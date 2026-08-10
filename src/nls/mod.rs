//! Neural local search: deploys a trained network as the repair operator of
//! a destroy/repair loop at inference time, decoupled from the
//! training loop entirely -- it only needs a loaded network, a problem, and
//! a destroy/decode operator pair.

pub mod decode;
pub mod destroy;

pub use decode::DecodingOperator;
pub use destroy::DestroyOperator;

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use burn::config::Config;
use burn::module::Module;
use burn::prelude::ElementConversion;
use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::learning::{Batch, Network, NetworkConfig};
use crate::modelling::Problem;

/// A solution returned by the solver, with its statistics
pub struct Solution {
    /// Number of seconds elapsed before finding the solution
    runtime: u64,
    /// Number of local search steps before finding the solution
    iterations: usize,
    /// Solution to the problem, None if the problem is UNSAT.
    solution: Option<Vec<isize>>,
    /// Status of the solution. Either proved SAT/UNSAT in the budget limits, or unknown if the
    /// process reached a limit
    status: Status,
}

#[derive(Clone, Copy)]
pub enum Status {
    Satisfiable,
    Unsatisfiable,
    Unknown,
}

impl Solution {
    pub fn runtime(&self) -> u64 {
        self.runtime
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn solution(&self) -> &Option<Vec<isize>> {
        &self.solution
    }

    pub fn is_sat(&self) -> bool {
        self.solution.is_some()
    }

    pub fn status(&self) -> Status {
        self.status
    }
}

/// Stopping criterion for the search; whichever limit is first reached stops it.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub time_limit: Duration,
    pub iteration_limit: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Budget {
            time_limit: Duration::MAX,
            iteration_limit: usize::MAX,
        }
    }
}

struct StoppingCriterion {
    budget: Budget,
    start: Instant,
    iters_done: usize,
}

impl StoppingCriterion {
    fn new(budget: Budget) -> Self {
        Self {
            budget,
            start: Instant::now(),
            iters_done: 0,
        }
    }

    fn tick(&mut self) {
        self.iters_done += 1;
    }

    fn is_exhausted(&self) -> bool {
        self.start.elapsed() >= self.budget.time_limit
            || self.iters_done >= self.budget.iteration_limit
    }

    fn log(&self, assignments: &Vec<Vec<isize>>, problem: &Arc<Problem>) {
        if self.iters_done % 10 == 0 {
            let n = problem.number_constraints() as f64;
            let satisfaction_rates = assignments
                .iter()
                .map(|assignment| {
                    problem
                        .iter_constraints()
                        .filter(|&cstr| problem[cstr].is_satisfied(assignment))
                        .count() as f64
                        / n
                })
                .collect::<Vec<f64>>();
            log::info!(
                "Iteration {}, elapsed: {} seconds. Satisfaction rates of candidates: {:?}",
                self.iters_done,
                self.start.elapsed().as_secs(),
                satisfaction_rates,
            );
        }
    }
}

/// Loads a network's hyperparameters (JSON config) and trained weights from
/// a checkpoint directory produced by `train_model`/`run_training`
pub fn load_network<B, NC>(checkpoint_dir: &Path, device: &B::Device) -> NC::N
where
    B: Backend,
    NC: NetworkConfig<B> + Config,
    NC::N: Module<B>,
{
    let config: NC =
        NC::load(checkpoint_dir.join("config.json")).expect("failed to load network config");
    config
        .init(device)
        .load_file(
            checkpoint_dir.join("weights"),
            &CompactRecorder::new(),
            device,
        )
        .expect("failed to load network weights")
}

pub struct NeuralLocalSearch<B: Backend, N: Network<B>> {
    /// Problem being solved
    problem: Arc<Problem>,
    /// Neural network used to guide the local search
    network: N,
    /// Heuristic for the destroy operator
    destroy_op: Box<dyn DestroyOperator>,
    /// How to decode (arg-max or sample)
    decode_op: Box<dyn DecodingOperator<B>>,
    /// Number of assignments ran in parallel
    population_size: usize,
    /// Devices used (cpu or gpu)
    device: B::Device,
}

impl<B: Backend, N: Network<B>> NeuralLocalSearch<B, N> {
    pub fn new(
        problem: Arc<Problem>,
        network: N,
        destroy_op: Box<dyn DestroyOperator>,
        decode_op: Box<dyn DecodingOperator<B>>,
        population_size: usize,
        device: B::Device,
    ) -> Self {
        Self {
            problem,
            network,
            destroy_op,
            decode_op,
            population_size,
            device,
        }
    }

    /// Runs the search until `budget` is exhausted or a feasible solution is
    /// found. `seed` controls the destroy operator's randomness
    /// Returns a solution if found, None otherwise
    pub fn run(&self, budget: Budget, seed: u64) -> Solution {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut stop = StoppingCriterion::new(budget);

        let n = self.problem.number_variables();
        let p = self.population_size;

        // Starts from a random assignment; note that each variable is sampled given its domain, so
        // assigned variables are taken into account
        let mut assignments = self.random_init();
        let mut rows = to_rows(&assignments, p, n);

        while !stop.is_exhausted() {
            let mut destroy_mask_data = vec![false; p * n];
            for (row_idx, row) in rows.iter().enumerate() {
                for var in self.destroy_op.destroy(&self.problem, row, &mut rng) {
                    destroy_mask_data[row_idx * n + var] = true;
                }
            }
            let destroy_mask: Tensor<B, 2, Bool> =
                Tensor::<B, 1, Bool>::from_data(destroy_mask_data.as_slice(), &self.device)
                    .reshape([p, n]);

            let batch = N::Batch::for_assignments(
                &self.problem,
                assignments.clone(),
                destroy_mask.clone(),
                &self.device,
            );
            let logits = self.network.forward(&batch);
            assignments = self.decode_op.decode(logits, destroy_mask, assignments);
            rows = to_rows(&assignments, p, n);

            for row in rows.iter() {
                if self
                    .problem
                    .iter_constraints()
                    .all(|cstr| self.problem[cstr].is_satisfied(row))
                {
                    let sol = row.to_owned();
                    return Solution {
                        runtime: stop.start.elapsed().as_secs(),
                        iterations: stop.iters_done + 1,
                        solution: Some(sol),
                        status: Status::Satisfiable,
                    };
                }
            }

            stop.tick();
            stop.log(&rows, &self.problem);
        }
        Solution {
            runtime: stop.start.elapsed().as_secs(),
            iterations: stop.iters_done,
            solution: None,
            status: Status::Unknown,
        }
    }

    fn random_init(&self) -> Tensor<B, 2, Int> {
        let n = self.problem.number_variables();
        let data: Vec<i64> = (0..self.population_size)
            .flat_map(|_| {
                self.problem
                    .iter_variables()
                    .map(|v| self.problem[v].sample() as i64)
            })
            .collect();
        Tensor::<B, 1, Int>::from_data(data.as_slice(), &self.device)
            .reshape([self.population_size, n])
    }
}

fn to_rows<B: Backend>(assignments: &Tensor<B, 2, Int>, p: usize, n: usize) -> Vec<Vec<isize>> {
    let flat: Vec<i64> = assignments
        .clone()
        .into_data()
        .to_vec::<B::IntElem>()
        .expect("assignment tensor should be integer")
        .into_iter()
        .map(|v| v.elem::<i64>())
        .collect();
    (0..p)
        .map(|i| {
            flat[i * n..(i + 1) * n]
                .iter()
                .map(|&v| v as isize)
                .collect()
        })
        .collect()
}
