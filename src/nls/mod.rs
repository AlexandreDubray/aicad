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
#[derive(Clone)]
pub struct Solution {
    /// Number of seconds elapsed before finding the solution
    pub(crate) runtime: u64,
    /// Number of local search steps before finding the solution
    pub(crate) iterations: usize,
    /// Solution to the problem, None if the problem is UNSAT.
    pub(crate) solution: Option<Vec<isize>>,
    /// Status of the solution. Either proved SAT/UNSAT in the budget limits, or unknown if the
    /// process reached a limit
    pub(crate) status: Status,
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

    /// Logs, per problem, the best (over its population) constraint satisfaction rate.
    fn log(&self, rows: &[Vec<isize>], problems: &[Arc<Problem>], population_size: usize) {
        if self.iters_done.is_multiple_of(100) {
            let solved = problems
                .iter()
                .enumerate()
                .filter(|(problem_idx, problem)| {
                    let base = problem_idx * population_size;
                    rows[base..base + population_size]
                        .iter()
                        .any(|row| {
                            problem.is_solution(row)
                        })
                }).count();
            log::info!(
                "Iteration {}, elapsed: {} seconds. Number solved {}/{}",
                self.iters_done,
                self.start.elapsed().as_secs(),
                solved,
                problems.len(),
            );
        }
    }
}

/// Loads a network's hyperparameters (JSON config) and trained weights from a checkpoint
/// directory produced by `train_model`/`run_training`. Returns the config alongside the network
/// (rather than just the network) so callers that also need a hyperparameter off the config --
/// e.g. `mask_fraction`, to pick a default `destroy_fraction` -- don't have to load `config.json`
/// a second time themselves.
///
/// Fallible rather than panicking: `checkpoint_dir` ultimately comes from a Python caller (a
/// typo'd path, a directory that isn't actually a checkpoint, or a `weights` file left over from
/// an incompatible config are all bad-input errors, not internal bugs), so this returns a boxed
/// `std::error::Error` for the pyo3 layer to turn into a catchable `PyErr` instead of aborting the
/// whole interpreter.
pub fn load_network<B, NC>(
    checkpoint_dir: &Path,
    problems: &[Arc<Problem>],
    device: &B::Device,
) -> Result<(NC, NC::N), Box<dyn std::error::Error>>
where
    B: Backend,
    NC: NetworkConfig<B> + Config + Clone,
    NC::N: Module<B>,
{
    let config: NC = NC::load(checkpoint_dir.join("config.json"))?;
    let network = config.clone().init(problems, device).load_file(
        checkpoint_dir.join("weights"),
        &CompactRecorder::new(),
        device,
    )?;
    Ok((config, network))
}

pub struct NeuralLocalSearch<B: Backend, N, Ba> {
    /// Neural network used to guide the local search
    network: N,
    /// Heuristic for the destroy operator
    destroy_op: Box<dyn DestroyOperator>,
    /// How to decode (arg-max or sample)
    decode_op: Box<dyn DecodingOperator<B>>,
    /// Number of assignments ran in parallel, per problem
    population_size: usize,
    /// Devices used (cpu or gpu)
    device: B::Device,
    /// Which batch type `N` is driven by at inference time. `NeuralLocalSearch` only ever needs
    /// `Ba::for_assignments`, so it's carried as a type parameter (rather than picked implicitly)
    /// so the same network type can still be paired with different batch types elsewhere (e.g. a
    /// training-only batch that doesn't support `for_assignments` at all).
    _batch: std::marker::PhantomData<Ba>,
}

impl<B: Backend, N, Ba> NeuralLocalSearch<B, N, Ba>
where
    Ba: Batch<B>,
    N: Network<B, Ba>,
{
    /// Builds the search engine: everything that's independent of *which*
    /// problems get solved (network weights, operators, device). Call `run`
    /// once per batch of problems -- the engine can be reused across several
    /// `run` calls on different problem sets without reloading the network
    /// (e.g. a caller that needs to keep the batch within some memory bound
    /// can call `run` once per chunk of problems).
    pub fn new(
        network: N,
        destroy_op: Box<dyn DestroyOperator>,
        decode_op: Box<dyn DecodingOperator<B>>,
        population_size: usize,
        device: B::Device,
    ) -> Self {
        Self {
            network,
            destroy_op,
            decode_op,
            population_size,
            device,
            _batch: std::marker::PhantomData,
        }
    }

    /// Runs the search on `problems`, batching every problem's population into
    /// a single forward pass per iteration, until `budget` is exhausted or
    /// every problem has found a feasible solution. `seed` controls the
    /// destroy operator's randomness. All problems must share the same
    /// `number_variables()`.
    ///
    /// A problem that finds a solution before the others is frozen in place
    /// (its rows stop being destroyed/repaired, but stay in the batch so
    /// tensor shapes remain consistent) while the rest keep iterating.
    ///
    /// Returns one `Solution` per problem, in the same order as `problems`.
    pub fn run(&self, problems: &[Arc<Problem>], budget: Budget, seed: u64) -> Vec<Solution> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut stop = StoppingCriterion::new(budget);

        let num_problems = problems.len();
        let n = problems[0].number_variables();
        let p = self.population_size;
        let total_rows = num_problems * p;

        // Starts from a random assignment; note that each variable is sampled given its domain, so
        // assigned variables are taken into account
        let mut assignments = self.random_init(problems);
        let mut rows = to_rows(&assignments, total_rows, n);

        let mut solutions: Vec<Option<Solution>> = vec![None; num_problems];

        while !stop.is_exhausted() && solutions.iter().any(Option::is_none) {
            let mut destroy_mask_data = vec![false; total_rows * n];
            for (row_idx, row) in rows.iter().enumerate() {
                let problem_idx = row_idx / p;
                // Frozen: this problem already has a solution, leave its rows untouched.
                if solutions[problem_idx].is_some() {
                    continue;
                }
                let problem = &problems[problem_idx];
                for var in self.destroy_op.destroy(problem, row, &mut rng) {
                    destroy_mask_data[row_idx * n + var] = true;
                }
            }
            let destroy_mask: Tensor<B, 2, Bool> =
                Tensor::<B, 1, Bool>::from_data(destroy_mask_data.as_slice(), &self.device)
                    .reshape([total_rows, n]);

            let batch = Ba::for_assignments(
                problems,
                p,
                assignments.clone(),
                destroy_mask.clone(),
                &self.device,
            );
            let logits = self.network.forward(&batch);
            assignments = self.decode_op.decode(logits, destroy_mask, assignments);
            rows = to_rows(&assignments, total_rows, n);

            stop.tick();

            for (problem_idx, problem) in problems.iter().enumerate() {
                if solutions[problem_idx].is_some() {
                    continue;
                }
                let base = problem_idx * p;
                if let Some(row) = rows[base..base + p]
                    .iter()
                    .find(|row| problem.is_solution(row))
                {
                    solutions[problem_idx] = Some(Solution {
                        runtime: stop.start.elapsed().as_secs(),
                        iterations: stop.iters_done,
                        solution: Some(row.to_owned()),
                        status: Status::Satisfiable,
                    });
                }
            }

            stop.log(&rows, problems, p);
        }

        solutions
            .into_iter()
            .map(|s| {
                s.unwrap_or(Solution {
                    runtime: stop.start.elapsed().as_secs(),
                    iterations: stop.iters_done,
                    solution: None,
                    status: Status::Unknown,
                })
            })
            .collect()
    }

    fn random_init(&self, problems: &[Arc<Problem>]) -> Tensor<B, 2, Int> {
        let n = problems[0].number_variables();
        let data: Vec<i64> = problems
            .iter()
            .flat_map(|problem| {
                (0..self.population_size)
                    .flat_map(move |_| problem.iter_variables().map(|v| problem[v].sample() as i64))
            })
            .collect();
        Tensor::<B, 1, Int>::from_data(data.as_slice(), &self.device)
            .reshape([problems.len() * self.population_size, n])
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
