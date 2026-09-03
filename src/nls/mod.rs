//! Neural local search: deploys a trained network as the repair operator of
//! a destroy/repair loop at inference time, decoupled from the
//! training loop entirely -- it only needs a loaded network, a problem, and
//! a destroy/decode operator pair.

pub mod config;
pub mod decode;
pub mod destroy;

pub use config::SolveConfig;
pub use decode::DecodingOperator;
pub use destroy::DestroyOperator;

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use burn::config::Config;
use burn::module::Module;
use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::learning::{Batch, Network, NetworkConfig};
use crate::modelling::Problem;
use crate::utils::tensor::*;

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

    fn log(&self, solutions: &[Option<Solution>]) {
        if self.iters_done.is_multiple_of(100) {
            let solved = solutions.iter().filter(|s| s.is_some()).count();
            log::info!(
                "Iteration {}, elapsed: {} seconds. Number solved {}/{}",
                self.iters_done,
                self.start.elapsed().as_secs(),
                solved,
                solutions.len(),
            );
        }
    }
}

/// Loads a network's hyperparameters (JSON config) and trained weights from a checkpoint
/// directory produced by `train_model`/`run_training`. Returns the config alongside the network
/// (rather than just the network) so callers also have access to training hyperparameters.
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

fn resolve_status<B: Backend>(
    decode_op: &dyn DecodingOperator<B>,
    problem: &Arc<Problem>,
    row: &[isize],
) -> Option<(Status, Option<Vec<isize>>)> {
    if problem.is_solution(row) {
        Some((Status::Satisfiable, Some(row.to_owned())))
    } else if decode_op.detect_unsat(problem) {
        Some((Status::Unsatisfiable, None))
    } else {
        None
    }
}

pub struct NeuralLocalSearch<B: Backend, N, Ba> {
    /// Neural network used to guide the local search
    network: N,
    /// Heuristic for the destroy operator
    destroy_op: Box<dyn DestroyOperator>,
    /// How to decode (arg-max or sample)
    decode_op: Box<dyn DecodingOperator<B>>,
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
    /// How often (in iterations) `run` physically drops solved problems out of the batch. See
    /// `run`'s doc for why this is periodic rather than immediate.
    const COMPACTION_INTERVAL: usize = 100;

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
        device: B::Device,
    ) -> Self {
        Self {
            network,
            destroy_op,
            decode_op,
            device,
            _batch: std::marker::PhantomData,
        }
    }

    pub fn run(&self, problems: &[Arc<Problem>], budget: Budget, seed: u64) -> Vec<Solution> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut stop = StoppingCriterion::new(budget);

        let mut active: Vec<usize> = (0..problems.len()).collect();
        let mut active_problems: Vec<Arc<Problem>> = problems.to_vec();
        let mut solutions: Vec<Option<Solution>> = vec![None; problems.len()];

        self.decode_op.prepare(&active_problems);

        // Starts from a random assignment; note that each variable is sampled given its domain, so
        // assigned variables are taken into account
        let mut assignments = self.random_init(problems);
        let n = problems[0].number_variables();
        let mut rows = to_rows(&assignments, problems.len(), n);

        while !stop.is_exhausted() && solutions.iter().any(Option::is_none) {
            let mut destroy_mask_data = vec![0i64; rows.len() * n];
            for (row_idx, row) in rows.iter().enumerate() {
                // Frozen: this problem already has a solution, leave its row untouched until it's
                // compacted out.
                if solutions[row_idx].is_some() {
                    continue;
                }
                let problem = &active_problems[row_idx];
                for var in self.destroy_op.destroy(problem, row, &mut rng) {
                    destroy_mask_data[row_idx * n + var] = 1;
                }
            }
            let destroy_mask: Tensor<B, 2, Int> =
                Tensor::<B, 1, Int>::from_data(destroy_mask_data.as_slice(), &self.device)
                    .reshape([rows.len(), n]);

            let batch = Ba::for_assignments(
                &active_problems,
                assignments.clone(),
                destroy_mask.clone(),
                &self.device,
            );
            let logits = self.network.forward(&batch);
            assignments =
                self.decode_op
                    .decode(logits, destroy_mask, assignments, &active_problems);
            rows = to_rows(&assignments, active_problems.len(), n);

            stop.tick();

            for (local_idx, &problem_idx) in active.iter().enumerate() {
                if solutions[problem_idx].is_some() {
                    continue;
                }
                let problem = &active_problems[local_idx];
                let row = &rows[local_idx];
                if let Some((status, solution)) =
                    resolve_status(self.decode_op.as_ref(), problem, row)
                {
                    solutions[problem_idx] = Some(Solution {
                        runtime: stop.start.elapsed().as_secs(),
                        iterations: stop.iters_done,
                        solution,
                        status,
                    });
                }
            }

            if stop.iters_done.is_multiple_of(Self::COMPACTION_INTERVAL) {
                let before = active.len();
                let mut still_active = Vec::with_capacity(before);
                let mut still_active_problems = Vec::with_capacity(active_problems.len());
                let mut still_rows = Vec::with_capacity(rows.len());
                for (local_idx, &problem_idx) in active.iter().enumerate() {
                    if solutions[problem_idx].is_some() {
                        continue;
                    }
                    still_active.push(problem_idx);
                    still_active_problems.push(active_problems[local_idx].clone());
                    still_rows.push(rows[local_idx].clone());
                }
                if still_active.len() < before {
                    active = still_active;
                    active_problems = still_active_problems;
                    rows = still_rows;
                    // Left stale when every remaining active problem just got compacted away --
                    // the `while` condition above exits before it's read again in that case.
                    if !active_problems.is_empty() {
                        assignments = rows_to_tensor(&rows, n, &self.device);
                    }
                }
            }

            stop.log(&solutions);
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
            .flat_map(|problem| problem.iter_variables().map(|v| problem[v].sample() as i64))
            .collect();
        Tensor::<B, 1, Int>::from_data(data.as_slice(), &self.device).reshape([problems.len(), n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modelling::not_equals;
    use burn::backend::ndarray::NdArray;

    /// A `DecodingOperator` whose `detect_unsat` always answers a fixed value, regardless of which
    /// problem is asked about -- `resolve_status` doesn't need a real MDD compilation (or even a
    /// real decode) to be exercised, just something that answers `detect_unsat`. `decode` is
    /// intentionally `unreachable!()`: `resolve_status` never calls it.
    struct FixedUnsatDecode(bool);

    impl DecodingOperator<NdArray> for FixedUnsatDecode {
        fn decode(
            &self,
            _logits: Tensor<NdArray, 3>,
            _destroy_mask: Tensor<NdArray, 2, Int>,
            _current: Tensor<NdArray, 2, Int>,
            _problems: &[Arc<Problem>],
        ) -> Tensor<NdArray, 2, Int> {
            unreachable!("resolve_status never calls decode")
        }

        fn detect_unsat(&self, _problem: &Arc<Problem>) -> bool {
            self.0
        }
    }

    /// `not_equals(x, y)` over `{0, 1}`: `[0, 1]`/`[1, 0]` solve it, `[0, 0]`/`[1, 1]` don't.
    fn not_equals_problem() -> Arc<Problem> {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1], None);
        let y = problem.add_variable(vec![0, 1], None);
        not_equals(&mut problem, x, y);
        Arc::new(problem)
    }

    #[test]
    fn resolve_status_reports_satisfiable_for_a_solving_row_even_if_the_operator_would_flag_unsat()
    {
        let problem = not_equals_problem();
        let op = FixedUnsatDecode(true);
        let (status, solution) =
            resolve_status(&op, &problem, &[0, 1]).expect("a solving row must resolve to a status");
        assert!(matches!(status, Status::Satisfiable));
        assert_eq!(solution, Some(vec![0, 1]));
    }

    #[test]
    fn resolve_status_reports_unsatisfiable_when_the_operator_flags_it_and_the_row_does_not_solve_it(
    ) {
        let problem = not_equals_problem();
        let op = FixedUnsatDecode(true);
        let (status, solution) = resolve_status(&op, &problem, &[0, 0])
            .expect("an operator-flagged problem must resolve to a status");
        assert!(matches!(status, Status::Unsatisfiable));
        assert_eq!(solution, None);
    }

    #[test]
    fn resolve_status_returns_none_when_neither_solved_nor_flagged() {
        let problem = not_equals_problem();
        let op = FixedUnsatDecode(false);
        assert!(resolve_status(&op, &problem, &[0, 0]).is_none());
    }
}
