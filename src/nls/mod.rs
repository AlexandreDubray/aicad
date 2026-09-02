//! Neural local search: deploys a trained network as the repair operator of
//! a destroy/repair loop at inference time, decoupled from the
//! training loop entirely -- it only needs a loaded network, a problem, and
//! a destroy/decode operator pair.

pub mod config;
pub mod decode;
pub mod destroy;

pub use config::SolveConfig;
pub use decode::DecodingOperator;
pub use destroy::{DestroyOperator, MaskSchedule};

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

/// Splits `problems` into the ones `decode_op` already knows are unsatisfiable (see
/// `DecodingOperator::detect_unsat`'s doc) and the ones that still need actual search. Returns
/// `(active, solutions)`: `active` lists the indices (into `problems`) of problems that still need
/// solving, in their original relative order; `solutions[i]` is `Some(...)` -- reporting
/// `Status::Unsatisfiable`, with `start` as its runtime baseline -- for exactly the problems
/// `detect_unsat` flagged, and `None` for every index also listed in `active`.
///
/// A free function, deliberately not a `NeuralLocalSearch` method: it only touches the decode
/// operator, not the network or destroy operator, so it doesn't need `N`/`Ba` at all -- which also
/// makes it directly testable with a bare `DecodingOperator` and no network/batch machinery.
fn partition_unsat<B: Backend>(
    decode_op: &dyn DecodingOperator<B>,
    problems: &[Arc<Problem>],
    start: Instant,
) -> (Vec<usize>, Vec<Option<Solution>>) {
    let mut solutions: Vec<Option<Solution>> = vec![None; problems.len()];
    let mut active = Vec::with_capacity(problems.len());
    for (i, problem) in problems.iter().enumerate() {
        if decode_op.detect_unsat(problem) {
            log::info!(
                "problem {i}: UNSAT detected during MDD compilation (a bucket-grouped set of \
                 constraints has no accepting path at all)"
            );
            solutions[i] = Some(Solution {
                runtime: start.elapsed().as_secs(),
                iterations: 0,
                solution: None,
                status: Status::Unsatisfiable,
            });
        } else {
            active.push(i);
        }
    }
    (active, solutions)
}

pub struct NeuralLocalSearch<B: Backend, N, Ba> {
    /// Neural network used to guide the local search
    network: N,
    /// Heuristic for the destroy operator
    destroy_op: Box<dyn DestroyOperator>,
    /// How the destroy fraction evolves across a `run` call's iterations
    mask_schedule: MaskSchedule,
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
        mask_schedule: MaskSchedule,
        decode_op: Box<dyn DecodingOperator<B>>,
        population_size: usize,
        device: B::Device,
    ) -> Self {
        Self {
            network,
            destroy_op,
            mask_schedule,
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
    /// A problem that finds a solution is frozen in place immediately (its rows stop being
    /// destroyed, so its solution can't be overwritten by a later iteration), but is only
    /// physically removed from the batch every `COMPACTION_INTERVAL` iterations -- rebuilding the
    /// row/tensor bookkeeping on every single solve is itself non-negligible overhead at these
    /// batch sizes, so it's amortised over a stretch of iterations instead. Either way,
    /// `Solution::iterations` records the exact iteration a problem solved on, not the (later)
    /// iteration it happened to be swept out of the batch on.
    ///
    /// Returns one `Solution` per problem, in the same order as `problems`.
    pub fn run(&self, problems: &[Arc<Problem>], budget: Budget, seed: u64) -> Vec<Solution> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut stop = StoppingCriterion::new(budget);

        let n = problems[0].number_variables();
        let p = self.population_size;

        // UNSAT is a static property of a problem's compiled MDDs -- e.g. a bucket-grouped clique
        // of constraints with no accepting path at all -- not something that changes as the
        // destroy/repair loop runs, so it's checked once up front rather than left for the search
        // to burn its whole budget failing to converge on. Only `BeliefPropagationDecode` can
        // actually answer this (see `DecodingOperator::detect_unsat`'s doc); every other decode
        // operator always says no, same as today's behaviour.
        let (mut active, mut solutions) =
            partition_unsat(self.decode_op.as_ref(), problems, stop.start);
        if active.is_empty() {
            return solutions
                .into_iter()
                .map(|s| s.expect("every problem was just marked Unsatisfiable above"))
                .collect();
        }

        // The problems whose rows are currently in `rows`/`assignments`, and `active[i]`'s index
        // back into `problems`/`solutions`. May include already-solved problems in between
        // compaction passes -- see `COMPACTION_INTERVAL` below.
        let mut active_problems: Vec<Arc<Problem>> =
            active.iter().map(|&i| Arc::clone(&problems[i])).collect();

        // Starts from a random assignment; note that each variable is sampled given its domain, so
        // assigned variables are taken into account
        let mut assignments = self.random_init(&active_problems);
        let mut rows = to_rows(&assignments, active_problems.len() * p, n);

        while !stop.is_exhausted() && solutions.iter().any(Option::is_none) {
            let fraction = self.mask_schedule.fraction_at(stop.iters_done);
            let mut destroy_mask_data = vec![0i64; rows.len() * n];
            for (row_idx, row) in rows.iter().enumerate() {
                let problem_idx = active[row_idx / p];
                // Frozen: this problem already has a solution, leave its row untouched until it's
                // compacted out.
                if solutions[problem_idx].is_some() {
                    continue;
                }
                let problem = &active_problems[row_idx / p];
                for var in self.destroy_op.destroy(problem, row, fraction, &mut rng) {
                    destroy_mask_data[row_idx * n + var] = 1;
                }
            }
            let destroy_mask: Tensor<B, 2, Int> =
                Tensor::<B, 1, Int>::from_data(destroy_mask_data.as_slice(), &self.device)
                    .reshape([rows.len(), n]);

            let batch = Ba::for_assignments(
                &active_problems,
                p,
                assignments.clone(),
                destroy_mask.clone(),
                &self.device,
            );
            let logits = self.network.forward(&batch);
            assignments =
                self.decode_op
                    .decode(logits, destroy_mask, assignments, &active_problems, p);
            rows = to_rows(&assignments, active_problems.len() * p, n);

            stop.tick();

            for (local_idx, &problem_idx) in active.iter().enumerate() {
                if solutions[problem_idx].is_some() {
                    continue;
                }
                let problem = &active_problems[local_idx];
                let base = local_idx * p;
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
                    let base = local_idx * p;
                    still_rows.extend_from_slice(&rows[base..base + p]);
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
            .flat_map(|problem| {
                (0..self.population_size)
                    .flat_map(move |_| problem.iter_variables().map(|v| problem[v].sample() as i64))
            })
            .collect();
        Tensor::<B, 1, Int>::from_data(data.as_slice(), &self.device)
            .reshape([problems.len() * self.population_size, n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    /// A `DecodingOperator` whose `detect_unsat` is driven purely by a marker (a 1-variable
    /// problem stands in for "flagged UNSAT during MDD compilation", any other variable count for
    /// "still needs solving") -- exactly the shape `BeliefPropagationDecode::detect_unsat` has
    /// (answerable from the problem alone, no tensors involved), without needing a real MDD
    /// compilation to produce that answer. `decode` is intentionally `unreachable!()`: nothing in
    /// these tests should ever call it.
    struct MarkedUnsatDecode;

    impl DecodingOperator<NdArray> for MarkedUnsatDecode {
        fn decode(
            &self,
            _logits: Tensor<NdArray, 3>,
            _destroy_mask: Tensor<NdArray, 2, Int>,
            _current: Tensor<NdArray, 2, Int>,
            _problems: &[Arc<Problem>],
            _population_size: usize,
        ) -> Tensor<NdArray, 2, Int> {
            unreachable!("decode should never be called in these tests")
        }

        fn detect_unsat(&self, problem: &Arc<Problem>) -> bool {
            problem.number_variables() == 1
        }
    }

    fn marked_problem(unsat: bool) -> Arc<Problem> {
        let mut problem = Problem::default();
        problem.add_variable(vec![0, 1], None);
        if !unsat {
            problem.add_variable(vec![0, 1], None);
        }
        Arc::new(problem)
    }

    #[test]
    fn partition_unsat_splits_out_flagged_problems_and_leaves_the_rest_active() {
        let sat = marked_problem(false);
        let unsat = marked_problem(true);
        let problems = vec![sat.clone(), unsat.clone(), sat.clone()];

        let op = MarkedUnsatDecode;
        let (active, solutions) = partition_unsat::<NdArray>(&op, &problems, Instant::now());

        assert_eq!(active, vec![0, 2]);
        assert!(solutions[0].is_none());
        assert!(solutions[2].is_none());

        let flagged = solutions[1]
            .as_ref()
            .expect("problem 1 was marked unsat and should have a Solution already");
        assert!(matches!(flagged.status, Status::Unsatisfiable));
        assert!(flagged.solution.is_none());
        assert_eq!(flagged.iterations, 0);
    }

    #[test]
    fn partition_unsat_with_nothing_flagged_leaves_every_index_active() {
        let problems = vec![marked_problem(false), marked_problem(false)];
        let op = MarkedUnsatDecode;
        let (active, solutions) = partition_unsat::<NdArray>(&op, &problems, Instant::now());

        assert_eq!(active, vec![0, 1]);
        assert!(solutions.iter().all(Option::is_none));
    }

    #[test]
    fn partition_unsat_with_everything_flagged_leaves_active_empty() {
        let problems = vec![marked_problem(true), marked_problem(true)];
        let op = MarkedUnsatDecode;
        let (active, solutions) = partition_unsat::<NdArray>(&op, &problems, Instant::now());

        assert!(active.is_empty());
        assert!(
            solutions
                .iter()
                .all(|s| matches!(s.as_ref().map(|s| s.status), Some(Status::Unsatisfiable)))
        );
    }
}
