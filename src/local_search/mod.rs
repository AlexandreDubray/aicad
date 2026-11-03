//! Local search: repeatedly destroys part of an assignment and repairs it, until the problem is
//! solved or the budget is spent.
//!
//! The engine ([`LocalSearch`]) is the loop around a destroy-and-repair step: it keeps the budget,
//! the clock, the assignments and which problems are solved. It does not say what a step is. That
//! belongs to a [`SearchArchitecture`], which builds a [`SearchState`] for a batch of problems and
//! performs the steps on the assignments the engine holds, so a search may be as symbolic as it
//! likes: a repair that never leaves the CPU needs no tensor, no network and no device.
//!
//! # The architectures
//!
//! - [`BeliefPropagationSearch`]: belief propagation over the MDDs from uniform probabilities, on
//!   the CPU only.
//! - `NeuralPriorSearch` and `NeuralPriorBpSearch`, with the `learning` feature: a trained network
//!   evaluated at each step, whose logits are decoded directly or used as the prior of belief
//!   propagation over the MDDs of the constraints. Any network of `crate::neural` can drive them.
//!
//! Each has its own settings, gathered in [`SearchConfig`]; [`repair`] holds the belief-propagation
//! repair they share.
//!
//! # Example
//!
//! Belief propagation needs no network, so it runs in the lightest build of the crate:
//!
//! ```
//! use aicad::constraints::all_different;
//! use aicad::local_search::{solve, SolveConfig};
//! use aicad::modelling::{Problem, Status};
//! use std::sync::Arc;
//!
//! let mut problem = Problem::default();
//! let vars = problem.add_variables(4, vec![0, 1, 2, 3]);
//! all_different(&mut problem, vars);
//!
//! // A seed makes the run reproducible; the default search is belief propagation.
//! let config = SolveConfig::default().with_iteration_limit(1000).with_seed(0);
//! let solutions = solve(&[Arc::new(problem)], &config, None).unwrap();
//! assert_eq!(solutions[0].status(), Status::Satisfiable);
//! ```

#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

mod bp_search;
mod config;
#[cfg(feature = "learning")]
mod neural_prior;
pub mod repair;

pub use bp_search::{BeliefPropagationSearch, BeliefPropagationSearchConfig};
pub use config::{Budget, SearchConfig, SolveConfig};
#[cfg(feature = "learning")]
pub use neural_prior::{
    NeuralPriorBpSearch, NeuralPriorBpSearchConfig, NeuralPriorSearch, NeuralPriorSearchConfig,
    solve_with_network,
};

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use crate::destroy::DestroyOperator;
use crate::modelling::{Problem, Solution, Status};

/// Stopping criterion, updated during the local search loop.
struct StoppingCriterion {
    /// Allowed computation budget.
    budget: Budget,
    /// Start time of the loop.
    start: Instant,
    /// Number of iterations performed.
    iters_done: usize,
}

impl StoppingCriterion {
    /// Starts the clock and the iteration count for a run with the given budget.
    fn new(budget: Budget) -> Self {
        Self {
            budget,
            start: Instant::now(),
            iters_done: 0,
        }
    }

    /// Updates the stopping criterion after an iteration of local search.
    fn tick(&mut self) {
        self.iters_done += 1;
    }

    /// Returns true of the allowed computational budget is exhausted.
    fn is_exhausted(&self) -> bool {
        self.start.elapsed() >= self.budget.time_limit
            || self.iters_done >= self.budget.iteration_limit
    }
}

/// The search of one architecture: what it needs from a batch of problems, and how it destroys and
/// repairs their assignments.
pub trait SearchArchitecture {
    /// Builds the state of a search over `problems`, which is never empty. It holds what the
    /// search needs about the problems (per-problem data) and its rows follow the order of
    /// `problems`.
    ///
    /// Returns an error naming the restriction when the architecture cannot search these problems
    /// together.
    fn setup<'a>(&'a self, problems: &[Arc<Problem>]) -> Result<Box<dyn SearchState + 'a>, String>;
}

/// The state of a search over a batch of problems, one row per problem still in the batch. The
/// assignments belong to the engine, which hands them to every step; the state holds only what
/// the architecture needs besides them.
pub trait SearchState {
    /// One iteration: destroys part of the assignment of each row flagged in `pending` and repairs
    /// it, in `assignments`. The other rows (their problem is solved but not yet dropped) must be
    /// left as they are. `problems` and `assignments` have one entry per row. `rng` is the source
    /// of the randomness of the iteration.
    fn step(
        &mut self,
        problems: &[Arc<Problem>],
        assignments: &mut [Vec<isize>],
        pending: &[bool],
        rng: &mut StdRng,
    );

    /// The rows flagged in `solved` have just been found to be solutions: the state may free what
    /// it holds for them. Does nothing by default.
    fn solved(&mut self, _solved: &[bool]) {}

    /// Drops the rows not flagged in `keep`; the others keep their order. Does nothing by default,
    /// for a state that holds nothing per row.
    fn retain(&mut self, _keep: &[bool]) {}
}

/// A random starting assignment for each of `problems`, each value drawn from its domain.
fn random_assignments(problems: &[Arc<Problem>]) -> Vec<Vec<isize>> {
    problems
        .iter()
        .map(|problem| crate::sampling::random_assignment(problem))
        .collect()
}

/// The variables each row frees, `destroy_op` choosing them with a seed drawn per row from `rng`.
/// A row that is not `pending` (solved, but not yet dropped) frees nothing.
pub fn destroy_rows(
    destroy_op: &dyn DestroyOperator,
    problems: &[Arc<Problem>],
    assignments: &[Vec<isize>],
    pending: &[bool],
    rng: &mut StdRng,
) -> Vec<Vec<usize>> {
    let seeds = row_seeds(problems.len(), rng);
    crate::utils::worker_pool().install(|| {
        assignments
            .par_iter()
            .zip(problems.par_iter())
            .zip(seeds.par_iter())
            .zip(pending.par_iter())
            .map(|(((row, problem), &seed), &is_pending)| {
                if !is_pending {
                    return Vec::new();
                }
                let mut rng = StdRng::seed_from_u64(seed);
                destroy_op.destroy(problem, row, &mut rng)
            })
            .collect()
    })
}

/// One seed per row, drawn in order from `rng`, so that work done on the rows in parallel stays
/// reproducible.
pub fn row_seeds(rows: usize, rng: &mut StdRng) -> Vec<u64> {
    (0..rows).map(|_| rng.random::<u64>()).collect()
}

/// The items flagged in `keep`, in order.
fn filtered<T>(items: Vec<T>, keep: &[bool]) -> Vec<T> {
    items
        .into_iter()
        .zip(keep)
        .filter_map(|(item, &kept)| kept.then_some(item))
        .collect()
}

/// Local search engine: repeatedly destroys part of each assignment and repairs it, until every
/// problem is solved or the budget is spent. What a destroy and repair step is belongs to the
/// [`SearchArchitecture`]: the engine only runs the loop, keeps the budget and tracks which
/// problems are solved.
pub struct LocalSearch {
    /// The architecture performing the steps.
    architecture: Box<dyn SearchArchitecture>,
}

impl LocalSearch {
    /// How often (in iterations) `run` physically drops solved problems out of the batch. See
    /// `run`'s doc for why this is periodic rather than immediate.
    const COMPACTION_INTERVAL: usize = 100;

    /// Builds the search engine for an architecture. The engine is independent of the problems
    /// being solved so it can be reused across batches of problems. Nothing is kept between runs,
    /// what an architecture needs per problem is set up at the start of each run, which reports
    /// that time as preprocessing, and owned by the run.
    pub fn new(architecture: Box<dyn SearchArchitecture>) -> Self {
        Self { architecture }
    }

    /// Runs the local search on a batch of problems with the allowed budget and the seed.
    /// Optionally, a progress bar is given to monitor the progress.
    ///
    /// To evaluate the runtime of a method in seconds, run a few (e.g., 100) iterations
    /// of each problem alone and record the average step time; then run in large batches and
    /// infer the runtime from the number of steps observed, plus the preprocessing time.
    ///
    /// A solved problem stays in the batch, untouched, until the next compaction, and is dropped
    /// then, so the state is not rebuilt at every solved problem.
    ///
    /// Returns an error when the architecture cannot search these problems together, see
    /// [`SearchArchitecture::setup`].
    pub fn run(
        &self,
        problems: &[Arc<Problem>],
        budget: Budget,
        seed: u64,
        progress: Option<&indicatif::ProgressBar>,
    ) -> Result<Vec<Solution>, String> {
        if problems.is_empty() {
            return Ok(Vec::new());
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let mut stop = StoppingCriterion::new(budget);

        // Index in `problems` of the problem of each row of the state.
        let mut active: Vec<usize> = (0..problems.len()).collect();
        // The problem and the assignment of each row, in the order of `active`.
        let mut row_problems: Vec<Arc<Problem>> = problems.to_vec();
        let mut assignments = random_assignments(problems);
        // Sets of found solution for this batch.
        let mut solutions: Vec<Option<Solution>> = vec![None; problems.len()];

        let preprocessing_start = Instant::now();
        let mut state = self.architecture.setup(problems)?;
        let preprocessing_runtime = preprocessing_start.elapsed().as_secs_f64();

        // Continues while the budget is not finished, and there is unsolved problem in the batch
        while !stop.is_exhausted() && solutions.iter().any(Option::is_none) {
            // Get unsolved problems
            let pending: Vec<bool> = active
                .iter()
                .map(|&problem_idx| solutions[problem_idx].is_none())
                .collect();

            state.step(&row_problems, &mut assignments, &pending, &mut rng);

            // One more iteration done.
            stop.tick();

            // Updated the solutions *only* for newly solved problems.
            let solved_now: Vec<bool> = crate::utils::worker_pool().install(|| {
                assignments
                    .par_iter()
                    .zip(row_problems.par_iter())
                    .zip(pending.par_iter())
                    .map(|((row, problem), &is_pending)| is_pending && problem.is_solution(row))
                    .collect()
            });
            state.solved(&solved_now);
            for (row, &problem_idx) in active.iter().enumerate() {
                if solved_now[row] {
                    solutions[problem_idx] = Some(
                        Solution::satisfiable(&problems[problem_idx], assignments[row].clone())
                            .with_runtime(stop.start.elapsed().as_secs_f64(), preprocessing_runtime)
                            .with_iterations(stop.iters_done),
                    );
                    if let Some(bar) = progress {
                        bar.inc(1);
                    }
                }
            }
            if let Some(bar) = progress {
                bar.set_message(format!("Solving instances (iteration {})", stop.iters_done));
            }

            if stop.iters_done.is_multiple_of(Self::COMPACTION_INTERVAL) {
                let keep: Vec<bool> = active
                    .iter()
                    .map(|&problem_idx| solutions[problem_idx].is_none())
                    .collect();
                if keep.iter().any(|&kept| !kept) {
                    state.retain(&keep);
                    row_problems = filtered(row_problems, &keep);
                    assignments = filtered(assignments, &keep);
                    active = filtered(active, &keep);
                }
            }
        }

        Ok(solutions
            .into_iter()
            .map(|s| {
                s.unwrap_or_else(|| {
                    Solution::unknown()
                        .with_runtime(stop.start.elapsed().as_secs_f64(), preprocessing_runtime)
                        .with_iterations(stop.iters_done)
                })
            })
            .collect())
    }
}

/// Solves `problems` in batches of at most `batch_size` (all of them in one batch if `None`), one
/// batch after the other, each with the full `budget`. The seed of a batch is `seed` plus its
/// index, so that the batches do not replay the same destroy sequence. Shows a progress bar and
/// logs the outcome of every batch.
///
/// Returns one solution per problem, in order, or the error of the architecture if it cannot search
/// a batch (see [`SearchArchitecture::setup`]).
pub fn run_batched(
    engine: &LocalSearch,
    problems: &[Arc<Problem>],
    batch_size: Option<usize>,
    budget: Budget,
    seed: u64,
) -> Result<Vec<Solution>, String> {
    let batch_size = batch_size.unwrap_or(problems.len()).max(1);
    let batches: Vec<&[Arc<Problem>]> = problems.chunks(batch_size).collect();
    log::info!(
        "Solving {} problems in {} batches of at most {}",
        problems.len(),
        batches.len(),
        batch_size,
    );

    let progress = ProgressBar::new(problems.len() as u64);
    progress.set_style(
        ProgressStyle::with_template(
            "{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )
        .expect("hard-coded progress bar template should always be valid"),
    );
    progress.set_message("Solving instances");

    let mut solutions = Vec::with_capacity(problems.len());
    for (batch_idx, batch) in batches.iter().enumerate() {
        let batch_start = Instant::now();
        let batch_solutions = engine.run(
            batch,
            budget,
            seed.wrapping_add(batch_idx as u64),
            Some(&progress),
        )?;
        log_status_counts(
            &format!(
                "batch {}/{} done in {:.1}s",
                batch_idx + 1,
                batches.len(),
                batch_start.elapsed().as_secs_f64()
            ),
            &batch_solutions,
        );
        // `run` already advanced the bar for every problem it solved; the ones it gave up on are
        // only known now.
        let unsolved = batch_solutions
            .iter()
            .filter(|s| matches!(s.status(), Status::Unknown))
            .count();
        progress.inc(unsolved as u64);
        solutions.extend(batch_solutions);
    }
    progress.finish_and_clear();
    log_status_counts("Done", &solutions);

    Ok(solutions)
}

/// Logs how many of `solutions` are satisfiable, unsatisfiable and unknown.
fn log_status_counts(label: &str, solutions: &[Solution]) {
    let count = |wanted: Status| solutions.iter().filter(|s| s.status() == wanted).count();
    let (sat, unsat) = (count(Status::Satisfiable), count(Status::Unsatisfiable));
    log::info!(
        "{label}: {sat} sat, {unsat} unsat, {} unknown/timed out (of {})",
        solutions.len() - sat - unsat,
        solutions.len(),
    );
}

/// Solves `problems` with the local search that `config` describes: the batching, the limits, the
/// seed and the architecture with its settings. Returns one solution per problem, in order.
///
/// The seed of `config` is drawn from the process-wide generator when unset, so it is reproducible
/// once [`crate::set_seed`] was called.
///
/// `checkpoint_dir` is the directory of a trained ConsFormer (`config.json` and `weights`, as
/// written by `learning::consformer::train_consformer`): the neural architectures need
/// it and load their network from it, on the best device found; the belief-propagation one uses no
/// network and refuses it. To search with another network, or on a device of your choice, build the
/// architecture yourself and use [`run_batched`].
///
/// # Errors
///
/// When `problems` is empty, when the checkpoint is missing for a neural architecture or given to
/// one that has no network, when it cannot be loaded, or when the architecture cannot search these
/// problems together.
pub fn solve(
    problems: &[Arc<Problem>],
    config: &SolveConfig,
    checkpoint_dir: Option<&Path>,
) -> Result<Vec<Solution>, String> {
    if problems.is_empty() {
        return Err("local search: there are no problems to solve".to_string());
    }
    let seed = config
        .seed
        .unwrap_or_else(|| crate::utils::with_rng(|rng| rng.random_range(0..u64::MAX)));
    match &config.search {
        SearchConfig::BeliefPropagation(search) => {
            if checkpoint_dir.is_some() {
                return Err(
                    "the belief-propagation search uses no network: do not give a checkpoint"
                        .to_string(),
                );
            }
            let engine = LocalSearch::new(Box::new(BeliefPropagationSearch::new(search)));
            run_batched(&engine, problems, config.batch_size, config.budget, seed)
        }
        #[cfg(feature = "learning")]
        _ => neural_prior::solve_with_consformer(problems, config, checkpoint_dir, seed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::destroy::DestroyOperator;
    use std::time::Duration;

    struct CallCounter {
        calls: std::sync::Mutex<std::collections::HashMap<usize, usize>>,
    }

    impl DestroyOperator for CallCounter {
        fn destroy(
            &self,
            problem: &Problem,
            _assignment: &[isize],
            _rng: &mut StdRng,
        ) -> Vec<usize> {
            *self
                .calls
                .lock()
                .unwrap()
                .entry(problem as *const Problem as usize)
                .or_default() += 1;
            Vec::new()
        }
    }

    /// A search that destroys through the counter and repairs nothing: its assignments never
    /// change, so a problem is solved from the start or never.
    struct Idle(Arc<CallCounter>);

    impl SearchArchitecture for Idle {
        fn setup<'a>(
            &'a self,
            _problems: &[Arc<Problem>],
        ) -> Result<Box<dyn SearchState + 'a>, String> {
            Ok(Box::new(IdleState { counter: &self.0 }))
        }
    }

    struct IdleState<'a> {
        counter: &'a CallCounter,
    }

    impl SearchState for IdleState<'_> {
        fn step(
            &mut self,
            problems: &[Arc<Problem>],
            assignments: &mut [Vec<isize>],
            pending: &[bool],
            rng: &mut StdRng,
        ) {
            destroy_rows(self.counter, problems, assignments, pending, rng);
        }
    }

    fn three_binary_variables(unsatisfiable: bool) -> Arc<Problem> {
        let mut problem = Problem::default();
        let vars: Vec<_> = (0..3).map(|_| problem.add_variable(vec![0, 1])).collect();
        if unsatisfiable {
            crate::constraints::all_different(&mut problem, vars);
        }
        Arc::new(problem)
    }

    #[test]
    fn unsolved_problems_keep_being_destroyed_after_others_were_compacted_away() {
        let problems = vec![
            three_binary_variables(false),
            three_binary_variables(false),
            three_binary_variables(true),
            three_binary_variables(true),
        ];
        let counter = Arc::new(CallCounter {
            calls: Default::default(),
        });
        let search = LocalSearch::new(Box::new(Idle(counter.clone())));
        let budget = Budget {
            time_limit: Duration::MAX,
            iteration_limit: 250,
        };
        let solutions = search.run(&problems, budget, 0, None).unwrap();

        assert!(solutions[0].assignment().is_some() && solutions[1].assignment().is_some());
        assert!(solutions[2].assignment().is_none() && solutions[3].assignment().is_none());
        let calls = counter.calls.lock().unwrap();
        for unsolved in &problems[2..] {
            let key = Arc::as_ptr(unsolved) as usize;
            assert_eq!(calls[&key], 250);
        }
    }
}
