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
//! - [`neural_prior`]: a trained network evaluated at each step, whose logits are decoded
//!   directly ([`NeuralPriorSearch`]) or used as the prior of belief propagation over the MDDs of
//!   the constraints ([`NeuralPriorBpSearch`]). Any network of [`crate::neural`] can drive them.
//! - [`bp_search`]: belief propagation over the MDDs from uniform probabilities, on the CPU only.
//!
//! Each has its own settings, gathered in [`SearchConfig`]; [`repair`] holds the belief-propagation
//! repair the last two share.

#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

pub(crate) mod bp_search;
pub(crate) mod config;
pub(crate) mod neural_prior;
pub(crate) mod repair;

pub(crate) use bp_search::{BeliefPropagationSearch, BeliefPropagationSearchConfig};
pub(crate) use config::{Budget, SearchConfig, SolveConfig};
pub(crate) use neural_prior::{
    NeuralPriorBpSearch, NeuralPriorBpSearchConfig, NeuralPriorSearch, NeuralPriorSearchConfig,
};

use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use crate::destroy::DestroyOperator;
use crate::modelling::{Problem, Solution};

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
pub(crate) trait SearchArchitecture {
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
pub(crate) trait SearchState {
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
        .map(|problem| {
            problem
                .iter_variables()
                .map(|v| problem[v].sample())
                .collect()
        })
        .collect()
}

/// The variables each row frees, `destroy_op` choosing them with a seed drawn per row from `rng`.
/// A row that is not `pending` (solved, but not yet dropped) frees nothing.
pub(crate) fn destroy_rows(
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
pub(crate) fn row_seeds(rows: usize, rng: &mut StdRng) -> Vec<u64> {
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
pub(crate) struct LocalSearch {
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
    pub(crate) fn new(architecture: Box<dyn SearchArchitecture>) -> Self {
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
    pub(crate) fn run(
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
        let vars: Vec<_> = (0..3)
            .map(|_| problem.add_variable(vec![0, 1], None))
            .collect();
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
