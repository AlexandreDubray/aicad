//! Local search: repeatedly destroys part of an assignment and repairs it, until the problem is
//! solved or the budget is spent. The repair step is guided by priors, which for now come from a
//! neural network (neural local search) or are uniform; destroy operators are non-neural.

#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

pub(crate) mod config;
pub(crate) mod decode;
pub(crate) mod prior;

pub(crate) use config::{DecodeKind, NetworkKind, PriorKind, SolveConfig};
pub(crate) use decode::{Decoder, Prepared};
pub(crate) use prior::{NetworkPrior, PriorHeuristic, UniformPrior};

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use burn::config::Config;
use burn::module::Module;
use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use crate::destroy::DestroyOperator;
use crate::learning::NetworkConfig;
use crate::modelling::{Problem, Solution};
use crate::utils::tensor::*;

/// Stopping criterion for the search; whichever limit is first reached stops it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Budget {
    /// Limit on the runtime.
    pub time_limit: Duration,
    /// Limit on the number of iterations.
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

/// Loads a network's hyperparameters (JSON config) and trained weights from a checkpoint
/// directory produced by `train_model`/`run_training`. Returns the config alongside the network
/// so callers also have access to training hyperparameters.
pub(crate) fn load_network<B, NC>(
    checkpoint_dir: &Path,
    problems: &[Arc<Problem>],
    device: &B::Device,
) -> Result<(NC, NC::N), Box<dyn std::error::Error>>
where
    B: Backend,
    NC: NetworkConfig<B> + Config + Clone,
    NC::N: Module<B>,
{
    // First, loads the config from the checkpoint
    let config: NC = NC::load(checkpoint_dir.join("config.json"))?;
    // Inits the network from the config and loads the weights.
    // The network initialisation is responsible for initiating problem-specific network's
    // component. For example, in ConsFormer, the variables positions are used in the positional
    // embedding. See each architecture's doc for further comment.
    let network = config.clone().init(problems, device).load_file(
        checkpoint_dir.join("weights"),
        &CompactRecorder::new(),
        device,
    )?;
    Ok((config, network))
}

/// Neural local search engine: repeatedly destroys part of each assignment and repairs it from
/// the priors, until every problem is solved or the budget is spent.
pub(crate) struct NeuralLocalSearch<B: Backend> {
    /// Source of the per-variable prior that guides each step (a trained network, uniform, ...)
    prior: Box<dyn PriorHeuristic<B>>,
    /// Heuristic for the destroy operator
    destroy_op: Box<dyn DestroyOperator>,
    /// How the priors become the next assignment
    decoder: Decoder,
    /// Devices used (cpu or gpu)
    device: B::Device,
}

impl<B: Backend> NeuralLocalSearch<B> {
    /// How often (in iterations) `run` physically drops solved problems out of the batch. See
    /// `run`'s doc for why this is periodic rather than immediate.
    const COMPACTION_INTERVAL: usize = 100;

    /// Builds the search engine with the given heuristics. The solver is independent of the
    /// problem being solved so it can be reused across batches of problems, which avoids the cost
    /// of reloading the network. The decoder keeps nothing between runs: what it needs per problem
    /// is prepared at the start of each run, which reports that time as preprocessing, and owned
    /// by the run.
    pub(crate) fn new(
        prior: Box<dyn PriorHeuristic<B>>,
        destroy_op: Box<dyn DestroyOperator>,
        decoder: Decoder,
        device: B::Device,
    ) -> Self {
        Self {
            prior,
            destroy_op,
            decoder,
            device,
        }
    }

    /// Runs the neural local search on a batch of problems with the allowed budget and the seed.
    /// Optionally, a progress bar is given to monitor the progress.
    ///
    /// The problems of a batch are stepped together: one evaluation of the prior (the network, on
    /// the device) and one decoding serve all of them.
    /// To evaluate the runtime of a method in seconds, run a few (e.g., 100) iterations
    /// of each problem alone and record the average step time; then run in large batches and
    /// infer the runtime from the number of steps observed, plus the preprocessing time.
    ///
    /// What the decoder prepared for a problem is dropped as soon as the problem is solved, and
    /// the rest when the run ends.
    ///
    /// # Panics
    ///
    /// If the problems do not all have the same number of variables.
    pub(crate) fn run(
        &self,
        problems: &[Arc<Problem>],
        budget: Budget,
        seed: u64,
        progress: Option<&indicatif::ProgressBar>,
    ) -> Vec<Solution> {
        if problems.is_empty() {
            return Vec::new();
        }
        // The tensors of a batch are rectangular.
        let n = problems[0].number_variables();
        assert!(
            problems.iter().all(|p| p.number_variables() == n),
            "all the problems of a batch must have the same number of variables"
        );

        let mut rng = StdRng::seed_from_u64(seed);
        let mut stop = StoppingCriterion::new(budget);

        // Sets of active problems indexes.
        let mut active: Vec<usize> = (0..problems.len()).collect();
        // Sets of active problems.
        let mut active_problems: Vec<Arc<Problem>> = problems.to_vec();
        // Sets of found solution for this batch.
        let mut solutions: Vec<Option<Solution>> = vec![None; problems.len()];

        let preprocessing_start = Instant::now();
        // What the decoder needs per problem, aligned with `active_problems`.
        let mut prepared: Vec<Prepared> = self.decoder.prepare(&active_problems);
        let preprocessing_runtime = preprocessing_start.elapsed().as_secs_f64();

        // Starts from random assignment
        // Tensor representations of the assignments. We need both a tensor representation for the
        // neural network and a CPU representation for the CPU-side processing (e.g., constraint
        // satisfaction check, mdd-decoding).
        let mut tensor_assignments = self.random_init(problems);
        // CPU representation of the assignments
        let mut cpu_assignments = to_rows(&tensor_assignments, problems.len(), n);

        // Continues while the budget is not finished, and there is unsolved problem in the batch
        while !stop.is_exhausted() && solutions.iter().any(Option::is_none) {
            // Get unsolved problems
            let pending: Vec<bool> = active
                .iter()
                .map(|&problem_idx| solutions[problem_idx].is_none())
                .collect();
            // Gets random seeds for the problems
            let seeds: Vec<u64> = (0..cpu_assignments.len())
                .map(|_| rng.random::<u64>())
                .collect();
            let destroy_op = &self.destroy_op;
            // Returns for each unsolved problem the index of the destroyed variables
            let destroyed_per_row: Vec<Vec<usize>> = crate::utils::worker_pool().install(|| {
                cpu_assignments
                    .par_iter()
                    .zip(active_problems.par_iter())
                    .zip(seeds.par_iter())
                    .zip(pending.par_iter())
                    .map(|(((row, problem), &seed), &is_pending)| {
                        // if the problem is solved, but not yet compacted out, return the empty
                        // vec: we don't touch it.
                        if !is_pending {
                            return Vec::new();
                        }
                        let mut row_rng = StdRng::seed_from_u64(seed);
                        destroy_op.destroy(problem, row, &mut row_rng)
                    })
                    .collect()
            });

            // We create a tensor for the masked data; priori heuristic may use it as they see fit
            // (e.g., consformer directly applies it in its architecture, uniform priors ignores
            // it).
            let mut destroy_mask_data = vec![0i64; cpu_assignments.len() * n];
            for (row_idx, destroyed) in destroyed_per_row.into_iter().enumerate() {
                for var in destroyed {
                    destroy_mask_data[row_idx * n + var] = 1;
                }
            }
            let destroy_mask: Tensor<B, 2, Int> =
                Tensor::<B, 1, Int>::from_data(destroy_mask_data.as_slice(), &self.device)
                    .reshape([cpu_assignments.len(), n]);

            // Actual call to get the priors. If the neural-network is used, this is where it is
            // called.
            let priors = self.prior.priors(
                &active_problems,
                &tensor_assignments,
                &destroy_mask,
                &self.device,
            );
            // Repair the solution based on the destroyed variable and the priors.
            tensor_assignments = self.decoder.decode(
                priors,
                destroy_mask,
                tensor_assignments,
                &active_problems,
                &prepared,
            );
            // Moves the new assignments back to the CPU for processing
            cpu_assignments = to_rows(&tensor_assignments, active_problems.len(), n);

            // One more iteration done.
            stop.tick();

            // Checks the newly solved problems
            let solved_now: Vec<bool> = crate::utils::worker_pool().install(|| {
                cpu_assignments
                    .par_iter()
                    .zip(active_problems.par_iter())
                    .zip(pending.par_iter())
                    .map(|((row, problem), &is_pending)| is_pending && problem.is_solution(row))
                    .collect()
            });

            // Updated the solutions *only* for newly solved problems.
            for (local_idx, &problem_idx) in active.iter().enumerate() {
                if !pending[local_idx] {
                    continue;
                }
                if solved_now[local_idx] {
                    solutions[problem_idx] = Some(
                        Solution::satisfiable(
                            &active_problems[local_idx],
                            cpu_assignments[local_idx].clone(),
                        )
                        .with_runtime(stop.start.elapsed().as_secs_f64(), preprocessing_runtime)
                        .with_iterations(stop.iters_done),
                    );
                    // The problem is done: frees what was prepared for it right away.
                    prepared[local_idx] = Prepared::Nothing;
                    if let Some(bar) = progress {
                        bar.inc(1);
                    }
                }
            }
            if let Some(bar) = progress {
                bar.set_message(format!("Solving instances (iteration {})", stop.iters_done));
            }

            if stop.iters_done.is_multiple_of(Self::COMPACTION_INTERVAL) {
                let before = active.len();
                let mut still_active = Vec::with_capacity(before);
                let mut still_active_problems = Vec::with_capacity(active_problems.len());
                let mut still_rows = Vec::with_capacity(cpu_assignments.len());
                let mut still_prepared = Vec::with_capacity(prepared.len());
                for (local_idx, &problem_idx) in active.iter().enumerate() {
                    if solutions[problem_idx].is_some() {
                        continue;
                    }
                    still_active.push(problem_idx);
                    still_active_problems.push(active_problems[local_idx].clone());
                    still_rows.push(cpu_assignments[local_idx].clone());
                    still_prepared.push(std::mem::replace(
                        &mut prepared[local_idx],
                        Prepared::Nothing,
                    ));
                }
                if still_active.len() < before {
                    active = still_active;
                    active_problems = still_active_problems;
                    cpu_assignments = still_rows;
                    prepared = still_prepared;
                    // Left stale when every remaining active problem just got compacted away --
                    // the `while` condition above exits before it's read again in that case.
                    if !active_problems.is_empty() {
                        tensor_assignments = rows_to_tensor(&cpu_assignments, n, &self.device);
                    }
                }
            }
        }

        solutions
            .into_iter()
            .map(|s| {
                s.unwrap_or_else(|| {
                    Solution::unknown()
                        .with_runtime(stop.start.elapsed().as_secs_f64(), preprocessing_runtime)
                        .with_iterations(stop.iters_done)
                })
            })
            .collect()
    }

    /// Random starting assignment, one row per problem, each value drawn from its domain.
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
    use burn::backend::ndarray::NdArray;

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
        use burn::backend::ndarray::NdArrayDevice;

        let problems = vec![
            three_binary_variables(false),
            three_binary_variables(false),
            three_binary_variables(true),
            three_binary_variables(true),
        ];
        let counter = Arc::new(CallCounter {
            calls: Default::default(),
        });
        struct Shared(Arc<CallCounter>);
        impl DestroyOperator for Shared {
            fn destroy(&self, p: &Problem, a: &[isize], rng: &mut StdRng) -> Vec<usize> {
                self.0.destroy(p, a, rng)
            }
        }
        let search = NeuralLocalSearch::<NdArray>::new(
            Box::new(UniformPrior::new(2)),
            Box::new(Shared(counter.clone())),
            // Nothing is destroyed, so the decoder never changes anything.
            Decoder::Argmax,
            NdArrayDevice::default(),
        );
        let budget = Budget {
            time_limit: Duration::MAX,
            iteration_limit: 250,
        };
        let solutions = search.run(&problems, budget, 0, None);

        assert!(solutions[0].assignment().is_some() && solutions[1].assignment().is_some());
        assert!(solutions[2].assignment().is_none() && solutions[3].assignment().is_none());
        let calls = counter.calls.lock().unwrap();
        for unsolved in &problems[2..] {
            let key = Arc::as_ptr(unsolved) as usize;
            assert_eq!(calls[&key], 250);
        }
    }
}
