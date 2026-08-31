//! Sequential-imputation solving: builds a full assignment by sampling every variable once, in a
//! freshly shuffled order, from its exact per-MDD conditional given only the variables already
//! imputed earlier in *that same* attempt -- every variable not yet decided is marginalised out
//! entirely. This is repeated for up to a fixed number of steps, re-deriving the guiding probabilities
//! (typically a network forward pass) from the previous step's assignment, until one attempt
//! satisfies the problem or the step budget runs out.
//!
//! See "Sequential Imputations and Bayesian Missing Data Problem" (Kong, Liu & Wong, 1994) for reference
//! on sequential imputation.

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::{Duration, Instant};

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use rand::seq::SliceRandom;
use rayon::prelude::*;

use crate::learning::{Batch, Network};
use crate::modelling::{Problem, ValueIndex, VariableIndex};
use crate::utils::tensor::rows_to_tensor;

use super::{argmax, sample_categorical, DecodeMode, MddSampler};

/// Outcome of `solve`/`solve_batch` for one problem.
pub struct ImputationResult {
    /// The last assignment built, whether or not it satisfies the problem.
    pub assignment: Vec<ValueIndex>,
    /// How many sequential-imputation attempts actually ran for this problem.
    pub steps: usize,
    /// Whether `assignment` satisfies every constraint of the problem it was built for.
    pub satisfied: bool,
    /// Wall-clock time from the call starting to this result being produced -- the moment this
    /// problem was found solved, or the whole call's total elapsed time if it never was.
    pub elapsed: Duration,
}

/// One sequential-imputation attempt: visits every variable exactly once, in a freshly shuffled
/// order, sampling each from `sampler`'s combined per-MDD conditional given only the variables
/// visited earlier in this same call. `decided` starts all-false, so the first variable in the
/// order is automatically drawn from its plain marginal.
fn sequential_imputation(
    sampler: &MddSampler,
    probs: &[Vec<f64>],
    mode: DecodeMode,
) -> Vec<ValueIndex> {
    let n = sampler.number_variables();
    let mut order: Vec<VariableIndex> = (0..n).map(VariableIndex).collect();
    crate::utils::with_rng(|rng| order.shuffle(rng));

    let mut assignment: Vec<ValueIndex> = vec![ValueIndex(0); n];
    let mut decided = vec![false; n];

    for &var in &order {
        let combined = sampler.combined_partial_conditional(var, probs, &assignment, &decided);
        assignment[var.0] = ValueIndex(match mode {
            DecodeMode::Greedy => argmax(&combined),
            DecodeMode::Sample => sample_categorical(&combined),
        });
        decided[var.0] = true;
    }

    assignment
}

/// Finds `value`'s index in `variable`'s domain -- the inverse of `Variable::value`, needed to
/// turn a raw sampled/network value back into the `ValueIndex` the MDD/imputation machinery works
/// in.
fn value_to_index(problem: &Problem, variable: VariableIndex, value: isize) -> ValueIndex {
    let index = problem[variable]
        .iter_domain()
        .position(|v| v == value)
        .unwrap_or_else(|| {
            panic!(
                "value {value} is not in variable {}'s domain -- current assignment is \
                 inconsistent with the problem it's paired with",
                variable.0
            )
        });
    ValueIndex(index)
}

/// Tensor-native front end for `solve_batch`, mirroring `nls::NeuralLocalSearch`: built once
/// around a loaded network (and reusable across several `run` calls, e.g. one per chunk of a
/// larger problem list), it owns every bit of tensor bookkeeping a step needs -- building the
/// active rows' assignment/destroy-mask tensors, running the network, and turning its softmax
/// output back into the per-variable `Vec<f64>` conditionals `MddSampler` needs -- so callers
/// never have to touch a `Tensor` themselves.
pub struct SequentialImputationSolver<B: Backend, N, Ba> {
    network: N,
    /// Width of the network's per-variable output head (the padded max domain size across every
    /// problem the network was trained on) -- needed to slice each problem's own domain out of
    /// the shared-width logits.
    domain_size: usize,
    device: B::Device,
    _batch: PhantomData<Ba>,
}

impl<B: Backend, N, Ba> SequentialImputationSolver<B, N, Ba>
where
    Ba: Batch<B>,
    N: Network<B, Ba>,
{
    pub fn new(network: N, domain_size: usize, device: B::Device) -> Self {
        Self {
            network,
            domain_size,
            device,
            _batch: PhantomData,
        }
    }

    /// Runs sequential imputation on `problems` (all sharing the same `number_variables()`),
    /// starting each from a random per-variable assignment. See `solve_batch`'s doc for the
    /// step-by-step semantics (active-set dropout, `max_steps`, `time_limit`); this only adds the
    /// network forward pass that produces each step's guiding probabilities.
    pub fn run(
        &self,
        problems: &[Arc<Problem>],
        samplers: &[MddSampler],
        mode: DecodeMode,
        max_steps: usize,
        time_limit: Option<Duration>,
    ) -> Vec<ImputationResult> {
        let n = problems[0].number_variables();
        let initial = self.random_init(problems);
        let problem_refs: Vec<&Problem> = problems.iter().map(|p| p.as_ref()).collect();

        self.solve_batch(
            &problem_refs,
            samplers,
            mode,
            initial,
            max_steps,
            time_limit,
            |active, active_assignments| self.probs_for(problems, active, active_assignments, n),
        )
    }

    /// Runs sequential imputation on many problems at once, so a single `probs_fn` call -- typically
    /// one batched network forward pass -- can cover every still-unsolved problem's next attempt in
    /// one shot instead of one problem at a time. Each step:
    ///
    ///   1. Collects the indices of problems that haven't been solved yet ("active").
    ///   2. Calls `probs_fn(active, active_assignments)` once, with `active_assignments[k]` the current
    ///      assignment of problem `active[k]` -- `probs_fn` must return one `probs` per entry of
    ///      `active`, in the same order.
    ///   3. Runs one `sequential_imputation` attempt per active problem *in parallel* (via rayon),
    ///      each against its own `samplers[i]`.
    ///   4. Marks any problem whose new attempt satisfies it as solved (it drops out of `active` from
    ///      the next step on -- `probs_fn` never sees a solved problem again).
    ///
    /// Stops once every problem is solved, `max_steps` attempts have been made, or (if given)
    /// `time_limit` has elapsed since the call started -- whichever comes first. A problem still
    /// unsolved when the run stops reports its last attempt and however many steps actually ran (which
    /// may be fewer than `max_steps` if `time_limit` cut the run short).
    ///
    /// `problems`, `samplers`, and `initial` must all have the same length, one entry per problem, in
    /// the same order.
    pub fn solve_batch(
        &self,
        problems: &[&Problem],
        samplers: &[MddSampler],
        mode: DecodeMode,
        initial: Vec<Vec<ValueIndex>>,
        max_steps: usize,
        time_limit: Option<Duration>,
        mut probs_fn: impl FnMut(&[usize], &[Vec<ValueIndex>]) -> Vec<Vec<Vec<f64>>>,
    ) -> Vec<ImputationResult> {
        if problems.len() != samplers.len() || problems.len() != initial.len() {
            panic!(
                "Solving batch problems with sequential inputs but problems ({}), samplers ({}) and initials ({}) don't have the same length",
                problems.len(),
                samplers.len(),
                initial.len()
            );
        }

        let num_problems = problems.len();
        let mut assignments = initial;
        let mut solved_at: Vec<Option<(usize, Duration)>> = vec![None; num_problems]; // (step, elapsed)
        let start = Instant::now();
        let mut steps_run = 0;

        for step in 0..max_steps {
            if time_limit.is_some_and(|limit| start.elapsed() >= limit) {
                break;
            }
            let active: Vec<usize> = (0..num_problems)
                .filter(|&i| solved_at[i].is_none())
                .collect();
            if active.is_empty() {
                break;
            }
            steps_run = step + 1;

            let active_assignments: Vec<Vec<ValueIndex>> =
                active.iter().map(|&i| assignments[i].clone()).collect();
            let probs_batch = probs_fn(&active, &active_assignments);
            assert_eq!(
                probs_batch.len(),
                active.len(),
                "probs_fn must return exactly one distribution set per active problem"
            );

            let updated: Vec<(usize, Vec<ValueIndex>)> = active
                .par_iter()
                .zip(probs_batch.par_iter())
                .map(|(&i, probs)| (i, sequential_imputation(&samplers[i], probs, mode)))
                .collect();

            for (i, new_assignment) in updated {
                let raw: Vec<isize> = (0..problems[i].number_variables())
                    .map(|v| problems[i][VariableIndex(v)].value(new_assignment[v]))
                    .collect();
                if problems[i].is_solution(&raw) {
                    solved_at[i] = Some((steps_run, start.elapsed()));
                }
                assignments[i] = new_assignment;
            }
        }

        let total_elapsed = start.elapsed();
        (0..num_problems)
            .map(|i| ImputationResult {
                assignment: assignments[i].clone(),
                steps: solved_at[i].map(|(step, _)| step).unwrap_or(steps_run),
                satisfied: solved_at[i].is_some(),
                elapsed: solved_at[i]
                    .map(|(_, elapsed)| elapsed)
                    .unwrap_or(total_elapsed),
            })
            .collect()
    }

    /// Starts from a random assignment; note that each variable is sampled given its domain, so
    /// hinted/fixed variables are taken into account.
    fn random_init(&self, problems: &[Arc<Problem>]) -> Vec<Vec<ValueIndex>> {
        problems
            .iter()
            .map(|problem| {
                (0..problem.number_variables())
                    .map(|v| {
                        let variable = VariableIndex(v);
                        value_to_index(problem, variable, problem[variable].sample())
                    })
                    .collect()
            })
            .collect()
    }

    /// One batched forward pass, covering exactly the currently-active rows: builds their
    /// assignment tensor (every position masked, matching training's MDD-guided recipe -- see this
    /// module's doc), runs the network, and slices each active problem's own domain back out of
    /// the shared-width softmax output.
    fn probs_for(
        &self,
        problems: &[Arc<Problem>],
        active: &[usize],
        active_assignments: &[Vec<ValueIndex>],
        n: usize,
    ) -> Vec<Vec<Vec<f64>>> {
        let rows = active.len();
        let active_problems: Vec<Arc<Problem>> =
            active.iter().map(|&i| Arc::clone(&problems[i])).collect();

        let raw_rows: Vec<Vec<isize>> = active_problems
            .iter()
            .zip(active_assignments)
            .map(|(problem, assignment)| {
                (0..n)
                    .map(|v| problem[VariableIndex(v)].value(assignment[v]))
                    .collect()
            })
            .collect();
        let assignments = rows_to_tensor::<B>(&raw_rows, n, &self.device);
        let destroy_mask: Tensor<B, 2, Int> =
            Tensor::<B, 1, Int>::from_data(vec![1i64; rows * n].as_slice(), &self.device)
                .reshape([rows, n]);
        let batch =
            Ba::for_assignments(&active_problems, 1, assignments, destroy_mask, &self.device);
        let logits = self.network.forward(&batch);
        let probs_flat: Vec<f32> = softmax(logits, 2)
            .into_data()
            .to_vec::<f32>()
            .expect("softmax output should be f32-convertible");

        let domain_size = self.domain_size;
        (0..rows)
            .map(|row| {
                let problem = &active_problems[row];
                (0..n)
                    .map(|v| {
                        let variable = VariableIndex(v);
                        let ds = problem[variable].domain_size();
                        (0..ds)
                            .map(|d| {
                                let value = problem[variable].value(ValueIndex(d));
                                let offset =
                                    row * n * domain_size + v * domain_size + value as usize;
                                probs_flat[offset] as f64
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
}
