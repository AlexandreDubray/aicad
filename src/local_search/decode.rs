//! Decoding of local search: how the priors of an iteration become the next assignment.
//!
//! The priors are unnormalised scores, one per value of each variable (a higher score is a more
//! likely value), whatever produced them: a trained network's logits, a uniform score, a
//! heuristic. A [`Decoder`] is one of the following strategies:
//!     - Argmax: always select the value with the highest prior
//!     - Sampling: sample proportionally to the softmax of the priors
//!     - DecomposedMddSampling: combine the priors with belief propagation over the problem's
//!       decomposed MDDs, one per constraint, before decoding
//!
//! # Per-problem state
//!
//! Some decoders need work that depends on the problem alone, not on the iteration
//! (DecomposedMddSampling compiles the constraints of the problem). [`Decoder::prepare`] does it
//! once per problem before the search and returns it as a [`Prepared`] value per problem. The
//! search loop owns these values and hands the matching ones back to [`Decoder::decode`]; it
//! drops the one of a problem as soon as that problem is solved, so the memory of a finished
//! problem is freed at once and the decoder itself keeps no state between runs.

use std::collections::HashMap;
use std::sync::Arc;

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::bp::belief_propagation;
use crate::mdd::arena::{MddCompilationConfig, compile_constraints};
use crate::mdd::{CompiledConstraint, MddArena};
use crate::modelling::{Problem, ValueIndex, VariableIndex};
use crate::utils::tensor::to_rows;
use crate::utils::{DecodeMode, argmax, sample_categorical};

/// What a decoder prepared for one problem, owned by the search loop while the problem is being
/// solved.
pub(crate) enum Prepared {
    /// The decoder needs nothing per problem, or the problem is solved and its state was dropped.
    Nothing,
    /// The decomposed MDDs of the problem, one compiled constraint per constraint. The `Arc` is
    /// shared by the copies of the same problem in a batch.
    Mdds(Arc<Vec<CompiledConstraint>>),
}

/// Combines the priors with belief propagation over the decomposed MDDs of the problem (one per
/// constraint, not one diagram for the whole problem), then takes the value of largest marginal or samples one, according to `mode`.
pub(crate) struct DecomposedMddSampling {
    /// How the constraints are compiled.
    compilation: MddCompilationConfig,
    /// Size of the nominal domain `0..domain_size` that the constraints are compiled over, which
    /// is the width of the priors, see `crate::mdd::arena`'s module doc.
    domain_size: usize,
    /// How a value is taken from a marginal.
    mode: DecodeMode,
    /// Rounds of belief propagation per decoding.
    bp_iterations: usize,
}

impl DecomposedMddSampling {
    /// `domain_size` must match the width of the priors: it is the nominal alphabet every shared
    /// constraint structure is compiled against, see `crate::mdd::arena`'s module doc.
    pub(crate) fn new(
        compilation: MddCompilationConfig,
        domain_size: usize,
        mode: DecodeMode,
        bp_iterations: usize,
    ) -> Self {
        Self {
            compilation,
            domain_size,
            mode,
            bp_iterations,
        }
    }

    /// Below this many distinct problems, `prepare` shows no progress bar: it would finish
    /// before it could be read.
    const PROGRESS_BAR_MIN_PROBLEMS: usize = 4;

    /// Compiles the constraints of every distinct problem of `problems` in parallel.
    ///
    /// The arena that shares the structures of the constraints is local to the call: it is
    /// dropped when `prepare` returns, and a diagram then lives exactly as long as the
    /// [`Prepared`] values that use it. A diagram that no other problem shares is therefore freed
    /// with its problem. An arena pays off when the constraints of the problems overlap a lot, as
    /// with one small diagram per constraint; with large grouped diagrams that nothing shares, it
    /// only adds a lookup, and the diagrams should be compiled directly into the list of the
    /// problem.
    fn prepare(&self, problems: &[Arc<Problem>]) -> Vec<Prepared> {
        let arena = MddArena::new();
        // The same problem can appear several times in a batch, for instance to run several
        // searches on it: it is compiled once.
        let mut first_of: HashMap<usize, usize> = HashMap::new();
        let mut unique: Vec<&Arc<Problem>> = Vec::new();
        let slots: Vec<usize> = problems
            .iter()
            .map(|problem| {
                *first_of
                    .entry(Arc::as_ptr(problem) as usize)
                    .or_insert_with(|| {
                        unique.push(problem);
                        unique.len() - 1
                    })
            })
            .collect();

        let compile = |problem: &&Arc<Problem>| {
            Arc::new(compile_constraints(
                problem,
                &arena,
                &self.compilation,
                self.domain_size,
            ))
        };
        let compiled: Vec<Arc<Vec<CompiledConstraint>>> =
            if unique.len() < Self::PROGRESS_BAR_MIN_PROBLEMS {
                crate::utils::worker_pool().install(|| unique.par_iter().map(compile).collect())
            } else {
                let progress = ProgressBar::new(unique.len() as u64);
                progress.set_style(
                    ProgressStyle::with_template(
                        "{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                    )
                    .expect("hard-coded progress bar template should always be valid"),
                );
                progress.set_message("Compiling MDDs");
                let compiled = crate::utils::worker_pool().install(|| {
                    unique
                        .par_iter()
                        .progress_with(progress.clone())
                        .map(compile)
                        .collect()
                });
                progress.finish_and_clear();
                compiled
            };

        slots
            .into_iter()
            .map(|slot| Prepared::Mdds(Arc::clone(&compiled[slot])))
            .collect()
    }

    /// Decodes the rows of `current` that have free positions. A row without any keeps its
    /// values and needs no `Prepared`.
    fn decode<B: Backend>(
        &self,
        priors: Tensor<B, 3>,
        destroy_mask: Tensor<B, 2, Int>,
        current: Tensor<B, 2, Int>,
        problems: &[Arc<Problem>],
        prepared: &[Prepared],
    ) -> Tensor<B, 2, Int> {
        let device = priors.device();
        let dims = current.dims();
        let (rows, n) = (dims[0], dims[1]);
        let domain_width = priors.dims()[2];

        let probs_flat: Vec<f32> = softmax(priors, 2)
            .into_data()
            .to_vec::<f32>()
            .expect("softmax output should be f32-convertible");
        let current_rows = to_rows(&current, rows, n);
        let mask_rows = to_rows(&destroy_mask, rows, n);

        let mut next_data = vec![0i64; rows * n];
        crate::utils::worker_pool().install(|| {
            next_data
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(row, next_row)| {
                    if mask_rows[row].iter().all(|&m| m == 0) {
                        // Nothing to change: a solved problem, whose state may be dropped.
                        for (v, next) in next_row.iter_mut().enumerate() {
                            *next = current_rows[row][v] as i64;
                        }
                        return;
                    }
                    let problem = &problems[row];
                    let Prepared::Mdds(mdds) = &prepared[row] else {
                        panic!(
                            "DecomposedMddSampling needs its problems prepared before it decodes them"
                        );
                    };

                    let mut assignment = vec![ValueIndex(0); n];
                    let mut decided = vec![false; n];
                    let mut probs: Vec<Vec<f64>> = Vec::with_capacity(n);
                    for v in 0..n {
                        let variable = VariableIndex(v);
                        // `mdds`' shared structures are compiled with no per-instance domain
                        // restriction (see `crate::mdd::arena`'s module doc), so a `ValueIndex`
                        // in one of their edges is exactly the raw domain value -- no translation
                        // through `problem`'s own (possibly narrower) domain enumeration.
                        assignment[v] = ValueIndex(current_rows[row][v] as usize);
                        // `destroy_mask == 1` marks a position as free to change this iteration --
                        // `decided` here is its opposite: everything the destroy/repair loop is
                        // holding fixed this round.
                        decided[v] = mask_rows[row][v] == 0;

                        // Masked the same way `layer_weights_from_probs` masks a training
                        // weight: a raw value outside `problem`'s actual (possibly narrowed --
                        // e.g. a fixed/given position) domain gets zero weight, even though the
                        // shared structure has a real edge for it -- see that function's doc.
                        let probs_v: Vec<f64> = (0..domain_width)
                            .map(|raw_value| {
                                if problem[variable].in_domain(raw_value as isize) {
                                    let offset =
                                        row * n * domain_width + v * domain_width + raw_value;
                                    probs_flat[offset] as f64
                                } else {
                                    0.0
                                }
                            })
                            .collect();
                        probs.push(probs_v);
                    }

                    let marginals =
                        belief_propagation(mdds, &probs, &assignment, &decided, self.bp_iterations);

                    for v in 0..n {
                        if mask_rows[row][v] == 0 {
                            // Untouched position -- keep the current value exactly, same contract
                            // `Argmax`/`Sampling` honour via `mask_where`.
                            next_row[v] = current_rows[row][v] as i64;
                            continue;
                        }
                        // `marginals[v]` is indexed by raw domain value directly (see the masked
                        // `probs_v` build above -- length `domain_width`, position == raw value),
                        // so `chosen` needs no translation back through `problem`'s own domain
                        // enumeration, unlike the pre-arena-split convention this replaced.
                        let chosen = match self.mode {
                            DecodeMode::Greedy => argmax(&marginals[v]),
                            DecodeMode::Sample => sample_categorical(&marginals[v]),
                        };
                        next_row[v] = chosen as i64;
                    }
                });
        });

        Tensor::<B, 1, Int>::from_data(next_data.as_slice(), &device).reshape([rows, n])
    }
}

/// How the priors of an iteration become the next assignment. Only positions flagged in
/// `destroy_mask` may change; everywhere else the current value is kept, regardless of what the
/// priors say there.
pub(crate) enum Decoder {
    /// Greedy / MAP decoding: takes the most likely value per variable.
    Argmax,
    /// Stochastic decoding: samples a value per variable from `softmax(priors / temperature)`.
    Sampling {
        /// Temperature of the softmax.
        temperature: f64,
    },
    /// Belief propagation over the compiled constraints, then a value per variable.
    DecomposedMddSampling(DecomposedMddSampling),
}

impl Decoder {
    /// Prepare the data needed for decoding (e.g., MDD compilation).
    /// Returns one [`Prepared`] per problem of `problems`, in order.
    ///
    /// The search calls it with the problems of a batch, in parallel over the problems, and
    /// reports its time apart from the search as the preprocessing time of the solutions.
    /// Decoding does not compile anything by itself: a problem that was not prepared cannot be
    /// decoded by MDD sampling.
    pub(crate) fn prepare(&self, problems: &[Arc<Problem>]) -> Vec<Prepared> {
        match self {
            Decoder::DecomposedMddSampling(decomposed) => decomposed.prepare(problems),
            Decoder::Argmax | Decoder::Sampling { .. } => {
                problems.iter().map(|_| Prepared::Nothing).collect()
            }
        }
    }

    /// `priors` has shape `[rows, variables, domain_size]`; `problems` and `prepared` have one
    /// entry per row.
    /// Returns an updated tensor-stored assignment for each problem.
    pub(crate) fn decode<B: Backend>(
        &self,
        priors: Tensor<B, 3>,
        destroy_mask: Tensor<B, 2, Int>,
        current: Tensor<B, 2, Int>,
        problems: &[Arc<Problem>],
        prepared: &[Prepared],
    ) -> Tensor<B, 2, Int> {
        match self {
            Decoder::Argmax => {
                let proposed: Tensor<B, 2, Int> = priors.argmax(2).squeeze_dim(2);
                current.mask_where(destroy_mask.equal_elem(1), proposed)
            }
            Decoder::Sampling { temperature } => {
                let device = priors.device();
                let u = Tensor::<B, 3>::random(
                    priors.dims(),
                    Distribution::Uniform(1e-20, 1.0),
                    &device,
                );
                let neg_log_u = -u.log(); // -ln(u), > 0 since u in (0, 1)
                let gumbel = -neg_log_u.log(); // Gumbel(0, 1) noise: -ln(-ln(u))

                let scaled = priors.div_scalar(*temperature) + gumbel;
                let proposed: Tensor<B, 2, Int> = scaled.argmax(2).squeeze_dim(2);
                current.mask_where(destroy_mask.equal_elem(1), proposed)
            }
            Decoder::DecomposedMddSampling(decomposed) => {
                decomposed.decode(priors, destroy_mask, current, problems, prepared)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::all_different;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    /// `size`-variable clique via a single `all_different` over `colours` values, compiled as its
    /// own diagram.
    fn clique_problem(size: usize, colours: usize) -> Arc<Problem> {
        let mut problem = Problem::default();
        let vars = problem.add_variables(size, (0..colours as isize).collect(), None);
        all_different(&mut problem, vars);
        Arc::new(problem)
    }

    fn decomposed(domain_size: usize) -> Decoder {
        Decoder::DecomposedMddSampling(DecomposedMddSampling::new(
            MddCompilationConfig::default(),
            domain_size,
            DecodeMode::Greedy,
            1,
        ))
    }

    /// The compiled constraints of a prepared problem.
    fn mdds(prepared: &Prepared) -> &Arc<Vec<CompiledConstraint>> {
        match prepared {
            Prepared::Mdds(mdds) => mdds,
            Prepared::Nothing => panic!("expected compiled constraints"),
        }
    }

    #[test]
    fn prepare_returns_one_state_per_problem_in_order() {
        let problems = vec![clique_problem(3, 3), clique_problem(4, 4)];
        let prepared = decomposed(4).prepare(&problems);
        assert_eq!(prepared.len(), 2);
        assert_eq!(mdds(&prepared[0]).len(), 1);
        assert_eq!(mdds(&prepared[1])[0].number_decisions(), 4);
    }

    #[test]
    fn prepare_compiles_each_distinct_problem_once_even_with_duplicates() {
        let a = clique_problem(6, 5);
        let b = clique_problem(4, 4);

        // `a` repeated three times (several searches on the same problem) plus `b` once: the
        // copies of `a` share one compiled list.
        let prepared = decomposed(5).prepare(&[a.clone(), a.clone(), a.clone(), b.clone()]);

        assert!(Arc::ptr_eq(mdds(&prepared[0]), mdds(&prepared[1])));
        assert!(Arc::ptr_eq(mdds(&prepared[0]), mdds(&prepared[2])));
        assert!(!Arc::ptr_eq(mdds(&prepared[0]), mdds(&prepared[3])));
    }

    #[test]
    fn the_other_decoders_prepare_nothing() {
        let problems = vec![clique_problem(3, 3)];
        assert!(matches!(
            Decoder::Argmax.prepare(&problems)[0],
            Prepared::Nothing
        ));
        assert!(matches!(
            Decoder::Sampling { temperature: 1.0 }.prepare(&problems)[0],
            Prepared::Nothing
        ));
    }

    #[test]
    fn decomposed_changes_only_the_free_positions_and_stays_in_the_domain() {
        let device = NdArrayDevice::default();
        let problems = vec![clique_problem(3, 3)];
        let decoder = decomposed(3);
        let prepared = decoder.prepare(&problems);

        let priors = Tensor::<NdArray, 3>::zeros([1, 3, 3], &device);
        let mask = Tensor::<NdArray, 2, Int>::from_data([[1, 0, 0]], &device);
        let current = Tensor::<NdArray, 2, Int>::from_data([[0, 1, 2]], &device);
        let next = decoder.decode(priors, mask, current, &problems, &prepared);

        let values: Vec<i64> = next.into_data().to_vec().unwrap();
        assert_eq!(values[1..], [1, 2]);
        assert!((0..3).contains(&values[0]));
    }

    #[test]
    fn a_row_without_free_position_is_kept_and_needs_no_prepared_state() {
        let device = NdArrayDevice::default();
        let problems = vec![clique_problem(3, 3)];
        let priors = Tensor::<NdArray, 3>::zeros([1, 3, 3], &device);
        let mask = Tensor::<NdArray, 2, Int>::zeros([1, 3], &device);
        let current = Tensor::<NdArray, 2, Int>::from_data([[2, 1, 0]], &device);
        let next = decomposed(3).decode(priors, mask, current, &problems, &[Prepared::Nothing]);

        let values: Vec<i64> = next.into_data().to_vec().unwrap();
        assert_eq!(values, [2, 1, 0]);
    }
}
