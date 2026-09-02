//! Decoding operators for neural local search.
//! The following decoding strategies are implemented:
//!     - Use an argmax: Always select the value associated with the highest logit
//!     - Use a softmax: sample proportionnaly to the logits
//!     - Use belief propagation over the problem's compiled MDDs to turn the network's raw,
//!       per-position logits into constraint-propagated marginals before decoding

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};

use rayon::prelude::*;

use crate::learning::consformer::MddCompilationConfig;
use crate::mdd::Mdd;
use crate::modelling::{Problem, ValueIndex, VariableIndex};
use crate::sampling::bp::belief_propagation;
use crate::sampling::solve::value_to_index;
use crate::sampling::{DecodeMode, argmax, sample_categorical};
use crate::utils::tensor::to_rows;

/// Turns this iteration's logits into the next assignment. Only positions
/// flagged in `destroy_mask` may change; everywhere else the current value
/// is kept, regardless of what the network predicted there.
pub trait DecodingOperator<B: Backend>: Send + Sync {
    fn decode(
        &self,
        logits: Tensor<B, 3>,
        destroy_mask: Tensor<B, 2, Int>,
        current: Tensor<B, 2, Int>,
        problems: &[Arc<Problem>],
        population_size: usize,
    ) -> Tensor<B, 2, Int>;

    /// Whether `problem` is already known to be unsatisfiable, independent of anything a
    /// destroy/repair loop could ever decode -- e.g. a bucket-grouped clique of constraints
    /// (`BeliefPropagationDecode`'s compiled MDDs) with no accepting path at all, detected once
    /// during MDD compilation rather than left for the search to burn its whole budget failing to
    /// converge on. `NeuralLocalSearch::run` calls this once per problem, up front, and reports
    /// `Status::Unsatisfiable` immediately for any that answer `true`, instead of ever destroying,
    /// decoding, or checking `is_solution` on them.
    ///
    /// Default `false`: an operator with no static UNSAT-detection mechanism of its own (`Argmax`,
    /// `Sampling` -- neither one looks at the problem's constraints at all, only at logits) just
    /// defers to the search itself, which reports `Status::Unknown` if it never finds a solution
    /// within budget, same as before this existed.
    fn detect_unsat(&self, _problem: &Arc<Problem>) -> bool {
        false
    }
}

/// Greedy / MAP decoding: takes the most likely value per variable.
pub struct Argmax;

impl<B: Backend> DecodingOperator<B> for Argmax {
    fn decode(
        &self,
        logits: Tensor<B, 3>,
        destroy_mask: Tensor<B, 2, Int>,
        current: Tensor<B, 2, Int>,
        _problems: &[Arc<Problem>],
        _population_size: usize,
    ) -> Tensor<B, 2, Int> {
        let proposed: Tensor<B, 2, Int> = logits.argmax(2).squeeze_dim(2);
        current.mask_where(destroy_mask.equal_elem(1), proposed)
    }
}

/// Stochastic decoding: samples a value per variable from
/// `softmax(logits / temperature)`.
pub struct Sampling {
    pub temperature: f64,
}

impl<B: Backend> DecodingOperator<B> for Sampling {
    fn decode(
        &self,
        logits: Tensor<B, 3>,
        destroy_mask: Tensor<B, 2, Int>,
        current: Tensor<B, 2, Int>,
        _problems: &[Arc<Problem>],
        _population_size: usize,
    ) -> Tensor<B, 2, Int> {
        let device = logits.device();
        let u = Tensor::<B, 3>::random(logits.dims(), Distribution::Uniform(1e-20, 1.0), &device);
        let neg_log_u = -u.log(); // -ln(u), > 0 since u in (0, 1)
        let gumbel = -neg_log_u.log(); // Gumbel(0, 1) noise: -ln(-ln(u))

        let scaled = logits.div_scalar(self.temperature) + gumbel;
        let proposed: Tensor<B, 2, Int> = scaled.argmax(2).squeeze_dim(2);
        current.mask_where(destroy_mask.equal_elem(1), proposed)
    }
}

/// Compiles one MDD per group `compilation.grouping` puts `problem`'s constraints into, refining
/// each to full exactness -- mirrors `pyaicad::sequential_imputation::compile_problem_mdds`, kept
/// as its own small copy here rather than shared, the same way `sampling::bp`'s tests duplicate
/// `sampling`'s `build_mdd` helper: this module has no reason to depend on `pyaicad`.
fn compile_mdds_for(problem: &Arc<Problem>, compilation: &MddCompilationConfig) -> Vec<Mdd> {
    compilation
        .grouping
        .groups(problem)
        .into_iter()
        .map(|constraints| {
            let mut mdd = Mdd::new(
                Arc::clone(problem),
                compilation.ordering.clone(),
                compilation.merge,
                compilation.select,
                &constraints,
            );
            mdd.refine(usize::MAX);
            mdd
        })
        .collect()
}

/// Decodes destroyed positions from constraint-propagated marginals instead of the network's raw,
/// per-position logits: `Argmax`/`Sampling` decide each position independently, so two individually
/// plausible values can both get committed even though committing both together violates a
/// constraint they share. This runs `sampling::bp::belief_propagation` per row instead -- seeded
/// from that row's softmax probabilities, with every position `destroy_mask` doesn't cover clamped
/// to its current value as hard evidence (the same `decided` convention `partial_alpha_at` uses) --
/// and decodes the resulting marginals at the destroyed positions only.
///
/// MDDs are compiled once per distinct `Problem` (keyed by its `Arc` pointer) and cached for the
/// life of this operator, since `NeuralLocalSearch::run` calls `decode` every iteration on the same
/// handful of problems -- recompiling them from scratch each time would be repeated, non-negligible
/// work for no benefit (the problem's constraints don't change between iterations).
pub struct BeliefPropagationDecode {
    compilation: MddCompilationConfig,
    /// How many belief-propagation rounds to run per row, per iteration -- kept low, per
    /// `belief_propagation`'s own doc: a handful of rounds captures most of the benefit, and this
    /// runs inside every step of the outer destroy/repair loop besides.
    iterations: usize,
    /// How to turn each destroyed position's resulting marginal into a value -- greedy (argmax) or
    /// sampled.
    mode: DecodeMode,
    cache: Mutex<HashMap<usize, Arc<Vec<Mdd>>>>,
}

impl BeliefPropagationDecode {
    pub fn new(compilation: MddCompilationConfig, iterations: usize, mode: DecodeMode) -> Self {
        Self {
            compilation,
            iterations,
            mode,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// MDDs for `problem`, compiling and caching them on first sight -- keyed by the `Arc`'s
    /// pointer, which stays stable across `NeuralLocalSearch::run`'s whole call for any problem
    /// that hasn't been compacted out (`run`'s compaction step only ever clones the same `Arc`, it
    /// never rebuilds a `Problem`).
    ///
    /// `decode` calls this from every row in parallel (see its doc), so the lock is only ever held
    /// for the cache lookup/insert itself, not across the compile -- two rows racing on the very
    /// first sight of a problem can both miss the cache and both compile it once, but that's a rare
    /// one-time duplicate cost, not a correctness issue (whichever insert loses just has its result
    /// discarded in favour of the other's, both being equivalent). Serialising every problem's
    /// first compile behind one held lock instead would be worse across a whole batch of distinct
    /// problems.
    fn mdds_for(&self, problem: &Arc<Problem>) -> Arc<Vec<Mdd>> {
        let key = Arc::as_ptr(problem) as usize;
        {
            let cache = self.cache.lock().expect("mdd cache lock poisoned");
            if let Some(mdds) = cache.get(&key) {
                return Arc::clone(mdds);
            }
        }
        let mdds = Arc::new(compile_mdds_for(problem, &self.compilation));
        self.cache
            .lock()
            .expect("mdd cache lock poisoned")
            .insert(key, Arc::clone(&mdds));
        mdds
    }
}

impl<B: Backend> DecodingOperator<B> for BeliefPropagationDecode {
    fn decode(
        &self,
        logits: Tensor<B, 3>,
        destroy_mask: Tensor<B, 2, Int>,
        current: Tensor<B, 2, Int>,
        problems: &[Arc<Problem>],
        population_size: usize,
    ) -> Tensor<B, 2, Int> {
        let device = logits.device();
        let dims = current.dims();
        let (rows, n) = (dims[0], dims[1]);
        // The padded per-variable output width -- the network's vocabulary index for a value is
        // that raw value itself (see `crate::utils::tensor`'s doc), so this is exactly the stride
        // needed to slice one row's one variable's distribution out of the flattened softmax
        // output, same as `SequentialImputationSolver::probs_for` does for its own network pass.
        let domain_width = logits.dims()[2];

        let probs_flat: Vec<f32> = softmax(logits, 2)
            .into_data()
            .to_vec::<f32>()
            .expect("softmax output should be f32-convertible");
        let current_rows = to_rows(&current, rows, n);
        let mask_rows = to_rows(&destroy_mask, rows, n);

        // Every row's belief-propagation run is independent of every other row's, so rows are
        // decoded in parallel across CPU cores -- but each row's own `belief_propagation` call
        // stays single-threaded (see that function's doc): parallelising *both* levels at once
        // would oversubscribe the available cores, so only one of the two is parallel, and it's
        // this one -- with `population_size` copies per problem, a batch typically has far more
        // rows than `belief_propagation` has MDDs to loop over internally, so this is the layer
        // with more independent work to spread across cores.
        let mut next_data = vec![0i64; rows * n];
        next_data
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(row, next_row)| {
                let problem = &problems[row / population_size];
                let mdds = self.mdds_for(problem);

                let mut assignment = vec![ValueIndex(0); n];
                let mut decided = vec![false; n];
                let mut probs: Vec<Vec<f64>> = Vec::with_capacity(n);
                for v in 0..n {
                    let variable = VariableIndex(v);
                    assignment[v] = value_to_index(problem, variable, current_rows[row][v]);
                    // `destroy_mask == 1` marks a position as free to change this iteration --
                    // `decided` here is its opposite: everything the destroy/repair loop is
                    // holding fixed this round.
                    decided[v] = mask_rows[row][v] == 0;

                    let domain_size = problem[variable].domain_size();
                    let probs_v: Vec<f64> = (0..domain_size)
                        .map(|d| {
                            let value = problem[variable].value(ValueIndex(d));
                            let offset =
                                row * n * domain_width + v * domain_width + value as usize;
                            probs_flat[offset] as f64
                        })
                        .collect();
                    probs.push(probs_v);
                }

                let marginals =
                    belief_propagation(&mdds, &probs, &assignment, &decided, self.iterations);

                for v in 0..n {
                    if mask_rows[row][v] == 0 {
                        // Untouched position -- keep the current value exactly, same contract
                        // `Argmax`/`Sampling` honour via `mask_where`.
                        next_row[v] = current_rows[row][v] as i64;
                        continue;
                    }
                    let chosen = match self.mode {
                        DecodeMode::Greedy => argmax(&marginals[v]),
                        DecodeMode::Sample => sample_categorical(&marginals[v]),
                    };
                    next_row[v] = problem[VariableIndex(v)].value(ValueIndex(chosen)) as i64;
                }
            });

        Tensor::<B, 1, Int>::from_data(next_data.as_slice(), &device).reshape([rows, n])
    }

    /// `true` if any of `problem`'s compiled MDDs (see `compile_mdds_for`) has no accepting path
    /// at all (`Mdd::is_unsat`) -- e.g. `mdd_grouping_size_bound` bucketed a clique of constraints
    /// too tight to ever be satisfied together, such as a 6-clique in a 5-colouring problem's
    /// bucket. Reuses `mdds_for`'s cache, so calling this before the first `decode` doesn't cost a
    /// second compilation -- whichever call happens first compiles and caches, the other just
    /// reads the cache.
    fn detect_unsat(&self, problem: &Arc<Problem>) -> bool {
        self.mdds_for(problem).iter().any(Mdd::is_unsat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modelling::all_different;
    use burn::backend::ndarray::NdArray;

    /// A `size`-clique modelled directly as one `all_different` constraint over `colours` values --
    /// exactly the "clique too tight for the number of colours" scenario the user described (a
    /// 6-clique needing 6 colours in a 5-colouring problem). Modelled as a single constraint
    /// (rather than `size * (size - 1) / 2` pairwise `not_equals`) so it always compiles to its own
    /// one MDD regardless of `compilation.grouping` -- a set of pairwise `not_equals` constraints
    /// covering the same clique can end up split across several buckets by
    /// `ConstraintGrouping`'s elimination-order-driven bucketing (each bucket keyed by its
    /// *earliest-eliminated* scope variable, so two edges that don't share an "earliest" variable
    /// never land in the same bucket even at a large `size_bound`), in which case no single
    /// compiled MDD would ever see the whole clique and `detect_unsat` would miss it -- `1 <=
    /// colours < size` makes the clique itself unsatisfiable no matter how it's grouped, so using
    /// `all_different` sidesteps that entirely.
    fn clique_problem(size: usize, colours: usize) -> Arc<Problem> {
        let mut problem = Problem::default();
        let domain: Vec<isize> = (0..colours as isize).collect();
        let vars: Vec<VariableIndex> = (0..size)
            .map(|_| problem.add_variable(domain.clone(), None))
            .collect();
        all_different(&mut problem, vars);
        Arc::new(problem)
    }

    #[test]
    fn detect_unsat_is_true_when_a_clique_has_fewer_colours_than_variables() {
        let op = BeliefPropagationDecode::new(MddCompilationConfig::default(), 5, DecodeMode::Greedy);
        let problem = clique_problem(6, 5);
        assert!(DecodingOperator::<NdArray>::detect_unsat(&op, &problem));
    }

    #[test]
    fn detect_unsat_is_false_when_a_clique_has_enough_colours() {
        let op = BeliefPropagationDecode::new(MddCompilationConfig::default(), 5, DecodeMode::Greedy);
        let problem = clique_problem(6, 6);
        assert!(!DecodingOperator::<NdArray>::detect_unsat(&op, &problem));
    }

    #[test]
    fn detect_unsat_caches_so_a_second_call_does_not_recompile() {
        let op = BeliefPropagationDecode::new(MddCompilationConfig::default(), 5, DecodeMode::Greedy);
        let problem = clique_problem(6, 5);
        assert!(DecodingOperator::<NdArray>::detect_unsat(&op, &problem));
        // Second call must read back the same (cached) UNSAT MDDs, not silently recompute
        // something different.
        assert!(DecodingOperator::<NdArray>::detect_unsat(&op, &problem));
        assert_eq!(op.cache.lock().unwrap().len(), 1);
    }
}
