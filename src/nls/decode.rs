//! Decoding operators for neural local search.
//! The following decoding strategies are implemented:
//!     - Use an argmax: Always select the value associated with the highest logit
//!     - Use a softmax: sample proportionnaly to the logits
//!     - Mdd-sampling: combine the network's raw, per-position logits with one round of belief
//!       propagation over the problem's compiled MDDs before decoding

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::learning::consformer::MddCompilationConfig;
use crate::mdd::arena::compile_constraint;
use crate::mdd::{CompiledConstraint, MddArena};
use crate::modelling::{Problem, ValueIndex, VariableIndex};
use crate::sampling::bp::belief_propagation;
use crate::sampling::{argmax, sample_categorical, DecodeMode};
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
    ) -> Tensor<B, 2, Int>;

    fn detect_unsat(&self, _problem: &Arc<Problem>) -> bool {
        false
    }
    fn prepare(&self, _problems: &[Arc<Problem>]) {}
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

/// Compiles `problem`'s constraints into `CompiledConstraint`s, sharing each constraint's
/// structure (via `arena`) with every other structurally-identical constraint -- whether from
/// this problem or any other problem this `MddCache` (and hence its `arena`) ever compiles for.
/// See `crate::mdd::arena`'s module doc.
fn compile_constraints_for(
    problem: &Arc<Problem>,
    arena: &MddArena,
    compilation: &MddCompilationConfig,
    domain_size: usize,
) -> Vec<CompiledConstraint> {
    problem
        .iter_constraints()
        .map(|c| {
            compile_constraint(
                arena,
                problem,
                c,
                &compilation.ordering,
                domain_size,
                compilation.max_width,
            )
        })
        .collect()
}

struct MddCache {
    compilation: MddCompilationConfig,
    /// Nominal domain size every constraint's shared structure is compiled against -- see
    /// `crate::mdd::arena`'s module doc. Must match the network's own configured domain size,
    /// same convention as `mdd_dataset::compile_constraint_mdds`'s `domain_size` parameter.
    domain_size: usize,
    /// Shared across every problem this cache ever compiles for -- structurally-identical
    /// constraints from different problems dedup here, not just within one problem's own
    /// constraints.
    arena: MddArena,
    /// Per-problem cache of already-compiled `CompiledConstraint`s (their own `order`s are
    /// specific to that problem, even when their `structure`s come from `arena` and may be
    /// shared).
    cache: Mutex<HashMap<usize, Arc<Vec<CompiledConstraint>>>>,
}

impl MddCache {
    fn new(compilation: MddCompilationConfig, domain_size: usize) -> Self {
        Self {
            compilation,
            domain_size,
            arena: MddArena::new(),
            cache: Mutex::new(HashMap::new()),
        }
    }

    fn mdds_for(&self, problem: &Arc<Problem>) -> Arc<Vec<CompiledConstraint>> {
        let key = Arc::as_ptr(problem) as usize;
        {
            let cache = self.cache.lock().expect("mdd cache lock poisoned");
            if let Some(mdds) = cache.get(&key) {
                return Arc::clone(mdds);
            }
        }
        let mdds = Arc::new(compile_constraints_for(
            problem,
            &self.arena,
            &self.compilation,
            self.domain_size,
        ));
        self.cache
            .lock()
            .expect("mdd cache lock poisoned")
            .insert(key, Arc::clone(&mdds));
        mdds
    }

    /// Below this many distinct problems, `prepare` skips the indicatif progress bar
    /// entirely and just compiles. Two reasons: the bar would finish before a human could
    /// ever see it for a handful of problems, and -- the one that actually matters since
    /// `prepare` runs once per chunk (see `chunked_run`'s doc) -- a fair per-instance
    /// benchmark's `--batch-size 1` means *every* chunk hits `prepare` with exactly one
    /// problem, so without this guard a `--max-concurrent-chunks 32` sweep would spawn up
    /// to 32 independent `ProgressBar`s drawing to the terminal at once from different
    /// threads with no coordination between them (indicatif doesn't serialize unrelated
    /// bars unless they share a `MultiProgress`), garbling the output -- purely cosmetic,
    /// never a correctness issue, but worth just not doing.
    const PROGRESS_BAR_MIN_PROBLEMS: usize = 4;

    fn prepare(&self, problems: &[Arc<Problem>]) {
        let mut seen = HashSet::new();
        let unique: Vec<&Arc<Problem>> = problems
            .iter()
            .filter(|p| seen.insert(Arc::as_ptr(*p) as usize))
            .collect();

        if unique.len() < Self::PROGRESS_BAR_MIN_PROBLEMS {
            crate::utils::worker_pool().install(|| {
                unique.into_par_iter().for_each(|problem| {
                    self.mdds_for(problem);
                });
            });
            return;
        }

        let progress = ProgressBar::new(unique.len() as u64);
        progress.set_style(
            ProgressStyle::with_template(
                "{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .expect("hard-coded progress bar template should always be valid"),
        );
        progress.set_message("Compiling MDDs");

        crate::utils::worker_pool().install(|| {
            unique
                .into_par_iter()
                .progress_with(progress.clone())
                .for_each(|problem| {
                    self.mdds_for(problem);
                });
            progress.finish_and_clear();
        });
    }
}

pub struct MddSamplingDecode {
    mdds: MddCache,
    mode: DecodeMode,
    bp_iterations: usize,
}

impl MddSamplingDecode {
    /// `domain_size` must match the network's configured domain size -- it's the nominal alphabet
    /// every shared constraint-group structure is compiled against, see `crate::mdd::arena`'s
    /// module doc and `MddCache::domain_size`.
    pub fn new(
        compilation: MddCompilationConfig,
        domain_size: usize,
        mode: DecodeMode,
        bp_iterations: usize,
    ) -> Self {
        Self {
            mdds: MddCache::new(compilation, domain_size),
            mode,
            bp_iterations,
        }
    }

    fn mdds_for(&self, problem: &Arc<Problem>) -> Arc<Vec<CompiledConstraint>> {
        self.mdds.mdds_for(problem)
    }
}

impl<B: Backend> DecodingOperator<B> for MddSamplingDecode {
    fn decode(
        &self,
        logits: Tensor<B, 3>,
        destroy_mask: Tensor<B, 2, Int>,
        current: Tensor<B, 2, Int>,
        problems: &[Arc<Problem>],
    ) -> Tensor<B, 2, Int> {
        let device = logits.device();
        let dims = current.dims();
        let (rows, n) = (dims[0], dims[1]);
        let domain_width = logits.dims()[2];

        let probs_flat: Vec<f32> = softmax(logits, 2)
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
                    let problem = &problems[row];
                    let mdds = self.mdds_for(problem);

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

                    let marginals = belief_propagation(
                        &mdds,
                        &probs,
                        &assignment,
                        &decided,
                        self.bp_iterations,
                    );

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

    fn detect_unsat(&self, problem: &Arc<Problem>) -> bool {
        // Every constraint's shared structure is compiled with no per-instance domain restriction
        // (see `crate::mdd::arena`'s module doc), so `structure.is_unsat()` only ever reflects a
        // shape that's unconditionally unsatisfiable -- an instance-specific hint (e.g. a Sudoku
        // given) that makes THIS `problem` unsatisfiable no longer shows up here.
        self.mdds_for(problem)
            .iter()
            .any(|constraint| constraint.structure.is_unsat())
    }

    fn prepare(&self, problems: &[Arc<Problem>]) {
        self.mdds.prepare(problems);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modelling::all_different;
    use burn::backend::ndarray::NdArray;

    /// `size`-variable clique via a single `all_different` over `colours < size` values -- always
    /// compiled as its own MDD, so this exercises `detect_unsat` straightforwardly.
    fn clique_problem(size: usize, colours: usize) -> Arc<Problem> {
        let mut problem = Problem::default();
        let vars = problem.add_variables(size, (0..colours as isize).collect(), None);
        all_different(&mut problem, vars);
        Arc::new(problem)
    }

    /// `domain_size` must match the shape's true nominal arity for `detect_unsat` to see a
    /// pigeonhole (too-few-colours) case as *unconditionally* unsat (see `MddSamplingDecode`'s
    /// `detect_unsat` doc): compiling the shared `AllDifferent` template against a domain_size
    /// larger than the real clique's colour count would make the generic template satisfiable
    /// even though this particular instance's own colours don't reach that far -- which is now a
    /// per-instance restriction, caught via `belief_propagation`'s weight masking at decode time,
    /// not via structural unsat. The two structural `detect_unsat` tests below pass the clique's
    /// own colour count as `domain_size` specifically to stay a test of structural unsat.
    fn mdd_sampling_decode(domain_size: usize) -> MddSamplingDecode {
        MddSamplingDecode::new(
            MddCompilationConfig::default(),
            domain_size,
            DecodeMode::Greedy,
            1,
        )
    }

    #[test]
    fn detect_unsat_is_true_when_a_clique_has_fewer_colours_than_variables() {
        let problem = clique_problem(6, 5);
        let op = mdd_sampling_decode(5);
        assert!(<MddSamplingDecode as DecodingOperator<NdArray>>::detect_unsat(&op, &problem));
    }

    #[test]
    fn detect_unsat_is_false_when_a_clique_has_enough_colours() {
        let problem = clique_problem(6, 6);
        let op = mdd_sampling_decode(6);
        assert!(
            !<MddSamplingDecode as DecodingOperator<NdArray>>::detect_unsat(&op, &problem)
        );
    }

    #[test]
    fn detect_unsat_caches_so_a_second_call_does_not_recompile() {
        let problem = clique_problem(6, 5);
        let op = mdd_sampling_decode(5);
        let first = op.mdds_for(&problem);
        let second = op.mdds_for(&problem);
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn prepare_warms_the_cache_so_decode_never_needs_to_compile() {
        let problem = clique_problem(6, 5);
        let op = mdd_sampling_decode(5);
        <MddSamplingDecode as DecodingOperator<NdArray>>::prepare(&op, &[problem.clone()]);

        // `mdds_for` after `prepare` must be a pure cache hit -- calling it twice more should
        // keep returning the exact same `Arc`, never a freshly compiled one.
        let first = op.mdds_for(&problem);
        let second = op.mdds_for(&problem);
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn prepare_compiles_each_distinct_problem_once_even_with_duplicates() {
        let a = clique_problem(6, 5);
        let b = clique_problem(4, 4);
        let op = mdd_sampling_decode(5);

        // `a` repeated three times (multiple search samples of the same problem) plus `b` once --
        // `prepare` must still only compile 2 distinct problems, not 4.
        <MddSamplingDecode as DecodingOperator<NdArray>>::prepare(
            &op,
            &[a.clone(), a.clone(), a.clone(), b.clone()],
        );

        assert_eq!(op.mdds.cache.lock().unwrap().len(), 2);
    }
}
