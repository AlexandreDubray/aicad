//! Dataset for the MDD-WMC ConsFormer training recipe: for each problem, compiles one exact MDD
//! per constraint group. The loss (see `loss.rs`) walks these `Mdd`s directly via
//! `crate::mdd::wmc`, so no padded/bucketed tensor representation is built here anymore.

use std::sync::Arc;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::learning::BatchProblems;
use crate::mdd::arena::compile_constraint;
use crate::mdd::heuristics::{MergeHeuristic, OrderingHeuristic, SelectHeuristic};
use crate::mdd::{CompiledConstraint, MddArena};
use crate::modelling::Problem;

use super::dataset::{
    consformer_mask_data, stack_masks_and_sample_assignments, ConsFormerMaskData,
};
use super::{ConsFormerDataConfig, ConsFormerInputs};

#[derive(Clone, Debug)]
pub struct MddCompilationConfig {
    pub ordering: OrderingHeuristic,
    pub merge: MergeHeuristic,
    pub select: SelectHeuristic,
    pub max_width: usize,
}

impl Default for MddCompilationConfig {
    fn default() -> Self {
        Self {
            ordering: OrderingHeuristic::MinDomMaxLinked,
            merge: MergeHeuristic::LessRelaxed,
            select: SelectHeuristic::Greedy,
            max_width: usize::MAX,
        }
    }
}

/// Compiles one `CompiledConstraint` per constraint of `problem`. Each constraint's *structure*
/// is looked up/compiled once in `arena` and shared with every other structurally-identical
/// constraint (from this problem or any other sharing the same `arena`).
/// Always compiles/refines the shared structure to full exactness (`usize::MAX`), regardless of
/// `compilation.max_width`.
///
/// `domain_size` must match the network's configured `ConsFormerConfig::domain_size`: it's the
/// nominal alphabet every shared template is compiled against, and the index the training loss
/// reads a raw domain value's weight from in the network's per-variable probability vector (see
/// `crate::learning::consformer::loss::layer_weights_from_probs`), so this validates up front
/// that every real variable actually referenced by a compiled constraint's scope has its whole
/// domain inside `[0, domain_size)` -- a mismatch there would otherwise silently read/write the
/// wrong slot instead of panicking loudly.
fn compile_constraint_mdds(
    problem: &Arc<Problem>,
    arena: &MddArena,
    compilation: &MddCompilationConfig,
    domain_size: usize,
) -> Vec<CompiledConstraint> {
    problem
        .iter_constraints()
        .map(|c| {
            let label = problem[c].name();

            for variable in problem[c].iter_scope() {
                for value in problem[variable].iter_domain() {
                    assert!(
                        value >= 0 && (value as usize) < domain_size,
                        "constraint `{}`: domain value {} out of the network's [0, {}) range -- \
                         `domain_size` passed to the MDD dataset doesn't match the network's \
                         configured domain_size",
                        label,
                        value,
                        domain_size,
                    );
                }
            }

            // Always compiled exactly, regardless of `compilation.max_width` -- see this
            // function's doc.
            let compiled = compile_constraint(
                arena,
                problem,
                c,
                &compilation.ordering,
                domain_size,
                usize::MAX,
            );

            if compiled.structure.is_unsat() {
                log::warn!(
                    "MDD constraint `{}`'s shape is unconditionally unsatisfiable -- its \
                     compiled structure has no accepting path regardless of instance, so its \
                     WMC will always be 0.",
                    label,
                );
            }

            compiled
        })
        .collect()
}

/// Sample used to train ConsFormer-MDD. Carries the same attention/var masks as the classical
/// `ConsFormerSample` (see `consformer_masks`), plus one `CompiledConstraint` per constraint of
/// `problem` -- each group's (possibly shared, see `crate::mdd::arena`) structure plus its own
/// real branching order. `mdds` is `Arc`-wrapped since `Dataset::get` clones the sample on every
/// access (once per epoch, per batch), and isn't free to deep-copy repeatedly.
pub struct ConsFormerMddSample<B: Backend> {
    problem: Arc<Problem>,
    attention_mask: Tensor<B, 2, Bool>,
    var_mask: Tensor<B, 1, Bool>,
    mdds: Arc<Vec<CompiledConstraint>>,
}

impl<B: Backend> ConsFormerMddSample<B> {
    pub fn problem(&self) -> &Arc<Problem> {
        &self.problem
    }

    pub fn attention_mask(&self) -> &Tensor<B, 2, Bool> {
        &self.attention_mask
    }

    pub fn var_mask(&self) -> &Tensor<B, 1, Bool> {
        &self.var_mask
    }

    pub fn mdds(&self) -> &Arc<Vec<CompiledConstraint>> {
        &self.mdds
    }
}

impl<B: Backend> Clone for ConsFormerMddSample<B> {
    fn clone(&self) -> Self {
        Self {
            problem: Arc::clone(&self.problem),
            attention_mask: self.attention_mask.clone(),
            var_mask: self.var_mask.clone(),
            mdds: Arc::clone(&self.mdds),
        }
    }
}

impl<B: Backend> std::fmt::Debug for ConsFormerMddSample<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsFormerMddSample")
            .field("number_vars", &self.problem.number_variables())
            .field("number_mdds", &self.mdds.len())
            .finish()
    }
}

/// A dataset is a vector of samples, one per problem, each with its constraint groups' MDDs
/// precompiled.
pub struct ConsFormerMddDataset<B: Backend> {
    samples: Vec<ConsFormerMddSample<B>>,
}

impl<B: Backend> ConsFormerMddDataset<B> {
    /// Compiles one `CompiledConstraint` per constraint for every problem, sharing structures
    /// across every group (from any problem in this call, or from any earlier/later call given
    /// the same `arena`) whose shape matches -- see `crate::mdd::arena`'s module doc.
    /// `data_config.domain_size` must match the network's configured `ConsFormerConfig::domain_size`
    /// -- see `compile_constraint_mdds`. Build `data_config` via
    /// `ConsFormerDataConfig::from(&network_config)` rather than by hand, so this and the
    /// `ConsFormerMddBatcher` built alongside it can't end up with different `domain_size`s.
    ///
    /// `arena` is a plain `&MddArena` rather than owned: callers building several datasets that
    /// should share compiled structures (e.g. `pyaicad::learn::run_training_mdd`'s train and
    /// validation datasets) pass the same `Arc<MddArena>` (dereferenced) to each call.
    pub fn new(
        problems: Vec<Arc<Problem>>,
        arena: &MddArena,
        compilation: MddCompilationConfig,
        data_config: ConsFormerDataConfig,
        device: &B::Device,
    ) -> Self {
        let domain_size = data_config.domain_size;
        let progress = ProgressBar::new(problems.len() as u64);
        progress.set_style(
            ProgressStyle::with_template(
                "{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .expect("hard-coded progress bar template should always be valid"),
        );
        progress.set_message("Compiling MDDs");

        let per_problem: Vec<(ConsFormerMaskData, Vec<CompiledConstraint>)> =
            crate::utils::worker_pool().install(|| {
                problems
                    .par_iter()
                    .progress_with(progress.clone())
                    .map(|problem| {
                        (
                            consformer_mask_data(problem),
                            compile_constraint_mdds(problem, arena, &compilation, domain_size),
                        )
                    })
                    .collect()
            });
        progress.finish_and_clear();

        let samples = problems
            .into_iter()
            .zip(per_problem)
            .map(|(problem, (mask_data, mdds))| {
                let (attention_mask, var_mask) = mask_data.into_tensors::<B>(device);
                ConsFormerMddSample {
                    problem,
                    attention_mask,
                    var_mask,
                    mdds: Arc::new(mdds),
                }
            })
            .collect();
        Self { samples }
    }
}

impl<B: Backend> Dataset<ConsFormerMddSample<B>> for ConsFormerMddDataset<B> {
    fn get(&self, index: usize) -> Option<ConsFormerMddSample<B>> {
        self.samples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

/// Batch used to train ConsFormer-MDD. Carries the same attention/var-mask/assignment inputs as
/// the classical `ConsFormerBatch` (see `ConsFormerInputs`), plus every sample's compiled MDDs.
pub struct ConsFormerMddBatch<B: Backend> {
    /// (batch_size, number_vars, number_vars)
    pub attention_masks: Tensor<B, 3, Bool>,
    /// (batch_size, number_vars)
    pub var_masks: Tensor<B, 2, Bool>,
    /// (batch_size, number_vars)
    pub assignments: Tensor<B, 2, Int>,
    /// Problems, used to compute satisfaction reports/metrics.
    pub problems: Vec<Arc<Problem>>,
    /// Every sample's compiled constraint groups, index-aligned with `problems`.
    pub mdds: Vec<Arc<Vec<CompiledConstraint>>>,
}

impl<B: Backend> Clone for ConsFormerMddBatch<B> {
    fn clone(&self) -> Self {
        ConsFormerMddBatch {
            attention_masks: self.attention_masks.clone(),
            var_masks: self.var_masks.clone(),
            assignments: self.assignments.clone(),
            problems: self.problems.clone(),
            mdds: self.mdds.clone(),
        }
    }
}

impl<B: Backend> std::fmt::Debug for ConsFormerMddBatch<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsFormerMddBatch")
            .field("attention_masks", &self.attention_masks)
            .field("var_masks", &self.var_masks)
            .field("problems", &format!("{} problem(s)", self.problems.len()))
            .field(
                "mdds",
                &format!(
                    "{} MDD(s)",
                    self.mdds.iter().map(|m| m.len()).sum::<usize>()
                ),
            )
            .finish()
    }
}

impl<B: Backend> BatchProblems<B> for ConsFormerMddBatch<B> {
    fn problems(&self) -> &[Arc<Problem>] {
        &self.problems
    }
}

impl<B: Backend> ConsFormerInputs<B> for ConsFormerMddBatch<B> {
    fn attention_masks(&self) -> Tensor<B, 3, Bool> {
        self.attention_masks.clone()
    }

    fn var_masks(&self) -> Tensor<B, 2, Bool> {
        self.var_masks.clone()
    }

    fn assignments(&self) -> Tensor<B, 2, Int> {
        self.assignments.clone()
    }
}

/// Builds `ConsFormerMddBatch`es from `ConsFormerMddSample`s.
#[derive(Clone, Copy)]
pub struct ConsFormerMddBatcher {
    mask_fraction: f64,
}

impl ConsFormerMddBatcher {
    pub fn new(data_config: ConsFormerDataConfig) -> Self {
        Self {
            mask_fraction: data_config.mask_fraction,
        }
    }
}

impl<B: Backend> Batcher<B, ConsFormerMddSample<B>, ConsFormerMddBatch<B>>
    for ConsFormerMddBatcher
{
    fn batch(
        &self,
        samples: Vec<ConsFormerMddSample<B>>,
        device: &B::Device,
    ) -> ConsFormerMddBatch<B> {
        let attention_mask_tensors: Vec<Tensor<B, 2, Bool>> =
            samples.iter().map(|s| s.attention_mask.clone()).collect();
        let var_mask_tensors: Vec<Tensor<B, 1, Bool>> =
            samples.iter().map(|s| s.var_mask.clone()).collect();
        let problems: Vec<Arc<Problem>> = samples.iter().map(|s| Arc::clone(&s.problem)).collect();
        let mdds: Vec<Arc<Vec<CompiledConstraint>>> =
            samples.iter().map(|s| Arc::clone(&s.mdds)).collect();

        let (attention_masks, var_masks, assignments) = stack_masks_and_sample_assignments(
            attention_mask_tensors,
            &var_mask_tensors,
            &problems,
            self.mask_fraction,
            device,
        );

        ConsFormerMddBatch {
            attention_masks,
            var_masks,
            assignments,
            problems,
            mdds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modelling::{all_different, among, not_equals, sum};

    /// `real_domain_size` is what `problem[variable].domain_size()` used to be before the arena
    /// split -- since the group's structure is now compiled against the full nominal
    /// `domain_size` regardless of any real narrowing, this builds the same masked weights
    /// `crate::learning::consformer::loss::layer_weights_from_probs` does (every real variable
    /// here has an unnarrowed `0..domain_size` domain, so the mask is a no-op, but this keeps the
    /// test helper honest about what production code actually builds).
    fn brute_force_wmc(
        constraint: &CompiledConstraint,
        real_problem: &Problem,
        probs: &[f64],
        domain_size: usize,
    ) -> f64 {
        let num_layers = constraint.number_layers() - 1;
        let mut weights = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let variable = constraint.decision_at_layer(layer);
            let mut layer_weights = vec![0.0; domain_size];
            for (raw_value, w) in layer_weights.iter_mut().enumerate() {
                if real_problem[variable].in_domain(raw_value as isize) {
                    *w = probs[variable.0 * domain_size + raw_value];
                }
            }
            weights.push(layer_weights);
        }
        crate::mdd::wmc::wmc(constraint, &weights)
    }

    #[test]
    fn compile_constraint_mdds_matches_brute_force_for_not_equals() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);

        let domain_size = 3;
        let arena = MddArena::new();
        let mdds = compile_constraint_mdds(
            &problem,
            &arena,
            &MddCompilationConfig::default(),
            domain_size,
        );
        assert_eq!(mdds.len(), 1);
        assert_eq!(mdds[0].number_layers() - 1, 2);

        let probs: Vec<f64> = vec![
            0.2, 0.5, 0.3, // x
            0.1, 0.3, 0.6, // y
        ];
        let wmc = brute_force_wmc(&mdds[0], &problem, &probs, domain_size);

        let mut brute = 0.0;
        for xv in 0..domain_size {
            for yv in 0..domain_size {
                if xv != yv {
                    brute += probs[xv] * probs[domain_size + yv];
                }
            }
        }
        assert!((wmc - brute).abs() < 1e-9, "wmc={wmc} brute={brute}");
    }

    #[test]
    fn compile_constraint_mdds_matches_brute_force_for_all_different_permutation() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        all_different(&mut problem, vars.clone());
        let problem = Arc::new(problem);

        let domain_size = 3;
        let arena = MddArena::new();
        let mdds = compile_constraint_mdds(
            &problem,
            &arena,
            &MddCompilationConfig::default(),
            domain_size,
        );
        assert_eq!(mdds.len(), 1);
        assert_eq!(mdds[0].number_layers() - 1, 3);

        let probs: Vec<f64> = vec![
            0.5, 0.3, 0.2, // var 0
            0.2, 0.2, 0.6, // var 1
            0.1, 0.4, 0.5, // var 2
        ];
        let wmc = brute_force_wmc(&mdds[0], &problem, &probs, domain_size);

        let mut brute = 0.0;
        for a in 0..domain_size {
            for b in 0..domain_size {
                for c in 0..domain_size {
                    if a != b && b != c && a != c {
                        brute += probs[a] * probs[domain_size + b] * probs[2 * domain_size + c];
                    }
                }
            }
        }
        assert!((wmc - brute).abs() < 1e-9, "wmc={wmc} brute={brute}");
    }

    #[test]
    fn compile_constraint_mdds_matches_brute_force_for_sum() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        sum(&mut problem, vars.clone(), 3);
        let problem = Arc::new(problem);

        let domain_size = 3;
        let arena = MddArena::new();
        let mdds = compile_constraint_mdds(
            &problem,
            &arena,
            &MddCompilationConfig::default(),
            domain_size,
        );

        let probs: Vec<f64> = vec![
            0.5, 0.3, 0.2, // var 0
            0.2, 0.2, 0.6, // var 1
            0.1, 0.4, 0.5, // var 2
        ];
        let wmc = brute_force_wmc(&mdds[0], &problem, &probs, domain_size);

        let mut brute = 0.0;
        for a in 0..domain_size {
            for b in 0..domain_size {
                for c in 0..domain_size {
                    if a + b + c == 3 {
                        brute += probs[a] * probs[domain_size + b] * probs[2 * domain_size + c];
                    }
                }
            }
        }
        assert!((wmc - brute).abs() < 1e-9, "wmc={wmc} brute={brute}");
    }

    #[test]
    fn compile_constraint_mdds_matches_brute_force_for_among() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        among(&mut problem, vars.clone(), vec![1, 2], 1, 2);
        let problem = Arc::new(problem);

        let domain_size = 3;
        let arena = MddArena::new();
        let mdds = compile_constraint_mdds(
            &problem,
            &arena,
            &MddCompilationConfig::default(),
            domain_size,
        );

        let probs: Vec<f64> = vec![
            0.5, 0.3, 0.2, // var 0
            0.2, 0.2, 0.6, // var 1
            0.1, 0.4, 0.5, // var 2
        ];
        let wmc = brute_force_wmc(&mdds[0], &problem, &probs, domain_size);

        let mut brute = 0.0;
        for a in 0..domain_size {
            for b in 0..domain_size {
                for c in 0..domain_size {
                    let count = [a, b, c].iter().filter(|&&v| v == 1 || v == 2).count();
                    if (1..=2).contains(&count) {
                        brute += probs[a] * probs[domain_size + b] * probs[2 * domain_size + c];
                    }
                }
            }
        }
        assert!((wmc - brute).abs() < 1e-9, "wmc={wmc} brute={brute}");
    }

    #[test]
    fn unsat_constraint_gets_zero_wmc() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1], None);
        all_different(&mut problem, vars.clone());
        let problem = Arc::new(problem);

        let domain_size = 2;
        let arena = MddArena::new();
        let mdds = compile_constraint_mdds(
            &problem,
            &arena,
            &MddCompilationConfig::default(),
            domain_size,
        );
        let probs: Vec<f64> = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        assert_eq!(
            brute_force_wmc(&mdds[0], &problem, &probs, domain_size),
            0.0
        );
    }

    #[test]
    #[should_panic(expected = "out of the network's")]
    fn compile_constraint_mdds_panics_when_domain_size_is_too_small() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);

        let arena = MddArena::new();
        compile_constraint_mdds(&problem, &arena, &MddCompilationConfig::default(), 2);
    }

    /// End-to-end: `ConsFormerMddDataset::new` compiles MDDs and builds mask tensors for many
    /// problems in parallel (see the `par_iter` in `compile_constraint_mdds` and `new` itself).
    /// This checks that parallel construction still (a) preserves problem order and (b) produces
    /// results identical to compiling the same problems one at a time -- i.e. that splitting the
    /// work across threads didn't introduce any cross-problem interference.
    #[test]
    fn dataset_construction_is_order_preserving_under_parallelism() {
        use burn::backend::ndarray::{NdArray, NdArrayDevice};

        let mut problems = Vec::new();
        for domain_max in 0isize..40 {
            let mut problem = Problem::default();
            let domain: Vec<isize> = (0..=(domain_max % 5 + 2)).collect();
            let vars = problem.add_variables(3, domain, None);
            all_different(&mut problem, vars.clone());
            not_equals(&mut problem, vars[0], vars[1]);
            problems.push(Arc::new(problem));
        }

        let device = NdArrayDevice::default();
        let domain_size = 7;
        let data_config = ConsFormerDataConfig {
            domain_size,
            mask_fraction: 0.0,
        };

        let arena = MddArena::new();
        let dataset = ConsFormerMddDataset::<NdArray>::new(
            problems.clone(),
            &arena,
            MddCompilationConfig::default(),
            data_config,
            &device,
        );

        assert_eq!(dataset.len(), problems.len());
        for (i, problem) in problems.iter().enumerate() {
            let sample = dataset.get(i).unwrap();
            assert!(Arc::ptr_eq(sample.problem(), problem));

            let expected_arena = MddArena::new();
            let expected = compile_constraint_mdds(
                problem,
                &expected_arena,
                &MddCompilationConfig::default(),
                domain_size,
            );
            let actual = sample.mdds();
            assert_eq!(actual.len(), expected.len());
            for (a, e) in actual.iter().zip(expected.iter()) {
                assert_eq!(a.number_layers(), e.number_layers());
                assert_eq!(a.number_nodes(), e.number_nodes());
                assert_eq!(a.number_edges(), e.number_edges());
            }
        }
    }

    #[test]
    fn batcher_carries_every_samples_mdds_through_in_order() {
        use burn::backend::ndarray::{NdArray, NdArrayDevice};

        let mut problems = Vec::new();
        for _ in 0..3 {
            let mut problem = Problem::default();
            let vars = problem.add_variables(3, vec![0, 1, 2], None);
            all_different(&mut problem, vars.clone());
            not_equals(&mut problem, vars[0], vars[1]);
            problems.push(Arc::new(problem));
        }

        let device = NdArrayDevice::default();
        let domain_size = 3;
        let data_config = ConsFormerDataConfig {
            domain_size,
            mask_fraction: 0.0,
        };

        let arena = MddArena::new();
        let dataset = ConsFormerMddDataset::<NdArray>::new(
            problems.clone(),
            &arena,
            MddCompilationConfig::default(),
            data_config,
            &device,
        );
        let samples: Vec<ConsFormerMddSample<NdArray>> = (0..dataset.len())
            .map(|i| dataset.get(i).unwrap())
            .collect();
        let expected_counts: Vec<usize> = samples.iter().map(|s| s.mdds().len()).collect();

        let batcher = ConsFormerMddBatcher::new(data_config);
        let batch = batcher.batch(samples, &device);

        assert_eq!(batch.mdds.len(), problems.len());
        for (actual, expected) in batch.mdds.iter().zip(&expected_counts) {
            assert_eq!(actual.len(), *expected);
        }
        assert_eq!(batch.attention_masks.dims(), [3, 3, 3]);
        assert_eq!(batch.var_masks.dims(), [3, 3]);
        assert_eq!(batch.assignments.dims(), [3, 3]);
    }
}
