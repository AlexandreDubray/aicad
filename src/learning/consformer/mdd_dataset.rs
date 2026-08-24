//! Dataset for the MDD-WMC ConsFormer training recipe: for each problem, compiles one exact MDD
//! per constraint, then reduces each MDD to a flat, padded representation that a later batched
//! DP pass (see the loss, added in a following step) can turn into a weighted model count without
//! ever walking the `Mdd` graph again at training time.

use std::collections::HashMap;
use std::sync::Arc;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::learning::BatchProblems;
use crate::mdd::heuristics::{
    ConstraintGrouping, MergeHeuristic, OrderingHeuristic, SelectHeuristic,
};
use crate::mdd::{Mdd, NodeIndex};
use crate::modelling::{ConstraintIndex, Problem};

use super::dataset::{
    consformer_mask_data, stack_masks_and_sample_assignments, ConsFormerMaskData,
};
use super::{ConsFormerDataConfig, ConsFormerInputs};

#[derive(Clone, Debug)]
pub struct MddCompilationConfig {
    pub ordering: OrderingHeuristic,
    pub merge: MergeHeuristic,
    pub select: SelectHeuristic,
    pub grouping: ConstraintGrouping,
}

impl Default for MddCompilationConfig {
    fn default() -> Self {
        Self {
            ordering: OrderingHeuristic::MinDomMaxLinked,
            merge: MergeHeuristic::LessRelaxed,
            select: SelectHeuristic::Greedy,
            grouping: ConstraintGrouping::PER_CONSTRAINT,
        }
    }
}

/// Padding bucket for a constraint's MDD, computed purely from its own shape. Two MDDs that land
/// on the same key have identically-shaped `MddInstance` arrays, so their WMC computation can
/// later be batched into one matmul chain instead of one per instance; two MDDs of very different
/// size never share a key, so a single large constraint (e.g. a wide `Sum`) can't force wasteful
/// padding on small ones (e.g. `NotEquals`).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MddBucketKey {
    /// Number of decision layers, i.e. the constraint's scope size. Exact, not padded: MDDs are
    /// only ever grouped with others of the identical scope size.
    pub num_layers: usize,
    /// Padding bound on nodes per layer: the next power of two at or above the widest node-layer
    /// this MDD actually has. Used to size the DP's running node-weight vector.
    pub max_nodes: usize,
    /// Padding bound on edges per layer: the next power of two at or above the most edges any
    /// single decision layer of this MDD actually has.
    pub max_edges: usize,
}

impl MddBucketKey {
    fn for_shape(num_layers: usize, raw_max_nodes: usize, raw_max_edges: usize) -> Self {
        MddBucketKey {
            num_layers,
            max_nodes: raw_max_nodes.max(1).next_power_of_two(),
            max_edges: raw_max_edges.max(1).next_power_of_two(),
        }
    }
}

#[derive(Clone)]
pub struct MddInstance {
    pub bucket: MddBucketKey,
    /// Flat local index into a per-sample-flattened `(number_vars * domain_size)` probability
    /// vector, one per padded `(layer, edge_slot)` cell. Not yet offset by this problem's
    /// position within a training batch -- offsetting by `position_in_batch * number_vars *
    /// domain_size` is left to the batcher.
    pub gather_index: Vec<i64>,
    /// Same shape as `gather_index`: true where that edge slot is a real MDD edge, false where
    /// it's padding.
    pub edge_mask: Vec<bool>,
    /// Local index (within `bucket.max_nodes`, in layer `l + 1`'s node-layer) of that edge's
    /// target node. Meaningless (left at 0) on padding slots.
    pub edge_to: Vec<i64>,
    /// Local index (within `bucket.max_nodes`, in layer `l`'s node-layer) of that edge's source
    /// node. Meaningless (left at 0) on padding slots.
    pub edge_from: Vec<i64>,
    pub constraints: Vec<ConstraintIndex>,
    pub label: String,
}

impl std::fmt::Debug for MddInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MddInstance")
            .field("label", &self.label)
            .field("bucket", &self.bucket)
            .field(
                "active_edges",
                &self.edge_mask.iter().filter(|&&m| m).count(),
            )
            .finish()
    }
}

/// One layer's worth of active edges, collected before the padding bound for the whole instance
/// is known.
struct RawEdge {
    from_idx: usize,
    to_idx: usize,
    flat_index: i64,
}

/// Human-readable label for a compiled group, used in diagnostics (`MddInstance::label`, unsat
/// warnings, out-of-range assertions) -- the group's constraint name(s), joined.
fn describe_group(problem: &Problem, constraints: &[ConstraintIndex]) -> String {
    constraints
        .iter()
        .map(|&c| problem[c].name())
        .collect::<Vec<_>>()
        .join(", ")
}

fn compile_constraint_mdds(
    problem: &Arc<Problem>,
    compilation: &MddCompilationConfig,
    domain_size: usize,
) -> Vec<MddInstance> {
    let groups = compilation.grouping.groups(problem);
    groups
        .into_par_iter()
        .map(|constraints| {
            let mut mdd = Mdd::new(
                Arc::clone(problem),
                compilation.ordering.clone(),
                compilation.merge.clone(),
                compilation.select.clone(),
                &constraints,
            );
            mdd.refine(usize::MAX);

            let label = describe_group(problem, &constraints);

            if mdd.is_unsat() {
                log::warn!(
                    "MDD group `{}` is unsatisfiable given its own scope's domains -- its \
                     compiled MDD has no accepting path, so its WMC will always be 0. This \
                     usually means a fixed/hint value already violates one of its constraints.",
                    label,
                );
            }

            // `Mdd::number_layers()` counts node-layers (decision layers + 1, the sink layer);
            // the number of *decision* layers -- and hence this constraint's scope size -- is one
            // less.
            let num_layers = mdd.number_layers() - 1;

            // First pass: walk every layer's active edges without committing to a padding bound
            // yet, since `max_nodes`/`max_edges` need the true maxima across the whole MDD first.
            let mut raw_edges: Vec<Vec<RawEdge>> = Vec::with_capacity(num_layers);
            let mut raw_max_nodes = 1;
            for layer in 0..num_layers {
                let variable = mdd.decision_at_layer(layer);
                raw_max_nodes = raw_max_nodes.max(mdd.number_nodes_in_layer(layer));
                let mut layer_edges = Vec::new();
                for from_idx in 0..mdd.number_nodes_in_layer(layer) {
                    let from_node = NodeIndex(layer, from_idx);
                    if !mdd[from_node].is_active() {
                        continue;
                    }
                    for edge_index in mdd[from_node].iter_children() {
                        let edge = &mdd[edge_index];
                        if !edge.is_active() {
                            continue;
                        }
                        let to_idx = edge.to().1;

                        let domain_value = problem[variable].value(edge.assignment());
                        // A real `assert!`, not `debug_assert!`: this fires only when the caller
                        // (ultimately, whatever `domain_size` the Python entrypoint was given)
                        // doesn't match the problem's own domains, which is a bad-input error, not
                        // an internal-logic bug -- it should surface as a clear panic in release
                        // builds too, rather than silently producing an out-of-range
                        // `gather_index` that corrupts training instead of failing loudly.
                        assert!(
                            domain_value >= 0 && (domain_value as usize) < domain_size,
                            "group `{}`: domain value {} out of the network's [0, {}) \
                             range -- `domain_size` passed to the MDD dataset doesn't match the \
                             network's configured domain_size",
                            label,
                            domain_value,
                            domain_size,
                        );
                        let flat_index =
                            variable.0 as i64 * domain_size as i64 + domain_value as i64;

                        layer_edges.push(RawEdge {
                            from_idx,
                            to_idx,
                            flat_index,
                        });
                    }
                }
                raw_edges.push(layer_edges);
            }
            raw_max_nodes = raw_max_nodes.max(mdd.number_nodes_in_layer(num_layers));
            let raw_max_edges = raw_edges.iter().map(Vec::len).max().unwrap_or(1);

            let bucket = MddBucketKey::for_shape(num_layers, raw_max_nodes, raw_max_edges);
            let max_edges = bucket.max_edges;

            let mut gather_index = vec![0i64; num_layers * max_edges];
            let mut edge_mask = vec![false; num_layers * max_edges];
            let mut edge_to = vec![0i64; num_layers * max_edges];
            let mut edge_from = vec![0i64; num_layers * max_edges];

            for (layer, layer_edges) in raw_edges.into_iter().enumerate() {
                debug_assert!(
                    layer_edges.len() <= max_edges,
                    "layer has more edges than the computed padding bound"
                );
                for (slot, edge) in layer_edges.into_iter().enumerate() {
                    let cell = layer * max_edges + slot;
                    gather_index[cell] = edge.flat_index;
                    edge_mask[cell] = true;
                    edge_to[cell] = edge.to_idx as i64;
                    edge_from[cell] = edge.from_idx as i64;
                }
            }

            MddInstance {
                bucket,
                gather_index,
                edge_mask,
                edge_to,
                edge_from,
                constraints,
                label,
            }
        })
        .collect()
}

/// Sample used to train ConsFormer-MDD. Carries the same attention/var masks as the classical
/// `ConsFormerSample` (see `consformer_masks`), plus one `MddInstance` per constraint of
/// `problem`. `constraint_mdds` is `Arc`-wrapped since `Dataset::get` clones the sample on every
/// access (once per epoch, per batch), and an `MddInstance`'s arrays are not free to deep-copy
/// repeatedly.
#[derive(Clone)]
pub struct ConsFormerMddSample<B: Backend> {
    problem: Arc<Problem>,
    attention_mask: Tensor<B, 2, Bool>,
    var_mask: Tensor<B, 1, Bool>,
    constraint_mdds: Arc<Vec<MddInstance>>,
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

    pub fn constraint_mdds(&self) -> &Arc<Vec<MddInstance>> {
        &self.constraint_mdds
    }
}

impl<B: Backend> std::fmt::Debug for ConsFormerMddSample<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsFormerMddSample")
            .field("number_vars", &self.problem.number_variables())
            .field("number_constraint_mdds", &self.constraint_mdds.len())
            .finish()
    }
}

/// A dataset is a vector of samples, one per problem, each with its constraints' MDDs precompiled
/// and reduced to `MddInstance`s.
pub struct ConsFormerMddDataset<B: Backend> {
    samples: Vec<ConsFormerMddSample<B>>,
}

impl<B: Backend> ConsFormerMddDataset<B> {
    /// Compiles one exact MDD per constraint for every problem and reduces each to its padded
    /// `MddInstance` representation. `data_config.domain_size` must match the network's
    /// configured `ConsFormerConfig::domain_size` -- see `compile_constraint_mdds`. Build
    /// `data_config` via `ConsFormerDataConfig::from(&network_config)` rather than by hand, so
    /// this and the `ConsFormerMddBatcher` built alongside it can't end up with different
    /// `domain_size`s.
    pub fn new(
        problems: Vec<Arc<Problem>>,
        compilation: MddCompilationConfig,
        data_config: ConsFormerDataConfig,
        device: &B::Device,
    ) -> Self {
        let domain_size = data_config.domain_size;
        // One tick per problem, not per constraint: `compile_constraint_mdds` parallelizes over a
        // problem's own constraints too (e.g. Sudoku's ~27), so ticking at that finer grain would
        // need passing a shared `ProgressBar` down into it, and per-problem is already the same
        // granularity the Python ConsFormer extension's `tqdm(executor.map(get_mdd, ...))` uses.
        // `ProgressBar` is internally `Arc`-backed, so cloning it into the parallel closure below
        // (via `progress_with`) is cheap and thread-safe.
        let progress = ProgressBar::new(problems.len() as u64);
        progress.set_style(
            ProgressStyle::with_template(
                "{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .expect("hard-coded progress bar template should always be valid"),
        );
        progress.set_message("Compiling MDDs");

        // MDD compilation (the expensive part -- see `compile_constraint_mdds`) and mask
        // construction are both independent per problem and touch no device state, so they run
        // on a capped worker pool (see `utils::worker_pool`) rather than rayon's all-cores
        // default -- this can run on a shared benchmarking machine, and even locally, saturating
        // every core leaves no headroom to interrupt a training run. `compile_constraint_mdds`
        // also parallelizes across a problem's own constraints; since it's called from within the
        // `install` below, rayon runs that nested parallelism on this same capped pool rather than
        // spilling onto the global one. Only the final `Tensor::from_data` calls stay
        // single-threaded, to avoid any backend-specific assumptions about building tensors
        // concurrently from multiple threads (e.g. around CUDA context handling).
        let per_problem: Vec<(ConsFormerMaskData, Vec<MddInstance>)> = crate::utils::worker_pool()
            .install(|| {
                problems
                    .par_iter()
                    .progress_with(progress.clone())
                    .map(|problem| {
                        (
                            consformer_mask_data(problem),
                            compile_constraint_mdds(problem, &compilation, domain_size),
                        )
                    })
                    .collect()
            });
        progress.finish_and_clear();

        let samples = problems
            .into_iter()
            .zip(per_problem)
            .map(|(problem, (mask_data, constraint_mdds))| {
                let (attention_mask, var_mask) = mask_data.into_tensors::<B>(device);
                ConsFormerMddSample {
                    problem,
                    attention_mask,
                    var_mask,
                    constraint_mdds: Arc::new(constraint_mdds),
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

/// One padding bucket's worth of `MddInstance`s, gathered across every sample in a batch and
/// stacked into tensors -- the unit the WMC loss's batched DP (added in a later step) actually
/// operates on. All four data tensors share the shape `(num_instances, num_layers, max_edges)`
/// (`num_layers`/`max_edges` from `key`, `num_instances` the number of `(sample, constraint)`
/// pairs that landed in this bucket); `sample_index` has shape `(num_instances,)` and gives, for
/// each instance, which row of the batch it belongs to -- needed to average WMCs back per-sample
/// (each sample's constraints are scattered across however many buckets their shapes fall into).
///
/// `gather_index` here is already offset by `sample_index[i] * number_vars * domain_size`, so it
/// indexes directly into the batch's flattened `(batch_size * number_vars, domain_size)`
/// probability tensor -- no per-instance offsetting is needed downstream. `edge_to`/`edge_from`
/// are left un-offset (local node indices within `key.max_nodes`): each instance keeps its own
/// local node space, and batching happens purely via the leading `num_instances` dimension.
#[derive(Clone, Debug)]
pub struct MddBucketBatch<B: Backend> {
    pub key: MddBucketKey,
    /// (num_instances, num_layers, max_edges), values already offset into the batch's flattened
    /// probability tensor.
    pub gather_index: Tensor<B, 3, Int>,
    /// (num_instances, num_layers, max_edges)
    pub edge_mask: Tensor<B, 3, Bool>,
    /// (num_instances, num_layers, max_edges), local node index within `key.max_nodes`.
    pub edge_to: Tensor<B, 3, Int>,
    /// (num_instances, num_layers, max_edges), local node index within `key.max_nodes`.
    pub edge_from: Tensor<B, 3, Int>,
    /// (num_instances,): which batch row each instance's problem occupies.
    pub sample_index: Tensor<B, 1, Int>,
}

/// Batch used to train ConsFormer-MDD. Carries the same attention/var-mask/assignment inputs as
/// the classical `ConsFormerBatch` (see `ConsFormerInputs`), plus every sample's constraint MDDs,
/// gathered into one `MddBucketBatch` per padding bucket that appears anywhere in the batch.
pub struct ConsFormerMddBatch<B: Backend> {
    /// (batch_size, number_vars, number_vars)
    pub attention_masks: Tensor<B, 3, Bool>,
    /// (batch_size, number_vars)
    pub var_masks: Tensor<B, 2, Bool>,
    /// (batch_size, number_vars)
    pub assignments: Tensor<B, 2, Int>,
    /// Problems, used to compute satisfaction reports/metrics.
    pub problems: Vec<Arc<Problem>>,
    /// Every `(sample, constraint)` pair in the batch, grouped by padding bucket.
    pub mdd_buckets: Vec<MddBucketBatch<B>>,
}

impl<B: Backend> Clone for ConsFormerMddBatch<B> {
    fn clone(&self) -> Self {
        ConsFormerMddBatch {
            attention_masks: self.attention_masks.clone(),
            var_masks: self.var_masks.clone(),
            assignments: self.assignments.clone(),
            problems: self.problems.clone(),
            mdd_buckets: self.mdd_buckets.clone(),
        }
    }
}

impl<B: Backend> std::fmt::Debug for ConsFormerMddBatch<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsFormerMddBatch")
            .field("attention_masks", &self.attention_masks)
            .field("var_masks", &self.var_masks)
            .field("problems", &format!("{} problem(s)", self.problems.len()))
            .field("mdd_buckets", &self.mdd_buckets)
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

/// Builds `ConsFormerMddBatch`es from `ConsFormerMddSample`s. `domain_size` must match the value
/// the dataset itself was built with, since it's needed to offset each instance's `gather_index`
/// into the batch's flattened probability tensor -- so this is only constructible from a
/// `ConsFormerDataConfig`, the same value the dataset is built from, rather than from independent
/// `mask_fraction`/`domain_size` arguments that could accidentally diverge from the dataset's.
#[derive(Clone, Copy)]
pub struct ConsFormerMddBatcher {
    mask_fraction: f64,
    domain_size: usize,
}

impl ConsFormerMddBatcher {
    pub fn new(data_config: ConsFormerDataConfig) -> Self {
        Self {
            mask_fraction: data_config.mask_fraction,
            domain_size: data_config.domain_size,
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

        let (attention_masks, var_masks, assignments) = stack_masks_and_sample_assignments(
            attention_mask_tensors,
            &var_mask_tensors,
            &problems,
            self.mask_fraction,
            device,
        );

        // Every problem in a batch shares the same variable count (batches are drawn from a
        // single dataset of same-shaped problems), so a single `number_vars` suffices to turn a
        // sample index into a flat probability-tensor offset.
        let number_vars = problems[0].number_variables() as i64;
        let domain_size = self.domain_size as i64;

        // Group every (sample, constraint) pair by padding bucket, so same-shaped instances --
        // regardless of which sample or which constraint they came from -- end up in the same
        // `MddBucketBatch` and can be run through the DP as one matmul chain.
        let mut grouped: HashMap<MddBucketKey, Vec<(usize, &MddInstance)>> = HashMap::new();
        for (sample_idx, sample) in samples.iter().enumerate() {
            for instance in sample.constraint_mdds.iter() {
                grouped
                    .entry(instance.bucket)
                    .or_default()
                    .push((sample_idx, instance));
            }
        }

        let mdd_buckets = grouped
            .into_iter()
            .map(|(key, instances)| {
                let num_layers = key.num_layers;
                let max_edges = key.max_edges;
                let num_instances = instances.len();
                let cells_per_instance = num_layers * max_edges;

                let mut gather_index = Vec::with_capacity(num_instances * cells_per_instance);
                let mut edge_mask = Vec::with_capacity(num_instances * cells_per_instance);
                let mut edge_to = Vec::with_capacity(num_instances * cells_per_instance);
                let mut edge_from = Vec::with_capacity(num_instances * cells_per_instance);
                let mut sample_index = Vec::with_capacity(num_instances);

                for (sample_idx, instance) in &instances {
                    // Offsets this instance's locally-flat `variable_index * domain_size +
                    // domain_value` indices into the batch's flattened `(batch_size *
                    // number_vars, domain_size)` probability tensor. Padding slots keep their
                    // local value of 0, which after offsetting is `sample_offset` -- still a
                    // valid in-bounds index (into that sample's own variable 0, value 0), it's
                    // just masked out by `edge_mask` before it can contribute anything.
                    let sample_offset = *sample_idx as i64 * number_vars * domain_size;
                    gather_index
                        .extend(instance.gather_index.iter().map(|&idx| idx + sample_offset));
                    edge_mask.extend(instance.edge_mask.iter().copied());
                    edge_to.extend(instance.edge_to.iter().copied());
                    edge_from.extend(instance.edge_from.iter().copied());
                    sample_index.push(*sample_idx as i64);
                }

                let shape = [num_instances, num_layers, max_edges];
                let gather_index: Tensor<B, 3, Int> =
                    Tensor::<B, 1, Int>::from_data(gather_index.as_slice(), device).reshape(shape);
                let edge_mask: Tensor<B, 3, Bool> =
                    Tensor::<B, 1, Bool>::from_data(edge_mask.as_slice(), device).reshape(shape);
                let edge_to: Tensor<B, 3, Int> =
                    Tensor::<B, 1, Int>::from_data(edge_to.as_slice(), device).reshape(shape);
                let edge_from: Tensor<B, 3, Int> =
                    Tensor::<B, 1, Int>::from_data(edge_from.as_slice(), device).reshape(shape);
                let sample_index: Tensor<B, 1, Int> =
                    Tensor::from_data(sample_index.as_slice(), device);

                MddBucketBatch {
                    key,
                    gather_index,
                    edge_mask,
                    edge_to,
                    edge_from,
                    sample_index,
                }
            })
            .collect();

        ConsFormerMddBatch {
            attention_masks,
            var_masks,
            assignments,
            problems,
            mdd_buckets,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modelling::{all_different, among, not_equals, sum};

    /// Plain-f64 reference implementation of the layer-by-layer WMC DP described in the module
    /// doc (`w[0] = 1`, `w(to) += w(from) * probs[edge]` summed over every active edge from
    /// `from` to `to`, `WMC = w[0]` after the last layer, root/sink both being local index 0).
    /// Used only to check `compile_constraint_mdds`'s output against brute force -- the real DP
    /// is tensor-based and added in a later step.
    fn compute_wmc(instance: &MddInstance, probs: &[f64]) -> f64 {
        let max_nodes = instance.bucket.max_nodes;
        let max_edges = instance.bucket.max_edges;
        let num_layers = instance.bucket.num_layers;
        let mut w = vec![0.0; max_nodes];
        w[0] = 1.0;
        for layer in 0..num_layers {
            let mut next = vec![0.0; max_nodes];
            for slot in 0..max_edges {
                let cell = layer * max_edges + slot;
                if instance.edge_mask[cell] {
                    let from = instance.edge_from[cell] as usize;
                    let to = instance.edge_to[cell] as usize;
                    next[to] += w[from] * probs[instance.gather_index[cell] as usize];
                }
            }
            w = next;
        }
        w[0]
    }

    #[test]
    fn wmc_matches_brute_force_for_not_equals() {
        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);

        let domain_size = 3;
        let instances =
            compile_constraint_mdds(&problem, &MddCompilationConfig::default(), domain_size);
        assert_eq!(instances.len(), 1);
        let instance = &instances[0];
        assert_eq!(instance.bucket.num_layers, 2);

        // Flattened the same way `gather_index` expects: flat[var * domain_size + value].
        let probs: Vec<f64> = vec![
            0.2, 0.5, 0.3, // x
            0.1, 0.3, 0.6, // y
        ];

        let wmc = compute_wmc(instance, &probs);

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
    fn wmc_matches_brute_force_for_all_different_permutation() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        all_different(&mut problem, vars.clone());
        let problem = Arc::new(problem);

        let domain_size = 3;
        let instances =
            compile_constraint_mdds(&problem, &MddCompilationConfig::default(), domain_size);
        assert_eq!(instances.len(), 1);
        let instance = &instances[0];
        assert_eq!(instance.bucket.num_layers, 3);

        let probs: Vec<f64> = vec![
            0.5, 0.3, 0.2, // var 0
            0.2, 0.2, 0.6, // var 1
            0.1, 0.4, 0.5, // var 2
        ];

        let wmc = compute_wmc(instance, &probs);

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
    fn wmc_matches_brute_force_for_all_different_non_permutation() {
        // Scope (2 vars) smaller than the domain (3 values): exercises the pairwise-collision
        // case (many distinct successors legitimately share a sink-ward node), not just the
        // permutation case above.
        let mut problem = Problem::default();
        let vars = problem.add_variables(2, vec![0, 1, 2], None);
        all_different(&mut problem, vars.clone());
        let problem = Arc::new(problem);

        let domain_size = 3;
        let instances =
            compile_constraint_mdds(&problem, &MddCompilationConfig::default(), domain_size);
        let instance = &instances[0];

        let probs: Vec<f64> = vec![
            0.2, 0.5, 0.3, // var 0
            0.1, 0.3, 0.6, // var 1
        ];
        let wmc = compute_wmc(instance, &probs);

        let mut brute = 0.0;
        for a in 0..domain_size {
            for b in 0..domain_size {
                if a != b {
                    brute += probs[a] * probs[domain_size + b];
                }
            }
        }
        assert!((wmc - brute).abs() < 1e-9, "wmc={wmc} brute={brute}");
    }

    #[test]
    fn wmc_matches_brute_force_for_sum() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        sum(&mut problem, vars.clone(), 3);
        let problem = Arc::new(problem);

        let domain_size = 3;
        let instances =
            compile_constraint_mdds(&problem, &MddCompilationConfig::default(), domain_size);
        let instance = &instances[0];

        let probs: Vec<f64> = vec![
            0.5, 0.3, 0.2, // var 0
            0.2, 0.2, 0.6, // var 1
            0.1, 0.4, 0.5, // var 2
        ];
        let wmc = compute_wmc(instance, &probs);

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
    fn wmc_matches_brute_force_for_among() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        among(&mut problem, vars.clone(), vec![1, 2], 1, 2);
        let problem = Arc::new(problem);

        let domain_size = 3;
        let instances =
            compile_constraint_mdds(&problem, &MddCompilationConfig::default(), domain_size);
        let instance = &instances[0];

        let probs: Vec<f64> = vec![
            0.5, 0.3, 0.2, // var 0
            0.2, 0.2, 0.6, // var 1
            0.1, 0.4, 0.5, // var 2
        ];
        let wmc = compute_wmc(instance, &probs);

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
        // Scope larger than the domain: AllDifferent can never be satisfied.
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1], None);
        all_different(&mut problem, vars.clone());
        let problem = Arc::new(problem);

        let domain_size = 2;
        let instances =
            compile_constraint_mdds(&problem, &MddCompilationConfig::default(), domain_size);
        let instance = &instances[0];

        let probs: Vec<f64> = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        assert_eq!(compute_wmc(instance, &probs), 0.0);
        assert!(instance.edge_mask.iter().all(|&m| !m));
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
            // A different-shaped problem per index, so a shuffle/interference bug would show up
            // as a mismatch rather than being masked by every problem looking the same.
            let mut problem = Problem::default();
            let domain: Vec<isize> = (0..=(domain_max % 5 + 2)).collect();
            let vars = problem.add_variables(3, domain, None);
            all_different(&mut problem, vars.clone());
            not_equals(&mut problem, vars[0], vars[1]);
            problems.push(Arc::new(problem));
        }

        let device = NdArrayDevice::default();
        let domain_size = 7; // covers every problem's domain_max % 5 + 2 above (max 6)
        let data_config = ConsFormerDataConfig {
            domain_size,
            mask_fraction: 0.0,
        };

        let dataset = ConsFormerMddDataset::<NdArray>::new(
            problems.clone(),
            MddCompilationConfig::default(),
            data_config,
            &device,
        );

        assert_eq!(dataset.len(), problems.len());
        for (i, problem) in problems.iter().enumerate() {
            let sample = dataset.get(i).unwrap();
            // Same problem, in the same order.
            assert!(Arc::ptr_eq(sample.problem(), problem));

            // Same MDDs as compiling that one problem in isolation would produce.
            let expected =
                compile_constraint_mdds(problem, &MddCompilationConfig::default(), domain_size);
            let actual = sample.constraint_mdds();
            assert_eq!(actual.len(), expected.len());
            for (a, e) in actual.iter().zip(expected.iter()) {
                assert_eq!(a.bucket, e.bucket);
                assert_eq!(a.gather_index, e.gather_index);
                assert_eq!(a.edge_mask, e.edge_mask);
                assert_eq!(a.edge_to, e.edge_to);
                assert_eq!(a.edge_from, e.edge_from);
            }
        }
    }

    /// `ConsFormerMddBatcher` groups every `(sample, constraint)` instance in a batch by padding
    /// bucket, offsets each instance's `gather_index` by its sample's position in the batch, and
    /// records which sample every instance came from. This checks all three: bucket grouping
    /// (identically-shaped constraints across different samples land in the same bucket),
    /// `sample_index` bookkeeping (each sample's constraints are found, with the right sample
    /// index, spread across however many buckets they fall into), and the `gather_index` offset
    /// arithmetic itself.
    #[test]
    fn batcher_groups_by_bucket_and_offsets_gather_index() {
        use burn::backend::ndarray::{NdArray, NdArrayDevice};

        // Every problem has the identical structure (3 vars, domain {0,1,2}, an AllDifferent
        // over all three plus a NotEquals over the first two) so their MDDs -- and hence their
        // bucket keys -- are identical across samples; only the sample they belong to differs.
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
        let number_vars = 3i64;
        let data_config = ConsFormerDataConfig {
            domain_size,
            mask_fraction: 0.0,
        };

        let dataset = ConsFormerMddDataset::<NdArray>::new(
            problems.clone(),
            MddCompilationConfig::default(),
            data_config,
            &device,
        );

        // Keep the per-sample, pre-batch instances around (in dataset order) to check the
        // batch's offset arithmetic against.
        let per_sample_instances: Vec<Arc<Vec<MddInstance>>> = (0..dataset.len())
            .map(|i| Arc::clone(dataset.get(i).unwrap().constraint_mdds()))
            .collect();

        let samples: Vec<ConsFormerMddSample<NdArray>> = (0..dataset.len())
            .map(|i| dataset.get(i).unwrap())
            .collect();

        let batcher = ConsFormerMddBatcher::new(data_config);
        let batch = batcher.batch(samples, &device);

        // Every sample has exactly 2 constraints (AllDifferent, NotEquals), each with a distinct
        // shape, so exactly 2 buckets should appear, and every instance across both should be
        // accounted for.
        assert_eq!(batch.mdd_buckets.len(), 2);
        let total_instances: usize = batch
            .mdd_buckets
            .iter()
            .map(|b| b.sample_index.dims()[0])
            .sum();
        assert_eq!(total_instances, problems.len() * 2);

        for bucket in &batch.mdd_buckets {
            let num_instances = bucket.sample_index.dims()[0];
            assert_eq!(
                bucket.gather_index.dims(),
                [num_instances, bucket.key.num_layers, bucket.key.max_edges]
            );

            let sample_indices: Vec<i64> = bucket
                .sample_index
                .clone()
                .into_data()
                .to_vec::<i64>()
                .expect("sample_index should convert to i32")
                .into_iter()
                .map(|v| v as i64)
                .collect();

            let gather_flat: Vec<i64> = bucket
                .gather_index
                .clone()
                .into_data()
                .to_vec::<i64>()
                .expect("gather_index should convert to i32")
                .into_iter()
                .map(|v| v as i64)
                .collect();

            let cells_per_instance = bucket.key.num_layers * bucket.key.max_edges;

            for (instance_idx, &sample_idx) in sample_indices.iter().enumerate() {
                // Find the one constraint of this sample whose bucket matches -- exactly one,
                // since AllDifferent and NotEquals have different shapes here.
                let source = per_sample_instances[sample_idx as usize]
                    .iter()
                    .find(|inst| inst.bucket == bucket.key)
                    .expect("every bucketed instance should trace back to a source MddInstance");

                let expected_offset = sample_idx * number_vars * domain_size as i64;
                let cell_start = instance_idx * cells_per_instance;
                for cell in 0..cells_per_instance {
                    assert_eq!(
                        gather_flat[cell_start + cell],
                        source.gather_index[cell] + expected_offset,
                        "gather_index mismatch at instance {instance_idx} cell {cell}"
                    );
                }
            }
        }

        // Sanity check the shared mask/assignment path still produces the expected shapes.
        assert_eq!(batch.attention_masks.dims(), [3, 3, 3]);
        assert_eq!(batch.var_masks.dims(), [3, 3]);
        assert_eq!(batch.assignments.dims(), [3, 3]);
    }
}
