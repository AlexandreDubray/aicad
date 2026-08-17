use std::collections::HashMap;

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, FloatDType, IndexingUpdateOp, Int, Tensor};

use crate::constraints::{AllDifferent, Constraint, NotEquals};
use crate::learning::{BatchProblems, Loss};

use super::dataset::ConsFormerBatch;
use super::mdd_dataset::{ConsFormerMddBatch, MddBucketBatch};

/// Loss trait for ConsFormer. Given a tensor (number_var, domain_size), computes a penalty term
/// for the constraint. We assume that the probability tensor sent to the constraint is restricted
/// to its scope
pub trait ConstraintLoss<B: Backend> {
    fn constraint_penalty(&self, probs: Tensor<B, 2>) -> Tensor<B, 1>;
}

/// "No two variables share a value" relaxation: penalizes any pair of variables in the scope
/// putting nonzero probability on the same value. Used for `NotEquals` and small-scope
/// `AllDifferent` -- i.e. binary/pairwise inequality constraints, as in graph coloring or max-cut.
///
/// Mathematically equivalent, up to a constant factor of 2, to `CustomGCOLLossDot` in the Python
/// reference this recipe was adapted from: that function sums `dot_products * adj_matrix` over
/// the *whole* (symmetric) adjacency matrix, counting each edge twice (once as `(i,j)`, once as
/// `(j,i)`); `triu(1)` here counts each declared constraint exactly once instead. A constant scale
/// factor changes neither the loss's optimum nor its gradient direction, so this is left as-is
/// rather than doubled to match bit-for-bit.
fn pairwise_collision_penalty<B: Backend>(probs: Tensor<B, 2>) -> Tensor<B, 1> {
    let collisions = probs.clone().matmul(probs.transpose());
    collisions.triu(1).sum().reshape([1])
}

/// Penalty for permutation constraints (e.g., all-diff with exactly the same number of domain
/// values as variables, such as every Sudoku row/column/box).
///
/// Mean- (not sum-) reduced over the coverage-error tensor, matching `CustomSudokuLossMSE
/// .batch_loss` in the Python reference this recipe was adapted from (`nn.MSELoss()`'s default
/// `reduction='mean'`). Note this only matches Python's *per-element* normalization: Python
/// additionally computes rows, columns, and boxes as three separately-averaged terms before
/// summing them, a Sudoku-specific "which of the 27 all-diffs is this" grouping that this generic,
/// constraint-type-agnostic code has no equivalent of (a `Problem`/`Constraint` here carries no
/// notion of belonging to one of several named groups). So this is not bit-exact with the Python
/// reference for Sudoku specifically -- just the same per-element normalization intent, applied
/// uniformly to however many permutation constraints a sample has.
fn permutation_penalty<B: Backend>(probs: Tensor<B, 2>) -> Tensor<B, 1> {
    let [_, domain_size] = probs.dims();
    let coverage: Tensor<B, 2> = probs.sum_dim(0).reshape([1, domain_size]);
    let diff = coverage.sub_scalar(1.0);
    (diff.clone() * diff).mean().reshape([1])
}

impl<B: Backend> ConstraintLoss<B> for AllDifferent {
    /// Uses the permutation relaxation when the scope exactly covers the
    /// domain (e.g. Sudoku), and falls back to the pairwise collision
    /// relaxation otherwise (scope smaller than the domain: not every value
    /// needs to appear, so "no duplicates" is the correct -- and only
    /// meaningful -- relaxation).
    fn constraint_penalty(&self, probs: Tensor<B, 2>) -> Tensor<B, 1> {
        let [scope_len, domain_size] = probs.dims();
        if scope_len == domain_size {
            permutation_penalty(probs)
        } else {
            pairwise_collision_penalty(probs)
        }
    }
}

impl<B: Backend> ConstraintLoss<B> for NotEquals {
    fn constraint_penalty(&self, probs: Tensor<B, 2>) -> Tensor<B, 1> {
        pairwise_collision_penalty(probs)
    }
}

/// Compute for an arbitrary constraint its penalty term.
fn constraint_loss<B: Backend>(
    constraint: &(dyn Constraint + Send + Sync + 'static),
    all_probs: &Tensor<B, 2>,
) -> Tensor<B, 1> {
    // First, get the scope of the constraint and limit the probabilities to it.
    let scope: Vec<i64> = constraint.iter_scope().map(|v| v.0 as i64).collect();
    let device = all_probs.device();
    let idx = Tensor::<B, 1, Int>::from_data(scope.as_slice(), &device);
    let scope_probs = all_probs.clone().select(0, idx);

    // Pattern matching to find the actual constraint
    if let Some(c) = constraint.as_any().downcast_ref::<AllDifferent>() {
        return c.constraint_penalty(scope_probs);
    }
    if let Some(c) = constraint.as_any().downcast_ref::<NotEquals>() {
        return c.constraint_penalty(scope_probs);
    }

    panic!(
        "no ConstraintLoss implementation for constraint type `{}` -- add one in src/learning/consformer/loss.rs",
        constraint.name()
    );
}

/// Soft (differentiable) Gumbel-softmax relaxation: adds Gumbel(0,1) noise
/// to `logits` before the softmax, instead of taking a plain softmax.
fn gumbel_softmax<B: Backend>(logits: Tensor<B, 3>) -> Tensor<B, 3> {
    let device = logits.device();
    let u = Tensor::<B, 3>::random(
        logits.dims(),
        burn::tensor::Distribution::Uniform(1e-20, 1.0),
        &device,
    );
    let neg_log_u = -u.log();
    let gumbel = -neg_log_u.log();
    softmax(logits + gumbel, 2)
}

fn blend_with_current<B: Backend>(
    probs: Tensor<B, 3>,
    assignments: Tensor<B, 2, Int>,
    var_masks: Tensor<B, 2, Bool>,
) -> Tensor<B, 3> {
    let [batch_size, number_vars, domain_size] = probs.dims();
    let device = probs.device();

    let arange: Tensor<B, 3, Int> = Tensor::<B, 1, Int>::arange(0..domain_size as i64, &device)
        .reshape([1, 1, domain_size])
        .repeat_dim(0, batch_size)
        .repeat_dim(1, number_vars);
    let assign_3d: Tensor<B, 3, Int> = assignments
        .reshape([batch_size, number_vars, 1])
        .repeat_dim(2, domain_size);
    let one_hot: Tensor<B, 3> = assign_3d.equal(arange).float();

    let mask_3d: Tensor<B, 3, Bool> = var_masks
        .reshape([batch_size, number_vars, 1])
        .repeat_dim(2, domain_size);

    one_hot.mask_where(mask_3d, probs)
}

pub struct ConsFormerLoss;

impl<B: Backend> Loss<B, ConsFormerBatch<B>> for ConsFormerLoss {
    fn loss(&self, logits: Tensor<B, 3>, batch: &ConsFormerBatch<B>) -> Tensor<B, 1> {
        let probs = gumbel_softmax(logits);
        let probs = blend_with_current(probs, batch.assignments.clone(), batch.var_masks.clone());
        let problems = batch.problems();
        let batch_size = problems.len();
        let [_, number_vars, domain_size] = probs.dims();
        let device = probs.device();

        // Flatten (batch, vars, domain) -> (batch*vars, domain) so a single
        // "global" variable index (sample_offset + local_index) can gather any
        // variable of any sample in one op.
        let flat_probs = probs
            .clone()
            .reshape([batch_size * number_vars, domain_size]);

        // Group constraints by which batched penalty they need, so each
        // group can be computed as a single tensor op instead of one
        // op-chain per instance. `NotEquals`, and `AllDifferent` whose scope
        // is smaller than the domain, use the pairwise collision penalty
        // (batched via matmul); `AllDifferent` whose scope exactly covers
        // the domain (e.g. every Sudoku row/col/box) uses the permutation
        // penalty instead (batched via a sum reduction.
        let mut collision_groups: HashMap<usize, Vec<i64>> = HashMap::new();
        let mut permutation_groups: HashMap<usize, Vec<i64>> = HashMap::new();
        let mut total = Tensor::<B, 1>::zeros([1], &device);

        for (i, problem) in problems.iter().enumerate() {
            let sample_offset = (i * number_vars) as i64;

            for constraint in problem.iter_constraints() {
                let c = &*problem[constraint];
                let scope_len = c.iter_scope().count();
                let is_all_different = c.as_any().downcast_ref::<AllDifferent>().is_some();
                let is_not_equals = c.as_any().downcast_ref::<NotEquals>().is_some();

                if is_all_different && scope_len == domain_size {
                    let scope: Vec<i64> =
                        c.iter_scope().map(|v| sample_offset + v.0 as i64).collect();
                    permutation_groups
                        .entry(scope_len)
                        .or_default()
                        .extend(scope);
                } else if is_all_different || is_not_equals {
                    let scope: Vec<i64> =
                        c.iter_scope().map(|v| sample_offset + v.0 as i64).collect();
                    collision_groups.entry(scope_len).or_default().extend(scope);
                } else {
                    let sample_probs: Tensor<B, 2> = probs.clone().slice([i..i + 1]).squeeze();
                    total = total + constraint_loss(c, &sample_probs);
                }
            }
        }

        // One batched matmul + triu + sum per group, instead of one op-chain per instance. See
        // `pairwise_collision_penalty`'s doc: this is the code path that actually runs for
        // `NotEquals`/small-scope `AllDifferent` during training (the free function above isn't
        // reachable for those two types), and it's already equivalent, up to a benign constant
        // factor of 2, to `CustomGCOLLossDot`.
        for (scope_len, flat_indices) in collision_groups {
            let num_instances = flat_indices.len() / scope_len;
            let idx = Tensor::<B, 1, Int>::from_data(flat_indices.as_slice(), &device);
            let group_probs: Tensor<B, 3> =
                flat_probs
                    .clone()
                    .select(0, idx)
                    .reshape([num_instances, scope_len, domain_size]);

            let collisions = group_probs.clone().matmul(group_probs.transpose());
            total = total + collisions.triu(1).sum().reshape([1]);
        }

        // One batched mean-reduction per group, instead of one op-chain per instance. See
        // `permutation_penalty`'s doc: mean- (not sum-) reduced to match `CustomSudokuLossMSE`'s
        // `nn.MSELoss()` reduction, modulo the row/column/box grouping this generic code can't
        // replicate. This is the code path that actually runs for `AllDifferent` with
        // `scope_len == domain_size` during training.
        for (scope_len, flat_indices) in permutation_groups {
            let num_instances = flat_indices.len() / scope_len;
            let idx = Tensor::<B, 1, Int>::from_data(flat_indices.as_slice(), &device);
            let group_probs: Tensor<B, 3> =
                flat_probs
                    .clone()
                    .select(0, idx)
                    .reshape([num_instances, scope_len, domain_size]);

            // Sum over the scope (dim 1): per group instance, how much
            // probability mass each value received across the whole scope.
            let coverage: Tensor<B, 2> =
                group_probs.sum_dim(1).reshape([num_instances, domain_size]);
            let diff = coverage.sub_scalar(1.0);
            total = total + (diff.clone() * diff).mean().reshape([1]);
        }

        total.div_scalar(batch_size as f32)
    }
}

/// Runs the batched forward WMC DP (see the module doc on `MddInstance`/`MddBucketBatch` for the
/// edge-list layout) over every instance in one padding bucket at once, entirely in log space,
/// and returns `log(WMC)` -- not WMC itself. Ports the same design as the Python ConsFormer
/// extension's `criterion/mdd.py::log_space_wmc`: log-space "multiply" is addition
/// (`w_from + log_probs_at_edges`), and log-space "sum over incoming edges" has no batched
/// primitive of its own (burn's `scatter` only implements `IndexingUpdateOp::Add`, not `Max`/
/// `Min`, so there's no cheap way to express `logsumexp` directly) -- so each layer exponentiates,
/// sums via the ordinary linear-space `scatter`/`Add` this DP already used, then takes `.log()` to
/// get back to log space for the next layer.
///
/// This round-trip through linear space is safe from overflow *without* the usual logsumexp
/// max-shift trick: every quantity here is a log-probability, so `w` (log of a sum of products of
/// probabilities, itself always in `[0, 1]`) and `flat_log_probs` are both always `<= 0`. Their
/// sum -- log-space multiplication -- is therefore always `<= 0` too, so `.exp()` of it can never
/// overflow; only underflow toward `0.0` is possible, and `.log()` of that (`log(0) = -inf`) is
/// safe and NaN-free, not a special case that needs guarding. This replaces an earlier version of
/// this function that computed WMC directly in linear space (`f64`-only, `+ epsilon` before the
/// one `-log` at the end): that version needed `epsilon` purely to keep `-log` finite whenever WMC
/// was legitimately, exactly zero (e.g. an already-conflicting fixed/"given" context), which also
/// silently floors the loss (and its gradient) whenever the true WMC is merely *very* small rather
/// than exactly zero -- log space has no such floor, since nothing here is ever exponentiated back
/// up to a linear WMC that could underflow.
///
/// Still runs in `f64`, not the backend's default `f32`, for the same reason as before: keep the
/// per-layer exponentiate/sum/log round-trip representable across the widest realistic range
/// before any real underflow (not just precision loss) could occur. `Tensor::cast` is
/// autodiff-aware (see `burn_autodiff`'s `float_cast` op), so this doesn't break backprop into
/// `logits` -- only this DP runs in double precision, not the rest of the network.
///
/// Returns `log(WMC)`, one per instance, `(num_instances,)`, in the same order as
/// `bucket.sample_index`. `-inf` for any instance whose fixed ("given") context is already
/// structurally unsatisfiable, regardless of what the network predicts for its free variables --
/// see `mdd_wmc_loss`'s doc for what that does to the loss.
fn bucket_log_wmc<B: Backend>(bucket: &MddBucketBatch<B>, flat_log_probs: &Tensor<B, 1>) -> Tensor<B, 1> {
    let device = flat_log_probs.device();
    let num_instances = bucket.sample_index.dims()[0];
    let max_nodes = bucket.key.max_nodes;
    let max_edges = bucket.key.max_edges;
    let num_layers = bucket.key.num_layers;

    let flat_log_probs: Tensor<B, 1> = flat_log_probs.clone().cast(FloatDType::F64);

    // w[i, 0] = log(1) = 0: every instance starts at its MDD's root, local node index 0 of layer
    // 0 (the `Mdd`'s own convention -- see `MddInstance`'s doc). Every other node starts at
    // log(0) = -inf (not yet reached). This is a constant, not derived from the network's
    // probabilities, so it's built directly rather than through a tracked op.
    let mut root = vec![f32::NEG_INFINITY; num_instances * max_nodes];
    for i in 0..num_instances {
        root[i * max_nodes] = 0.0;
    }
    let mut w: Tensor<B, 2> = Tensor::<B, 1>::from_data(root.as_slice(), &device)
        .reshape([num_instances, max_nodes])
        .cast(FloatDType::F64);

    for layer in 0..num_layers {
        let gather_layer: Tensor<B, 2, Int> = bucket
            .gather_index
            .clone()
            .narrow(1, layer, 1)
            .squeeze_dim(1);
        let mask_layer: Tensor<B, 2, Bool> =
            bucket.edge_mask.clone().narrow(1, layer, 1).squeeze_dim(1);
        let to_layer: Tensor<B, 2, Int> = bucket.edge_to.clone().narrow(1, layer, 1).squeeze_dim(1);
        let from_layer: Tensor<B, 2, Int> =
            bucket.edge_from.clone().narrow(1, layer, 1).squeeze_dim(1);

        // The log-probability each edge's decision puts on its assignment: one gather from the
        // whole batch's flattened log-probability vector per edge slot (`gather_index` is already
        // offset per-sample -- see `ConsFormerMddBatcher`).
        let flat_gather: Tensor<B, 1, Int> = gather_layer.reshape([num_instances * max_edges]);
        let log_probs_at_edges: Tensor<B, 2> = flat_log_probs
            .clone()
            .select(0, flat_gather)
            .reshape([num_instances, max_edges]);

        // Log-space "multiply": each edge's contribution is its source node's current log-weight
        // plus its own decision's log-probability -- always `<= 0` (see doc above), so the
        // `.exp()` just below can never overflow.
        let w_from = w.clone().gather(1, from_layer);
        let contributions = w_from + log_probs_at_edges;

        // Exponentiate to linear space so incoming edges can be *summed* (log space has no
        // batched "sum", only "logsumexp", which isn't available here -- see doc above). Padding
        // slots are zeroed via `mask_layer` right after exponentiating, same as the old
        // linear-space DP -- whatever garbage `contributions` holds for them (their `edge_to`/
        // `edge_from`/`gather_index` values are in-bounds but not meaningful) doesn't matter once
        // multiplied by 0.
        let exp_contributions =
            contributions.exp() * mask_layer.float().cast(FloatDType::F64);

        // Sum every edge's linear-space contribution into its target node -- the DP's "sum over
        // incoming edges" step, via the same `scatter`/`Add` the old linear-space DP used --
        // then take `.log()` to get back to log space for the next layer. `log(0.0) = -inf` for
        // any node no edge reached this layer; that's a real, meaningful value here (that node is
        // unreachable), not an error case to special-case around.
        let next = Tensor::<B, 2>::zeros([num_instances, max_nodes], &device).cast(FloatDType::F64);
        let next_linear = next.scatter(1, to_layer, exp_contributions, IndexingUpdateOp::Add);
        w = next_linear.log();
    }

    // log(WMC) = the sink's log-weight, local node index 0 of the last layer -- same convention
    // as root. Cast back down to the backend's default `f32` here, at the very end, so every
    // caller is unaffected by this function's internal precision -- and unlike the old
    // linear-space `bucket_wmc`, this is safe: a log-probability is a normally-scaled number (not
    // an astronomically tiny one), so there's no underflow risk left to guard against by staying
    // in `f64` past this point.
    let sink_idx = Tensor::<B, 1, Int>::from_data([0i64], &device);
    w.select(1, sink_idx)
        .reshape([num_instances])
        .cast(FloatDType::F32)
}

/// Trains ConsFormer against the *exact* weighted model count of each constraint's precompiled
/// MDD, instead of the classical hand-written per-constraint-type penalty (`ConstraintLoss`).
/// Loss per sample is the average, over that sample's own constraints, of `-log(wmc)`; the batch
/// loss is the average of that over every sample. No `epsilon` floor: see `bucket_log_wmc`'s doc
/// for why running the whole DP in log space makes one unnecessary. This does mean a sample whose
/// fixed ("given") context is already structurally unsatisfiable for one of its constraints -- not
/// underflow, a real `WMC = 0` -- produces a literal `-log(0) = inf` loss for that constraint, with
/// no gradient (the earlier `epsilon`-floored version silently hid this as a large-but-finite,
/// still-misleadingly-differentiable number instead). See `ConsFormerMddDataset`/`MddInstance` for
/// how the MDDs are compiled and reduced to tensors, and `bucket_log_wmc` for the batched DP
/// itself.
pub struct ConsFormerMddLoss;

/// The actual WMC-loss computation, given `probs` directly rather than the raw `logits` --
/// pulled out of `Loss::loss` so tests can exercise it with deterministic probabilities instead
/// of going through `gumbel_softmax`'s randomness.
fn mdd_wmc_loss<B: Backend>(probs: Tensor<B, 3>, batch: &ConsFormerMddBatch<B>) -> Tensor<B, 1> {
    let batch_size = batch.problems().len();
    let [_, number_vars, domain_size] = probs.dims();
    let device = probs.device();

    // Flatten (batch, vars, domain) -> (batch*vars*domain,): `MddBucketBatch::gather_index` is
    // already offset by `sample_index * number_vars * domain_size` (see `ConsFormerMddBatcher`),
    // so it indexes directly into this. Logged once here, rather than per-bucket, since every
    // bucket shares the same underlying `probs`.
    let flat_log_probs: Tensor<B, 1> =
        probs.log().reshape([batch_size * number_vars * domain_size]);

    // Every constraint's `-log(wmc)`, scattered into its sample's running sum, plus a running
    // count of how many constraints landed in each sample -- both needed for the per-sample
    // average before averaging again over the batch. Constraints of one sample can be spread
    // across several buckets (different constraint shapes), so this accumulates across every
    // bucket rather than assuming one bucket has the whole picture.
    let mut per_sample_sum = Tensor::<B, 1>::zeros([batch_size], &device);
    let mut per_sample_count = Tensor::<B, 1>::zeros([batch_size], &device);

    for bucket in &batch.mdd_buckets {
        let log_wmc = bucket_log_wmc(bucket, &flat_log_probs);
        let neg_log_wmc = -log_wmc;
        let num_instances = bucket.sample_index.dims()[0];
        let ones = Tensor::<B, 1>::ones([num_instances], &device);

        per_sample_sum = per_sample_sum.scatter(
            0,
            bucket.sample_index.clone(),
            neg_log_wmc,
            IndexingUpdateOp::Add,
        );
        per_sample_count =
            per_sample_count.scatter(0, bucket.sample_index.clone(), ones, IndexingUpdateOp::Add);
    }

    // `clamp_min(1.0)` only matters for a sample with zero constraints of its own (its sum is 0
    // too, so the divide is 0/1 either way) -- guards the division without changing any real
    // sample's result.
    let per_sample_avg = per_sample_sum / per_sample_count.clamp_min(1.0);
    per_sample_avg.mean()
}

impl<B: Backend> Loss<B, ConsFormerMddBatch<B>> for ConsFormerMddLoss {
    fn loss(&self, logits: Tensor<B, 3>, batch: &ConsFormerMddBatch<B>) -> Tensor<B, 1> {
        let probs = gumbel_softmax(logits);
        let probs = blend_with_current(probs, batch.assignments.clone(), batch.var_masks.clone());
        mdd_wmc_loss(probs, batch)
    }
}

#[cfg(test)]
mod mdd_loss_tests {
    use std::sync::Arc;

    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::data::dataloader::batcher::Batcher;
    use burn::data::dataset::Dataset;

    use crate::modelling::{all_different, not_equals, Problem};

    use super::super::mdd_dataset::{ConsFormerMddBatcher, ConsFormerMddDataset, MddCompilationConfig};
    use super::super::ConsFormerDataConfig;
    use super::*;

    /// Every problem is 3 variables, domain `{0,1,2}`, with an `AllDifferent` over all three (a
    /// permutation-shaped MDD, 3 layers) and a `NotEquals` over the first two (2 layers) -- so a
    /// batch of them exercises two distinct buckets, each with one instance per sample. Generic
    /// over the backend so the same builder can be used both with plain `NdArray` (the WMC/loss
    /// correctness tests) and `Autodiff<NdArray>` (the gradient-flow test).
    ///
    /// `mask_fraction` is exposed rather than fixed: the correctness tests use `0.0` since they
    /// only care about `bucket_wmc`/`mdd_wmc_loss`'s own math and want deterministic
    /// `probs`-derived values, but a gradient-flow test needs at least some variables actually
    /// routed through the network's probabilities (`var_masks` true) -- with every variable
    /// pinned to its fixed initial assignment (`blend_with_current`'s `mask_fraction = 0.0`
    /// case), the loss wouldn't depend on `logits` at all, and the gradient would be trivially
    /// zero for a reason that has nothing to do with `bucket_wmc`.
    fn two_sample_batch<B: Backend>(device: &B::Device, mask_fraction: f64) -> ConsFormerMddBatch<B> {
        let mut problems = Vec::new();
        for _ in 0..2 {
            let mut problem = Problem::default();
            let vars = problem.add_variables(3, vec![0, 1, 2], None);
            all_different(&mut problem, vars.clone());
            not_equals(&mut problem, vars[0], vars[1]);
            problems.push(Arc::new(problem));
        }

        let data_config = ConsFormerDataConfig {
            domain_size: 3,
            mask_fraction,
        };
        let dataset = ConsFormerMddDataset::<B>::new(
            problems,
            MddCompilationConfig::default(),
            data_config,
            device,
        );
        let samples: Vec<_> = (0..dataset.len()).map(|i| dataset.get(i).unwrap()).collect();

        let batcher = ConsFormerMddBatcher::new(data_config);
        batcher.batch(samples, device)
    }

    fn brute_force_not_equals(probs: &[f64], domain_size: usize) -> f64 {
        let mut brute = 0.0;
        for a in 0..domain_size {
            for b in 0..domain_size {
                if a != b {
                    brute += probs[a] * probs[domain_size + b];
                }
            }
        }
        brute
    }

    fn brute_force_all_different_permutation(probs: &[f64], domain_size: usize) -> f64 {
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
        brute
    }

    /// `bucket_log_wmc` is the tensor-batched DP; this checks `.exp()` of its output against a
    /// plain brute-force enumeration for every instance of every bucket in a two-sample batch,
    /// using a different probability distribution per sample so a bug in the per-sample
    /// `gather_index` offset (rather than the DP itself) would also be caught.
    #[test]
    fn bucket_log_wmc_matches_brute_force_across_a_batch() {
        let device = NdArrayDevice::default();
        let domain_size = 3;
        let batch = two_sample_batch(&device, 0.0);

        let per_sample_probs: Vec<Vec<f64>> = vec![
            vec![0.2, 0.5, 0.3, 0.1, 0.3, 0.6, 0.4, 0.4, 0.2],
            vec![0.6, 0.3, 0.1, 0.2, 0.2, 0.6, 0.5, 0.25, 0.25],
        ];
        let flat: Vec<f32> = per_sample_probs.iter().flatten().map(|&v| v as f32).collect();
        let flat_probs: Tensor<NdArray, 1> = Tensor::from_data(flat.as_slice(), &device);
        let flat_log_probs = flat_probs.log();

        assert_eq!(batch.mdd_buckets.len(), 2);
        for bucket in &batch.mdd_buckets {
            let wmc: Vec<f32> = bucket_log_wmc(bucket, &flat_log_probs)
                .exp()
                .into_data()
                .to_vec::<f32>()
                .expect("wmc should convert to f32");
            let sample_indices: Vec<i64> = bucket
                .sample_index
                .clone()
                .into_data()
                .to_vec::<i64>()
                .expect("sample_index should convert to i64");

            for (instance_idx, &sample_idx) in sample_indices.iter().enumerate() {
                let probs = &per_sample_probs[sample_idx as usize];
                let expected = if bucket.key.num_layers == 2 {
                    brute_force_not_equals(probs, domain_size)
                } else {
                    brute_force_all_different_permutation(probs, domain_size)
                };
                assert!(
                    (wmc[instance_idx] as f64 - expected).abs() < 1e-5,
                    "bucket num_layers={} instance {instance_idx} sample {sample_idx}: got {} expected {expected}",
                    bucket.key.num_layers,
                    wmc[instance_idx],
                );
            }
        }
    }

    /// Regression test for the f32-underflow bug `bucket_log_wmc`'s log-space DP fixes (see that
    /// function's doc): a Sudoku-shaped permutation constraint (scope 9, domain 9) needs a 9-way
    /// product of per-edge probabilities to reach any valid assignment. With a sharply peaked
    /// distribution -- exactly what `gumbel_softmax` plus a small `ConsFormerConfig::tau` produce,
    /// even at initialization -- the "off-peak" probabilities are small enough that a naive
    /// linear-space `f32` DP would silently underflow every permutation's product to a literal
    /// `0.0`; log space sidesteps this entirely (see the function doc), rather than merely
    /// widening the floor the way an `f64`-only linear-space DP would.
    #[test]
    fn bucket_log_wmc_survives_sharply_peaked_probabilities() {
        let device = NdArrayDevice::default();
        let domain_size = 9;

        let mut problem = Problem::default();
        let vars = problem.add_variables(domain_size, (0..domain_size as isize).collect(), None);
        all_different(&mut problem, vars);
        let problems = vec![Arc::new(problem)];

        let data_config = ConsFormerDataConfig {
            domain_size,
            mask_fraction: 0.0,
        };
        let dataset = ConsFormerMddDataset::<NdArray>::new(
            problems,
            MddCompilationConfig::default(),
            data_config,
            &device,
        );
        let samples: Vec<_> = (0..dataset.len()).map(|i| dataset.get(i).unwrap()).collect();
        let batcher = ConsFormerMddBatcher::new(data_config);
        let batch = batcher.batch(samples, &device);

        // Every one of the 9 variables puts 0.99999 on the *same* value (index 0) and splits the
        // remaining 0.00001 evenly over the other 8 -- a degenerate-but-plausible early-training
        // state (e.g. attention hasn't yet learned to differentiate the variables). Off-peak
        // probability is ~1.25e-6. Since only one variable can actually take value 0 in any valid
        // permutation, every one of the 9! permutations routes through 8 off-peak edges and just
        // 1 on-peak edge -- there is no permutation that avoids the off-peak probabilities
        // entirely (unlike e.g. a diagonal/identity permutation, which trivially would).
        let dominant = 0.99999_f64;
        let off_peak = (1.0 - dominant) / (domain_size as f64 - 1.0);
        let mut single_var_probs = vec![off_peak as f32; domain_size];
        single_var_probs[0] = dominant as f32;
        let probs_data: Vec<f32> = single_var_probs.repeat(domain_size);
        let probs: Tensor<NdArray, 3> = Tensor::<NdArray, 1>::from_data(probs_data.as_slice(), &device)
            .reshape([1, domain_size, domain_size]);
        let flat_probs: Tensor<NdArray, 1> = probs.reshape([domain_size * domain_size]);
        let flat_log_probs = flat_probs.log();

        // True WMC = 9! * dominant * off_peak^8: every permutation puts exactly one variable on
        // the dominant value and the other 8 on distinct off-peak values, and there are 9! ways
        // to choose which variable gets which value. ~2.16e-42 -- tiny, but each individual
        // per-permutation product bottoms out around `off_peak^8 ~ 6e-48`, well past f32's
        // ~1.18e-38 normal floor (and even its ~1.4e-45 subnormal floor) partway through the DP,
        // well before any summing. A naive linear-space `f32` DP underflows every single path to
        // exactly `0.0`; log space keeps this representable without even needing the widened but
        // still-finite `f64` floor a linear-space DP would.
        let factorial_9 = (1..=9u32).product::<u32>() as f64;
        let expected_wmc = factorial_9 * dominant * off_peak.powi(8);

        assert_eq!(batch.mdd_buckets.len(), 1, "a single AllDifferent has one bucket");
        let wmc: Vec<f32> = bucket_log_wmc(&batch.mdd_buckets[0], &flat_log_probs)
            .exp()
            .into_data()
            .to_vec::<f32>()
            .expect("wmc should convert to f32");

        // A relative check, not absolute: `expected_wmc` is tiny enough that even f64's own
        // rounding error at this magnitude is a meaningful fraction of the value itself.
        let relative_error = ((wmc[0] as f64 - expected_wmc) / expected_wmc).abs();
        assert!(
            relative_error < 0.05,
            "wmc should be close to the analytically-expected {expected_wmc:e} (9! permutations, \
             each contributing dominant * off_peak^8) -- got {}, relative error {relative_error:.4} \
             -- looks like the DP underflowed to 0 (or somewhere close to it) instead of computing \
             the true, tiny-but-nonzero WMC",
            wmc[0],
        );
    }

    /// End-to-end aggregation check on `mdd_wmc_loss`: with deterministic `probs` (bypassing
    /// `gumbel_softmax`'s randomness), the per-sample average and batch average should match a
    /// plain scalar computation built from the same brute-force WMCs used above.
    #[test]
    fn mdd_wmc_loss_averages_per_sample_then_per_batch() {
        let device = NdArrayDevice::default();
        let domain_size = 3;
        let batch = two_sample_batch(&device, 0.0);

        let per_sample_probs: Vec<Vec<f64>> = vec![
            vec![0.2, 0.5, 0.3, 0.1, 0.3, 0.6, 0.4, 0.4, 0.2],
            vec![0.6, 0.3, 0.1, 0.2, 0.2, 0.6, 0.5, 0.25, 0.25],
        ];
        let flat: Vec<f32> = per_sample_probs.iter().flatten().map(|&v| v as f32).collect();
        let probs: Tensor<NdArray, 3> =
            Tensor::<NdArray, 1>::from_data(flat.as_slice(), &device).reshape([2, 3, domain_size]);

        let loss = mdd_wmc_loss(probs, &batch);
        let loss_value: f32 = loss.into_data().to_vec::<f32>().unwrap()[0];

        // Each sample has exactly 2 constraints (AllDifferent, NotEquals): its own average is the
        // mean of their `-log(wmc)`, and the batch loss is the mean of those two.
        let mut expected_per_sample = Vec::new();
        for probs in &per_sample_probs {
            let not_equals_wmc = brute_force_not_equals(probs, domain_size);
            let all_different_wmc = brute_force_all_different_permutation(probs, domain_size);
            let sample_avg = (-(not_equals_wmc).ln() - (all_different_wmc).ln()) / 2.0;
            expected_per_sample.push(sample_avg);
        }
        let expected = expected_per_sample.iter().sum::<f64>() / expected_per_sample.len() as f64;

        assert!(
            (loss_value as f64 - expected).abs() < 1e-4,
            "got {loss_value} expected {expected}"
        );
    }

    /// `bucket_log_wmc`'s DP leans on `gather`/`scatter`, which -- unlike the plain matmul the
    /// classical `ConsFormerLoss` uses -- aren't things every autodiff backend is guaranteed to
    /// support cleanly. This checks that burn's autodiff backend does back-propagate through the
    /// whole batched WMC DP: with `logits` requiring grad, the full `Loss::loss` (including
    /// `gumbel_softmax` and `blend_with_current`) should produce a finite, non-all-zero gradient
    /// of the expected shape.
    #[test]
    fn loss_backpropagates_through_the_batched_wmc_dp() {
        use burn::backend::Autodiff;
        use burn::tensor::Distribution;

        type ADBackend = Autodiff<NdArray>;

        let device = NdArrayDevice::default();
        let batch = two_sample_batch::<ADBackend>(&device, 1.0);

        let logits: Tensor<ADBackend, 3> =
            Tensor::random([2, 3, 3], Distribution::Uniform(-1.0, 1.0), &device).require_grad();

        let loss = ConsFormerMddLoss.loss(logits.clone(), &batch);
        let loss_value: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
        assert!(loss_value.is_finite(), "loss should be finite, got {loss_value}");

        let grads = loss.backward();
        let grad = logits
            .grad(&grads)
            .expect("logits should have a gradient after backward()");
        assert_eq!(grad.dims(), [2, 3, 3]);

        let grad_values: Vec<f32> = grad.into_data().to_vec::<f32>().unwrap();
        assert!(
            grad_values.iter().all(|v| v.is_finite()),
            "every gradient entry should be finite"
        );
        assert!(
            grad_values.iter().any(|&v| v != 0.0),
            "gradient should not be identically zero"
        );
    }
}
