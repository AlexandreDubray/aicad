use std::collections::HashMap;

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, IndexingUpdateOp, Int, Tensor};

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

/// "No two variables share a value" relaxation: penalizes any pair of
/// variables in the scope putting nonzero probability on the same value.
fn pairwise_collision_penalty<B: Backend>(probs: Tensor<B, 2>) -> Tensor<B, 1> {
    let collisions = probs.clone().matmul(probs.transpose());
    collisions.triu(1).sum().reshape([1])
}

/// Penalty for permutation constraints (e.g., all-diff with exactly the same number of domain
/// value as variables)
fn permutation_penalty<B: Backend>(probs: Tensor<B, 2>) -> Tensor<B, 1> {
    let [_, domain_size] = probs.dims();
    let coverage: Tensor<B, 2> = probs.sum_dim(0).reshape([1, domain_size]);
    let diff = coverage.sub_scalar(1.0);
    (diff.clone() * diff).sum().reshape([1])
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

        // One batched matmul + triu + sum per group, instead of one op-chain
        // per instance.
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

        // One batched sum-reduction per group, instead of one op-chain per
        // instance.
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
            total = total + (diff.clone() * diff).sum().reshape([1]);
        }

        total.div_scalar(batch_size as f32)
    }
}

/// Runs the batched forward WMC DP (see the module doc on `MddInstance`/`MddBucketBatch` for the
/// edge-list layout) over every instance in one padding bucket at once.
/// Returns one WMC per instance, `(batch_size,)`, in the same order as `bucket.sample_index`.
fn bucket_wmc<B: Backend>(bucket: &MddBucketBatch<B>, flat_probs: &Tensor<B, 1>) -> Tensor<B, 1> {
    let device = flat_probs.device();
    let num_instances = bucket.sample_index.dims()[0];
    let max_nodes = bucket.key.max_nodes;
    let max_edges = bucket.key.max_edges;
    let num_layers = bucket.key.num_layers;

    // w[i, 0] = 1: every instance starts at its MDD's root, local node index 0 of layer 0
    let mut root = vec![0.0f32; num_instances * max_nodes];
    for i in 0..num_instances {
        root[i * max_nodes] = 1.0;
    }
    let mut w: Tensor<B, 2> =
        Tensor::<B, 1>::from_data(root.as_slice(), &device).reshape([num_instances, max_nodes]);

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

        // The probability each edge's decision puts on its assignment: one gather from the
        // whole batch's flattened probability vector per edge slot (`gather_index` is already
        // offset per-sample -- see `ConsFormerMddBatcher`).
        let flat_gather: Tensor<B, 1, Int> = gather_layer.reshape([num_instances * max_edges]);
        let probs_at_edges: Tensor<B, 2> = flat_probs
            .clone()
            .select(0, flat_gather)
            .reshape([num_instances, max_edges]);

        // Each edge's contribution is its source node's current weight times its own decision's
        // probability; padding slots are zeroed via `mask_layer` before they can add anything --
        // their `edge_to`/`edge_from`/`gather_index` values are in-bounds but not meaningful.
        let w_from = w.clone().gather(1, from_layer);
        let contributions = (w_from * probs_at_edges) * mask_layer.float();

        // Sum every edge's contribution into its target node -- the DP's "sum over incoming
        // edges" step. `scatter`'s `Add` update naturally sums the routine case of two edges
        // sharing a target node, which is exactly why this is an edge list rather than a dense
        // `(to, from)` grid (see `MddInstance`'s doc).
        let next = Tensor::<B, 2>::zeros([num_instances, max_nodes], &device);
        w = next.scatter(1, to_layer, contributions, IndexingUpdateOp::Add);
    }

    // WMC = the sink's weight, local node index 0 of the last layer -- same convention as root.
    let sink_idx = Tensor::<B, 1, Int>::from_data([0i64], &device);
    w.select(1, sink_idx).reshape([num_instances])
}

/// Trains ConsFormer against the *exact* weighted model count of each constraint's
/// Loss per sample is the average, over that sample's own constraints, of `-log(wmc + epsilon)`;
/// the batch loss is the average of that over every sample.
pub struct ConsFormerMddLoss {
    pub epsilon: f64,
}

impl Default for ConsFormerMddLoss {
    fn default() -> Self {
        Self { epsilon: 1e-6 }
    }
}

/// The actual WMC-loss computation, given `probs` directly rather than the raw `logits`
fn mdd_wmc_loss<B: Backend>(
    probs: Tensor<B, 3>,
    batch: &ConsFormerMddBatch<B>,
    epsilon: f64,
) -> Tensor<B, 1> {
    let batch_size = batch.problems().len();
    let [_, number_vars, domain_size] = probs.dims();
    let device = probs.device();

    // Flatten (batch, vars, domain) -> (batch*vars*domain,): `MddBucketBatch::gather_index` is
    // already offset by `sample_index * number_vars * domain_size` (see `ConsFormerMddBatcher`),
    // so it indexes directly into this.
    let flat_probs: Tensor<B, 1> = probs.reshape([batch_size * number_vars * domain_size]);

    // Every constraint's `-log(wmc + eps)`, scattered into its sample's running sum, plus a
    // running count of how many constraints landed in each sample
    let mut per_sample_sum = Tensor::<B, 1>::zeros([batch_size], &device);
    let mut per_sample_count = Tensor::<B, 1>::zeros([batch_size], &device);

    for bucket in &batch.mdd_buckets {
        let wmc = bucket_wmc(bucket, &flat_probs);
        let neg_log_wmc = -(wmc.add_scalar(epsilon).log());
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

    let per_sample_avg = per_sample_sum / per_sample_count.clamp_min(1.0);
    per_sample_avg.mean()
}

impl<B: Backend> Loss<B, ConsFormerMddBatch<B>> for ConsFormerMddLoss {
    fn loss(&self, logits: Tensor<B, 3>, batch: &ConsFormerMddBatch<B>) -> Tensor<B, 1> {
        let probs = gumbel_softmax(logits);
        let probs = blend_with_current(probs, batch.assignments.clone(), batch.var_masks.clone());
        mdd_wmc_loss(probs, batch, self.epsilon)
    }
}

#[cfg(test)]
mod mdd_loss_tests {
    use std::sync::Arc;

    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::data::dataloader::batcher::Batcher;
    use burn::data::dataset::Dataset;

    use crate::modelling::{all_different, not_equals, Problem};

    use super::super::mdd_dataset::{
        ConsFormerMddBatcher, ConsFormerMddDataset, MddCompilationConfig,
    };
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
    fn two_sample_batch<B: Backend>(
        device: &B::Device,
        mask_fraction: f64,
    ) -> ConsFormerMddBatch<B> {
        let mut problems = Vec::new();
        for _ in 0..2 {
            let mut problem = Problem::default();
            let vars = problem.add_variables(3, vec![0, 1, 2], None);
            all_different(&mut problem, vars.clone());
            not_equals(&mut problem, vars[0], vars[1]);
            problems.push(Arc::new(problem));
        }

        let domain_size = 3;
        let dataset = ConsFormerMddDataset::<B>::new(
            problems,
            MddCompilationConfig::default(),
            domain_size,
            device,
        );
        let samples: Vec<_> = (0..dataset.len())
            .map(|i| dataset.get(i).unwrap())
            .collect();

        let batcher = ConsFormerMddBatcher {
            mask_fraction,
            domain_size,
        };
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

    /// `bucket_wmc` is the tensor-batched DP; this checks it against a plain brute-force
    /// enumeration for every instance of every bucket in a two-sample batch, using a different
    /// probability distribution per sample so a bug in the per-sample `gather_index` offset
    /// (rather than the DP itself) would also be caught.
    #[test]
    fn bucket_wmc_matches_brute_force_across_a_batch() {
        let device = NdArrayDevice::default();
        let domain_size = 3;
        let batch = two_sample_batch(&device, 0.0);

        let per_sample_probs: Vec<Vec<f64>> = vec![
            vec![0.2, 0.5, 0.3, 0.1, 0.3, 0.6, 0.4, 0.4, 0.2],
            vec![0.6, 0.3, 0.1, 0.2, 0.2, 0.6, 0.5, 0.25, 0.25],
        ];
        let flat: Vec<f32> = per_sample_probs
            .iter()
            .flatten()
            .map(|&v| v as f32)
            .collect();
        let flat_probs: Tensor<NdArray, 1> = Tensor::from_data(flat.as_slice(), &device);

        assert_eq!(batch.mdd_buckets.len(), 2);
        for bucket in &batch.mdd_buckets {
            let wmc: Vec<f32> = bucket_wmc(bucket, &flat_probs)
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
        let flat: Vec<f32> = per_sample_probs
            .iter()
            .flatten()
            .map(|&v| v as f32)
            .collect();
        let probs: Tensor<NdArray, 3> =
            Tensor::<NdArray, 1>::from_data(flat.as_slice(), &device).reshape([2, 3, domain_size]);

        let epsilon = 1e-6;
        let loss = mdd_wmc_loss(probs, &batch, epsilon);
        let loss_value: f32 = loss.into_data().to_vec::<f32>().unwrap()[0];

        // Each sample has exactly 2 constraints (AllDifferent, NotEquals): its own average is the
        // mean of their `-log(wmc + eps)`, and the batch loss is the mean of those two.
        let mut expected_per_sample = Vec::new();
        for probs in &per_sample_probs {
            let not_equals_wmc = brute_force_not_equals(probs, domain_size);
            let all_different_wmc = brute_force_all_different_permutation(probs, domain_size);
            let sample_avg =
                (-(not_equals_wmc + epsilon).ln() - (all_different_wmc + epsilon).ln()) / 2.0;
            expected_per_sample.push(sample_avg);
        }
        let expected = expected_per_sample.iter().sum::<f64>() / expected_per_sample.len() as f64;

        assert!(
            (loss_value as f64 - expected).abs() < 1e-4,
            "got {loss_value} expected {expected}"
        );
    }

    /// `bucket_wmc`'s DP leans on `gather`/`scatter`, which -- unlike the plain matmul the
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

        let loss = ConsFormerMddLoss::default().loss(logits.clone(), &batch);
        let loss_value: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            loss_value.is_finite(),
            "loss should be finite, got {loss_value}"
        );

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
