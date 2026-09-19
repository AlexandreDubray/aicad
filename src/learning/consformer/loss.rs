use std::collections::HashMap;

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use rayon::prelude::*;

use crate::constraints::{AllDifferent, Constraint, NotEquals};
use crate::learning::{BatchProblems, Loss};
use crate::mdd::wmc::wmc_and_gradient;
use crate::mdd::CompiledConstraint;
use crate::modelling::Problem;

use super::dataset::ConsFormerBatch;
use super::mdd_dataset::ConsFormerMddBatch;

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
/// summing them, a Sudoku-specific "which of the 27 all-diffs is this" constrainting that this generic,
/// constraint-type-agnostic code has no equivalent of (a `Problem`/`Constraint` here carries no
/// notion of belonging to one of several named constraints). So this is not bit-exact with the Python
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
        // constraint can be computed as a single tensor op instead of one
        // op-chain per instance. `NotEquals`, and `AllDifferent` whose scope
        // is smaller than the domain, use the pairwise collision penalty
        // (batched via matmul); `AllDifferent` whose scope exactly covers
        // the domain (e.g. every Sudoku row/col/box) uses the permutation
        // penalty instead (batched via a sum reduction.
        let mut collision_constraints: HashMap<usize, Vec<i64>> = HashMap::new();
        let mut permutation_constraints: HashMap<usize, Vec<i64>> = HashMap::new();
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
                    permutation_constraints
                        .entry(scope_len)
                        .or_default()
                        .extend(scope);
                } else if is_all_different || is_not_equals {
                    let scope: Vec<i64> =
                        c.iter_scope().map(|v| sample_offset + v.0 as i64).collect();
                    collision_constraints
                        .entry(scope_len)
                        .or_default()
                        .extend(scope);
                } else {
                    let sample_probs: Tensor<B, 2> = probs.clone().slice([i..i + 1]).squeeze();
                    total = total + constraint_loss(c, &sample_probs);
                }
            }
        }

        // One batched matmul + triu + sum per constraint, instead of one op-chain per instance. See
        // `pairwise_collision_penalty`'s doc: this is the code path that actually runs for
        // `NotEquals`/small-scope `AllDifferent` during training (the free function above isn't
        // reachable for those two types), and it's already equivalent, up to a benign constant
        // factor of 2, to `CustomGCOLLossDot`.
        for (scope_len, flat_indices) in collision_constraints {
            let num_instances = flat_indices.len() / scope_len;
            let idx = Tensor::<B, 1, Int>::from_data(flat_indices.as_slice(), &device);
            let constraint_probs: Tensor<B, 3> =
                flat_probs
                    .clone()
                    .select(0, idx)
                    .reshape([num_instances, scope_len, domain_size]);

            let collisions = constraint_probs
                .clone()
                .matmul(constraint_probs.transpose());
            total = total + collisions.triu(1).sum().reshape([1]);
        }

        // One batched mean-reduction per constraint, instead of one op-chain per instance. See
        // `permutation_penalty`'s doc: mean- (not sum-) reduced to match `CustomSudokuLossMSE`'s
        // `nn.MSELoss()` reduction, modulo the row/column/box constrainting this generic code can't
        // replicate. This is the code path that actually runs for `AllDifferent` with
        // `scope_len == domain_size` during training.
        for (scope_len, flat_indices) in permutation_constraints {
            let num_instances = flat_indices.len() / scope_len;
            let idx = Tensor::<B, 1, Int>::from_data(flat_indices.as_slice(), &device);
            let constraint_probs: Tensor<B, 3> =
                flat_probs
                    .clone()
                    .select(0, idx)
                    .reshape([num_instances, scope_len, domain_size]);

            // Sum over the scope (dim 1): per constraint instance, how much
            // probability mass each value received across the whole scope.
            let coverage: Tensor<B, 2> = constraint_probs
                .sum_dim(1)
                .reshape([num_instances, domain_size]);
            let diff = coverage.sub_scalar(1.0);
            total = total + (diff.clone() * diff).mean().reshape([1]);
        }

        total.div_scalar(batch_size as f32)
    }
}

const WMC_EPS: f64 = 1e-6;
const SATISFACTION_WEIGHT_FLOOR: f64 = 1e-3;

/// One constraint MDD's contribution to a sample's loss and to `∂loss/∂weight`, computed by a
/// plain sink-to-root chain-rule pass (`crate::mdd::wmc::wmc_and_gradient`) instead of
/// automatic differentiation.
/// `probs_for_sample` is that sample's flattened `(number_vars, domain_size)` probability slice,
/// indexed by each variable's *raw domain value*, not by `ValueIndex` position.
/// `grad` accumulates `∂(weight * -log(wmc+eps))/∂probs_for_sample` into that same indexing, with
/// `weight` treated as a constant (stop-gradient on `wmc`)
/// Builds `constraint`'s per-layer weights from the raw network probabilities, **masking out** every
/// raw domain value that isn't actually in `real_problem`'s (possibly narrowed -- e.g. a Sudoku
/// given, or a masked/destroyed variable) domain for that layer's variable.
fn layer_weights_from_probs(
    constraint: &CompiledConstraint,
    real_problem: &Problem,
    probs_for_sample: &[f32],
    domain_size: usize,
) -> Vec<Vec<f64>> {
    let num_layers = constraint.number_layers() - 1;
    let mut weights: Vec<Vec<f64>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let variable = constraint.decision_at_layer(layer);
        let mut layer_weights = vec![0.0; domain_size];
        for (raw_value, w) in layer_weights.iter_mut().enumerate() {
            if real_problem[variable].in_domain(raw_value as isize) {
                *w = probs_for_sample[variable.0 * domain_size + raw_value] as f64;
            }
        }
        weights.push(layer_weights);
    }
    weights
}

fn mdd_loss_and_gradient(
    constraint: &CompiledConstraint,
    real_problem: &Problem,
    probs_for_sample: &[f32],
    domain_size: usize,
    grad: &mut [f32],
    gamma: f64,
) -> (f64, f64) {
    let weights = layer_weights_from_probs(constraint, real_problem, probs_for_sample, domain_size);
    let (wmc, gradient) = wmc_and_gradient(constraint, &weights);
    let weight = (1.0 - wmc).clamp(0.0, 1.0).powf(gamma) + SATISFACTION_WEIGHT_FLOOR;
    let chain = -weight / (wmc + WMC_EPS);

    for (layer, layer_gradient) in gradient.iter().enumerate() {
        let variable = constraint.decision_at_layer(layer);
        for (raw_value, &g) in layer_gradient.iter().enumerate() {
            // Masked the same way `layer_weights_from_probs` masked the forward weight: a raw
            // value outside `real_problem`'s actual domain has a *constant* (zero) weight,
            // independent of `probs_for_sample` -- so its contribution to d(loss)/d(probs) is
            // genuinely zero, not whatever d(wmc)/d(weight) the shared, unhinted structure
            // reports for that (structurally still-present, but never actually selectable) edge.
            if real_problem[variable].in_domain(raw_value as isize) {
                grad[variable.0 * domain_size + raw_value] += (chain * g) as f32;
            }
        }
    }

    (weight * -(wmc + WMC_EPS).ln(), weight)
}

/// A sample's satisfaction-weighted average `-log(wmc+eps)` over its constraint MDDs (each
/// constraint weighted by `(1 - wmc)^gamma + FLOOR`, so nearly-satisfied constraints contribute
/// little to the loss and to `grad`, and the still-violated ones dominate), plus
/// `∂(that weighted average)/∂probs` (`mdds` iterated sequentially -- parallelism is across
/// samples, see `mdd_wmc_loss`). `gamma = 0.0` recovers the plain, unweighted per-constraint mean
/// exactly: `(1 - wmc)^0 + FLOOR` is the same constant `1.0 + FLOOR` for every constraint
/// regardless of its own `wmc`, which cancels out of the weighted average.
fn sample_loss_and_gradient(
    constraints: &[CompiledConstraint],
    real_problem: &Problem,
    probs_for_sample: &[f32],
    number_vars: usize,
    domain_size: usize,
    gamma: f64,
) -> (f64, Vec<f32>) {
    let mut grad = vec![0.0f32; number_vars * domain_size];
    let mut weighted_loss_sum = 0.0;
    let mut weight_sum = 0.0;
    for constraint in constraints {
        let (weighted_loss, weight) = mdd_loss_and_gradient(
            constraint,
            real_problem,
            probs_for_sample,
            domain_size,
            &mut grad,
            gamma,
        );
        weighted_loss_sum += weighted_loss;
        weight_sum += weight;
    }
    let denom = weight_sum.max(SATISFACTION_WEIGHT_FLOOR);
    let loss = weighted_loss_sum / denom;
    for g in &mut grad {
        *g /= denom as f32;
    }
    (loss, grad)
}

/// `gamma` controls the focal-style satisfaction weighting in `sample_loss_and_gradient` --
/// `gamma = 0.0` recovers the plain unweighted per-constraint mean (the pre-focal-weighting
/// behaviour); larger `gamma` pushes down-weighting of nearly-satisfied constraints harder.
pub struct ConsFormerMddLoss {
    pub gamma: f64,
}

/// Computes the MDD-WMC loss and its exact gradient wrt `probs` by hand (see
/// `mdd_loss_and_gradient`), then hands that gradient to Burn's autodiff via the standard
/// stop-gradient identity `value + stop_gradient(desired_value - value)`, here as
/// `grad_tensor * (probs - probs.detach())`: its *value* is always zero (`probs.clone()` and
/// `probs.detach()` hold the same numbers), so the returned tensor's value is exactly
/// `loss_value`, but its derivative wrt `probs` is exactly `grad_tensor`, since `grad_tensor` and
/// `probs.detach()` are constants as far as autodiff is concerned. This keeps the whole hand-rolled
/// backward pass outside Burn's graph -- nothing per-layer gets recorded or replayed -- while
/// still letting `loss.backward()` reach every parameter upstream of `probs` normally.
fn mdd_wmc_loss<B: Backend>(
    probs: Tensor<B, 3>,
    batch: &ConsFormerMddBatch<B>,
    gamma: f64,
) -> Tensor<B, 1> {
    let device = probs.device();
    let [batch_size, number_vars, domain_size] = probs.dims();

    let probs_data: Vec<f32> = probs
        .clone()
        .into_data()
        .to_vec::<f32>()
        .expect("probs should convert to f32");

    let results: Vec<(f64, Vec<f32>)> = crate::utils::worker_pool().install(|| {
        batch
            .mdds
            .par_iter()
            .zip(batch.problems.par_iter())
            .enumerate()
            .map(|(i, (constraints, real_problem))| {
                let start = i * number_vars * domain_size;
                let sample_probs = &probs_data[start..start + number_vars * domain_size];
                sample_loss_and_gradient(
                    constraints,
                    real_problem,
                    sample_probs,
                    number_vars,
                    domain_size,
                    gamma,
                )
            })
            .collect()
    });

    let loss_value: f64 = results.iter().map(|(loss, _)| loss).sum::<f64>() / batch_size as f64;

    let mut grad_data = vec![0.0f32; batch_size * number_vars * domain_size];
    for (i, (_, grad)) in results.iter().enumerate() {
        let start = i * number_vars * domain_size;
        for (j, &g) in grad.iter().enumerate() {
            grad_data[start + j] = g / batch_size as f32;
        }
    }

    let grad_tensor: Tensor<B, 3> = Tensor::<B, 1>::from_data(grad_data.as_slice(), &device)
        .reshape([batch_size, number_vars, domain_size]);
    let loss_tensor: Tensor<B, 1> = Tensor::from_data([loss_value as f32].as_slice(), &device);

    (grad_tensor * (probs.clone() - probs.detach()))
        .sum()
        .reshape([1])
        + loss_tensor
}

impl<B: Backend> Loss<B, ConsFormerMddBatch<B>> for ConsFormerMddLoss {
    fn loss(&self, logits: Tensor<B, 3>, batch: &ConsFormerMddBatch<B>) -> Tensor<B, 1> {
        let probs = gumbel_softmax(logits);
        let probs = blend_with_current(probs, batch.assignments.clone(), batch.var_masks.clone());
        mdd_wmc_loss(probs, batch, self.gamma)
    }
}

#[cfg(test)]
mod mdd_loss_tests {
    use std::sync::Arc;

    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::data::dataloader::batcher::Batcher;
    use burn::data::dataset::Dataset;

    use crate::mdd::{CompiledConstraint, MddArena};
    use crate::modelling::{all_different, not_equals, Problem};

    use super::super::mdd_dataset::{
        ConsFormerMddBatcher, ConsFormerMddDataset, ConsFormerMddSample, MddCompilationConfig,
    };
    use super::super::ConsFormerDataConfig;
    use super::*;

    /// Every problem is 3 variables, domain `{0,1,2}`, with an `AllDifferent` over all three and
    /// a `NotEquals` over the first two. Generic over the backend so the same builder can be used
    /// both with plain `NdArray` (the WMC/loss correctness tests) and `Autodiff<NdArray>` (the
    /// gradient-flow tests).
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

        let data_config = ConsFormerDataConfig {
            domain_size: 3,
            mask_fraction,
        };
        let arena = MddArena::new();
        let dataset = ConsFormerMddDataset::<B>::new(
            problems,
            &arena,
            MddCompilationConfig::default(),
            data_config,
            device,
        );
        let samples: Vec<_> = (0..dataset.len())
            .map(|i| dataset.get(i).unwrap())
            .collect();

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

    fn constraint_wmcs(
        constraints: &[CompiledConstraint],
        real_problem: &Problem,
        probs_for_sample: &[f32],
        domain_size: usize,
    ) -> Vec<f64> {
        constraints
            .iter()
            .map(|constraint| {
                let weights = layer_weights_from_probs(
                    constraint,
                    real_problem,
                    probs_for_sample,
                    domain_size,
                );
                wmc_and_gradient(constraint, &weights).0
            })
            .collect()
    }

    fn frozen_weights_for(wmcs: &[f64], gamma: f64) -> Vec<f64> {
        wmcs.iter()
            .map(|&wmc| (1.0 - wmc).clamp(0.0, 1.0).powf(gamma) + SATISFACTION_WEIGHT_FLOOR)
            .collect()
    }

    /// Loss for one sample's constraints with `frozen_weights` held fixed instead of recomputed
    /// from the (possibly perturbed) `probs_for_sample` -- this is what the finite-difference
    /// checks below compare `sample_loss_and_gradient`'s hand-rolled gradient against, since that
    /// gradient deliberately treats each constraint's weight as a constant (stop-gradient on
    /// `wmc`), not as a differentiable function of `probs`.
    fn sample_loss_with_frozen_weights(
        constraints: &[CompiledConstraint],
        real_problem: &Problem,
        probs_for_sample: &[f32],
        domain_size: usize,
        frozen_weights: &[f64],
    ) -> f64 {
        let wmcs = constraint_wmcs(constraints, real_problem, probs_for_sample, domain_size);
        let mut weighted_loss_sum = 0.0;
        let mut weight_sum = 0.0;
        for (&wmc, &weight) in wmcs.iter().zip(frozen_weights) {
            weighted_loss_sum += weight * -(wmc + WMC_EPS).ln();
            weight_sum += weight;
        }
        weighted_loss_sum / weight_sum.max(SATISFACTION_WEIGHT_FLOOR)
    }

    fn mdd_wmc_loss_with_frozen_weights(
        probs: Tensor<NdArray, 3>,
        batch: &ConsFormerMddBatch<NdArray>,
        frozen_weights: &[Vec<f64>],
    ) -> f64 {
        let [batch_size, number_vars, domain_size] = probs.dims();
        let probs_data: Vec<f32> = probs.into_data().to_vec::<f32>().unwrap();
        let mut loss_sum = 0.0;
        for (i, mdds) in batch.mdds.iter().enumerate() {
            let start = i * number_vars * domain_size;
            let sample_probs = &probs_data[start..start + number_vars * domain_size];
            loss_sum += sample_loss_with_frozen_weights(
                mdds,
                &batch.problems[i],
                sample_probs,
                domain_size,
                &frozen_weights[i],
            );
        }
        loss_sum / batch_size as f64
    }

    fn weighted_average(wmcs: &[f64], gamma: f64) -> f64 {
        let mut weighted_loss_sum = 0.0;
        let mut weight_sum = 0.0;
        for &wmc in wmcs {
            let weight = (1.0 - wmc).clamp(0.0, 1.0).powf(gamma) + SATISFACTION_WEIGHT_FLOOR;
            weighted_loss_sum += weight * -(wmc + WMC_EPS).ln();
            weight_sum += weight;
        }
        weighted_loss_sum / weight_sum.max(SATISFACTION_WEIGHT_FLOOR)
    }

    #[test]
    fn sample_loss_and_gradient_matches_brute_force_average() {
        let device = NdArrayDevice::default();
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        all_different(&mut problem, vars.clone());
        not_equals(&mut problem, vars[0], vars[1]);
        let problem = Arc::new(problem);

        let data_config = ConsFormerDataConfig {
            domain_size: 3,
            mask_fraction: 0.0,
        };
        let arena = MddArena::new();
        let dataset = ConsFormerMddDataset::<NdArray>::new(
            vec![Arc::clone(&problem)],
            &arena,
            MddCompilationConfig::default(),
            data_config,
            &device,
        );
        let sample: ConsFormerMddSample<NdArray> = dataset.get(0).unwrap();
        let mdds = sample.mdds().clone();

        let probs: Vec<f32> = vec![0.2, 0.5, 0.3, 0.1, 0.3, 0.6, 0.4, 0.4, 0.2];
        let (loss, _grad) = sample_loss_and_gradient(&mdds, &problem, &probs, 3, 3, 1.0);

        let probs_f64: Vec<f64> = probs.iter().map(|&v| v as f64).collect();
        let not_equals_wmc = brute_force_not_equals(&probs_f64, 3);
        let all_different_wmc = brute_force_all_different_permutation(&probs_f64, 3);
        let expected = weighted_average(&[not_equals_wmc, all_different_wmc], 1.0);

        assert!(
            (loss - expected).abs() < 1e-6,
            "got {loss} expected {expected}"
        );
    }

    #[test]
    fn sample_loss_and_gradient_matches_finite_differences() {
        let device = NdArrayDevice::default();
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        all_different(&mut problem, vars.clone());
        not_equals(&mut problem, vars[0], vars[1]);
        let problem = Arc::new(problem);

        let data_config = ConsFormerDataConfig {
            domain_size: 3,
            mask_fraction: 0.0,
        };
        let arena = MddArena::new();
        let dataset = ConsFormerMddDataset::<NdArray>::new(
            vec![Arc::clone(&problem)],
            &arena,
            MddCompilationConfig::default(),
            data_config,
            &device,
        );
        let sample: ConsFormerMddSample<NdArray> = dataset.get(0).unwrap();
        let mdds = sample.mdds().clone();

        let probs: Vec<f32> = vec![0.2, 0.5, 0.3, 0.1, 0.3, 0.6, 0.4, 0.4, 0.2];
        let (base_loss, grad) = sample_loss_and_gradient(&mdds, &problem, &probs, 3, 3, 1.0);
        let frozen_weights = frozen_weights_for(&constraint_wmcs(&mdds, &problem, &probs, 3), 1.0);

        let eps = 1e-4f32;
        for i in 0..probs.len() {
            let mut bumped = probs.clone();
            bumped[i] += eps;
            let bumped_loss =
                sample_loss_with_frozen_weights(&mdds, &problem, &bumped, 3, &frozen_weights);
            let finite_diff = (bumped_loss - base_loss) / eps as f64;
            assert!(
                (finite_diff - grad[i] as f64).abs() < 1e-2,
                "index {i}: analytic={} finite_diff={finite_diff}",
                grad[i],
            );
        }
    }

    #[test]
    fn mdd_wmc_loss_averages_per_sample_then_per_batch() {
        let device = NdArrayDevice::default();
        let domain_size = 3;
        let batch = two_sample_batch::<NdArray>(&device, 0.0);

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

        let loss = mdd_wmc_loss(probs, &batch, 1.0);
        let loss_value: f32 = loss.into_data().to_vec::<f32>().unwrap()[0];

        let mut expected_per_sample = Vec::new();
        for probs in &per_sample_probs {
            let not_equals_wmc = brute_force_not_equals(probs, domain_size);
            let all_different_wmc = brute_force_all_different_permutation(probs, domain_size);
            expected_per_sample.push(weighted_average(&[not_equals_wmc, all_different_wmc], 1.0));
        }
        let expected = expected_per_sample.iter().sum::<f64>() / expected_per_sample.len() as f64;

        assert!(
            (loss_value as f64 - expected).abs() < 1e-4,
            "got {loss_value} expected {expected}"
        );
    }

    /// The custom gradient this loss hands to Burn's autodiff (`mdd_loss_and_gradient`'s
    /// sink-to-root chain rule) bypasses Burn's own backward machinery entirely, so this checks it
    /// against finite differences of `mdd_wmc_loss` itself -- computed on a plain, non-autodiff
    /// `NdArray` backend at perturbed `probs` -- rather than against a second differentiation path
    /// through Burn. `mdd_wmc_loss` is deterministic (no `gumbel_softmax` involved -- that's called
    /// only by `Loss::loss`, above this function), so this is an exact check, not just a
    /// finite/non-zero sanity check.
    #[test]
    fn mdd_wmc_loss_gradient_matches_finite_differences_via_autodiff() {
        use burn::backend::Autodiff;

        type ADBackend = Autodiff<NdArray>;
        let ad_device = NdArrayDevice::default();
        let plain_device = NdArrayDevice::default();
        let domain_size = 3;

        let batch_ad = two_sample_batch::<ADBackend>(&ad_device, 0.0);
        let batch_plain = two_sample_batch::<NdArray>(&plain_device, 0.0);

        let flat: Vec<f32> = vec![
            0.2, 0.5, 0.3, 0.1, 0.3, 0.6, 0.4, 0.4, 0.2, 0.6, 0.3, 0.1, 0.2, 0.2, 0.6, 0.5, 0.25,
            0.25,
        ];

        let probs_ad: Tensor<ADBackend, 3> =
            Tensor::<ADBackend, 1>::from_data(flat.as_slice(), &ad_device)
                .reshape([2, 3, domain_size])
                .require_grad();
        let loss = mdd_wmc_loss(probs_ad.clone(), &batch_ad, 1.0);
        let grads = loss.backward();
        let grad = probs_ad
            .grad(&grads)
            .expect("probs should have a gradient after backward()");
        let grad_values: Vec<f32> = grad.into_data().to_vec::<f32>().unwrap();

        let base_tensor: Tensor<NdArray, 3> =
            Tensor::<NdArray, 1>::from_data(flat.as_slice(), &plain_device).reshape([
                2,
                3,
                domain_size,
            ]);
        let base_data: Vec<f32> = base_tensor.clone().into_data().to_vec::<f32>().unwrap();
        let frozen_weights: Vec<Vec<f64>> = batch_plain
            .mdds
            .iter()
            .enumerate()
            .map(|(i, mdds)| {
                let start = i * 3 * domain_size;
                let sample_probs = &base_data[start..start + 3 * domain_size];
                frozen_weights_for(
                    &constraint_wmcs(mdds, &batch_plain.problems[i], sample_probs, domain_size),
                    1.0,
                )
            })
            .collect();
        let base_loss =
            mdd_wmc_loss_with_frozen_weights(base_tensor, &batch_plain, &frozen_weights);

        let eps = 1e-4f32;
        for i in 0..flat.len() {
            let mut bumped = flat.clone();
            bumped[i] += eps;
            let bumped_tensor: Tensor<NdArray, 3> =
                Tensor::<NdArray, 1>::from_data(bumped.as_slice(), &plain_device).reshape([
                    2,
                    3,
                    domain_size,
                ]);
            let bumped_loss =
                mdd_wmc_loss_with_frozen_weights(bumped_tensor, &batch_plain, &frozen_weights);
            let finite_diff = (bumped_loss - base_loss) / eps as f64;
            assert!(
                (finite_diff - grad_values[i] as f64).abs() < 1e-2,
                "index {i}: analytic={} finite_diff={finite_diff}",
                grad_values[i],
            );
        }
    }

    /// End-to-end through the full `Loss::loss` path (`gumbel_softmax` + `blend_with_current` +
    /// `mdd_wmc_loss`): with `logits` requiring grad, `.backward()` should reach `logits` with a
    /// finite, non-all-zero gradient of the expected shape. Unlike the test above, this can't be
    /// checked against an exact finite-difference value -- `gumbel_softmax` draws fresh noise on
    /// every call -- so it's a sanity check that the gradient actually flows end to end, not a
    /// correctness check on the custom gradient math itself (that's covered above).
    #[test]
    fn loss_backpropagates_through_gumbel_and_blend() {
        use burn::backend::Autodiff;
        use burn::tensor::Distribution;

        type ADBackend = Autodiff<NdArray>;

        let device = NdArrayDevice::default();
        let batch = two_sample_batch::<ADBackend>(&device, 1.0);

        let logits: Tensor<ADBackend, 3> =
            Tensor::random([2, 3, 3], Distribution::Uniform(-1.0, 1.0), &device).require_grad();

        let loss = ConsFormerMddLoss { gamma: 1.0 }.loss(logits.clone(), &batch);
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

    /// A `NotEquals` constraint both of whose variables are pinned (`var_masks` all false) to the
    /// *same* value is unconditionally violated: `wmc = 0` for that MDD regardless of what the
    /// network predicts, so `mdd_loss_and_gradient`'s `-log(wmc+eps)`/`-1/(wmc+eps)` genuinely hit
    /// their `WMC_EPS` floor rather than some comfortably-nonzero value. This checks the loss and
    /// its gradient both stay finite in that regime -- the case `WMC_EPS` exists for.
    #[test]
    fn loss_stays_finite_when_a_pinned_assignment_violates_a_constraint() {
        use burn::backend::Autodiff;

        type ADBackend = Autodiff<NdArray>;
        let device = NdArrayDevice::default();

        let mut problem = Problem::default();
        let x = problem.add_variable(vec![0, 1, 2], None);
        let y = problem.add_variable(vec![0, 1, 2], None);
        not_equals(&mut problem, x, y);
        let problem = Arc::new(problem);

        let data_config = ConsFormerDataConfig {
            domain_size: 3,
            mask_fraction: 0.0,
        };
        let arena = MddArena::new();
        let dataset = ConsFormerMddDataset::<ADBackend>::new(
            vec![problem],
            &arena,
            MddCompilationConfig::default(),
            data_config,
            &device,
        );
        let samples: Vec<_> = (0..dataset.len())
            .map(|i| dataset.get(i).unwrap())
            .collect();
        let batcher = ConsFormerMddBatcher::new(data_config);
        let batch = batcher.batch(samples, &device);

        // `mask_fraction = 0.0` above means `var_masks` is all-false, so `blend_with_current`
        // pins every variable to `batch.assignments` regardless of `logits` -- setting both
        // assignments to 0 violates `not_equals(x, y)` unconditionally.
        let assignments: Tensor<ADBackend, 2, Int> =
            Tensor::<ADBackend, 1, Int>::from_data([0i64, 0i64].as_slice(), &device)
                .reshape([1, 2]);
        let batch = ConsFormerMddBatch {
            assignments,
            ..batch
        };

        let logits: Tensor<ADBackend, 3> = Tensor::zeros([1, 2, 3], &device).require_grad();

        let loss = ConsFormerMddLoss { gamma: 1.0 }.loss(logits.clone(), &batch);
        let loss_value: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
        assert!(
            loss_value.is_finite(),
            "loss should be finite, got {loss_value}"
        );
        assert!(
            (loss_value as f64 - (-(WMC_EPS).ln())).abs() < 1e-3,
            "an unconditionally-violated constraint should read back as -log(WMC_EPS), got {loss_value}",
        );

        let grads = loss.backward();
        let grad = logits
            .grad(&grads)
            .expect("logits should have a gradient after backward()");
        let grad_values: Vec<f32> = grad.into_data().to_vec::<f32>().unwrap();
        assert!(
            grad_values.iter().all(|v| v.is_finite()),
            "every gradient entry should be finite even when a constraint is unconditionally violated -- got {:?}",
            grad_values,
        );
    }

    /// Direct correctness check for the weight-masking fix `layer_weights_from_probs` needs now
    /// that `constraint.structure` is compiled with no per-instance domain restriction (see
    /// `crate::mdd::arena`'s module doc): a real "given" position (its domain narrowed to a
    /// single value, e.g. a Sudoku clue) must have every *other* raw value's weight forced to
    /// zero, even though the shared, un-hinted `AllDifferent` structure has a real edge for each
    /// of them. Checks both that the masked weight vector itself is a one-hot at the given value,
    /// and that `wmc` over those masked weights exactly matches a brute-force count restricted to
    /// assignments honouring the given -- i.e. sampling/`wmc`/gradient never actually credits an
    /// assignment that uses a masked-out value at the given position.
    #[test]
    fn layer_weights_from_probs_masks_out_a_given_values_forbidden_domain() {
        let domain_size = 4;
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, (0..domain_size as isize).collect(), None);
        all_different(&mut problem, vars.clone());
        // `vars[0]` is a "given": its real domain is narrowed to a single value, but the shared
        // template below is still compiled over the full nominal domain (no hint baked in).
        crate::modelling::equal(&mut problem, vars[0], 2);
        let problem = Arc::new(problem);

        let arena = MddArena::new();
        let constraint = problem
            .iter_constraints()
            .next()
            .expect("problem must have exactly one constraint");
        let constraint = crate::mdd::arena::compile_constraint(
            &arena,
            &problem,
            constraint,
            &crate::mdd::heuristics::OrderingHeuristic::MinDomMaxLinked,
            domain_size,
            usize::MAX,
        );

        // Uniform raw probabilities everywhere -- if masking didn't happen, every position
        // (including the given's 3 forbidden values) would carry equal, nonzero weight.
        let probs_for_sample: Vec<f32> = vec![0.25; 4 * domain_size];
        let weights =
            layer_weights_from_probs(&constraint, &problem, &probs_for_sample, domain_size);

        let given_layer = (0..constraint.number_layers() - 1)
            .find(|&l| constraint.decision_at_layer(l) == vars[0])
            .expect("vars[0] must be in the compiled constraint's scope");
        assert_eq!(
            weights[given_layer],
            vec![0.0, 0.0, 0.25, 0.0],
            "a given position's weight vector must be zero everywhere but its one allowed value"
        );

        let wmc = crate::mdd::wmc::wmc(&constraint, &weights);

        // Brute force: every permutation of {0,1,2,3} over the 4 variables with vars[0] fixed to
        // 2, each weighted 0.25^4 (uniform raw probability at every position).
        let mut brute = 0.0;
        for a in 0..domain_size {
            for b in 0..domain_size {
                for c in 0..domain_size {
                    let full = [2usize, a, b, c];
                    let vals: std::collections::HashSet<usize> = full.iter().copied().collect();
                    if vals.len() == 4 {
                        brute += 0.25f64.powi(4);
                    }
                }
            }
        }
        assert!(
            (wmc - brute).abs() < 1e-9,
            "wmc={wmc} brute={brute} -- masked wmc must match assignments honouring the given"
        );
    }
}
