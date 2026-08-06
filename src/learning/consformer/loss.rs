use std::collections::HashMap;

use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::constraints::{AllDifferent, Constraint, NotEquals};
use crate::learning::{HasProblems, Loss};

use super::architecture::ConsFormer;
use super::dataset::ConsFormerBatch;

/// Loss trait for ConsFormer. Given a tensor (number_var, domain_size), computes a penalty term
/// for the constraint. We assume that the probability tensor sent to the constraint is restricted
/// to its scope
pub trait ConstraintLoss<B: Backend> {
    fn constraint_penalty(&self, probs: Tensor<B, 2>) -> Tensor<B, 1>;
}

fn pairwise_collision_penalty<B: Backend>(probs: Tensor<B, 2>) -> Tensor<B, 1> {
    let collisions = probs.clone().matmul(probs.transpose());
    collisions.triu(1).sum().reshape([1])
}

impl<B: Backend> ConstraintLoss<B> for AllDifferent {
    // TODO: Handle the two cases defined in ConsFormer paper's based on the domain size and the
    // number of variable in the scope
    fn constraint_penalty(&self, probs: Tensor<B, 2>) -> Tensor<B, 1> {
        pairwise_collision_penalty(probs)
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

pub struct ConsFormerLoss;

impl<B: Backend> Loss<B, ConsFormer<B>> for ConsFormerLoss {
    fn loss(&self, logits: Tensor<B, 3>, batch: &ConsFormerBatch<B>) -> Tensor<B, 1> {
        let probs = softmax(logits, 2);
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

        // Group collision based constraints (i.e., the only constraints currently supported by
        // ConsFormer) so their loss can be computed as a single matmul operation, leading to
        // faster gradient computation
        let mut collision_groups: HashMap<usize, Vec<i64>> = HashMap::new();
        let mut total = Tensor::<B, 1>::zeros([1], &device);

        for (i, problem) in problems.iter().enumerate() {
            let sample_offset = (i * number_vars) as i64;

            for constraint in problem.iter_constraints() {
                let c = &*problem[constraint];
                let is_collision = c.as_any().downcast_ref::<AllDifferent>().is_some()
                    || c.as_any().downcast_ref::<NotEquals>().is_some();

                if is_collision {
                    let scope: Vec<i64> =
                        c.iter_scope().map(|v| sample_offset + v.0 as i64).collect();
                    collision_groups
                        .entry(scope.len())
                        .or_default()
                        .extend(scope);
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

        total.div_scalar(batch_size as f32)
    }
}
