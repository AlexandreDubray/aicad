use burn::tensor::activation::softmax;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};

use crate::constraints::{AllDifferent, Constraint};
use crate::learning::{HasProblems, Loss};

use super::architecture::ConsFormer;
use super::dataset::ConsFormerBatch;

/// Loss trait for ConsFormer. Given a tensor (number_var, domain_size), computes a penalty term
/// for the constraint. We assume that the probability tensor sent to the constraint is restricted
/// to its scope
pub trait ConstraintLoss<B: Backend> {
    fn constraint_penalty(&self, probs: Tensor<B, 2>) -> Tensor<B, 1>;
}

impl<B: Backend> ConstraintLoss<B> for AllDifferent {
    // TODO: Handle the two cases defined in ConsFormer paper's based on the domain size and the
    // number of variable in the scope
    fn constraint_penalty(&self, probs: Tensor<B, 2>) -> Tensor<B, 1> {
        let collisions = probs.clone().matmul(probs.transpose());
        collisions.triu(1).sum().reshape([1])
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

    panic!(
        "no ConstraintLoss implementation for constraint type `{}` -- add one in src/learning/consformer/loss.rs",
        constraint.name()
    );
}

pub struct ConsFormerLoss;

impl<B: AutodiffBackend> Loss<B, ConsFormer<B>> for ConsFormerLoss {
    fn loss(&self, logits: Tensor<B, 3>, batch: &ConsFormerBatch<B>) -> Tensor<B, 1> {
        let probs = softmax(logits, 2);
        let problems = batch.problems();
        let batch_size = problems.len();
        let device = probs.device();

        let mut total = Tensor::<B, 1>::zeros([1], &device);

        for (i, problem) in problems.iter().enumerate() {
            let sample_probs: Tensor<B, 2> = probs.clone().slice([i..i + 1]).squeeze();

            for constraint in problem.iter_constraints() {
                let loss = constraint_loss(&*problem[constraint], &sample_probs);
                total = total + loss;
            }
        }

        total.div_scalar(batch_size as f32)
    }
}
