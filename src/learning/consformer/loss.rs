//! ConsFormer original loss. This is a reimplementation of ConsFormer original loss. In order to
//! be efficient while retaining some genericity (as opposed to the [initial ConsFormer
//! code](https://github.com/khalil-research/ConsFormer) that uses one loss per problem), we group
//! constraint that can be grouped together so that matrix operations are used on the tensors to
//! compute the loss.
//!
//! # References
//!
//! "Self-Supervised Transformers as Iterative Solution Improvers for Constraint Satisfaction" - Xu
//! et al. ICML 2025

use std::collections::BTreeMap;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::constraints::{AllDifferent, Constraint, NotEquals};
use crate::learning::{Batch, Loss};

use super::dataset::ConsFormerBatch;
use super::utils::{blend_with_current, gumbel_softmax};

/// The differentiable penalty ConsFormer uses for a constraint. Constraints with the same penalty
/// and the same scope size are scored together, as one tensor operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Penalty {
    /// "No two variables share a value": penalizes any pair of variables in the scope putting
    /// nonzero probability on the same value. Used for `NotEquals`.
    Pairwise,
    /// Permutation constraints (e.g., all-diff with exactly the same number of domain values as
    /// variables, such as every Sudoku row/column/box): every value must receive a total
    /// probability of one across the scope.
    Permutation,
}

impl Penalty {
    /// The penalty for `constraint` given the number of values the network predicts, or `None` if
    /// ConsFormer has no relaxation for this type of constraint.
    fn of(constraint: &dyn Constraint, domain_size: usize) -> Option<Self> {
        let any = constraint.as_any();
        if any.is::<NotEquals>() {
            Some(Self::Pairwise)
        } else if any.is::<AllDifferent>() {
            // When the scope covers the domain every value must appear (e.g. Sudoku); with a
            // smaller scope only "no duplicates" is meaningful.
            if constraint.iter_scope().count() == domain_size {
                Some(Self::Permutation)
            } else {
                Some(Self::Pairwise)
            }
        } else {
            None
        }
    }

    /// The penalties, summed over instances, of constraints with `probs` the probabilities of
    /// their scopes, of shape `(instances, scope_len, domain_size)`.
    fn batched<B: Backend>(self, probs: Tensor<B, 3>) -> Tensor<B, 1> {
        match self {
            Self::Pairwise => {
                let collisions = probs.clone().matmul(probs.transpose());
                collisions.triu(1).sum().reshape([1])
            }
            Self::Permutation => {
                let [instances, _, domain_size] = probs.dims();
                // Per instance, how much probability mass each value received across the scope.
                let coverage: Tensor<B, 2> = probs.sum_dim(1).reshape([instances, domain_size]);
                let diff = coverage.sub_scalar(1.0);
                (diff.clone() * diff).mean().reshape([1])
            }
        }
    }
}

pub(crate) struct ConsFormerLoss;

impl<B: Backend> Loss<B, ConsFormerBatch<B>> for ConsFormerLoss {
    /// # Panics
    ///
    /// If a problem has a constraint other than `NotEquals` or `AllDifferent`.
    fn loss(&self, logits: Tensor<B, 3>, batch: &ConsFormerBatch<B>) -> Tensor<B, 1> {
        let probs = gumbel_softmax(logits);
        let probs = blend_with_current(probs, batch.assignments.clone(), batch.var_masks.clone());
        let problems = batch.problems();
        let batch_size = problems.len();
        let [_, number_vars, domain_size] = probs.dims();
        let device = probs.device();

        // Flatten (batch, vars, domain) -> (batch*vars, domain) so a single "global" variable
        // index (sample_offset + local_index) can gather any variable of any sample in one op.
        let flat_probs = probs.reshape([batch_size * number_vars, domain_size]);

        // Group the scopes of all the constraints of the batch by penalty and scope size, so that
        // each group is one tensor op instead of one op-chain per constraint. A `BTreeMap` fixes
        // the order the groups are summed in, so that the loss does not depend on a hash seed.
        let mut groups: BTreeMap<(Penalty, usize), Vec<i64>> = BTreeMap::new();
        for (i, problem) in problems.iter().enumerate() {
            let sample_offset = (i * number_vars) as i64;
            for constraint in problem.iter_constraints() {
                let c = &*problem[constraint];
                let Some(penalty) = Penalty::of(c, domain_size) else {
                    panic!(
                        "ConsFormer's loss has no penalty for constraint type `{}`: it supports \
                         `NotEquals` and `AllDifferent` (see `Penalty` in \
                         src/learning/consformer/loss.rs)",
                        c.name()
                    );
                };
                let scope: Vec<i64> = c.iter_scope().map(|v| sample_offset + v.0 as i64).collect();
                groups
                    .entry((penalty, scope.len()))
                    .or_default()
                    .extend(scope);
            }
        }

        let mut total = Tensor::<B, 1>::zeros([1], &device);
        for ((penalty, scope_len), flat_indices) in groups {
            let instances = flat_indices.len() / scope_len;
            let idx = Tensor::<B, 1, Int>::from_data(flat_indices.as_slice(), &device);
            let scopes: Tensor<B, 3> =
                flat_probs
                    .clone()
                    .select(0, idx)
                    .reshape([instances, scope_len, domain_size]);
            total = total + penalty.batched(scopes);
        }

        total.div_scalar(batch_size as f32)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use burn::backend::Autodiff;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use burn::data::dataloader::batcher::Batcher;
    use burn::data::dataset::Dataset;
    use burn::tensor::Distribution;

    use crate::constraints::{all_different, not_equals, sum};
    use crate::modelling::Problem;

    use super::super::dataset::{ConsFormerBatcher, ConsFormerDataset};
    use super::*;

    type Plain = NdArray;
    type Ad = Autodiff<NdArray>;

    fn scalar(t: Tensor<Plain, 1>) -> f32 {
        t.into_data().to_vec::<f32>().unwrap()[0]
    }

    /// One instance of `rows` variables, given row by row.
    fn instance(rows: &[&[f32]]) -> Tensor<Plain, 3> {
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Tensor::<Plain, 1>::from_data(flat.as_slice(), &NdArrayDevice::default()).reshape([
            1,
            rows.len(),
            rows[0].len(),
        ])
    }

    #[test]
    fn pairwise_penalty_counts_the_probability_of_sharing_a_value() {
        let same = instance(&[&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]]);
        assert!((scalar(Penalty::Pairwise.batched(same)) - 1.0).abs() < 1e-6);

        let different = instance(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]]);
        assert!(scalar(Penalty::Pairwise.batched(different)).abs() < 1e-6);

        // 0.5 * 0.5 + 0.5 * 0.5 on the two values both variables may take.
        let uncertain = instance(&[&[0.5, 0.5, 0.0], &[0.5, 0.5, 0.0]]);
        assert!((scalar(Penalty::Pairwise.batched(uncertain)) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn permutation_penalty_is_zero_exactly_when_every_value_is_covered_once() {
        let permutation = instance(&[&[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0], &[1.0, 0.0, 0.0]]);
        assert!(scalar(Penalty::Permutation.batched(permutation)).abs() < 1e-6);

        // Coverage (3, 0, 0): squared errors 4 + 1 + 1, averaged over the 3 values.
        let repeated = instance(&[&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]]);
        assert!((scalar(Penalty::Permutation.batched(repeated)) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn the_penalty_depends_on_the_constraint_and_on_the_scope_size() {
        let mut problem = Problem::default();
        let vars = problem.add_variables(4, vec![0, 1, 2], None);
        not_equals(&mut problem, vars[0], vars[1]);
        all_different(&mut problem, vars[..3].to_vec());
        all_different(&mut problem, vars.clone());
        sum(&mut problem, vars.clone(), 3);

        let penalties: Vec<Option<Penalty>> = problem
            .iter_constraints()
            .map(|c| Penalty::of(&*problem[c], 3))
            .collect();
        assert_eq!(
            penalties,
            [
                Some(Penalty::Pairwise),
                Some(Penalty::Permutation),
                Some(Penalty::Pairwise),
                None
            ]
        );
    }

    fn batch_of<B: Backend>(problems: Vec<Arc<Problem>>, device: &B::Device) -> ConsFormerBatch<B> {
        let dataset = ConsFormerDataset::<B>::new(problems, device);
        let samples: Vec<_> = (0..dataset.len())
            .map(|i| dataset.get(i).unwrap())
            .collect();
        ConsFormerBatcher {
            mask_fraction: 1.0,
            destroy: None,
        }
        .batch(samples, device)
    }

    fn colouring_problem() -> Arc<Problem> {
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        not_equals(&mut problem, vars[0], vars[1]);
        not_equals(&mut problem, vars[1], vars[2]);
        all_different(&mut problem, vars);
        Arc::new(problem)
    }

    #[test]
    fn the_loss_is_finite_and_reaches_the_logits() {
        let device = NdArrayDevice::default();
        let batch = batch_of::<Ad>(vec![colouring_problem(), colouring_problem()], &device);
        let logits: Tensor<Ad, 3> =
            Tensor::random([2, 3, 3], Distribution::Uniform(-1.0, 1.0), &device).require_grad();

        let loss = ConsFormerLoss.loss(logits.clone(), &batch);
        let value = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
        assert!(value.is_finite() && value > 0.0, "loss = {value}");

        let grads = loss.backward();
        let grad: Vec<f32> = logits
            .grad(&grads)
            .expect("logits should have a gradient")
            .into_data()
            .to_vec()
            .unwrap();
        assert!(grad.iter().all(|g| g.is_finite()));
        assert!(grad.iter().any(|&g| g != 0.0));
    }

    #[test]
    #[should_panic(expected = "no penalty for constraint type")]
    fn a_constraint_without_a_penalty_is_refused() {
        let device = NdArrayDevice::default();
        let mut problem = Problem::default();
        let vars = problem.add_variables(3, vec![0, 1, 2], None);
        sum(&mut problem, vars, 3);
        let batch = batch_of::<Plain>(vec![Arc::new(problem)], &device);
        let logits: Tensor<Plain, 3> = Tensor::zeros([1, 3, 3], &device);
        ConsFormerLoss.loss(logits, &batch);
    }
}
