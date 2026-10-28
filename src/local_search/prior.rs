//! Where the per-position prior over values comes from at each local-search step. The decode
//! operator turns whatever this returns into the next assignment, so swapping the prior changes
//! what the search is guided by without touching decoding.

use std::marker::PhantomData;
use std::sync::Arc;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::learning::{Batch, Network};
use crate::modelling::Problem;

/// Source of the per-value priors that guide each step of the search.
pub(crate) trait PriorHeuristic<B: Backend>: Send + Sync {
    /// Priors of shape `[rows, variables, domain_size]`, one row per problem in `problems`: unnormalised
    /// scores per value (a network's logits, for instance), a higher score being a more likely value.
    fn priors(
        &self,
        problems: &[Arc<Problem>],
        assignments: &Tensor<B, 2, Int>,
        destroy_mask: &Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> Tensor<B, 3>;
}

/// Priors given by a trained network: the batch the network expects is built from the current
/// assignments and destroy mask, and its output is the priors.
pub(crate) struct NetworkPrior<N, Ba> {
    /// The network evaluated at each step.
    network: N,
    /// Marks the batch type `Ba` the network consumes without storing one.
    _batch: PhantomData<fn() -> Ba>,
}

impl<N, Ba> NetworkPrior<N, Ba> {
    /// Wraps a loaded network.
    pub(crate) fn new(network: N) -> Self {
        Self {
            network,
            _batch: PhantomData,
        }
    }
}

impl<B, N, Ba> PriorHeuristic<B> for NetworkPrior<N, Ba>
where
    B: Backend,
    Ba: Batch<B>,
    N: Network<B, Ba> + Send + Sync,
{
    fn priors(
        &self,
        problems: &[Arc<Problem>],
        assignments: &Tensor<B, 2, Int>,
        destroy_mask: &Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let batch =
            Ba::for_assignments(problems, assignments.clone(), destroy_mask.clone(), device);
        self.network.forward(&batch)
    }
}

/// Equal priors everywhere, so every in-domain value starts equally likely. No network is
/// evaluated and no batch is built.
pub(crate) struct UniformPrior {
    /// Number of values per variable in the priors produced.
    domain_size: usize,
}

impl UniformPrior {
    /// Uniform priors over `domain_size` values per variable.
    pub(crate) fn new(domain_size: usize) -> Self {
        Self { domain_size }
    }

    /// Smallest alphabet that covers every value any variable of any problem can take.
    pub(crate) fn domain_size_covering(problems: &[Arc<Problem>]) -> usize {
        problems
            .iter()
            .flat_map(|problem| {
                problem
                    .iter_variables()
                    .flat_map(|v| problem[v].iter_domain().collect::<Vec<_>>())
            })
            .map(|value| value as usize + 1)
            .max()
            .unwrap_or(1)
    }
}

impl<B: Backend> PriorHeuristic<B> for UniformPrior {
    fn priors(
        &self,
        problems: &[Arc<Problem>],
        assignments: &Tensor<B, 2, Int>,
        _destroy_mask: &Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let n = assignments.dims()[1];
        Tensor::zeros([problems.len(), n, self.domain_size], device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::not_equals;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    fn problem_over(domain: Vec<isize>) -> Arc<Problem> {
        let mut problem = Problem::default();
        let x = problem.add_variable(domain.clone(), None);
        let y = problem.add_variable(domain, None);
        not_equals(&mut problem, x, y);
        Arc::new(problem)
    }

    #[test]
    fn uniform_prior_has_one_equal_row_per_problem() {
        let device = NdArrayDevice::default();
        let problems = vec![problem_over(vec![0, 1, 2]), problem_over(vec![0, 1, 2])];
        let assignments = Tensor::<NdArray, 2, Int>::zeros([2, 2], &device);
        let mask = Tensor::<NdArray, 2, Int>::ones([2, 2], &device);
        let priors = PriorHeuristic::<NdArray>::priors(
            &UniformPrior::new(3),
            &problems,
            &assignments,
            &mask,
            &device,
        );
        assert_eq!(priors.dims(), [2, 2, 3]);
        let values: Vec<f32> = priors.into_data().to_vec().unwrap();
        assert!(values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn covering_domain_size_is_the_largest_value_plus_one() {
        let problems = vec![problem_over(vec![0, 1]), problem_over(vec![0, 1, 4])];
        assert_eq!(UniformPrior::domain_size_covering(&problems), 5);
    }
}
