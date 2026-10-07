//! Where the per-position prior over values comes from at each local-search step. The decode
//! operator turns whatever this returns into the next assignment, so swapping the prior changes
//! what the search is guided by without touching decoding.

use std::marker::PhantomData;
use std::sync::Arc;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::learning::{Batch, Network};
use crate::modelling::Problem;

pub trait PriorHeuristic<B: Backend>: Send + Sync {
    /// Logits of shape `[rows, variables, domain_size]`, one row per problem in `problems`.
    fn priors(
        &self,
        problems: &[Arc<Problem>],
        assignments: &Tensor<B, 2, Int>,
        destroy_mask: &Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> Tensor<B, 3>;
}

pub struct NetworkPrior<N, Ba> {
    network: N,
    _batch: PhantomData<fn() -> Ba>,
}

impl<N, Ba> NetworkPrior<N, Ba> {
    pub fn new(network: N) -> Self {
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

/// Equal logits everywhere, so every in-domain value starts equally likely. No network is
/// evaluated and no batch is built.
pub struct UniformPrior {
    domain_size: usize,
}

impl UniformPrior {
    pub fn new(domain_size: usize) -> Self {
        Self { domain_size }
    }

    /// Smallest alphabet that covers every value any variable of any problem can take.
    pub fn domain_size_covering(problems: &[Arc<Problem>]) -> usize {
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
    use crate::modelling::not_equals;
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
        let logits = PriorHeuristic::<NdArray>::priors(
            &UniformPrior::new(3),
            &problems,
            &assignments,
            &mask,
            &device,
        );
        assert_eq!(logits.dims(), [2, 2, 3]);
        let values: Vec<f32> = logits.into_data().to_vec().unwrap();
        assert!(values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn covering_domain_size_is_the_largest_value_plus_one() {
        let problems = vec![problem_over(vec![0, 1]), problem_over(vec![0, 1, 4])];
        assert_eq!(UniformPrior::domain_size_covering(&problems), 5);
    }
}
