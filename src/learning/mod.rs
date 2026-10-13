pub(crate) mod consformer;
pub(crate) mod monitoring;
pub(crate) mod train;

use std::sync::Arc;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::modelling::Problem;

pub(crate) trait BatchProblems<B: Backend> {
    fn problems(&self) -> &[Arc<Problem>];
}

pub(crate) trait Batch<B: Backend>:
    BatchProblems<B> + Clone + Send + Sync + std::fmt::Debug + 'static
{
    fn for_assignments(
        problems: &[Arc<Problem>],
        assignments: Tensor<B, 2, Int>,
        destroy_mask: Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> Self;
}

pub(crate) trait Network<B: Backend, Ba>: Module<B> {
    fn forward(&self, batch: &Ba) -> Tensor<B, 3>;
}

pub(crate) trait NetworkConfig<B: Backend> {
    type N: Module<B>;

    fn init(&self, problems: &[Arc<Problem>], device: &B::Device) -> Self::N;
}

pub(crate) trait Loss<B: Backend, Ba> {
    fn loss(&self, logits: Tensor<B, 3>, batch: &Ba) -> Tensor<B, 1>;
}
