pub(crate) mod consformer;
pub(crate) mod monitoring;
pub(crate) mod train;

use std::sync::Arc;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use crate::modelling::Problem;

/// A group of problems prepared for one forward pass of a network: what the network reads, what
/// the loss scores and what the training monitor decodes.
///
/// Every row is one problem, and every problem of a batch has the same number of variables.
pub(crate) trait Batch<B: Backend>: Clone + Send + Sync + std::fmt::Debug + 'static {
    /// The problems of the batch, one per row.
    fn problems(&self) -> &[Arc<Problem>];

    /// The current value of every variable, of shape `(batch_size, number_vars)`.
    fn assignments(&self) -> Tensor<B, 2, Int>;

    /// The variables the network must re-predict, of shape `(batch_size, number_vars)`. Every
    /// other variable keeps its value in [`Batch::assignments`].
    fn var_masks(&self) -> Tensor<B, 2, Bool>;

    /// Builds the batch for `assignments` (shape `(problems.len(), number_vars)`) at inference
    /// time, where `destroy_mask` holds 1 for the variables to re-predict and 0 for the others.
    fn for_assignments(
        problems: &[Arc<Problem>],
        assignments: Tensor<B, 2, Int>,
        destroy_mask: Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> Self;
}

pub(crate) trait Network<B: Backend, Ba: Batch<B>>: Module<B> {
    fn forward(&self, batch: &Ba) -> Tensor<B, 3>;
}

pub(crate) trait NetworkConfig<B: Backend> {
    type N: Module<B>;

    fn init(&self, problems: &[Arc<Problem>], device: &B::Device) -> Self::N;
}

pub(crate) trait Loss<B: Backend, Ba: Batch<B>> {
    /// Prepares whatever the loss needs about `problems`, once, before training starts. It is
    /// given every problem the loss will score, training and validation alike. Does nothing by
    /// default.
    fn prepare(&mut self, _problems: &[Arc<Problem>]) {}

    fn loss(&self, logits: Tensor<B, 3>, batch: &Ba) -> Tensor<B, 1>;
}
