pub mod consformer;
pub mod monitoring;
pub mod train;

use std::sync::Arc;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use crate::modelling::Problem;

/// Batch type each network operates on. Covers both what's needed at
/// training time and at inference time. One trait, implemented once per
/// architecture.
pub trait Batch<B: Backend>: Clone + Send + Sync + std::fmt::Debug + 'static {
    fn problems(&self) -> &[Arc<Problem>];

    /// Builds a batch for `problems.len()` problems, each solved with
    /// `population_size` parallel candidate assignments (`population_size ==
    /// 1` is the plain single-candidate case). All problems must share the
    /// same `number_variables()`.
    ///
    /// `assignments` and `destroy_mask` have shape `(problems.len() *
    /// population_size, number_vars)`, with rows grouped by problem in
    /// order: problem 0's `population_size` rows first, then problem 1's,
    /// etc. `destroy_mask` flags which variables are being modified this
    /// iteration.
    fn for_assignments(
        problems: &[Arc<Problem>],
        population_size: usize,
        assignments: Tensor<B, 2, Int>,
        destroy_mask: Tensor<B, 2, Bool>,
        device: &B::Device,
    ) -> Self;
}

/// Trait that each network, independent of its architecture, must implement.
pub trait Network<B: Backend>: Module<B> {
    type Batch: Batch<B>;

    /// Evaluate the network. It receive a batch, which can be of arbitrary size during training,
    /// or a singleton at inference, and return logits over the variable domains (which can then be
    /// transformed into probability distributions). We leave the choice of the transformation up
    /// to the use case (e.g., log-softmax or softmax).
    fn forward(&self, batch: &Self::Batch) -> Tensor<B, 3>;
}

/// Trait that must be implemented for each network configuration. A network configuration can be
/// seen as a list of its hyper-parameters
pub trait NetworkConfig<B: Backend> {
    type N: Network<B>;

    /// Initialise the network with the given hyper-parameters.
    fn init(&self, device: &B::Device) -> Self::N;
}

/// Generic loss function. It receives the logits of the network as input and returns the loss of
/// each sample in the batch.
pub trait Loss<B: Backend, N: Network<B>> {
    fn loss(&self, logits: Tensor<B, 3>, batch: &N::Batch) -> Tensor<B, 1>;
}
