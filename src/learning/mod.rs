pub mod consformer;
pub mod monitoring;
pub mod train;

use std::sync::Arc;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::modelling::Problem;

/// Trait that must be implemented by the network's batch to get access to the underlying problems
pub trait HasProblems<B: Backend> {
    fn problems(&self) -> &[Arc<Problem>];
}

/// Trait that each network, independent of its architecture, must implement.
pub trait Network<B: Backend>: Module<B> {
    type Batch: HasProblems<B> + Clone + Send + Sync + std::fmt::Debug + 'static;

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
