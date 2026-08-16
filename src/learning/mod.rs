pub mod consformer;
pub mod monitoring;
pub mod train;

use std::sync::Arc;

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};

use crate::modelling::Problem;

/// Minimal capability every training/validation batch must provide: which
/// problem instance each row of the batch came from. Kept separate from
/// `Batch` so that a batch used only to compute a training loss
pub trait BatchProblems<B: Backend> {
    fn problems(&self) -> &[Arc<Problem>];
}

/// Batch type usable at inference time: on top of `BatchProblems`, it can be
/// rebuilt directly from a set of (possibly partial) assignments, which is
/// what local search / inference needs.
pub trait Batch<B: Backend>:
    BatchProblems<B> + Clone + Send + Sync + std::fmt::Debug + 'static
{
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
/// Generic over the batch type `Ba` rather than tying it to a single
/// associated type, so the same network (e.g. `ConsFormer`) can be driven by
/// several different batch structs
pub trait Network<B: Backend, Ba>: Module<B> {
    /// Evaluate the network. It receive a batch, which can be of arbitrary size during training,
    /// or a singleton at inference, and return logits over the variable domains (which can then be
    /// transformed into probability distributions). We leave the choice of the transformation up
    /// to the use case (e.g., log-softmax or softmax).
    fn forward(&self, batch: &Ba) -> Tensor<B, 3>;
}

/// Trait that must be implemented for each network configuration. A network configuration can be
/// seen as a list of its hyper-parameters
pub trait NetworkConfig<B: Backend> {
    type N: Module<B>;

    /// Initialise the network with the given hyper-parameters.
    fn init(&self, problems: &[Arc<Problem>], device: &B::Device) -> Self::N;
}

/// Generic loss function, over the batch type `Ba`. It receives the logits of the network as
/// input and returns the loss of each sample in the batch.
pub trait Loss<B: Backend, Ba> {
    fn loss(&self, logits: Tensor<B, 3>, batch: &Ba) -> Tensor<B, 1>;
}
