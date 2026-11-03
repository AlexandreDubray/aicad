//! The neural networks of the crate, and what both training and local search need to use them.
//!
//! # Meaning
//!
//! A network reads a [`Batch`] of problems with their current assignments and returns one logit
//! per value of every variable ([`Network`]). It is built from a [`NetworkConfig`] and the
//! problems it will read, and [`load_network`] brings a trained one back from a checkpoint.
//! [`consformer`] is the architecture the crate provides. `learning` trains these networks and
//! `local_search` runs them; neither depends on the other.
//!
//! # Design notes
//!
//! - **What a batch can hold is the architecture's business.** [`Batch::check_problems`] states
//!   the requirement (ConsFormer needs the same variables in every problem) next to the code that
//!   relies on it, and every caller asks the batch type instead of assuming it.

#![deny(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]

pub mod consformer;

use std::path::Path;

use std::sync::Arc;

use burn::config::Config;
use burn::module::Module;
use burn::record::CompactRecorder;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Bool, Int, Tensor};

use crate::modelling::Problem;

/// Whether a CUDA device can be used. The Cuda backend is built in, but a machine without a GPU
/// would panic when touching it, so a tensor is created and read: lazy initialisation would
/// otherwise hide the failure until later. The panic is caught, and its message silenced.
pub fn cuda_available() -> bool {
    use burn::backend::cuda::{Cuda, CudaDevice};

    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(|| {
        let device = CudaDevice::default();
        let tensor = burn::tensor::Tensor::<Cuda, 1>::from_data([0.0f32], &device);
        let _ = tensor.into_data();
    });
    std::panic::set_hook(prev_hook);
    result.is_ok()
}

/// Seeds the random generators of the backends: the `NdArray` one always, the `Cuda` one only if
/// [`cuda_available`] (touching an absent GPU would panic). Both are seeded, rather than the one
/// a later call will use, because the caller cannot know which that will be.
///
/// Use [`crate::set_seed`] to seed the whole crate at once.
pub fn seed_backends(seed: u64) {
    use burn::backend::cuda::{Cuda, CudaDevice};
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    NdArray::<f32>::seed(&NdArrayDevice::default(), seed);
    if cuda_available() {
        Cuda::<f32>::seed(&CudaDevice::default(), seed);
    }
}

/// Calls `$call::<Backend>(device, args...)` with the backend of the best device found, chosen at
/// run time: `Cuda` if [`cuda_available`], `NdArray` otherwise. The `training` arm wraps it in
/// `Autodiff`, as training needs; the `inference` arm does not.
macro_rules! dispatch_backend {
    (training, $call:ident, $($args:expr),* $(,)?) => {
        if $crate::neural::cuda_available() {
            $call::<burn::backend::Autodiff<burn::backend::cuda::Cuda>>(
                burn::backend::cuda::CudaDevice::default(), $($args),*)
        } else {
            $call::<burn::backend::Autodiff<burn::backend::ndarray::NdArray>>(
                burn::backend::ndarray::NdArrayDevice::default(), $($args),*)
        }
    };
    (inference, $call:ident, $($args:expr),* $(,)?) => {
        if $crate::neural::cuda_available() {
            $call::<burn::backend::cuda::Cuda>(burn::backend::cuda::CudaDevice::default(), $($args),*)
        } else {
            $call::<burn::backend::ndarray::NdArray>(
                burn::backend::ndarray::NdArrayDevice::default(), $($args),*)
        }
    };
}
pub(crate) use dispatch_backend;

/// Work to run on the best inference backend (see [`run_inference`]), written once for every
/// backend. A generic method cannot be given as a closure, hence a trait.
pub trait InferenceJob {
    /// What the job returns.
    type Output;

    /// Runs the job on the backend `B`, on `device`.
    fn run<B: Backend>(self, device: B::Device) -> Self::Output;
}

/// Work to run on the best training backend (see [`run_training`]): the backend is wrapped in
/// `Autodiff`.
pub trait TrainingJob {
    /// What the job returns.
    type Output;

    /// Runs the job on the backend `B`, on `device`.
    fn run<B: AutodiffBackend>(self, device: B::Device) -> Self::Output;
}

/// Runs `job` on the best device found at run time: `Cuda` if [`cuda_available`], `NdArray`
/// otherwise. A network defined outside the crate is used through this, as the crate's own entry
/// points do, without choosing a backend by hand.
pub fn run_inference<J: InferenceJob>(job: J) -> J::Output {
    use burn::backend::cuda::{Cuda, CudaDevice};
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    if cuda_available() {
        job.run::<Cuda>(CudaDevice::default())
    } else {
        job.run::<NdArray>(NdArrayDevice::default())
    }
}

/// Like [`run_inference`], for work that trains: the backend is wrapped in `Autodiff`.
pub fn run_training<J: TrainingJob>(job: J) -> J::Output {
    use burn::backend::Autodiff;
    use burn::backend::cuda::{Cuda, CudaDevice};
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    if cuda_available() {
        job.run::<Autodiff<Cuda>>(CudaDevice::default())
    } else {
        job.run::<Autodiff<NdArray>>(NdArrayDevice::default())
    }
}

/// A group of problems prepared for one forward pass of a network: what the network reads, what
/// the loss scores and what the training monitor decodes.
///
/// Every row is one problem. What problems a batch can hold together (for instance whether they
/// need the same number of variables) is a requirement of each architecture, checked by its own
/// batch type.
pub trait Batch<B: Backend>: Clone + Send + Sync + std::fmt::Debug + 'static {
    /// The problems of the batch, one per row.
    fn problems(&self) -> &[Arc<Problem>];

    /// The current value of every variable, of shape `(batch_size, number_vars)`: one row per
    /// problem.
    fn assignments(&self) -> Tensor<B, 2, Int>;

    /// The variables the network must re-predict, of shape `(batch_size, number_vars)`. Every
    /// other variable keeps its value in [`Batch::assignments`].
    fn var_masks(&self) -> Tensor<B, 2, Bool>;

    /// The assignments the network proposes for `logits` (shape `(batch_size, number_vars,
    /// domain_size)`), one row per problem with one value per variable of that problem. Only the
    /// variables of [`Batch::var_masks`] take the network's choice; every other keeps its value in
    /// [`Batch::assignments`], as it does in local search.
    fn decode(&self, logits: Tensor<B, 3>) -> Vec<Vec<isize>>;

    /// Whether `problems` can be handled together by this kind of batch: the requirement of the
    /// architecture on the problems it reads (for instance that they all have the same variables).
    /// Returns a message explaining what is wrong when they cannot.
    fn check_problems(problems: &[Arc<Problem>]) -> Result<(), String>;

    /// Builds the batch for `assignments` (shape `(problems.len(), number_vars)`) at inference
    /// time, where `destroy_mask` holds 1 for the variables to re-predict and 0 for the others.
    fn for_assignments(
        problems: &[Arc<Problem>],
        assignments: Tensor<B, 2, Int>,
        destroy_mask: Tensor<B, 2, Int>,
        device: &B::Device,
    ) -> Self;
}

/// A neural network that reads a [`Batch`] of type `Ba` and returns its logits.
pub trait Network<B: Backend, Ba: Batch<B>>: Module<B> {
    /// The logits for `batch`, of shape `(batch_size, number_vars, domain_size)`: one per value of
    /// every variable.
    fn forward(&self, batch: &Ba) -> Tensor<B, 3>;
}

/// The hyperparameters of a network: what it takes to build one for a set of problems.
pub trait NetworkConfig<B: Backend> {
    /// The network that the configuration builds.
    type N: Module<B>;

    /// Builds a network with randomly initialised weights. Whatever depends on the problems (for
    /// instance the positions of their variables) is read from `problems`; all of them must be
    /// accepted by the batch type of the network (see [`Batch::check_problems`]).
    fn init(&self, problems: &[Arc<Problem>], device: &B::Device) -> Self::N;
}

/// Loads a network's hyperparameters (JSON config) and trained weights from a checkpoint
/// directory produced by `train_model`/`run_training`. Returns the config alongside the network
/// so callers also have access to training hyperparameters.
pub fn load_network<B, NC>(
    checkpoint_dir: &Path,
    problems: &[Arc<Problem>],
    device: &B::Device,
) -> Result<(NC, NC::N), Box<dyn std::error::Error>>
where
    B: Backend,
    NC: NetworkConfig<B> + Config + Clone,
    NC::N: Module<B>,
{
    // First, loads the config from the checkpoint
    let config: NC = NC::load(checkpoint_dir.join("config.json"))?;
    // Inits the network from the config and loads the weights.
    // The network initialisation is responsible for initiating problem-specific network's
    // component. For example, in ConsFormer, the variables positions are used in the positional
    // embedding. See each architecture's doc for further comment.
    let network = config.clone().init(problems, device).load_file(
        checkpoint_dir.join("weights"),
        &CompactRecorder::new(),
        device,
    )?;
    Ok((config, network))
}
