use std::path::{Path, PathBuf};
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::config::Config;
use burn::tensor::backend::{AutodiffBackend, Backend};

use rand::seq::SliceRandom;

use crate::learning::consformer::architecture::MaskingKind;
use crate::learning::consformer::{
    ConsFormerBatch, ConsFormerBatcher, ConsFormerConfig, ConsFormerDataConfig, ConsFormerDataset,
    ConsFormerLoss, ConsFormerMddBatch, ConsFormerMddBatcher, ConsFormerMddDataset,
    ConsFormerMddLoss, ConsFormerMddSample, ConsFormerSample, MddCompilationConfig,
};
use crate::learning::train::{train_model, ModelSelection, TrainingConfig};
use crate::modelling::Problem;
use crate::utils::with_rng;

use super::heuristics::{PyMergeHeuristic, PyOrderingHeuristic, PySelectHeuristic};
use super::problem::PyProblem;

#[pyclass]
pub struct PyConsFormerConfig {
    pub domain_size: usize,
    pub embedding_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub expand_size: usize,
    pub num_layers: usize,
    pub drop_out: f64,
    pub bias: bool,
    pub positional_encoding_dimensions: usize,
    pub mask_fraction: f64,
    pub tau: f64,
    /// If true, the multi-head attention blocks replace hard -inf masking of non-primal-graph
    /// pairs with a learned per-head bias (see `architecture::MaskingKind::Learned`). Useful
    /// for problems whose primal graph has large cliques (e.g. nurse rostering's per-day
    /// coverage constraints) where hard masking still leaves each row attending near-uniformly
    /// over many neighbours.
    pub learn_attention_mask: bool,
}

#[pymethods]
impl PyConsFormerConfig {
    #[new]
    #[pyo3(signature = (domain_size=1,
            embedding_size=128,
            hidden_size=128,
            num_heads=1,
            expand_size=128,
            mask_fraction=0.5,
            tau=0.1,
            num_layers=1,
            drop_out=0.0,
            bias=true,
            positional_encoding_dimensions=0,
            learn_attention_mask=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        domain_size: usize,
        embedding_size: usize,
        hidden_size: usize,
        num_heads: usize,
        expand_size: usize,
        mask_fraction: f64,
        tau: f64,
        num_layers: usize,
        drop_out: f64,
        bias: bool,
        positional_encoding_dimensions: usize,
        learn_attention_mask: bool,
    ) -> Self {
        PyConsFormerConfig {
            domain_size,
            embedding_size,
            hidden_size,
            num_heads,
            expand_size,
            num_layers,
            drop_out,
            bias,
            positional_encoding_dimensions,
            mask_fraction,
            tau,
            learn_attention_mask,
        }
    }
}

impl From<&PyConsFormerConfig> for ConsFormerConfig {
    fn from(c: &PyConsFormerConfig) -> Self {
        ConsFormerConfig {
            domain_size: c.domain_size,
            embedding_size: c.embedding_size,
            hidden_size: c.hidden_size,
            num_heads: c.num_heads,
            expand_size: c.expand_size,
            num_layers: c.num_layers,
            drop_out: c.drop_out,
            bias: c.bias,
            positional_encoding_dimensions: c.positional_encoding_dimensions,
            mask_fraction: c.mask_fraction,
            tau: c.tau,
            masking_kind: if c.learn_attention_mask {
                MaskingKind::Learned
            } else {
                MaskingKind::Hard
            },
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyModelSelection {
    Loss,
    ConstraintSatisfaction,
}

impl From<&PyModelSelection> for ModelSelection {
    fn from(c: &PyModelSelection) -> Self {
        match c {
            PyModelSelection::Loss => Self::Loss,
            PyModelSelection::ConstraintSatisfaction => Self::ConstraintSatisfaction,
        }
    }
}

#[pyclass]
pub struct PyTrainingConfig {
    pub lr: f64,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub validation_interval: usize,
    pub model_selection: PyModelSelection,
    /// Adam's first moment (gradient mean) decay rate.
    pub beta_1: f64,
    /// Adam's second moment (gradient variance) decay rate.
    pub beta_2: f64,
    /// Numerical-stability floor added to Adam's denominator.
    pub epsilon: f64,
    /// Whether to use the AMSGrad variant of Adam.
    pub amsgrad: bool,
    /// L2 weight decay penalty. `None` disables it.
    pub weight_decay: Option<f64>,
    /// Clip each gradient tensor's global L2 norm to this value, if set. Takes precedence over
    /// `grad_clip_value` when both are set.
    pub grad_clip_norm: Option<f64>,
    /// Clip each gradient element to `[-grad_clip_value, grad_clip_value]`, if set. Ignored when
    /// `grad_clip_norm` is also set.
    pub grad_clip_value: Option<f64>,
    /// If true, also save the best-so-far model at a decaying-density set of training-epoch
    /// horizons under `checkpoint_dir/horizons/`, so performance can later be swept as a function
    /// of training budget. See `train::compute_horizons`.
    pub save_horizons: bool,
    /// How many horizon checkpoints to save across `[0, num_epochs]` when `save_horizons` is set.
    pub num_checkpoints: usize,
}

#[pymethods]
impl PyTrainingConfig {
    #[new]
    #[pyo3(signature = (lr=3e-4, num_epochs=10, batch_size=32, validation_interval=10,
            model_selection=PyModelSelection::Loss, beta_1=0.9, beta_2=0.999, epsilon=1e-5,
            amsgrad=false, weight_decay=None, grad_clip_norm=None, grad_clip_value=None,
            save_horizons=false, num_checkpoints=15))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lr: f64,
        num_epochs: usize,
        batch_size: usize,
        validation_interval: usize,
        model_selection: PyModelSelection,
        beta_1: f64,
        beta_2: f64,
        epsilon: f64,
        amsgrad: bool,
        weight_decay: Option<f64>,
        grad_clip_norm: Option<f64>,
        grad_clip_value: Option<f64>,
        save_horizons: bool,
        num_checkpoints: usize,
    ) -> Self {
        PyTrainingConfig {
            lr,
            num_epochs,
            batch_size,
            validation_interval,
            model_selection,
            beta_1,
            beta_2,
            epsilon,
            amsgrad,
            weight_decay,
            grad_clip_norm,
            grad_clip_value,
            save_horizons,
            num_checkpoints,
        }
    }
}

impl From<&PyTrainingConfig> for TrainingConfig {
    fn from(c: &PyTrainingConfig) -> Self {
        TrainingConfig {
            lr: c.lr,
            num_epochs: c.num_epochs,
            batch_size: c.batch_size,
            validation_interval: c.validation_interval,
            model_selection: ModelSelection::from(&c.model_selection),
            beta_1: c.beta_1,
            beta_2: c.beta_2,
            epsilon: c.epsilon,
            amsgrad: c.amsgrad,
            weight_decay: c.weight_decay,
            grad_clip_norm: c.grad_clip_norm,
            grad_clip_value: c.grad_clip_value,
            save_horizons: c.save_horizons,
            num_checkpoints: c.num_checkpoints,
        }
    }
}

/// Runtime CUDA availability check. Burn's backend is a compile-time generic, so
/// "auto-detection" is done as follows: try to grab a CUDA device, and fall back to
/// the NdArray (CPU) backend if that fails.
///
/// `CudaDevice::default()` alone is NOT a valid check: `CudaDevice` is just a
/// `#[derive(Default)]` wrapper around a device index (`CudaDevice(0)`). To actually touch the
/// GPU, we need to create a tensor. To avoid lazy initialisation, we turn it into_data(). If no
/// cude device is available, this should panic; which we catch manually.
pub(super) fn cuda_available() -> bool {
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

/// Seeds every independently-seedable source of randomness this crate uses, so a subsequent
/// training/search run reproduces bit-for-bit given the same seed (weight init, `gumbel_softmax`
/// sampling, decode-time sampling, the train/validation shuffle, and per-batch variable masking --
/// see `utils::rng`'s doc for the one source this deliberately leaves out, and why).
///
/// Seeds *both* backends' RNGs unconditionally (`NdArray` always; `Cuda` only if
/// `cuda_available()`, since touching an absent GPU would itself panic) rather than just whichever
/// one the next call happens to dispatch to: `set_seed` has no way to know in advance which
/// backend a later `train_consformer`/`neural_local_search` call will pick, and seeding the one
/// that won't be used is harmless.
#[pyfunction]
pub fn set_seed(seed: u64) {
    crate::utils::rng::set_seed(seed);
    NdArray::<f32>::seed(&NdArrayDevice::default(), seed);
    if cuda_available() {
        Cuda::<f32>::seed(&CudaDevice::default(), seed);
    }
}

/// Picks `Autodiff<Cuda>` or `Autodiff<NdArray>` at runtime (via `cuda_available`) and calls
/// `$call` with the matching device already threaded in as the first argument, so every training
/// entry point doesn't have to hand-write the same `if cuda_available() { ... } else { ... }`
/// dispatch. Add one `train_consformer_<recipe>` here for every new recipe (nurse rostering, etc)
/// and it gets this for free.
macro_rules! dispatch_backend {
    ($call:ident, $($args:expr),* $(,)?) => {
        if cuda_available() {
            $call::<Autodiff<Cuda>>(CudaDevice::default(), $($args),*)
        } else {
            $call::<Autodiff<NdArray>>(NdArrayDevice::default(), $($args),*)
        }
    };
}

/// `(all, train, validation)`, as returned by `split_train_validation`.
type ProblemSplit = (Vec<Arc<Problem>>, Vec<Arc<Problem>>, Vec<Arc<Problem>>);

/// Splits `problems` into a `(all, train, validation)` triple: `all` is an unshuffled clone (the
/// full problem set that `train_model` wants for e.g. constraint-satisfaction model selection),
/// while `train`/`validation` are a random 80/20 split of a shuffled copy. Shared by every
/// `run_training_*` function so the split logic can't drift between recipes.
fn split_train_validation(mut problems: Vec<Arc<Problem>>) -> ProblemSplit {
    let all_problems = problems.clone();
    let training_size = (problems.len() as f64 * 0.8).round() as usize;
    with_rng(|rng| problems.shuffle(rng));
    let validation_problems = problems.split_off(training_size);
    (all_problems, problems, validation_problems)
}

/// Creates `checkpoint_dir` (if missing) and saves `network_config` as `config.json` inside it,
/// so it can be reloaded at inference time. Shared by every `run_training_*` function.
fn prepare_checkpoint_dir<C: Config>(checkpoint_dir: &Path, network_config: &C) -> PyResult<()> {
    std::fs::create_dir_all(checkpoint_dir)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create checkpoint dir: {e}")))?;
    network_config
        .save(checkpoint_dir.join("config.json"))
        .map_err(|e| PyRuntimeError::new_err(format!("failed to save network config: {e}")))
}

/// Trains consformer on a set of problems.
///
/// Releases the GIL for the whole training run (`py.detach`): everything this needs --
/// `problems`, `network_config`, `training_config`, `checkpoint_dir` -- is extracted into plain,
/// `Send` Rust values *before* the release, so nothing inside the closure touches Python. Without
/// this, the calling thread holds the GIL for the entire (potentially long) blocking call, which
/// silently breaks progressive logging: `log::info!` calls from inside the training loop (e.g.
/// the per-epoch loss in `train.rs`) go through `pyo3-log`, which has to re-acquire the GIL to
/// forward each record to Python -- but the GIL isn't free again until this function returns, so
/// every log line was queued up and only delivered in a burst at the very end.
#[pyfunction]
pub fn train_consformer(
    py: Python<'_>,
    problems: Vec<PyRef<PyProblem>>,
    config: &PyConsFormerConfig,
    training: &PyTrainingConfig,
    checkpoint_dir: String,
) -> PyResult<()> {
    let problems: Vec<Arc<Problem>> = problems.iter().map(|p| p.arc()).collect();
    let network_config: ConsFormerConfig = config.into();
    let training_config: TrainingConfig = training.into();
    let checkpoint_dir = PathBuf::from(checkpoint_dir);

    py.detach(move || {
        dispatch_backend!(
            run_training,
            problems,
            network_config,
            training_config,
            &checkpoint_dir,
        )
    })
}

/// runs the training loop for a set of training problems, a network/training loop configuration.
/// The network is saved to the checkpoint.
fn run_training<B: AutodiffBackend>(
    device: B::Device,
    problems: Vec<Arc<Problem>>,
    network_config: ConsFormerConfig,
    training_config: TrainingConfig,
    checkpoint_dir: &Path,
) -> PyResult<()> {
    let (all_problems, problems, validation_problems) = split_train_validation(problems);
    let train_dataset = ConsFormerDataset::<B>::new(problems, &device);
    let validation_dataset =
        ConsFormerDataset::<B::InnerBackend>::new(validation_problems, &device);
    let batcher = ConsFormerBatcher {
        mask_fraction: network_config.mask_fraction,
    };

    prepare_checkpoint_dir(checkpoint_dir, &network_config)?;

    // Create and train the network
    let _network = train_model::<
        B,
        ConsFormerConfig,
        ConsFormerSample<B>,
        ConsFormerBatch<B>,
        ConsFormerBatcher,
        ConsFormerLoss,
        ConsFormerSample<B::InnerBackend>,
        ConsFormerBatch<B::InnerBackend>,
    >(
        network_config,
        &all_problems,
        train_dataset,
        validation_dataset,
        batcher,
        ConsFormerLoss,
        training_config,
        checkpoint_dir,
        &device,
    );

    Ok(())
}

/// Trains ConsFormer against the exact per-constraint MDD weighted model count (see
/// `ConsFormerMddLoss`) instead of the classical hand-written per-constraint-type penalty --
/// otherwise identical to `train_consformer`: same network architecture, same config, same
/// checkpointing. No `epsilon`: the loss's gradient is computed by hand, walking each compiled
/// `Mdd` directly (`crate::mdd::wmc`) instead of through Burn's autodiff, with its own small fixed
/// floor (`loss::WMC_EPS`) on WMC -- a constraint whose fixed context is already structurally
/// unsatisfiable reads back as `-log(WMC_EPS)` rather than `-log(0) = inf`. `pyordering`/
/// `pymerge`/`pyselect` control how each constraint's MDD is compiled (`pymerge` is accepted for
/// parity with `Compiler::compile` but has no effect here -- this recipe always refines to an
/// exact, unbounded-width MDD, and `merge` only ever triggers once a width bound would be
/// exceeded; see `MddCompilationConfig`'s doc).
///
/// Releases the GIL for the whole training run -- see `train_consformer`'s doc for why.
#[pyfunction]
#[pyo3(signature = (problems, config, training, checkpoint_dir,
        pyordering=PyOrderingHeuristic::MinDomMaxLinked(),
        pymerge=PyMergeHeuristic::LessRelaxed,
        pyselect=PySelectHeuristic::Greedy))]
#[allow(clippy::too_many_arguments)]
pub fn train_consformer_mdd(
    py: Python<'_>,
    problems: Vec<PyRef<PyProblem>>,
    config: &PyConsFormerConfig,
    training: &PyTrainingConfig,
    checkpoint_dir: String,
    pyordering: PyOrderingHeuristic,
    pymerge: PyMergeHeuristic,
    pyselect: PySelectHeuristic,
) -> PyResult<()> {
    let problems: Vec<Arc<Problem>> = problems.iter().map(|p| p.arc()).collect();
    let network_config: ConsFormerConfig = config.into();
    let training_config: TrainingConfig = training.into();
    let checkpoint_dir = PathBuf::from(checkpoint_dir);
    let compilation = MddCompilationConfig {
        ordering: pyordering.into(),
        merge: pymerge.into(),
        select: pyselect.into(),
        grouping: crate::mdd::heuristics::ConstraintGrouping::default(),
        max_width: usize::MAX,
    };

    py.detach(move || {
        dispatch_backend!(
            run_training_mdd,
            problems,
            network_config,
            training_config,
            compilation,
            &checkpoint_dir,
        )
    })
}

/// Runs the ConsFormer-MDD training loop -- see `run_training` for the classical recipe this
/// mirrors. The only structural differences are the dataset/batcher/loss types (`ConsFormerMdd*`
/// instead of `ConsFormer*`) and the `compilation`/`data_config` the MDD dataset needs to compile
/// each problem's constraints into exact MDDs. `data_config` is derived from `network_config` via
/// `ConsFormerDataConfig::from` -- not built by hand -- so the dataset and its batcher can't end
/// up with different `domain_size`s (see `ConsFormerDataConfig`'s doc).
fn run_training_mdd<B: AutodiffBackend>(
    device: B::Device,
    problems: Vec<Arc<Problem>>,
    network_config: ConsFormerConfig,
    training_config: TrainingConfig,
    compilation: MddCompilationConfig,
    checkpoint_dir: &Path,
) -> PyResult<()> {
    let (all_problems, problems, validation_problems) = split_train_validation(problems);

    let data_config = ConsFormerDataConfig::from(&network_config);
    let train_dataset =
        ConsFormerMddDataset::<B>::new(problems, compilation.clone(), data_config, &device);
    let validation_dataset = ConsFormerMddDataset::<B::InnerBackend>::new(
        validation_problems,
        compilation,
        data_config,
        &device,
    );
    let batcher = ConsFormerMddBatcher::new(data_config);

    prepare_checkpoint_dir(checkpoint_dir, &network_config)?;

    // Create and train the network
    let _network = train_model::<
        B,
        ConsFormerConfig,
        ConsFormerMddSample<B>,
        ConsFormerMddBatch<B>,
        ConsFormerMddBatcher,
        ConsFormerMddLoss,
        ConsFormerMddSample<B::InnerBackend>,
        ConsFormerMddBatch<B::InnerBackend>,
    >(
        network_config,
        &all_problems,
        train_dataset,
        validation_dataset,
        batcher,
        ConsFormerMddLoss,
        training_config,
        checkpoint_dir,
        &device,
    );

    Ok(())
}
