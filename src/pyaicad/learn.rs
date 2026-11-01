use std::path::{Path, PathBuf};
use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use burn::backend::Autodiff;
use burn::backend::cuda::{Cuda, CudaDevice};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::config::Config;
use burn::tensor::backend::{AutodiffBackend, Backend};

use rand::seq::SliceRandom;

use crate::destroy::DestroyOperator;
use crate::learning::consformer::architecture::MaskingKind;
use crate::learning::consformer::{
    ConsFormerBatch, ConsFormerBatcher, ConsFormerConfig, ConsFormerDataset, ConsFormerLoss,
    ConsFormerSample, DecomposedMddLoss,
};
use crate::learning::optimiser::OptimiserConfig;
use crate::learning::train::{TrainingConfig, train_model};
use crate::mdd::arena::MddCompilationConfig;
use crate::mdd::heuristics::MergeHeuristic;
use crate::modelling::Problem;
use crate::utils::with_rng;

use super::heuristics::{PyMergeHeuristic, PyOrderingHeuristic, PySelectHeuristic};
use super::local_search::PyDestroyKind;
use super::optimiser::PyOptimiser;
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

#[pyclass]
pub struct PyTrainingConfig {
    /// The optimiser and its hyper-parameters, learning rate included.
    pub optimiser: PyOptimiser,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub validation_interval: usize,
    /// Stop training once the validation loss hasn't improved by more than
    /// `early_stopping_min_delta` for this many consecutive validation checks. `None` disables
    /// early stopping.
    pub early_stopping_patience: Option<usize>,
    /// How much the validation loss must improve to count as progress rather than
    /// stagnation. Only used when `early_stopping_patience` is set.
    pub early_stopping_min_delta: f64,
    pub destroy_kind: PyDestroyKind,
    /// A checkpoint directory whose weights initialise the network before training.
    pub initial_weights: Option<String>,
}

#[pymethods]
impl PyTrainingConfig {
    #[new]
    #[pyo3(signature = (optimiser, num_epochs=10, batch_size=32, validation_interval=10,
            early_stopping_patience=None, early_stopping_min_delta=1e-4, destroy_kind=PyDestroyKind::Random,
            initial_weights=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        optimiser: PyOptimiser,
        num_epochs: usize,
        batch_size: usize,
        validation_interval: usize,
        early_stopping_patience: Option<usize>,
        early_stopping_min_delta: f64,
        destroy_kind: PyDestroyKind,
        initial_weights: Option<String>,
    ) -> Self {
        PyTrainingConfig {
            optimiser,
            num_epochs,
            batch_size,
            validation_interval,
            early_stopping_patience,
            early_stopping_min_delta,
            destroy_kind,
            initial_weights,
        }
    }
}

impl From<&PyTrainingConfig> for TrainingConfig {
    fn from(c: &PyTrainingConfig) -> Self {
        TrainingConfig {
            optimiser: OptimiserConfig::from(&c.optimiser),
            num_epochs: c.num_epochs,
            batch_size: c.batch_size,
            validation_interval: c.validation_interval,
            early_stopping_patience: c.early_stopping_patience,
            early_stopping_min_delta: c.early_stopping_min_delta,
            initial_weights: c.initial_weights.as_ref().map(PathBuf::from),
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
pub(crate) fn set_seed(seed: u64) {
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
/// full problem set that `train_model` hands to the network and the loss to prepare),
/// while `train`/`validation` are a random 80/20 split of a shuffled copy. Shared by every
/// `run_training_*` function so the split logic can't drift between recipes.
fn split_train_validation(
    mut problems: Vec<Arc<Problem>>,
    validation: Option<Vec<Arc<Problem>>>,
) -> ProblemSplit {
    if let Some(validation_problems) = validation {
        let mut all_problems = problems.clone();
        all_problems.extend(validation_problems.iter().cloned());
        return (all_problems, problems, validation_problems);
    }
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
#[pyo3(signature = (problems, config, training, checkpoint_dir, validation=None))]
pub(crate) fn train_consformer(
    py: Python<'_>,
    problems: Vec<PyRef<PyProblem>>,
    config: &PyConsFormerConfig,
    training: &PyTrainingConfig,
    checkpoint_dir: String,
    validation: Option<Vec<PyRef<PyProblem>>>,
) -> PyResult<()> {
    let problems: Vec<Arc<Problem>> = problems.iter().map(|p| p.arc()).collect();
    let validation: Option<Vec<Arc<Problem>>> =
        validation.map(|v| v.iter().map(|p| p.arc()).collect());
    let network_config: ConsFormerConfig = config.into();
    let training_config: TrainingConfig = training.into();
    let destroy = training
        .destroy_kind
        .training_operator(network_config.mask_fraction);
    let checkpoint_dir = PathBuf::from(checkpoint_dir);

    py.detach(move || {
        dispatch_backend!(
            run_training,
            problems,
            validation,
            network_config,
            training_config,
            destroy,
            &checkpoint_dir,
        )
    })
}

/// runs the training loop for a set of training problems, a network/training loop configuration.
/// The network is saved to the checkpoint.
fn run_training<B: AutodiffBackend>(
    device: B::Device,
    problems: Vec<Arc<Problem>>,
    validation: Option<Vec<Arc<Problem>>>,
    network_config: ConsFormerConfig,
    training_config: TrainingConfig,
    destroy: Option<Arc<dyn DestroyOperator>>,
    checkpoint_dir: &Path,
) -> PyResult<()> {
    prepare_checkpoint_dir(checkpoint_dir, &network_config)?;

    let (all_problems, problems, validation_problems) =
        split_train_validation(problems, validation);
    let train_dataset = ConsFormerDataset::<B>::new(problems, &device);
    let validation_dataset =
        ConsFormerDataset::<B::InnerBackend>::new(validation_problems, &device);
    let batcher = ConsFormerBatcher {
        mask_fraction: network_config.mask_fraction,
        destroy,
    };

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
    )
    .map_err(PyValueError::new_err)?;

    Ok(())
}

/// Trains ConsFormer against the exact per-constraint MDD weighted model count (see
/// `DecomposedMddLoss`) instead of the classical hand-written per-constraint-type penalty --
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
        pyselect=PySelectHeuristic::Greedy,
        validation=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn train_consformer_mdd(
    py: Python<'_>,
    problems: Vec<PyRef<PyProblem>>,
    config: &PyConsFormerConfig,
    training: &PyTrainingConfig,
    checkpoint_dir: String,
    pyordering: PyOrderingHeuristic,
    pymerge: PyMergeHeuristic,
    pyselect: PySelectHeuristic,
    validation: Option<Vec<PyRef<PyProblem>>>,
) -> PyResult<()> {
    let problems: Vec<Arc<Problem>> = problems.iter().map(|p| p.arc()).collect();
    let validation: Option<Vec<Arc<Problem>>> =
        validation.map(|v| v.iter().map(|p| p.arc()).collect());
    let network_config: ConsFormerConfig = config.into();
    let training_config: TrainingConfig = training.into();
    if !matches!(training.destroy_kind, PyDestroyKind::Random) {
        return Err(PyValueError::new_err(
            "train_consformer_mdd masks every variable and does not take a destroy_kind",
        ));
    }
    let checkpoint_dir = PathBuf::from(checkpoint_dir);
    // The merge heuristic is not read while the MDDs are compiled with unbounded width, and the
    // problems of a training set do not share their probabilities: `MostLikely` takes those of
    // the first problem so that the setting has a value.
    let merge = problems
        .first()
        .map_or(MergeHeuristic::LessRelaxed, |problem| {
            pymerge.into_heuristic(problem)
        });
    let compilation = MddCompilationConfig {
        ordering: pyordering.into(),
        merge,
        select: pyselect.into(),
        max_width: usize::MAX,
    };

    py.detach(move || {
        dispatch_backend!(
            run_training_mdd,
            problems,
            validation,
            network_config,
            training_config,
            compilation,
            &checkpoint_dir,
        )
    })
}

/// Runs the ConsFormer-MDD training loop -- see `run_training` for the classical recipe this
/// mirrors. The only structural difference is the loss: `DecomposedMddLoss` scores each problem
/// through its constraints compiled into exact MDDs, which `train_model` has it compile, once,
/// for every training and validation problem.
fn run_training_mdd<B: AutodiffBackend>(
    device: B::Device,
    problems: Vec<Arc<Problem>>,
    validation: Option<Vec<Arc<Problem>>>,
    network_config: ConsFormerConfig,
    training_config: TrainingConfig,
    compilation: MddCompilationConfig,
    checkpoint_dir: &Path,
) -> PyResult<()> {
    prepare_checkpoint_dir(checkpoint_dir, &network_config)?;

    let (all_problems, problems, validation_problems) =
        split_train_validation(problems, validation);

    let loss = DecomposedMddLoss::new(compilation, network_config.domain_size);
    let train_dataset = ConsFormerDataset::<B>::new(problems, &device);
    let validation_dataset =
        ConsFormerDataset::<B::InnerBackend>::new(validation_problems, &device);
    let batcher = ConsFormerBatcher {
        mask_fraction: network_config.mask_fraction,
        destroy: None,
    };

    // Create and train the network
    let _network = train_model::<
        B,
        ConsFormerConfig,
        ConsFormerSample<B>,
        ConsFormerBatch<B>,
        ConsFormerBatcher,
        DecomposedMddLoss,
        ConsFormerSample<B::InnerBackend>,
        ConsFormerBatch<B::InnerBackend>,
    >(
        network_config,
        &all_problems,
        train_dataset,
        validation_dataset,
        batcher,
        loss,
        training_config,
        checkpoint_dir,
        &device,
    )
    .map_err(PyValueError::new_err)?;

    Ok(())
}
