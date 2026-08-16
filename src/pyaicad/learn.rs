use std::path::{Path, PathBuf};
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::backend::Autodiff;
use burn::config::Config;
use burn::tensor::backend::AutodiffBackend;

use rand::rng;
use rand::seq::SliceRandom;

use crate::learning::consformer::{
    ConsFormerBatch, ConsFormerBatcher, ConsFormerConfig, ConsFormerDataset, ConsFormerLoss,
    ConsFormerMddBatch, ConsFormerMddBatcher, ConsFormerMddDataset, ConsFormerMddLoss,
    ConsFormerMddSample, ConsFormerSample, MddCompilationConfig,
};
use crate::learning::train::{train_model, ModelSelection, TrainingConfig};
use crate::modelling::Problem;

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
            positional_encoding_dimensions=0))]
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
}

#[pymethods]
impl PyTrainingConfig {
    #[new]
    #[pyo3(signature = (lr=3e-4, num_epochs=10, batch_size=32, validation_interval=10, model_selection=PyModelSelection::Loss))]
    fn new(
        lr: f64,
        num_epochs: usize,
        batch_size: usize,
        validation_interval: usize,
        model_selection: PyModelSelection,
    ) -> Self {
        PyTrainingConfig {
            lr,
            num_epochs,
            batch_size,
            validation_interval,
            model_selection,
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

/// Trains consformer on a set of problems.
#[pyfunction]
pub fn train_consformer(
    problems: Vec<PyRef<PyProblem>>,
    config: &PyConsFormerConfig,
    training: &PyTrainingConfig,
    checkpoint_dir: String,
) -> PyResult<()> {
    let problems: Vec<Arc<Problem>> = problems.iter().map(|p| p.arc()).collect();
    let network_config: ConsFormerConfig = config.into();
    let training_config: TrainingConfig = training.into();
    let checkpoint_dir = PathBuf::from(checkpoint_dir);

    if cuda_available() {
        run_training::<Autodiff<Cuda>>(
            CudaDevice::default(),
            problems,
            network_config,
            training_config,
            &checkpoint_dir,
        )
    } else {
        run_training::<Autodiff<NdArray>>(
            NdArrayDevice::default(),
            problems,
            network_config,
            training_config,
            &checkpoint_dir,
        )
    }
}

/// runs the training loop for a set of training problems, a network/training loop configuration.
/// The network is saved to the checkpoint.
fn run_training<B: AutodiffBackend>(
    device: B::Device,
    mut problems: Vec<Arc<Problem>>,
    network_config: ConsFormerConfig,
    training_config: TrainingConfig,
    checkpoint_dir: &Path,
) -> PyResult<()> {
    let _problems = problems.clone();
    let training_size = (problems.len() as f64 * 0.8).round() as usize;
    problems.shuffle(&mut rng());
    let validation_problems = problems.split_off(training_size);
    let train_dataset = ConsFormerDataset::<B>::new(problems, &device);
    let validation_dataset =
        ConsFormerDataset::<B::InnerBackend>::new(validation_problems, &device);
    let batcher = ConsFormerBatcher {
        mask_fraction: network_config.mask_fraction,
    };

    std::fs::create_dir_all(checkpoint_dir)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create checkpoint dir: {e}")))?;

    // Saves the hyper-parameters of the network so that we can just load them at inference
    network_config
        .save(checkpoint_dir.join("config.json"))
        .map_err(|e| PyRuntimeError::new_err(format!("failed to save network config: {e}")))?;

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
        &_problems,
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
/// checkpointing. `epsilon` is added to each constraint's WMC before taking `-log` (see
/// `ConsFormerMddLoss`); `pyordering`/`pymerge`/`pyselect` control how each constraint's MDD is
/// compiled (`pymerge` is accepted for parity with `Compiler::compile` but has no effect here --
/// this recipe always refines to an exact, unbounded-width MDD, and `merge` only ever triggers
/// once a width bound would be exceeded; see `MddCompilationConfig`'s doc).
#[pyfunction]
#[pyo3(signature = (problems, config, training, checkpoint_dir, epsilon=1e-6,
        pyordering=PyOrderingHeuristic::MinDomMaxLinked(),
        pymerge=PyMergeHeuristic::LessRelaxed,
        pyselect=PySelectHeuristic::Greedy))]
#[allow(clippy::too_many_arguments)]
pub fn train_consformer_mdd(
    problems: Vec<PyRef<PyProblem>>,
    config: &PyConsFormerConfig,
    training: &PyTrainingConfig,
    checkpoint_dir: String,
    epsilon: f64,
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
    };

    if cuda_available() {
        run_training_mdd::<Autodiff<Cuda>>(
            CudaDevice::default(),
            problems,
            network_config,
            training_config,
            compilation,
            epsilon,
            &checkpoint_dir,
        )
    } else {
        run_training_mdd::<Autodiff<NdArray>>(
            NdArrayDevice::default(),
            problems,
            network_config,
            training_config,
            compilation,
            epsilon,
            &checkpoint_dir,
        )
    }
}

/// Runs the ConsFormer-MDD training loop -- see `run_training` for the classical recipe this
/// mirrors. The only structural differences are the dataset/batcher/loss types (`ConsFormerMdd*`
/// instead of `ConsFormer*`) and the extra `compilation`/`domain_size` the MDD dataset needs to
/// compile each problem's constraints into exact MDDs (`domain_size` is read off
/// `network_config` -- it must already match what the network itself was configured with, since
/// it's also the width of the per-variable probability vector `ConsFormerMddDataset`'s gather
/// indices are computed against).
fn run_training_mdd<B: AutodiffBackend>(
    device: B::Device,
    mut problems: Vec<Arc<Problem>>,
    network_config: ConsFormerConfig,
    training_config: TrainingConfig,
    compilation: MddCompilationConfig,
    epsilon: f64,
    checkpoint_dir: &Path,
) -> PyResult<()> {
    let _problems = problems.clone();
    let training_size = (problems.len() as f64 * 0.8).round() as usize;
    problems.shuffle(&mut rng());
    let validation_problems = problems.split_off(training_size);

    let domain_size = network_config.domain_size;
    let train_dataset =
        ConsFormerMddDataset::<B>::new(problems, compilation.clone(), domain_size, &device);
    let validation_dataset = ConsFormerMddDataset::<B::InnerBackend>::new(
        validation_problems,
        compilation,
        domain_size,
        &device,
    );
    let batcher = ConsFormerMddBatcher {
        mask_fraction: network_config.mask_fraction,
        domain_size,
    };

    std::fs::create_dir_all(checkpoint_dir)
        .map_err(|e| PyRuntimeError::new_err(format!("failed to create checkpoint dir: {e}")))?;

    // Saves the hyper-parameters of the network so that we can just load them at inference
    network_config
        .save(checkpoint_dir.join("config.json"))
        .map_err(|e| PyRuntimeError::new_err(format!("failed to save network config: {e}")))?;

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
        &_problems,
        train_dataset,
        validation_dataset,
        batcher,
        ConsFormerMddLoss { epsilon },
        training_config,
        checkpoint_dir,
        &device,
    );

    Ok(())
}
