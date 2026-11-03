use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::learning::consformer::{train_consformer as train, train_consformer_mdd as train_mdd};
use crate::learning::{OptimiserConfig, TrainingConfig};
use crate::mdd::MddCompilationConfig;
use crate::mdd::heuristics::MergeHeuristic;
use crate::modelling::Problem;
use crate::neural::consformer::{ConsFormerConfig, MaskingKind};

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
        train(
            problems,
            validation,
            network_config,
            training_config,
            destroy,
            &checkpoint_dir,
        )
    })
    .map_err(PyValueError::new_err)
}

/// Trains ConsFormer against the exact per-constraint MDD weighted model count (see
/// `DecomposedMddLoss`) instead of the classical hand-written per-constraint-type penalty --
/// otherwise identical to `train_consformer`: same network architecture, same config, same
/// checkpointing. `pyordering`/`pymerge`/`pyselect` control how each constraint's MDD is compiled
/// (`pymerge` is accepted for parity with `Compiler::compile` but has no effect here -- this
/// recipe always refines to an exact, unbounded-width MDD, and `merge` only ever triggers once a
/// width bound would be exceeded; see `MddCompilationConfig`'s doc).
///
/// Releases the GIL for the whole training run -- see `train_consformer`'s doc for why.
#[pyfunction]
#[pyo3(signature = (problems, config, training, checkpoint_dir,
        pyordering=PyOrderingHeuristic::MinDomMaxLinked(),
        pymerge=PyMergeHeuristic::StateSimilarity,
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
    // problems of a training set do not share their domains: `MostLikely` takes the uniform
    // probabilities of the first problem so that the setting has a value.
    let merge = problems
        .first()
        .map_or(MergeHeuristic::default(), |problem| {
            pymerge.into_heuristic(problem)
        });
    let compilation = MddCompilationConfig {
        ordering: pyordering.into(),
        merge,
        select: pyselect.into(),
        max_width: usize::MAX,
    };

    py.detach(move || {
        train_mdd(
            problems,
            validation,
            network_config,
            training_config,
            compilation,
            &checkpoint_dir,
        )
    })
    .map_err(PyValueError::new_err)
}
