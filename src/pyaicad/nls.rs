use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::config::Config;
use burn::tensor::backend::Backend;

use rand::RngExt;

use crate::learning::consformer::{ConsFormer, ConsFormerBatch, ConsFormerConfig};
use crate::learning::Network;
use crate::modelling::Problem;
use crate::nls::decode::{Argmax, DecodingOperator, Sampling};
use crate::nls::destroy::{DestroyOperator, RandomDestroy, RelatedDestroy, WorstDestroy};
use crate::nls::{
    load_network, Budget, MaskSchedule, NeuralLocalSearch, Solution, SolveConfig, Status,
};

use super::learn::cuda_available;
use super::problem::PyProblem;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PySolution {
    #[pyo3(get)]
    runtime: u64,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    solution: Option<Vec<isize>>,
    #[pyo3(get)]
    status: PyStatus,
}

impl From<&Solution> for PySolution {
    fn from(s: &Solution) -> Self {
        PySolution {
            runtime: s.runtime(),
            iterations: s.iterations(),
            solution: s.solution().clone(),
            status: (&s.status()).into(),
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyStatus {
    Satisfiable,
    Unsatisfiable,
    Unknown,
}

impl From<&Status> for PyStatus {
    fn from(s: &Status) -> Self {
        match s {
            Status::Satisfiable => Self::Satisfiable,
            Status::Unsatisfiable => Self::Unsatisfiable,
            Status::Unknown => Self::Unknown,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyNetworkKind {
    ConsFormer,
}

impl PyNetworkKind {
    fn tag(&self) -> &'static str {
        match self {
            PyNetworkKind::ConsFormer => "consformer",
        }
    }

    fn parse(tag: &str) -> PyResult<Self> {
        match tag {
            "consformer" => Ok(PyNetworkKind::ConsFormer),
            other => Err(PyValueError::new_err(format!(
                "unknown network_kind {other:?}"
            ))),
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyDestroyKind {
    Random,
    Worst,
    Related,
}

impl PyDestroyKind {
    fn build(&self) -> Box<dyn DestroyOperator> {
        match self {
            PyDestroyKind::Random => Box::new(RandomDestroy),
            PyDestroyKind::Worst => Box::new(WorstDestroy),
            PyDestroyKind::Related => Box::new(RelatedDestroy),
        }
    }

    fn tag(&self) -> &'static str {
        match self {
            PyDestroyKind::Random => "random",
            PyDestroyKind::Worst => "worst",
            PyDestroyKind::Related => "related",
        }
    }

    fn parse(tag: &str) -> PyResult<Self> {
        match tag {
            "random" => Ok(PyDestroyKind::Random),
            "worst" => Ok(PyDestroyKind::Worst),
            "related" => Ok(PyDestroyKind::Related),
            other => Err(PyValueError::new_err(format!(
                "unknown destroy_kind {other:?}"
            ))),
        }
    }
}

/// Builds the decode operator for neural local search: `stochastic_decode`
/// picks greedy (`Argmax`) vs. stochastic (`Sampling`) decoding of the network's logits.
fn build_decode_op<B: Backend>(
    stochastic_decode: bool,
    temperature: f64,
) -> Box<dyn DecodingOperator<B>> {
    if stochastic_decode {
        Box::new(Sampling { temperature })
    } else {
        Box::new(Argmax)
    }
}

/// Accepts either a single `PyProblem` or a list of them, so
/// `neural_local_search` can solve one problem or a whole batch through the
/// same entry point.
#[derive(FromPyObject)]
pub enum PyProblemsArg<'py> {
    Many(Vec<PyRef<'py, PyProblem>>),
    Single(PyRef<'py, PyProblem>),
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PySolveConfig {
    #[pyo3(get, set)]
    pub network_kind: PyNetworkKind,
    #[pyo3(get, set)]
    pub batch_size: Option<usize>,
    #[pyo3(get, set)]
    pub destroy_kind: PyDestroyKind,
    #[pyo3(get, set)]
    pub destroy_fraction_max: f64,
    #[pyo3(get, set)]
    pub destroy_fraction_min: f64,
    #[pyo3(get, set)]
    pub mask_schedule_epochs: usize,
    #[pyo3(get, set)]
    pub stochastic_decode: bool,
    #[pyo3(get, set)]
    pub temperature: f64,
    #[pyo3(get, set)]
    pub time_limit: Option<u64>,
    #[pyo3(get, set)]
    pub iteration_limit: Option<usize>,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl PySolveConfig {
    #[new]
    #[pyo3(signature = (
        network_kind=PyNetworkKind::ConsFormer,
        batch_size=None,
        destroy_kind=PyDestroyKind::Random,
        destroy_fraction_max=1.0,
        destroy_fraction_min=1.0,
        mask_schedule_epochs=0,
        stochastic_decode=false,
        temperature=1.0,
        time_limit=None,
        iteration_limit=None,
        seed=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        network_kind: PyNetworkKind,
        batch_size: Option<usize>,
        destroy_kind: PyDestroyKind,
        destroy_fraction_max: f64,
        destroy_fraction_min: f64,
        mask_schedule_epochs: usize,
        stochastic_decode: bool,
        temperature: f64,
        time_limit: Option<u64>,
        iteration_limit: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let config = PySolveConfig {
            network_kind,
            batch_size,
            destroy_kind,
            destroy_fraction_max,
            destroy_fraction_min,
            mask_schedule_epochs,
            stochastic_decode,
            temperature,
            time_limit,
            iteration_limit,
            seed,
        };
        SolveConfig::from(&config)
            .validate()
            .map_err(PyValueError::new_err)?;
        Ok(config)
    }

    #[staticmethod]
    fn from_json(path: String) -> PyResult<Self> {
        let config = SolveConfig::load_lenient(&path)
            .map_err(|e| PyValueError::new_err(format!("failed to load {path}: {e}")))?;
        (&config).try_into()
    }

    fn save_json(&self, path: String) -> PyResult<()> {
        let config: SolveConfig = self.into();
        config
            .save(&path)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to save {path}: {e}")))
    }
}

impl From<&PySolveConfig> for SolveConfig {
    fn from(c: &PySolveConfig) -> Self {
        SolveConfig {
            network_kind: c.network_kind.tag().to_string(),
            batch_size: c.batch_size,
            destroy_kind: c.destroy_kind.tag().to_string(),
            destroy_fraction_max: c.destroy_fraction_max,
            destroy_fraction_min: c.destroy_fraction_min,
            mask_schedule_epochs: c.mask_schedule_epochs,
            stochastic_decode: c.stochastic_decode,
            temperature: c.temperature,
            time_limit: c.time_limit,
            iteration_limit: c.iteration_limit,
            seed: c.seed,
        }
    }
}

impl TryFrom<&SolveConfig> for PySolveConfig {
    type Error = PyErr;

    fn try_from(c: &SolveConfig) -> Result<Self, Self::Error> {
        Ok(PySolveConfig {
            network_kind: PyNetworkKind::parse(&c.network_kind)?,
            batch_size: c.batch_size,
            destroy_kind: PyDestroyKind::parse(&c.destroy_kind)?,
            destroy_fraction_max: c.destroy_fraction_max,
            destroy_fraction_min: c.destroy_fraction_min,
            mask_schedule_epochs: c.mask_schedule_epochs,
            stochastic_decode: c.stochastic_decode,
            temperature: c.temperature,
            time_limit: c.time_limit,
            iteration_limit: c.iteration_limit,
            seed: c.seed,
        })
    }
}

/// Runs neural local search on `problems` (a single `Problem` or a list of
/// them, batched together into one search) using a network loaded from
/// `checkpoint_dir` (the `config.json` + `weights` produced by
/// `train_consformer`). Returns a single `PySolution` when given a single
/// problem, or a list of `PySolution` (in the same order as `problems`) when
/// given a list. Every problem in a batch must have the same number of
/// variables.
///
/// `batch_size` caps how many problems are ever loaded onto the device at
/// once; when the full problem list doesn't fit, it's processed in
/// sequential chunks of at most that size, reusing the same loaded network.
/// Left unset, every problem is batched together in a single pass (today's
/// behaviour). `time_limit` and `iteration_limit` apply per problem,
/// matching the classical-CP convention of a private timeout per instance:
/// every chunk gets its own full budget, so e.g. `time_limit=10` means each
/// problem gets up to 10 seconds to solve, regardless of how many chunks it
/// took to get through the whole list -- not 10 seconds total across the
/// call.
///
/// `config` gathers every knob unrelated to `problems`/`checkpoint_dir` (see `PySolveConfig`);
/// left unset, it's `PySolveConfig()` -- today's zero-config defaults.
#[pyfunction]
#[pyo3(signature = (problems, checkpoint_dir, config=None))]
pub fn neural_local_search(
    py: Python<'_>,
    problems: PyProblemsArg<'_>,
    checkpoint_dir: String,
    config: Option<PySolveConfig>,
) -> PyResult<Py<PyAny>> {
    let is_single = matches!(problems, PyProblemsArg::Single(_));
    let problems: Vec<Arc<Problem>> = match problems {
        PyProblemsArg::Single(p) => vec![p.arc()],
        PyProblemsArg::Many(ps) => ps.iter().map(|p| p.arc()).collect(),
    };
    if problems.is_empty() {
        return Err(PyValueError::new_err(
            "neural_local_search: `problems` must be non-empty",
        ));
    }
    let n = problems[0].number_variables();
    if problems.iter().any(|p| p.number_variables() != n) {
        return Err(PyValueError::new_err(
            "neural_local_search: all problems in a batch must have the same number of variables",
        ));
    }

    let checkpoint_dir = PathBuf::from(checkpoint_dir);
    let config: SolveConfig = config.as_ref().map(SolveConfig::from).unwrap_or_default();
    // `PySolveConfig::new` already validates this, but its fields are individually settable from
    // Python afterwards (`#[pyo3(get, set)]`), so re-check here at the point of use.
    config.validate().map_err(PyValueError::new_err)?;
    let budget = Budget {
        time_limit: config
            .time_limit
            .map(Duration::from_secs)
            .unwrap_or(Duration::MAX),
        iteration_limit: config.iteration_limit.unwrap_or(usize::MAX),
    };
    // Drawn from the process-wide RNG (see `crate::utils::rng`) so that, when `set_seed` has been
    // called, even a caller that leaves `seed` unset gets a reproducible destroy sequence instead
    // of a fresh OS-entropy one each time.
    let seed = config
        .seed
        .unwrap_or_else(|| crate::utils::with_rng(|rng| rng.random_range(0..u64::MAX)));

    // Releases the GIL for the actual search -- see `train_consformer`'s doc for why. Everything
    // captured here (`problems: Vec<Arc<Problem>>`, `config`, `checkpoint_dir`, `budget`, `seed`)
    // is already a Python-free, `Send` Rust value by this point, so the closure doesn't touch
    // anything GIL-bound.
    let solutions = py.detach(move || {
        if cuda_available() {
            run::<Cuda>(
                CudaDevice::default(),
                problems,
                &checkpoint_dir,
                &config,
                budget,
                seed,
            )
        } else {
            run::<NdArray>(
                NdArrayDevice::default(),
                problems,
                &checkpoint_dir,
                &config,
                budget,
                seed,
            )
        }
    })?;

    if is_single {
        Ok(Py::new(py, PySolution::from(&solutions[0]))?.into_any())
    } else {
        let items = solutions
            .iter()
            .map(|s| Py::new(py, PySolution::from(s)))
            .collect::<PyResult<Vec<Py<PySolution>>>>()?;
        let list = PyList::empty(py);
        for item in items {
            list.append(item)?;
        }
        Ok(list.into_any().unbind())
    }
}

fn run<B: Backend>(
    device: B::Device,
    problems: Vec<Arc<Problem>>,
    checkpoint_dir: &Path,
    config: &SolveConfig,
    budget: Budget,
    seed: u64,
) -> PyResult<Vec<Solution>> {
    let network_kind = PyNetworkKind::parse(&config.network_kind)?;
    let destroy_kind = PyDestroyKind::parse(&config.destroy_kind)?;

    match network_kind {
        PyNetworkKind::ConsFormer => {
            let (_network_config, network) =
                load_network::<B, ConsFormerConfig>(checkpoint_dir, &problems, &device).map_err(
                    |e| {
                        PyRuntimeError::new_err(format!(
                            "failed to load network from checkpoint {}: {e}",
                            checkpoint_dir.display()
                        ))
                    },
                )?;
            let mask_schedule = MaskSchedule {
                max: config.destroy_fraction_max,
                min: config.destroy_fraction_min,
                epochs: config.mask_schedule_epochs,
            };
            let destroy_op = destroy_kind.build();
            let decode_op = build_decode_op::<B>(config.stochastic_decode, config.temperature);

            let nls = NeuralLocalSearch::<B, ConsFormer<B>, ConsFormerBatch<B>>::new(
                network,
                destroy_op,
                mask_schedule,
                decode_op,
                1,
                device,
            );
            Ok(chunked_run(
                &nls,
                &problems,
                config.batch_size,
                budget,
                seed,
            ))
        }
    }
}

fn chunked_run<B: Backend, N: Network<B, Ba>, Ba: crate::learning::Batch<B>>(
    nls: &NeuralLocalSearch<B, N, Ba>,
    problems: &[Arc<Problem>],
    batch_size: Option<usize>,
    budget: Budget,
    seed: u64,
) -> Vec<Solution> {
    let chunk_size = batch_size.unwrap_or(problems.len()).max(1);
    let mut solutions = Vec::with_capacity(problems.len());
    log::info!(
        "Solving {} problems by chunk of size {}",
        problems.len(),
        chunk_size
    );
    for (chunk_idx, chunk) in problems.chunks(chunk_size).enumerate() {
        log::info!("Solving chunk {}", chunk_idx);
        // Vary the seed per chunk so chunks don't replay the exact same destroy sequence.
        let chunk_seed = seed.wrapping_add(chunk_idx as u64);
        solutions.extend(nls.run(chunk, budget, chunk_seed));
    }

    solutions
}
