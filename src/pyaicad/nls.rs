use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::backend::Backend;

use rand::RngExt;

use crate::learning::consformer::{ConsFormer, ConsFormerBatch, ConsFormerConfig, MddCompilationConfig};
use crate::learning::Network;
use crate::modelling::Problem;
use crate::nls::decode::{Argmax, DecodingOperator, Sampling};
use crate::nls::destroy::{DestroyOperator, RandomDestroy, RelatedDestroy, WorstDestroy};
use crate::nls::{load_network, Budget, MddGibbsDecoding, NeuralLocalSearch, Solution, Status};
use crate::sampling::DecodeMode;

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

#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyDestroyKind {
    Random,
    Worst,
    Related,
}

impl PyDestroyKind {
    fn build(&self, fraction: f64) -> Box<dyn DestroyOperator> {
        match self {
            PyDestroyKind::Random => Box::new(RandomDestroy { fraction }),
            PyDestroyKind::Worst => Box::new(WorstDestroy { fraction }),
            PyDestroyKind::Related => Box::new(RelatedDestroy { fraction }),
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyDecodeKind {
    Argmax,
    Sampling,
    MddGibbs,
}

#[allow(clippy::too_many_arguments)]
impl PyDecodeKind {
    fn build<B: Backend>(
        &self,
        temperature: f64,
        domain_size: usize,
        mdd_rounds: usize,
        mdd_greedy: bool,
        mdd_gibbs_cleanup: bool,
    ) -> Box<dyn DecodingOperator<B>> {
        match self {
            PyDecodeKind::Argmax => Box::new(Argmax),
            PyDecodeKind::Sampling => Box::new(Sampling { temperature }),
            PyDecodeKind::MddGibbs => {
                let mode = if mdd_greedy {
                    DecodeMode::Greedy
                } else {
                    DecodeMode::Sample
                };
                Box::new(MddGibbsDecoding::new(
                    MddCompilationConfig::default(),
                    domain_size,
                    mdd_rounds,
                    mode,
                    mdd_gibbs_cleanup,
                ))
            }
        }
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

/// Runs neural local search on `problems` (a single `Problem` or a list of
/// them, batched together into one search) using a network loaded from
/// `checkpoint_dir` (the `config.json` + `weights` produced by
/// `train_consformer`). Returns a single `PySolution` when given a single
/// problem, or a list of `PySolution` (in the same order as `problems`) when
/// given a list. Every problem in a batch must have the same number of
/// variables.
///
/// `max_batch_size` caps how many problems (times `population_size` rows
/// each) are ever loaded onto the device at once; when the full problem list
/// doesn't fit, it's processed in sequential chunks of at most that size,
/// reusing the same loaded network. Left unset, every problem is batched
/// together in a single pass (today's behaviour). `time_limit` and
/// `iteration_limit` apply per problem, matching the classical-CP convention
/// of a private timeout per instance: every chunk gets its own full budget,
/// so e.g. `time_limit=10` means each problem gets up to 10 seconds to
/// solve, regardless of how many chunks it took to get through the whole
/// list -- not 10 seconds total across the call.
#[pyfunction]
#[pyo3(signature = (
    problems,
    checkpoint_dir,
    network_kind=PyNetworkKind::ConsFormer,
    time_limit=None,
    iteration_limit=None,
    population_size=1,
    max_batch_size=None,
    destroy_kind=PyDestroyKind::Random,
    destroy_fraction=None,
    decode_kind=PyDecodeKind::Argmax,
    temperature=0.1,
    mdd_rounds=4,
    mdd_greedy=false,
    mdd_gibbs_cleanup=true,
    seed=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn neural_local_search(
    py: Python<'_>,
    problems: PyProblemsArg<'_>,
    checkpoint_dir: String,
    network_kind: PyNetworkKind,
    time_limit: Option<u64>,
    iteration_limit: Option<usize>,
    population_size: usize,
    max_batch_size: Option<usize>,
    destroy_kind: PyDestroyKind,
    destroy_fraction: Option<f64>,
    decode_kind: PyDecodeKind,
    temperature: f64,
    mdd_rounds: usize,
    mdd_greedy: bool,
    mdd_gibbs_cleanup: bool,
    seed: Option<u64>,
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
    let budget = Budget {
        time_limit: time_limit.map(Duration::from_secs).unwrap_or(Duration::MAX),
        iteration_limit: iteration_limit.unwrap_or(usize::MAX),
    };
    // Drawn from the process-wide RNG (see `crate::utils::rng`) so that, when `set_seed` has been
    // called, even a caller that leaves `seed` unset gets a reproducible destroy sequence instead
    // of a fresh OS-entropy one each time.
    let seed = seed.unwrap_or_else(|| crate::utils::with_rng(|rng| rng.random_range(0..u64::MAX)));

    // Releases the GIL for the actual search -- see `train_consformer`'s doc for why. Everything
    // captured here (`problems: Vec<Arc<Problem>>`, the plain-enum `network_kind`/`destroy_kind`/
    // `decode_kind`, `checkpoint_dir`, `budget`, `seed`) is already a Python-free, `Send` Rust
    // value by this point, so the closure doesn't touch anything GIL-bound.
    let solutions = py.detach(move || {
        if cuda_available() {
            run::<Cuda>(
                CudaDevice::default(),
                problems,
                max_batch_size,
                &checkpoint_dir,
                &network_kind,
                &destroy_kind,
                destroy_fraction,
                &decode_kind,
                temperature,
                mdd_rounds,
                mdd_greedy,
                mdd_gibbs_cleanup,
                population_size,
                budget,
                seed,
            )
        } else {
            run::<NdArray>(
                NdArrayDevice::default(),
                problems,
                max_batch_size,
                &checkpoint_dir,
                &network_kind,
                &destroy_kind,
                destroy_fraction,
                &decode_kind,
                temperature,
                mdd_rounds,
                mdd_greedy,
                mdd_gibbs_cleanup,
                population_size,
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

#[allow(clippy::too_many_arguments)]
fn run<B: Backend>(
    device: B::Device,
    problems: Vec<Arc<Problem>>,
    max_batch_size: Option<usize>,
    checkpoint_dir: &Path,
    network_kind: &PyNetworkKind,
    destroy_kind: &PyDestroyKind,
    destroy_fraction: Option<f64>,
    decode_kind: &PyDecodeKind,
    temperature: f64,
    mdd_rounds: usize,
    mdd_greedy: bool,
    mdd_gibbs_cleanup: bool,
    population_size: usize,
    budget: Budget,
    seed: u64,
) -> PyResult<Vec<Solution>> {
    match network_kind {
        PyNetworkKind::ConsFormer => {
            let (config, network) =
                load_network::<B, ConsFormerConfig>(checkpoint_dir, &problems, &device).map_err(
                    |e| {
                        PyRuntimeError::new_err(format!(
                            "failed to load network from checkpoint {}: {e}",
                            checkpoint_dir.display()
                        ))
                    },
                )?;
            let fraction = destroy_fraction.unwrap_or(config.mask_fraction);
            let destroy_op = destroy_kind.build(fraction);
            let decode_op = decode_kind.build::<B>(
                temperature,
                config.domain_size,
                mdd_rounds,
                mdd_greedy,
                mdd_gibbs_cleanup,
            );

            let nls = NeuralLocalSearch::<B, ConsFormer<B>, ConsFormerBatch<B>>::new(
                network,
                destroy_op,
                decode_op,
                population_size,
                device,
            );
            Ok(chunked_run(&nls, &problems, max_batch_size, budget, seed))
        }
    }
}

fn chunked_run<B: Backend, N: Network<B, Ba>, Ba: crate::learning::Batch<B>>(
    nls: &NeuralLocalSearch<B, N, Ba>,
    problems: &[Arc<Problem>],
    max_batch_size: Option<usize>,
    budget: Budget,
    seed: u64,
) -> Vec<Solution> {
    let chunk_size = max_batch_size.unwrap_or(problems.len()).max(1);
    let mut solutions = Vec::with_capacity(problems.len());
    log::info!("Solving {} problems by chunk of size {}", problems.len(), chunk_size);
    for (chunk_idx, chunk) in problems.chunks(chunk_size).enumerate() {
        log::info!("Solving chunk {}", chunk_idx);
        // Vary the seed per chunk so chunks don't replay the exact same destroy sequence.
        let chunk_seed = seed.wrapping_add(chunk_idx as u64);
        solutions.extend(nls.run(chunk, budget, chunk_seed));
    }

    solutions
}
