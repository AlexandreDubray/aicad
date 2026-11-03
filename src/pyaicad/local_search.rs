use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::destroy::{DestroyKind, DestroyOperator};
use crate::local_search::{
    BeliefPropagationSearchConfig, NeuralPriorBpSearchConfig, NeuralPriorSearchConfig,
    SearchConfig, SolveConfig, solve,
};
use crate::modelling::{Problem, Solution, Status};

use super::problem::PyProblem;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub(crate) struct PySolution {
    /// Seconds elapsed before finding the solution (or before the budget ran out).
    /// Sub-second precision -- see `Solution::runtime`'s doc.
    #[pyo3(get)]
    runtime: f64,
    /// Seconds spent compiling MDDs (`decode_kind=MddSampling`'s `DecomposedMddSampling`'s preparation)
    /// for the `run` call this solution came out of -- 0.0 for `decode_kind=Logits`, which has no
    /// compilation step. Already included in `runtime`, not additional to it -- see
    /// `Solution::compilation_runtime`'s doc.
    #[pyo3(get)]
    compilation_runtime: f64,
    /// `None` for a method without a search.
    #[pyo3(get)]
    iterations: Option<usize>,
    #[pyo3(get)]
    solution: Option<Vec<isize>>,
    #[pyo3(get)]
    status: PyStatus,
}

impl From<&Solution> for PySolution {
    fn from(s: &Solution) -> Self {
        PySolution {
            runtime: s.runtime(),
            compilation_runtime: s.compilation_runtime(),
            iterations: s.iterations(),
            solution: s.assignment().map(<[isize]>::to_vec),
            status: s.status().into(),
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub(crate) enum PyStatus {
    Satisfiable,
    Unsatisfiable,
    Unknown,
}

impl From<Status> for PyStatus {
    fn from(s: Status) -> Self {
        match s {
            Status::Satisfiable => Self::Satisfiable,
            Status::Unsatisfiable => Self::Unsatisfiable,
            Status::Unknown => Self::Unknown,
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

impl From<&PyDestroyKind> for DestroyKind {
    fn from(kind: &PyDestroyKind) -> Self {
        match kind {
            PyDestroyKind::Random => DestroyKind::Random,
            PyDestroyKind::Worst => DestroyKind::Worst,
            PyDestroyKind::Related => DestroyKind::Related,
        }
    }
}

impl PyDestroyKind {
    /// The operator to train with, `None` for `Random`, which is the training default.
    pub(super) fn training_operator(&self, fraction: f64) -> Option<Arc<dyn DestroyOperator>> {
        match self {
            PyDestroyKind::Random => None,
            other => Some(DestroyKind::from(other).build(fraction).into()),
        }
    }
}

/// Accepts either a single `PyProblem` or a list of them, so
/// `local_search` can solve one problem or a whole batch through the
/// same entry point.
#[derive(FromPyObject)]
pub(crate) enum PyProblemsArg<'py> {
    Many(Vec<PyRef<'py, PyProblem>>),
    Single(PyRef<'py, PyProblem>),
}

/// The architecture of a local search and its settings, one variant per architecture.
///
/// Every variant holds every setting, so that the Python side picks an architecture by building
/// the matching variant; the defaults live in Python (the classes of `pyaicad.solve`).
#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PySearch {
    /// A trained network, decoding its logits. `temperature` only matters when `stochastic`.
    NeuralPrior {
        destroy_kind: PyDestroyKind,
        destroy_fraction: f64,
        stochastic: bool,
        temperature: f64,
    },
    /// A trained network whose probabilities are the prior of belief propagation over the MDDs of
    /// the constraints.
    NeuralPriorBp {
        destroy_kind: PyDestroyKind,
        destroy_fraction: f64,
        stochastic: bool,
        bp_iterations: usize,
    },
    /// Belief propagation over the MDDs of the constraints from uniform probabilities, on the CPU
    /// only: no network and no checkpoint.
    BeliefPropagation {
        destroy_kind: PyDestroyKind,
        destroy_fraction: f64,
        stochastic: bool,
        bp_iterations: usize,
    },
}

impl From<&PySearch> for SearchConfig {
    fn from(search: &PySearch) -> Self {
        match search {
            PySearch::NeuralPrior {
                destroy_kind,
                destroy_fraction,
                stochastic,
                temperature,
            } => SearchConfig::NeuralPrior(NeuralPriorSearchConfig {
                destroy_kind: destroy_kind.into(),
                destroy_fraction: *destroy_fraction,
                stochastic: *stochastic,
                temperature: *temperature,
            }),
            PySearch::NeuralPriorBp {
                destroy_kind,
                destroy_fraction,
                stochastic,
                bp_iterations,
            } => SearchConfig::NeuralPriorBp(NeuralPriorBpSearchConfig {
                destroy_kind: destroy_kind.into(),
                destroy_fraction: *destroy_fraction,
                stochastic: *stochastic,
                bp_iterations: *bp_iterations,
            }),
            PySearch::BeliefPropagation {
                destroy_kind,
                destroy_fraction,
                stochastic,
                bp_iterations,
            } => SearchConfig::BeliefPropagation(BeliefPropagationSearchConfig {
                destroy_kind: destroy_kind.into(),
                destroy_fraction: *destroy_fraction,
                stochastic: *stochastic,
                bp_iterations: *bp_iterations,
            }),
        }
    }
}

/// What a run takes beyond its search: the batching, the limits and the seed.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PySolveConfig {
    #[pyo3(get, set)]
    pub search: PySearch,
    #[pyo3(get, set)]
    pub batch_size: Option<usize>,
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
    #[pyo3(signature = (search, batch_size=None, time_limit=None, iteration_limit=None, seed=None))]
    fn new(
        search: PySearch,
        batch_size: Option<usize>,
        time_limit: Option<u64>,
        iteration_limit: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        PySolveConfig {
            search,
            batch_size,
            time_limit,
            iteration_limit,
            seed,
        }
    }
}

impl From<&PySolveConfig> for SolveConfig {
    fn from(c: &PySolveConfig) -> Self {
        let mut config = SolveConfig::default().with_search((&c.search).into());
        if let Some(seconds) = c.time_limit {
            config = config.with_runtime_limit(seconds);
        }
        if let Some(iterations) = c.iteration_limit {
            config = config.with_iteration_limit(iterations);
        }
        if let Some(batch_size) = c.batch_size {
            config = config.with_batch_size(batch_size);
        }
        if let Some(seed) = c.seed {
            config = config.with_seed(seed);
        }
        config
    }
}

/// Runs a local search on `problems` (a single `Problem` or a list of them, batched together into
/// one search). Returns a single `PySolution` when given a single problem, or a list of
/// `PySolution` (in the same order as `problems`) when given a list.
///
/// `config` holds the architecture of the search with its settings (see `PySearch`), the
/// batching, the limits and the seed. The neural architectures use the network of
/// `checkpoint_dir` (the `config.json` + `weights` produced by `train_consformer`) and need it;
/// the belief-propagation one uses none and refuses it.
///
/// `batch_size` caps how many problems are stepped together; the list is processed in batches of
/// at most that size, one after the other, reusing the same loaded network. Left unset, every
/// problem is in a single batch. `time_limit` and `iteration_limit` apply per batch, matching the
/// classical-CP convention of a private timeout per instance when `batch_size=1`:
/// `time_limit=10` gives each batch up to 10 seconds, not 10 seconds in total across the call.
///
/// Releases the GIL for the whole search: everything it needs is extracted into plain, `Send`
/// Rust values *before* the release, so nothing inside the closure touches Python (see
/// `train_consformer`'s doc for why that matters for progressive logging).
#[pyfunction]
#[pyo3(name = "local_search", signature = (problems, config, checkpoint_dir=None))]
pub(crate) fn run_local_search(
    py: Python<'_>,
    problems: PyProblemsArg<'_>,
    config: PySolveConfig,
    checkpoint_dir: Option<String>,
) -> PyResult<Py<PyAny>> {
    let is_single = matches!(problems, PyProblemsArg::Single(_));
    let problems: Vec<Arc<Problem>> = match problems {
        PyProblemsArg::Single(p) => vec![p.arc()],
        PyProblemsArg::Many(ps) => ps.iter().map(|p| p.arc()).collect(),
    };
    if problems.is_empty() {
        return Err(PyValueError::new_err(
            "local_search: `problems` must be non-empty",
        ));
    }
    let checkpoint_dir = checkpoint_dir.map(PathBuf::from);
    let config = SolveConfig::from(&config);

    let solutions = py
        .detach(move || solve(&problems, &config, checkpoint_dir.as_deref()))
        .map_err(PyValueError::new_err)?;

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
