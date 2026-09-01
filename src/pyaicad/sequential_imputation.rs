//! Python entry point for `crate::sampling::solve`'s sequential-imputation solver: repeatedly
//! builds a full assignment from the network's own probabilities combined with the problem's
//! compiled MDDs, re-running the network on each attempt's result, until one attempt satisfies the
//! problem, `max_steps` is exhausted, or `time_limit` elapses. See `crate::sampling::solve`'s
//! module doc for what this is and isn't (a sequential-imputation Monte Carlo method, not Gibbs
//! sampling, no importance weighting).
//!
//! Deliberately separate from `pyaicad::nls`: this isn't a destroy/decode neural local search --
//! it owns its own step loop instead of plugging into `NeuralLocalSearch`/`DecodingOperator`.
//!
//! Problems are solved in batches of up to `batch_size` at once (default: every problem in one
//! batch): every still-unsolved problem in a batch shares one network forward pass per step (via
//! `sampling::solve::SequentialImputationSolver`, which owns the network and all tensor bookkeeping
//! -- this module only ever deals in `Problem`s and `Solution`s), and each batch's own per-row
//! sequential-imputation attempts run in parallel across CPU cores. A problem that solves early
//! drops out of later steps' forward passes for the rest of its batch, so an easy problem doesn't
//! keep paying for a hard batchmate's remaining steps.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::backend::Backend;

use rayon::prelude::*;

use crate::learning::consformer::{
    ConsFormer, ConsFormerBatch, ConsFormerConfig, MddCompilationConfig,
};
use crate::mdd::heuristics::ConstraintGrouping;
use crate::mdd::Mdd;
use crate::modelling::{Problem, VariableIndex};
use crate::nls::{load_network, Solution, Status};
use crate::sampling::solve::SequentialImputationSolver;
use crate::sampling::{DecodeMode, DestroyRule, MddSampler};

use super::learn::cuda_available;
use super::nls::{PyNetworkKind, PyProblemsArg, PySolution};

/// Which of `sampling::DestroyRule`'s variants to use -- see that type's doc.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyDestroyRule {
    Deterministic,
    Probabilistic,
}

impl From<&PyDestroyRule> for DestroyRule {
    fn from(rule: &PyDestroyRule) -> Self {
        match rule {
            PyDestroyRule::Deterministic => DestroyRule::Deterministic,
            PyDestroyRule::Probabilistic => DestroyRule::Probabilistic,
        }
    }
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PySequentialImputationConfig {
    #[pyo3(get, set)]
    pub network_kind: PyNetworkKind,
    /// Which rule decides whether an MDD's scope gets resampled each step -- see `DestroyRule`.
    #[pyo3(get, set)]
    pub destroy_rule: PyDestroyRule,
    /// K in the sequential-imputation loop: how many full-assignment attempts a still-unsolved
    /// problem gets before giving up on it.
    #[pyo3(get, set)]
    pub max_steps: usize,
    /// If true, sample each variable's value from its combined conditional; if false, take the
    /// most likely value (greedy).
    #[pyo3(get, set)]
    pub stochastic_decode: bool,
    /// How the problem's constraints are grouped into MDDs before compilation -- 0 means one MDD
    /// per constraint; see `EliminationOrdering::buckets`'s doc for what a larger bound buys.
    #[pyo3(get, set)]
    pub mdd_grouping_size_bound: usize,
    /// How many problems are ever loaded onto the device and stepped together at once. Left unset,
    /// every problem is batched together in a single run. Problems are still solved independently
    /// within a batch -- a solved one drops out of later steps' network calls -- this only bounds
    /// how many are ever live in memory/on the device simultaneously.
    #[pyo3(get, set)]
    pub batch_size: Option<usize>,
    /// If present, caps how long *each batch* gets, in seconds -- matching the classical-CP
    /// convention of a private timeout per instance, every batch gets its own full budget rather
    /// than sharing one across the whole problem list. A problem still unsolved when its batch's
    /// deadline passes is reported as `Unknown`, not an error.
    #[pyo3(get, set)]
    pub time_limit: Option<u64>,
}

#[pymethods]
impl PySequentialImputationConfig {
    #[new]
    #[pyo3(signature = (
        network_kind=PyNetworkKind::ConsFormer,
        destroy_rule=PyDestroyRule::Probabilistic,
        max_steps=50,
        stochastic_decode=false,
        mdd_grouping_size_bound=0,
        batch_size=None,
        time_limit=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        network_kind: PyNetworkKind,
        destroy_rule: PyDestroyRule,
        max_steps: usize,
        stochastic_decode: bool,
        mdd_grouping_size_bound: usize,
        batch_size: Option<usize>,
        time_limit: Option<u64>,
    ) -> Self {
        Self {
            network_kind,
            destroy_rule,
            max_steps,
            stochastic_decode,
            mdd_grouping_size_bound,
            batch_size,
            time_limit,
        }
    }
}

/// Compiles one MDD per group `compilation.grouping` puts the problem's constraints into,
/// refining each to full exactness. Mirrors what the old `MddGibbsDecoding` used to do at decode
/// time.
fn compile_problem_mdds(problem: &Arc<Problem>, compilation: &MddCompilationConfig) -> Vec<Mdd> {
    compilation
        .grouping
        .groups(problem)
        .into_iter()
        .map(|constraints| {
            let mut mdd = Mdd::new(
                Arc::clone(problem),
                compilation.ordering.clone(),
                compilation.merge,
                compilation.select,
                &constraints,
            );
            mdd.refine(usize::MAX);
            if mdd.is_unsat() {
                let label = constraints
                    .iter()
                    .map(|&c| problem[c].name())
                    .collect::<Vec<_>>()
                    .join(", ");
                log::warn!(
                    "MDD group `{label}` is unsatisfiable given its own scope's domains -- its \
                     compiled MDD has no accepting path, so the sequential-imputation solver will \
                     always abstain (uniform conditional) for it. This usually means a fixed/hint \
                     value already violates one of its constraints."
                );
            }
            mdd
        })
        .collect()
}

/// Runs the sequential-imputation solver on one batch of same-size problems, sharing one network
/// forward pass per step across every problem still unsolved in the batch.
fn solve_batch_chunk<B: Backend>(
    solver: &SequentialImputationSolver<B, ConsFormer<B>, ConsFormerBatch<B>>,
    chunk: &[Arc<Problem>],
    compilation: &MddCompilationConfig,
    mode: DecodeMode,
    max_steps: usize,
    time_limit: Option<Duration>,
) -> Vec<Solution> {
    let n = chunk[0].number_variables();
    let start = Instant::now();

    // MDD compilation is the expensive, embarrassingly parallel, per-problem one-time cost --
    // spread it across cores rather than doing it one problem at a time.
    let mdds_per_problem: Vec<Vec<Mdd>> = chunk
        .par_iter()
        .map(|problem| compile_problem_mdds(problem, compilation))
        .collect();
    let samplers: Vec<MddSampler> = mdds_per_problem
        .iter()
        .map(|mdds| MddSampler::new(mdds))
        .collect();

    let results = solver.run(chunk, &samplers, mode, max_steps, time_limit);

    log::info!(
        "sequential-imputation batch of {} problems: {}/{} solved in {}s",
        chunk.len(),
        results.iter().filter(|r| r.satisfied).count(),
        chunk.len(),
        start.elapsed().as_secs(),
    );

    results
        .into_iter()
        .zip(chunk)
        .map(|(result, problem)| {
            let runtime = result.elapsed.as_secs();
            if result.satisfied {
                let raw: Vec<isize> = (0..n)
                    .map(|v| problem[VariableIndex(v)].value(result.assignment[v]))
                    .collect();
                Solution {
                    runtime,
                    iterations: result.steps,
                    solution: Some(raw),
                    status: Status::Satisfiable,
                }
            } else {
                Solution {
                    runtime,
                    iterations: result.steps,
                    solution: None,
                    status: Status::Unknown,
                }
            }
        })
        .collect()
}

fn run<B: Backend>(
    device: B::Device,
    problems: Vec<Arc<Problem>>,
    checkpoint_dir: &Path,
    config: &PySequentialImputationConfig,
) -> PyResult<Vec<Solution>> {
    match &config.network_kind {
        PyNetworkKind::ConsFormer => {
            let (network_config, network) =
                load_network::<B, ConsFormerConfig>(checkpoint_dir, &problems, &device).map_err(
                    |e| {
                        PyRuntimeError::new_err(format!(
                            "failed to load network from checkpoint {}: {e}",
                            checkpoint_dir.display()
                        ))
                    },
                )?;

            let mode = if config.stochastic_decode {
                DecodeMode::Sample
            } else {
                DecodeMode::Greedy
            };
            let compilation = MddCompilationConfig {
                grouping: ConstraintGrouping {
                    size_bound: config.mdd_grouping_size_bound,
                    ..ConstraintGrouping::PER_CONSTRAINT
                },
                ..MddCompilationConfig::default()
            };
            let time_limit = config.time_limit.map(Duration::from_secs);
            let chunk_size = config.batch_size.unwrap_or(problems.len()).max(1);

            let solver = SequentialImputationSolver::<B, ConsFormer<B>, ConsFormerBatch<B>>::new(
                network,
                network_config.domain_size,
                DestroyRule::from(&config.destroy_rule),
                device,
            );

            log::info!(
                "Solving {} problems by batch of size {}",
                problems.len(),
                chunk_size
            );
            let mut solutions = Vec::with_capacity(problems.len());
            for (chunk_idx, chunk) in problems.chunks(chunk_size).enumerate() {
                log::info!("Solving batch {chunk_idx}");
                solutions.extend(solve_batch_chunk::<B>(
                    &solver,
                    chunk,
                    &compilation,
                    mode,
                    config.max_steps,
                    time_limit,
                ));
            }
            Ok(solutions)
        }
    }
}

/// Solves `problems` (a single `Problem` or a list of them) with the sequential-imputation
/// procedure in `crate::sampling::solve`, using a network loaded from `checkpoint_dir` (the
/// `config.json` + `weights` produced by `train_consformer`/`train_consformer_mdd`). Returns a
/// single `PySolution` for a single problem, or a list of them (in the same order as `problems`)
/// for a list. Every problem must have the same number of variables.
///
/// Problems are grouped into batches of `config.batch_size` (default: all of them in one batch)
/// and, within a batch, solved together -- one shared network forward pass per step across every
/// problem still unsolved in that batch, with each problem's own per-step sequential-imputation
/// attempt computed in parallel across CPU cores. This is meant for trying the solver out across
/// many instances at once (to see where it actually struggles, not just whether it works on one),
/// not for squeezing out maximum throughput.
///
/// `config` gathers every knob unrelated to `problems`/`checkpoint_dir` (see
/// `PySequentialImputationConfig`); left unset, it's `PySequentialImputationConfig()` --
/// `max_steps=50`, greedy decoding, one MDD per constraint, one batch, no time limit.
#[pyfunction]
#[pyo3(signature = (problems, checkpoint_dir, config=None))]
pub fn sequential_imputation_solve(
    py: Python<'_>,
    problems: PyProblemsArg<'_>,
    checkpoint_dir: String,
    config: Option<PySequentialImputationConfig>,
) -> PyResult<Py<PyAny>> {
    let is_single = matches!(problems, PyProblemsArg::Single(_));
    let problems: Vec<Arc<Problem>> = match problems {
        PyProblemsArg::Single(p) => vec![p.arc()],
        PyProblemsArg::Many(ps) => ps.iter().map(|p| p.arc()).collect(),
    };
    if problems.is_empty() {
        return Err(PyValueError::new_err(
            "sequential_imputation_solve: `problems` must be non-empty",
        ));
    }
    let n = problems[0].number_variables();
    if problems.iter().any(|p| p.number_variables() != n) {
        return Err(PyValueError::new_err(
            "sequential_imputation_solve: all problems in a batch must have the same number of \
             variables",
        ));
    }

    let checkpoint_dir = PathBuf::from(checkpoint_dir);
    let config = config.unwrap_or_else(|| {
        PySequentialImputationConfig::new(
            PyNetworkKind::ConsFormer,
            PyDestroyRule::Probabilistic,
            50,
            false,
            0,
            None,
            None,
        )
    });

    let solutions = py.detach(move || {
        if cuda_available() {
            run::<Cuda>(CudaDevice::default(), problems, &checkpoint_dir, &config)
        } else {
            run::<NdArray>(NdArrayDevice::default(), problems, &checkpoint_dir, &config)
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
