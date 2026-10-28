use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::config::Config;
use burn::tensor::backend::Backend;

use indicatif::{ProgressBar, ProgressStyle};
use rand::RngExt;

use crate::destroy::{DestroyKind, DestroyOperator};
use crate::learning::consformer::{ConsFormer, ConsFormerBatch, ConsFormerConfig};
use crate::local_search::decode::{Decoder, DecomposedMddSampling};
use crate::local_search::{
    Budget, DecodeKind, NetworkKind, NetworkPrior, NeuralLocalSearch, PriorHeuristic, PriorKind,
    SolveConfig, UniformPrior, load_network,
};
use crate::mdd::arena::MddCompilationConfig;
use crate::modelling::{Problem, Solution, Status};
use crate::utils::DecodeMode;

use super::learn::cuda_available;
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
pub enum PyNetworkKind {
    ConsFormer,
}

impl From<&PyNetworkKind> for NetworkKind {
    fn from(kind: &PyNetworkKind) -> Self {
        match kind {
            PyNetworkKind::ConsFormer => NetworkKind::ConsFormer,
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

/// Where each step's prior over values comes from -- see `local_search::prior::PriorHeuristic`.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyPriorKind {
    Network,
    Uniform,
}

impl From<&PyPriorKind> for PriorKind {
    fn from(kind: &PyPriorKind) -> Self {
        match kind {
            PyPriorKind::Network => PriorKind::Network,
            PyPriorKind::Uniform => PriorKind::Uniform,
        }
    }
}

/// Which decoding operator to build -- see `local_search::DecodeKind`.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyDecodeKind {
    Logits,
    MddSampling,
}

impl From<&PyDecodeKind> for DecodeKind {
    fn from(kind: &PyDecodeKind) -> Self {
        match kind {
            PyDecodeKind::Logits => DecodeKind::Logits,
            PyDecodeKind::MddSampling => DecodeKind::MddSampling,
        }
    }
}

/// Builds the decoder of neural local search. `DecodeKind::Logits` picks greedy (`Argmax`)
/// vs. stochastic (`Sampling`) decoding of the network's raw logits, per `stochastic_decode`;
/// `DecodeKind::MddSampling` instead refines those same logits through belief propagation
/// over the problem's compiled MDDs before decoding -- see `DecomposedMddSampling`'s doc.
fn build_decoder(
    decode_kind: DecodeKind,
    stochastic_decode: bool,
    temperature: f64,
    bp_iterations: usize,
    domain_size: usize,
) -> Decoder {
    match decode_kind {
        DecodeKind::Logits => {
            if stochastic_decode {
                Decoder::Sampling { temperature }
            } else {
                Decoder::Argmax
            }
        }
        DecodeKind::MddSampling => {
            let mode = if stochastic_decode {
                DecodeMode::Sample
            } else {
                DecodeMode::Greedy
            };
            Decoder::DecomposedMddSampling(DecomposedMddSampling::new(
                MddCompilationConfig::default(),
                domain_size,
                mode,
                bp_iterations,
            ))
        }
    }
}

/// Accepts either a single `PyProblem` or a list of them, so
/// `neural_local_search` can solve one problem or a whole batch through the
/// same entry point.
#[derive(FromPyObject)]
pub(crate) enum PyProblemsArg<'py> {
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
    pub destroy_fraction: f64,
    #[pyo3(get, set)]
    pub stochastic_decode: bool,
    #[pyo3(get, set)]
    pub temperature: f64,
    /// Which decode operator to use
    #[pyo3(get, set)]
    pub decode_kind: PyDecodeKind,
    /// Number of loopy belief propagation rounds `DecomposedMddSampling` runs over the problem's
    /// compiled MDDs before decoding -- see `belief_propagation`'s doc.
    #[pyo3(get, set)]
    pub bp_iterations: usize,
    /// Where each step's prior comes from: `Network` (the checkpoint's network) or `Uniform`
    /// (no network evaluated; needs `decode_kind=MddSampling`).
    #[pyo3(get, set)]
    pub prior_kind: PyPriorKind,
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
        destroy_fraction=1.0,
        stochastic_decode=false,
        temperature=1.0,
        decode_kind=PyDecodeKind::Logits,
        bp_iterations=1,
        prior_kind=PyPriorKind::Network,
        time_limit=None,
        iteration_limit=None,
        seed=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        network_kind: PyNetworkKind,
        batch_size: Option<usize>,
        destroy_kind: PyDestroyKind,
        destroy_fraction: f64,
        stochastic_decode: bool,
        temperature: f64,
        decode_kind: PyDecodeKind,
        bp_iterations: usize,
        prior_kind: PyPriorKind,
        time_limit: Option<u64>,
        iteration_limit: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        PySolveConfig {
            network_kind,
            batch_size,
            destroy_kind,
            destroy_fraction,
            stochastic_decode,
            temperature,
            decode_kind,
            bp_iterations,
            prior_kind,
            time_limit,
            iteration_limit,
            seed,
        }
    }
}

impl From<&PySolveConfig> for SolveConfig {
    fn from(c: &PySolveConfig) -> Self {
        SolveConfig {
            network_kind: (&c.network_kind).into(),
            batch_size: c.batch_size,
            destroy_kind: (&c.destroy_kind).into(),
            destroy_fraction: c.destroy_fraction,
            stochastic_decode: c.stochastic_decode,
            temperature: c.temperature,
            decode_kind: (&c.decode_kind).into(),
            bp_iterations: c.bp_iterations,
            prior_kind: (&c.prior_kind).into(),
            time_limit: c.time_limit,
            iteration_limit: c.iteration_limit,
            seed: c.seed,
        }
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
/// `batch_size` caps how many problems are stepped together on the device; the list is processed
/// in batches of at most that size, one after the other, reusing the same loaded network. Left
/// unset, every problem is in a single batch. `time_limit` and `iteration_limit` apply per
/// batch, matching the classical-CP convention of a private timeout per instance when
/// `batch_size=1`: `time_limit=10` gives each batch up to 10 seconds, not 10 seconds in total
/// across the call.
///
/// `checkpoint_dir` may be omitted only with `prior_kind=Uniform`, which evaluates no network at
/// all; if given alongside it, only its `config.json` is read (for the value alphabet).
///
/// `config` gathers every knob unrelated to `problems`/`checkpoint_dir` (see `PySolveConfig`);
/// left unset, it's `PySolveConfig()` -- today's zero-config defaults.
#[pyfunction]
#[pyo3(signature = (problems, checkpoint_dir=None, config=None))]
pub(crate) fn neural_local_search(
    py: Python<'_>,
    problems: PyProblemsArg<'_>,
    checkpoint_dir: Option<String>,
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

    let checkpoint_dir = checkpoint_dir.map(PathBuf::from);
    let config: SolveConfig = config.as_ref().map(SolveConfig::from).unwrap_or_default();
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
                checkpoint_dir.as_deref(),
                &config,
                budget,
                seed,
            )
        } else {
            run::<NdArray>(
                NdArrayDevice::default(),
                problems,
                checkpoint_dir.as_deref(),
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
    checkpoint_dir: Option<&Path>,
    config: &SolveConfig,
    budget: Budget,
    seed: u64,
) -> PyResult<Vec<Solution>> {
    let SolveConfig {
        network_kind,
        prior_kind,
        decode_kind,
        ..
    } = *config;

    if prior_kind == PriorKind::Uniform && decode_kind != DecodeKind::MddSampling {
        return Err(PyValueError::new_err(
            "prior_kind=uniform needs decode_kind=mdd_sampling: decoding flat logits directly \
             carries no information",
        ));
    }

    let load_error = |dir: &Path, e: &dyn std::fmt::Display| {
        PyRuntimeError::new_err(format!(
            "failed to load network from checkpoint {}: {e}",
            dir.display()
        ))
    };

    let (prior, domain_size): (Box<dyn PriorHeuristic<B>>, usize) = match network_kind {
        NetworkKind::ConsFormer => match prior_kind {
            PriorKind::Network => {
                let dir = checkpoint_dir.ok_or_else(|| {
                    PyValueError::new_err("prior_kind=network needs a checkpoint_dir")
                })?;
                let (network_config, network) =
                    load_network::<B, ConsFormerConfig>(dir, &problems, &device)
                        .map_err(|e| load_error(dir, &e))?;
                (
                    Box::new(NetworkPrior::<ConsFormer<B>, ConsFormerBatch<B>>::new(
                        network,
                    )),
                    network_config.domain_size,
                )
            }
            PriorKind::Uniform => {
                // The alphabet the MDDs are compiled over has to match the one the network
                // variant would use, so take it from the checkpoint's config when one is given
                // (its weights are not loaded); otherwise cover the problems' own values.
                let domain_size = match checkpoint_dir {
                    Some(dir) => {
                        ConsFormerConfig::load(dir.join("config.json"))
                            .map_err(|e| load_error(dir, &e))?
                            .domain_size
                    }
                    None => UniformPrior::domain_size_covering(&problems),
                };
                (Box::new(UniformPrior::new(domain_size)), domain_size)
            }
        },
    };

    let destroy_op = config.destroy_kind.build(config.destroy_fraction);
    let decoder = build_decoder(
        decode_kind,
        config.stochastic_decode,
        config.temperature,
        config.bp_iterations,
        domain_size,
    );

    let nls = NeuralLocalSearch::<B>::new(prior, destroy_op, decoder, device);
    Ok(batched_run(
        &nls,
        &problems,
        config.batch_size,
        budget,
        seed,
    ))
}

/// Solves `problems` in batches of at most `batch_size` (all of them in one batch if unset), one
/// batch after the other. The seed of a batch is `seed` plus its index, so that the batches do not
/// replay the same destroy sequence. Every batch gets the full `budget`.
fn batched_run<B: Backend>(
    nls: &NeuralLocalSearch<B>,
    problems: &[Arc<Problem>],
    batch_size: Option<usize>,
    budget: Budget,
    seed: u64,
) -> Vec<Solution> {
    let batch_size = batch_size.unwrap_or(problems.len()).max(1);
    let batches: Vec<&[Arc<Problem>]> = problems.chunks(batch_size).collect();
    log::info!(
        "Solving {} problems in {} batches of at most {}",
        problems.len(),
        batches.len(),
        batch_size,
    );

    let progress = ProgressBar::new(problems.len() as u64);
    progress.set_style(
        ProgressStyle::with_template(
            "{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )
        .expect("hard-coded progress bar template should always be valid"),
    );
    progress.set_message("Solving instances");

    let mut solutions = Vec::with_capacity(problems.len());
    for (batch_idx, batch) in batches.iter().enumerate() {
        let batch_start = Instant::now();
        let batch_solutions = nls.run(
            batch,
            budget,
            seed.wrapping_add(batch_idx as u64),
            Some(&progress),
        );
        log_status_counts(
            &format!(
                "batch {}/{} done in {:.1}s",
                batch_idx + 1,
                batches.len(),
                batch_start.elapsed().as_secs_f64()
            ),
            &batch_solutions,
        );
        // `run` already advanced the bar for every problem it solved; the ones it gave up on are
        // only known now.
        let unsolved = batch_solutions
            .iter()
            .filter(|s| matches!(s.status(), Status::Unknown))
            .count();
        progress.inc(unsolved as u64);
        solutions.extend(batch_solutions);
    }
    progress.finish_and_clear();
    log_status_counts("Done", &solutions);

    solutions
}

fn log_status_counts(label: &str, solutions: &[Solution]) {
    let sat = solutions
        .iter()
        .filter(|s| matches!(s.status(), Status::Satisfiable))
        .count();
    let unsat = solutions
        .iter()
        .filter(|s| matches!(s.status(), Status::Unsatisfiable))
        .count();
    let unknown = solutions.len() - sat - unsat;
    log::info!(
        "{label}: {sat} sat, {unsat} unsat, {unknown} unknown/timed out (of {})",
        solutions.len(),
    );
}
