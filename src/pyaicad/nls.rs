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

use indicatif::{ProgressBar, ProgressStyle};
use rand::RngExt;
use rayon::prelude::*;

use crate::learning::consformer::{
    ConsFormer, ConsFormerBatch, ConsFormerConfig, MddCompilationConfig,
};
use crate::learning::Network;
use crate::modelling::Problem;
use crate::nls::decode::{Argmax, DecodingOperator, MddSamplingDecode, Sampling};
use crate::nls::destroy::{DestroyOperator, RandomDestroy, RelatedDestroy, WorstDestroy};
use crate::nls::{load_network, Budget, NeuralLocalSearch, Solution, SolveConfig, Status};
use crate::sampling::DecodeMode;

use super::learn::cuda_available;
use super::problem::PyProblem;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PySolution {
    #[pyo3(get)]
    runtime: u64,
    /// Seconds spent compiling MDDs (`decode_kind=MddSampling`'s `MddSamplingDecode::prepare`)
    /// for the `run` call this solution came out of -- 0.0 for `decode_kind=Logits`, which has no
    /// compilation step. Already included in `runtime`, not additional to it -- see
    /// `nls::Solution::compilation_runtime`'s doc.
    #[pyo3(get)]
    compilation_runtime: f64,
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
            compilation_runtime: s.compilation_runtime(),
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
    fn build(&self, fraction: f64) -> Box<dyn DestroyOperator> {
        match self {
            PyDestroyKind::Random => Box::new(RandomDestroy { fraction }),
            PyDestroyKind::Worst => Box::new(WorstDestroy { fraction }),
            PyDestroyKind::Related => Box::new(RelatedDestroy { fraction }),
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

/// Which of `nls::decode`'s operators to build -- see `SolveConfig::decode_kind`'s doc.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub enum PyDecodeKind {
    Logits,
    MddSampling,
}

impl PyDecodeKind {
    fn tag(&self) -> &'static str {
        match self {
            PyDecodeKind::Logits => "logits",
            PyDecodeKind::MddSampling => "mdd_sampling",
        }
    }

    fn parse(tag: &str) -> PyResult<Self> {
        match tag {
            "logits" => Ok(PyDecodeKind::Logits),
            "mdd_sampling" => Ok(PyDecodeKind::MddSampling),
            other => Err(PyValueError::new_err(format!(
                "unknown decode_kind {other:?}"
            ))),
        }
    }
}

/// Builds the decode operator for neural local search. `decode_kind == "logits"` picks greedy
/// (`Argmax`) vs. stochastic (`Sampling`) decoding of the network's raw logits, per
/// `stochastic_decode`; `decode_kind == "mdd_sampling"` instead refines those same logits through
/// one round of `sampling::bp::belief_propagation` over the problem's compiled MDDs before
/// decoding -- see `MddSamplingDecode`'s doc.
fn build_decode_op<B: Backend>(
    decode_kind: &PyDecodeKind,
    stochastic_decode: bool,
    temperature: f64,
    bp_iterations: usize,
    domain_size: usize,
) -> Box<dyn DecodingOperator<B>> {
    match decode_kind {
        PyDecodeKind::Logits => {
            if stochastic_decode {
                Box::new(Sampling { temperature })
            } else {
                Box::new(Argmax)
            }
        }
        PyDecodeKind::MddSampling => {
            let compilation = MddCompilationConfig::default();
            let mode = if stochastic_decode {
                DecodeMode::Sample
            } else {
                DecodeMode::Greedy
            };
            Box::new(MddSamplingDecode::new(
                compilation,
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
    pub destroy_fraction: f64,
    #[pyo3(get, set)]
    pub stochastic_decode: bool,
    #[pyo3(get, set)]
    pub temperature: f64,
    /// Which decode operator to use
    #[pyo3(get, set)]
    pub decode_kind: PyDecodeKind,
    /// Number of loopy belief propagation rounds `MddSamplingDecode` runs over the problem's
    /// compiled MDDs before decoding -- see `belief_propagation`'s doc.
    #[pyo3(get, set)]
    pub bp_iterations: usize,
    /// Upper bound on how many `batch_size`-sized chunks may run concurrently -- a separate knob
    /// from the CPU worker pool's thread count, since GPU memory (not CPU cores) is the resource
    /// this bounds. Default `1`: fully sequential over chunks, matching the pre-existing
    /// behaviour. Raise it only when `batch_size` is small enough that several chunks resident on
    /// the device at once is safe -- e.g. `batch_size=1` with `max_concurrent_chunks` set to
    /// however many instances you're comfortable holding in device memory simultaneously.
    #[pyo3(get, set)]
    pub max_concurrent_chunks: usize,
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
        max_concurrent_chunks=1,
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
        max_concurrent_chunks: usize,
        time_limit: Option<u64>,
        iteration_limit: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        Ok(PySolveConfig {
            network_kind,
            batch_size,
            destroy_kind,
            destroy_fraction,
            stochastic_decode,
            temperature,
            decode_kind,
            bp_iterations,
            max_concurrent_chunks,
            time_limit,
            iteration_limit,
            seed,
        })
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
            destroy_fraction: c.destroy_fraction,
            stochastic_decode: c.stochastic_decode,
            temperature: c.temperature,
            decode_kind: c.decode_kind.tag().to_string(),
            bp_iterations: c.bp_iterations,
            max_concurrent_chunks: c.max_concurrent_chunks,
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
            destroy_fraction: c.destroy_fraction,
            stochastic_decode: c.stochastic_decode,
            temperature: c.temperature,
            decode_kind: PyDecodeKind::parse(&c.decode_kind)?,
            bp_iterations: c.bp_iterations,
            max_concurrent_chunks: c.max_concurrent_chunks,
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
            let (network_config, network) =
                load_network::<B, ConsFormerConfig>(checkpoint_dir, &problems, &device).map_err(
                    |e| {
                        PyRuntimeError::new_err(format!(
                            "failed to load network from checkpoint {}: {e}",
                            checkpoint_dir.display()
                        ))
                    },
                )?;
            let destroy_op = destroy_kind.build(config.destroy_fraction);
            let decode_kind = PyDecodeKind::parse(&config.decode_kind)?;
            // A builder, not a single shared instance: each chunk gets its own freshly-built
            // decode operator (see `chunked_run`) so `MddSamplingDecode`'s compiled-MDD cache
            // never carries structure over from one chunk to the next -- every chunk (an entire
            // instance of its own with `batch_size=1`, the convention a fair per-instance
            // benchmark uses) pays its own compilation cost from a cold cache, the same way a
            // single real deployment solving that one problem would.
            let stochastic_decode = config.stochastic_decode;
            let temperature = config.temperature;
            let bp_iterations = config.bp_iterations;
            let domain_size = network_config.domain_size;
            let build_decode_op_for_chunk = move || -> Box<dyn DecodingOperator<B>> {
                build_decode_op::<B>(
                    &decode_kind,
                    stochastic_decode,
                    temperature,
                    bp_iterations,
                    domain_size,
                )
            };

            let nls = NeuralLocalSearch::<B, ConsFormer<B>, ConsFormerBatch<B>>::new(
                network, destroy_op, device,
            );
            Ok(chunked_run(
                &nls,
                &build_decode_op_for_chunk,
                &problems,
                config.batch_size,
                config.max_concurrent_chunks,
                budget,
                seed,
            ))
        }
    }
}

/// Seeds are assigned by chunk *index*, not by completion order, so the destroy sequence for a
/// given problem stays reproducible regardless of how the scheduler happens to interleave chunks
/// within a group; `.collect()` likewise preserves chunk order in the returned `Vec` even though
/// chunks within a group may finish out of order.
///
/// `decode_op_builder` is called once per chunk (never shared across chunks) -- see the doc where
/// it's built in `run`. This is also why the decode operator can't be a field of `nls` the way
/// `network`/`destroy_op` are: those two have nothing to gain from per-chunk isolation (the
/// network's weights are read-only, and no destroy operator caches anything keyed by which
/// problem it last saw), but `MddSamplingDecode` does, and sharing `nls` itself across every
/// `par_iter` task already means whatever it owned would be shared too.
fn chunked_run<B: Backend, N: Network<B, Ba> + Sync, Ba: crate::learning::Batch<B>>(
    nls: &NeuralLocalSearch<B, N, Ba>,
    decode_op_builder: &(dyn Fn() -> Box<dyn DecodingOperator<B>> + Sync),
    problems: &[Arc<Problem>],
    batch_size: Option<usize>,
    max_concurrent_chunks: usize,
    budget: Budget,
    seed: u64,
) -> Vec<Solution> {
    let chunk_size = batch_size.unwrap_or(problems.len()).max(1);
    let max_concurrent_chunks = max_concurrent_chunks.max(1);
    let chunks: Vec<&[Arc<Problem>]> = problems.chunks(chunk_size).collect();
    log::info!(
        "Solving {} problems by chunk of size {} ({} chunks, at most {} running concurrently, worker pool has {} threads)",
        problems.len(),
        chunk_size,
        chunks.len(),
        max_concurrent_chunks,
        crate::utils::worker_pool().current_num_threads(),
    );

    // One bar for the whole call, ticked as each chunk finishes (not per-chunk bars -- see
    // `MddCache::prepare`'s doc for why several independent bars drawing concurrently, one per
    // in-flight chunk under `--max-concurrent-chunks`, would garble the terminal instead of
    // helping). `ProgressBar` is `Clone` + internally synchronized, so calling `.inc()` from
    // several rayon threads at once (one per concurrently-finishing chunk) is safe.
    let progress = ProgressBar::new(problems.len() as u64);
    progress.set_style(
        ProgressStyle::with_template(
            "{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )
        .expect("hard-coded progress bar template should always be valid"),
    );
    progress.set_message("Solving instances");

    let mut solutions = Vec::with_capacity(problems.len());
    for (group_idx, group) in chunks.chunks(max_concurrent_chunks).enumerate() {
        let group_start = group_idx * max_concurrent_chunks;
        let group_solutions: Vec<Vec<Solution>> = crate::utils::worker_pool().install(|| {
            group
                .par_iter()
                .enumerate()
                .map(|(local_idx, chunk)| {
                    let chunk_idx = group_start + local_idx;
                    log::debug!("Solving chunk {}", chunk_idx);
                    // Vary the seed per chunk so chunks don't replay the exact same destroy
                    // sequence -- keyed by chunk_idx (not completion order) so this stays
                    // reproducible under concurrent scheduling.
                    let chunk_seed = seed.wrapping_add(chunk_idx as u64);
                    let decode_op = decode_op_builder();
                    let chunk_start = std::time::Instant::now();
                    let chunk_solutions = nls.run(chunk, decode_op.as_ref(), budget, chunk_seed);
                    log_chunk_summary(chunk_idx, chunks.len(), &chunk_solutions, chunk_start.elapsed());
                    progress.inc(chunk_solutions.len() as u64);
                    chunk_solutions
                })
                .collect()
        });
        solutions.extend(group_solutions.into_iter().flatten());
    }
    progress.finish_and_clear();
    log_status_counts("Done", &solutions);

    solutions
}

/// One `info`-level line per finished chunk -- this, plus the progress bar, is what makes a
/// `--max-concurrent-chunks`-parallel sweep's overall progress visible without drowning in
/// `StoppingCriterion::log`'s per-100-iterations `debug` output (see that method's doc).
/// `elapsed` is the wall-clock time this specific chunk's `nls.run` call took, not summed across
/// `chunk_solutions` -- those may individually report a smaller `runtime` (whichever iteration
/// they were decided on), and summing them would double-count the shared search loop.
fn log_chunk_summary(chunk_idx: usize, total_chunks: usize, chunk_solutions: &[Solution], elapsed: Duration) {
    log_status_counts(
        &format!("chunk {}/{total_chunks} done in {:.1}s", chunk_idx + 1, elapsed.as_secs_f64()),
        chunk_solutions,
    );
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
