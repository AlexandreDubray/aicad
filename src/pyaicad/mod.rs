use pyo3::prelude::*;

mod build;
mod compiler;
mod heuristics;
#[cfg(feature = "learning")]
mod learn;
mod local_search;
mod logging;
#[cfg(feature = "learning")]
mod optimiser;
mod problem;
mod threads;

pub(crate) use build::build_info;
pub(crate) use compiler::{PyCompiler, PyMdd};
pub(crate) use heuristics::{PyMergeHeuristic, PyOrderingHeuristic, PySelectHeuristic};
#[cfg(feature = "learning")]
pub(crate) use learn::{
    PyConsFormerConfig, PyTrainingConfig, train_consformer, train_consformer_mdd,
};
pub(crate) use local_search::{
    PyDestroyKind, PySearch, PySolution, PySolveConfig, PyStatus, run_local_search,
};
pub(crate) use logging::{
    disable_data_log, enable_data_log, set_verbosity_debug, set_verbosity_error,
    set_verbosity_info, set_verbosity_off, set_verbosity_trace, set_verbosity_warning,
};
#[cfg(feature = "learning")]
pub(crate) use optimiser::PyOptimiser;
pub(crate) use problem::PyProblem;
pub(crate) use threads::{get_num_threads, set_num_threads, set_seed};

#[pymodule]
fn _core(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Scope pyo3-log to *this* crate's own log records only. The blanket
    // `pyo3_log::init()` forwards every crate's log records -- including
    // internal `log::*!` calls from dependencies (burn, its rayon-backed
    // ndarray backend, etc.) -- to Python, which means those can fire from
    // worker threads pyo3 doesn't manage, each one calling `Python::attach`
    // to acquire the GIL from a thread it was never meant to be acquired
    // from. `filter(Off)` + `filter_target("aicad", Trace)` means anything
    // not under our own "aicad"/"aicad::..." targets is dropped by the
    // cheap Rust-side filter before it ever reaches that GIL-acquiring code
    // path.
    pyo3_log::Logger::new(py, pyo3_log::Caching::LoggersAndLevels)
        .unwrap()
        .filter(log::LevelFilter::Off)
        .filter_target("aicad".to_owned(), log::LevelFilter::Trace)
        .install()
        .expect("pyo3-log logger already installed");

    // Logging is opt-in: silent until a `set_verbosity_*` function below is
    // called from Python, like a CLI tool with no `-v` passed.
    logging::disable_by_default(py)?;

    m.add_class::<PyProblem>()?;
    m.add_class::<PyCompiler>()?;
    m.add_class::<PyMdd>()?;
    m.add_class::<PyOrderingHeuristic>()?;
    m.add_class::<PyMergeHeuristic>()?;
    m.add_class::<PySelectHeuristic>()?;
    m.add_class::<PyDestroyKind>()?;
    m.add_class::<PySearch>()?;
    m.add_class::<PySolveConfig>()?;
    m.add_class::<PySolution>()?;
    m.add_class::<PyStatus>()?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_function(wrap_pyfunction!(set_seed, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(run_local_search, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_off, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_error, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_warning, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_info, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_debug, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_trace, m)?)?;
    m.add_function(wrap_pyfunction!(enable_data_log, m)?)?;
    m.add_function(wrap_pyfunction!(disable_data_log, m)?)?;

    // Whether this build ships the training and the neural searches (the `learning` feature); the
    // Python package reads it to refuse them with a clear message instead of failing on a missing
    // attribute.
    m.add("HAS_LEARNING", cfg!(feature = "learning"))?;
    #[cfg(feature = "learning")]
    {
        m.add_class::<PyConsFormerConfig>()?;
        m.add_class::<PyOptimiser>()?;
        m.add_class::<PyTrainingConfig>()?;
        m.add_function(wrap_pyfunction!(train_consformer, m)?)?;
        m.add_function(wrap_pyfunction!(train_consformer_mdd, m)?)?;
    }
    Ok(())
}
