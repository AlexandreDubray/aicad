use pyo3::prelude::*;

mod compiler;
mod heuristics;
mod learn;
mod logging;
mod nls;
mod problem;

pub use compiler::{Compiler, PyMdd};
pub use heuristics::{PyMergeHeuristic, PyOrderingHeuristic, PySelectHeuristic};
pub use learn::{train_consformer, train_consformer_mdd, PyConsFormerConfig, PyTrainingConfig};
pub use logging::{
    enable_console_logging, set_verbosity_debug, set_verbosity_error, set_verbosity_info,
    set_verbosity_off, set_verbosity_trace, set_verbosity_warning,
};
pub use nls::{
    neural_local_search, PyDecodeKind, PyDestroyKind, PyNetworkKind, PySolution, PyStatus,
};
pub use problem::PyProblem;

#[pymodule]
fn pyaicad(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
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

    logging::enable_console_logging(py)?;
    // Logging is opt-in: silent until a `set_verbosity_*` function below is
    // called from Python, like a CLI tool with no `-v` passed.
    logging::disable_by_default(py)?;

    m.add_class::<PyProblem>()?;
    m.add_class::<Compiler>()?;
    m.add_class::<PyMdd>()?;
    m.add_class::<PyOrderingHeuristic>()?;
    m.add_class::<PyMergeHeuristic>()?;
    m.add_class::<PySelectHeuristic>()?;
    m.add_class::<PyConsFormerConfig>()?;
    m.add_class::<PyTrainingConfig>()?;
    m.add_class::<PyNetworkKind>()?;
    m.add_class::<PyDestroyKind>()?;
    m.add_class::<PyDecodeKind>()?;
    m.add_class::<PySolution>()?;
    m.add_class::<PyStatus>()?;
    m.add_function(wrap_pyfunction!(train_consformer, m)?)?;
    m.add_function(wrap_pyfunction!(train_consformer_mdd, m)?)?;
    m.add_function(wrap_pyfunction!(neural_local_search, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_off, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_error, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_warning, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_info, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_debug, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_trace, m)?)?;
    m.add_function(wrap_pyfunction!(enable_console_logging, m)?)?;
    Ok(())
}
