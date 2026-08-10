use pyo3::prelude::*;

mod heuristics;
mod learn;
mod logging;
mod nls;
mod problem;
mod solver;

pub use heuristics::{PyMergeHeuristic, PyOrderingHeuristic, PySelectHeuristic};
pub use learn::{train_consformer, PyConsFormerConfig, PyPositionalEncoding, PyTrainingConfig};
pub use logging::{
    set_verbosity_debug, set_verbosity_error, set_verbosity_info, set_verbosity_off,
    set_verbosity_trace, set_verbosity_warning,
};
pub use nls::{neural_local_search, PyDecodeKind, PyDestroyKind, PyNetworkKind, PySolution};
pub use problem::PyProblem;
pub use solver::Solver;

#[pymodule]
fn pyaicad(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();

    logging::disable_by_default(py)?;

    m.add_class::<PyProblem>()?;
    m.add_class::<Solver>()?;
    m.add_class::<PyOrderingHeuristic>()?;
    m.add_class::<PyMergeHeuristic>()?;
    m.add_class::<PySelectHeuristic>()?;
    m.add_class::<PyConsFormerConfig>()?;
    m.add_class::<PyPositionalEncoding>()?;
    m.add_class::<PyTrainingConfig>()?;
    m.add_class::<PyNetworkKind>()?;
    m.add_class::<PyDestroyKind>()?;
    m.add_class::<PyDecodeKind>()?;
    m.add_class::<PySolution>()?;
    m.add_function(wrap_pyfunction!(train_consformer, m)?)?;
    m.add_function(wrap_pyfunction!(neural_local_search, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_off, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_error, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_warning, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_info, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_debug, m)?)?;
    m.add_function(wrap_pyfunction!(set_verbosity_trace, m)?)?;
    Ok(())
}
