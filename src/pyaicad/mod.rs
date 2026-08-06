use pyo3::prelude::*;

mod heuristics;
mod learn;
mod problem;
mod solver;

pub use heuristics::{PyMergeHeuristic, PyOrderingHeuristic, PySelectHeuristic};
pub use learn::{train_consformer, PyConsFormerConfig, PyTrainingConfig};
pub use problem::PyProblem;
pub use solver::Solver;

#[pymodule]
fn pyaicad(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProblem>()?;
    m.add_class::<Solver>()?;
    m.add_class::<PyOrderingHeuristic>()?;
    m.add_class::<PyMergeHeuristic>()?;
    m.add_class::<PySelectHeuristic>()?;
    m.add_class::<PyConsFormerConfig>()?;
    m.add_class::<PyTrainingConfig>()?;
    m.add_function(wrap_pyfunction!(train_consformer, m)?)?;
    Ok(())
}
