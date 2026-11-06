//! The Python function that reports which build of the library is running.

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The fields of [`crate::build_info::BuildInfo`] as a dict; the Python wrapper turns it into a
/// `BuildInfo` object. An unknown commit is `None`.
#[pyfunction]
pub(crate) fn build_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let info = crate::build_info::version();
    let dict = PyDict::new(py);
    dict.set_item("version", info.version)?;
    dict.set_item("commit", info.commit)?;
    dict.set_item("rustc", info.rustc)?;
    dict.set_item("target", info.target)?;
    dict.set_item("profile", info.profile)?;
    dict.set_item("opt_level", info.opt_level)?;
    dict.set_item("features", info.features.to_vec())?;
    Ok(dict)
}
