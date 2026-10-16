//! The Python functions that set and read the number of threads of the worker pool.

use pyo3::prelude::*;

/// Sets the number of threads of the worker pool (see [`crate::utils::set_num_threads`]). The
/// Python wrapper has already checked that `n` is a positive integer.
#[pyfunction]
pub(crate) fn set_num_threads(n: usize) {
    crate::utils::set_num_threads(n);
}

/// The number of threads of the worker pool: the value last set, or the default.
#[pyfunction]
pub(crate) fn get_num_threads() -> usize {
    crate::utils::num_threads()
}
