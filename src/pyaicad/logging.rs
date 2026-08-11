use pyo3::prelude::*;

const LOGGER_NAME: &str = "aicad";

// Mirrors Python's `logging` module level constants (`logging.DEBUG == 10`,
// etc.), so we don't need to import/call into `logging` just to read them.
const LEVEL_TRACE: i32 = 5;
const LEVEL_DEBUG: i32 = 10;
const LEVEL_INFO: i32 = 20;
const LEVEL_WARNING: i32 = 30;
const LEVEL_ERROR: i32 = 40;
const LEVEL_OFF: i32 = 60;

fn set_level(py: Python<'_>, level: i32) -> PyResult<()> {
    let logging = py.import("logging")?;
    let logger = logging.call_method1("getLogger", (LOGGER_NAME,))?;
    logger.call_method1("setLevel", (level,))?;
    Ok(())
}

/// Called once from the module's `#[pymodule]` init so logging starts out
/// silent, matching a CLI tool with no `-v` passed.
pub(crate) fn disable_by_default(py: Python<'_>) -> PyResult<()> {
    set_level(py, LEVEL_OFF)
}

/// Disables all logging from this crate. Equivalent to the default state
/// before any `set_verbosity_*` call.
#[pyfunction]
pub fn set_verbosity_off(py: Python<'_>) -> PyResult<()> {
    set_level(py, LEVEL_OFF)
}

/// Only `log::error!` records are emitted.
#[pyfunction]
pub fn set_verbosity_error(py: Python<'_>) -> PyResult<()> {
    set_level(py, LEVEL_ERROR)
}

/// `log::warn!` and above.
#[pyfunction]
pub fn set_verbosity_warning(py: Python<'_>) -> PyResult<()> {
    set_level(py, LEVEL_WARNING)
}

/// `log::info!` and above.
#[pyfunction]
pub fn set_verbosity_info(py: Python<'_>) -> PyResult<()> {
    set_level(py, LEVEL_INFO)
}

/// `log::debug!` and above.
#[pyfunction]
pub fn set_verbosity_debug(py: Python<'_>) -> PyResult<()> {
    set_level(py, LEVEL_DEBUG)
}

/// Everything, including `log::trace!`.
#[pyfunction]
pub fn set_verbosity_trace(py: Python<'_>) -> PyResult<()> {
    set_level(py, LEVEL_TRACE)
}

#[pyfunction]
pub fn enable_console_logging(py: Python<'_>) -> PyResult<()> {
    py.import("logging")?.call_method0("basicConfig")?;
    Ok(())
}
