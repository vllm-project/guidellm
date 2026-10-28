//! Python bindings for GuideLLM's Rust libraries.

use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
fn hello_world() -> &'static str {
    guidellm_core::hello_world()
}

#[pyfunction]
fn version_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("git_sha", option_env!("VERGEN_GIT_SHA").unwrap_or("unknown"))?;
    dict.set_item(
        "git_branch",
        option_env!("VERGEN_GIT_BRANCH").unwrap_or("unknown"),
    )?;
    dict.set_item(
        "git_describe",
        option_env!("VERGEN_GIT_DESCRIBE").unwrap_or("unknown"),
    )?;
    Ok(dict.into())
}

#[pymodule(name = "_rust")]
fn guidellm_connector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_world, m)?)?;
    m.add_function(wrap_pyfunction!(version_info, m)?)?;
    Ok(())
}
