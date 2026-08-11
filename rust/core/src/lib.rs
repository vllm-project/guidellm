use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
fn version_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("git_sha", env!("VERGEN_GIT_SHA"))?;
    dict.set_item("git_branch", env!("VERGEN_GIT_BRANCH"))?;
    dict.set_item("git_describe", env!("VERGEN_GIT_DESCRIBE"))?;
    Ok(dict.into())
}

#[pymodule(name = "_rust")]
fn guidellm_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version_info, m)?)?;
    Ok(())
}
