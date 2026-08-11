use pyo3::prelude::*;

#[pyfunction]
fn hello_world() -> &'static str {
    "Hello from GuideLLM Rust!"
}

#[pymodule(name = "_rust")]
fn guidellm_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_world, m)?)?;
    Ok(())
}
