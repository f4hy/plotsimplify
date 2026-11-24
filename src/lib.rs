pub mod core;

use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn check_len(a: usize, b: usize) -> PyResult<()> {
    if a != b {
        return Err(PyValueError::new_err("arrays must have the same length"));
    }
    Ok(())
}

/// scan(x, y, valid) -> (sorted_ok, first, last, ymin, ymax); updates `valid` in place.
#[pyfunction]
fn scan<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    mut valid: PyReadwriteArray1<'py, bool>,
) -> PyResult<(bool, i64, i64, f64, f64)> {
    let (x, y) = (x.as_slice()?, y.as_slice()?);
    let valid = valid.as_slice_mut()?;
    check_len(x.len(), y.len())?;
    check_len(x.len(), valid.len())?;
    Ok(py.detach(|| core::scan(x, y, valid)))
}

/// rdp_keep(x, y, valid, eps, vertical, max_points, keep) -> max_error; updates `keep` in place.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rdp_keep<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    valid: PyReadonlyArray1<'py, bool>,
    eps: f64,
    vertical: bool,
    max_points: i64,
    mut keep: PyReadwriteArray1<'py, bool>,
) -> PyResult<f64> {
    let (x, y, valid) = (x.as_slice()?, y.as_slice()?, valid.as_slice()?);
    let keep = keep.as_slice_mut()?;
    check_len(x.len(), y.len())?;
    check_len(x.len(), valid.len())?;
    check_len(x.len(), keep.len())?;
    // Release the GIL: other Python threads keep running while we crunch.
    Ok(py.detach(|| core::rdp_keep(x, y, valid, eps, vertical, max_points, keep)))
}

/// Which SIMD path the distance loops use on this CPU: "avx512", "avx2" or "baseline".
#[pyfunction]
fn simd_level() -> &'static str {
    core::simd_level()
}

#[pymodule]
fn _plotsimplify(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simd_level, m)?)?;
    m.add_function(wrap_pyfunction!(scan, m)?)?;
    m.add_function(wrap_pyfunction!(rdp_keep, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
