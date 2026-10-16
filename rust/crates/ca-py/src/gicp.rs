//! Owned fixed-correspondence GICP linearization for Python.

use ca_core::gicp::{Correspondence, Matrix3};
use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn shapes(array: &PyReadonlyArray3<f64>, count: usize, name: &str) -> PyResult<Vec<Matrix3>> {
    if array.shape() != [count, 3, 3] {
        return Err(PyValueError::new_err(format!(
            "{name} needs shape (N, 3, 3)"
        )));
    }
    let view = array.as_array();
    Ok((0..count)
        .map(|k| std::array::from_fn(|i| std::array::from_fn(|j| view[[k, i, j]])))
        .collect())
}

/// Fixed matches and metric: H=J.T W J, b=-J.T W residual, half-squared cost.
/// Source points/shapes use sensor axes; targets/shapes use world axes. The
/// transport C_target+R C_source R.T is held fixed while differentiating the
/// residual. Shapes are not inferred or treated as calibrated uncertainty.
#[pyfunction]
#[pyo3(signature = (source, target, source_shapes, target_shapes, pose, center_world, *, weights=None, parallel=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gicp_linearization<'py>(
    py: Python<'py>,
    source: PyReadonlyArray2<f64>,
    target: PyReadonlyArray2<f64>,
    source_shapes: PyReadonlyArray3<f64>,
    target_shapes: PyReadonlyArray3<f64>,
    pose: PyReadonlyArray2<f64>,
    center_world: PyReadonlyArray1<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
    parallel: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let source = crate::points(&source)?;
    let target = crate::points(&target)?;
    if target.len() != source.len() {
        return Err(PyValueError::new_err(
            "target needs one point per source point",
        ));
    }
    let n = source.len();
    let source_shapes = shapes(&source_shapes, n, "source_shapes")?;
    let target_shapes = shapes(&target_shapes, n, "target_shapes")?;
    let weights = if let Some(weights) = weights {
        if weights.len() != n {
            return Err(PyValueError::new_err("weights needs one value per pair"));
        }
        weights.as_array().iter().copied().collect::<Vec<_>>()
    } else {
        vec![1.0; n]
    };
    if pose.shape() != [4, 4] {
        return Err(PyValueError::new_err("pose needs shape (4, 4)"));
    }
    let view = pose.as_array();
    if !(0..4).all(|j| {
        view[[3, j]].is_finite() && (view[[3, j]] - f64::from(u8::from(j == 3))).abs() <= 1e-12
    }) {
        return Err(PyValueError::new_err(
            "pose needs homogeneous last row [0,0,0,1]",
        ));
    }
    let pose = ca_core::icp::Rigid {
        rotation: std::array::from_fn(|i| std::array::from_fn(|j| view[[i, j]])),
        translation: std::array::from_fn(|i| view[[i, 3]]),
    };
    if center_world.shape() != [3] {
        return Err(PyValueError::new_err("center_world needs shape (3,)"));
    }
    let center = center_world.as_array();
    let center = [center[0], center[1], center[2]];
    let pairs: Vec<_> = (0..n)
        .map(|i| Correspondence {
            source: source[i],
            target: target[i],
            source_shape: source_shapes[i],
            target_shape: target_shapes[i],
            weight: weights[i],
        })
        .collect();
    let result = py
        .detach(|| ca_core::gicp::linearize(&pairs, &pose, center, parallel))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let out = PyDict::new(py);
    out.set_item(
        "normal_matrix",
        Array2::from_shape_vec((6, 6), result.normal_matrix.into_iter().flatten().collect())
            .expect("6 x 6 normal matrix")
            .into_pyarray(py),
    )?;
    out.set_item(
        "right_hand_side",
        result.right_hand_side.to_vec().into_pyarray(py),
    )?;
    out.set_item("cost", result.cost)?;
    out.set_item("weight_sum", result.weight_sum)?;
    out.set_item("positive_weight_points", result.positive_weight_points)?;
    out.set_item("source_world", crate::to_array2(py, result.source_world))?;
    out.set_item(
        "combined_shapes",
        Array3::from_shape_vec(
            (n, 3, 3),
            result
                .combined_shapes
                .into_iter()
                .flatten()
                .flatten()
                .collect(),
        )
        .expect("N x 3 x 3 shapes")
        .into_pyarray(py),
    )?;
    out.set_item(
        "mahalanobis_squared",
        result.mahalanobis_squared.into_pyarray(py),
    )?;
    Ok(out)
}
