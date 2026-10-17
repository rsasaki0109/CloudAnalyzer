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

/// Normalized f64 surface shapes and explicit support, with optional exact kNN.
#[pyfunction]
#[pyo3(signature = (points, *, neighbors=20, parallel=false, capture_neighbors=false))]
pub(crate) fn gicp_shapes<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<f64>,
    neighbors: usize,
    parallel: bool,
    capture_neighbors: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let points = crate::points(&points)?;
    let n = points.len();
    let result = py
        .detach(|| ca_core::gicp_shapes::estimate(&points, neighbors, parallel, capture_neighbors))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let width = if capture_neighbors {
        neighbors.min(n)
    } else {
        0
    };
    let out = PyDict::new(py);
    out.set_item("shapes", shape_array(py, result.shapes))?;
    out.set_item("supported", result.supported.into_pyarray(py))?;
    out.set_item(
        "scatter_eigenvalues",
        crate::to_array2(py, result.scatter_eigenvalues),
    )?;
    out.set_item(
        "neighbor_counts",
        result
            .neighbor_counts
            .into_iter()
            .map(|v| v as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    out.set_item(
        "neighbor_indices",
        Array2::from_shape_vec(
            (n, width),
            result
                .neighbor_indices
                .into_iter()
                .flatten()
                .map(|v| v as u64)
                .collect(),
        )
        .expect("N x K indices")
        .into_pyarray(py),
    )?;
    out.set_item(
        "neighbor_squared_distances",
        Array2::from_shape_vec(
            (n, width),
            result
                .neighbor_squared_distances
                .into_iter()
                .flatten()
                .collect(),
        )
        .expect("N x K distances")
        .into_pyarray(py),
    )?;
    Ok(out)
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

fn rigid(array: &PyReadonlyArray2<f64>) -> PyResult<ca_core::icp::Rigid> {
    if array.shape() != [4, 4] {
        return Err(PyValueError::new_err("pose needs shape (4, 4)"));
    }
    let v = array.as_array();
    if !(0..4)
        .all(|j| v[[3, j]].is_finite() && (v[[3, j]] - f64::from(u8::from(j == 3))).abs() <= 1e-12)
    {
        return Err(PyValueError::new_err(
            "pose needs homogeneous last row [0,0,0,1]",
        ));
    }
    Ok(ca_core::icp::Rigid {
        rotation: std::array::from_fn(|i| std::array::from_fn(|j| v[[i, j]])),
        translation: std::array::from_fn(|i| v[[i, 3]]),
    })
}

fn pose_array<'py>(py: Python<'py>, pose: ca_core::icp::Rigid) -> Bound<'py, numpy::PyArray2<f64>> {
    Array2::from_shape_vec((4, 4), pose.to_matrix().to_vec())
        .expect("4 x 4 pose")
        .into_pyarray(py)
}

fn shape_array<'py>(py: Python<'py>, shapes: Vec<Matrix3>) -> Bound<'py, numpy::PyArray3<f64>> {
    Array3::from_shape_vec(
        (shapes.len(), 3, 3),
        shapes.into_iter().flatten().flatten().collect(),
    )
    .expect("N x 3 x 3 shapes")
    .into_pyarray(py)
}

fn weights(array: Option<PyReadonlyArray1<f64>>, n: usize) -> PyResult<Vec<f64>> {
    match array {
        Some(array) if array.len() == n => Ok(array.as_array().iter().copied().collect()),
        Some(_) => Err(PyValueError::new_err("weights needs one value per pair")),
        None => Ok(vec![1.0; n]),
    }
}

/// Evaluate candidate residuals using supplied baseline combined shapes.
/// No candidate-pose shape transport or rematching is performed.
#[pyfunction]
#[pyo3(signature = (source, target, combined_shapes, pose, *, weights=None, parallel=false))]
pub(crate) fn gicp_frozen_cost<'py>(
    py: Python<'py>,
    source: PyReadonlyArray2<f64>,
    target: PyReadonlyArray2<f64>,
    combined_shapes: PyReadonlyArray3<f64>,
    pose: PyReadonlyArray2<f64>,
    weights: Option<PyReadonlyArray1<f64>>,
    parallel: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let source = crate::points(&source)?;
    let target = crate::points(&target)?;
    let shapes = shapes(&combined_shapes, source.len(), "combined_shapes")?;
    let pose = rigid(&pose)?;
    let weights = self::weights(weights, source.len())?;
    let result = py
        .detach(|| {
            ca_core::gicp_registration::FrozenMetric::new(&source, &target, &shapes, &weights)?
                .evaluate(&pose, parallel)
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let out = PyDict::new(py);
    out.set_item("cost", result.cost)?;
    out.set_item(
        "mahalanobis_squared",
        result.mahalanobis_squared.into_pyarray(py),
    )?;
    Ok(out)
}

/// Immutable copied world-frame points/shapes and an exact target search tree.
#[pyclass(frozen, module = "cloudanalyzer_core._core")]
pub(crate) struct GicpTarget {
    inner: ca_core::gicp_registration::PreparedTarget,
}

#[pymethods]
impl GicpTarget {
    #[new]
    fn new(
        py: Python<'_>,
        target: PyReadonlyArray2<f64>,
        target_shapes: PyReadonlyArray3<f64>,
    ) -> PyResult<Self> {
        let target = crate::points(&target)?;
        let target_shapes = shapes(&target_shapes, target.len(), "target_shapes")?;
        let inner = py
            .detach(|| ca_core::gicp_registration::PreparedTarget::new(&target, &target_shapes))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Register sensor-frame source points/shapes against this fixed target.
    #[pyo3(signature = (source, source_shapes, initial_pose, center_world, *, max_correspondence_distance, max_iterations=30, translation_tolerance=1e-4, rotation_tolerance=1e-5, robust_scale=None, parallel=false, capture_history=false))]
    #[allow(clippy::too_many_arguments)]
    fn register<'py>(
        &self,
        py: Python<'py>,
        source: PyReadonlyArray2<f64>,
        source_shapes: PyReadonlyArray3<f64>,
        initial_pose: PyReadonlyArray2<f64>,
        center_world: PyReadonlyArray1<f64>,
        max_correspondence_distance: f64,
        max_iterations: usize,
        translation_tolerance: f64,
        rotation_tolerance: f64,
        robust_scale: Option<f64>,
        parallel: bool,
        capture_history: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let source = crate::points(&source)?;
        let source_shapes = shapes(&source_shapes, source.len(), "source_shapes")?;
        let pose = rigid(&initial_pose)?;
        if center_world.shape() != [3] {
            return Err(PyValueError::new_err("center_world needs shape (3,)"));
        }
        let c = center_world.as_array();
        let center = [c[0], c[1], c[2]];
        let params = ca_core::gicp_registration::RegistrationParams {
            max_correspondence_distance,
            max_iterations,
            translation_tolerance,
            rotation_tolerance,
            robust_scale,
            parallel,
            capture_history,
        };
        let result = py
            .detach(|| {
                self.inner
                    .register(&source, &source_shapes, pose, center, params)
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        registration_dict(py, result)
    }
}

/// Iterative global nearest-neighbor GICP on explicit source/world SPD shapes.
/// Returns the last accepted pose and an explicit termination reason.
#[pyfunction]
#[pyo3(signature = (source, target, source_shapes, target_shapes, initial_pose, center_world, *, max_correspondence_distance, max_iterations=30, translation_tolerance=1e-4, rotation_tolerance=1e-5, robust_scale=None, parallel=false, capture_history=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gicp_register<'py>(
    py: Python<'py>,
    source: PyReadonlyArray2<f64>,
    target: PyReadonlyArray2<f64>,
    source_shapes: PyReadonlyArray3<f64>,
    target_shapes: PyReadonlyArray3<f64>,
    initial_pose: PyReadonlyArray2<f64>,
    center_world: PyReadonlyArray1<f64>,
    max_correspondence_distance: f64,
    max_iterations: usize,
    translation_tolerance: f64,
    rotation_tolerance: f64,
    robust_scale: Option<f64>,
    parallel: bool,
    capture_history: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let source = crate::points(&source)?;
    let target = crate::points(&target)?;
    let source_shapes = shapes(&source_shapes, source.len(), "source_shapes")?;
    let target_shapes = shapes(&target_shapes, target.len(), "target_shapes")?;
    let pose = rigid(&initial_pose)?;
    if center_world.shape() != [3] {
        return Err(PyValueError::new_err("center_world needs shape (3,)"));
    }
    let c = center_world.as_array();
    let center = [c[0], c[1], c[2]];
    let params = ca_core::gicp_registration::RegistrationParams {
        max_correspondence_distance,
        max_iterations,
        translation_tolerance,
        rotation_tolerance,
        robust_scale,
        parallel,
        capture_history,
    };
    let result = py
        .detach(|| {
            ca_core::gicp_registration::register(
                &source,
                &target,
                &source_shapes,
                &target_shapes,
                pose,
                center,
                params,
            )
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    registration_dict(py, result)
}

fn registration_dict<'py>(
    py: Python<'py>,
    result: ca_core::gicp_registration::Registration,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("pose", pose_array(py, result.pose))?;
    out.set_item("termination", result.termination.as_str())?;
    out.set_item(
        "converged",
        result.termination == ca_core::gicp_registration::Termination::ConvergedStep,
    )?;
    out.set_item("iterations", result.iterations)?;
    out.set_item("accepted_steps", result.accepted_steps)?;
    out.set_item("final_cost", result.final_cost)?;
    out.set_item(
        "matched_source_indices",
        result
            .matched_source_indices
            .into_iter()
            .map(|i| i as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    out.set_item(
        "matched_target_indices",
        result
            .matched_target_indices
            .into_iter()
            .map(|i| i as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    out.set_item(
        "squared_distances",
        result.squared_distances.into_pyarray(py),
    )?;
    out.set_item("weights", result.weights.into_pyarray(py))?;
    out.set_item("failure_reason", result.failure_reason)?;
    let history = pyo3::types::PyList::empty(py);
    for item in result.history {
        let row = PyDict::new(py);
        row.set_item("pose_before", pose_array(py, item.pose_before))?;
        row.set_item("pose_after", pose_array(py, item.pose_after))?;
        row.set_item(
            "source_indices",
            item.source_indices
                .into_iter()
                .map(|i| i as u64)
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        row.set_item(
            "target_indices",
            item.target_indices
                .into_iter()
                .map(|i| i as u64)
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        row.set_item("squared_distances", item.squared_distances.into_pyarray(py))?;
        row.set_item("weights", item.weights.into_pyarray(py))?;
        row.set_item("combined_shapes", shape_array(py, item.combined_shapes))?;
        row.set_item(
            "normal_matrix",
            Array2::from_shape_vec((6, 6), item.normal_matrix.into_iter().flatten().collect())
                .expect("6 x 6 system")
                .into_pyarray(py),
        )?;
        row.set_item(
            "right_hand_side",
            item.right_hand_side.to_vec().into_pyarray(py),
        )?;
        row.set_item("cost_before", item.cost_before)?;
        row.set_item("cost_after", item.cost_after)?;
        row.set_item("step", item.step.map(|v| v.to_vec().into_pyarray(py)))?;
        row.set_item("trial_scales", item.trial_scales.into_pyarray(py))?;
        row.set_item("trial_costs", item.trial_costs)?;
        row.set_item("accepted_scale", item.accepted_scale)?;
        history.append(row)?;
    }
    out.set_item("history", history)?;
    Ok(out)
}
