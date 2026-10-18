//! Copied NumPy representation of an opt-in registration snapshot.

use ca_core::odometry_linearization::RegistrationLinearization;
use numpy::IntoPyArray;
use numpy::ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) fn to_dict<'py>(
    py: Python<'py>,
    snapshot: &RegistrationLinearization,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item(
        "pose",
        Array2::from_shape_vec((4, 4), snapshot.pose.to_matrix().to_vec())
            .expect("4 x 4 pose")
            .into_pyarray(py),
    )?;
    result.set_item(
        "center_world",
        snapshot.center_world.to_vec().into_pyarray(py),
    )?;
    for (name, value) in [
        ("threshold_sq", snapshot.threshold_sq),
        ("kernel", snapshot.kernel),
        ("map_voxel", snapshot.map_voxel),
        ("max_range", snapshot.max_range),
        ("weight_sum", snapshot.weight_sum),
    ] {
        result.set_item(name, value)?;
    }
    result.set_item("map_points", snapshot.map_points)?;
    result.set_item("alignment_eligible", snapshot.alignment_eligible)?;
    result.set_item("target_balance", snapshot.target_balance)?;
    result.set_item("finite", snapshot.finite)?;
    result.set_item("weighted_rms_radius_m", snapshot.weighted_rms_radius_m)?;
    for (name, rows) in [
        ("source_sensor", &snapshot.source_sensor),
        ("source_world", &snapshot.source_world),
        ("target_world", &snapshot.target_world),
        ("inserted_world", &snapshot.inserted_world),
    ] {
        result.set_item(name, crate::to_array2(py, rows.clone()))?;
    }
    result.set_item(
        "matched_source_indices",
        snapshot
            .matched_source_indices
            .iter()
            .map(|&index| index as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    result.set_item(
        "squared_distances",
        snapshot.squared_distances.clone().into_pyarray(py),
    )?;
    result.set_item("weights", snapshot.weights.clone().into_pyarray(py))?;
    result.set_item(
        "robust_weights",
        snapshot.robust_weights.clone().into_pyarray(py),
    )?;
    result.set_item(
        "target_multiplicity",
        snapshot
            .target_multiplicity
            .iter()
            .map(|&n| n as i64)
            .collect::<Vec<_>>()
            .into_pyarray(py),
    )?;
    result.set_item(
        "normal_matrix",
        Array2::from_shape_vec(
            (6, 6),
            snapshot.normal_matrix.into_iter().flatten().collect(),
        )
        .expect("6 x 6 normal matrix")
        .into_pyarray(py),
    )?;
    result.set_item(
        "right_hand_side",
        snapshot.right_hand_side.to_vec().into_pyarray(py),
    )?;
    Ok(result)
}
