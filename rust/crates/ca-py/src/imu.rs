//! Reusable inertial factors; a bag is read only when the factor is constructed.

use ca_core::imu::{ImuError, PreintegratedImu, PreintegrationParams};
use ca_core::imu_factor::{ImuEvaluation, ImuFactor as Factor, ImuState};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn err(error: ImuError) -> PyErr {
    PyValueError::new_err(error.0)
}

pub(crate) fn params(
    gyro: f64,
    accel: f64,
    gap: f64,
    bias: Option<[f64; 6]>,
) -> PreintegrationParams {
    let bias = bias.unwrap_or([0.0; 6]);
    PreintegrationParams {
        gyro_noise_density: gyro,
        accel_noise_density: accel,
        max_gap_s: gap,
        gyro_bias: [bias[0], bias[1], bias[2]],
        accel_bias: [bias[3], bias[4], bias[5]],
    }
}

fn matrix<'py, const M: usize, const N: usize>(
    py: Python<'py>,
    data: &[[f64; N]; M],
) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_vec((M, N), data.iter().flatten().copied().collect())
        .expect("matrix dimensions")
        .into_pyarray(py)
}

pub(crate) fn measurement_dict<'py>(
    py: Python<'py>,
    m: &PreintegratedImu,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("delta_rotation", matrix(py, &m.delta_rotation))?;
    result.set_item("delta_velocity", m.delta_velocity.to_vec().into_pyarray(py))?;
    result.set_item("delta_position", m.delta_position.to_vec().into_pyarray(py))?;
    result.set_item("covariance", matrix(py, &m.covariance))?;
    result.set_item("bias_jacobian", matrix(py, &m.bias_jacobian))?;
    result.set_item(
        "linearization_bias",
        m.linearization_bias.to_vec().into_pyarray(py),
    )?;
    result.set_item("delta_time_s", m.delta_time_s)?;
    result.set_item("integration_steps", m.integration_steps)?;
    result.set_item("frame_id", &m.frame_id)?;
    Ok(result)
}

fn state(pose: PyReadonlyArray2<'_, f64>, velocity: [f64; 3]) -> PyResult<ImuState> {
    if pose.shape() != [4, 4] {
        return Err(PyValueError::new_err("expected a (4, 4) IMU-to-world pose"));
    }
    let view = pose.as_array();
    if !(0..4).all(|j| {
        view[[3, j]].is_finite() && (view[[3, j]] - f64::from(u8::from(j == 3))).abs() <= 1e-12
    }) {
        return Err(PyValueError::new_err(
            "IMU pose must have homogeneous last row [0, 0, 0, 1]",
        ));
    }
    Ok(ImuState {
        rotation: std::array::from_fn(|i| std::array::from_fn(|j| view[[i, j]])),
        position: std::array::from_fn(|i| view[[i, 3]]),
        velocity,
    })
}

fn evaluation_dict<'py>(py: Python<'py>, out: &ImuEvaluation) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("residual", out.residual.to_vec().into_pyarray(py))?;
    result.set_item(
        "whitened_residual",
        out.whitened_residual.to_vec().into_pyarray(py),
    )?;
    result.set_item("cost", out.cost)?;
    Ok(result)
}

/// A preintegrated IMU factor, created by BagReader.imu_factor().
#[pyclass(frozen, module = "cloudanalyzer_core._core")]
pub struct ImuFactor {
    factor: Factor,
}

impl ImuFactor {
    pub(crate) fn new(measurement: PreintegratedImu) -> PyResult<Self> {
        Ok(Self {
            factor: Factor::new(measurement).map_err(err)?,
        })
    }
}

#[pymethods]
impl ImuFactor {
    /// A copy of the preintegrated measurement and its bias linearization point.
    fn measurement<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        measurement_dict(py, self.factor.measurement())
    }

    /// Predict an IMU-to-world pose and world velocity, adding world gravity.
    #[pyo3(signature = (pose_i, velocity_i, *, gravity, bias=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        pose_i: PyReadonlyArray2<f64>,
        velocity_i: [f64; 3],
        gravity: [f64; 3],
        bias: Option<[f64; 6]>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let initial = state(pose_i, velocity_i)?;
        let predicted = self
            .factor
            .predict(
                &initial,
                gravity,
                bias.unwrap_or(self.factor.measurement().linearization_bias),
            )
            .map_err(err)?;
        let pose: [[f64; 4]; 4] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i == 3 {
                    f64::from(u8::from(j == 3))
                } else if j == 3 {
                    predicted.position[i]
                } else {
                    predicted.rotation[i][j]
                }
            })
        });
        let result = PyDict::new(py);
        result.set_item("pose", matrix(py, &pose))?;
        result.set_item("velocity", predicted.velocity.to_vec().into_pyarray(py))?;
        Ok(result)
    }

    /// Rotation/velocity/position residual, whitened residual and half squared cost.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (pose_i, velocity_i, pose_j, velocity_j, *, gravity, bias=None))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        pose_i: PyReadonlyArray2<f64>,
        velocity_i: [f64; 3],
        pose_j: PyReadonlyArray2<f64>,
        velocity_j: [f64; 3],
        gravity: [f64; 3],
        bias: Option<[f64; 6]>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let initial = state(pose_i, velocity_i)?;
        let terminal = state(pose_j, velocity_j)?;
        let out = self
            .factor
            .evaluate(
                &initial,
                &terminal,
                gravity,
                bias.unwrap_or(self.factor.measurement().linearization_bias),
            )
            .map_err(err)?;
        evaluation_dict(py, &out)
    }

    /// Analytical Jacobians: right-local rotation, world velocity, world position.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (pose_i, velocity_i, pose_j, velocity_j, *, gravity, bias=None))]
    fn linearize<'py>(
        &self,
        py: Python<'py>,
        pose_i: PyReadonlyArray2<f64>,
        velocity_i: [f64; 3],
        pose_j: PyReadonlyArray2<f64>,
        velocity_j: [f64; 3],
        gravity: [f64; 3],
        bias: Option<[f64; 6]>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let initial = state(pose_i, velocity_i)?;
        let terminal = state(pose_j, velocity_j)?;
        let out = self
            .factor
            .linearize(
                &initial,
                &terminal,
                gravity,
                bias.unwrap_or(self.factor.measurement().linearization_bias),
            )
            .map_err(err)?;
        let result = evaluation_dict(py, &out.evaluation)?;
        result.set_item("jacobian_i", matrix(py, &out.jacobian_i))?;
        result.set_item("jacobian_j", matrix(py, &out.jacobian_j))?;
        result.set_item("jacobian_bias", matrix(py, &out.jacobian_bias))?;
        result.set_item("jacobian_gravity", matrix(py, &out.jacobian_gravity))?;
        result.set_item("whitened_jacobian_i", matrix(py, &out.whitened_jacobian_i))?;
        result.set_item("whitened_jacobian_j", matrix(py, &out.whitened_jacobian_j))?;
        result.set_item(
            "whitened_jacobian_bias",
            matrix(py, &out.whitened_jacobian_bias),
        )?;
        result.set_item(
            "whitened_jacobian_gravity",
            matrix(py, &out.whitened_jacobian_gravity),
        )?;
        Ok(result)
    }
}
