//! Reusable inertial factors; a bag is read only when the factor is constructed.

use ca_core::imu::{ImuError, PreintegratedImu, PreintegrationParams};
use ca_core::imu_factor::{ImuEvaluation, ImuFactor as Factor, ImuState};
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) fn err(error: ImuError) -> PyErr {
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
    result.set_item("start_ns", m.start_ns)?;
    result.set_item("end_ns", m.end_ns)?;
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

pub(crate) fn state(pose: PyReadonlyArray2<'_, f64>, velocity: [f64; 3]) -> PyResult<ImuState> {
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

/// An immutable, in-memory topic snapshot. BagReader.imu_series() decodes once;
/// factors then search timestamps and integrate only their overlapping samples.
#[pyclass(frozen, module = "cloudanalyzer_core._core")]
pub struct ImuSeries {
    pub(crate) series: ca_core::imu::ImuSeries,
}

#[pymethods]
impl ImuSeries {
    /// Configured raw-acceleration multiplier; bias/noise remain in SI units.
    #[getter]
    fn acceleration_scale(&self) -> f64 {
        self.series.acceleration_scale()
    }

    fn __len__(&self) -> usize {
        self.series.samples().len()
    }

    /// Exact header timestamps, copied so callers cannot change the snapshot.
    fn timestamps_ns<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.series
            .samples()
            .iter()
            .map(|s| s.stamp_ns)
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    fn frame_ids(&self) -> Vec<String> {
        self.series
            .samples()
            .iter()
            .map(|s| s.frame_id.clone())
            .collect()
    }

    /// Fixed-anchor nominal motion; state, gravity and bias must be declared.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (start_ns, end_ns, pose_i, velocity_i, *, gravity, bias, max_gap_s=0.05))]
    fn trajectory(
        &self,
        py: Python<'_>,
        start_ns: i64,
        end_ns: i64,
        pose_i: PyReadonlyArray2<'_, f64>,
        velocity_i: [f64; 3],
        gravity: [f64; 3],
        bias: [f64; 6],
        max_gap_s: f64,
    ) -> PyResult<ImuTrajectory> {
        let initial = state(pose_i, velocity_i)?;
        let params = ca_core::imu_trajectory::TrajectoryParams {
            gravity,
            gyro_bias: [bias[0], bias[1], bias[2]],
            accel_bias: [bias[3], bias[4], bias[5]],
            max_gap_s,
        };
        let trajectory = py
            .detach(|| self.series.trajectory(start_ns, end_ns, initial, &params))
            .map_err(err)?;
        Ok(ImuTrajectory {
            trajectory: std::sync::Arc::new(trajectory),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (start_ns, end_ns, *, gyro_noise_density, accel_noise_density, max_gap_s=0.05, bias=None))]
    fn preintegrate<'py>(
        &self,
        py: Python<'py>,
        start_ns: i64,
        end_ns: i64,
        gyro_noise_density: f64,
        accel_noise_density: f64,
        max_gap_s: f64,
        bias: Option<[f64; 6]>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let p = params(gyro_noise_density, accel_noise_density, max_gap_s, bias);
        let out = py
            .detach(|| self.series.preintegrate(start_ns, end_ns, &p))
            .map_err(err)?;
        measurement_dict(py, &out)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (start_ns, end_ns, *, gyro_noise_density, accel_noise_density, max_gap_s=0.05, bias=None))]
    fn factor(
        &self,
        py: Python<'_>,
        start_ns: i64,
        end_ns: i64,
        gyro_noise_density: f64,
        accel_noise_density: f64,
        max_gap_s: f64,
        bias: Option<[f64; 6]>,
    ) -> PyResult<ImuFactor> {
        let p = params(gyro_noise_density, accel_noise_density, max_gap_s, bias);
        let out = py
            .detach(|| self.series.preintegrate(start_ns, end_ns, &p))
            .map_err(err)?;
        ImuFactor::new(out)
    }
}

/// Owns covered motion knots; no recording/series access and no fitted state.
#[pyclass(frozen, module = "cloudanalyzer_core._core")]
pub struct ImuTrajectory {
    pub(crate) trajectory: std::sync::Arc<ca_core::imu_trajectory::ImuTrajectory>,
}

#[pymethods]
impl ImuTrajectory {
    #[getter]
    fn start_ns(&self) -> i64 {
        self.trajectory.start_ns()
    }
    #[getter]
    fn end_ns(&self) -> i64 {
        self.trajectory.end_ns()
    }
    #[getter]
    fn frame_id(&self) -> String {
        self.trajectory.frame_id().to_owned()
    }
    #[getter]
    fn acceleration_scale(&self) -> f64 {
        self.trajectory.acceleration_scale()
    }
    fn __len__(&self) -> usize {
        self.trajectory.knot_count()
    }
    fn timestamps_ns<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.trajectory
            .timestamps_ns()
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    fn state<'py>(&self, py: Python<'py>, stamp_ns: i64) -> PyResult<Bound<'py, PyDict>> {
        let state = self.trajectory.state(stamp_ns).map_err(err)?;
        let pose: [[f64; 4]; 4] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i == 3 {
                    f64::from(u8::from(j == 3))
                } else if j == 3 {
                    state.position[i]
                } else {
                    state.rotation[i][j]
                }
            })
        });
        let result = PyDict::new(py);
        result.set_item("pose", matrix(py, &pose))?;
        result.set_item("velocity", state.velocity.to_vec().into_pyarray(py))?;
        Ok(result)
    }

    /// LiDAR coordinates at the explicit reference time; preserves input order.
    #[pyo3(signature = (points, point_times_ns, reference_ns, *, imu_from_lidar, parallel=false))]
    fn deskew<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
        point_times_ns: PyReadonlyArray1<'_, i64>,
        reference_ns: i64,
        imu_from_lidar: PyReadonlyArray2<'_, f64>,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if points.shape()[1] != 3 {
            return Err(PyValueError::new_err(
                "deskew points must have shape (N, 3)",
            ));
        }
        let extrinsic = lidar_transform(&imu_from_lidar)?;
        // Snapshot Python buffers before releasing the GIL; no caller-owned
        // buffer is read while the core map runs on worker threads.
        let points: Vec<[f64; 3]> = points
            .as_array()
            .outer_iter()
            .map(|r| [r[0], r[1], r[2]])
            .collect();
        let times: Vec<i64> = point_times_ns.as_array().iter().copied().collect();
        let output = py
            .detach(|| {
                if parallel {
                    self.trajectory
                        .deskew_parallel(&points, &times, reference_ns, &extrinsic)
                } else {
                    self.trajectory
                        .deskew(&points, &times, reference_ns, &extrinsic)
                }
            })
            .map_err(err)?;
        let count = output.len();
        Ok(
            Array2::from_shape_vec((count, 3), output.into_iter().flatten().collect())
                .expect("deskew point dimensions")
                .into_pyarray(py),
        )
    }
}

pub(crate) fn lidar_transform(matrix: &PyReadonlyArray2<'_, f64>) -> PyResult<ca_core::icp::Rigid> {
    if matrix.shape() != [4, 4] {
        return Err(PyValueError::new_err(
            "IMU-from-LiDAR transform must have shape (4, 4)",
        ));
    }
    let transform = matrix.as_array();
    if !(0..4).all(|j| {
        transform[[3, j]].is_finite()
            && (transform[[3, j]] - f64::from(u8::from(j == 3))).abs() <= 1e-12
    }) {
        return Err(PyValueError::new_err(
            "IMU-from-LiDAR transform must have homogeneous last row [0, 0, 0, 1]",
        ));
    }
    Ok(ca_core::icp::Rigid {
        rotation: std::array::from_fn(|i| std::array::from_fn(|j| transform[[i, j]])),
        translation: std::array::from_fn(|i| transform[[i, 3]]),
    })
}

/// A preintegrated IMU factor, created by BagReader.imu_factor().
#[pyclass(frozen, module = "cloudanalyzer_core._core")]
pub struct ImuFactor {
    pub(crate) factor: Factor,
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
