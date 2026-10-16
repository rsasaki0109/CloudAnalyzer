//! Gyro-only rotation compensation; no fabricated translational state.

use crate::imu::{err, lidar_transform};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyclass(frozen, module = "cloudanalyzer_core._core")]
pub struct ImuRotationTrajectory {
    pub(crate) trajectory: ca_core::imu_rotation::ImuRotationTrajectory,
}

#[pymethods]
impl ImuRotationTrajectory {
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
    fn gyro_bias(&self) -> (f64, f64, f64) {
        let bias = self.trajectory.gyro_bias();
        (bias[0], bias[1], bias[2])
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
    /// IMU-to-start-gauge rotation; the start gauge is identity, not an attitude estimate.
    fn rotation<'py>(&self, py: Python<'py>, stamp_ns: i64) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(crate::to_array2(
            py,
            self.trajectory.rotation(stamp_ns).map_err(err)?.to_vec(),
        ))
    }
    /// Fixed IMU origin within the scan; includes the LiDAR lever arm under rotation.
    /// Translation of the IMU origin remains unmodeled.
    #[pyo3(signature=(points, point_times_ns, reference_ns, *, imu_from_lidar, parallel=false))]
    fn deskew<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
        point_times_ns: PyReadonlyArray1<'_, i64>,
        reference_ns: i64,
        imu_from_lidar: PyReadonlyArray2<'_, f64>,
        parallel: bool,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let extrinsic = lidar_transform(&imu_from_lidar)?;
        // Copy strided caller buffers before releasing the GIL.
        let points = crate::points(&points)?;
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
        Ok(crate::to_array2(py, output))
    }
}
