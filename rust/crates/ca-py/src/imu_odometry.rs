//! Explicit IMU compensation before ICP, with one reference time per pose.

use ca_core::PointCloud;
use ca_core::icp::Rigid;
use ca_core::imu::ImuError;
use ca_core::imu_trajectory::ImuTrajectory;
use ca_core::odometry::{Odometry, OdometryParams};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

/// ICP poses and placed map points use the first accepted reference LiDAR frame.
/// Owns the odometry and shares only immutable nominal IMU motion knots.
#[pyclass(module = "cloudanalyzer_core._core")]
pub struct ImuLidarOdometry {
    trajectory: Arc<ImuTrajectory>,
    imu_from_lidar: Rigid,
    odometry: crate::odometry::LidarOdometry,
    parallel: bool,
    reference_times: Vec<i64>,
    world_from_first_lidar: Option<Rigid>,
    last_prediction: Option<Rigid>,
}

fn matrix<'py>(py: Python<'py>, pose: &Rigid) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_vec((4, 4), pose.to_matrix().to_vec())
        .expect("16 pose elements")
        .into_pyarray(py)
}

#[pymethods]
impl ImuLidarOdometry {
    #[new]
    #[allow(clippy::too_many_arguments)] // Explicit Python initialization/calibration inputs.
    #[pyo3(signature = (trajectory, imu_from_lidar, *, min_range=1.5, max_range=80.0, map_voxel=1.0, map_points=20, parallel=false, diagnostics=false, registration_voxel=None))]
    fn new(
        trajectory: &crate::imu::ImuTrajectory,
        imu_from_lidar: PyReadonlyArray2<'_, f64>,
        min_range: f64,
        max_range: f64,
        map_voxel: f64,
        map_points: usize,
        parallel: bool,
        diagnostics: bool,
        registration_voxel: Option<f64>,
    ) -> PyResult<Self> {
        if !min_range.is_finite()
            || min_range < 0.0
            || !max_range.is_finite()
            || max_range <= min_range
            || !map_voxel.is_finite()
            || map_voxel <= 0.0
            || map_points == 0
        {
            return Err(PyValueError::new_err(
                "need finite 0 <= min_range < max_range, positive map_voxel and map_points",
            ));
        }
        let extrinsic = crate::imu::lidar_transform(&imu_from_lidar)?;
        let motion = Arc::clone(&trajectory.trajectory);
        motion
            .relative_lidar_pose(motion.start_ns(), motion.start_ns(), &extrinsic)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut inner = Odometry::new(OdometryParams {
            min_range,
            max_range,
            map_voxel,
            map_points,
            diagnostics,
            deskew: false,
            ..OdometryParams::default()
        });
        inner
            .set_registration_voxel(registration_voxel)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            trajectory: motion,
            imu_from_lidar: extrinsic,
            odometry: crate::odometry::LidarOdometry { inner },
            parallel,
            reference_times: Vec::new(),
            world_from_first_lidar: None,
            last_prediction: None,
        })
    }

    /// Explicit independent source voxel in metres; None keeps legacy sampling.
    #[getter]
    fn registration_voxel(&self) -> Option<f64> {
        self.odometry.registration_voxel()
    }

    /// Compensate once, register, then place the same reference-frame points.
    /// Reference timestamps must strictly increase. All input validation precedes
    /// mutation of the map, poses, diagnostics and reference history.
    fn register<'py>(
        &mut self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, f64>,
        point_times_ns: PyReadonlyArray1<'_, i64>,
        reference_ns: i64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if self
            .reference_times
            .last()
            .is_some_and(|&previous| reference_ns <= previous)
        {
            return Err(PyValueError::new_err(
                "reference timestamps must strictly increase",
            ));
        }
        // No caller-owned array is accessed while the GIL is released.
        let points = crate::points(&points)?;
        let times: Vec<i64> = point_times_ns.as_array().iter().copied().collect();
        let (pose, prediction, first_pose) = py
            .detach(|| -> Result<_, ImuError> {
                let prediction = self
                    .reference_times
                    .last()
                    .map(|&previous| {
                        self.trajectory.relative_lidar_pose(
                            previous,
                            reference_ns,
                            &self.imu_from_lidar,
                        )
                    })
                    .transpose()?;
                let first_pose = if self.reference_times.is_empty() {
                    let state = self.trajectory.state(reference_ns)?;
                    let pose = Rigid {
                        rotation: state.rotation,
                        translation: state.position,
                    }
                    .compose(&self.imu_from_lidar);
                    if !pose
                        .rotation
                        .iter()
                        .flatten()
                        .chain(&pose.translation)
                        .all(|v| v.is_finite())
                    {
                        return Err(ImuError(
                            "initial world-from-LiDAR pose is not finite".into(),
                        ));
                    }
                    Some(pose)
                } else {
                    None
                };
                let positions = if self.parallel {
                    self.trajectory.deskew_parallel(
                        &points,
                        &times,
                        reference_ns,
                        &self.imu_from_lidar,
                    )?
                } else {
                    self.trajectory
                        .deskew(&points, &times, reference_ns, &self.imu_from_lidar)?
                };
                let scan = PointCloud {
                    positions,
                    ..PointCloud::default()
                };
                let pose = match prediction {
                    Some(prediction) => self
                        .odometry
                        .inner
                        .register_with_prediction(&scan, &prediction)
                        .map_err(|e| ImuError(e.to_string()))?,
                    None => self.odometry.inner.register(&scan),
                };
                Ok((pose, prediction, first_pose))
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.reference_times.push(reference_ns);
        self.last_prediction = prediction;
        if let Some(pose) = first_pose {
            self.world_from_first_lidar = Some(pose);
        }
        Ok(matrix(py, &pose))
    }

    /// Alignment of the first-reference local frame to the nominal IMU world.
    #[getter]
    fn world_from_first_lidar<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.world_from_first_lidar.as_ref().map(|p| matrix(py, p))
    }

    /// Copied integer reference timestamps corresponding exactly to poses().
    fn reference_timestamps_ns<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.reference_times.clone().into_pyarray(py)
    }

    fn poses<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        self.odometry.poses(py)
    }

    /// Copied points placed by the last scan, after range filtering and thinning.
    /// Uses the same first-reference local frame as poses(); not an accumulated map.
    fn placed<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        crate::to_array2(py, self.odometry.inner.placed().to_vec())
    }

    /// The last IMU initial guess, mapping current into preceding reference frame.
    fn last_prediction<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.last_prediction.as_ref().map(|p| matrix(py, p))
    }

    fn last_iterations(&self) -> usize {
        self.odometry.inner.last_iterations()
    }

    fn last_report<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.odometry.last_report(py)
    }
}
