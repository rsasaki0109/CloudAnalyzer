//! LiDAR odometry for Python: scans in, poses out.

use ca_core::PointCloud;
use ca_core::odometry::{Odometry, OdometryParams, TIME};
use ca_core::{Attribute, AttributeValues};
use numpy::ndarray::{Array2, Array3};
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Registers scans one after another onto a local map of the ones before
/// (see `ca_core::odometry`): the KISS-ICP recipe in the Rust core.
#[pyclass]
pub struct LidarOdometry {
    pub(crate) inner: Odometry,
}

fn matrix(m: [f64; 16]) -> Array2<f64> {
    Array2::from_shape_vec((4, 4), m.to_vec()).expect("16 numbers")
}

#[pymethods]
impl LidarOdometry {
    /// Points nearer than `min_range` or further than `max_range` (metres)
    /// are left out; the local map keeps `map_points` per voxel of
    /// `map_voxel` metres; `deskew` undoes the motion during each scan.
    #[new]
    #[pyo3(signature = (min_range=1.5, max_range=80.0, map_voxel=1.0, map_points=20, deskew=false, *, diagnostics=false))]
    fn new(
        min_range: f64,
        max_range: f64,
        map_voxel: f64,
        map_points: usize,
        deskew: bool,
        diagnostics: bool,
    ) -> Self {
        LidarOdometry {
            inner: Odometry::new(OdometryParams {
                min_range,
                max_range,
                map_voxel,
                map_points,
                deskew,
                diagnostics,
                ..OdometryParams::default()
            }),
        }
    }

    /// Register the next scan, `positions` (N, 3) in its sensor's frame,
    /// with `times` (N,) saying when in the sweep each point was taken (any
    /// unit; only for deskewing): its pose as a 4x4 matrix.
    /// `relative_prediction`, when supplied, maps the new scan into the previous
    /// scan frame. It changes the ICP initial pose, not the deskew motion model.
    #[pyo3(signature = (positions, times=None, *, relative_prediction=None))]
    fn register<'py>(
        &mut self,
        py: Python<'py>,
        positions: PyReadonlyArray2<f64>,
        times: Option<PyReadonlyArray1<f32>>,
        relative_prediction: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let positions = crate::points(&positions)?;
        let mut scan = PointCloud {
            positions,
            ..PointCloud::default()
        };
        if let Some(times) = times {
            if times.len() != scan.len() {
                return Err(PyValueError::new_err("times needs one number per point"));
            }
            scan.attributes.push(Attribute {
                name: TIME.to_string(),
                values: AttributeValues::F32(times.as_slice()?.to_vec()),
            });
        }
        let prediction = relative_prediction
            .map(|pose| {
                if pose.shape() != [4, 4] {
                    return Err(PyValueError::new_err(
                        "relative_prediction needs a (4, 4) pose",
                    ));
                }
                let view = pose.as_array();
                if !(0..4).all(|j| {
                    view[[3, j]].is_finite()
                        && (view[[3, j]] - f64::from(u8::from(j == 3))).abs() <= 1e-12
                }) {
                    return Err(PyValueError::new_err(
                        "relative_prediction needs homogeneous last row [0, 0, 0, 1]",
                    ));
                }
                Ok::<_, PyErr>(ca_core::icp::Rigid {
                    rotation: std::array::from_fn(|i| std::array::from_fn(|j| view[[i, j]])),
                    translation: std::array::from_fn(|i| view[[i, 3]]),
                })
            })
            .transpose()?;
        let pose = py.detach(|| match prediction {
            Some(prediction) => self
                .inner
                .register_with_prediction(&scan, &prediction)
                .map_err(|e| PyValueError::new_err(e.to_string())),
            None => Ok(self.inner.register(&scan)),
        })?;
        Ok(matrix(pose.to_matrix()).into_pyarray(py))
    }

    /// Gauss-Newton steps of the last registration; zero without alignment.
    fn last_iterations(&self) -> usize {
        self.inner.last_iterations()
    }

    /// A copied report for the last finished scan, or None before the first.
    /// With diagnostics=True, residuals use the returned pose and the map before
    /// insertion. They use ICP's search/threshold and do not certify alignment.
    pub(crate) fn last_report<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(report) = self.inner.last_report() else {
            return Ok(None);
        };
        let result = PyDict::new(py);
        result.set_item("termination", report.termination.as_str())?;
        result.set_item("iterations", report.iterations)?;
        result.set_item("input_points", report.input_points)?;
        result.set_item("registration_points", report.registration_points)?;
        result.set_item("threshold_m", report.threshold_m)?;
        if let Some(stats) = &report.correspondences {
            let correspondences = PyDict::new(py);
            correspondences.set_item("matched_points", stats.matched_points)?;
            correspondences.set_item("matched_fraction", stats.matched_fraction)?;
            correspondences.set_item("rmse_m", stats.rmse_m)?;
            correspondences.set_item("max_residual_m", stats.max_residual_m)?;
            result.set_item("correspondences", correspondences)?;
        } else {
            result.set_item("correspondences", py.None())?;
        }
        Ok(Some(result))
    }

    /// Every pose so far, (K, 4, 4).
    pub(crate) fn poses<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let poses = self.inner.poses();
        let flat: Vec<f64> = poses.iter().flat_map(|p| p.to_matrix()).collect();
        Array3::from_shape_vec((poses.len(), 4, 4), flat)
            .expect("16 numbers a pose")
            .into_pyarray(py)
    }
}
