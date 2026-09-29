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

/// Registers scans one after another onto a local map of the ones before
/// (see `ca_core::odometry`): the KISS-ICP recipe in the Rust core.
#[pyclass]
pub struct LidarOdometry {
    inner: Odometry,
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
    #[pyo3(signature = (min_range=1.5, max_range=80.0, map_voxel=1.0, map_points=20, deskew=false))]
    fn new(
        min_range: f64,
        max_range: f64,
        map_voxel: f64,
        map_points: usize,
        deskew: bool,
    ) -> Self {
        LidarOdometry {
            inner: Odometry::new(OdometryParams {
                min_range,
                max_range,
                map_voxel,
                map_points,
                deskew,
                ..OdometryParams::default()
            }),
        }
    }

    /// Register the next scan, `positions` (N, 3) in its sensor's frame,
    /// with `times` (N,) saying when in the sweep each point was taken (any
    /// unit; only for deskewing): its pose as a 4x4 matrix.
    #[pyo3(signature = (positions, times=None))]
    fn register<'py>(
        &mut self,
        py: Python<'py>,
        positions: PyReadonlyArray2<f64>,
        times: Option<PyReadonlyArray1<f32>>,
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
        let pose = py.detach(|| self.inner.register(&scan));
        Ok(matrix(pose.to_matrix()).into_pyarray(py))
    }

    /// Every pose so far, (K, 4, 4).
    fn poses<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let poses = self.inner.poses();
        let flat: Vec<f64> = poses.iter().flat_map(|p| p.to_matrix()).collect();
        Array3::from_shape_vec((poses.len(), 4, 4), flat)
            .expect("16 numbers a pose")
            .into_pyarray(py)
    }
}
