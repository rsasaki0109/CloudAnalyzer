//! LiDAR odometry for scans that come without poses (e.g. from a ROS bag).

use ca_core::odometry::{Odometry, OdometryParams};
use wasm_bindgen::prelude::*;

use crate::Cloud;

/// Registers scans one after another onto a local map of the ones before
/// (see `ca_core::odometry`).
#[wasm_bindgen]
pub struct LidarOdometry {
    inner: Odometry,
}

#[wasm_bindgen]
impl LidarOdometry {
    /// Points nearer than `min_range` or further than `max_range` (metres)
    /// are left out.
    #[wasm_bindgen(constructor)]
    pub fn new(min_range: f64, max_range: f64) -> LidarOdometry {
        LidarOdometry {
            inner: Odometry::new(OdometryParams {
                min_range,
                max_range,
                ..OdometryParams::default()
            }),
        }
    }

    /// Register the next scan (in its sensor's frame): its pose, row-major 4x4.
    pub fn register(&mut self, cloud: &Cloud) -> Vec<f64> {
        self.inner.register(&cloud.inner).to_matrix().to_vec()
    }
}
