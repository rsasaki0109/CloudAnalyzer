//! LiDAR odometry for scans that come without poses (e.g. from a ROS bag).
//!
//! A scan is registered in steps so that the normal equations of each step
//! can be summed over the worker pool, each worker holding a copy of the
//! local map (`OdometryMap`) fed the same points.

use ca_core::icp::Rigid;
use ca_core::odometry::{LocalMap, Odometry, OdometryParams, Registration as Inner};
use wasm_bindgen::prelude::*;

use crate::Cloud;

/// Registers scans one after another onto a local map of the ones before
/// (see `ca_core::odometry`).
#[wasm_bindgen]
pub struct LidarOdometry {
    inner: Odometry,
}

/// A scan's registration in progress (see `LidarOdometry.begin`).
#[wasm_bindgen]
pub struct Registration {
    inner: Inner,
}

/// A copy of the local map, for summing a step's normal equations elsewhere.
#[wasm_bindgen]
pub struct OdometryMap {
    inner: LocalMap,
}

fn flat(points: &[[f64; 3]]) -> Vec<f64> {
    points.iter().flatten().copied().collect()
}

fn points(flat: &[f64]) -> Vec<[f64; 3]> {
    flat.as_chunks::<3>().0.to_vec()
}

fn rigid(m: &[f64]) -> Result<Rigid, JsError> {
    let m: &[f64; 16] = m
        .try_into()
        .map_err(|_| JsError::new("a transform needs 16 numbers"))?;
    Ok(Rigid::from_matrix(m))
}

/// The normal equations as 42 numbers: the 6x6 matrix row by row, then the 6 right-hand sides.
fn packed(system: ([[f64; 6]; 6], [f64; 6])) -> Vec<f64> {
    let mut out: Vec<f64> = system.0.iter().flatten().copied().collect();
    out.extend(system.1);
    out
}

fn unpacked(equations: &[f64]) -> Result<([[f64; 6]; 6], [f64; 6]), JsError> {
    if equations.len() != 42 {
        return Err(JsError::new("the normal equations are 42 numbers"));
    }
    let mut jtj = [[0.0; 6]; 6];
    for (i, row) in jtj.iter_mut().enumerate() {
        row.copy_from_slice(&equations[6 * i..6 * i + 6]);
    }
    let mut jtr = [0.0; 6];
    jtr.copy_from_slice(&equations[36..]);
    Ok((jtj, jtr))
}

#[wasm_bindgen]
impl LidarOdometry {
    /// Points nearer than `min_range` or further than `max_range` (metres)
    /// are left out; `deskew` undoes the motion during each scan (from its
    /// `time` attribute, else from the order a spinning sensor takes points).
    #[wasm_bindgen(constructor)]
    pub fn new(min_range: f64, max_range: f64, deskew: bool) -> LidarOdometry {
        LidarOdometry {
            inner: Odometry::new(OdometryParams {
                min_range,
                max_range,
                deskew,
                ..OdometryParams::default()
            }),
        }
    }

    /// Register the next scan (in its sensor's frame): its pose, row-major 4x4.
    pub fn register(&mut self, cloud: &Cloud) -> Vec<f64> {
        self.inner.register(&cloud.inner).to_matrix().to_vec()
    }

    /// The local map's voxel size and points per voxel, for copies of it.
    #[wasm_bindgen(getter, js_name = mapVoxel)]
    pub fn map_voxel(&self) -> f64 {
        self.inner.params().map_voxel
    }

    #[wasm_bindgen(getter, js_name = mapPoints)]
    pub fn map_points(&self) -> usize {
        self.inner.params().map_points
    }

    #[wasm_bindgen(getter, js_name = maxRange)]
    pub fn max_range(&self) -> f64 {
        self.inner.params().max_range
    }

    /// Start registering the next scan: then, while `step` says to go on,
    /// sum its normal equations (`equations`, or `OdometryMap.equations` on
    /// copies of the map over the registration's `source` in parts) and
    /// `step`; then `finish`.
    pub fn begin(&self, cloud: &Cloud) -> Registration {
        Registration {
            inner: self.inner.begin(&cloud.inner),
        }
    }

    /// The registration's normal equations from this map (42 numbers).
    pub fn equations(&self, registration: &Registration) -> Vec<f64> {
        packed(self.inner.equations(&registration.inner))
    }

    /// One Gauss-Newton step from the summed normal equations: true when done.
    pub fn step(
        &self,
        registration: &mut Registration,
        equations: &[f64],
    ) -> Result<bool, JsError> {
        Ok(self
            .inner
            .step(&mut registration.inner, unpacked(equations)?))
    }

    /// The scan's pose (row-major 4x4); the scan joins the map (see `placed`).
    pub fn finish(&mut self, registration: Registration) -> Vec<f64> {
        self.inner.finish(registration.inner).to_matrix().to_vec()
    }

    /// The world points the last finished scan added to the map (three per point).
    pub fn placed(&self) -> Vec<f64> {
        flat(self.inner.placed())
    }
}

#[wasm_bindgen]
impl Registration {
    /// Whether the scan is registered at all (the map has points and the scan enough).
    #[wasm_bindgen(getter, js_name = needsAlignment)]
    pub fn needs_alignment(&self) -> bool {
        self.inner.needs_alignment()
    }

    /// The points registered, in the world at the prediction (three per point).
    pub fn source(&self) -> Vec<f64> {
        flat(self.inner.source())
    }

    /// The correction so far (row-major 4x4), to apply to the source points.
    pub fn total(&self) -> Vec<f64> {
        self.inner.total().to_matrix().to_vec()
    }

    /// Where the sensor is (rotations are about it), the pair threshold
    /// (squared) and the kernel's scale: `[x, y, z, threshold_sq, kernel]`.
    pub fn terms(&self) -> Vec<f64> {
        let c = self.inner.center();
        vec![
            c[0],
            c[1],
            c[2],
            self.inner.threshold_sq(),
            self.inner.kernel(),
        ]
    }
}

#[wasm_bindgen]
impl OdometryMap {
    #[wasm_bindgen(constructor)]
    pub fn new(voxel: f64, max_points: usize) -> OdometryMap {
        OdometryMap {
            inner: LocalMap::new(voxel, max_points),
        }
    }

    /// Add world points (three each) and drop the voxels further than `range` from `origin`.
    pub fn update(&mut self, placed: &[f64], origin: &[f64], range: f64) -> Result<(), JsError> {
        self.inner.insert(&points(placed));
        let origin: [f64; 3] = origin
            .try_into()
            .map_err(|_| JsError::new("the origin needs three numbers"))?;
        self.inner.retain_within(origin, range);
        Ok(())
    }

    /// The normal equations (42 numbers) of `source` points (three each) moved
    /// by `total` (row-major 4x4), with `terms` as `Registration.terms` gives them.
    pub fn equations(
        &self,
        source: &[f64],
        total: &[f64],
        terms: &[f64],
    ) -> Result<Vec<f64>, JsError> {
        if terms.len() != 5 {
            return Err(JsError::new("terms are five numbers"));
        }
        let center = [terms[0], terms[1], terms[2]];
        let system = self.inner.normal_equations(
            &points(source),
            &rigid(total)?,
            center,
            terms[3],
            terms[4],
        );
        Ok(packed(system))
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}
