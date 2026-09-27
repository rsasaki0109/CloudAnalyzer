//! WebAssembly bindings for the CloudAnalyzer Web viewer.

use ca_core::{DistanceStats, PointCloud};
use wasm_bindgen::prelude::*;

/// A loaded point cloud. Coordinates stay in `f64` on the Rust side.
#[wasm_bindgen]
pub struct Cloud {
    inner: PointCloud,
}

#[wasm_bindgen]
impl Cloud {
    /// Parse a file; the format is detected from `name` and the leading bytes.
    pub fn parse(name: &str, bytes: &[u8]) -> Result<Cloud, JsError> {
        Ok(Cloud {
            inner: ca_core::read(name, bytes)?,
        })
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// `[minX, minY, minZ, maxX, maxY, maxZ]`.
    pub fn bounds(&self) -> Vec<f64> {
        self.inner
            .bounds()
            .map(|b| b.min.into_iter().chain(b.max).collect())
            .unwrap_or_default()
    }

    /// Shift that brings this cloud near the origin (zero for small coordinates).
    #[wasm_bindgen(js_name = suggestedShift)]
    pub fn suggested_shift(&self) -> Vec<f64> {
        self.inner.suggested_shift().to_vec()
    }

    /// Interleaved `xyz` positions minus `shift`, narrowed to `f32` for rendering.
    pub fn positions(&self, shift: &[f64]) -> Result<Vec<f32>, JsError> {
        let shift: [f64; 3] = shift
            .try_into()
            .map_err(|_| JsError::new("shift must have 3 components"))?;
        Ok(self
            .inner
            .positions
            .iter()
            .flat_map(|p| (0..3).map(move |i| (p[i] - shift[i]) as f32))
            .collect())
    }

    /// Interleaved `rgb` bytes, or `undefined` when the file has no colors.
    pub fn colors(&self) -> Option<Vec<u8>> {
        self.inner
            .colors
            .as_ref()
            .map(|c| c.iter().flatten().copied().collect())
    }
}

/// Result of a cloud-to-cloud distance computation.
#[wasm_bindgen]
pub struct C2cResult {
    distances: Vec<f32>,
    stats: DistanceStats,
}

#[wasm_bindgen]
impl C2cResult {
    /// Per-point distances of the compared cloud, in its point order.
    pub fn distances(&self) -> Vec<f32> {
        self.distances.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn count(&self) -> usize {
        self.stats.count
    }
    #[wasm_bindgen(getter)]
    pub fn min(&self) -> f64 {
        self.stats.min
    }
    #[wasm_bindgen(getter)]
    pub fn max(&self) -> f64 {
        self.stats.max
    }
    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> f64 {
        self.stats.mean
    }
    #[wasm_bindgen(getter)]
    pub fn rms(&self) -> f64 {
        self.stats.rms
    }
    #[wasm_bindgen(getter, js_name = stdDev)]
    pub fn std_dev(&self) -> f64 {
        self.stats.std_dev
    }
    #[wasm_bindgen(getter)]
    pub fn median(&self) -> f64 {
        self.stats.median
    }
}

/// Distance from every point of `compared` to its nearest neighbour in `reference`.
#[wasm_bindgen(js_name = cloudToCloud)]
pub fn cloud_to_cloud(compared: &Cloud, reference: &Cloud) -> Result<C2cResult, JsError> {
    let distances = ca_core::cloud_to_cloud(&compared.inner, &reference.inner)
        .ok_or_else(|| JsError::new("reference cloud is empty"))?;
    let stats = DistanceStats::from_distances(&distances)
        .ok_or_else(|| JsError::new("compared cloud is empty"))?;
    Ok(C2cResult {
        distances: distances.iter().map(|&d| d as f32).collect(),
        stats,
    })
}
