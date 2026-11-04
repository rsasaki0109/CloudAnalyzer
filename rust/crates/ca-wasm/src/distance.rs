//! Cloud-to-cloud distances, map quality and distance summaries.

use super::*;

/// Map quality of `estimate` against the ground-truth map `truth` by
/// voxel Gaussians (see `ca_core::map_quality`): `[AWD, SCS, voxels]`.
#[wasm_bindgen(js_name = mapQuality)]
pub fn map_quality(
    estimate: &Cloud,
    truth: &Cloud,
    voxel: f64,
    min_points: usize,
    simd: Option<bool>,
) -> Vec<f64> {
    let s = ca_core::map_quality::voxel_scores_with_simd(
        &estimate.inner.positions,
        &truth.inner.positions,
        voxel,
        min_points,
        simd.unwrap_or(false),
    );
    vec![s.awd, s.scs, s.voxels as f64]
}

/// Immutable voxel statistics for one reference and parameter set.
#[wasm_bindgen]
pub struct VoxelReference {
    inner: ca_core::map_quality::VoxelReference,
}

#[wasm_bindgen]
impl VoxelReference {
    #[wasm_bindgen(constructor)]
    pub fn new(truth: &Cloud, voxel: f64, min_points: usize, simd: Option<bool>) -> Self {
        Self {
            inner: ca_core::map_quality::VoxelReference::new(
                &truth.inner.positions,
                voxel,
                min_points,
                simd.unwrap_or(false),
            ),
        }
    }

    #[wasm_bindgen(getter, js_name = retainedBytes)]
    pub fn retained_bytes(&self) -> usize {
        self.inner.retained_bytes()
    }
    #[wasm_bindgen(getter, js_name = rootsReused)]
    pub fn roots_reused(&self) -> usize {
        self.inner.roots_reused()
    }
    #[wasm_bindgen(getter, js_name = rootsComputed)]
    pub fn roots_computed(&self) -> usize {
        self.inner.roots_computed()
    }

    #[wasm_bindgen(getter, js_name = neighborVoxels)]
    pub fn neighbor_voxels(&self) -> usize {
        self.inner.neighbor_voxels()
    }

    #[wasm_bindgen(getter, js_name = neighborsUsed)]
    pub fn neighbors_used(&self) -> usize {
        self.inner.neighbors_used()
    }

    pub fn compute(&mut self, estimate: &Cloud) -> Vec<f64> {
        let s = self.inner.compute(&estimate.inner.positions);
        vec![s.awd, s.scs, s.voxels as f64]
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
pub fn cloud_to_cloud(
    compared: &Cloud,
    reference: &Cloud,
    simd: Option<bool>,
    sort_median: Option<bool>,
) -> Result<C2cResult, JsError> {
    let distances = ca_core::distance::cloud_to_cloud_with_simd(
        &compared.inner,
        &reference.inner,
        simd.unwrap_or(false),
    )
    .ok_or_else(|| JsError::new("reference cloud is empty"))?;
    summarize_distances(&distances, sort_median)
}

pub(crate) fn from_flat(xyz: &[f64]) -> Result<PointCloud, JsError> {
    if !xyz.len().is_multiple_of(3) {
        return Err(JsError::new("positions length must be a multiple of 3"));
    }
    Ok(PointCloud {
        positions: xyz.as_chunks::<3>().0.to_vec(),
        colors: None,
        attributes: Vec::new(),
    })
}

#[wasm_bindgen(js_name = nearestDistances)]
pub fn nearest_distances(
    reference: &[f64],
    queries: &[f64],
    simd: Option<bool>,
) -> Result<Vec<f64>, JsError> {
    ca_core::distance::cloud_to_cloud_with_simd(
        &from_flat(queries)?,
        &from_flat(reference)?,
        simd.unwrap_or(false),
    )
    .ok_or_else(|| JsError::new("reference cloud is empty"))
}

/// A pool lane's exact immutable C2C inputs and prepared nearest-neighbour tree.
#[wasm_bindgen]
pub struct C2cPrepared {
    inner: ca_core::distance::PreparedC2c,
}

#[wasm_bindgen]
impl C2cPrepared {
    #[wasm_bindgen(constructor)]
    pub fn new(reference: &[f64], queries: &[f64]) -> Result<Self, JsError> {
        let reference = from_flat(reference)?;
        let queries = from_flat(queries)?;
        let inner = ca_core::distance::PreparedC2c::new(queries.positions, &reference.positions)
            .ok_or_else(|| JsError::new("reference cloud is empty"))?;
        Ok(Self { inner })
    }

    pub fn distances(&self, simd: Option<bool>) -> Vec<f64> {
        self.inner.distances(simd.unwrap_or(false))
    }
}

/// Whether this build supports ordinary C2C's optional paired f64 leaves.
#[wasm_bindgen(js_name = c2cSimdEnabled)]
pub fn c2c_simd_enabled() -> bool {
    cfg!(all(target_arch = "wasm32", target_feature = "simd128"))
}

/// A C2C job split into independent, spatially compact parts
/// (see [`ca_core::partition_c2c`]). Each part can run on its own worker via
/// [`nearest_distances`], and its results belong at [`C2cPlan::take_query_indices`].
#[wasm_bindgen]
pub struct C2cPlan {
    parts: Vec<(Vec<u32>, Vec<f64>, Vec<f64>)>,
}

#[wasm_bindgen]
impl C2cPlan {
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.parts.len()
    }

    // The getters below hand the buffers over instead of copying them, so
    // each may be called once per part.

    /// Indices into the compared cloud for part `k`.
    #[wasm_bindgen(js_name = takeQueryIndices)]
    pub fn take_query_indices(&mut self, k: usize) -> Vec<u32> {
        std::mem::take(&mut self.parts[k].0)
    }

    /// Interleaved `xyz` of the compared points of part `k`.
    #[wasm_bindgen(js_name = takeQueries)]
    pub fn take_queries(&mut self, k: usize) -> Vec<f64> {
        std::mem::take(&mut self.parts[k].1)
    }

    /// Interleaved `xyz` of the reference points part `k` needs.
    #[wasm_bindgen(js_name = takeReference)]
    pub fn take_reference(&mut self, k: usize) -> Vec<f64> {
        std::mem::take(&mut self.parts[k].2)
    }
}

/// Split a C2C job into up to `parts` pieces for parallel workers.
#[wasm_bindgen(js_name = planCloudToCloud)]
pub fn plan_cloud_to_cloud(
    compared: &Cloud,
    reference: &Cloud,
    parts: usize,
) -> Result<C2cPlan, JsError> {
    let gather = |cloud: &PointCloud, idx: &[u32]| -> Vec<f64> {
        idx.iter()
            .flat_map(|&i| cloud.positions[i as usize])
            .collect()
    };
    let parts = ca_core::partition_c2c(&compared.inner, &reference.inner, parts)
        .ok_or_else(|| JsError::new("reference cloud is empty"))?
        .into_iter()
        .map(|part| {
            let queries = gather(&compared.inner, &part.queries);
            let reference = gather(&reference.inner, &part.reference);
            (part.queries, queries, reference)
        })
        .collect();
    Ok(C2cPlan { parts })
}

/// Summary statistics (and `f32` copies for coloring) of per-point distances.
#[wasm_bindgen(js_name = summarizeDistances)]
pub fn summarize_distances(
    distances: &[f64],
    sort_median: Option<bool>,
) -> Result<C2cResult, JsError> {
    let stats =
        DistanceStats::from_distances_with_median_sort(distances, sort_median.unwrap_or(false))
            .ok_or_else(|| JsError::new("compared cloud is empty"))?;
    Ok(C2cResult {
        distances: distances.iter().map(|&d| d as f32).collect(),
        stats,
    })
}
