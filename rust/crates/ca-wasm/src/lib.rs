//! WebAssembly bindings for the CloudAnalyzer Web viewer.

use ca_core::icp::{IcpMetric, IcpParams, Rigid};
use ca_core::octree::{NO_CHILD, Octree, OctreeParams};
use ca_core::{DistanceStats, PointCloud};
use wasm_bindgen::prelude::*;

/// Numbers per node in [`Cloud::lod_nodes`].
const NODE_STRIDE: usize = 15;

/// A loaded point cloud. Coordinates stay in `f64` on the Rust side.
///
/// Points are reordered at load time into the order of a level-of-detail
/// octree, where each node's points are contiguous. Everything this type
/// returns per point (positions, colors, C2C distances) uses that order.
#[wasm_bindgen]
pub struct Cloud {
    inner: PointCloud,
    lod: Octree,
}

#[wasm_bindgen]
impl Cloud {
    /// Parse a file; the format is detected from `name` and the leading bytes.
    pub fn parse(name: &str, bytes: &[u8]) -> Result<Cloud, JsError> {
        let mut inner = ca_core::read(name, bytes)?;
        let lod = build_lod(&mut inner)?;
        Ok(Cloud { inner, lod })
    }

    /// Apply a rigid transform (row-major 4x4) to every point and rebuild the
    /// octree, which also changes the point order.
    pub fn transform(&mut self, matrix: &[f64]) -> Result<(), JsError> {
        let matrix: &[f64; 16] = matrix
            .try_into()
            .map_err(|_| JsError::new("matrix must have 16 entries"))?;
        let rigid = Rigid::from_matrix(matrix);
        for p in &mut self.inner.positions {
            *p = rigid.apply(p);
        }
        self.lod = build_lod(&mut self.inner)?;
        Ok(())
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
            .flat_map(|p| (0..3).map(move |a| (p[a] - shift[a]) as f32))
            .collect())
    }

    /// Exact `f64` coordinates of point `index` (octree order), or
    /// `undefined` when out of range.
    pub fn point(&self, index: usize) -> Option<Vec<f64>> {
        self.inner.positions.get(index).map(|p| p.to_vec())
    }

    /// Interleaved `rgb` bytes, or `undefined` when the file has no colors.
    pub fn colors(&self) -> Option<Vec<u8>> {
        self.inner
            .colors
            .as_ref()
            .map(|c| c.iter().flatten().copied().collect())
    }

    /// Octree nodes, 15 numbers each: `start, count, minX, minY, minZ, size,
    /// level, child0..child7` (children are node indices or -1). `min` is in
    /// original coordinates; subtract the shift used for `positions`.
    #[wasm_bindgen(js_name = lodNodes)]
    pub fn lod_nodes(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.lod.nodes.len() * NODE_STRIDE);
        for n in &self.lod.nodes {
            out.extend([n.start as f64, n.count as f64]);
            out.extend(n.min);
            out.extend([n.size, n.level as f64]);
            out.extend(
                n.children
                    .map(|c| if c == NO_CHILD { -1.0 } else { c as f64 }),
            );
        }
        out
    }

    /// Subsampling lattice resolution per node edge; a node's point spacing is
    /// about `size / lodGrid`.
    #[wasm_bindgen(getter, js_name = lodGrid)]
    pub fn lod_grid(&self) -> u32 {
        self.lod.grid
    }
}

/// Reorder `cloud` in place into octree order (colors follow) and return
/// the octree without the permutation, which the viewer does not need.
fn build_lod(cloud: &mut PointCloud) -> Result<Octree, JsError> {
    let mut lod = Octree::build_in_place(&mut cloud.positions, OctreeParams::default())
        .ok_or_else(|| JsError::new("cloud is empty or too large"))?;
    if let Some(colors) = cloud.colors.as_mut() {
        *colors = lod.order.iter().map(|&i| colors[i as usize]).collect();
    }
    lod.order = Vec::new();
    Ok(lod)
}

/// Outcome of [`register_icp`].
#[wasm_bindgen]
pub struct IcpOutcome {
    matrix: [f64; 16],
    #[wasm_bindgen(readonly, js_name = rmsInitial)]
    pub rms_initial: f64,
    #[wasm_bindgen(readonly, js_name = rmsFinal)]
    pub rms_final: f64,
    #[wasm_bindgen(readonly)]
    pub iterations: usize,
    #[wasm_bindgen(readonly)]
    pub converged: bool,
}

#[wasm_bindgen]
impl IcpOutcome {
    /// Row-major 4x4 transform taking the moving cloud onto the reference.
    pub fn matrix(&self) -> Vec<f64> {
        self.matrix.to_vec()
    }
}

/// Register `moving` onto `reference` with ICP. The clouds are not changed;
/// apply the result with [`Cloud::transform`].
#[wasm_bindgen(js_name = registerIcp)]
pub fn register_icp(
    moving: &Cloud,
    reference: &Cloud,
    max_iterations: usize,
    overlap: f64,
    match_centroids: bool,
    point_to_plane: bool,
) -> Result<IcpOutcome, JsError> {
    let params = IcpParams {
        metric: if point_to_plane {
            IcpMetric::PointToPlane
        } else {
            IcpMetric::PointToPoint
        },
        max_iterations,
        overlap,
        match_centroids,
        ..IcpParams::default()
    };
    let result = ca_core::icp::icp(&moving.inner, &reference.inner, params)
        .ok_or_else(|| JsError::new("ICP needs non-empty clouds and at least 3 pairs"))?;
    Ok(IcpOutcome {
        matrix: result.transform.to_matrix(),
        rms_initial: result.rms_initial,
        rms_final: result.rms_final,
        iterations: result.iterations,
        converged: result.converged,
    })
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
    summarize_distances(&distances)
}

fn from_flat(xyz: &[f64]) -> Result<PointCloud, JsError> {
    if !xyz.len().is_multiple_of(3) {
        return Err(JsError::new("positions length must be a multiple of 3"));
    }
    Ok(PointCloud {
        positions: xyz.as_chunks::<3>().0.to_vec(),
        colors: None,
    })
}

/// Nearest-neighbour distances for a batch of interleaved `xyz` queries.
/// Used by the worker pool, where each worker handles one slice of the
/// compared cloud against its own copy of the reference.
#[wasm_bindgen(js_name = nearestDistances)]
pub fn nearest_distances(reference: &[f64], queries: &[f64]) -> Result<Vec<f64>, JsError> {
    ca_core::cloud_to_cloud(&from_flat(queries)?, &from_flat(reference)?)
        .ok_or_else(|| JsError::new("reference cloud is empty"))
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
pub fn summarize_distances(distances: &[f64]) -> Result<C2cResult, JsError> {
    let stats = DistanceStats::from_distances(distances)
        .ok_or_else(|| JsError::new("compared cloud is empty"))?;
    Ok(C2cResult {
        distances: distances.iter().map(|&d| d as f32).collect(),
        stats,
    })
}
