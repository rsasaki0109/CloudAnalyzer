//! Browser M3C2: core points are sliced while exact immutable trees preserve
//! neighbour iteration and f64 accumulation order across workers and jobs.

use crate::{Cloud, build_lod};
use ca_core::kdtree::KdTree;
#[cfg(test)]
use ca_core::m3c2::m3c2;
use ca_core::m3c2::{M3c2, M3c2PairScratch, M3c2Params, M3c2Result, m3c2_with_projection};
use ca_core::{Attribute, AttributeValues, PointCloud};
use wasm_bindgen::prelude::*;

/// Own exact trees and an optional bounded normal set; no source Cloud buffers.
#[wasm_bindgen]
pub struct M3c2Prepared {
    reference: KdTree,
    compared: KdTree,
    normals: Option<NormalCache>,
    normal_cache_hit: bool,
}

// Keep the validity tag: Some(NaN) and a failed fit must remain distinct.
const NORMAL_BYTES_PER_CORE: usize =
    std::mem::size_of::<[f64; 3]>() + std::mem::size_of::<Option<[f64; 3]>>();
const MAX_NORMAL_CACHE_BYTES: usize = 64 * 1024 * 1024;
const USE_M3C2_PAIR_EIGEN: bool = true;

fn fresh_prepared_values(
    m: &M3c2<'_>,
    cores: &[[f64; 3]],
    shared: bool,
    mut normals: Option<&mut Vec<Option<[f64; 3]>>>,
) -> Vec<f64> {
    if USE_M3C2_PAIR_EIGEN && cfg!(all(target_arch = "wasm32", target_feature = "simd128")) {
        let mut scratch = M3c2PairScratch::default();
        let mut values = Vec::with_capacity(cores.len() * 2);
        let mut append = |(distance, lod, normal)| {
            values.extend_from_slice(&[distance, lod]);
            if let Some(normals) = &mut normals {
                normals.push(normal);
            }
        };
        let (pairs, tail) = cores.as_chunks::<2>();
        for pair in pairs {
            for result in m.at_pair_with_search(pair, &mut scratch, shared) {
                append(result);
            }
        }
        if let Some(core) = tail.first() {
            append(m.at_single_with_pair_scratch(core, &mut scratch, shared));
        }
        return values;
    }
    let (mut scratch, mut projection) = (Vec::new(), Vec::new());
    cores
        .iter()
        .flat_map(|point| {
            let (distance, lod, normal) =
                m.at_with_search(point, &mut scratch, &mut projection, shared);
            if let Some(normals) = &mut normals {
                normals.push(normal);
            }
            [distance, lod]
        })
        .collect()
}

struct NormalCache {
    cores: Box<[[f64; 3]]>,
    normals: Vec<Option<[f64; 3]>>,
    radius: u64,
    bounds: bool,
}

impl NormalCache {
    fn matches(&self, cores: &[[f64; 3]], radius: f64, bounds: bool) -> bool {
        self.radius == radius.to_bits()
            && self.bounds == bounds
            && self.cores.len() == cores.len()
            && self
                .cores
                .iter()
                .zip(cores)
                .all(|(a, b)| a.map(f64::to_bits) == b.map(f64::to_bits))
    }
}

#[wasm_bindgen]
impl M3c2Prepared {
    #[wasm_bindgen(constructor)]
    pub fn new(compared: &Cloud, reference: &Cloud) -> Result<Self, JsError> {
        Ok(Self {
            reference: KdTree::new(&reference.inner.positions)
                .ok_or_else(|| JsError::new("M3C2 needs non-empty inputs"))?,
            compared: KdTree::new(&compared.inner.positions)
                .ok_or_else(|| JsError::new("M3C2 needs non-empty inputs"))?,
            normals: None,
            normal_cache_hit: false,
        })
    }

    #[wasm_bindgen(js_name = fromSnapshots)]
    pub fn from_snapshots(
        reference: &[u8],
        compared: &[u8],
        reference_count: usize,
        compared_count: usize,
    ) -> Result<Self, JsError> {
        let reference = KdTree::from_snapshot(reference, reference_count).map_err(JsError::new)?;
        let compared = KdTree::from_snapshot(compared, compared_count).map_err(JsError::new)?;
        if reference.len() != reference_count || compared.len() != compared_count {
            return Err(JsError::new("M3C2 snapshot point count changed"));
        }
        Ok(Self {
            reference,
            compared,
            normals: None,
            normal_cache_hit: false,
        })
    }

    #[wasm_bindgen(js_name = referenceSnapshot)]
    pub fn reference_snapshot(&self) -> Vec<u8> {
        self.reference.snapshot()
    }

    #[wasm_bindgen(js_name = comparedSnapshot)]
    pub fn compared_snapshot(&self) -> Vec<u8> {
        self.compared.snapshot()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &self,
        core: &[f64],
        normal_radius: f64,
        projection_radius: f64,
        max_depth: f64,
        radius_bounds: Option<bool>,
        shared_search: Option<bool>,
        fused_projection: Option<bool>,
    ) -> Result<Vec<f64>, JsError> {
        let m = M3c2::with_trees(
            &self.reference,
            &self.compared,
            params(normal_radius, projection_radius, max_depth),
            radius_bounds.unwrap_or(false),
        )
        .ok_or_else(|| JsError::new("M3C2 needs non-empty inputs and positive radii"))?
        .with_fused_projection(fused_projection.unwrap_or(false));
        Ok(fresh_prepared_values(
            &m,
            triples(core)?,
            shared_search.unwrap_or(true),
            None,
        ))
    }

    /// Cache only exact fitted normals for one ordered core set. The caller
    /// divides its pair-wide budget among lanes; a local cap also bounds direct use.
    #[wasm_bindgen(js_name = computeCached)]
    // Keep existing JS positional arguments; append the independent search control.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_cached(
        &mut self,
        core: &[f64],
        normal_radius: f64,
        projection_radius: f64,
        max_depth: f64,
        radius_bounds: Option<bool>,
        max_cached_cores: usize,
        shared_search: Option<bool>,
        fused_projection: Option<bool>,
    ) -> Result<Vec<f64>, JsError> {
        self.normal_cache_hit = false;
        let core = triples(core)?;
        let bounds = radius_bounds.unwrap_or(false);
        let m = M3c2::with_trees(
            &self.reference,
            &self.compared,
            params(normal_radius, projection_radius, max_depth),
            bounds,
        )
        .ok_or_else(|| JsError::new("M3C2 needs non-empty inputs and positive radii"))?;
        let eligible = !core.is_empty()
            && core.len() <= max_cached_cores.min(MAX_NORMAL_CACHE_BYTES / NORMAL_BYTES_PER_CORE);
        self.normal_cache_hit = eligible
            && self
                .normals
                .as_ref()
                .is_some_and(|cache| cache.matches(core, normal_radius, bounds));
        let m = m.with_fused_projection(fused_projection.unwrap_or(self.normal_cache_hit));
        let mut scratch = Vec::new();
        if self.normal_cache_hit {
            let cache = self.normals.as_ref().unwrap();
            return Ok(core
                .iter()
                .zip(&cache.normals)
                .flat_map(|(point, &normal)| {
                    let (distance, lod, _) = m.at_with_normal(point, normal, &mut scratch);
                    [distance, lod]
                })
                .collect());
        }
        // Release the old set before allocating a replacement, including when
        // the current core set exceeds the budget. No two sets remain resident.
        self.normals = None;
        let mut normals = eligible.then(|| Vec::with_capacity(core.len()));
        let values =
            fresh_prepared_values(&m, core, shared_search.unwrap_or(true), normals.as_mut());
        if let Some(normals) = normals {
            self.normals = Some(NormalCache {
                cores: core.into(),
                normals,
                radius: normal_radius.to_bits(),
                bounds,
            });
        }
        Ok(values)
    }

    #[wasm_bindgen(getter, js_name = normalCacheHit)]
    pub fn normal_cache_hit(&self) -> bool {
        self.normal_cache_hit
    }

    #[wasm_bindgen(getter, js_name = normalCacheBytes)]
    pub fn normal_cache_bytes(&self) -> usize {
        self.normals
            .as_ref()
            .map_or(0, |cache| cache.cores.len() * NORMAL_BYTES_PER_CORE)
    }
}

/// Owned core-point snapshot, including all compared-cloud attributes.
#[wasm_bindgen]
pub struct M3c2Plan {
    core: PointCloud,
}

#[wasm_bindgen]
impl M3c2Plan {
    #[wasm_bindgen(constructor)]
    pub fn new(compared: &Cloud, core_spacing: f64) -> Self {
        let core = if core_spacing > 0.0 {
            compared.inner.select(&ca_core::filter::voxel_subsample(
                &compared.inner,
                core_spacing,
            ))
        } else {
            compared.inner.clone()
        };
        Self { core }
    }

    /// Stable original point indices; geometry must be unchanged when reused.
    #[wasm_bindgen(js_name = selectedIndices)]
    pub fn selected_indices(compared: &Cloud, core_spacing: f64) -> Result<Vec<u32>, JsError> {
        if compared.inner.len() > u32::MAX as usize {
            return Err(JsError::new("M3C2 core indices exceed u32 capacity"));
        }
        let indices = if core_spacing > 0.0 {
            ca_core::filter::voxel_subsample(&compared.inner, core_spacing)
        } else {
            (0..compared.inner.len()).collect()
        };
        Ok(indices.into_iter().map(|i| i as u32).collect())
    }

    /// Take a fresh coordinate/attribute snapshot from a validated selection.
    #[wasm_bindgen(js_name = fromIndices)]
    pub fn from_indices(compared: &Cloud, indices: &[u32]) -> Result<Self, JsError> {
        if indices.iter().any(|&i| i as usize >= compared.inner.len())
            || indices.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(JsError::new("Invalid ordered M3C2 core indices"));
        }
        let indices: Vec<_> = indices.iter().map(|&i| i as usize).collect();
        Ok(Self {
            core: compared.inner.select(&indices),
        })
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.core.len()
    }

    /// Coordinates before result indexing, in original f64 precision.
    pub fn points(&self, first: usize, count: usize) -> Result<Vec<f64>, JsError> {
        let end = first
            .checked_add(count)
            .filter(|&end| end <= self.length())
            .ok_or_else(|| JsError::new("M3C2 core range is out of bounds"))?;
        Ok(self.core.positions[first..end].as_flattened().to_vec())
    }

    /// Small contiguous blocks distributed cyclically across workers. Only
    /// core order changes; each core still sees the complete original inputs.
    #[wasm_bindgen(js_name = interleavedPoints)]
    pub fn interleaved_points(
        &self,
        part: usize,
        parts: usize,
        block: usize,
    ) -> Result<Vec<f64>, JsError> {
        let step = parts
            .checked_mul(block)
            .filter(|&step| step > 0 && part < parts)
            .ok_or_else(|| JsError::new("Invalid M3C2 core partition"))?;
        let full = self.length() / step * step;
        let tail = self.length() - full;
        let tail_count = tail / parts + usize::from(part < tail % parts);
        let tail_first = full + part * (tail / parts) + part.min(tail % parts);
        let count = full / parts + tail_count;
        let mut points = Vec::with_capacity(count * 3);
        for first in (part * block..full).step_by(step) {
            points.extend_from_slice(self.core.positions[first..first + block].as_flattened());
        }
        // Balance the last incomplete cycle so lane sizes differ by at most
        // one core, including jobs near the minimum cores-per-worker limit.
        points.extend_from_slice(
            self.core.positions[tail_first..tail_first + tail_count].as_flattened(),
        );
        Ok(points)
    }

    /// Complete `distance, lod95` pairs in core order; consumes the snapshot.
    /// Serial and parallel results use the same attribute/index construction.
    pub fn finish(self, results: &[f64]) -> Result<Cloud, JsError> {
        if results.len() != self.length() * 2 {
            return Err(JsError::new("M3C2 results do not match the core points"));
        }
        let mut inner = self.core;
        inner
            .attributes
            .retain(|a| !matches!(a.name.as_str(), "m3c2_distance" | "lod95" | "significant"));
        let pairs = results.as_chunks::<2>().0;
        inner.attributes.push(Attribute {
            name: "m3c2_distance".into(),
            values: AttributeValues::F32(pairs.iter().map(|p| p[0] as f32).collect()),
        });
        inner.attributes.push(Attribute {
            name: "lod95".into(),
            values: AttributeValues::F32(pairs.iter().map(|p| p[1] as f32).collect()),
        });
        inner.attributes.push(Attribute {
            name: "significant".into(),
            values: AttributeValues::U8(
                pairs.iter().map(|p| u8::from(p[0].abs() > p[1])).collect(),
            ),
        });
        let lod = build_lod(&mut inner)?;
        Ok(Cloud {
            inner,
            lod,
            layout: None,
        })
    }
}

fn params(normal_radius: f64, projection_radius: f64, max_depth: f64) -> M3c2Params {
    M3c2Params {
        normal_radius,
        projection_radius,
        max_depth,
        ..M3c2Params::default()
    }
}

fn pairs(result: M3c2Result) -> Vec<f64> {
    result
        .distance
        .into_iter()
        .zip(result.lod95)
        .flat_map(|(d, lod)| [d, lod])
        .collect()
}

fn triples(xyz: &[f64]) -> Result<&[[f64; 3]], JsError> {
    let (points, remainder) = xyz.as_chunks::<3>();
    if !remainder.is_empty() {
        return Err(JsError::new("positions length must be a multiple of 3"));
    }
    Ok(points)
}

/// One independent worker's core points against the complete input clouds.
#[wasm_bindgen(js_name = m3c2Distances)]
#[allow(clippy::too_many_arguments)]
pub fn m3c2_distances(
    reference: &[f64],
    compared: &[f64],
    core: &[f64],
    normal_radius: f64,
    projection_radius: f64,
    max_depth: f64,
    radius_bounds: Option<bool>,
    shared_search: Option<bool>,
    fused_projection: Option<bool>,
) -> Result<Vec<f64>, JsError> {
    // Borrow the binding's buffers instead of making another owned copy of
    // each complete cloud in the worker's WASM heap.
    m3c2_with_projection(
        triples(core)?,
        triples(reference)?,
        triples(compared)?,
        params(normal_radius, projection_radius, max_depth),
        radius_bounds.unwrap_or(false),
        shared_search.unwrap_or(true),
        fused_projection.unwrap_or(false),
    )
    .map(pairs)
    .ok_or_else(|| JsError::new("M3C2 needs non-empty clouds and positive radii"))
}

/// Core lane computed on the analysis worker without copying complete input
/// clouds through JS/WASM bindings. This synchronous call borrows the source
/// clouds only until it returns, before the parallel driver's first yield.
#[wasm_bindgen(js_name = m3c2CloudDistances)]
#[allow(clippy::too_many_arguments)]
pub fn m3c2_cloud_distances(
    compared: &Cloud,
    reference: &Cloud,
    core: &[f64],
    normal_radius: f64,
    projection_radius: f64,
    max_depth: f64,
    radius_bounds: Option<bool>,
    shared_search: Option<bool>,
    fused_projection: Option<bool>,
) -> Result<Vec<f64>, JsError> {
    m3c2_with_projection(
        triples(core)?,
        &reference.inner.positions,
        &compared.inner.positions,
        params(normal_radius, projection_radius, max_depth),
        radius_bounds.unwrap_or(false),
        shared_search.unwrap_or(true),
        fused_projection.unwrap_or(false),
    )
    .map(pairs)
    .ok_or_else(|| JsError::new("M3C2 needs non-empty clouds and positive radii"))
}

/// M3C2 at compared-cloud core points (all, or one per core-spacing voxel),
/// returned as an indexed cloud with distance, LoD95 and significance.
#[wasm_bindgen(js_name = computeM3c2)]
#[allow(clippy::too_many_arguments)]
pub fn compute_m3c2(
    compared: &Cloud,
    reference: &Cloud,
    normal_radius: f64,
    projection_radius: f64,
    max_depth: f64,
    core_spacing: f64,
    radius_bounds: Option<bool>,
    shared_search: Option<bool>,
    fused_projection: Option<bool>,
) -> Result<Cloud, JsError> {
    let plan = M3c2Plan::new(compared, core_spacing);
    let result = m3c2_with_projection(
        &plan.core.positions,
        &reference.inner.positions,
        &compared.inner.positions,
        params(normal_radius, projection_radius, max_depth),
        radius_bounds.unwrap_or(false),
        shared_search.unwrap_or(true),
        fused_projection.unwrap_or(false),
    )
    .ok_or_else(|| JsError::new("M3C2 needs non-empty clouds and positive radii"))?;
    plan.finish(&pairs(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud(lift: f64) -> Cloud {
        let n = 323;
        Cloud::unindexed(PointCloud {
            positions: (0..n)
                .map(|i| {
                    let x = (i % 19) as f64 * 0.1;
                    let y = (i / 19) as f64 * 0.1;
                    // UTM offsets, a slope and an isolated undefined core.
                    if i == n - 1 {
                        [400_000.0, 4_000_000.0, 50.0 + lift]
                    } else {
                        [368_000.0 + x, 3_955_000.0 + y, x * 0.2 + lift]
                    }
                })
                .collect(),
            colors: Some((0..n).map(|i| [i as u8, 10, 20]).collect()),
            attributes: vec![
                Attribute {
                    name: "intensity".into(),
                    values: AttributeValues::F32((0..n).map(|i| i as f32).collect()),
                },
                Attribute {
                    name: "m3c2_distance".into(),
                    values: AttributeValues::F32(vec![99.0; n]),
                },
            ],
        })
    }

    #[test]
    fn selected_core_indices_rebuild_exact_snapshots_with_current_attributes() {
        let mut compared = cloud(0.07);
        for spacing in [0.0, -0.0, -1.0, f64::NAN, 0.08, 0.17, 1.0, f64::INFINITY] {
            let indices = M3c2Plan::selected_indices(&compared, spacing).unwrap();
            for revision in [10, 20] {
                compared.inner.colors.as_mut().unwrap()[0] = [revision, 30, 40];
                if let AttributeValues::F32(values) = &mut compared.inner.attributes[0].values {
                    values[0] = f32::from(revision);
                }
                let old = M3c2Plan::new(&compared, spacing);
                let selected = M3c2Plan::from_indices(&compared, &indices).unwrap();
                assert_eq!(old.core.positions, selected.core.positions);
                assert_eq!(old.core.colors, selected.core.colors);
                assert_eq!(old.core.attributes, selected.core.attributes);
                for interleaved in [false, true] {
                    for lane in 0..3 {
                        let points = |plan: &M3c2Plan| {
                            if interleaved {
                                plan.interleaved_points(lane, 3, 17).unwrap()
                            } else {
                                let per = plan.length().div_ceil(3);
                                let first = (lane * per).min(plan.length());
                                plan.points(first, per.min(plan.length() - first)).unwrap()
                            }
                        };
                        assert_eq!(points(&old), points(&selected));
                    }
                }
            }
        }
    }

    #[test]
    fn worker_slices_keep_exact_f64_results_and_indexed_attributes() {
        let reference = cloud(0.0);
        let compared = cloud(0.07);
        for spacing in [0.0, 0.17] {
            let plan = M3c2Plan::new(&compared, spacing);
            let serial = m3c2(
                &plan.core.positions,
                &reference.inner.positions,
                &compared.inner.positions,
                params(0.4, 0.25, 0.5),
            )
            .unwrap();
            let expected = pairs(serial);
            let mut joined = Vec::new();
            for first in (0..plan.length()).step_by(37) {
                joined.extend(
                    m3c2_distances(
                        reference.inner.positions.as_flattened(),
                        compared.inner.positions.as_flattened(),
                        &plan.points(first, 37.min(plan.length() - first)).unwrap(),
                        0.4,
                        0.25,
                        0.5,
                        None,
                        None,
                        None,
                    )
                    .unwrap(),
                );
            }
            assert_eq!(
                joined.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert!(joined.iter().any(|v| v.is_nan()));
            let parallel = plan.finish(&joined).unwrap();
            let serial = compute_m3c2(
                &compared, &reference, 0.4, 0.25, 0.5, spacing, None, None, None,
            )
            .unwrap();
            assert_eq!(parallel.inner.positions, serial.inner.positions);
            assert_eq!(parallel.inner.colors, serial.inner.colors);
            assert_eq!(
                parallel.inner.attributes.len(),
                serial.inner.attributes.len()
            );
            for (a, b) in parallel
                .inner
                .attributes
                .iter()
                .zip(&serial.inner.attributes)
            {
                assert_eq!(a.name, b.name);
                // Attribute equality treats NaN as unequal; compare exported bytes.
            }
            assert_eq!(
                parallel.export("ply", None, None).unwrap(),
                serial.export("ply", None, None).unwrap()
            );
        }
    }

    #[test]
    fn resident_cloud_lane_matches_copied_inputs_and_owns_output() {
        let mut reference = cloud(0.0);
        let mut compared = cloud(0.07);
        let plan = M3c2Plan::new(&compared, 0.0);
        for bounds in [None, Some(false), Some(true)] {
            for (part, parts) in [(0, 2), (1, 2), (2, 3)] {
                let cores = plan.interleaved_points(part, parts, 7).unwrap();
                let resident = m3c2_cloud_distances(
                    &compared, &reference, &cores, 0.4, 0.25, 0.5, bounds, None, None,
                )
                .unwrap();
                let copied = m3c2_distances(
                    reference.inner.positions.as_flattened(),
                    compared.inner.positions.as_flattened(),
                    &cores,
                    0.4,
                    0.25,
                    0.5,
                    bounds,
                    None,
                    None,
                )
                .unwrap();
                assert_eq!(
                    resident.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    copied.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
        let cores = plan.points(0, plan.length()).unwrap();
        let output = m3c2_cloud_distances(
            &compared, &reference, &cores, 0.4, 0.25, 0.5, None, None, None,
        )
        .unwrap();
        let expected: Vec<_> = output.iter().map(|v| v.to_bits()).collect();
        reference.inner.positions.fill([0.0; 3]);
        compared.inner.positions.fill([0.0; 3]);
        drop(reference);
        drop(compared);
        assert_eq!(
            output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn prepared_trees_preserve_f64_pairs_exports_and_own_geometry() {
        let mut reference = cloud(0.0);
        let mut compared = cloud(0.07);
        let prepared = M3c2Prepared::new(&compared, &reference).unwrap();
        let imported = M3c2Prepared::from_snapshots(
            &prepared.reference_snapshot(),
            &prepared.compared_snapshot(),
            reference.length(),
            compared.length(),
        )
        .unwrap();
        let plan = M3c2Plan::new(&compared, 0.17);
        let cores = plan.points(0, plan.length()).unwrap();
        for bounds in [None, Some(false), Some(true)] {
            for (normal, projection, depth) in [(0.1, 0.1, 0.2), (0.4, 0.25, 0.5), (1.0, 0.5, 2.0)]
            {
                let expected = m3c2_cloud_distances(
                    &compared, &reference, &cores, normal, projection, depth, bounds, None, None,
                )
                .unwrap();
                for state in [&prepared, &imported] {
                    let actual = state
                        .compute(&cores, normal, projection, depth, bounds, None, None)
                        .unwrap();
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
        }
        let expected = compute_m3c2(
            &compared, &reference, 0.4, 0.25, 0.5, 0.17, None, None, None,
        )
        .unwrap()
        .export("ply", None, None)
        .unwrap();
        reference.inner.positions.fill([0.0; 3]);
        compared.inner.positions.fill([0.0; 3]);
        drop(reference);
        drop(compared);
        let actual = imported
            .compute(&cores, 0.4, 0.25, 0.5, None, None, None)
            .unwrap();
        assert_eq!(
            plan.finish(&actual)
                .unwrap()
                .export("ply", None, None)
                .unwrap(),
            expected
        );
    }

    #[test]
    fn cached_normals_preserve_pairs_projection_changes_and_exact_core_keys() {
        assert_eq!(NORMAL_BYTES_PER_CORE, 56);
        let reference = cloud(0.0);
        let compared = cloud(0.07);
        let mut prepared = M3c2Prepared::new(&compared, &reference).unwrap();
        let plan = M3c2Plan::new(&compared, 0.17);
        let mut cores = plan.points(0, plan.length()).unwrap();
        cores.extend([
            -0.0,
            0.0,
            0.0,
            f64::from_bits(0x7ff8_0000_0000_0123),
            0.0,
            0.0,
        ]);
        drop(reference);
        drop(compared);
        let bits = |values: Vec<f64>| values.into_iter().map(f64::to_bits).collect::<Vec<_>>();
        for bounds in [false, true] {
            for normal in [0.1, 0.4, 1.0] {
                let first = prepared
                    .compute_cached(
                        &cores,
                        normal,
                        0.25,
                        0.5,
                        Some(bounds),
                        cores.len(),
                        None,
                        None,
                    )
                    .unwrap();
                assert!(!prepared.normal_cache_hit());
                assert_eq!(
                    bits(first),
                    bits(
                        prepared
                            .compute(&cores, normal, 0.25, 0.5, Some(bounds), None, None)
                            .unwrap()
                    )
                );
                for (projection, depth) in [(0.1, 0.2), (0.35, 0.7), (0.25, 0.5)] {
                    let actual = prepared
                        .compute_cached(
                            &cores,
                            normal,
                            projection,
                            depth,
                            Some(bounds),
                            cores.len(),
                            None,
                            None,
                        )
                        .unwrap();
                    assert!(prepared.normal_cache_hit());
                    assert_eq!(
                        bits(actual),
                        bits(
                            prepared
                                .compute(
                                    &cores,
                                    normal,
                                    projection,
                                    depth,
                                    Some(bounds),
                                    None,
                                    None
                                )
                                .unwrap()
                        )
                    );
                }
            }
        }
        assert_eq!(
            prepared.normal_cache_bytes(),
            cores.len() / 3 * NORMAL_BYTES_PER_CORE
        );
        let end = cores.len() - 6;
        cores[end] = 0.0; // Same numerical coordinate, different bit key.
        prepared
            .compute_cached(&cores, 1.0, 0.25, 0.5, Some(true), cores.len(), None, None)
            .unwrap();
        assert!(!prepared.normal_cache_hit());
        let last = cores.len() - 3;
        cores[last] = f64::from_bits(0x7ff8_0000_0000_0456);
        prepared
            .compute_cached(&cores, 1.0, 0.25, 0.5, Some(true), cores.len(), None, None)
            .unwrap();
        assert!(!prepared.normal_cache_hit());
        cores[..6].rotate_left(3); // Count alone must not identify an ordered core set.
        let actual = prepared
            .compute_cached(&cores, 1.0, 0.25, 0.5, Some(true), cores.len(), None, None)
            .unwrap();
        assert!(!prepared.normal_cache_hit());
        assert_eq!(
            bits(actual),
            bits(
                prepared
                    .compute(&cores, 1.0, 0.25, 0.5, Some(true), None, None)
                    .unwrap()
            )
        );
        let actual = prepared
            .compute_cached(&cores, 1.0, 0.25, 0.5, Some(true), 0, None, None)
            .unwrap();
        assert!(!prepared.normal_cache_hit());
        assert_eq!(prepared.normal_cache_bytes(), 0);
        assert_eq!(
            bits(actual),
            bits(
                prepared
                    .compute(&cores, 1.0, 0.25, 0.5, Some(true), None, None)
                    .unwrap()
            )
        );
        prepared
            .compute_cached(&[], 1.0, 0.25, 0.5, Some(true), usize::MAX, None, None)
            .unwrap();
        assert_eq!(prepared.normal_cache_bytes(), 0);
    }

    #[test]
    fn cyclic_core_blocks_restore_exact_results_and_attributes() {
        let reference = cloud(0.0);
        let compared = cloud(0.07);
        for spacing in [0.0, 0.17] {
            let expected = compute_m3c2(
                &compared, &reference, 0.4, 0.25, 0.5, spacing, None, None, None,
            )
            .unwrap()
            .export("ply", None, None)
            .unwrap();
            for parts in [2, 3, 4] {
                for block in [1, 7, 64, 1024] {
                    let plan = M3c2Plan::new(&compared, spacing);
                    let serial = pairs(
                        m3c2(
                            &plan.core.positions,
                            &reference.inner.positions,
                            &compared.inner.positions,
                            params(0.4, 0.25, 0.5),
                        )
                        .unwrap(),
                    );
                    let mut joined = vec![f64::NAN; serial.len()];
                    let mut seen = vec![false; plan.length()];
                    for part in 0..parts {
                        let points = plan.interleaved_points(part, parts, block).unwrap();
                        assert_eq!(
                            points.len() / 3,
                            plan.length() / parts + usize::from(part < plan.length() % parts)
                        );
                        let values = m3c2_distances(
                            reference.inner.positions.as_flattened(),
                            compared.inner.positions.as_flattened(),
                            &points,
                            0.4,
                            0.25,
                            0.5,
                            None,
                            None,
                            None,
                        )
                        .unwrap();
                        let mut offset = 0;
                        let full = plan.length() / (parts * block) * (parts * block);
                        let tail = plan.length() - full;
                        let tail_first = full + part * (tail / parts) + part.min(tail % parts);
                        let tail_count = tail / parts + usize::from(part < tail % parts);
                        let ranges = (part * block..full)
                            .step_by(parts * block)
                            .map(|first| (first, block))
                            .chain(std::iter::once((tail_first, tail_count)));
                        for (first, count) in ranges {
                            assert!(seen[first..first + count].iter().all(|&v| !v));
                            seen[first..first + count].fill(true);
                            joined[first * 2..(first + count) * 2]
                                .copy_from_slice(&values[offset * 2..(offset + count) * 2]);
                            offset += count;
                        }
                        assert_eq!(offset * 2, values.len());
                    }
                    assert!(seen.into_iter().all(|v| v));
                    assert_eq!(
                        joined.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        serial.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    assert_eq!(
                        plan.finish(&joined)
                            .unwrap()
                            .export("ply", None, None)
                            .unwrap(),
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn plan_owns_core_positions_colors_and_attributes_after_source_edits() {
        let mut compared = cloud(0.1);
        let plan = M3c2Plan::new(&compared, 0.0);
        let positions = compared.inner.positions.clone();
        let colors = compared.inner.colors.clone();
        compared.inner.positions.fill([0.0; 3]);
        compared.inner.colors = None;
        compared.inner.attributes.clear();
        assert_eq!(
            plan.points(0, plan.length()).unwrap(),
            positions.as_flattened()
        );
        let n = plan.length();
        let completed = plan.finish(&vec![0.1; n * 2]).unwrap();
        assert_eq!(completed.inner.len(), n);
        assert!(completed.inner.colors.is_some());
        assert_eq!(
            completed.inner.colors.as_ref().unwrap().len(),
            colors.unwrap().len()
        );
        assert!(completed.inner.attribute("intensity").is_some());
    }
}
