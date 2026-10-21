//! GPU candidates with authoritative f64 nearest-neighbour verification.

use std::rc::Rc;

use ca_core::kdtree::CandidateKdTree;
use wasm_bindgen::prelude::*;

use crate::Cloud;

const MAX_POINTS: usize = 2_000_000;

/// Immutable reference snapshot, shared by jobs without rebuilding the tree.
#[wasm_bindgen]
pub struct GpuC2cReference {
    tree: Rc<CandidateKdTree>,
    origin: [f64; 3],
}

#[wasm_bindgen]
impl GpuC2cReference {
    #[wasm_bindgen(constructor)]
    pub fn new(reference: &Cloud) -> Result<GpuC2cReference, JsError> {
        check_count(reference)?;
        Ok(Self {
            tree: Rc::new(CandidateKdTree::new(&reference.inner.positions).unwrap()),
            origin: reference.inner.positions[0],
        })
    }

    /// Packed reference buffers are produced only on the first GPU upload.
    #[wasm_bindgen(js_name = candidateBuffers)]
    pub fn candidate_buffers(&self) -> GpuC2cBuffers {
        let (points, nodes) = self.tree.candidate_buffers(self.origin);
        GpuC2cBuffers { points, nodes }
    }

    /// Immutable original f64 coordinates for independent verification workers.
    pub fn xyz(&self) -> Vec<f64> {
        self.tree.original_points().into_iter().flatten().collect()
    }

    /// Transfer the already built tree without repeating median partitioning.
    pub fn snapshot(&self) -> Vec<u8> {
        self.tree.snapshot()
    }

    #[wasm_bindgen(js_name = fromSnapshot)]
    pub fn from_snapshot(
        snapshot: &[u8],
        reuse_bounds: Option<bool>,
    ) -> Result<GpuC2cReference, JsError> {
        let tree = CandidateKdTree::from_snapshot_with_reused_bounds(
            snapshot,
            MAX_POINTS,
            reuse_bounds.unwrap_or(true),
        )
        .map_err(JsError::new)?;
        // The origin only affects candidate packing, not f64 verification.
        let origin = tree.original_point(0).unwrap();
        Ok(Self {
            tree: Rc::new(tree),
            origin,
        })
    }

    #[wasm_bindgen(js_name = fromXyz)]
    pub fn from_xyz(xyz: &[f64]) -> Result<GpuC2cReference, JsError> {
        if xyz.is_empty() || !xyz.len().is_multiple_of(3) || xyz.len() / 3 > MAX_POINTS {
            return Err(JsError::new("invalid GPU verification reference"));
        }
        let (points, _) = xyz.as_chunks::<3>();
        Ok(Self {
            tree: Rc::new(CandidateKdTree::new(points).unwrap()),
            origin: points[0],
        })
    }

    /// Verify a worker-owned batch using the same tree-order hints and f64
    /// arithmetic as the main job. Invalid hints still trigger a full search.
    pub fn verify(
        &self,
        xyz: &[f64],
        candidates: &[u32],
        bounds: bool,
        simd: bool,
    ) -> Result<GpuC2cVerification, JsError> {
        if xyz.len() / 3 != candidates.len() || !xyz.len().is_multiple_of(3) {
            return Err(JsError::new("invalid GPU verification batch"));
        }
        let (queries, _) = xyz.as_chunks::<3>();
        Ok(verify_queries(
            &self.tree, queries, candidates, bounds, simd,
        ))
    }
}

fn verify_queries(
    tree: &CandidateKdTree,
    queries: &[[f64; 3]],
    candidates: &[u32],
    bounds: bool,
    simd: bool,
) -> GpuC2cVerification {
    let mut certified = 0;
    let distances = queries
        .iter()
        .zip(candidates)
        .map(|(q, &slot)| {
            let (hit, certificate) = tree.nearest_with_options(q, slot, bounds, simd);
            certified += usize::from(certificate);
            hit.distance_sq.sqrt()
        })
        .collect();
    GpuC2cVerification {
        distances,
        certified,
    }
}

/// A worker's immutable exact queries, kept in WASM for one asynchronous job.
#[wasm_bindgen]
pub struct GpuC2cVerifier {
    tree: Rc<CandidateKdTree>,
    xyz: Vec<f64>,
}

#[wasm_bindgen]
impl GpuC2cVerifier {
    #[wasm_bindgen(constructor)]
    pub fn new(reference: &GpuC2cReference, xyz: Vec<f64>) -> Result<Self, JsError> {
        if xyz.is_empty() || !xyz.len().is_multiple_of(3) || xyz.len() / 3 > MAX_POINTS {
            return Err(JsError::new("invalid GPU verification queries"));
        }
        Ok(Self {
            tree: reference.tree.clone(),
            xyz,
        })
    }

    /// Only hints cross the WASM boundary after preparation; coordinates stay exact.
    pub fn verify(
        &self,
        first: usize,
        candidates: &[u32],
        bounds: bool,
        simd: bool,
    ) -> Result<GpuC2cVerification, JsError> {
        let end = first
            .checked_add(candidates.len())
            .ok_or_else(|| JsError::new("invalid staged query range"))?;
        let (queries, _) = self.xyz.as_chunks::<3>();
        let queries = queries
            .get(first..end)
            .ok_or_else(|| JsError::new("invalid staged query range"))?;
        Ok(verify_queries(
            &self.tree, queries, candidates, bounds, simd,
        ))
    }
}

#[wasm_bindgen]
pub struct GpuC2cVerification {
    distances: Vec<f64>,
    certified: usize,
}

#[wasm_bindgen]
impl GpuC2cVerification {
    #[wasm_bindgen(js_name = takeDistances)]
    pub fn take_distances(&mut self) -> Vec<f64> {
        std::mem::take(&mut self.distances)
    }

    #[wasm_bindgen(getter)]
    pub fn certified(&self) -> usize {
        self.certified
    }
}

#[wasm_bindgen]
pub struct GpuC2cBuffers {
    points: Vec<f32>,
    nodes: Vec<u32>,
}

#[wasm_bindgen]
impl GpuC2cBuffers {
    #[wasm_bindgen(js_name = takePoints)]
    pub fn take_points(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.points)
    }

    #[wasm_bindgen(js_name = takeNodes)]
    pub fn take_nodes(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.nodes)
    }
}

/// Owns a query snapshot and a shared reference so async work never borrows
/// mutable clouds or a cache entry that can be invalidated while it runs.
#[wasm_bindgen]
pub struct GpuC2cJob {
    tree: Rc<CandidateKdTree>,
    queries: Vec<[f64; 3]>,
    origin: [f64; 3],
    certified: usize,
    verify_bounds: bool,
    verify_simd: bool,
}

fn check_count(cloud: &Cloud) -> Result<(), JsError> {
    if cloud.inner.len() > MAX_POINTS {
        return Err(JsError::new(
            "WebGPU C2C currently supports up to 2,000,000 points per cloud",
        ));
    }
    if cloud.inner.is_empty() {
        return Err(JsError::new("WebGPU C2C requires nonempty clouds"));
    }
    Ok(())
}

#[wasm_bindgen]
impl GpuC2cJob {
    #[wasm_bindgen(constructor)]
    pub fn new(compared: &Cloud, reference: &GpuC2cReference) -> Result<GpuC2cJob, JsError> {
        check_count(compared)?;
        Ok(Self {
            tree: reference.tree.clone(),
            queries: compared.inner.positions.clone(),
            origin: reference.origin,
            certified: 0,
            verify_bounds: true,
            verify_simd: true,
        })
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.queries.len()
    }

    #[wasm_bindgen(getter)]
    pub fn certified(&self) -> usize {
        self.certified
    }

    /// Retain the plane-only path for controlled within-worker benchmarks.
    #[wasm_bindgen(js_name = useBounds)]
    pub fn use_bounds(&mut self, enabled: bool) {
        self.verify_bounds = enabled;
    }

    /// Retain scalar leaf arithmetic for controlled WASM parity/benchmarks.
    #[wasm_bindgen(js_name = useSimd)]
    pub fn use_simd(&mut self, enabled: bool) {
        self.verify_simd = enabled;
    }

    #[wasm_bindgen(getter, js_name = simdEnabled)]
    pub fn simd_enabled(&self) -> bool {
        self.verify_simd && cfg!(all(target_arch = "wasm32", target_feature = "simd128"))
    }

    /// A bounded batch in original query order, shifted and padded to vec4.
    pub fn queries(&self, first: usize, count: usize) -> Result<Vec<f32>, JsError> {
        let end = first
            .checked_add(count)
            .ok_or_else(|| JsError::new("invalid query range"))?;
        let queries = self
            .queries
            .get(first..end)
            .ok_or_else(|| JsError::new("invalid query range"))?;
        Ok(queries
            .iter()
            .flat_map(|q| {
                [
                    (q[0] - self.origin[0]) as f32,
                    (q[1] - self.origin[1]) as f32,
                    (q[2] - self.origin[2]) as f32,
                    0.0,
                ]
            })
            .collect())
    }

    /// Transfer exact coordinates to pool workers from this job's snapshot.
    #[wasm_bindgen(js_name = queriesF64)]
    pub fn queries_f64(&self, first: usize, count: usize) -> Result<Vec<f64>, JsError> {
        let end = first
            .checked_add(count)
            .ok_or_else(|| JsError::new("invalid query range"))?;
        let queries = self
            .queries
            .get(first..end)
            .ok_or_else(|| JsError::new("invalid query range"))?;
        Ok(queries.iter().flatten().copied().collect())
    }

    /// Experimental contiguous export; retain the ordinary iterator API unchanged.
    #[wasm_bindgen(js_name = queriesF64WithCopy)]
    pub fn queries_f64_with_copy(
        &self,
        first: usize,
        count: usize,
        slice_copy: bool,
    ) -> Result<Vec<f64>, JsError> {
        if !slice_copy {
            return self.queries_f64(first, count);
        }
        let end = first
            .checked_add(count)
            .ok_or_else(|| JsError::new("invalid query range"))?;
        let queries = self
            .queries
            .get(first..end)
            .ok_or_else(|| JsError::new("invalid query range"))?;
        Ok(queries.as_flattened().to_vec())
    }

    /// Concatenate one lane's ranges from all batches, preserving original f64 bits.
    #[wasm_bindgen(js_name = workerQueries)]
    pub fn worker_queries(
        &self,
        lane: usize,
        workers: usize,
        batch_size: usize,
        main_worker: Option<bool>,
    ) -> Result<Vec<f64>, JsError> {
        if workers == 0 || workers > 8 || lane >= workers || batch_size == 0 || batch_size > 262_144
        {
            return Err(JsError::new("invalid staged query partition"));
        }
        let lane_count = |count: usize| {
            let lanes = workers.min(count);
            if lane < lanes.saturating_sub(usize::from(main_worker.unwrap_or(false))) {
                (lane + 1) * count / lanes - lane * count / lanes
            } else {
                0
            }
        };
        let full = self.queries.len() / batch_size;
        let last = self.queries.len() % batch_size;
        let mut xyz = Vec::with_capacity(3 * (full * lane_count(batch_size) + lane_count(last)));
        for batch in self.queries.chunks(batch_size) {
            let lanes = workers.min(batch.len());
            if lane < lanes - usize::from(main_worker.unwrap_or(false)) {
                let first = lane * batch.len() / lanes;
                let end = (lane + 1) * batch.len() / lanes;
                xyz.extend(batch[first..end].iter().flatten().copied());
            }
        }
        Ok(xyz)
    }

    /// Exact leaf scan plus a certificate, then bounded sibling searches or
    /// the ordinary complete f64 search if the certificate did not suffice.
    /// Verify from the resident snapshot without changing the serial certificate
    /// counter. A parallel batch can discard this result if another lane fails.
    pub fn verify(&self, first: usize, candidates: &[u32]) -> Result<GpuC2cVerification, JsError> {
        let end = first
            .checked_add(candidates.len())
            .ok_or_else(|| JsError::new("invalid candidate range"))?;
        let queries = self
            .queries
            .get(first..end)
            .ok_or_else(|| JsError::new("invalid candidate range"))?;
        Ok(verify_queries(
            &self.tree,
            queries,
            candidates,
            self.verify_bounds,
            self.verify_simd,
        ))
    }

    pub fn complete(&mut self, first: usize, candidates: &[u32]) -> Result<Vec<f64>, JsError> {
        let verified = self.verify(first, candidates)?;
        self.certified += verified.certified;
        Ok(verified.distances)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ca_core::PointCloud;

    #[test]
    fn resident_verification_is_transactional_and_staged_queries_exclude_its_lane() {
        let reference = query_test_cloud(
            (0..73)
                .map(|i| [i as f64 * 0.13, (i % 7) as f64, 0.0])
                .collect(),
        );
        let index = GpuC2cReference::new(&reference).unwrap();
        for n in [1, 2, 7, 8, 9, 31, 65, 257] {
            let cloud = query_test_cloud(
                (0..n)
                    .map(|i| [i as f64 * 0.07, (i % 7) as f64, 0.01])
                    .collect(),
            );
            for batch_size in [1, 3, 7, 64] {
                for workers in [2, 3, 4, 8] {
                    let mut job = GpuC2cJob::new(&cloud, &index).unwrap();
                    let staged: Vec<_> = (0..workers)
                        .map(|lane| {
                            job.worker_queries(lane, workers, batch_size, Some(true))
                                .unwrap()
                        })
                        .collect();
                    let mut offsets = vec![0; workers];
                    let mut resident_count = 0;
                    for first in (0..n).step_by(batch_size) {
                        let count = batch_size.min(n - first);
                        let lanes = workers.min(count);
                        for lane in 0..lanes {
                            let start = lane * count / lanes;
                            let end = (lane + 1) * count / lanes;
                            let xyz = job.queries_f64(first + start, end - start).unwrap();
                            let hints = vec![u32::MAX; end - start];
                            for bounds in [false, true] {
                                for simd in [false, true] {
                                    job.use_bounds(bounds);
                                    job.use_simd(simd);
                                    let resident = job.verify(first + start, &hints).unwrap();
                                    let copied = index.verify(&xyz, &hints, bounds, simd).unwrap();
                                    assert_eq!(resident.certified, copied.certified);
                                    assert!(
                                        resident
                                            .distances
                                            .iter()
                                            .zip(&copied.distances)
                                            .all(|(a, b)| a.to_bits() == b.to_bits())
                                    );
                                    assert_eq!(job.certified(), 0);
                                }
                            }
                            if lane == lanes - 1 {
                                resident_count += end - start;
                            } else {
                                let offset = offsets[lane];
                                assert!(
                                    staged[lane][offset..offset + xyz.len()]
                                        .iter()
                                        .zip(&xyz)
                                        .all(|(a, b)| a.to_bits() == b.to_bits())
                                );
                                offsets[lane] += xyz.len();
                            }
                        }
                    }
                    assert_eq!(
                        staged.iter().map(Vec::len).sum::<usize>() / 3 + resident_count,
                        n
                    );
                    assert!(
                        staged
                            .iter()
                            .zip(offsets)
                            .all(|(xyz, used)| xyz.len() == used)
                    );
                    let candidates = vec![0; n];
                    let verified = job.verify(0, &candidates).unwrap();
                    let expected = verified.certified;
                    let distances = verified.distances;
                    assert_eq!(job.certified(), 0);
                    assert!(
                        job.complete(0, &candidates)
                            .unwrap()
                            .iter()
                            .zip(&distances)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                    assert_eq!(job.certified(), expected);
                    drop(job);
                    assert_eq!(distances.len(), n);
                }
            }
        }
    }

    fn query_test_cloud(positions: Vec<[f64; 3]>) -> Cloud {
        Cloud::unindexed(PointCloud {
            positions,
            colors: None,
            attributes: vec![],
        })
    }

    #[test]
    fn staged_lane_queries_preserve_bits_partial_batches_and_exact_results() {
        let reference = query_test_cloud(
            (0..73)
                .map(|i| {
                    [
                        368_000.0 + i as f64 * 0.13,
                        3_955_000.0 + (i % 7) as f64,
                        i as f64 * 0.01,
                    ]
                })
                .collect(),
        );
        let index = GpuC2cReference::new(&reference).unwrap();
        for n in [1, 2, 7, 8, 9, 16, 17, 31, 65, 131, 257] {
            let points: Vec<_> = (0..n)
                .map(|i| {
                    if i == 0 {
                        [-0.0, f64::from_bits(1), 0.0]
                    } else {
                        [
                            368_000.0 + i as f64 * 0.07,
                            3_955_000.0 + (i % 7) as f64,
                            0.015,
                        ]
                    }
                })
                .collect();
            let cloud = query_test_cloud(points);
            for batch_size in [1, 3, 7, 17, 64, 131_072] {
                for workers in [2, 3, 4, 8] {
                    let job = GpuC2cJob::new(&cloud, &index).unwrap();
                    let lanes = workers.min(batch_size).min(n);
                    let staged: Vec<_> = (0..lanes)
                        .map(|lane| {
                            let xyz = job.worker_queries(lane, workers, batch_size, None).unwrap();
                            GpuC2cVerifier::new(&index, xyz).unwrap()
                        })
                        .collect();
                    let mut offsets = vec![0; lanes];
                    for first in (0..n).step_by(batch_size) {
                        let count = batch_size.min(n - first);
                        let batch_lanes = workers.min(count);
                        for (lane, offset) in offsets.iter_mut().enumerate().take(batch_lanes) {
                            let start = lane * count / batch_lanes;
                            let end = (lane + 1) * count / batch_lanes;
                            let expected = job.queries_f64(first + start, end - start).unwrap();
                            let actual =
                                &staged[lane].xyz[*offset * 3..(*offset + end - start) * 3];
                            assert!(
                                actual
                                    .iter()
                                    .zip(&expected)
                                    .all(|(a, b)| a.to_bits() == b.to_bits())
                            );
                            let candidates: Vec<_> = (first + start..first + end)
                                .map(|i| {
                                    if i % 3 == 0 {
                                        u32::MAX
                                    } else {
                                        (i % 73) as u32
                                    }
                                })
                                .collect();
                            for bounds in [false, true] {
                                for simd in [false, true] {
                                    let streamed =
                                        index.verify(&expected, &candidates, bounds, simd).unwrap();
                                    let held = staged[lane]
                                        .verify(*offset, &candidates, bounds, simd)
                                        .unwrap();
                                    assert_eq!(streamed.certified, held.certified);
                                    assert!(
                                        streamed
                                            .distances
                                            .iter()
                                            .zip(&held.distances)
                                            .all(|(a, b)| a.to_bits() == b.to_bits())
                                    );
                                }
                            }
                            *offset += end - start;
                        }
                    }
                    assert_eq!(offsets.iter().sum::<usize>(), n);
                    for (lane, count) in offsets.into_iter().enumerate() {
                        assert_eq!(staged[lane].xyz.len(), count * 3);
                    }
                }
            }
        }
    }

    #[test]
    fn staged_verifiers_own_queries_and_reference_after_source_release() {
        let reference = query_test_cloud(vec![[0.0; 3], [1.0, 2.0, 3.0]]);
        let index = GpuC2cReference::new(&reference).unwrap();
        let xyz = vec![0.1, 0.2, 0.3, 0.9, 2.0, 3.0];
        let candidates = [u32::MAX, 1];
        let expected = index.verify(&xyz, &candidates, true, true).unwrap();
        let verifier = GpuC2cVerifier::new(&index, xyz).unwrap();
        drop(index);
        drop(reference);
        let actual = verifier.verify(0, &candidates, true, true).unwrap();
        assert_eq!(actual.certified, expected.certified);
        assert!(
            actual
                .distances
                .iter()
                .zip(expected.distances)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }

    #[test]
    fn independent_verifiers_preserve_slots_snapshots_and_distance_bits() {
        let points: Vec<_> = (0..513)
            .map(|i| {
                [
                    6_378_137.0 + (i % 37) as f64 * 0.000_001,
                    (i % 19) as f64 * 0.1,
                    (i % 7) as f64,
                ]
            })
            .collect();
        let reference = Cloud::unindexed(PointCloud {
            positions: points.clone(),
            colors: None,
            attributes: vec![],
        });
        let index = GpuC2cReference::new(&reference).unwrap();
        let mut snapshot = index.xyz();
        assert_eq!(
            snapshot,
            points.iter().flatten().copied().collect::<Vec<_>>()
        );
        let copy = GpuC2cReference::from_xyz(&snapshot).unwrap();
        let imported = GpuC2cReference::from_snapshot(&index.snapshot(), None).unwrap();
        snapshot.fill(0.0);
        assert_eq!(
            index.candidate_buffers().take_points(),
            copy.candidate_buffers().take_points()
        );
        assert_eq!(
            index.candidate_buffers().take_nodes(),
            copy.candidate_buffers().take_nodes()
        );
        let compared = Cloud::unindexed(PointCloud {
            positions: points
                .iter()
                .map(|p| [p[0] + 0.000_003, p[1] + 0.02, p[2] + 0.1])
                .collect(),
            colors: None,
            attributes: vec![],
        });
        let expected = ca_core::cloud_to_cloud(&compared.inner, &reference.inner).unwrap();
        for bounds in [false, true] {
            for simd in [false, true] {
                let mut job = GpuC2cJob::new(&compared, &index).unwrap();
                job.use_bounds(bounds);
                job.use_simd(simd);
                for first in (0..job.length()).step_by(31) {
                    let count = 31.min(job.length() - first);
                    let hints: Vec<_> = (first..first + count)
                        .map(|i| {
                            if i % 3 == 0 {
                                u32::MAX
                            } else {
                                (i * 11 % 513) as u32
                            }
                        })
                        .collect();
                    let main = job.complete(first, &hints).unwrap();
                    for worker in [&copy, &imported] {
                        let mut verified = worker
                            .verify(
                                &job.queries_f64(first, count).unwrap(),
                                &hints,
                                bounds,
                                simd,
                            )
                            .unwrap();
                        let distances = verified.take_distances();
                        for ((a, b), c) in main
                            .iter()
                            .zip(&distances)
                            .zip(&expected[first..first + count])
                        {
                            assert_eq!(a.to_bits(), b.to_bits());
                            assert_eq!(a.to_bits(), c.to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn exact_snapshot_import_keeps_original_origin_and_candidate_buffers() {
        let reference = Cloud::unindexed(PointCloud {
            positions: (0..257)
                .rev()
                .map(|i| [368_000.0 + i as f64 * 0.001, -0.0, 4.0])
                .collect(),
            colors: None,
            attributes: vec![],
        });
        let index = GpuC2cReference::new(&reference).unwrap();
        let mut bytes = index.snapshot();
        let restored = GpuC2cReference::from_snapshot(&bytes, None).unwrap();
        bytes.fill(0);
        assert_eq!(
            index.origin.map(f64::to_bits),
            restored.origin.map(f64::to_bits)
        );
        assert_eq!(index.snapshot(), restored.snapshot());
        assert_eq!(
            index.candidate_buffers().take_nodes(),
            restored.candidate_buffers().take_nodes()
        );
        assert_eq!(
            index.candidate_buffers().take_points(),
            restored.candidate_buffers().take_points()
        );
    }

    #[test]
    fn rounded_and_invalid_seeds_match_cpu_with_owned_snapshots() {
        let mut reference = Cloud::unindexed(PointCloud {
            positions: vec![[6_378_137.0, 0.0, 0.0], [6_378_137.000_01, 0.0, 0.0]],
            colors: None,
            attributes: vec![],
        });
        let mut compared = Cloud::unindexed(PointCloud {
            positions: vec![[6_378_137.000_009, 0.0, 0.0], [6_378_137.000_003, 0.0, 0.0]],
            colors: None,
            attributes: vec![],
        });
        let expected = ca_core::cloud_to_cloud(&compared.inner, &reference.inner).unwrap();
        let index = GpuC2cReference::new(&reference).unwrap();
        let mut job = GpuC2cJob::new(&compared, &index).unwrap();
        let mut second_job = GpuC2cJob::new(&compared, &index).unwrap();
        assert!(Rc::ptr_eq(&job.tree, &second_job.tree));
        let mut buffers = index.candidate_buffers();
        assert_eq!(buffers.take_points().len(), 8);
        assert_eq!(buffers.take_nodes().len(), 5);
        drop(index);
        reference.inner.positions.clear();
        compared.inner.positions.clear();
        assert_eq!(job.complete(0, &[0, u32::MAX]).unwrap(), expected);
        assert_eq!(job.certified(), 1);
        assert_eq!(second_job.complete(0, &[1, 1]).unwrap(), expected);
        assert_eq!(second_job.certified(), 2);
        assert_eq!(job.complete(1, &[1]).unwrap(), expected[1..]);
        assert_eq!(job.queries(0, 2).unwrap().len(), 8);
    }

    #[test]
    fn verifier_selection_is_per_job_and_preserves_exact_results() {
        let reference = Cloud::unindexed(PointCloud {
            positions: (0..32).map(|i| [i as f64, (i % 2) as f64, 0.0]).collect(),
            colors: None,
            attributes: vec![],
        });
        let compared = Cloud::unindexed(PointCloud {
            positions: vec![[15.5, 10.0, 0.0]],
            colors: None,
            attributes: vec![],
        });
        let expected = ca_core::cloud_to_cloud(&compared.inner, &reference.inner).unwrap();
        let index = GpuC2cReference::new(&reference).unwrap();
        let mut bounded = GpuC2cJob::new(&compared, &index).unwrap();
        let mut planes = GpuC2cJob::new(&compared, &index).unwrap();
        let mut scalar = GpuC2cJob::new(&compared, &index).unwrap();
        planes.use_bounds(false);
        scalar.use_simd(false);
        assert_eq!(bounded.complete(0, &[0]).unwrap(), expected);
        assert_eq!(planes.complete(0, &[0]).unwrap(), expected);
        assert_eq!(scalar.complete(0, &[0]).unwrap(), expected);
        assert_eq!(bounded.certified(), 1);
        assert_eq!(planes.certified(), 0);
        assert_eq!(scalar.certified(), bounded.certified());
    }

    #[test]
    fn query_copy_paths_preserve_ieee_bits_ranges_and_snapshot_ownership() {
        let reference = Cloud::unindexed(PointCloud {
            positions: vec![[0.0; 3]],
            colors: None,
            attributes: vec![],
        });
        let original = vec![
            [
                -0.0,
                f64::from_bits(1),
                f64::from_bits(0x7ff8_1234_5678_9abc),
            ],
            [f64::MAX, f64::NEG_INFINITY, f64::INFINITY],
            [6_378_137.000_009, -f64::from_bits(1), 0.0],
        ];
        let mut compared = Cloud::unindexed(PointCloud {
            positions: original.clone(),
            colors: None,
            attributes: vec![],
        });
        let index = GpuC2cReference::new(&reference).unwrap();
        let job = GpuC2cJob::new(&compared, &index).unwrap();
        compared.inner.positions.clear();
        drop(index);
        for (first, count) in [(0, 0), (0, 3), (1, 1), (2, 1), (3, 0)] {
            let expected: Vec<_> = original[first..first + count]
                .as_flattened()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            for control in [false, true] {
                let mut exported = job.queries_f64_with_copy(first, count, control).unwrap();
                assert_eq!(
                    exported.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected
                );
                exported.fill(42.0);
                assert_eq!(
                    job.queries_f64_with_copy(first, count, control)
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                );
            }
        }
    }
}
