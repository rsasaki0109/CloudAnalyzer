//! Shared exact verification of untrusted WebGPU C2C candidates.

use ca_core::kdtree::CandidateKdTree;
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "threads", target_arch = "wasm32"))]
pub use wasm_bindgen_rayon::init_thread_pool;

const MAX_POINTS: usize = 2_000_000;

/// One validated tree and one owned query job for every Rayon lane.
#[wasm_bindgen]
pub struct SharedGpuC2c {
    tree: CandidateKdTree,
    queries: Vec<f64>,
}

impl SharedGpuC2c {
    fn validate_queries(xyz: &[f64]) -> Result<(), &'static str> {
        if xyz.is_empty() || !xyz.len().is_multiple_of(3) || xyz.len() / 3 > MAX_POINTS {
            return Err("expected 1..2000000 query XYZ triples");
        }
        if !xyz.iter().all(|v| v.is_finite()) {
            return Err("query coordinates must be finite");
        }
        Ok(())
    }

    fn prepare(snapshot: &[u8], xyz: &[f64]) -> Result<Self, &'static str> {
        Self::validate_queries(xyz)?;
        let queries = xyz.to_vec();
        let tree = CandidateKdTree::from_snapshot(snapshot, MAX_POINTS)?;
        Ok(Self { tree, queries })
    }

    fn replace_queries(&mut self, xyz: Vec<f64>) -> Result<(), &'static str> {
        // Own the transferred WASM allocation, with no second coordinate copy.
        // Validation failure leaves the current job intact.
        Self::validate_queries(&xyz)?;
        self.queries = xyz;
        Ok(())
    }

    fn calculate(
        &self,
        first: usize,
        candidates: &[u32],
        bounds: bool,
        simd: bool,
        parallel: bool,
    ) -> Result<SharedVerification, &'static str> {
        let end = first
            .checked_add(candidates.len())
            .ok_or("invalid staged query range")?;
        let queries = self
            .queries
            .as_chunks::<3>()
            .0
            .get(first..end)
            .ok_or("invalid staged query range")?;
        let mut distances = vec![0.0; candidates.len()];
        let verify = |((query, &slot), distance): ((&[f64; 3], &u32), &mut f64)| {
            let (hit, certified) = self.tree.nearest_with_options(query, slot, bounds, simd);
            *distance = hit.distance_sq.sqrt();
            usize::from(certified)
        };
        let certified = if parallel {
            #[cfg(feature = "threads")]
            {
                use rayon::prelude::*;
                queries
                    .par_iter()
                    .zip(candidates.par_iter())
                    .zip(distances.par_iter_mut())
                    .map(verify)
                    .sum()
            }
            #[cfg(not(feature = "threads"))]
            return Err("shared verification requires the threads build");
        } else {
            queries
                .iter()
                .zip(candidates)
                .zip(&mut distances)
                .map(verify)
                .sum()
        };
        Ok(SharedVerification {
            distances,
            certified,
        })
    }
}

#[wasm_bindgen]
impl SharedGpuC2c {
    #[wasm_bindgen(constructor)]
    pub fn new(snapshot: &[u8], queries: &[f64]) -> Result<Self, JsError> {
        Self::prepare(snapshot, queries).map_err(JsError::new)
    }

    /// Start a fresh query job without importing the reference tree again.
    #[wasm_bindgen(js_name = setQueries)]
    pub fn set_queries(&mut self, queries: Vec<f64>) -> Result<(), JsError> {
        self.replace_queries(queries).map_err(JsError::new)
    }

    #[wasm_bindgen(js_name = releaseQueries)]
    pub fn release_queries(&mut self) {
        self.queries = Vec::new();
    }

    /// Invalid hints use the core's complete f64 search, just as in production.
    pub fn verify(
        &self,
        first: usize,
        candidates: &[u32],
        bounds: bool,
        simd: bool,
        parallel: bool,
    ) -> Result<SharedVerification, JsError> {
        self.calculate(first, candidates, bounds, simd, parallel)
            .map_err(JsError::new)
    }
}

#[wasm_bindgen]
pub struct SharedVerification {
    distances: Vec<f64>,
    certified: usize,
}

#[wasm_bindgen]
impl SharedVerification {
    #[wasm_bindgen(js_name = takeDistances)]
    pub fn take_distances(&mut self) -> Vec<f64> {
        std::mem::take(&mut self.distances)
    }

    #[wasm_bindgen(getter)]
    pub fn certified(&self) -> usize {
        self.certified
    }
}

#[wasm_bindgen(js_name = threadIndices)]
pub fn thread_indices() -> Vec<u32> {
    #[cfg(feature = "threads")]
    return rayon::broadcast(|context| context.index() as u32);
    #[cfg(not(feature = "threads"))]
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_snapshots_queries_and_ranges() {
        let tree = CandidateKdTree::new(&[[1.0, 2.0, 3.0]]).unwrap();
        let bytes = tree.snapshot();
        for xyz in [
            &[][..],
            &[0.0; 2],
            &[f64::NAN, 0.0, 0.0],
            &[f64::INFINITY; 3],
        ] {
            assert!(SharedGpuC2c::prepare(&bytes, xyz).is_err());
        }
        assert!(SharedGpuC2c::prepare(&bytes, &vec![0.0; (MAX_POINTS + 1) * 3]).is_err());
        assert!(SharedGpuC2c::prepare(&[], &[0.0; 3]).is_err());
        let mut corrupt = bytes.clone();
        corrupt[0] ^= 1;
        assert!(SharedGpuC2c::prepare(&corrupt, &[0.0; 3]).is_err());
        let job = SharedGpuC2c::prepare(&bytes, &[0.0; 3]).unwrap();
        assert!(job.calculate(usize::MAX, &[0], true, true, false).is_err());
        assert!(job.calculate(1, &[0], true, true, false).is_err());
        assert!(job.calculate(2, &[], true, true, false).is_err());
        assert!(
            job.calculate(1, &[], true, true, false)
                .unwrap()
                .distances
                .is_empty()
        );
        #[cfg(not(feature = "threads"))]
        assert!(job.calculate(0, &[0], true, true, true).is_err());
    }

    fn fixture() -> (Vec<[f64; 3]>, Vec<f64>) {
        let points = (0..513)
            .map(|i| {
                [
                    3_955_000.0 + (i % 31) as f64 * 0.125,
                    368_000.0 + (i % 5) as f64,
                    if i % 2 == 0 { -0.0 } else { 0.0 },
                ]
            })
            .collect();
        let queries = (0..12_345)
            .flat_map(|i| {
                [
                    3_955_000.0 + (i % 61) as f64 * 0.0625,
                    368_000.000_009,
                    -0.0,
                ]
            })
            .collect();
        (points, queries)
    }

    #[test]
    fn owns_inputs_and_recovers_wrong_or_invalid_hints() {
        let (points, mut queries) = fixture();
        let mut bytes = CandidateKdTree::new(&points).unwrap().snapshot();
        let job = SharedGpuC2c::prepare(&bytes, &queries).unwrap();
        let expected: Vec<u64> = queries
            .as_chunks::<3>()
            .0
            .iter()
            .map(|q| {
                points
                    .iter()
                    .map(|p| (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2))
                    .reduce(f64::min)
                    .unwrap()
                    .sqrt()
                    .to_bits()
            })
            .collect();
        bytes.fill(0);
        queries.fill(0.0);
        let hints: Vec<u32> = (0..expected.len())
            .map(|i| [u32::MAX, 0, 512, 900][i % 4])
            .collect();
        for bounds in [false, true] {
            for simd in [false, true] {
                let result = job.calculate(0, &hints, bounds, simd, false).unwrap();
                assert_eq!(
                    result
                        .distances
                        .iter()
                        .map(|d| d.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                );
                for (first, count) in [(0, 1), (7, 4097), (12_344, 1)] {
                    let batch = job
                        .calculate(first, &hints[first..first + count], bounds, simd, false)
                        .unwrap();
                    assert_eq!(batch.distances, result.distances[first..first + count]);
                }
            }
        }
    }

    #[test]
    fn replaces_and_releases_jobs_without_changing_reference_or_distance_bits() {
        let points = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]];
        let bytes = CandidateKdTree::new(&points).unwrap().snapshot();
        let mut job = SharedGpuC2c::prepare(&bytes, &[1.0, 0.0, 0.0]).unwrap();
        let reference_bytes = job.tree.snapshot();
        let mut next = [2.0, 0.0, 0.0, 4.0, 0.0, 0.0];
        job.replace_queries(next.to_vec()).unwrap();
        next.fill(999.0);
        for invalid in [&[][..], &[1.0, 2.0], &[f64::NAN, 0.0, 0.0]] {
            assert!(job.replace_queries(invalid.to_vec()).is_err());
        }
        let expected = job.calculate(0, &[u32::MAX, 1], true, true, false).unwrap();
        assert_eq!(expected.distances, [2.0, 0.0]);
        assert_eq!(job.tree.snapshot(), reference_bytes);
        #[cfg(feature = "threads")]
        for n in [1, 2, 4] {
            let actual = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap()
                .install(|| job.calculate(0, &[u32::MAX, 1], true, true, true).unwrap());
            assert_eq!(
                actual
                    .distances
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .distances
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(actual.certified, expected.certified);
        }
        job.release_queries();
        job.release_queries();
        assert!(job.queries.is_empty());
        assert!(job.calculate(0, &[0], true, true, false).is_err());
        job.replace_queries(vec![3.0, 0.0, 0.0]).unwrap();
        assert_eq!(
            job.calculate(0, &[0], true, true, false).unwrap().distances,
            [1.0]
        );
        assert_eq!(job.tree.snapshot(), reference_bytes);
    }

    #[cfg(feature = "threads")]
    #[test]
    fn all_lanes_preserve_distance_bits_and_certificate_counts() {
        let (points, queries) = fixture();
        let tree = CandidateKdTree::new(&points).unwrap();
        let job = SharedGpuC2c::prepare(&tree.snapshot(), &queries).unwrap();
        let hints: Vec<u32> = (0..queries.len() / 3)
            .map(|i| [u32::MAX, 0, 127, 512][i % 4])
            .collect();
        for n in [1, 2, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap()
                .install(|| {
                    assert_eq!(thread_indices(), (0..n as u32).collect::<Vec<_>>());
                    for bounds in [false, true] {
                        for simd in [false, true] {
                            let serial = job.calculate(0, &hints, bounds, simd, false).unwrap();
                            let parallel = job.calculate(0, &hints, bounds, simd, true).unwrap();
                            assert_eq!(parallel.certified, serial.certified);
                            assert_eq!(
                                parallel
                                    .distances
                                    .iter()
                                    .map(|d| d.to_bits())
                                    .collect::<Vec<_>>(),
                                serial
                                    .distances
                                    .iter()
                                    .map(|d| d.to_bits())
                                    .collect::<Vec<_>>()
                            );
                        }
                    }
                });
        }
    }
}
