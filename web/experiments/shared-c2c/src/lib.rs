//! Optional shared-memory C2C experiment; the production WASM is unchanged.

use ca_core::distance::{DistanceStats, PreparedC2c};
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "threads", target_arch = "wasm32"))]
pub use wasm_bindgen_rayon::init_thread_pool;

const MAX_POINTS: usize = 2_000_000;

fn triples(xyz: &[f64]) -> Result<&[[f64; 3]], &'static str> {
    if xyz.is_empty() || !xyz.len().is_multiple_of(3) || xyz.len() / 3 > MAX_POINTS {
        return Err("expected 1..2000000 XYZ triples");
    }
    if !xyz.iter().all(|v| v.is_finite()) {
        return Err("coordinates must be finite");
    }
    Ok(xyz.as_chunks::<3>().0)
}

/// One immutable reference tree and query snapshot, shared by all Rayon lanes.
#[wasm_bindgen]
pub struct SharedC2c {
    inner: PreparedC2c,
}

impl SharedC2c {
    fn prepare(reference: &[f64], queries: &[f64]) -> Result<Self, &'static str> {
        let reference = triples(reference)?;
        let queries = triples(queries)?;
        Ok(Self {
            inner: PreparedC2c::new(queries.to_vec(), reference)
                .ok_or("reference cloud is empty")?,
        })
    }

    fn calculate(&self, parallel: bool) -> Result<Vec<f64>, &'static str> {
        if parallel {
            #[cfg(feature = "threads")]
            return Ok(self.inner.distances_par());
            #[cfg(not(feature = "threads"))]
            return Err("shared parallel computation requires the threads build");
        }
        Ok(self.inner.distances(false))
    }
}

#[wasm_bindgen]
impl SharedC2c {
    #[wasm_bindgen(constructor)]
    pub fn new(reference: &[f64], queries: &[f64]) -> Result<Self, JsError> {
        Self::prepare(reference, queries).map_err(JsError::new)
    }

    /// Logical retained vector capacity; excludes allocator and transient data.
    #[wasm_bindgen(getter, js_name = retainedBytes)]
    pub fn retained_bytes(&self) -> usize {
        self.inner.retained_bytes()
    }

    /// Same f32 outputs and ordered f64 summaries as the production C2C path.
    pub fn compute(&self, parallel: bool) -> Result<SharedResult, JsError> {
        let distances = self.calculate(parallel).map_err(JsError::new)?;
        let stats = DistanceStats::from_distances(&distances)
            .ok_or_else(|| JsError::new("compared cloud is empty"))?;
        Ok(SharedResult {
            distances: distances.into_iter().map(|d| d as f32).collect(),
            stats,
        })
    }

    /// Untimed diagnostic: compare every authoritative f64 bit as well.
    pub fn exact(&self, parallel: bool) -> Result<Vec<f64>, JsError> {
        self.calculate(parallel).map_err(JsError::new)
    }
}

#[wasm_bindgen]
pub struct SharedResult {
    distances: Vec<f32>,
    stats: DistanceStats,
}

#[wasm_bindgen]
impl SharedResult {
    #[wasm_bindgen(js_name = takeDistances)]
    pub fn take_distances(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.distances)
    }

    /// count, min, max, mean, rms, standard deviation, median.
    pub fn summary(&self) -> Vec<f64> {
        let s = self.stats;
        vec![
            s.count as f64,
            s.min,
            s.max,
            s.mean,
            s.rms,
            s.std_dev,
            s.median,
        ]
    }
}

/// An untimed broadcast proves that every initialized lane is responding.
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
    fn validates_inputs_before_snapshot_allocation() {
        for invalid in [
            &[][..],
            &[1.0, 2.0],
            &[0.0, f64::NAN, 0.0],
            &[f64::INFINITY; 3],
        ] {
            assert!(SharedC2c::prepare(invalid, &[0.0; 3]).is_err());
            assert!(SharedC2c::prepare(&[0.0; 3], invalid).is_err());
        }
        assert!(triples(&vec![0.0; (MAX_POINTS + 1) * 3]).is_err());
    }

    #[test]
    fn owns_snapshots_and_recomputes_distances() {
        let mut reference = [3_955_000.0, 368_000.0, -0.0, 3_955_001.0, 368_000.0, 1.0];
        let mut queries = [3_955_000.5, 368_000.0, 0.0];
        let snapshot = SharedC2c::prepare(&reference, &queries).unwrap();
        reference.fill(0.0);
        queries.fill(100.0);
        assert_eq!(snapshot.calculate(false).unwrap(), [0.5]);
        assert_eq!(snapshot.calculate(false).unwrap(), [0.5]);
        assert!(snapshot.retained_bytes() >= 3 * std::mem::size_of::<f64>());
        #[cfg(not(feature = "threads"))]
        assert!(snapshot.calculate(true).is_err());
    }

    #[cfg(feature = "threads")]
    #[test]
    fn shared_threads_preserve_every_distance_and_summary_bit() {
        let reference: Vec<f64> = (0..513)
            .flat_map(|i| {
                [
                    3_955_000.0 + (i % 17) as f64 * 0.125,
                    368_000.0,
                    (i % 3) as f64,
                ]
            })
            .collect();
        // Multiple Morton chunks, duplicate coordinates and nearest-hit ties.
        let queries: Vec<f64> = (0..12_345)
            .flat_map(|i| {
                [
                    3_955_000.0 + (i % 31) as f64 * 0.0625,
                    368_000.000_009,
                    -0.0,
                ]
            })
            .collect();
        let snapshot = SharedC2c::prepare(&reference, &queries).unwrap();
        let expected = snapshot.calculate(false).unwrap();
        for n in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap();
            pool.install(|| {
                assert_eq!(thread_indices(), (0..n as u32).collect::<Vec<_>>());
                let actual = snapshot.calculate(true).unwrap();
                assert_eq!(
                    actual.iter().map(|d| d.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|d| d.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(
                    DistanceStats::from_distances(&actual),
                    DistanceStats::from_distances(&expected)
                );
            });
        }
    }
}
