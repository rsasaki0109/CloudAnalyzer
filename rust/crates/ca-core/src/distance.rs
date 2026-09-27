//! Cloud-to-cloud (C2C) nearest-neighbour distances.

use kiddo::{ImmutableKdTree, SquaredEuclidean};

use crate::PointCloud;

/// Summary statistics over a set of distances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistanceStats {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub rms: f64,
    pub std_dev: f64,
    pub median: f64,
}

impl DistanceStats {
    pub fn from_distances(distances: &[f64]) -> Option<Self> {
        if distances.is_empty() {
            return None;
        }
        let n = distances.len() as f64;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for &d in distances {
            min = min.min(d);
            max = max.max(d);
            sum += d;
            sum_sq += d * d;
        }
        let mean = sum / n;
        let mut sorted = distances.to_vec();
        sorted.sort_unstable_by(f64::total_cmp);
        let mid = sorted.len() / 2;
        let median = if sorted.len().is_multiple_of(2) {
            0.5 * (sorted[mid - 1] + sorted[mid])
        } else {
            sorted[mid]
        };
        Some(Self {
            count: distances.len(),
            min,
            max,
            mean,
            rms: (sum_sq / n).sqrt(),
            std_dev: (sum_sq / n - mean * mean).max(0.0).sqrt(),
            median,
        })
    }
}

/// For every point of `compared`, the Euclidean distance to its nearest
/// neighbour in `reference`. Returns `None` when `reference` is empty.
pub fn cloud_to_cloud(compared: &PointCloud, reference: &PointCloud) -> Option<Vec<f64>> {
    if reference.is_empty() {
        return None;
    }
    let tree = ImmutableKdTree::<f64, 3>::new_from_slice(&reference.positions).ok()?;
    let mut distances = vec![0.0; compared.len()];
    // Querying in spatial order keeps the tree's hot path in cache; on
    // unordered inputs this is several times faster than file order.
    for i in morton_order(&compared.positions) {
        distances[i] = tree
            .query(&compared.positions[i])
            .nearest_one::<SquaredEuclidean<f64>>()
            .execute()
            .distance
            .sqrt();
    }
    Some(distances)
}

/// Indices of `points` sorted along a Z-order (Morton) curve.
fn morton_order(points: &[[f64; 3]]) -> Vec<usize> {
    let Some(first) = points.first() else {
        return Vec::new();
    };
    let (mut lo, mut hi) = (*first, *first);
    for p in points {
        for i in 0..3 {
            lo[i] = lo[i].min(p[i]);
            hi[i] = hi[i].max(p[i]);
        }
    }
    const CELLS: f64 = ((1u32 << 21) - 1) as f64;
    let scale: [f64; 3] = std::array::from_fn(|i| {
        let extent = hi[i] - lo[i];
        if extent > 0.0 { CELLS / extent } else { 0.0 }
    });
    let mut keyed: Vec<(u64, usize)> = points
        .iter()
        .enumerate()
        .map(|(idx, p)| {
            let cell = |i: usize| ((p[i] - lo[i]) * scale[i]) as u64;
            (
                spread_bits(cell(0)) | spread_bits(cell(1)) << 1 | spread_bits(cell(2)) << 2,
                idx,
            )
        })
        .collect();
    keyed.sort_unstable_by_key(|&(key, _)| key);
    keyed.into_iter().map(|(_, idx)| idx).collect()
}

/// Insert two zero bits between each of the low 21 bits of `v`.
fn spread_bits(v: u64) -> u64 {
    let mut x = v & 0x1f_ffff;
    x = (x | x << 32) & 0x1f_0000_0000_ffff;
    x = (x | x << 16) & 0x1f_0000_ff00_00ff;
    x = (x | x << 8) & 0x100f_00f0_0f00_f00f;
    x = (x | x << 4) & 0x10c3_0c30_c30c_30c3;
    x = (x | x << 2) & 0x1249_2492_4924_9249;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud(points: &[[f64; 3]]) -> PointCloud {
        PointCloud {
            positions: points.to_vec(),
            colors: None,
        }
    }

    #[test]
    fn distances_match_brute_force() {
        let reference = cloud(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]);
        let compared = cloud(&[[0.0, 0.0, 0.5], [1.0, 1.0, 0.0], [5.0, 0.0, 0.0]]);
        let d = cloud_to_cloud(&compared, &reference).unwrap();
        let expected = [0.5, 1.0, 4.0];
        for (got, want) in d.iter().zip(expected) {
            assert!((got - want).abs() < 1e-12, "{got} != {want}");
        }
    }

    #[test]
    fn spread_bits_interleaves() {
        assert_eq!(spread_bits(0b1), 0b1);
        assert_eq!(spread_bits(0b11), 0b1001);
        assert_eq!(spread_bits(0x1f_ffff), 0x1249_2492_4924_9249);
    }

    #[test]
    fn morton_order_is_a_permutation() {
        let points: Vec<[f64; 3]> = (0..100)
            .map(|i| {
                let f = i as f64;
                [(f * 7.3) % 10.0, (f * 3.1) % 5.0, 1.0]
            })
            .collect();
        let mut order = morton_order(&points);
        order.sort_unstable();
        assert_eq!(order, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn empty_reference_yields_none() {
        assert!(cloud_to_cloud(&cloud(&[[0.0; 3]]), &PointCloud::default()).is_none());
    }

    #[test]
    fn stats_are_correct() {
        let s = DistanceStats::from_distances(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(s.count, 4);
        assert_eq!(s.min, 1.0);
        assert_eq!(s.max, 4.0);
        assert_eq!(s.mean, 2.5);
        assert_eq!(s.median, 2.5);
        assert!((s.rms - 7.5f64.sqrt()).abs() < 1e-12);
        assert!((s.std_dev - 1.25f64.sqrt()).abs() < 1e-12);
    }
}
