//! Point cloud filters: spatial and random subsampling, statistical outlier
//! removal. Each returns the indices of the points to keep (in their original
//! order), so colors and attributes follow via [`PointCloud::select`].

use crate::PointCloud;
use crate::kdtree::KdTree;

/// Keep one point per cubic voxel of edge `voxel` (the first point, in cloud
/// order, that falls into each occupied voxel). Original points are kept, not
/// averaged, so their coordinates, colors and attributes stay exact.
pub fn voxel_subsample(cloud: &PointCloud, voxel: f64) -> Vec<usize> {
    // Also rejects NaN.
    if cloud.is_empty() || voxel.is_nan() || voxel <= 0.0 {
        return (0..cloud.len()).collect();
    }
    let lo = cloud.bounds().map(|b| b.min).unwrap_or([0.0; 3]);
    let mut keyed: Vec<(u64, u32)> = cloud
        .positions
        .iter()
        .zip(0u32..)
        .map(|(p, i)| {
            let cell = |a: usize| (((p[a] - lo[a]) / voxel) as u64).min((1 << 21) - 1);
            (cell(0) | cell(1) << 21 | cell(2) << 42, i)
        })
        .collect();
    // Stable sort keeps the first point of each voxel first.
    keyed.sort_by_key(|&(key, _)| key);
    let mut keep: Vec<usize> = Vec::new();
    let mut previous = None;
    for &(key, i) in &keyed {
        if previous != Some(key) {
            keep.push(i as usize);
            previous = Some(key);
        }
    }
    keep.sort_unstable();
    keep
}

/// Keep `count` points chosen uniformly at random (deterministic for a seed).
pub fn random_subsample(cloud: &PointCloud, count: usize, seed: u64) -> Vec<usize> {
    let n = cloud.len();
    if count >= n {
        return (0..n).collect();
    }
    // Partial Fisher-Yates over an index array with a xorshift generator.
    let mut indices: Vec<usize> = (0..n).collect();
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for i in 0..count {
        let j = i + (next() % (n - i) as u64) as usize;
        indices.swap(i, j);
    }
    indices.truncate(count);
    indices.sort_unstable();
    indices
}

/// Statistical outlier removal: drop points whose mean distance to their
/// `k` nearest neighbours exceeds the global mean by more than `ratio`
/// standard deviations (PCL's `StatisticalOutlierRemoval`).
pub fn statistical_outliers(cloud: &PointCloud, k: usize, ratio: f64) -> Vec<usize> {
    let n = cloud.len();
    let Some(tree) = KdTree::new(&cloud.positions) else {
        return Vec::new();
    };
    if n <= k || k == 0 {
        return (0..n).collect();
    }
    // Query in Morton order so consecutive searches touch the same part of
    // the tree. k + 1 because the nearest neighbour of a point is itself.
    let mut means = vec![0.0; n];
    for i in crate::distance::morton_order(&cloud.positions) {
        let hits = tree.nearest_k(&cloud.positions[i], k + 1);
        means[i] = hits.iter().skip(1).map(|&(_, d2)| d2.sqrt()).sum::<f64>() / k as f64;
    }
    let mean = means.iter().sum::<f64>() / n as f64;
    let var = means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / n as f64;
    let limit = mean + ratio * var.sqrt();
    (0..n).filter(|&i| means[i] <= limit).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Attribute, AttributeValues, INTENSITY};

    fn cloud(positions: Vec<[f64; 3]>) -> PointCloud {
        PointCloud {
            positions,
            colors: None,
            attributes: Vec::new(),
        }
    }

    fn grid(n: usize, step: f64) -> Vec<[f64; 3]> {
        (0..n * n)
            .map(|i| [(i % n) as f64 * step, (i / n) as f64 * step, 0.0])
            .collect()
    }

    #[test]
    fn voxel_keeps_one_point_per_voxel() {
        // A 100 x 100 grid with 0.1 spacing, subsampled with 0.5 voxels.
        let c = cloud(grid(100, 0.1));
        let keep = voxel_subsample(&c, 0.5);
        assert_eq!(keep.len(), 20 * 20);
        assert!(keep.windows(2).all(|w| w[0] < w[1]));
        // The first point of each voxel is kept, e.g. the origin.
        assert_eq!(keep[0], 0);
        // A non-positive size keeps everything.
        assert_eq!(voxel_subsample(&c, 0.0).len(), c.len());
    }

    #[test]
    fn random_subsample_is_exact_and_deterministic() {
        let c = cloud(grid(50, 1.0));
        let a = random_subsample(&c, 300, 7);
        let b = random_subsample(&c, 300, 7);
        assert_eq!(a, b);
        assert_eq!(a.len(), 300);
        let mut unique = a.clone();
        unique.dedup();
        assert_eq!(unique.len(), 300);
        assert_ne!(a, random_subsample(&c, 300, 8));
        assert_eq!(random_subsample(&c, 10_000, 1).len(), c.len());
    }

    #[test]
    fn sor_removes_isolated_points() {
        let mut positions = grid(40, 0.1);
        let outliers = [[10.0, 10.0, 5.0], [-8.0, 3.0, 2.0], [2.0, 2.0, 9.0]];
        positions.extend(outliers);
        let c = cloud(positions);
        let keep = statistical_outliers(&c, 8, 1.0);
        assert_eq!(keep.len(), 40 * 40, "only the three far points go");
        assert!(keep.iter().all(|&i| i < 1600));
    }

    #[test]
    fn select_keeps_attributes() {
        let mut c = cloud(grid(10, 1.0));
        c.attributes.push(Attribute {
            name: INTENSITY.into(),
            values: AttributeValues::F32((0..100).map(|i| i as f32).collect()),
        });
        let kept = c.select(&voxel_subsample(&c, 2.0));
        assert_eq!(kept.len(), 25);
        let AttributeValues::F32(v) = &kept.attributes[0].values else {
            panic!()
        };
        assert_eq!(v[1], 2.0);
    }
}
