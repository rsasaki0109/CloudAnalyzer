//! Map quality against a ground-truth map, after MapEval (JokerJohn/
//! Cloud_Map_Evaluation): both maps are cut into voxels, the points of
//! each voxel summed up as a Gaussian, and matching voxels compared by the
//! Wasserstein distance between their Gaussians. Its average (AWD) is
//! small only when the local structure matches, not just the nearest
//! points; the spatial consistency score (SCS) is the spread of that
//! distance among neighbouring voxels, large where a map is warped.
//!
//! Point-wise accuracy and completeness come from nearest distances (see
//! the web app's Map quality method).

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gaussian {
    pub mean: [f64; 3],
    pub cov: [[f64; 3]; 3],
}

/// The Gaussians of the voxels holding at least `min_points` points.
pub fn voxel_gaussians(
    points: &[[f64; 3]],
    voxel: f64,
    min_points: usize,
) -> HashMap<[i64; 3], Gaussian> {
    type Sums = (usize, [f64; 3], [[f64; 3]; 3]);
    let mut sums: HashMap<[i64; 3], Sums> = HashMap::new();
    for p in points {
        let key = p.map(|v| (v / voxel).floor() as i64);
        let (n, s, ss) = sums.entry(key).or_insert((0, [0.0; 3], [[0.0; 3]; 3]));
        *n += 1;
        for a in 0..3 {
            s[a] += p[a];
            for b in 0..3 {
                ss[a][b] += p[a] * p[b];
            }
        }
    }
    sums.into_iter()
        .filter(|(_, (n, _, _))| *n >= min_points.max(2))
        .map(|(key, (n, s, ss))| {
            let n = n as f64;
            let mean = s.map(|v| v / n);
            let mut cov = [[0.0; 3]; 3];
            for a in 0..3 {
                for b in 0..3 {
                    cov[a][b] = ss[a][b] / n - mean[a] * mean[b];
                }
            }
            (key, Gaussian { mean, cov })
        })
        .collect()
}

/// Square root of a symmetric positive semi-definite matrix.
fn sqrt_psd(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let (values, vectors) = crate::icp::symmetric_eigen(m);
    let mut out = [[0.0; 3]; 3];
    for (k, &value) in values.iter().enumerate() {
        let s = value.max(0.0).sqrt();
        for a in 0..3 {
            for b in 0..3 {
                out[a][b] += s * vectors[a][k] * vectors[b][k];
            }
        }
    }
    out
}

fn mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn trace(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] + m[1][1] + m[2][2]
}

/// The 2-Wasserstein distance between two Gaussians (metres).
pub fn wasserstein(a: &Gaussian, b: &Gaussian) -> f64 {
    let d2: f64 = (0..3).map(|k| (a.mean[k] - b.mean[k]).powi(2)).sum();
    let root_b = sqrt_psd(b.cov);
    let cross = sqrt_psd(mul(&mul(&root_b, &a.cov), &root_b));
    (d2 + trace(&a.cov) + trace(&b.cov) - 2.0 * trace(&cross))
        .max(0.0)
        .sqrt()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxelScores {
    /// Average Wasserstein distance over the voxels both maps fill (metres).
    pub awd: f64,
    /// Spatial consistency: the mean, over those voxels, of the standard
    /// deviation of the distance within each voxel's 3x3x3 neighbourhood.
    pub scs: f64,
    /// Voxels both maps fill.
    pub voxels: usize,
}

/// [`VoxelScores`] of `estimate` against `truth`, with voxels of `voxel`
/// metres holding at least `min_points` points in both.
pub fn voxel_scores(
    estimate: &[[f64; 3]],
    truth: &[[f64; 3]],
    voxel: f64,
    min_points: usize,
) -> VoxelScores {
    let ours = voxel_gaussians(estimate, voxel, min_points);
    let theirs = voxel_gaussians(truth, voxel, min_points);
    let distances: HashMap<[i64; 3], f64> = ours
        .iter()
        .filter_map(|(key, g)| Some((*key, wasserstein(g, theirs.get(key)?))))
        .collect();
    if distances.is_empty() {
        return VoxelScores {
            awd: f64::NAN,
            scs: f64::NAN,
            voxels: 0,
        };
    }
    let awd = distances.values().sum::<f64>() / distances.len() as f64;
    let mut spread = 0.0;
    let mut counted = 0usize;
    for key in distances.keys() {
        let mut near = Vec::with_capacity(27);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(&w) = distances.get(&[key[0] + dx, key[1] + dy, key[2] + dz]) {
                        near.push(w);
                    }
                }
            }
        }
        if near.len() >= 2 {
            let mean = near.iter().sum::<f64>() / near.len() as f64;
            spread +=
                (near.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / near.len() as f64).sqrt();
            counted += 1;
        }
    }
    VoxelScores {
        awd,
        scs: if counted > 0 {
            spread / counted as f64
        } else {
            0.0
        },
        voxels: distances.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wall(offset: f64, bend: f64) -> Vec<[f64; 3]> {
        let mut out = Vec::new();
        for x in 0..100 {
            for z in 0..20 {
                let x = x as f64 * 0.1;
                out.push([x, offset + bend * x * x, z as f64 * 0.1]);
            }
        }
        out
    }

    #[test]
    fn wasserstein_of_shifted_gaussians_is_the_shift() {
        let a = Gaussian {
            mean: [0.0; 3],
            cov: [[1.0, 0.2, 0.0], [0.2, 0.5, 0.0], [0.0, 0.0, 0.1]],
        };
        let b = Gaussian {
            mean: [0.3, 0.0, 0.4],
            ..a
        };
        assert!((wasserstein(&a, &b) - 0.5).abs() < 1e-9);
        assert!(wasserstein(&a, &a) < 1e-4);
    }

    #[test]
    fn the_same_map_scores_zero_and_a_warped_one_does_not() {
        let truth = wall(0.25, 0.0);
        let same = voxel_scores(&truth, &truth, 1.0, 10);
        assert!(same.voxels > 10 && same.awd < 1e-4 && same.scs < 1e-4);
        let shifted = voxel_scores(&wall(0.35, 0.0), &truth, 1.0, 10);
        assert!((shifted.awd - 0.1).abs() < 1e-3, "{shifted:?}");
        // A shift is consistent everywhere; a bend is not.
        assert!(shifted.scs < 1e-3);
        let bent = voxel_scores(&wall(0.25, 0.006), &truth, 1.0, 10);
        assert!(bent.awd > 0.05 && bent.scs > 0.01, "{bent:?}");
    }
}
