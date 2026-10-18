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

// Measured cross-matrix pairing did not consistently improve whole Map calls.
const USE_VOXEL_PAIR_EIGEN: bool = false;
const USE_VOXEL_NEIGHBORS: bool = true;

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
    voxel_gaussians_with_simd(points, voxel, min_points, false)
}

/// Optional WASM SIMD across independent moment channels. Each channel keeps
/// the scalar point order and separate multiply/add, without horizontal sums.
/// Other targets retain the scalar implementation.
pub fn voxel_gaussians_with_simd(
    points: &[[f64; 3]],
    voxel: f64,
    min_points: usize,
    simd: bool,
) -> HashMap<[i64; 3], Gaussian> {
    if simd {
        gaussians::<true>(points, voxel, min_points)
    } else {
        gaussians::<false>(points, voxel, min_points)
    }
}

fn add_moments<const SIMD: bool>(p: &[f64; 3], s: &mut [f64; 3], ss: &mut [[f64; 3]; 3]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if SIMD {
        use core::arch::wasm32::*;
        // SAFETY: each pointer addresses an array of three f64 values. These
        // unaligned loads/stores touch only the first two elements (16 bytes).
        unsafe {
            let xy = v128_load(p.as_ptr().cast());
            v128_store(
                s.as_mut_ptr().cast(),
                f64x2_add(v128_load(s.as_ptr().cast()), xy),
            );
            s[2] += p[2];
            for a in 0..3 {
                let row = &mut ss[a];
                let products = f64x2_mul(f64x2_splat(p[a]), xy);
                v128_store(
                    row.as_mut_ptr().cast(),
                    f64x2_add(v128_load(row.as_ptr().cast()), products),
                );
                row[2] += p[a] * p[2];
            }
        }
        return;
    }
    for a in 0..3 {
        s[a] += p[a];
        for b in 0..3 {
            ss[a][b] += p[a] * p[b];
        }
    }
}

fn gaussians<const SIMD: bool>(
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
        add_moments::<SIMD>(p, s, ss);
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
    let root_b = sqrt_psd(b.cov);
    wasserstein_with_root(a, b, &root_b)
}

fn wasserstein_with_root(a: &Gaussian, b: &Gaussian, root_b: &[[f64; 3]; 3]) -> f64 {
    let d2: f64 = (0..3).map(|k| (a.mean[k] - b.mean[k]).powi(2)).sum();
    let cross = sqrt_psd(mul(&mul(root_b, &a.cov), root_b));
    (d2 + trace(&a.cov) + trace(&b.cov) - 2.0 * trace(&cross))
        .max(0.0)
        .sqrt()
}

/// Independent cross-matrix decompositions share SIMD lanes; each matrix root
/// and distance retains the original scalar reconstruction and sum order.
fn wasserstein_pair_with_roots(
    a: [&Gaussian; 2],
    b: [&Gaussian; 2],
    roots: [[[f64; 3]; 3]; 2],
) -> [f64; 2] {
    let matrices = std::array::from_fn(|i| mul(&mul(&roots[i], &a[i].cov), &roots[i]));
    let Some(eigen) = crate::normal_pair::solve(matrices) else {
        return std::array::from_fn(|i| wasserstein_with_root(a[i], b[i], &roots[i]));
    };
    std::array::from_fn(|i| {
        let (values, vectors) = eigen[i];
        let mut cross = [[0.0; 3]; 3];
        for (k, &value) in values.iter().enumerate() {
            let s = value.max(0.0).sqrt();
            for row in 0..3 {
                for col in 0..3 {
                    cross[row][col] += s * vectors[row][k] * vectors[col][k];
                }
            }
        }
        let d2: f64 = (0..3).map(|k| (a[i].mean[k] - b[i].mean[k]).powi(2)).sum();
        (d2 + trace(&a[i].cov) + trace(&b[i].cov) - 2.0 * trace(&cross))
            .max(0.0)
            .sqrt()
    })
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

struct ReferenceGaussian {
    gaussian: Gaussian,
    root: Option<[[f64; 3]; 3]>,
    slot: usize,
}

/// Owned reference voxel statistics; lazy roots retain exact f64 arithmetic.
/// No original input buffers or final compared scores are retained.
pub struct VoxelReference {
    truth: HashMap<[i64; 3], ReferenceGaussian>,
    neighbors: Option<Vec<[u32; 27]>>,
    voxel: f64,
    min_points: usize,
    simd: bool,
    roots_reused: usize,
    roots_computed: usize,
    neighbors_used: usize,
}

impl VoxelReference {
    pub fn new(truth: &[[f64; 3]], voxel: f64, min_points: usize, simd: bool) -> Self {
        Self::new_with_neighbors(truth, voxel, min_points, simd, USE_VOXEL_NEIGHBORS)
    }

    fn new_with_neighbors(
        truth: &[[f64; 3]],
        voxel: f64,
        min_points: usize,
        simd: bool,
        neighbors: bool,
    ) -> Self {
        let truth: HashMap<_, _> = voxel_gaussians_with_simd(truth, voxel, min_points, simd)
            .into_iter()
            .enumerate()
            .map(|(slot, (key, gaussian))| {
                (
                    key,
                    ReferenceGaussian {
                        gaussian,
                        root: None,
                        slot,
                    },
                )
            })
            .collect();
        let neighbors = neighbors.then(|| reference_neighbors(&truth)).flatten();
        Self {
            truth,
            neighbors,
            voxel,
            min_points,
            simd,
            roots_reused: 0,
            roots_computed: 0,
            neighbors_used: 0,
        }
    }

    /// Conservative retained payload allowance, not total process memory.
    pub fn retained_bytes(&self) -> usize {
        self.truth
            .capacity()
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<([i64; 3], ReferenceGaussian)>())
            .saturating_add(self.neighbors.as_ref().map_or(0, |n| {
                n.capacity()
                    .saturating_mul(2)
                    .saturating_mul(std::mem::size_of::<[u32; 27]>())
            }))
            .saturating_add(4096)
    }

    pub fn roots_reused(&self) -> usize {
        self.roots_reused
    }
    pub fn roots_computed(&self) -> usize {
        self.roots_computed
    }
    pub fn neighbor_voxels(&self) -> usize {
        self.neighbors.as_ref().map_or(0, Vec::len)
    }
    pub fn neighbors_used(&self) -> usize {
        self.neighbors_used
    }

    pub fn compute(&mut self, estimate: &[[f64; 3]]) -> VoxelScores {
        self.compute_with_pair(
            estimate,
            USE_VOXEL_PAIR_EIGEN && cfg!(all(target_arch = "wasm32", target_feature = "simd128")),
        )
    }

    fn compute_with_pair(&mut self, estimate: &[[f64; 3]], pair: bool) -> VoxelScores {
        self.roots_reused = 0;
        self.roots_computed = 0;
        self.neighbors_used = 0;
        let ours = voxel_gaussians_with_simd(estimate, self.voxel, self.min_points, self.simd);
        if !pair && let Some(neighbors) = &self.neighbors {
            let mut distances = vec![None; self.truth.len()];
            let mut order = Vec::with_capacity(ours.len().min(self.truth.len()));
            for (key, a) in &ours {
                let Some(b) = self.truth.get_mut(key) else {
                    continue;
                };
                if b.root.is_some() {
                    self.roots_reused += 1;
                } else {
                    self.roots_computed += 1;
                }
                let root = b.root.get_or_insert_with(|| sqrt_psd(b.gaussian.cov));
                distances[b.slot] = Some(wasserstein_with_root(a, &b.gaussian, root));
                order.push(b.slot);
            }
            self.neighbors_used = order.len();
            return scores_from_neighbors(&distances, neighbors, &order);
        }
        if pair {
            let mut jobs = ours.iter().filter_map(|(key, a)| {
                let b = self.truth.get_mut(key)?;
                if b.root.is_some() {
                    self.roots_reused += 1;
                } else {
                    self.roots_computed += 1;
                }
                let root = b.root.get_or_insert_with(|| sqrt_psd(b.gaussian.cov));
                Some((*key, *a, b.gaussian, *root))
            });
            let mut pending = None;
            let distances: HashMap<_, _> = std::iter::from_fn(|| {
                if let Some(next) = pending.take() {
                    return Some(next);
                }
                let first = jobs.next()?;
                let Some(second) = jobs.next() else {
                    return Some((first.0, wasserstein_with_root(&first.1, &first.2, &first.3)));
                };
                let [a, b] = wasserstein_pair_with_roots(
                    [&first.1, &second.1],
                    [&first.2, &second.2],
                    [first.3, second.3],
                );
                pending = Some((second.0, b));
                Some((first.0, a))
            })
            .collect();
            return scores_from_distances(&distances, true);
        }
        let distances: HashMap<_, _> = ours
            .iter()
            .filter_map(|(key, a)| {
                let b = self.truth.get_mut(key)?;
                if b.root.is_some() {
                    self.roots_reused += 1;
                } else {
                    self.roots_computed += 1;
                }
                let root = b.root.get_or_insert_with(|| sqrt_psd(b.gaussian.cov));
                Some((*key, wasserstein_with_root(a, &b.gaussian, root)))
            })
            .collect();
        scores_from_distances(&distances, true)
    }
}

/// Fixed reference connectivity in historical dx/dy/dz order. Reserve the
/// all-ones slot for absence; extreme keys keep the preceding arithmetic path.
fn reference_neighbors(truth: &HashMap<[i64; 3], ReferenceGaussian>) -> Option<Vec<[u32; 27]>> {
    if truth.len() >= u32::MAX as usize
        || truth
            .keys()
            .any(|key| key.iter().any(|&v| v == i64::MIN || v == i64::MAX))
    {
        return None;
    }
    let mut neighbors = vec![[u32::MAX; 27]; truth.len()];
    for (key, value) in truth {
        let mut index = 0;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(other) = truth.get(&[key[0] + dx, key[1] + dy, key[2] + dz]) {
                        neighbors[value.slot][index] = other.slot as u32;
                    }
                    index += 1;
                }
            }
        }
    }
    Some(neighbors)
}

fn scores_from_neighbors(
    distances: &[Option<f64>],
    neighbors: &[[u32; 27]],
    order: &[usize],
) -> VoxelScores {
    if order.is_empty() {
        return VoxelScores {
            awd: f64::NAN,
            scs: f64::NAN,
            voxels: 0,
        };
    }
    let awd = order
        .iter()
        .map(|&slot| distances[slot].unwrap())
        .sum::<f64>()
        / order.len() as f64;
    let mut spread = 0.0;
    let mut counted = 0usize;
    let mut near = Vec::with_capacity(27);
    for &slot in order {
        near.clear();
        for &other in &neighbors[slot] {
            if other != u32::MAX
                && let Some(w) = distances[other as usize]
            {
                near.push(w);
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
        voxels: order.len(),
    }
}

/// [`VoxelScores`] of `estimate` against `truth`, with voxels of `voxel`
/// metres holding at least `min_points` points in both.
pub fn voxel_scores(
    estimate: &[[f64; 3]],
    truth: &[[f64; 3]],
    voxel: f64,
    min_points: usize,
) -> VoxelScores {
    voxel_scores_with_simd(estimate, truth, voxel, min_points, false)
}

/// Same scores with optional lane-wise WASM moment accumulation.
pub fn voxel_scores_with_simd(
    estimate: &[[f64; 3]],
    truth: &[[f64; 3]],
    voxel: f64,
    min_points: usize,
    simd: bool,
) -> VoxelScores {
    voxel_scores_with_scratch(estimate, truth, voxel, min_points, simd, true)
}

/// Retain the 27-neighbour SCS buffer across voxels; false is the allocating control.
pub fn voxel_scores_with_scratch(
    estimate: &[[f64; 3]],
    truth: &[[f64; 3]],
    voxel: f64,
    min_points: usize,
    simd: bool,
    reuse: bool,
) -> VoxelScores {
    let ours = voxel_gaussians_with_simd(estimate, voxel, min_points, simd);
    let theirs = voxel_gaussians_with_simd(truth, voxel, min_points, simd);
    let distances: HashMap<[i64; 3], f64> = ours
        .iter()
        .filter_map(|(key, g)| Some((*key, wasserstein(g, theirs.get(key)?))))
        .collect();
    scores_from_distances(&distances, reuse)
}

fn scores_from_distances(distances: &HashMap<[i64; 3], f64>, reuse: bool) -> VoxelScores {
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
    let mut near = Vec::with_capacity(27);
    for key in distances.keys() {
        if reuse {
            near.clear();
        } else {
            near = Vec::with_capacity(27);
        }
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

    fn close_scores(a: VoxelScores, b: VoxelScores) {
        assert_eq!(a.voxels, b.voxels);
        for (x, y) in [(a.awd, b.awd), (a.scs, b.scs)] {
            assert!(
                x.to_bits() == y.to_bits()
                    || x.is_nan() && y.is_nan()
                    || (x - y).abs() <= 1e-12 * x.abs().max(y.abs()).max(1.0),
                "{a:?} != {b:?}"
            );
        }
    }

    #[test]
    fn retained_neighbors_keep_masked_neighborhood_bits_and_replacement_scores() {
        let points: Vec<_> = (-3..=3)
            .flat_map(|x| {
                (-2..=2).flat_map(move |y| {
                    (-1..=1).flat_map(move |z| {
                        (0..5).map(move |j| {
                            [
                                x as f64 + 0.2 + j as f64 * 0.02,
                                y as f64 + 0.3,
                                z as f64 + 0.4 + j as f64 * 0.01,
                            ]
                        })
                    })
                })
            })
            .collect();
        for simd in [false, true] {
            let mut cached = VoxelReference::new_with_neighbors(&points, 1.0, 5, simd, true);
            let mut legacy = VoxelReference::new_with_neighbors(&points, 1.0, 5, simd, false);
            let bytes = cached.retained_bytes();
            assert!(cached.neighbors.is_some());
            for step in 0..7 {
                let mut values = vec![None; cached.truth.len()];
                let mut map = HashMap::new();
                let mut order = Vec::new();
                for (key, value) in &cached.truth {
                    if (value.slot + step) % 4 == 0 {
                        continue;
                    }
                    let distance = match (value.slot + step) % 7 {
                        0 => -0.0,
                        1 => f64::from_bits(1),
                        2 => f64::INFINITY,
                        3 => f64::NAN,
                        _ => value.slot as f64 * 0.03,
                    };
                    values[value.slot] = Some(distance);
                    map.insert(*key, distance);
                    order.push(value.slot);
                }
                for (key, reference) in &cached.truth {
                    let mut expected = Vec::new();
                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                if let Some(&w) = map.get(&[key[0] + dx, key[1] + dy, key[2] + dz])
                                {
                                    expected.push(w.to_bits());
                                }
                            }
                        }
                    }
                    let actual: Vec<_> = cached.neighbors.as_ref().unwrap()[reference.slot]
                        .iter()
                        .filter_map(|&slot| {
                            if slot == u32::MAX {
                                None
                            } else {
                                values[slot as usize].map(f64::to_bits)
                            }
                        })
                        .collect();
                    assert_eq!(actual, expected);
                }
                close_scores(
                    scores_from_neighbors(&values, cached.neighbors.as_ref().unwrap(), &order),
                    scores_from_distances(&map, true),
                );
                let query: Vec<_> = points
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| (i / 5 + step) % 4 != 0)
                    .map(|(_, p)| [p[0], p[1] + step as f64 * 0.005, p[2]])
                    .collect();
                let actual = cached.compute(&query);
                close_scores(actual, legacy.compute(&query));
                assert_eq!(cached.neighbors_used(), actual.voxels);
                assert_eq!(legacy.neighbors_used(), 0);
                assert_eq!(cached.roots_reused(), legacy.roots_reused());
                assert_eq!(cached.roots_computed(), legacy.roots_computed());
                assert_eq!(cached.retained_bytes(), bytes);
            }
            close_scores(cached.compute(&[]), legacy.compute(&[]));
            assert_eq!(cached.neighbors_used(), 0);
            close_scores(cached.compute(&points), legacy.compute(&points));
            assert_eq!(cached.retained_bytes(), bytes);
        }
    }

    #[test]
    fn retained_neighbors_reject_extreme_keys_and_charge_capacity() {
        for key in [[i64::MIN, 0, 0], [0, i64::MAX, 0], [0, 0, i64::MIN]] {
            let truth = HashMap::from([(
                key,
                ReferenceGaussian {
                    gaussian: Gaussian {
                        mean: [0.0; 3],
                        cov: [[0.0; 3]; 3],
                    },
                    root: None,
                    slot: 0,
                },
            )]);
            assert!(reference_neighbors(&truth).is_none());
        }
        let points = wall(0.25, 0.0);
        let prepared = VoxelReference::new_with_neighbors(&points, 1.0, 5, false, true);
        let expected =
            prepared.truth.capacity() * 2 * std::mem::size_of::<([i64; 3], ReferenceGaussian)>()
                + prepared.neighbors.as_ref().unwrap().capacity()
                    * 2
                    * std::mem::size_of::<[u32; 27]>()
                + 4096;
        assert_eq!(prepared.retained_bytes(), expected);
    }

    #[test]
    fn reference_gaussians_and_lazy_roots_preserve_scores_after_source_changes() {
        for offset in [0.0, 368000.0] {
            let truth: Vec<_> = wall(0.25, 0.0)
                .iter()
                .map(|p| [p[0] + offset, p[1] + offset * 10.0, p[2] + 40.0])
                .collect();
            for voxel in [0.1, 1.0, 1000.0] {
                for min_points in [0, 2, 5, 4096] {
                    for simd in [false, true] {
                        let mut source = truth.clone();
                        let mut prepared = VoxelReference::new(&source, voxel, min_points, simd);
                        source.fill([0.0; 3]);
                        let bytes = prepared.retained_bytes();
                        let gaussians = voxel_gaussians_with_simd(&truth, voxel, min_points, simd);
                        for (key, g) in &gaussians {
                            let actual = &prepared.truth[key].gaussian;
                            assert_eq!(g.mean.map(f64::to_bits), actual.mean.map(f64::to_bits));
                            assert_eq!(
                                g.cov.map(|r| r.map(f64::to_bits)),
                                actual.cov.map(|r| r.map(f64::to_bits))
                            );
                        }
                        let cold = prepared.compute(&truth);
                        assert_eq!(prepared.roots_computed(), cold.voxels);
                        assert_eq!(prepared.roots_reused(), 0);
                        for bend in [0.0, 0.006, 0.0] {
                            let estimate: Vec<_> = wall(0.35, bend)
                                .iter()
                                .map(|p| [p[0] + offset, p[1] + offset * 10.0, p[2] + 40.0])
                                .collect();
                            close_scores(
                                prepared.compute(&estimate),
                                voxel_scores_with_simd(&estimate, &truth, voxel, min_points, simd),
                            );
                            assert_eq!(prepared.retained_bytes(), bytes);
                        }
                        close_scores(prepared.compute(&truth), cold);
                        assert_eq!(prepared.roots_reused(), cold.voxels);
                        assert_eq!(prepared.roots_computed(), 0);
                        for b in prepared.truth.values() {
                            if let Some(root) = b.root {
                                for a in gaussians.values().take(7) {
                                    assert_eq!(
                                        wasserstein(a, &b.gaussian).to_bits(),
                                        wasserstein_with_root(a, &b.gaussian, &root).to_bits()
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        close_scores(
            VoxelReference::new(&[], 1.0, 5, false).compute(&[]),
            voxel_scores(&[], &[], 1.0, 5),
        );
    }

    #[test]
    fn stored_roots_preserve_degenerate_and_nonfinite_pair_bits() {
        for scale in [0.0, f64::from_bits(1), 1.0, 1e150, f64::INFINITY, f64::NAN] {
            for sign in [-1.0, 1.0] {
                let b = Gaussian {
                    mean: [sign * scale, -0.0, scale],
                    cov: [[scale, 0.0, 0.0], [0.0, -scale, 0.0], [0.0, 0.0, scale]],
                };
                let root = sqrt_psd(b.cov);
                for a in [
                    b,
                    Gaussian {
                        mean: [1.0, 2.0, 3.0],
                        cov: [[1.0, 0.2, 0.0], [0.2, 0.5, 0.0], [0.0, 0.0, 0.1]],
                    },
                ] {
                    assert_eq!(
                        wasserstein(&a, &b).to_bits(),
                        wasserstein_with_root(&a, &b, &root).to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn paired_cross_roots_preserve_individual_wasserstein_bits_and_fallbacks() {
        let scales = [
            0.0,
            f64::from_bits(1),
            1e-160,
            1e-100,
            1.0,
            1e50,
            1e100,
            f64::INFINITY,
            f64::NAN,
        ];
        for step in 0..128 {
            let gaussian = |scale: f64, diagonal: bool| Gaussian {
                mean: [scale * 0.3, -0.0, scale * -0.2],
                cov: if diagonal {
                    [
                        [scale, 0.0, 0.0],
                        [0.0, scale * 0.5, 0.0],
                        [0.0, 0.0, -scale * 0.1],
                    ]
                } else {
                    let off = (step as f64 * 0.07).sin() * scale * 0.2;
                    [
                        [scale, off, scale * 0.03],
                        [off, scale * 0.5, -off],
                        [scale * 0.03, -off, scale * 0.1],
                    ]
                },
            };
            for (i, &scale) in scales.iter().enumerate() {
                let a = [
                    gaussian(scale, step % 3 == 0),
                    gaussian(scales[(i + 3) % scales.len()], false),
                ];
                let b = [
                    gaussian(scale * 0.8, step % 5 == 0),
                    gaussian(scales[(i + 5) % scales.len()], step % 7 == 0),
                ];
                let roots = b.map(|g| sqrt_psd(g.cov));
                let actual = wasserstein_pair_with_roots([&a[0], &a[1]], [&b[0], &b[1]], roots);
                for lane in 0..2 {
                    assert_eq!(
                        actual[lane].to_bits(),
                        wasserstein_with_root(&a[lane], &b[lane], &roots[lane]).to_bits(),
                        "step {step}, scale {i}, lane {lane}"
                    );
                }
            }
        }
    }

    #[test]
    fn paired_reference_scores_keep_odd_tails_root_reuse_and_capacity() {
        for count in [0, 1, 2, 3, 17, 64] {
            let points: Vec<_> = (0..count)
                .flat_map(|i| {
                    (0..5).map(move |j| {
                        [
                            i as f64 * 2.0 + j as f64 * 0.02,
                            0.2 + (j % 2) as f64 * 0.03,
                            0.3 + j as f64 * 0.01,
                        ]
                    })
                })
                .collect();
            let mut scalar = VoxelReference::new(&points, 1.0, 5, false);
            let mut paired = VoxelReference::new(&points, 1.0, 5, false);
            let bytes = paired.retained_bytes();
            for shift in [0.0, 0.01, 0.02, 0.0] {
                let query: Vec<_> = points.iter().map(|p| [p[0], p[1] + shift, p[2]]).collect();
                let a = scalar.compute_with_pair(&query, false);
                let b = paired.compute_with_pair(&query, true);
                assert_eq!(b.voxels, count);
                close_scores(a, b);
                assert_eq!(scalar.roots_computed(), paired.roots_computed());
                assert_eq!(scalar.roots_reused(), paired.roots_reused());
                assert_eq!(paired.retained_bytes(), bytes);
            }
        }
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
    fn optional_moment_lanes_keep_scalar_gaussian_bits() {
        let points: Vec<_> = (0..1027)
            .map(|i| {
                let x = (i % 19) as f64 * 0.01;
                let y = (i % 7) as f64 * 0.02;
                [368_000.0 + x, 3_955_000.0 + y, 40.0 + x * y]
            })
            .chain([
                [-0.0, 0.0, f64::MIN_POSITIVE],
                [0.0, -0.0, -f64::MIN_POSITIVE],
                [f64::INFINITY, 1.0, 2.0],
                [f64::INFINITY, 2.0, 3.0],
                [f64::NAN, 4.0, 5.0],
                [f64::NAN, 4.0, 6.0],
            ])
            .collect();
        for voxel in [0.1, 1.0, 1e6] {
            for min_points in [0, 2, 20, 2048] {
                let scalar = voxel_gaussians(&points, voxel, min_points);
                let lanes = voxel_gaussians_with_simd(&points, voxel, min_points, true);
                assert_eq!(scalar.len(), lanes.len());
                for (key, a) in &scalar {
                    let b = &lanes[key];
                    assert_eq!(a.mean.map(f64::to_bits), b.mean.map(f64::to_bits));
                    assert_eq!(
                        a.cov.map(|row| row.map(f64::to_bits)),
                        b.cov.map(|row| row.map(f64::to_bits))
                    );
                }
            }
        }
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
