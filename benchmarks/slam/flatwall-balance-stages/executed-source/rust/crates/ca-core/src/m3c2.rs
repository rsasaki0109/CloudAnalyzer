//! M3C2 distance (Lague, Brodu & Leroux 2013): change between two clouds
//! measured along local surface normals, with a per-point confidence bound.
//!
//! For each core point, a normal is fitted to the first cloud within
//! `normal_radius`; both clouds are then sampled in a cylinder of radius
//! `projection_radius` along that normal (up to `max_depth` either way), and
//! the distance is the difference of their mean positions along the normal.
//! The 95 % level of detection (LoD95) comes from the spread of both
//! samples; a change larger than it is significant.

use crate::kdtree::KdTree;
use std::borrow::Cow;

// Extra length guarding did not consistently improve whole measured calls.
const USE_M3C2_SHORT_PAIR_GUARD: bool = false;

#[derive(Debug, Clone, Copy)]
pub struct M3c2Params {
    /// Neighbourhood radius for the normal (the paper's D / 2).
    pub normal_radius: f64,
    /// Cylinder radius for averaging each cloud (the paper's d / 2).
    pub projection_radius: f64,
    /// Half-length of the cylinder along the normal.
    pub max_depth: f64,
    /// Points needed in each cylinder for a distance.
    pub min_points: usize,
    /// Registration error added to the LoD95.
    pub registration_error: f64,
}

impl Default for M3c2Params {
    fn default() -> Self {
        Self {
            normal_radius: 1.0,
            projection_radius: 0.5,
            max_depth: 2.0,
            min_points: 5,
            registration_error: 0.0,
        }
    }
}

/// Per-core-point results; NaN where a distance could not be computed.
#[derive(Debug, Clone, PartialEq)]
pub struct M3c2Result {
    /// Signed distance from cloud 1 to cloud 2 along the (upward) normal.
    pub distance: Vec<f64>,
    pub lod95: Vec<f64>,
    /// |distance| > LoD95.
    pub significant: Vec<bool>,
    /// Normals used (unit, z >= 0), NaN where none could be fitted.
    pub normals: Vec<[f64; 3]>,
}

/// Two independent reference hit lists while fitting adjacent cores together.
#[derive(Default)]
pub struct M3c2PairScratch {
    normals: [Vec<usize>; 2],
    projections: [Vec<usize>; 2],
}

type FreshResult = (f64, f64, Option<[f64; 3]>);

/// Fit-and-project state shared by the serial and parallel drivers.
pub struct M3c2<'a> {
    tree1: Cow<'a, KdTree>,
    tree2: Cow<'a, KdTree>,
    params: M3c2Params,
    radius_bounds: bool,
    reach: f64,
    projection_radius_sq: f64,
    cached_geometry: bool,
    fused_projection: bool,
}

impl<'a> M3c2<'a> {
    /// Returns `None` if either cloud is empty or a radius is not positive.
    pub fn new(cloud1: &'a [[f64; 3]], cloud2: &'a [[f64; 3]], params: M3c2Params) -> Option<Self> {
        Self::with_radius_bounds(cloud1, cloud2, params, false)
    }

    /// Select radius pruning independently of point scheduling and arithmetic.
    pub fn with_radius_bounds(
        cloud1: &'a [[f64; 3]],
        cloud2: &'a [[f64; 3]],
        params: M3c2Params,
        radius_bounds: bool,
    ) -> Option<Self> {
        Self::from_trees(
            Cow::Owned(KdTree::new(cloud1)?),
            Cow::Owned(KdTree::new(cloud2)?),
            params,
            radius_bounds,
        )
    }

    /// Reuse immutable exact trees without original coordinate buffers.
    pub fn with_trees(
        tree1: &'a KdTree,
        tree2: &'a KdTree,
        params: M3c2Params,
        radius_bounds: bool,
    ) -> Option<Self> {
        Self::from_trees(
            Cow::Borrowed(tree1),
            Cow::Borrowed(tree2),
            params,
            radius_bounds,
        )
    }

    fn from_trees(
        tree1: Cow<'a, KdTree>,
        tree2: Cow<'a, KdTree>,
        params: M3c2Params,
        radius_bounds: bool,
    ) -> Option<Self> {
        let positive = |v: f64| v > 0.0 && v.is_finite();
        if !(positive(params.normal_radius)
            && positive(params.projection_radius)
            && positive(params.max_depth))
        {
            return None;
        }
        Some(Self {
            tree1,
            tree2,
            reach: params.projection_radius.hypot(params.max_depth),
            projection_radius_sq: params.projection_radius.powi(2),
            cached_geometry: false,
            fused_projection: false,
            params,
            radius_bounds,
        })
    }

    /// Internal comparison: compute invariant cylinder geometry once per job.
    pub fn with_cached_geometry(mut self, enabled: bool) -> Self {
        self.cached_geometry = enabled;
        self
    }

    /// Fuse independent projection queries with their ordered accumulations.
    pub fn with_fused_projection(mut self, enabled: bool) -> Self {
        self.fused_projection = enabled;
        self
    }

    fn projection_reach(&self) -> f64 {
        if self.cached_geometry {
            self.reach
        } else {
            self.params.projection_radius.hypot(self.params.max_depth)
        }
    }

    /// `(distance, lod95, normal)` at one core point.
    pub fn at(&self, core: &[f64; 3], scratch: &mut Vec<usize>) -> (f64, f64, [f64; 3]) {
        self.at_with_normal(core, self.normal_at(core, scratch), scratch)
    }

    /// Normal fitting depends only on the reference tree, core and normal radius.
    pub fn normal_at(&self, core: &[f64; 3], scratch: &mut Vec<usize>) -> Option<[f64; 3]> {
        self.tree1.within_slots_with_bounds(
            core,
            self.params.normal_radius,
            scratch,
            self.radius_bounds,
        );
        fit_normal(self.tree1.tree_points(), scratch)
    }

    /// Project with a previously fitted normal, retaining the original query
    /// and accumulation order. `None` also preserves undefined normal results.
    pub fn at_with_normal(
        &self,
        core: &[f64; 3],
        normal: Option<[f64; 3]>,
        scratch: &mut Vec<usize>,
    ) -> (f64, f64, [f64; 3]) {
        const NAN3: [f64; 3] = [f64::NAN; 3];
        let Some(normal) = normal else {
            return (f64::NAN, f64::NAN, NAN3);
        };
        let reach = self.projection_reach();
        let mut project = |tree: &KdTree| {
            if self.fused_projection {
                return self.project_tree(tree, core, &normal, reach);
            }
            tree.within_slots_with_bounds(core, reach, scratch, self.radius_bounds);
            self.project_slots(tree.tree_points(), core, &normal, scratch)
        };
        self.finish_projection(project(&self.tree1), project(&self.tree2), normal)
    }

    /// Fit fresh normals and project, optionally sharing the reference search.
    /// The validity tag distinguishes a failed fit from a non-finite normal.
    pub fn at_with_search(
        &self,
        core: &[f64; 3],
        scratch: &mut Vec<usize>,
        projection: &mut Vec<usize>,
        shared_search: bool,
    ) -> (f64, f64, Option<[f64; 3]>) {
        if !shared_search {
            let normal = self.normal_at(core, scratch);
            let (distance, lod, _) = self.at_with_normal(core, normal, scratch);
            return (distance, lod, normal);
        }
        let reach = self.projection_reach();
        self.tree1.within_two_slots_with_bounds(
            core,
            [self.params.normal_radius, reach],
            [scratch, projection],
            self.radius_bounds,
        );
        let normal = fit_normal(self.tree1.tree_points(), scratch);
        let Some(n) = normal else {
            return (f64::NAN, f64::NAN, None);
        };
        let reference = self.project_slots(self.tree1.tree_points(), core, &n, projection);
        let compared = if self.fused_projection {
            self.project_tree(&self.tree2, core, &n, reach)
        } else {
            self.tree2
                .within_slots_with_bounds(core, reach, projection, self.radius_bounds);
            self.project_slots(self.tree2.tree_points(), core, &n, projection)
        };
        let (distance, lod, _) = self.finish_projection(reference, compared, n);
        (distance, lod, normal)
    }

    /// Pair only independent normal eigensolves, preserving each core's shared
    /// search results and projection order. The scratch belongs to one slice.
    pub fn at_pair_with_search(
        &self,
        cores: &[[f64; 3]; 2],
        scratch: &mut M3c2PairScratch,
        shared_search: bool,
    ) -> [FreshResult; 2] {
        let reach = self.projection_reach();
        for (i, core) in cores.iter().enumerate() {
            if shared_search {
                self.tree1.within_two_slots_with_bounds(
                    core,
                    [self.params.normal_radius, reach],
                    [&mut scratch.normals[i], &mut scratch.projections[i]],
                    self.radius_bounds,
                );
            } else {
                self.tree1.within_slots_with_bounds(
                    core,
                    self.params.normal_radius,
                    &mut scratch.normals[i],
                    self.radius_bounds,
                );
            }
        }
        let normals = fit_normal_pair(
            self.tree1.tree_points(),
            [&scratch.normals[0], &scratch.normals[1]],
        );
        std::array::from_fn(|i| {
            let normal = normals[i];
            if !shared_search {
                let (distance, lod, _) =
                    self.at_with_normal(&cores[i], normal, &mut scratch.normals[i]);
                return (distance, lod, normal);
            }
            let Some(n) = normal else {
                return (f64::NAN, f64::NAN, None);
            };
            let reference = self.project_slots(
                self.tree1.tree_points(),
                &cores[i],
                &n,
                &scratch.projections[i],
            );
            let compared = if self.fused_projection {
                self.project_tree(&self.tree2, &cores[i], &n, reach)
            } else {
                self.tree2.within_slots_with_bounds(
                    &cores[i],
                    reach,
                    &mut scratch.projections[i],
                    self.radius_bounds,
                );
                self.project_slots(
                    self.tree2.tree_points(),
                    &cores[i],
                    &n,
                    &scratch.projections[i],
                )
            };
            let (distance, lod, _) = self.finish_projection(reference, compared, n);
            (distance, lod, normal)
        })
    }

    /// Original individual fit for an odd tail, reusing the paired scratch.
    pub fn at_single_with_pair_scratch(
        &self,
        core: &[f64; 3],
        scratch: &mut M3c2PairScratch,
        shared_search: bool,
    ) -> FreshResult {
        self.at_with_search(
            core,
            &mut scratch.normals[0],
            &mut scratch.projections[0],
            shared_search,
        )
    }

    fn project_slots(
        &self,
        cloud: &[[f64; 3]],
        core: &[f64; 3],
        normal: &[f64; 3],
        slots: &[usize],
    ) -> Option<(f64, f64, f64)> {
        let r2 = if self.cached_geometry {
            self.projection_radius_sq
        } else {
            self.params.projection_radius.powi(2)
        };
        let (mut n, mut sum, mut sum2) = (0usize, 0.0, 0.0);
        for &i in slots.iter() {
            let v = [
                cloud[i][0] - core[0],
                cloud[i][1] - core[1],
                cloud[i][2] - core[2],
            ];
            let along = v[0] * normal[0] + v[1] * normal[1] + v[2] * normal[2];
            let radial2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - along * along;
            if along.abs() <= self.params.max_depth && radial2 <= r2 {
                n += 1;
                sum += along;
                sum2 += along * along;
            }
        }
        self.projection_stats(n, sum, sum2)
    }

    fn project_tree(
        &self,
        tree: &KdTree,
        core: &[f64; 3],
        normal: &[f64; 3],
        reach: f64,
    ) -> Option<(f64, f64, f64)> {
        let r2 = if self.cached_geometry {
            self.projection_radius_sq
        } else {
            self.params.projection_radius.powi(2)
        };
        let (mut n, mut sum, mut sum2) = (0usize, 0.0, 0.0);
        tree.visit_within_with_bounds(core, reach, self.radius_bounds, &mut |_, v, distance_sq| {
            let along = v[0] * normal[0] + v[1] * normal[1] + v[2] * normal[2];
            let radial2 = distance_sq - along * along;
            if along.abs() <= self.params.max_depth && radial2 <= r2 {
                n += 1;
                sum += along;
                sum2 += along * along;
            }
        });
        self.projection_stats(n, sum, sum2)
    }

    fn projection_stats(&self, n: usize, sum: f64, sum2: f64) -> Option<(f64, f64, f64)> {
        (n >= self.params.min_points.max(1)).then(|| {
            let mean = sum / n as f64;
            // Sample variance (n - 1), zero for a single point.
            let var = if n > 1 {
                ((sum2 - n as f64 * mean * mean) / (n - 1) as f64).max(0.0)
            } else {
                0.0
            };
            (mean, var, n as f64)
        })
    }

    fn finish_projection(
        &self,
        reference: Option<(f64, f64, f64)>,
        compared: Option<(f64, f64, f64)>,
        normal: [f64; 3],
    ) -> (f64, f64, [f64; 3]) {
        let (Some((m1, v1, n1)), Some((m2, v2, n2))) = (reference, compared) else {
            return (f64::NAN, f64::NAN, normal);
        };
        let lod = 1.96 * (v1 / n1 + v2 / n2).sqrt() + self.params.registration_error;
        (m2 - m1, lod, normal)
    }
}

/// M3C2 at every core point.
pub fn m3c2(
    core: &[[f64; 3]],
    cloud1: &[[f64; 3]],
    cloud2: &[[f64; 3]],
    params: M3c2Params,
) -> Option<M3c2Result> {
    m3c2_with_radius_bounds(core, cloud1, cloud2, params, false)
}

/// Same results and accumulation order as [`m3c2`], with a pruning control.
pub fn m3c2_with_radius_bounds(
    core: &[[f64; 3]],
    cloud1: &[[f64; 3]],
    cloud2: &[[f64; 3]],
    params: M3c2Params,
    radius_bounds: bool,
) -> Option<M3c2Result> {
    m3c2_with_search(core, cloud1, cloud2, params, radius_bounds, true)
}

/// Select shared reference traversal independently of scheduling and caching.
pub fn m3c2_with_search(
    core: &[[f64; 3]],
    cloud1: &[[f64; 3]],
    cloud2: &[[f64; 3]],
    params: M3c2Params,
    radius_bounds: bool,
    shared_search: bool,
) -> Option<M3c2Result> {
    m3c2_with_projection(
        core,
        cloud1,
        cloud2,
        params,
        radius_bounds,
        shared_search,
        false,
    )
}

/// Independent ordered projection control, preserving normal fitting.
pub fn m3c2_with_projection(
    core: &[[f64; 3]],
    cloud1: &[[f64; 3]],
    cloud2: &[[f64; 3]],
    params: M3c2Params,
    radius_bounds: bool,
    shared_search: bool,
    fused_projection: bool,
) -> Option<M3c2Result> {
    let m = M3c2::with_radius_bounds(cloud1, cloud2, params, radius_bounds)?
        .with_fused_projection(fused_projection);
    let (mut scratch, mut projection) = (Vec::new(), Vec::new());
    Some(collect(core.iter().map(|c| {
        let (distance, lod, normal) =
            m.at_with_search(c, &mut scratch, &mut projection, shared_search);
        (distance, lod, normal.unwrap_or([f64::NAN; 3]))
    })))
}

/// Multi-threaded [`m3c2`] (feature `parallel`); identical results.
#[cfg(feature = "parallel")]
pub fn m3c2_par(
    core: &[[f64; 3]],
    cloud1: &[[f64; 3]],
    cloud2: &[[f64; 3]],
    params: M3c2Params,
) -> Option<M3c2Result> {
    use rayon::prelude::*;
    let m = M3c2::new(cloud1, cloud2, params)?;
    let parts: Vec<(f64, f64, [f64; 3])> = core
        .par_chunks(1024)
        .flat_map_iter(|chunk| {
            let (mut scratch, mut projection) = (Vec::new(), Vec::new());
            chunk
                .iter()
                .map(|c| {
                    let (d, lod, normal) = m.at_with_search(c, &mut scratch, &mut projection, true);
                    (d, lod, normal.unwrap_or([f64::NAN; 3]))
                })
                .collect::<Vec<_>>()
        })
        .collect();
    Some(collect(parts.into_iter()))
}

fn collect(results: impl Iterator<Item = (f64, f64, [f64; 3])>) -> M3c2Result {
    let mut out = M3c2Result {
        distance: Vec::new(),
        lod95: Vec::new(),
        significant: Vec::new(),
        normals: Vec::new(),
    };
    for (d, lod, n) in results {
        out.distance.push(d);
        out.lod95.push(lod);
        out.significant.push(d.abs() > lod);
        out.normals.push(n);
    }
    out
}

/// Unit normal of the points at `indices` by PCA (smallest-variance axis),
/// oriented upward (z >= 0, or x >= 0 for horizontal normals).
fn fit_normal(points: &[[f64; 3]], indices: &[usize]) -> Option<[f64; 3]> {
    if indices.len() < 3 {
        return None;
    }
    let n = indices.len() as f64;
    let mut c = [0.0; 3];
    for &i in indices {
        for a in 0..3 {
            c[a] += points[i][a] / n;
        }
    }
    let mut cov = [[0.0; 3]; 3];
    for &i in indices {
        let d = [
            points[i][0] - c[0],
            points[i][1] - c[1],
            points[i][2] - c[2],
        ];
        crate::normals::accumulate_covariance(&mut cov, &d, false);
    }
    let (values, vectors) = crate::icp::symmetric_eigen(cov);
    let k = (0..3).min_by(|&a, &b| values[a].total_cmp(&values[b]))?;
    let mid = (0..3)
        .filter(|&a| a != k)
        .map(|a| values[a])
        .fold(f64::INFINITY, f64::min);
    if mid
        <= 1e-12
            * values
                .iter()
                .copied()
                .fold(0.0, f64::max)
                .max(f64::MIN_POSITIVE)
    {
        return None; // collinear or a single point: no plane
    }
    let mut normal = [vectors[0][k], vectors[1][k], vectors[2][k]];
    let flip = normal[2] < 0.0 || (normal[2] == 0.0 && normal[0] < 0.0);
    if flip {
        normal = normal.map(|v| -v);
    }
    Some(normal)
}

fn fit_normal_pair(points: &[[f64; 3]], indices: [&[usize]; 2]) -> [Option<[f64; 3]>; 2] {
    // Both covariance closures would run before a short member triggers the
    // scalar fallback. Skip that staging so its valid partner is fitted once.
    if USE_M3C2_SHORT_PAIR_GUARD && indices.iter().any(|hits| hits.len() < 3) {
        return indices.map(|hits| fit_normal(points, hits));
    }
    let covariance = |indices: &[usize]| {
        if indices.len() < 3 {
            return None;
        }
        let n = indices.len() as f64;
        let mut c = [0.0; 3];
        for &i in indices {
            for a in 0..3 {
                c[a] += points[i][a] / n;
            }
        }
        let mut cov = [[0.0; 3]; 3];
        for &i in indices {
            let d = [
                points[i][0] - c[0],
                points[i][1] - c[1],
                points[i][2] - c[2],
            ];
            crate::normals::accumulate_covariance(&mut cov, &d, false);
        }
        Some(cov)
    };
    let [Some(first), Some(second)] = indices.map(covariance) else {
        return indices.map(|hits| fit_normal(points, hits));
    };
    let Some(eigen) = crate::normal_pair::solve([first, second]) else {
        return indices.map(|hits| fit_normal(points, hits));
    };
    eigen.map(|(values, vectors)| {
        let k = (0..3).min_by(|&a, &b| values[a].total_cmp(&values[b]))?;
        let mid = (0..3)
            .filter(|&a| a != k)
            .map(|a| values[a])
            .fold(f64::INFINITY, f64::min);
        if mid
            <= 1e-12
                * values
                    .iter()
                    .copied()
                    .fold(0.0, f64::max)
                    .max(f64::MIN_POSITIVE)
        {
            return None;
        }
        let mut normal = [vectors[0][k], vectors[1][k], vectors[2][k]];
        let flip = normal[2] < 0.0 || (normal[2] == 0.0 && normal[0] < 0.0);
        if flip {
            normal = normal.map(|v| -v);
        }
        Some(normal)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_paired_neighborhoods_preserve_partner_bits_in_both_orders() {
        for scale in [1e-160, 1.0, 1e60, 1e155] {
            for offset in [0.0, 368000.0] {
                let points: Vec<_> = (0..73)
                    .map(|i| {
                        [
                            offset + (i % 9) as f64 * 0.1 * scale,
                            offset * 10.0 + (i / 9) as f64 * 0.1 * scale,
                            40.0 + (i as f64 * 0.7).sin() * 0.01 * scale,
                        ]
                    })
                    .collect();
                let all: Vec<_> = (0..points.len()).collect();
                for short in 0..3 {
                    for full in [3, 17, 73] {
                        for hits in [[&all[..short], &all[..full]], [&all[..full], &all[..short]]] {
                            let actual = fit_normal_pair(&points, hits)
                                .map(|n| n.map(|n| n.map(f64::to_bits)));
                            let expected =
                                hits.map(|h| fit_normal(&points, h).map(|n| n.map(f64::to_bits)));
                            assert_eq!(actual, expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn paired_fresh_fits_preserve_search_projection_bits_and_validity_tags() {
        let bits = |result: FreshResult| {
            (
                result.0.to_bits(),
                result.1.to_bits(),
                result.2.map(|normal| normal.map(f64::to_bits)),
            )
        };
        for scale in [1e-160, 1e-100, 1.0, 1e60, 1e80, 1e155] {
            for mode in [0, 1, 2, 3] {
                let points: Vec<_> = (0..73)
                    .map(|i| {
                        let p = match mode {
                            1 => [1.0, -0.0, 0.0],
                            2 => [i as f64 * 0.1, 0.0, 0.0],
                            _ => [
                                i as f64 % 9.0 * 0.1,
                                (i / 9) as f64 * 0.1,
                                (i as f64 * 0.7).sin() * 0.01,
                            ],
                        };
                        p.map(|v| v * scale)
                    })
                    .collect();
                let compared: Vec<_> = points
                    .iter()
                    .map(|p| [p[0], p[1], p[2] + scale * 0.02])
                    .collect();
                let mut cores: Vec<_> = points.iter().step_by(5).copied().collect();
                cores.extend([[100.0 * scale; 3], [f64::NAN; 3], [f64::INFINITY, 0.0, 0.0]]);
                if mode == 3 {
                    for p in &mut cores {
                        p[0] += scale * 368000.0;
                    }
                }
                for bounds in [false, true] {
                    for shared in [false, true] {
                        for fused in [false, true] {
                            let params = M3c2Params {
                                normal_radius: 0.4 * scale,
                                projection_radius: 0.25 * scale,
                                max_depth: 0.5 * scale,
                                ..M3c2Params::default()
                            };
                            let m = M3c2::with_radius_bounds(&points, &compared, params, bounds)
                                .unwrap()
                                .with_fused_projection(fused)
                                .with_cached_geometry(true);
                            let (mut normal, mut projection) = (Vec::new(), Vec::new());
                            let expected: Vec<_> = cores
                                .iter()
                                .map(|c| {
                                    bits(m.at_with_search(c, &mut normal, &mut projection, shared))
                                })
                                .collect();
                            let mut scratch = M3c2PairScratch::default();
                            let (pairs, tail) = cores.as_chunks::<2>();
                            let mut actual = Vec::new();
                            for pair in pairs {
                                actual.extend(
                                    m.at_pair_with_search(pair, &mut scratch, shared).map(bits),
                                );
                            }
                            if let Some(c) = tail.first() {
                                actual.push(bits(m.at_single_with_pair_scratch(
                                    c,
                                    &mut scratch,
                                    shared,
                                )));
                            }
                            assert_eq!(
                                actual, expected,
                                "scale {scale}, mode {mode}, bounds {bounds}, shared {shared}, fused {fused}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A plane z = h sampled every 0.1 m over 20 x 20 m, with deterministic
    /// jitter of amplitude `noise`.
    fn plane(h: f64, noise: f64, seed: u64) -> Vec<[f64; 3]> {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut out = Vec::new();
        for j in 0..200 {
            for i in 0..200 {
                out.push([
                    i as f64 * 0.1 + 0.05 * next(),
                    j as f64 * 0.1 + 0.05 * next(),
                    h + noise * next(),
                ]);
            }
        }
        out
    }

    #[test]
    fn measures_a_uniform_rise_along_the_normal() {
        let before = plane(0.0, 0.02, 1);
        let after = plane(0.3, 0.02, 2);
        let core: Vec<[f64; 3]> = vec![[5.0, 5.0, 0.0], [10.0, 12.0, 0.0], [15.0, 3.0, 0.0]];
        let r = m3c2(&core, &before, &after, M3c2Params::default()).unwrap();
        for k in 0..core.len() {
            assert!((r.distance[k] - 0.3).abs() < 0.01, "{}", r.distance[k]);
            assert!(r.lod95[k] > 0.0 && r.lod95[k] < 0.02, "{}", r.lod95[k]);
            assert!(r.significant[k]);
            assert!((r.normals[k][2] - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn radius_bounds_preserve_all_result_bits_and_undefined_cores() {
        let before = plane(0.0, 0.02, 41);
        let after = plane(0.1, 0.02, 43);
        let mut core: Vec<_> = before.iter().step_by(317).copied().collect();
        core.extend([[100.0, 100.0, 100.0], [f64::NAN; 3]]);
        for (normal_radius, projection_radius, max_depth) in [(0.2, 0.15, 0.2), (1.0, 0.5, 2.0)] {
            let params = M3c2Params {
                normal_radius,
                projection_radius,
                max_depth,
                ..M3c2Params::default()
            };
            let a = m3c2_with_radius_bounds(&core, &before, &after, params, false).unwrap();
            let b = m3c2_with_radius_bounds(&core, &before, &after, params, true).unwrap();
            let bits = |r: &M3c2Result| {
                r.distance
                    .iter()
                    .chain(&r.lod95)
                    .chain(r.normals.iter().flatten())
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(&a), bits(&b));
            assert_eq!(a.significant, b.significant);
            assert!(b.distance.iter().any(|v| v.is_nan()));
            assert!(b.distance.iter().any(|v| v.is_finite()));
        }
    }

    #[test]
    fn noise_alone_is_not_significant() {
        let before = plane(0.0, 0.05, 3);
        let after = plane(0.0, 0.05, 4);
        let core: Vec<[f64; 3]> = (0..50)
            .map(|k| {
                [
                    2.0 + (k % 10) as f64 * 1.6,
                    2.0 + (k / 10) as f64 * 3.0,
                    0.0,
                ]
            })
            .collect();
        let r = m3c2(&core, &before, &after, M3c2Params::default()).unwrap();
        let significant = r.significant.iter().filter(|&&s| s).count();
        // 95 % bound: a few false positives at most.
        assert!(significant <= 5, "{significant} of 50 significant");
        assert!(r.distance.iter().all(|d| d.abs() < 0.02));
    }

    #[test]
    fn measures_along_a_tilted_surface_normal() {
        // A 45-degree slope shifted 0.2 m along +z: the normal distance is 0.2 / sqrt(2).
        let slope = |dz: f64| -> Vec<[f64; 3]> {
            (0..40_000)
                .map(|k| {
                    let (x, y) = ((k % 200) as f64 * 0.1, (k / 200) as f64 * 0.1);
                    [x, y, x + dz]
                })
                .collect()
        };
        let (before, after) = (slope(0.0), slope(0.2));
        let r = m3c2(
            &[[10.0, 10.0, 10.0]],
            &before,
            &after,
            M3c2Params::default(),
        )
        .unwrap();
        assert!(
            (r.distance[0] - 0.2 / 2f64.sqrt()).abs() < 1e-6,
            "{}",
            r.distance[0]
        );
    }

    #[test]
    fn no_data_gives_nan() {
        let before = plane(0.0, 0.0, 5);
        let r = m3c2(
            &[[100.0, 100.0, 0.0]],
            &before,
            &before,
            M3c2Params::default(),
        )
        .unwrap();
        assert!(r.distance[0].is_nan() && !r.significant[0]);
        assert!(
            m3c2(
                &[[0.0; 3]],
                &before,
                &before,
                M3c2Params {
                    normal_radius: 0.0,
                    ..M3c2Params::default()
                }
            )
            .is_none()
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_matches_serial() {
        let (before, after) = (plane(0.0, 0.02, 6), plane(0.1, 0.02, 7));
        let core: Vec<[f64; 3]> = before.iter().step_by(97).copied().collect();
        let p = M3c2Params::default();
        assert_eq!(
            m3c2_par(&core, &before, &after, p),
            m3c2(&core, &before, &after, p)
        );
    }

    #[test]
    fn fused_projection_preserves_fresh_cached_and_nonfinite_normal_bits() {
        for offset in [0.0, 368000.0] {
            let before: Vec<_> = (0..809)
                .map(|i| {
                    let x = (i % 31) as f64 * 0.1;
                    let y = (i / 31) as f64 * 0.1;
                    [
                        offset + x,
                        offset * 10.0 + y,
                        40.0 + 0.2 * x + 0.01 * (4.0 * y).sin(),
                    ]
                })
                .chain([
                    [0.0; 3],
                    [-0.0; 3],
                    [f64::MAX; 3],
                    [f64::from_bits(1); 3],
                    [f64::INFINITY; 3],
                    [f64::NAN; 3],
                ])
                .collect();
            let after: Vec<_> = before.iter().map(|p| [p[0], p[1], p[2] + 0.07]).collect();
            let cores: Vec<_> = after
                .iter()
                .step_by(29)
                .copied()
                .chain([[f64::NAN; 3], [f64::MAX; 3], [-0.0; 3]])
                .collect();
            for (normal_radius, projection_radius, max_depth) in [
                (0.1, 0.1, 0.2),
                (0.4, 0.25, 0.5),
                (2.0, 0.1, 0.2),
                (0.5_f64.hypot(2.0), 0.5, 2.0),
                (1e200, 1e200, 1e200),
                (f64::from_bits(1), f64::from_bits(1), f64::from_bits(1)),
            ] {
                for bounds in [false, true] {
                    let params = M3c2Params {
                        normal_radius,
                        projection_radius,
                        max_depth,
                        registration_error: 0.002,
                        ..M3c2Params::default()
                    };
                    let ordinary = M3c2::with_radius_bounds(&before, &after, params, bounds)
                        .unwrap()
                        .with_fused_projection(false);
                    let fused = M3c2::with_trees(&ordinary.tree1, &ordinary.tree2, params, bounds)
                        .unwrap()
                        .with_fused_projection(true);
                    let (mut scratch, mut projection) = (Vec::new(), Vec::new());
                    for core in &cores {
                        let fitted = ordinary.normal_at(core, &mut scratch);
                        for normal in [
                            fitted,
                            None,
                            Some([f64::NAN; 3]),
                            Some([f64::INFINITY; 3]),
                            Some([0.6, 0.0, 0.8]),
                            Some([-0.0, 0.0, 1.0]),
                        ] {
                            let a = ordinary.at_with_normal(core, normal, &mut scratch);
                            let b = fused.at_with_normal(core, normal, &mut scratch);
                            assert_eq!(
                                (a.0.to_bits(), a.1.to_bits(), a.2.map(f64::to_bits)),
                                (b.0.to_bits(), b.1.to_bits(), b.2.map(f64::to_bits))
                            );
                        }
                        for shared in [false, true] {
                            let a = ordinary.at_with_search(
                                core,
                                &mut scratch,
                                &mut projection,
                                shared,
                            );
                            let b =
                                fused.at_with_search(core, &mut scratch, &mut projection, shared);
                            assert_eq!(
                                (
                                    a.0.to_bits(),
                                    a.1.to_bits(),
                                    a.2.map(|n| n.map(f64::to_bits))
                                ),
                                (
                                    b.0.to_bits(),
                                    b.1.to_bits(),
                                    b.2.map(|n| n.map(f64::to_bits))
                                )
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn shared_search_preserves_normals_distances_lod_and_missing_fit_bits() {
        for offset in [0.0, 368_000.0] {
            let before: Vec<_> = (0..1201)
                .map(|i| {
                    let x = (i % 37) as f64 * 0.1;
                    let y = (i / 37) as f64 * 0.1;
                    [
                        offset + x,
                        offset * 10.0 + y,
                        40.0 + 0.2 * x + 0.01 * (y * 4.0).sin(),
                    ]
                })
                .chain([
                    [offset + 100.0, offset * 10.0 + 100.0, 100.0],
                    [f64::MAX; 3],
                    [f64::from_bits(1); 3],
                    [-0.0; 3],
                ])
                .collect();
            let after: Vec<_> = before.iter().map(|p| [p[0], p[1], p[2] + 0.07]).collect();
            let cores: Vec<_> = after
                .iter()
                .step_by(19)
                .copied()
                .chain([[f64::NAN; 3], [f64::INFINITY; 3], [-0.0; 3], [f64::MAX; 3]])
                .collect();
            for (normal_radius, projection_radius, max_depth) in [
                (0.1, 0.1, 0.2),
                (0.4, 0.25, 0.5),
                (2.0, 0.1, 0.2),
                (0.5_f64.hypot(2.0), 0.5, 2.0),
                (1e200, 1e200, 1e200),
                (f64::from_bits(1), f64::from_bits(1), f64::from_bits(1)),
            ] {
                for bounds in [false, true] {
                    let m = M3c2::with_radius_bounds(
                        &before,
                        &after,
                        M3c2Params {
                            normal_radius,
                            projection_radius,
                            max_depth,
                            ..M3c2Params::default()
                        },
                        bounds,
                    )
                    .unwrap();
                    let (mut scratch, mut projection) = (Vec::new(), Vec::new());
                    for core in &cores {
                        let expected_normal = m.normal_at(core, &mut scratch);
                        let expected = m.at_with_normal(core, expected_normal, &mut scratch);
                        let (distance, lod, normal) =
                            m.at_with_search(core, &mut scratch, &mut projection, true);
                        assert_eq!(
                            normal.map(|n| n.map(f64::to_bits)),
                            expected_normal.map(|n| n.map(f64::to_bits))
                        );
                        assert_eq!(distance.to_bits(), expected.0.to_bits());
                        assert_eq!(lod.to_bits(), expected.1.to_bits());
                    }
                }
            }
        }
    }
}

/// A changed object: a cluster of core points whose change was significant.
#[derive(Debug, Clone, PartialEq)]
pub struct ChangedObject {
    pub count: usize,
    pub centroid: [f64; 3],
    pub min: [f64; 3],
    pub max: [f64; 3],
    /// Mean M3C2 distance of its points: positive where the compared
    /// surface lies in front of the reference along the normal.
    pub mean_change: f64,
}

/// The significant changes grouped into objects: core points whose change
/// is significant and at least `min_change` in size, closer than `link`
/// and changed the same way (towards or away along the normal), are one
/// object; objects of fewer than `min_points` are dropped. Largest first,
/// with each input point's object (or [`crate::cluster::NOISE`]).
/// Without `min_change` and the split by sign, small differences of both
/// signs (leaves, a rough wall) chain into sprawling objects of no net change.
pub fn changed_objects(
    positions: &[[f64; 3]],
    change: &[f64],
    significant: &[bool],
    min_change: f64,
    link: f64,
    min_points: usize,
) -> (Vec<ChangedObject>, Vec<u32>) {
    let mut found: Vec<(ChangedObject, Vec<usize>)> = Vec::new();
    for sign in [1.0, -1.0] {
        let chosen: Vec<usize> = (0..positions.len())
            .filter(|&i| {
                significant.get(i).copied().unwrap_or(false) && sign * change[i] >= min_change
            })
            .collect();
        let points: Vec<[f64; 3]> = chosen.iter().map(|&i| positions[i]).collect();
        let clusters = crate::cluster::euclidean_clusters(&points, link, min_points.max(1));
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); clusters.sizes.len()];
        for (k, &label) in clusters.labels.iter().enumerate() {
            if label != crate::cluster::NOISE {
                members[label as usize].push(chosen[k]);
            }
        }
        for points in members {
            let n = points.len() as f64;
            let mut o = ChangedObject {
                count: points.len(),
                centroid: [0.0; 3],
                min: [f64::INFINITY; 3],
                max: [f64::NEG_INFINITY; 3],
                mean_change: 0.0,
            };
            for &i in &points {
                for (a, &v) in positions[i].iter().enumerate() {
                    o.centroid[a] += v / n;
                    o.min[a] = o.min[a].min(v);
                    o.max[a] = o.max[a].max(v);
                }
                o.mean_change += change[i] / n;
            }
            found.push((o, points));
        }
    }
    found.sort_by_key(|(o, _)| std::cmp::Reverse(o.count));
    let mut labels = vec![crate::cluster::NOISE; positions.len()];
    for (rank, (_, points)) in found.iter().enumerate() {
        for &i in points {
            labels[i] = rank as u32;
        }
    }
    (found.into_iter().map(|(o, _)| o).collect(), labels)
}

#[cfg(test)]
mod change_tests {
    use super::*;

    #[test]
    fn significant_changes_group_into_objects() {
        // A 3 x 3 patch of raised points, a lone raised point, and unchanged ground.
        let mut positions = Vec::new();
        let mut change = Vec::new();
        let mut significant = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                positions.push([i as f64 * 0.5, j as f64 * 0.5, 0.0]);
                change.push(1.5);
                significant.push(true);
            }
        }
        positions.push([20.0, 0.0, 0.0]);
        change.push(-2.0);
        significant.push(true);
        for i in 0..10 {
            positions.push([5.0 + i as f64, 5.0, 0.0]);
            change.push(0.01);
            significant.push(false);
        }
        // A lowered patch next to the raised one is a separate object.
        for i in 0..4 {
            positions.push([2.0 + i as f64 * 0.5, 0.0, 0.0]);
            change.push(-0.8);
            significant.push(true);
        }
        let (objects, labels) = changed_objects(&positions, &change, &significant, 0.3, 1.0, 3);
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[1].count, 4);
        assert!((objects[1].mean_change + 0.8).abs() < 1e-12);
        let o = &objects[0];
        assert_eq!(o.count, 9);
        assert!((o.mean_change - 1.5).abs() < 1e-12);
        assert!((o.centroid[0] - 0.5).abs() < 1e-12 && (o.centroid[1] - 0.5).abs() < 1e-12);
        assert_eq!((o.min[0], o.max[0]), (0.0, 1.0));
        assert_eq!(labels[..9], [0; 9]);
        assert!(labels[9..20].iter().all(|&l| l == crate::cluster::NOISE));
        assert!(labels[20..].iter().all(|&l| l == 1));
    }
}
