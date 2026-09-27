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

/// Fit-and-project state shared by the serial and parallel drivers.
pub struct M3c2<'a> {
    cloud1: &'a [[f64; 3]],
    cloud2: &'a [[f64; 3]],
    tree1: KdTree,
    tree2: KdTree,
    params: M3c2Params,
}

impl<'a> M3c2<'a> {
    /// Returns `None` if either cloud is empty or a radius is not positive.
    pub fn new(cloud1: &'a [[f64; 3]], cloud2: &'a [[f64; 3]], params: M3c2Params) -> Option<Self> {
        let positive = |v: f64| v > 0.0 && v.is_finite();
        if !(positive(params.normal_radius)
            && positive(params.projection_radius)
            && positive(params.max_depth))
        {
            return None;
        }
        Some(Self {
            tree1: KdTree::new(cloud1)?,
            tree2: KdTree::new(cloud2)?,
            cloud1,
            cloud2,
            params,
        })
    }

    /// `(distance, lod95, normal)` at one core point.
    pub fn at(&self, core: &[f64; 3], scratch: &mut Vec<usize>) -> (f64, f64, [f64; 3]) {
        const NAN3: [f64; 3] = [f64::NAN; 3];
        self.tree1.within(core, self.params.normal_radius, scratch);
        let Some(normal) = fit_normal(self.cloud1, scratch) else {
            return (f64::NAN, f64::NAN, NAN3);
        };
        let reach = self.params.projection_radius.hypot(self.params.max_depth);
        let mut project = |tree: &KdTree, cloud: &[[f64; 3]]| {
            tree.within(core, reach, scratch);
            let r2 = self.params.projection_radius.powi(2);
            let (mut n, mut sum, mut sum2) = (0usize, 0.0, 0.0);
            for &i in scratch.iter() {
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
        };
        let (Some((m1, v1, n1)), Some((m2, v2, n2))) = (
            project(&self.tree1, self.cloud1),
            project(&self.tree2, self.cloud2),
        ) else {
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
    let m = M3c2::new(cloud1, cloud2, params)?;
    let mut scratch = Vec::new();
    Some(collect(core.iter().map(|c| m.at(c, &mut scratch))))
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
            let mut scratch = Vec::new();
            chunk
                .iter()
                .map(|c| m.at(c, &mut scratch))
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
        for r in 0..3 {
            for k in 0..3 {
                cov[r][k] += d[r] * d[k];
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
