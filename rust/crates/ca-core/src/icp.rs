//! Rigid ICP registration (point-to-plane or point-to-point).
//!
//! Each iteration pairs a subsample of the moving cloud with its nearest
//! reference points and optionally keeps only the closest fraction of pairs
//! ("overlap"). Point-to-plane minimises distances along reference normals
//! (estimated by PCA of nearby reference points, on demand) with a
//! linearised 6x6 solve; point-to-point uses Horn's closed-form quaternion
//! solution. All arithmetic happens relative to the reference centroid so
//! georeferenced coordinates keep full precision.

use crate::PointCloud;
use crate::kdtree::KdTree;

/// A rigid transform `x' = R x + t`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rigid {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
}

impl Rigid {
    pub const IDENTITY: Self = Self {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0; 3],
    };

    pub fn apply(&self, p: &[f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        std::array::from_fn(|i| {
            r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2] + self.translation[i]
        })
    }

    /// `self` after `first`: `x -> self(first(x))`.
    pub fn compose(&self, first: &Rigid) -> Rigid {
        let a = &self.rotation;
        let b = &first.rotation;
        let rotation = std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum())
        });
        Rigid {
            rotation,
            translation: self.apply(&first.translation),
        }
    }

    /// Row-major 4x4 homogeneous matrix.
    pub fn to_matrix(&self) -> [f64; 16] {
        let r = &self.rotation;
        let t = &self.translation;
        [
            r[0][0], r[0][1], r[0][2], t[0], //
            r[1][0], r[1][1], r[1][2], t[1], //
            r[2][0], r[2][1], r[2][2], t[2], //
            0.0, 0.0, 0.0, 1.0,
        ]
    }

    /// Read a row-major 4x4 matrix; the bottom row is ignored.
    pub fn from_matrix(m: &[f64; 16]) -> Rigid {
        Rigid {
            rotation: [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]],
            translation: [m[3], m[7], m[11]],
        }
    }

    /// Express a transform that acts on coordinates relative to `origin`
    /// (`x - origin`) as one acting on absolute coordinates.
    fn around(&self, origin: &[f64; 3]) -> Rigid {
        // x' = R (x - o) + t + o  =>  translation = t + o - R o
        let ro = Rigid {
            rotation: self.rotation,
            translation: [0.0; 3],
        }
        .apply(origin);
        Rigid {
            rotation: self.rotation,
            translation: std::array::from_fn(|i| self.translation[i] + origin[i] - ro[i]),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IcpMetric {
    /// Distances along reference normals; converges in far fewer iterations
    /// on surfaces sampled differently in the two clouds.
    PointToPlane,
    PointToPoint,
}

#[derive(Debug, Clone, Copy)]
pub struct IcpParams {
    pub metric: IcpMetric,
    pub max_iterations: usize,
    /// Stop when the RMS improves by less than this fraction.
    pub tolerance: f64,
    /// At most this many moving points are used per iteration.
    pub sample: usize,
    /// Fraction of closest pairs kept each iteration (1.0 keeps all).
    pub overlap: f64,
    /// Translate the moving centroid onto the reference centroid first.
    pub match_centroids: bool,
}

impl Default for IcpParams {
    fn default() -> Self {
        Self {
            metric: IcpMetric::PointToPlane,
            max_iterations: 50,
            tolerance: 1e-6,
            sample: 50_000,
            overlap: 1.0,
            match_centroids: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IcpResult {
    /// Maps the moving cloud onto the reference.
    pub transform: Rigid,
    /// RMS of the kept pair distances before the first and after the last step.
    pub rms_initial: f64,
    pub rms_final: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Register `moving` onto `reference`. Returns `None` if either cloud is
/// empty or `overlap` leaves fewer than three pairs.
pub fn icp(moving: &PointCloud, reference: &PointCloud, params: IcpParams) -> Option<IcpResult> {
    if moving.is_empty() || reference.is_empty() {
        return None;
    }
    let origin = centroid(&reference.positions);
    let local = |p: &[f64; 3]| -> [f64; 3] { std::array::from_fn(|i| p[i] - origin[i]) };
    let reference_local: Vec<[f64; 3]> = reference.positions.iter().map(local).collect();
    let tree = KdTree::new(&reference_local)?;
    let stride = moving.len().div_ceil(params.sample.max(3));
    let samples: Vec<[f64; 3]> = moving.positions.iter().step_by(stride).map(local).collect();
    let keep = ((samples.len() as f64 * params.overlap.clamp(0.0, 1.0)).round() as usize)
        .min(samples.len());
    if keep < 3 {
        return None;
    }

    let mut current = Rigid::IDENTITY;
    if params.match_centroids {
        let (mc, rc) = (centroid(&samples), centroid(&reference_local));
        current.translation = std::array::from_fn(|i| rc[i] - mc[i]);
    }

    let mut normals = Normals::new(&reference_local, &tree);
    let mut pairs = Vec::with_capacity(samples.len());
    let mut rms_initial = f64::NAN;
    let mut rms_previous = f64::INFINITY;
    let mut iterations = 0;
    let mut converged = false;
    let rms = loop {
        let rms = match_pairs(
            &samples,
            &current,
            &tree,
            &reference_local,
            keep,
            &mut pairs,
        );
        if iterations == 0 {
            rms_initial = rms;
        }
        let stalled =
            rms_previous.is_finite() && rms_previous - rms <= params.tolerance * rms_previous;
        if rms == 0.0 || stalled {
            converged = true;
            break rms;
        }
        if iterations == params.max_iterations {
            break rms;
        }
        rms_previous = rms;
        let step = match params.metric {
            IcpMetric::PointToPoint => horn(&pairs),
            IcpMetric::PointToPlane => {
                point_to_plane(&pairs, &mut normals).unwrap_or_else(|| horn(&pairs))
            }
        };
        current = step.compose(&current);
        iterations += 1;
    };
    Some(IcpResult {
        transform: current.around(&origin),
        rms_initial,
        rms_final: rms,
        iterations,
        converged,
    })
}

/// A moving point matched to a reference point.
struct Pair {
    distance_sq: f64,
    moving: [f64; 3],
    reference: [f64; 3],
    reference_index: usize,
}

/// Pair every transformed sample with its nearest reference point, keep the
/// `keep` closest pairs, and return their RMS distance.
fn match_pairs(
    samples: &[[f64; 3]],
    transform: &Rigid,
    tree: &KdTree,
    reference: &[[f64; 3]],
    keep: usize,
    pairs: &mut Vec<Pair>,
) -> f64 {
    pairs.clear();
    let mut guess = None;
    for s in samples {
        let p = transform.apply(s);
        let hit = tree.nearest(&p, guess);
        guess = Some(hit);
        pairs.push(Pair {
            distance_sq: hit.distance_sq,
            moving: p,
            reference: reference[hit.index],
            reference_index: hit.index,
        });
    }
    if keep < pairs.len() {
        pairs.select_nth_unstable_by(keep - 1, |a, b| a.distance_sq.total_cmp(&b.distance_sq));
        pairs.truncate(keep);
    }
    (pairs.iter().map(|p| p.distance_sq).sum::<f64>() / pairs.len() as f64).sqrt()
}

/// Reference normals, estimated lazily by PCA over nearby points.
struct Normals<'a> {
    points: &'a [[f64; 3]],
    tree: &'a KdTree,
    /// `None` = not computed yet; `Some(None)` = degenerate neighbourhood.
    cache: Vec<Option<Option<[f64; 3]>>>,
}

impl<'a> Normals<'a> {
    const NEIGHBOURS: usize = 12;

    fn new(points: &'a [[f64; 3]], tree: &'a KdTree) -> Self {
        Self {
            points,
            tree,
            cache: vec![None; points.len()],
        }
    }

    fn get(&mut self, index: usize) -> Option<[f64; 3]> {
        if let Some(normal) = self.cache[index] {
            return normal;
        }
        let neighbours = self.tree.nearest_k(&self.points[index], Self::NEIGHBOURS);
        let normal = if neighbours.len() < 3 {
            None
        } else {
            let pts: Vec<[f64; 3]> = neighbours.iter().map(|&(i, _)| self.points[i]).collect();
            let c = centroid(&pts);
            let mut cov = [[0.0; 3]; 3];
            for p in &pts {
                for i in 0..3 {
                    for j in 0..3 {
                        cov[i][j] += (p[i] - c[i]) * (p[j] - c[j]);
                    }
                }
            }
            let (values, vectors) = jacobi(cov);
            let order = sorted_indices(&values);
            // A plane needs two clearly non-zero spreads.
            if values[order[1]] <= 1e-12 * values[order[2]].max(f64::MIN_POSITIVE) {
                None
            } else {
                let k = order[0];
                Some([vectors[0][k], vectors[1][k], vectors[2][k]])
            }
        };
        self.cache[index] = Some(normal);
        normal
    }
}

fn sorted_indices(values: &[f64; 3]) -> [usize; 3] {
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    order
}

/// Linearised point-to-plane step: minimise `sum(((R p + t - q) . n)^2)`
/// for small rotations. `None` when too few pairs have normals.
fn point_to_plane(pairs: &[Pair], normals: &mut Normals) -> Option<Rigid> {
    let mut ata = [[0.0; 6]; 6];
    let mut atb = [0.0; 6];
    let mut used = 0;
    for pair in pairs {
        let Some(n) = normals.get(pair.reference_index) else {
            continue;
        };
        let p = pair.moving;
        let c = [
            p[1] * n[2] - p[2] * n[1],
            p[2] * n[0] - p[0] * n[2],
            p[0] * n[1] - p[1] * n[0],
        ];
        let row = [c[0], c[1], c[2], n[0], n[1], n[2]];
        let residual: f64 = (0..3).map(|i| (pair.reference[i] - p[i]) * n[i]).sum();
        for i in 0..6 {
            atb[i] += row[i] * residual;
            for j in 0..6 {
                ata[i][j] += row[i] * row[j];
            }
        }
        used += 1;
    }
    if used < 6 {
        return None;
    }
    // A touch of damping keeps directions the scene cannot constrain (e.g.
    // sliding along a flat floor) near zero instead of blowing up.
    let trace: f64 = (0..6).map(|i| ata[i][i]).sum();
    for (i, row) in ata.iter_mut().enumerate() {
        row[i] += 1e-9 * trace / 6.0 + f64::MIN_POSITIVE;
    }
    let x = solve6(ata, atb)?;
    let (sa, ca) = x[0].sin_cos();
    let (sb, cb) = x[1].sin_cos();
    let (sg, cg) = x[2].sin_cos();
    // R = Rz(g) Ry(b) Rx(a)
    let rotation = [
        [cg * cb, cg * sb * sa - sg * ca, cg * sb * ca + sg * sa],
        [sg * cb, sg * sb * sa + cg * ca, sg * sb * ca - cg * sa],
        [-sb, cb * sa, cb * ca],
    ];
    Some(Rigid {
        rotation,
        translation: [x[3], x[4], x[5]],
    })
}

/// Solve a 6x6 linear system by Gaussian elimination with partial pivoting.
fn solve6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
    for col in 0..6 {
        let pivot = (col..6).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let pivot_row = a[col];
        for row in col + 1..6 {
            let f = a[row][col] / pivot_row[col];
            for (value, &p) in a[row][col..].iter_mut().zip(&pivot_row[col..]) {
                *value -= f * p;
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; 6];
    for row in (0..6).rev() {
        let s: f64 = (row + 1..6).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

fn centroid(points: &[[f64; 3]]) -> [f64; 3] {
    let mut sum = [0.0; 3];
    for p in points {
        for a in 0..3 {
            sum[a] += p[a];
        }
    }
    sum.map(|s| s / points.len() as f64)
}

/// Least-squares rigid motion taking each moving point onto its reference
/// point (Horn 1987).
fn horn(pairs: &[Pair]) -> Rigid {
    rigid_fit(pairs.iter().map(|pair| (pair.moving, pair.reference)))
}

/// Least-squares rigid motion taking each `p` onto its `q` in `(p, q)`
/// pairs, always a proper rotation (Horn's closed form, as [`horn`]).
pub(crate) fn rigid_fit<I>(pairs: I) -> Rigid
where
    I: Iterator<Item = ([f64; 3], [f64; 3])> + Clone,
{
    let n = pairs.clone().count() as f64;
    let mut mp = [0.0; 3];
    let mut mq = [0.0; 3];
    for (p, q) in pairs.clone() {
        for a in 0..3 {
            mp[a] += p[a] / n;
            mq[a] += q[a] / n;
        }
    }
    // Cross-covariance S[i][j] = sum (p_i - mp_i)(q_j - mq_j).
    let mut s = [[0.0; 3]; 3];
    for (p, q) in pairs {
        for i in 0..3 {
            for j in 0..3 {
                s[i][j] += (p[i] - mp[i]) * (q[j] - mq[j]);
            }
        }
    }
    let [[sxx, sxy, sxz], [syx, syy, syz], [szx, szy, szz]] = s;
    let m = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let (values, vectors) = jacobi(m);
    let best = (0..4)
        .max_by(|&i, &j| values[i].total_cmp(&values[j]))
        .unwrap();
    let q = [
        vectors[0][best],
        vectors[1][best],
        vectors[2][best],
        vectors[3][best],
    ];
    let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let [w, x, y, z] = q.map(|v| v / norm);
    let rotation = [
        [
            w * w + x * x - y * y - z * z,
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            w * w - x * x + y * y - z * z,
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            w * w - x * x - y * y + z * z,
        ],
    ];
    let r_mp = Rigid {
        rotation,
        translation: [0.0; 3],
    }
    .apply(&mp);
    Rigid {
        rotation,
        translation: std::array::from_fn(|i| mq[i] - r_mp[i]),
    }
}

/// Eigen-decomposition of a symmetric 3x3 matrix (see [`jacobi`]).
pub(crate) fn symmetric_eigen(m: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    jacobi(m)
}

/// Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations:
/// eigenvalues and the matrix whose columns are the eigenvectors.
fn jacobi<const N: usize>(mut a: [[f64; N]; N]) -> ([f64; N], [[f64; N]; N]) {
    let mut v = [[0.0; N]; N];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..50 {
        let off: f64 = (0..N)
            .flat_map(|i| (0..N).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        let scale: f64 = (0..N).map(|i| a[i][i] * a[i][i]).sum();
        if off <= 1e-30 * scale.max(f64::MIN_POSITIVE) {
            break;
        }
        for p in 0..N - 1 {
            for q in p + 1..N {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                let (row_p, row_q) = (a[p], a[q]);
                for k in 0..N {
                    a[p][k] = c * row_p[k] - s * row_q[k];
                    a[q][k] = s * row_p[k] + c * row_q[k];
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    (std::array::from_fn(|i| a[i][i]), v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rotation(rx: f64, ry: f64, rz: f64) -> [[f64; 3]; 3] {
        let (sx, cx) = rx.sin_cos();
        let (sy, cy) = ry.sin_cos();
        let (sz, cz) = rz.sin_cos();
        let x = Rigid {
            rotation: [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
            translation: [0.0; 3],
        };
        let y = Rigid {
            rotation: [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
            translation: [0.0; 3],
        };
        let z = Rigid {
            rotation: [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0; 3],
        };
        z.compose(&y.compose(&x)).rotation
    }

    /// A bumpy 40 m x 30 m surface, offset like UTM data.
    fn surface(n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let (x, y) = (next() * 40.0, next() * 30.0);
                let z = 2.0 * (x / 5.0).sin() * (y / 4.0).cos() + 0.05 * x;
                [x + 368_000.0, y + 3_955_000.0, z + 40.0]
            })
            .collect()
    }

    fn cloud(positions: Vec<[f64; 3]>) -> PointCloud {
        PointCloud {
            positions,
            colors: None,
            attributes: Vec::new(),
        }
    }

    /// Largest disagreement between two transforms over `points`. Comparing
    /// raw translations would be meaningless for georeferenced data, where a
    /// micro-radian rotation about the far-away origin moves metres.
    fn displacement_error(a: &Rigid, b: &Rigid, points: &[[f64; 3]]) -> f64 {
        points
            .iter()
            .map(|p| {
                let (x, y) = (a.apply(p), b.apply(p));
                (0..3).map(|i| (x[i] - y[i]).powi(2)).sum::<f64>().sqrt()
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn horn_recovers_an_exact_motion() {
        let truth = Rigid {
            rotation: rotation(0.3, -0.2, 1.1),
            translation: [1.0, -2.0, 0.5],
        };
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ];
        let pairs: Vec<_> = points
            .iter()
            .map(|p| Pair {
                distance_sq: 0.0,
                moving: *p,
                reference: truth.apply(p),
                reference_index: 0,
            })
            .collect();
        let error = displacement_error(&horn(&pairs), &truth, &points);
        assert!(error < 1e-12, "{error}");
    }

    #[test]
    fn matrix_roundtrip_and_composition() {
        let a = Rigid {
            rotation: rotation(0.1, 0.2, 0.3),
            translation: [1.0, 2.0, 3.0],
        };
        assert_eq!(Rigid::from_matrix(&a.to_matrix()), a);
        let p = [4.0, -5.0, 6.0];
        let b = Rigid {
            rotation: rotation(-0.4, 0.0, 0.9),
            translation: [0.0, 1.0, 0.0],
        };
        let composed = b.compose(&a).apply(&p);
        let sequential = b.apply(&a.apply(&p));
        for i in 0..3 {
            assert!((composed[i] - sequential[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn registers_a_displaced_scan() {
        let reference = cloud(surface(40_000, 1));
        // A different sampling of the same surface, displaced about its middle.
        let center = [368_020.0, 3_955_015.0, 40.0];
        let truth = Rigid {
            rotation: rotation(0.02, -0.015, 0.05),
            translation: [0.4, -0.3, 0.15],
        }
        .around(&center);
        let inverse_truth = invert(&truth);
        let moving = cloud(
            surface(20_000, 2)
                .iter()
                .map(|p| inverse_truth.apply(p))
                .collect(),
        );

        for metric in [IcpMetric::PointToPlane, IcpMetric::PointToPoint] {
            let params = IcpParams {
                metric,
                max_iterations: 200,
                ..IcpParams::default()
            };
            let result = icp(&moving, &reference, params).unwrap();
            let error = displacement_error(&result.transform, &truth, &moving.positions);
            assert!(
                result.rms_final < result.rms_initial,
                "{metric:?} {result:?}"
            );
            // Point-to-plane is exact up to sampling; point-to-point slides
            // slowly along the surface, so it only gets close.
            let limit = if metric == IcpMetric::PointToPlane {
                0.01
            } else {
                0.1
            };
            assert!(error < limit, "{metric:?} error={error} {result:?}");
            if metric == IcpMetric::PointToPlane {
                assert!(result.converged && result.iterations < 30, "{result:?}");
            }
        }
    }

    #[test]
    fn overlap_trimming_ignores_extra_points() {
        let reference = cloud(surface(30_000, 3));
        let shift = Rigid {
            rotation: rotation(0.0, 0.0, 0.0),
            translation: [0.3, 0.2, -0.1],
        };
        let mut moving: Vec<[f64; 3]> = surface(15_000, 4).iter().map(|p| shift.apply(p)).collect();
        // 20 % outliers floating well above the surface.
        let outliers: Vec<[f64; 3]> = moving
            .iter()
            .step_by(4)
            .map(|p| [p[0], p[1], p[2] + 8.0])
            .collect();
        moving.extend(outliers);
        let params = IcpParams {
            overlap: 0.75,
            ..IcpParams::default()
        };
        let moving = cloud(moving);
        let result = icp(&moving, &reference, params).unwrap();
        let truth = Rigid {
            rotation: rotation(0.0, 0.0, 0.0),
            translation: [-0.3, -0.2, 0.1],
        };
        let error = displacement_error(&result.transform, &truth, &moving.positions[..15_000]);
        assert!(error < 0.02, "error={error} {result:?}");
    }

    #[test]
    fn match_centroids_handles_large_offsets() {
        let reference = cloud(surface(20_000, 5));
        let moving = cloud(
            surface(20_000, 6)
                .iter()
                .map(|p| [p[0] + 500.0, p[1] - 200.0, p[2] + 30.0])
                .collect(),
        );
        let params = IcpParams {
            match_centroids: true,
            ..IcpParams::default()
        };
        let result = icp(&moving, &reference, params).unwrap();
        let truth = Rigid {
            rotation: rotation(0.0, 0.0, 0.0),
            translation: [-500.0, 200.0, -30.0],
        };
        let error = displacement_error(&result.transform, &truth, &moving.positions);
        assert!(error < 0.02, "error={error} {result:?}");
    }

    fn invert(t: &Rigid) -> Rigid {
        let r = t.rotation;
        let rotation = std::array::from_fn(|i| std::array::from_fn(|j| r[j][i]));
        let inv = Rigid {
            rotation,
            translation: [0.0; 3],
        };
        let rt = inv.apply(&t.translation);
        Rigid {
            rotation,
            translation: rt.map(|v| -v),
        }
    }
}
