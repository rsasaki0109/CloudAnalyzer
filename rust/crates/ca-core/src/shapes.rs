//! RANSAC primitive detection: planes, spheres or cylinders are extracted
//! one after another, each the best of many candidates built from minimal
//! samples with their normals, then refitted by least squares. In the
//! spirit of Schnabel et al. (2007) and CloudCompare's RANSAC Shape
//! Detection, without their octree sampling and connectivity bitmaps.

use crate::kdtree::KdTree;

/// The kind of shape to look for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primitive {
    Plane,
    Sphere,
    Cylinder,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape {
    /// Points with `normal · p + d = 0`; `normal` is a unit vector.
    Plane {
        normal: [f64; 3],
        d: f64,
    },
    Sphere {
        center: [f64; 3],
        radius: f64,
    },
    /// An infinite cylinder around the line through `point` along the unit
    /// vector `axis`; `point` is the middle of the inliers along the axis
    /// and `length` their extent.
    Cylinder {
        point: [f64; 3],
        axis: [f64; 3],
        radius: f64,
        length: f64,
    },
}

impl Shape {
    /// Distance from `p` to the surface.
    pub fn distance(&self, p: &[f64; 3]) -> f64 {
        match *self {
            Shape::Plane { normal, d } => (dot(normal, *p) + d).abs(),
            Shape::Sphere { center, radius } => (norm(sub(*p, center)) - radius).abs(),
            Shape::Cylinder {
                point,
                axis,
                radius,
                ..
            } => (norm(radial(sub(*p, point), axis)) - radius).abs(),
        }
    }

    /// Radius of a sphere or cylinder (infinite for a plane).
    fn radius(&self) -> f64 {
        match *self {
            Shape::Plane { .. } => f64::INFINITY,
            Shape::Sphere { radius, .. } | Shape::Cylinder { radius, .. } => radius,
        }
    }

    /// Surface normal at (the projection of) `p`, up to sign; zero on the
    /// axis or centre.
    fn normal_at(&self, p: &[f64; 3]) -> [f64; 3] {
        match *self {
            Shape::Plane { normal, .. } => normal,
            Shape::Sphere { center, .. } => unit(sub(*p, center)),
            Shape::Cylinder { point, axis, .. } => unit(radial(sub(*p, point), axis)),
        }
    }
}

/// A detected shape with the indices of its points.
#[derive(Debug, Clone, PartialEq)]
pub struct Detected {
    pub shape: Shape,
    pub indices: Vec<usize>,
    /// Root mean square distance of the points to the shape.
    pub rms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RansacParams {
    pub primitive: Primitive,
    /// Largest distance from a point to its shape.
    pub threshold: f64,
    /// Fewest points a shape must have.
    pub min_support: usize,
    pub max_shapes: usize,
    /// Largest angle (degrees) between a point's normal and the shape's.
    pub max_normal_deviation: f64,
    /// Candidates tried for each shape.
    pub candidates: usize,
    pub seed: u64,
}

impl Default for RansacParams {
    fn default() -> Self {
        Self {
            primitive: Primitive::Plane,
            threshold: 0.01,
            min_support: 500,
            max_shapes: 10,
            max_normal_deviation: 25.0,
            candidates: 400,
            seed: 0x5eed,
        }
    }
}

/// Points candidates are scored on (a random subset of the rest).
const SAMPLE: usize = 20_000;
/// Neighbourhood sizes (within the scoring sample) the other points of a
/// minimal sample are drawn from: local ones find small shapes, large
/// ones make well-conditioned candidates for big shapes; 0 is the whole
/// sample.
const SCALES: [usize; 4] = [16, 64, 256, 0];
/// Least [`normal_spread`] of a sphere or cylinder (about a 20° arc).
const MIN_SPREAD: f64 = 0.01;
/// Rounds in a row that may find no shape before giving up.
const MAX_FAILURES: usize = 3;

/// Extract up to `params.max_shapes` shapes, best first. `normals` has one
/// entry per point (a zero normal is unknown and only its distance is
/// checked); spheres and cylinders are sampled from points with normals.
/// Deterministic for a given `params.seed`.
pub fn detect_shapes(
    points: &[[f64; 3]],
    normals: &[[f32; 3]],
    params: &RansacParams,
) -> Vec<Detected> {
    let n = points.len();
    let normal = |i: usize| -> [f64; 3] { normals.get(i).map_or([0.0; 3], |v| v.map(f64::from)) };
    // Candidates larger than the cloud are rejected.
    let scale = {
        let (lo, hi) = crate::distance::bounds(points.iter());
        norm(sub(hi, lo))
    };
    let cos_max = params.max_normal_deviation.to_radians().cos();
    let fits = |shape: &Shape, i: usize| -> bool {
        let p = &points[i];
        if shape.distance(p) > params.threshold {
            return false;
        }
        let ni = normal(i);
        ni == [0.0; 3] || dot(ni, shape.normal_at(p)).abs() >= cos_max
    };
    let mut rng = Rng(params.seed | 1);
    let mut remaining: Vec<usize> = (0..n).collect();
    let mut out = Vec::new();
    let mut failures = 0;
    let min_support = params.min_support.max(3);
    let curved = params.primitive != Primitive::Plane;
    while out.len() < params.max_shapes && remaining.len() >= min_support {
        // Candidates are scored on a sample of the remaining points.
        let sample = pick(&remaining, SAMPLE, &mut rng);
        let sample_points: Vec<[f64; 3]> = sample.iter().map(|&i| points[i]).collect();
        let tree = KdTree::new(&sample_points).expect("sample is not empty");
        let mut best: Option<(Shape, usize)> = None;
        for c in 0..params.candidates {
            let k = SCALES[c % SCALES.len()];
            let a = sample[rng.below(sample.len())];
            let near = if k == 0 {
                Vec::new()
            } else {
                tree.nearest_k(&points[a], k)
            };
            let mut other = || {
                if near.is_empty() {
                    sample[rng.below(sample.len())]
                } else {
                    sample[near[rng.below(near.len())].0]
                }
            };
            let shape = match params.primitive {
                Primitive::Plane => {
                    let (b, c) = (other(), other());
                    plane_from([points[a], points[b], points[c]])
                        .filter(|s| [a, b, c].iter().all(|&i| fits(s, i)))
                }
                Primitive::Sphere | Primitive::Cylinder => {
                    let b = other();
                    let (na, nb) = (normal(a), normal(b));
                    if na == [0.0; 3] || nb == [0.0; 3] {
                        continue;
                    }
                    let from = if params.primitive == Primitive::Sphere {
                        sphere_from
                    } else {
                        cylinder_from
                    };
                    from([points[a], points[b]], [na, nb], scale)
                        .filter(|s| fits(s, a) && fits(s, b))
                }
            };
            let Some(shape) = shape else { continue };
            let hits: Vec<usize> = sample
                .iter()
                .copied()
                .filter(|&i| fits(&shape, i))
                .collect();
            if curved && normal_spread(&hits, &normal) < MIN_SPREAD {
                continue;
            }
            let score = hits.len();
            if best.is_none_or(|(_, s)| score > s) {
                best = Some((shape, score));
            }
        }
        let Some((mut shape, _)) = best else {
            break;
        };
        // Refit to all the points of the best candidate, twice (the refit
        // gathers points the rough candidate missed).
        let mut inliers: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(|&i| fits(&shape, i))
            .collect();
        for _ in 0..2 {
            let Some(refit) = refit(&shape, points, &inliers, &normal) else {
                break;
            };
            let more: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|&i| fits(&refit, i))
                .collect();
            if more.len() < inliers.len() {
                break;
            }
            (shape, inliers) = (refit, more);
        }
        let bogus =
            curved && (normal_spread(&inliers, &normal) < MIN_SPREAD || shape.radius() >= scale);
        if inliers.len() < min_support || bogus {
            failures += 1;
            if failures >= MAX_FAILURES {
                break;
            }
            continue;
        }
        failures = 0;
        if let Shape::Cylinder {
            point,
            axis,
            radius,
            ..
        } = shape
        {
            shape = cylinder_extent(points, &inliers, point, axis, radius);
        }
        let rms = (inliers
            .iter()
            .map(|&i| shape.distance(&points[i]).powi(2))
            .sum::<f64>()
            / inliers.len() as f64)
            .sqrt();
        let mut taken = vec![false; n];
        for &i in &inliers {
            taken[i] = true;
        }
        remaining.retain(|&i| !taken[i]);
        out.push(Detected {
            shape,
            indices: inliers,
            rms,
        });
    }
    out
}

/// How much the normals of `indices` turn: the middle eigenvalue of their
/// mean outer product, 0 when all are parallel. A flat patch fits huge
/// spheres and cylinders tangent to it, but its normals do not turn.
fn normal_spread(indices: &[usize], normal: &impl Fn(usize) -> [f64; 3]) -> f64 {
    let mut m = [[0.0; 3]; 3];
    let mut count = 0;
    for &i in indices {
        let v = normal(i);
        if v != [0.0; 3] {
            outer_add(&mut m, v);
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    let (mut values, _) = crate::icp::symmetric_eigen(m);
    values.sort_by(f64::total_cmp);
    values[1] / count as f64
}

/// Up to `count` entries of `from`, drawn without replacement.
fn pick(from: &[usize], count: usize, rng: &mut Rng) -> Vec<usize> {
    if from.len() <= count {
        return from.to_vec();
    }
    let mut v = from.to_vec();
    for i in 0..count {
        let j = i + rng.below(v.len() - i);
        v.swap(i, j);
    }
    v.truncate(count);
    v
}

fn plane_from(p: [[f64; 3]; 3]) -> Option<Shape> {
    let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
    let len = norm(n);
    // Reject (nearly) collinear samples.
    if len <= 1e-12 * norm(sub(p[1], p[0])).powi(2).max(f64::MIN_POSITIVE) {
        return None;
    }
    let normal = canonical(n.map(|v| v / len));
    Some(Shape::Plane {
        normal,
        d: -dot(normal, p[0]),
    })
}

/// The sphere through two points whose normals meet at its centre.
fn sphere_from(p: [[f64; 3]; 2], n: [[f64; 3]; 2], scale: f64) -> Option<Shape> {
    let (t, s) = closest_on_lines(p[0], n[0], p[1], n[1])?;
    let (c0, c1) = (add(p[0], scale_by(n[0], t)), add(p[1], scale_by(n[1], s)));
    let center = scale_by(add(c0, c1), 0.5);
    let (r0, r1) = (norm(sub(p[0], center)), norm(sub(p[1], center)));
    let radius = 0.5 * (r0 + r1);
    // The normal lines must (nearly) meet, at a sensible distance.
    if norm(sub(c0, c1)) > 0.1 * radius || !(radius > 0.0 && radius < scale) {
        return None;
    }
    Some(Shape::Sphere { center, radius })
}

/// The cylinder through two points: its axis is perpendicular to both
/// normals, and the normals meet it.
fn cylinder_from(p: [[f64; 3]; 2], n: [[f64; 3]; 2], scale: f64) -> Option<Shape> {
    let a = cross(n[0], n[1]);
    // Near-parallel normals (e.g. both on one side, or on a plane) give no axis.
    if norm(a) < 0.05 {
        return None;
    }
    let axis = canonical(unit(a));
    // Solve in the plane through p0 perpendicular to the axis.
    let q1 = sub(p[1], scale_by(axis, dot(sub(p[1], p[0]), axis)));
    let (m0, m1) = (unit(radial(n[0], axis)), unit(radial(n[1], axis)));
    let (t, _) = closest_on_lines(p[0], m0, q1, m1)?;
    let point = add(p[0], scale_by(m0, t));
    let (r0, r1) = (norm(sub(p[0], point)), norm(sub(q1, point)));
    let radius = 0.5 * (r0 + r1);
    if (r0 - r1).abs() > 0.1 * radius || !(radius > 0.0 && radius < scale) {
        return None;
    }
    Some(Shape::Cylinder {
        point,
        axis,
        radius,
        length: 0.0,
    })
}

/// Parameters `(t, s)` of the closest points of the lines `p + t u` and
/// `q + s v`, or `None` when they are parallel.
fn closest_on_lines(p: [f64; 3], u: [f64; 3], q: [f64; 3], v: [f64; 3]) -> Option<(f64, f64)> {
    let w = sub(p, q);
    let (a, b, c) = (dot(u, u), dot(u, v), dot(v, v));
    let (d, e) = (dot(u, w), dot(v, w));
    let denom = a * c - b * b;
    if denom <= 1e-9 * a * c {
        return None;
    }
    Some(((b * e - c * d) / denom, (a * e - b * d) / denom))
}

/// Least-squares fit of the same kind of shape to `indices`.
fn refit(
    shape: &Shape,
    points: &[[f64; 3]],
    indices: &[usize],
    normal: &impl Fn(usize) -> [f64; 3],
) -> Option<Shape> {
    if indices.len() < 4 {
        return None;
    }
    let m = indices.len() as f64;
    let mut c = [0.0; 3];
    for &i in indices {
        c = add(c, scale_by(points[i], 1.0 / m));
    }
    match *shape {
        Shape::Plane { .. } => {
            // PCA: the direction of least spread.
            let mut cov = [[0.0; 3]; 3];
            for &i in indices {
                outer_add(&mut cov, sub(points[i], c));
            }
            let normal = canonical(smallest_eigenvector(cov));
            Some(Shape::Plane {
                normal,
                d: -dot(normal, c),
            })
        }
        Shape::Sphere { .. } => {
            // Algebraic fit |x|² + D·x + E = 0 around the centroid.
            let mut ata = [[0.0; 4]; 4];
            let mut atb = [0.0; 4];
            for &i in indices {
                let x = sub(points[i], c);
                let row = [x[0], x[1], x[2], 1.0];
                let rhs = -dot(x, x);
                for r in 0..4 {
                    atb[r] += row[r] * rhs;
                    for s in 0..4 {
                        ata[r][s] += row[r] * row[s];
                    }
                }
            }
            let sol = solve(ata, atb)?;
            let center = [-0.5 * sol[0], -0.5 * sol[1], -0.5 * sol[2]];
            let r2 = dot(center, center) - sol[3];
            (r2 > 0.0).then(|| Shape::Sphere {
                center: add(center, c),
                radius: r2.sqrt(),
            })
        }
        Shape::Cylinder { axis, .. } => {
            // The axis is the direction the normals are perpendicular to.
            let mut nn = [[0.0; 3]; 3];
            let mut with_normals = 0;
            for &i in indices {
                let v = normal(i);
                if v != [0.0; 3] {
                    outer_add(&mut nn, v);
                    with_normals += 1;
                }
            }
            let axis = if with_normals >= 3 {
                canonical(smallest_eigenvector(nn))
            } else {
                axis
            };
            // Then a circle (algebraic fit) across the axis.
            let (u, v) = basis(axis);
            let mut ata = [[0.0; 3]; 3];
            let mut atb = [0.0; 3];
            for &i in indices {
                let x = sub(points[i], c);
                let (pu, pv) = (dot(x, u), dot(x, v));
                let row = [pu, pv, 1.0];
                let rhs = -(pu * pu + pv * pv);
                for r in 0..3 {
                    atb[r] += row[r] * rhs;
                    for s in 0..3 {
                        ata[r][s] += row[r] * row[s];
                    }
                }
            }
            let sol = solve(ata, atb)?;
            let (cu, cv) = (-0.5 * sol[0], -0.5 * sol[1]);
            let r2 = cu * cu + cv * cv - sol[2];
            (r2 > 0.0).then(|| Shape::Cylinder {
                point: add(c, add(scale_by(u, cu), scale_by(v, cv))),
                axis,
                radius: r2.sqrt(),
                length: 0.0,
            })
        }
    }
}

/// The cylinder with `point` moved to the middle of the inliers along the
/// axis and their extent as `length`.
fn cylinder_extent(
    points: &[[f64; 3]],
    indices: &[usize],
    point: [f64; 3],
    axis: [f64; 3],
    radius: f64,
) -> Shape {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &i in indices {
        let t = dot(sub(points[i], point), axis);
        lo = lo.min(t);
        hi = hi.max(t);
    }
    Shape::Cylinder {
        point: add(point, scale_by(axis, 0.5 * (lo + hi))),
        axis,
        radius,
        length: hi - lo,
    }
}

/// Unit eigenvector of the smallest eigenvalue of a symmetric matrix.
fn smallest_eigenvector(m: [[f64; 3]; 3]) -> [f64; 3] {
    let (values, vectors) = crate::icp::symmetric_eigen(m);
    let k = (0..3)
        .min_by(|&a, &b| values[a].total_cmp(&values[b]))
        .unwrap_or(2);
    unit([vectors[0][k], vectors[1][k], vectors[2][k]])
}

/// Solve `a x = b` by Gaussian elimination with partial pivoting.
fn solve<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let pivot = (col..N).max_by(|&r, &s| a[r][col].abs().total_cmp(&a[s][col].abs()))?;
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for r in col + 1..N {
            let f = a[r][col] / a[col][col];
            let pivot_row = a[col];
            for (v, p) in a[r].iter_mut().zip(pivot_row).skip(col) {
                *v -= f * p;
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0; N];
    for r in (0..N).rev() {
        let s: f64 = (r + 1..N).map(|s| a[r][s] * x[s]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Two unit vectors perpendicular to `axis` and to each other.
fn basis(axis: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let helper = if axis[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = unit(cross(axis, helper));
    (u, cross(axis, u))
}

/// Of the two opposite directions, the one whose last non-zero component
/// (z, then y, then x) is positive, so results do not depend on sampling.
fn canonical(v: [f64; 3]) -> [f64; 3] {
    let key = [v[2], v[1], v[0]]
        .into_iter()
        .find(|c| c.abs() > 1e-9)
        .unwrap_or(0.0);
    if key < 0.0 { v.map(|c| -c) } else { v }
}

/// The part of `v` perpendicular to the unit vector `axis`.
fn radial(v: [f64; 3], axis: [f64; 3]) -> [f64; 3] {
    sub(v, scale_by(axis, dot(v, axis)))
}

fn outer_add(m: &mut [[f64; 3]; 3], v: [f64; 3]) {
    for r in 0..3 {
        for s in 0..3 {
            m[r][s] += v[r] * v[s];
        }
    }
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale_by(a: [f64; 3], s: f64) -> [f64; 3] {
    a.map(|v| v * s)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// `a` scaled to unit length (zero stays zero).
fn unit(a: [f64; 3]) -> [f64; 3] {
    let len = norm(a);
    if len > 0.0 { a.map(|v| v / len) } else { a }
}

/// Deterministic xorshift generator.
struct Rng(u64);

impl Rng {
    /// Uniform in `0..n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) % n as u64) as usize
    }

    #[cfg(test)]
    fn unit(&mut self) -> f64 {
        self.below(1 << 30) as f64 / (1u64 << 30) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normals::{Orientation, estimate_normals};

    /// Uniform noise in the box `[lo, hi]`.
    fn noise(rng: &mut Rng, count: usize, lo: f64, hi: f64) -> Vec<[f64; 3]> {
        (0..count)
            .map(|_| std::array::from_fn(|_| lo + (hi - lo) * rng.unit()))
            .collect()
    }

    fn detect(points: &[[f64; 3]], params: RansacParams) -> Vec<Detected> {
        let normals = estimate_normals(points, 12, Orientation::Up);
        detect_shapes(points, &normals, &params)
    }

    #[test]
    fn two_planes_and_noise() {
        let mut rng = Rng(42);
        let mut points = Vec::new();
        // A floor z = 0 (10 x 10 m) and a wall x = 2 (10 m wide, 5 m high,
        // from 1 m up so no normal mixes both), with a little jitter.
        for j in 0..100 {
            for i in 0..100 {
                let (x, y) = (i as f64 * 0.1, j as f64 * 0.1);
                points.push([x, y, 0.004 * (rng.unit() - 0.5)]);
            }
        }
        for j in 0..50 {
            for i in 0..100 {
                let (y, z) = (i as f64 * 0.1, 1.0 + j as f64 * 0.1);
                points.push([2.0 + 0.004 * (rng.unit() - 0.5), y, z]);
            }
        }
        points.extend(noise(&mut rng, 500, 0.0, 10.0));
        let params = RansacParams {
            threshold: 0.02,
            min_support: 1000,
            ..RansacParams::default()
        };
        let found = detect(&points, params);
        assert_eq!(found.len(), 2, "{found:?}");
        let expect = [
            ([0.0, 0.0, 1.0], 0.0, 10_000),
            ([1.0, 0.0, 0.0], -2.0, 5000),
        ];
        for (d, (normal, offset, count)) in found.iter().zip(expect) {
            let Shape::Plane { normal: n, d: off } = d.shape else {
                panic!("not a plane");
            };
            assert!(dot(n, normal) > 0.9999, "{n:?}");
            assert!((off - offset).abs() < 0.005, "{off}");
            // Every plane point, plus the odd noise point that lies on it.
            assert!(
                d.indices.len() >= count && d.indices.len() < count + 20,
                "{}",
                d.indices.len()
            );
            assert!(d.rms < 0.01, "{}", d.rms);
        }
        // Deterministic.
        assert_eq!(detect(&points, params), found);
    }

    #[test]
    fn a_cylinder_next_to_a_plane() {
        // A tilted cylinder (radius 0.5, 4 m long) standing on a floor.
        let axis = unit([0.2, 0.1, 1.0]);
        let (u, v) = basis(axis);
        let base = [5.0, 5.0, 0.5];
        let mut points = Vec::new();
        for j in 0..80 {
            for i in 0..120 {
                let (t, a) = (j as f64 * 0.05, i as f64 * std::f64::consts::TAU / 120.0);
                let r = add(scale_by(u, 0.5 * a.cos()), scale_by(v, 0.5 * a.sin()));
                points.push(add(add(base, scale_by(axis, t)), r));
            }
        }
        for j in 0..100 {
            for i in 0..100 {
                points.push([i as f64 * 0.1, j as f64 * 0.1, 0.0]);
            }
        }
        let params = RansacParams {
            primitive: Primitive::Cylinder,
            threshold: 0.01,
            min_support: 500,
            ..RansacParams::default()
        };
        let found = detect(&points, params);
        assert_eq!(found.len(), 1, "{found:?}");
        let Shape::Cylinder {
            point,
            axis: a,
            radius,
            length,
        } = found[0].shape
        else {
            panic!("not a cylinder");
        };
        assert!(dot(a, axis).abs() > 0.9999, "{a:?}");
        assert!((radius - 0.5).abs() < 0.002, "{radius}");
        assert!((length - 3.95).abs() < 0.01, "{length}");
        let middle = add(base, scale_by(axis, 3.95 / 2.0));
        assert!(norm(sub(point, middle)) < 0.01, "{point:?}");
        assert!(found[0].indices.len() >= 9600 && found[0].indices.iter().all(|&i| i < 9600));
    }

    #[test]
    fn two_spheres() {
        let mut points = Vec::new();
        for (center, radius) in [([0.0, 0.0, 0.0], 1.0), ([4.0, 1.0, 0.5], 0.6)] {
            for i in 0..60 {
                for j in 0..120 {
                    let (t, f) = (
                        std::f64::consts::PI * (i as f64 + 0.5) / 60.0,
                        std::f64::consts::TAU * j as f64 / 120.0,
                    );
                    let d = [t.sin() * f.cos(), t.sin() * f.sin(), t.cos()];
                    points.push(add(center, scale_by(d, radius)));
                }
            }
        }
        let params = RansacParams {
            primitive: Primitive::Sphere,
            threshold: 0.01,
            min_support: 1000,
            ..RansacParams::default()
        };
        let found = detect(&points, params);
        assert_eq!(found.len(), 2, "{found:?}");
        let mut radii: Vec<f64> = found
            .iter()
            .map(|d| match d.shape {
                Shape::Sphere { radius, .. } => radius,
                _ => panic!("not a sphere"),
            })
            .collect();
        radii.sort_by(f64::total_cmp);
        assert!(
            (radii[0] - 0.6).abs() < 0.002 && (radii[1] - 1.0).abs() < 0.002,
            "{radii:?}"
        );
        assert!(found.iter().all(|d| d.indices.len() == 7200));
    }

    #[test]
    fn nothing_in_pure_noise() {
        let mut rng = Rng(7);
        let points = noise(&mut rng, 3000, 0.0, 10.0);
        let params = RansacParams {
            threshold: 0.01,
            min_support: 200,
            ..RansacParams::default()
        };
        assert!(detect(&points, params).is_empty());
        assert!(detect_shapes(&[], &[], &params).is_empty());
    }
}
