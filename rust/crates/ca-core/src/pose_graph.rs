//! 3D pose graphs: g2o I/O and Levenberg–Marquardt optimisation.
//!
//! Nodes are SE(3) poses (body to world); an edge from `i` to `j` measures
//! `Z = Xi^-1 Xj` with a 6x6 information matrix in g2o's convention
//! (translation, then quaternion vector part). The optimiser minimises
//! `sum rho(e^T W e)` over the non-fixed poses, where
//! `e = [t, log R]` of `Z^-1 Xi^-1 Xj` and `W` is the information rescaled
//! from quaternion to rotation-vector units, so costs match g2o's to first
//! order. Loop edges can use a Huber kernel so a wrong loop closure bends
//! the graph less. The normal equations are solved with a block-sparse
//! Cholesky factorisation in minimum-degree order, which keeps the fill of
//! chain-plus-loops graphs small; no dependencies beyond `std`.
//!
//! Plane landmarks (e.g. the floor, seen from many keyframes) are optional
//! extra variables: an edge from a node to a plane says where the node sees
//! it, as `n · x + d = 0` in the node's frame. Tying keyframes to one floor
//! removes the tilt and height drift odometry leaves.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};

use crate::icp::{IcpParams, IcpResult, Rigid, icp};
use crate::kdtree::KdTree;
use crate::trajectory::{quaternion, rotation_matrix};
use crate::{Attribute, AttributeValues, INTENSITY, PointCloud};

type Mat3 = [[f64; 3]; 3];
type Vec6 = [f64; 6];
type Mat6 = [[f64; 6]; 6];

const ZERO6: Mat6 = [[0.0; 6]; 6];

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct PoseGraphError(pub String);

fn fail<T>(message: impl Into<String>) -> Result<T, PoseGraphError> {
    Err(PoseGraphError(message.into()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EdgeKind {
    /// Between poses that follow each other.
    Odometry,
    /// Between poses far apart in time: a loop closure.
    Loop,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Node {
    /// The id in the source file (the frame index for trajectories).
    pub id: i64,
    pub pose: Rigid,
    /// Held in place by the optimiser.
    pub fixed: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Edge {
    /// Node indices (into [`PoseGraph::nodes`], not ids).
    pub from: usize,
    pub to: usize,
    /// The pose of `to` in the frame of `from`.
    pub measurement: Rigid,
    /// 6x6 information in g2o order: x, y, z, qx, qy, qz.
    pub information: Mat6,
    pub kind: EdgeKind,
}

/// A plane landmark `n · x + d = 0` in world coordinates, `n` a unit vector.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Plane {
    /// Its vertex id in g2o files (distinct from the node ids).
    pub id: i64,
    pub coefficients: [f64; 4],
    pub fixed: bool,
}

/// A node's view of a plane: its coefficients in the node's frame, with a
/// 3x3 information on (normal tilt about two axes, offset).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PlaneEdge {
    pub node: usize,
    pub plane: usize,
    pub measurement: [f64; 4],
    pub information: [[f64; 3]; 3],
}

/// A keyframe's measured up direction (a unit vector in its frame), e.g.
/// from an IMU's roll and pitch, with a 2x2 information on its tilt.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GravityEdge {
    pub node: usize,
    pub up: [f64; 3],
    pub information: [[f64; 2]; 2],
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PoseGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub planes: Vec<Plane>,
    pub plane_edges: Vec<PlaneEdge>,
    /// Up in world coordinates (a unit vector), for the gravity edges.
    pub gravity: [f64; 3],
    pub gravity_edges: Vec<GravityEdge>,
}

/// Information for an up direction measured with standard deviation
/// `sigma` (radians, per tilt axis).
pub fn gravity_information(sigma: f64) -> [[f64; 2]; 2] {
    let a = 1.0 / (sigma * sigma);
    [[a, 0.0], [0.0, a]]
}

/// Residual of a gravity edge, padded to six entries: the predicted up
/// direction (world up seen from `x`) along the measured one's tangent basis.
fn gravity_residual(x: &Rigid, world_up: &[f64; 3], measured: &[f64; 3]) -> Vec6 {
    let r = &x.rotation;
    let predicted: [f64; 3] = std::array::from_fn(|i| (0..3).map(|k| r[k][i] * world_up[k]).sum());
    let (u, v) = tangent_basis(measured);
    [
        dot3(&u, &predicted),
        dot3(&v, &predicted),
        0.0,
        0.0,
        0.0,
        0.0,
    ]
}

fn gravity_weight(information: &[[f64; 2]; 2]) -> Mat6 {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i < 2 && j < 2 {
                information[i][j]
            } else {
                0.0
            }
        })
    })
}

fn gravity_jacobian(x: &Rigid, world_up: &[f64; 3], measured: &[f64; 3]) -> Mat6 {
    const H: f64 = 1e-6;
    let mut j = ZERO6;
    for c in 0..6 {
        let mut d = [0.0; 6];
        d[c] = H;
        let plus = gravity_residual(&retract(x, &d), world_up, measured);
        d[c] = -H;
        let minus = gravity_residual(&retract(x, &d), world_up, measured);
        for r in 0..6 {
            j[r][c] = (plus[r] - minus[r]) / (2.0 * H);
        }
    }
    j
}

/// Replace the gravity ties with `measured` (node index and up in its
/// frame), each with standard deviation `sigma` (radians): world up comes
/// from the first 1 % of the measurements (1 to 10), where odometry has
/// drifted least (see [`tie_to_gravity`]). False when none is usable.
pub fn set_gravity(graph: &mut PoseGraph, measured: &[(usize, [f64; 3])], sigma: f64) -> bool {
    graph.gravity_edges.clear();
    let reference = (measured.len() / 100).clamp(1, 10);
    tie_to_gravity(graph, measured, gravity_information(sigma), reference)
}

fn apply3(m: &Mat3, v: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|k| m[i][k] * v[k]).sum())
}

/// Each up in `ups` (a sensor's frame) turned by `mount` and seen in the
/// world through its keyframe's `rotations`, and their mean direction.
fn seen_ups(rotations: &[Mat3], ups: &[[f64; 3]], mount: &Mat3) -> (Vec<[f64; 3]>, [f64; 3]) {
    let seen: Vec<[f64; 3]> = rotations
        .iter()
        .zip(ups)
        .map(|(r, u)| apply3(r, &apply3(mount, u)))
        .collect();
    let mut mean = [0.0; 3];
    for v in &seen {
        for k in 0..3 {
            mean[k] += v[k];
        }
    }
    let norm = dot3(&mean, &mean).sqrt().max(1e-12);
    (seen, mean.map(|v| v / norm))
}

fn angle(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let n = (dot3(a, a) * dot3(b, b)).sqrt().max(1e-24);
    (dot3(a, b) / n).clamp(-1.0, 1.0).acos()
}

/// Per keyframe, the angle (degrees) between its measured up, turned by
/// `mount` and seen in the world through its rotation, and the mean of
/// them: all 0 when the up directions and the poses agree. A skewed mount
/// shows as the drive turns; so does odometry drift, over a long drive.
pub fn up_spread(rotations: &[Mat3], ups: &[[f64; 3]], mount: &Mat3) -> Vec<f64> {
    let (seen, mean) = seen_ups(rotations, ups, mount);
    seen.iter().map(|v| angle(v, &mean).to_degrees()).collect()
}

/// How noisy the up directions are (degrees), turned by `mount` and seen
/// through the rotations: from keyframes 1 and 2 apart, the part of their
/// disagreement that does not grow with the distance between them. Odometry
/// drift d per keyframe adds d² and 4d² to the mean squared angles A1² and
/// A2², noise σ adds 2σ² to both, so σ² = (4 A1² - A2²) / 6.
pub fn up_noise(rotations: &[Mat3], ups: &[[f64; 3]], mount: &Mat3) -> f64 {
    let (seen, _) = seen_ups(rotations, ups, mount);
    let mean_square = |k: usize| {
        let angles: Vec<f64> = (0..seen.len().saturating_sub(k))
            .map(|i| angle(&seen[i], &seen[i + k]))
            .collect();
        angles.iter().map(|a| a * a).sum::<f64>() / angles.len().max(1) as f64
    };
    ((4.0 * mean_square(1) - mean_square(2)) / 6.0)
        .max(0.0)
        .sqrt()
        .to_degrees()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied().unwrap_or(0.0)
}

/// The rotation of an IMU's frame into the scans' frame that its up
/// directions `ups` agree with best, seen through the keyframes'
/// `rotations`: the one that makes them, turned and seen in the world,
/// closest to their mean. A drive that turns and tilts pins it down.
/// Gauss-Newton on a rotation vector, from an upright and two upside-down
/// starts; the up's sign is free (tying to gravity does not depend on it).
pub fn calibrate_ups(rotations: &[Mat3], ups: &[[f64; 3]]) -> Mat3 {
    let residual = |x: &[f64; 3]| -> Vec<f64> {
        let (seen, mean) = seen_ups(rotations, ups, &exp_so3(x));
        seen.iter()
            .flat_map(|v| [v[0] - mean[0], v[1] - mean[1], v[2] - mean[2]])
            .collect()
    };
    let cost = |r: &[f64]| r.iter().map(|v| v * v).sum::<f64>();
    let mut best = ([0.0; 3], f64::INFINITY);
    for start in [
        [0.0; 3],
        [std::f64::consts::PI, 0.0, 0.0],
        [0.0, std::f64::consts::PI, 0.0],
    ] {
        let mut x = start;
        let mut lambda = 1e-3;
        let mut r = residual(&x);
        let mut c = cost(&r);
        for _ in 0..100 {
            // Numerical Jacobian (3 columns), then damped normal equations.
            let h = 1e-6;
            let columns: Vec<Vec<f64>> = (0..3)
                .map(|k| {
                    let mut xk = x;
                    xk[k] += h;
                    residual(&xk)
                        .iter()
                        .zip(&r)
                        .map(|(a, b)| (a - b) / h)
                        .collect()
                })
                .collect();
            let mut a = [[0.0; 3]; 3];
            let mut g = [0.0; 3];
            for i in 0..3 {
                for j in 0..3 {
                    a[i][j] = columns[i].iter().zip(&columns[j]).map(|(p, q)| p * q).sum();
                }
                g[i] = -columns[i].iter().zip(&r).map(|(p, q)| p * q).sum::<f64>();
            }
            let mut improved = false;
            for _ in 0..10 {
                let mut damped = a;
                for (i, row) in damped.iter_mut().enumerate() {
                    row[i] += lambda * (1.0 + a[i][i]);
                }
                let Some(step) = solve3(&damped, &g) else {
                    break;
                };
                let candidate = [x[0] + step[0], x[1] + step[1], x[2] + step[2]];
                let rc = residual(&candidate);
                let cc = cost(&rc);
                if cc < c {
                    (x, r, c) = (candidate, rc, cc);
                    lambda = (lambda * 0.3).max(1e-9);
                    improved = true;
                    break;
                }
                lambda *= 10.0;
            }
            if !improved || c < 1e-18 {
                break;
            }
        }
        if c < best.1 {
            best = (x, c);
        }
    }
    exp_so3(&best.0)
}

fn solve3(a: &Mat3, b: &[f64; 3]) -> Option<[f64; 3]> {
    let det = |m: &Mat3| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(a);
    if d.abs() < 1e-300 {
        return None;
    }
    Some(std::array::from_fn(|k| {
        let mut m = *a;
        for i in 0..3 {
            m[i][k] = b[i];
        }
        det(&m) / d
    }))
}

/// An estimated IMU mount is used only when it takes this much (degrees)
/// off the up directions' spread, and at least half of it...
const MOUNT_GAIN_DEG: f64 = 1.0;
/// ...over a drive of this many keyframes at least: over a few, drift that
/// follows the heading can pass for a skewed mount.
const MOUNT_MIN_KEYFRAMES: usize = 50;

/// What [`set_gravity_calibrated`] did.
#[derive(Debug, Clone, PartialEq)]
pub struct GravityTie {
    /// Keyframes tied.
    pub tied: usize,
    /// Median spread (degrees) of the up directions as measured, see [`up_spread`].
    pub spread: f64,
    /// The IMU's rotation into the scans' frame, when estimating it made
    /// them agree better, and their median spread then.
    pub mount: Option<(Mat3, f64)>,
    /// Their noise (degrees, see [`up_noise`]), with the rotation if used.
    pub noise: f64,
    /// The standard deviation used (degrees): at least the noise.
    pub sigma_deg: f64,
}

/// [`set_gravity`] for a real IMU: with `calibrate`, its rotation into the
/// scans' frame is estimated from the drive ([`calibrate_ups`]) and used
/// when it makes the ups agree clearly better (half the spread, and a
/// degree less at least); the standard deviation is at least
/// their noise ([`up_noise`], `sigma_deg` otherwise), so a noisy IMU levels
/// the map without bending it while drift is still removed. None when no
/// up is usable.
pub fn set_gravity_calibrated(
    graph: &mut PoseGraph,
    measured: &[(usize, [f64; 3])],
    sigma_deg: f64,
    calibrate: bool,
) -> Option<GravityTie> {
    let rotations: Vec<Mat3> = measured
        .iter()
        .map(|&(i, _)| graph.nodes[i].pose.rotation)
        .collect();
    let ups: Vec<[f64; 3]> = measured.iter().map(|&(_, u)| u).collect();
    let identity = exp_so3(&[0.0; 3]);
    let spread = median(up_spread(&rotations, &ups, &identity));
    let mut mount = None;
    if calibrate && measured.len() >= MOUNT_MIN_KEYFRAMES {
        let m = calibrate_ups(&rotations, &ups);
        let after = median(up_spread(&rotations, &ups, &m));
        // Only a clear gain: drift that follows the heading (a lap round a
        // block) can pass for a slightly skewed mount.
        if after < 0.5 * spread && spread - after > MOUNT_GAIN_DEG {
            mount = Some((m, after));
        }
    }
    let used: Vec<(usize, [f64; 3])> = match &mount {
        Some((m, _)) => measured.iter().map(|&(i, u)| (i, apply3(m, &u))).collect(),
        None => measured.to_vec(),
    };
    let noise = up_noise(
        &rotations,
        &ups,
        mount.as_ref().map_or(&identity, |(m, _)| m),
    );
    let sigma = sigma_deg.max(noise);
    set_gravity(graph, &used, sigma.to_radians()).then_some(GravityTie {
        tied: used.len(),
        spread,
        mount,
        noise,
        sigma_deg: sigma,
    })
}

/// Tie keyframes to gravity from the up direction each measured (`ups`:
/// node index and up in its frame, e.g. from an IMU's roll and pitch).
/// World up is taken as the mean of the first `reference` measurements
/// seen through their poses, where odometry has drifted least. Unlike a
/// floor, this assumes nothing about the ground, so hills stay hills while
/// tilt drift, and the height drift it causes, go.
pub fn tie_to_gravity(
    graph: &mut PoseGraph,
    ups: &[(usize, [f64; 3])],
    information: [[f64; 2]; 2],
    reference: usize,
) -> bool {
    let mut sum = [0.0; 3];
    for &(node, up) in ups.iter().take(reference.max(1)) {
        let r = &graph.nodes[node].pose.rotation;
        for i in 0..3 {
            sum[i] += (0..3).map(|k| r[i][k] * up[k]).sum::<f64>();
        }
    }
    if ups.is_empty() || dot3(&sum, &sum) == 0.0 {
        return false;
    }
    graph.gravity = unit3(sum);
    graph
        .gravity_edges
        .extend(ups.iter().map(|&(node, up)| GravityEdge {
            node,
            up: unit3(up),
            information,
        }));
    true
}

/// Information for a plane seen with standard deviations `sigma_angle`
/// (radians, per tilt axis) and `sigma_offset` (metres).
pub fn plane_information(sigma_angle: f64, sigma_offset: f64) -> [[f64; 3]; 3] {
    let a = 1.0 / (sigma_angle * sigma_angle);
    [
        [a, 0.0, 0.0],
        [0.0, a, 0.0],
        [0.0, 0.0, 1.0 / (sigma_offset * sigma_offset)],
    ]
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit3(v: [f64; 3]) -> [f64; 3] {
    let n = dot3(&v, &v).sqrt();
    v.map(|x| x / n)
}

/// Two unit vectors orthogonal to the unit vector `n` and to each other.
fn tangent_basis(n: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    let helper = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = unit3(cross3(n, &helper));
    (u, cross3(n, &u))
}

/// A world plane as seen from pose `x` (its coefficients in `x`'s frame).
pub fn plane_in_frame(plane: &[f64; 4], x: &Rigid) -> [f64; 4] {
    let n = [plane[0], plane[1], plane[2]];
    let r = &x.rotation;
    let local: [f64; 3] = std::array::from_fn(|i| (0..3).map(|k| r[k][i] * n[k]).sum());
    [
        local[0],
        local[1],
        local[2],
        dot3(&n, &x.translation) + plane[3],
    ]
}

/// A plane seen from pose `x` in world coordinates (the inverse of [`plane_in_frame`]).
pub fn plane_in_world(local: &[f64; 4], x: &Rigid) -> [f64; 4] {
    let l = [local[0], local[1], local[2]];
    let r = &x.rotation;
    let n: [f64; 3] = std::array::from_fn(|i| (0..3).map(|k| r[i][k] * l[k]).sum());
    [n[0], n[1], n[2], local[3] - dot3(&n, &x.translation)]
}

/// The plane moved by `d = [tilt u, tilt v, offset]` (tilts about its own tangent basis).
fn retract_plane(plane: &[f64; 4], d: &Vec6) -> [f64; 4] {
    let n = [plane[0], plane[1], plane[2]];
    let (u, v) = tangent_basis(&n);
    let moved = unit3(std::array::from_fn(|i| n[i] + d[0] * u[i] + d[1] * v[i]));
    [moved[0], moved[1], moved[2], plane[3] + d[2]]
}

/// Residual of a plane edge, padded to six entries: the predicted normal's
/// components along the measured normal's tangent basis, and the offset
/// difference.
fn plane_residual(x: &Rigid, plane: &[f64; 4], measurement: &[f64; 4]) -> Vec6 {
    let predicted = plane_in_frame(plane, x);
    let m = [measurement[0], measurement[1], measurement[2]];
    let (u, v) = tangent_basis(&m);
    let n = [predicted[0], predicted[1], predicted[2]];
    [
        dot3(&u, &n),
        dot3(&v, &n),
        predicted[3] - measurement[3],
        0.0,
        0.0,
        0.0,
    ]
}

/// A plane edge's 3x3 information padded to 6x6.
fn plane_weight(information: &[[f64; 3]; 3]) -> Mat6 {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i < 3 && j < 3 {
                information[i][j]
            } else {
                0.0
            }
        })
    })
}

fn plane_jacobians(x: &Rigid, plane: &[f64; 4], measurement: &[f64; 4]) -> (Mat6, Mat6) {
    const H: f64 = 1e-6;
    let mut jx = ZERO6;
    let mut jp = ZERO6;
    for c in 0..6 {
        let mut d = [0.0; 6];
        d[c] = H;
        let plus_x = plane_residual(&retract(x, &d), plane, measurement);
        let plus_p = (c < 3).then(|| plane_residual(x, &retract_plane(plane, &d), measurement));
        d[c] = -H;
        let minus_x = plane_residual(&retract(x, &d), plane, measurement);
        let minus_p = (c < 3).then(|| plane_residual(x, &retract_plane(plane, &d), measurement));
        for r in 0..6 {
            jx[r][c] = (plus_x[r] - minus_x[r]) / (2.0 * H);
            if let (Some(p), Some(m)) = (plus_p, minus_p) {
                jp[r][c] = (p[r] - m[r]) / (2.0 * H);
            }
        }
    }
    (jx, jp)
}

/// Information for standard deviations `sigma_t` (metres, per axis) and
/// `sigma_r` (radians, per axis), in g2o's quaternion units.
pub fn isotropic_information(sigma_t: f64, sigma_r: f64) -> Mat6 {
    let mut m = ZERO6;
    for i in 0..3 {
        m[i][i] = 1.0 / (sigma_t * sigma_t);
        // A rotation of theta has a quaternion vector part of about theta / 2.
        m[i + 3][i + 3] = 4.0 / (sigma_r * sigma_r);
    }
    m
}

// --- SE(3) helpers ----------------------------------------------------------

/// The inverse rigid transform.
pub fn inverse(x: &Rigid) -> Rigid {
    let r = &x.rotation;
    let rotation: Mat3 = std::array::from_fn(|i| std::array::from_fn(|j| r[j][i]));
    let t = &x.translation;
    let translation = std::array::from_fn(|i| -(0..3).map(|k| rotation[i][k] * t[k]).sum::<f64>());
    Rigid {
        rotation,
        translation,
    }
}

/// Rotation of the rotation vector `w` (Rodrigues).
pub(crate) fn exp_so3(w: &[f64; 3]) -> Mat3 {
    let theta2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    let theta = theta2.sqrt();
    let (a, b) = if theta < 1e-8 {
        (1.0 - theta2 / 6.0, 0.5 - theta2 / 24.0)
    } else {
        (theta.sin() / theta, (1.0 - theta.cos()) / theta2)
    };
    let k = [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]];
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let kk: f64 = (0..3).map(|m| k[i][m] * k[m][j]).sum();
            f64::from(u8::from(i == j)) + a * k[i][j] + b * kk
        })
    })
}

/// Rotation vector of `r`.
pub(crate) fn log_so3(r: &Mat3) -> [f64; 3] {
    let v = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
    // atan2 keeps small angles precise, where acos of the trace does not.
    let sin = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt() / 2.0;
    let cos = ((r[0][0] + r[1][1] + r[2][2] - 1.0) / 2.0).clamp(-1.0, 1.0);
    let theta = sin.atan2(cos);
    if theta < 1e-4 {
        return v.map(|x| x * (0.5 + theta * theta / 12.0));
    }
    if std::f64::consts::PI - theta > 1e-4 {
        let s = theta / (2.0 * sin);
        return v.map(|x| x * s);
    }
    // Near pi: the axis from the diagonal, its sign from the skew part.
    let i = (0..3)
        .max_by(|&a, &b| r[a][a].total_cmp(&r[b][b]))
        .unwrap_or(0);
    let mut axis = [0.0; 3];
    axis[i] = ((r[i][i] - cos) / (1.0 - cos)).max(0.0).sqrt();
    for j in 0..3 {
        if j != i {
            axis[j] = (r[i][j] + r[j][i]) / (2.0 * (1.0 - cos) * axis[i]);
        }
    }
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let sign = if axis.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>() < 0.0 {
        -1.0
    } else {
        1.0
    };
    axis.map(|a| sign * theta * a / norm)
}

/// `x` moved by `d = [dt, dtheta]` in its own frame.
fn retract(x: &Rigid, d: &Vec6) -> Rigid {
    x.compose(&Rigid {
        rotation: exp_so3(&[d[3], d[4], d[5]]),
        translation: [d[0], d[1], d[2]],
    })
}

fn residual(xi: &Rigid, xj: &Rigid, z_inv: &Rigid) -> Vec6 {
    let e = z_inv.compose(&inverse(xi).compose(xj));
    let w = log_so3(&e.rotation);
    [
        e.translation[0],
        e.translation[1],
        e.translation[2],
        w[0],
        w[1],
        w[2],
    ]
}

/// Information in `[t, rotation vector]` units.
fn weight(information: &Mat6) -> Mat6 {
    let d = |i: usize| if i < 3 { 1.0 } else { 0.5 };
    std::array::from_fn(|i| std::array::from_fn(|j| information[i][j] * d(i) * d(j)))
}

fn quadratic(e: &Vec6, w: &Mat6) -> f64 {
    (0..6)
        .map(|i| e[i] * (0..6).map(|j| w[i][j] * e[j]).sum::<f64>())
        .sum()
}

/// Huber cost and IRLS weight of a squared error.
fn robust(chi2: f64, delta: Option<f64>) -> (f64, f64) {
    match delta {
        Some(d) if chi2 > d * d => {
            let s = chi2.sqrt();
            (2.0 * d * s - d * d, d / s)
        }
        _ => (chi2, 1.0),
    }
}

// --- g2o I/O ----------------------------------------------------------------

fn pose_from(v: &[f64]) -> Rigid {
    let norm = (v[3] * v[3] + v[4] * v[4] + v[5] * v[5] + v[6] * v[6]).sqrt();
    let q = [v[3] / norm, v[4] / norm, v[5] / norm, v[6] / norm];
    Rigid {
        rotation: rotation_matrix(&q),
        translation: [v[0], v[1], v[2]],
    }
}

fn pose_text(x: &Rigid) -> String {
    let mut q = quaternion(&x.rotation);
    if q[3] < 0.0 {
        q = q.map(|c| -c);
    }
    let t = &x.translation;
    format!(
        "{} {} {} {} {} {} {}",
        t[0], t[1], t[2], q[0], q[1], q[2], q[3]
    )
}

impl PoseGraph {
    /// Read `VERTEX_SE3:QUAT`, `EDGE_SE3:QUAT`, `VERTEX_PLANE`,
    /// `EDGE_SE3_PLANE` and `FIX` lines; other element types are skipped.
    /// Edges between consecutive ids are odometry, the rest loops.
    pub fn from_g2o(text: &str) -> Result<PoseGraph, PoseGraphError> {
        let mut graph = PoseGraph::default();
        let mut index = HashMap::new();
        let mut plane_index = HashMap::new();
        let mut edges = Vec::new();
        let mut plane_edges = Vec::new();
        let mut fixed = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let mut fields = line.split_whitespace();
            let Some(tag) = fields.next() else { continue };
            let parse = |f: &str| {
                f.parse::<f64>()
                    .map_err(|_| PoseGraphError(format!("line {}: {f:?} is not a number", n + 1)))
            };
            let id = |f: Option<&str>| -> Result<i64, PoseGraphError> {
                f.and_then(|f| f.parse().ok())
                    .ok_or_else(|| PoseGraphError(format!("line {}: bad vertex id", n + 1)))
            };
            match tag {
                "VERTEX_SE3:QUAT" => {
                    let v = id(fields.next())?;
                    let values = fields.map(parse).collect::<Result<Vec<_>, _>>()?;
                    if values.len() < 7 {
                        return fail(format!("line {}: a vertex needs 7 numbers", n + 1));
                    }
                    if index.insert(v, graph.nodes.len()).is_some() {
                        return fail(format!("line {}: vertex {v} appears twice", n + 1));
                    }
                    graph.nodes.push(Node {
                        id: v,
                        pose: pose_from(&values),
                        fixed: false,
                    });
                }
                "EDGE_SE3:QUAT" => {
                    let a = id(fields.next())?;
                    let b = id(fields.next())?;
                    let values = fields.map(parse).collect::<Result<Vec<_>, _>>()?;
                    if values.len() < 28 {
                        return fail(format!(
                            "line {}: an edge needs 7 numbers and 21 information entries",
                            n + 1
                        ));
                    }
                    let mut information = ZERO6;
                    let mut upper = values[7..].iter();
                    for (i, j) in (0..6).flat_map(|i| (i..6).map(move |j| (i, j))) {
                        let x = *upper.next().unwrap_or(&0.0);
                        information[i][j] = x;
                        information[j][i] = x;
                    }
                    edges.push((n + 1, a, b, pose_from(&values), information));
                }
                "VERTEX_PLANE" => {
                    let v = id(fields.next())?;
                    let values = fields.map(parse).collect::<Result<Vec<_>, _>>()?;
                    if values.len() < 4 {
                        return fail(format!("line {}: a plane needs 4 numbers", n + 1));
                    }
                    let norm = dot3(
                        &[values[0], values[1], values[2]],
                        &[values[0], values[1], values[2]],
                    )
                    .sqrt();
                    plane_index.insert(v, graph.planes.len());
                    graph.planes.push(Plane {
                        id: v,
                        coefficients: [values[0], values[1], values[2], values[3]]
                            .map(|c| c / norm),
                        fixed: false,
                    });
                }
                "EDGE_SE3_PLANE" => {
                    let a = id(fields.next())?;
                    let b = id(fields.next())?;
                    let values = fields.map(parse).collect::<Result<Vec<_>, _>>()?;
                    if values.len() < 10 {
                        return fail(format!(
                            "line {}: a plane edge needs 4 numbers and 6 information entries",
                            n + 1
                        ));
                    }
                    let mut information = [[0.0; 3]; 3];
                    let mut upper = values[4..].iter();
                    for (i, j) in (0..3).flat_map(|i| (i..3).map(move |j| (i, j))) {
                        let x = *upper.next().unwrap_or(&0.0);
                        information[i][j] = x;
                        information[j][i] = x;
                    }
                    let norm = dot3(
                        &[values[0], values[1], values[2]],
                        &[values[0], values[1], values[2]],
                    )
                    .sqrt();
                    let measurement =
                        [values[0], values[1], values[2], values[3]].map(|c| c / norm);
                    plane_edges.push((n + 1, a, b, measurement, information));
                }
                "FIX" => {
                    for f in fields {
                        fixed.push(id(Some(f))?);
                    }
                }
                _ => {}
            }
        }
        if graph.nodes.is_empty() {
            return fail("no VERTEX_SE3:QUAT lines");
        }
        let lookup = |line: usize, v: i64| {
            index
                .get(&v)
                .copied()
                .ok_or_else(|| PoseGraphError(format!("line {line}: no vertex {v}")))
        };
        for (line, a, b, measurement, information) in edges {
            graph.edges.push(Edge {
                from: lookup(line, a)?,
                to: lookup(line, b)?,
                measurement,
                information,
                kind: if (a - b).abs() == 1 {
                    EdgeKind::Odometry
                } else {
                    EdgeKind::Loop
                },
            });
        }
        for (line, a, b, measurement, information) in plane_edges {
            let plane = plane_index
                .get(&b)
                .copied()
                .ok_or_else(|| PoseGraphError(format!("line {line}: no plane {b}")))?;
            graph.plane_edges.push(PlaneEdge {
                node: lookup(line, a)?,
                plane,
                measurement,
                information,
            });
        }
        for v in fixed {
            if let Some(&i) = index.get(&v) {
                graph.nodes[i].fixed = true;
            }
            if let Some(&p) = plane_index.get(&v) {
                graph.planes[p].fixed = true;
            }
        }
        Ok(graph)
    }

    pub fn to_g2o(&self) -> String {
        let mut out = String::new();
        for node in &self.nodes {
            out += &format!("VERTEX_SE3:QUAT {} {}\n", node.id, pose_text(&node.pose));
        }
        for plane in &self.planes {
            let c = plane.coefficients;
            out += &format!(
                "VERTEX_PLANE {} {} {} {} {}\n",
                plane.id, c[0], c[1], c[2], c[3]
            );
        }
        for node in self.nodes.iter().filter(|n| n.fixed) {
            out += &format!("FIX {}\n", node.id);
        }
        for plane in self.planes.iter().filter(|p| p.fixed) {
            out += &format!("FIX {}\n", plane.id);
        }
        for edge in &self.edges {
            out += &format!(
                "EDGE_SE3:QUAT {} {} {}",
                self.nodes[edge.from].id,
                self.nodes[edge.to].id,
                pose_text(&edge.measurement)
            );
            for i in 0..6 {
                for j in i..6 {
                    out += &format!(" {}", edge.information[i][j]);
                }
            }
            out.push('\n');
        }
        for edge in &self.plane_edges {
            let m = edge.measurement;
            out += &format!(
                "EDGE_SE3_PLANE {} {} {} {} {} {}",
                self.nodes[edge.node].id, self.planes[edge.plane].id, m[0], m[1], m[2], m[3]
            );
            for i in 0..3 {
                for j in i..3 {
                    out += &format!(" {}", edge.information[i][j]);
                }
            }
            out.push('\n');
        }
        out
    }

    /// A new plane landmark at `coefficients` (world), with an id past every
    /// node and plane id; returns its index.
    pub fn add_plane(&mut self, coefficients: [f64; 4]) -> usize {
        let top = self
            .nodes
            .iter()
            .map(|n| n.id)
            .chain(self.planes.iter().map(|p| p.id))
            .max()
            .unwrap_or(-1);
        self.planes.push(Plane {
            id: top + 1,
            coefficients,
            fixed: false,
        });
        self.planes.len() - 1
    }

    /// Squared error of every gravity edge.
    pub fn gravity_edge_errors(&self) -> Vec<f64> {
        self.gravity_edges
            .iter()
            .map(|e| {
                let r = gravity_residual(&self.nodes[e.node].pose, &self.gravity, &e.up);
                quadratic(&r, &gravity_weight(&e.information))
            })
            .collect()
    }

    /// Squared error of every plane edge.
    pub fn plane_edge_errors(&self) -> Vec<f64> {
        self.plane_edges
            .iter()
            .map(|e| {
                let r = plane_residual(
                    &self.nodes[e.node].pose,
                    &self.planes[e.plane].coefficients,
                    &e.measurement,
                );
                quadratic(&r, &plane_weight(&e.information))
            })
            .collect()
    }

    /// A chain of odometry edges through `poses`, measured from the poses
    /// themselves, with ids `0..n` and the first pose fixed.
    pub fn from_poses(poses: &[Rigid], information: Mat6) -> PoseGraph {
        let nodes = poses
            .iter()
            .enumerate()
            .map(|(i, &pose)| Node {
                id: i as i64,
                pose,
                fixed: i == 0,
            })
            .collect();
        let edges = poses
            .windows(2)
            .enumerate()
            .map(|(i, w)| Edge {
                from: i,
                to: i + 1,
                measurement: inverse(&w[0]).compose(&w[1]),
                information,
                kind: EdgeKind::Odometry,
            })
            .collect();
        PoseGraph {
            nodes,
            edges,
            ..PoseGraph::default()
        }
    }

    /// Relative pose of node `to` in the frame of node `from`, as the graph
    /// currently has it.
    pub fn relative(&self, from: usize, to: usize) -> Rigid {
        inverse(&self.nodes[from].pose).compose(&self.nodes[to].pose)
    }

    /// Add a loop edge measuring `to` in the frame of `from`; returns its index.
    pub fn add_loop(
        &mut self,
        from: usize,
        to: usize,
        measurement: Rigid,
        information: Mat6,
    ) -> usize {
        self.edges.push(Edge {
            from,
            to,
            measurement,
            information,
            kind: EdgeKind::Loop,
        });
        self.edges.len() - 1
    }

    /// The poses as a KITTI file: a row-major 3x4 matrix per line.
    pub fn to_kitti(&self) -> String {
        let mut out = String::new();
        for node in &self.nodes {
            let m = node.pose.to_matrix();
            let row: Vec<String> = m[..12].iter().map(f64::to_string).collect();
            out += &row.join(" ");
            out.push('\n');
        }
        out
    }

    /// The poses as a TUM file, with `timestamps` (one per node) or, when
    /// they do not match, the node ids as times.
    pub fn to_tum(&self, timestamps: &[f64]) -> String {
        let mut out = String::new();
        for (i, node) in self.nodes.iter().enumerate() {
            let time = if timestamps.len() == self.nodes.len() {
                timestamps[i]
            } else {
                node.id as f64
            };
            out += &format!("{time} {}\n", pose_text(&node.pose));
        }
        out
    }

    /// Squared error `e^T W e` of every edge (before any robust kernel).
    pub fn edge_errors(&self) -> Vec<f64> {
        self.edges
            .iter()
            .map(|e| {
                let r = residual(
                    &self.nodes[e.from].pose,
                    &self.nodes[e.to].pose,
                    &inverse(&e.measurement),
                );
                quadratic(&r, &weight(&e.information))
            })
            .collect()
    }

    fn cost(&self, params: &OptimizeParams) -> f64 {
        let poses: f64 = self
            .edge_errors()
            .iter()
            .zip(&self.edges)
            .map(|(&chi2, e)| robust(chi2, params.kernel(e.kind)).0)
            .sum();
        poses
            + self.plane_edge_errors().iter().sum::<f64>()
            + self.gravity_edge_errors().iter().sum::<f64>()
    }
}

// --- Optimisation -------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct OptimizeParams {
    pub max_iterations: usize,
    /// Stop when the cost improves by less than this fraction.
    pub tolerance: f64,
    /// Huber threshold on `sqrt(e^T W e)` for loop edges; `None` for plain
    /// least squares.
    pub loop_kernel: Option<f64>,
}

impl Default for OptimizeParams {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-9,
            loop_kernel: None,
        }
    }
}

impl OptimizeParams {
    fn kernel(&self, kind: EdgeKind) -> Option<f64> {
        match kind {
            EdgeKind::Loop => self.loop_kernel,
            EdgeKind::Odometry => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimizeReport {
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Lower-triangular block matrix whose columns have the structure of the
/// Cholesky factor.
struct BlockMatrix {
    diagonal: Vec<Mat6>,
    /// Per column, `(row, block)` sorted by row; rows are below the column.
    below: Vec<Vec<(usize, Mat6)>>,
}

impl BlockMatrix {
    fn block(&mut self, row: usize, column: usize) -> &mut Mat6 {
        if row == column {
            return &mut self.diagonal[row];
        }
        let (row, column) = (row.max(column), row.min(column));
        let blocks = &mut self.below[column];
        let k = blocks
            .binary_search_by_key(&row, |b| b.0)
            .expect("block outside the factor's structure");
        &mut blocks[k].1
    }
}

/// Minimum-degree elimination order of the variables and, for each position,
/// the later positions its factor column touches.
fn symbolic(variables: usize, adjacency: &[BTreeSet<usize>]) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut adjacency = adjacency.to_vec();
    let mut eliminated = vec![false; variables];
    let mut heap: BinaryHeap<_> = (0..variables)
        .map(|v| Reverse((adjacency[v].len(), v)))
        .collect();
    let mut order = Vec::with_capacity(variables);
    let mut columns = Vec::with_capacity(variables);
    while let Some(Reverse((degree, v))) = heap.pop() {
        if eliminated[v] || degree != adjacency[v].len() {
            continue;
        }
        eliminated[v] = true;
        let neighbours: Vec<usize> = adjacency[v].iter().copied().collect();
        for &a in &neighbours {
            adjacency[a].remove(&v);
            for &b in &neighbours {
                if a != b {
                    adjacency[a].insert(b);
                }
            }
            heap.push(Reverse((adjacency[a].len(), a)));
        }
        order.push(v);
        columns.push(neighbours);
    }
    let mut position = vec![0; variables];
    for (p, &v) in order.iter().enumerate() {
        position[v] = p;
    }
    let columns = columns
        .into_iter()
        .map(|c| {
            let mut rows: Vec<usize> = c.into_iter().map(|v| position[v]).collect();
            rows.sort_unstable();
            rows
        })
        .collect();
    (position, columns)
}

fn cholesky6(a: &Mat6) -> Option<Mat6> {
    let mut l = ZERO6;
    for j in 0..6 {
        let d = a[j][j] - (0..j).map(|k| l[j][k] * l[j][k]).sum::<f64>();
        if d <= 0.0 || !d.is_finite() {
            return None;
        }
        l[j][j] = d.sqrt();
        for i in j + 1..6 {
            l[i][j] = (a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>()) / l[j][j];
        }
    }
    Some(l)
}

/// `b L^-T` for lower-triangular `l`.
fn right_solve(b: &Mat6, l: &Mat6) -> Mat6 {
    let mut x = ZERO6;
    for r in 0..6 {
        for j in 0..6 {
            let s: f64 = (0..j).map(|k| x[r][k] * l[j][k]).sum();
            x[r][j] = (b[r][j] - s) / l[j][j];
        }
    }
    x
}

/// `a -= b c^T`.
fn subtract_abt(a: &mut Mat6, b: &Mat6, c: &Mat6) {
    for (ai, bi) in a.iter_mut().zip(b) {
        for (aij, cj) in ai.iter_mut().zip(c) {
            *aij -= bi[0] * cj[0]
                + bi[1] * cj[1]
                + bi[2] * cj[2]
                + bi[3] * cj[3]
                + bi[4] * cj[4]
                + bi[5] * cj[5];
        }
    }
}

/// Factor in place (`diagonal` and `below` become `L`); false if the matrix
/// is not positive definite.
fn factor(m: &mut BlockMatrix) -> bool {
    for k in 0..m.diagonal.len() {
        let Some(l) = cholesky6(&m.diagonal[k]) else {
            return false;
        };
        m.diagonal[k] = l;
        let mut column = std::mem::take(&mut m.below[k]);
        for (_, block) in &mut column {
            *block = right_solve(block, &l);
        }
        for (a, (i, li)) in column.iter().enumerate() {
            for (j, lj) in &column[..=a] {
                subtract_abt(m.block(*i, *j), li, lj);
            }
        }
        m.below[k] = column;
    }
    true
}

/// Solve `L L^T x = b` in place.
fn substitute(m: &BlockMatrix, x: &mut [Vec6]) {
    let n = m.diagonal.len();
    for k in 0..n {
        let l = &m.diagonal[k];
        for i in 0..6 {
            let s: f64 = (0..i).map(|j| l[i][j] * x[k][j]).sum();
            x[k][i] = (x[k][i] - s) / l[i][i];
        }
        let xk = x[k];
        for (row, block) in &m.below[k] {
            for i in 0..6 {
                x[*row][i] -= (0..6).map(|j| block[i][j] * xk[j]).sum::<f64>();
            }
        }
    }
    for k in (0..n).rev() {
        let mut v = x[k];
        for (row, block) in &m.below[k] {
            for j in 0..6 {
                v[j] -= (0..6).map(|i| block[i][j] * x[*row][i]).sum::<f64>();
            }
        }
        let l = &m.diagonal[k];
        for i in (0..6).rev() {
            let s: f64 = (i + 1..6).map(|j| l[j][i] * v[j]).sum();
            v[i] = (v[i] - s) / l[i][i];
        }
        x[k] = v;
    }
}

/// Numerical Jacobians of an edge's residual with respect to perturbations
/// of its two poses.
fn jacobians(xi: &Rigid, xj: &Rigid, z_inv: &Rigid) -> (Mat6, Mat6) {
    const H: f64 = 1e-6;
    let mut ji = ZERO6;
    let mut jj = ZERO6;
    for c in 0..6 {
        let mut d = [0.0; 6];
        d[c] = H;
        let plus_i = residual(&retract(xi, &d), xj, z_inv);
        let plus_j = residual(xi, &retract(xj, &d), z_inv);
        d[c] = -H;
        let minus_i = residual(&retract(xi, &d), xj, z_inv);
        let minus_j = residual(xi, &retract(xj, &d), z_inv);
        for r in 0..6 {
            ji[r][c] = (plus_i[r] - minus_i[r]) / (2.0 * H);
            jj[r][c] = (plus_j[r] - minus_j[r]) / (2.0 * H);
        }
    }
    (ji, jj)
}

/// `a^T w b`.
fn at_w_b(a: &Mat6, w: &Mat6, b: &Mat6) -> Mat6 {
    let wb: Mat6 =
        std::array::from_fn(|i| std::array::from_fn(|j| (0..6).map(|k| w[i][k] * b[k][j]).sum()));
    std::array::from_fn(|i| std::array::from_fn(|j| (0..6).map(|k| a[k][i] * wb[k][j]).sum()))
}

fn at_w_e(a: &Mat6, w: &Mat6, e: &Vec6) -> Vec6 {
    let we: Vec6 = std::array::from_fn(|i| (0..6).map(|k| w[i][k] * e[k]).sum());
    std::array::from_fn(|i| (0..6).map(|k| a[k][i] * we[k]).sum())
}

/// Optimise the non-fixed poses in place. With no fixed node the first one
/// is held. Returns `None` if the graph has no free node.
pub fn optimize(graph: &mut PoseGraph, params: &OptimizeParams) -> Option<OptimizeReport> {
    let mut fixed: Vec<bool> = graph.nodes.iter().map(|n| n.fixed).collect();
    if !fixed.iter().any(|&f| f) {
        fixed[0] = true;
    }
    let mut variable = vec![usize::MAX; graph.nodes.len()];
    let mut variables = 0;
    for (i, &f) in fixed.iter().enumerate() {
        if !f {
            variable[i] = variables;
            variables += 1;
        }
    }
    // Planes follow the poses; their blocks use three of the six entries.
    let mut plane_variable = vec![usize::MAX; graph.planes.len()];
    for (p, plane) in graph.planes.iter().enumerate() {
        if !plane.fixed {
            plane_variable[p] = variables;
            variables += 1;
        }
    }
    if variables == 0 {
        return None;
    }
    let mut adjacency = vec![BTreeSet::new(); variables];
    let pairs = graph
        .edges
        .iter()
        .map(|e| (variable[e.from], variable[e.to]))
        .chain(
            graph
                .plane_edges
                .iter()
                .map(|e| (variable[e.node], plane_variable[e.plane])),
        );
    for (a, b) in pairs {
        if a != usize::MAX && b != usize::MAX && a != b {
            adjacency[a].insert(b);
            adjacency[b].insert(a);
        }
    }
    let (position, columns) = symbolic(variables, &adjacency);

    let initial_cost = graph.cost(params);
    let mut cost = initial_cost;
    let mut lambda = 1e-4;
    let mut iterations = 0;
    let mut converged = false;
    while iterations < params.max_iterations {
        iterations += 1;
        // Normal equations at the current poses, in factor order.
        let mut h = BlockMatrix {
            diagonal: vec![ZERO6; variables],
            below: columns
                .iter()
                .map(|rows| rows.iter().map(|&r| (r, ZERO6)).collect())
                .collect(),
        };
        let mut g = vec![[0.0; 6]; variables];
        // The unused entries of plane blocks get a unit diagonal (their step stays zero).
        for &p in plane_variable.iter().filter(|&&p| p != usize::MAX) {
            for i in 3..6 {
                h.diagonal[position[p]][i][i] = 1.0;
            }
        }
        let pose_terms = graph.edges.iter().map(|e| {
            let (xi, xj) = (&graph.nodes[e.from].pose, &graph.nodes[e.to].pose);
            let z_inv = inverse(&e.measurement);
            let r = residual(xi, xj, &z_inv);
            let mut w = weight(&e.information);
            let s = robust(quadratic(&r, &w), params.kernel(e.kind)).1;
            w = w.map(|row| row.map(|x| x * s));
            let (ji, jj) = jacobians(xi, xj, &z_inv);
            (r, w, [(variable[e.from], ji), (variable[e.to], jj)])
        });
        let plane_terms = graph.plane_edges.iter().map(|e| {
            let (x, plane) = (
                &graph.nodes[e.node].pose,
                &graph.planes[e.plane].coefficients,
            );
            let r = plane_residual(x, plane, &e.measurement);
            let (jx, jp) = plane_jacobians(x, plane, &e.measurement);
            (
                r,
                plane_weight(&e.information),
                [(variable[e.node], jx), (plane_variable[e.plane], jp)],
            )
        });
        let gravity_terms = graph.gravity_edges.iter().map(|e| {
            let x = &graph.nodes[e.node].pose;
            (
                gravity_residual(x, &graph.gravity, &e.up),
                gravity_weight(&e.information),
                [
                    (variable[e.node], gravity_jacobian(x, &graph.gravity, &e.up)),
                    (usize::MAX, ZERO6),
                ],
            )
        });
        for (r, w, ends) in pose_terms.chain(plane_terms).chain(gravity_terms) {
            for &(a, ref ja) in &ends {
                if a == usize::MAX {
                    continue;
                }
                let pa = position[a];
                let ga = at_w_e(ja, &w, &r);
                for i in 0..6 {
                    g[pa][i] += ga[i];
                }
                for &(b, ref jb) in &ends {
                    if b == usize::MAX || position[b] > pa {
                        continue;
                    }
                    let block = at_w_b(ja, &w, jb);
                    let target = h.block(pa, position[b]);
                    for i in 0..6 {
                        for j in 0..6 {
                            target[i][j] += block[i][j];
                        }
                    }
                }
            }
        }
        // Try damped steps until one lowers the cost.
        let mut improved = false;
        for _ in 0..10 {
            let mut damped = BlockMatrix {
                diagonal: h
                    .diagonal
                    .iter()
                    .map(|d| {
                        let mut d = *d;
                        for (i, row) in d.iter_mut().enumerate() {
                            row[i] += lambda * row[i].max(1e-9);
                        }
                        d
                    })
                    .collect(),
                below: h.below.clone(),
            };
            if !factor(&mut damped) {
                lambda *= 10.0;
                continue;
            }
            let mut step: Vec<Vec6> = g.iter().map(|v| v.map(|x| -x)).collect();
            substitute(&damped, &mut step);
            let mut trial = graph.clone();
            for (i, node) in trial.nodes.iter_mut().enumerate() {
                if variable[i] != usize::MAX {
                    node.pose = retract(&node.pose, &step[position[variable[i]]]);
                }
            }
            for (p, plane) in trial.planes.iter_mut().enumerate() {
                if plane_variable[p] != usize::MAX {
                    plane.coefficients =
                        retract_plane(&plane.coefficients, &step[position[plane_variable[p]]]);
                }
            }
            let trial_cost = trial.cost(params);
            if trial_cost <= cost {
                let gain = cost - trial_cost;
                *graph = trial;
                converged = gain <= params.tolerance * cost.max(f64::MIN_POSITIVE);
                cost = trial_cost;
                lambda = (lambda / 3.0).max(1e-12);
                improved = true;
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            converged = true;
        }
        if converged {
            break;
        }
    }
    Some(OptimizeReport {
        initial_cost,
        final_cost: cost,
        iterations,
        converged,
    })
}

// --- Scans ------------------------------------------------------------------

/// `cloud` with every point moved by `x`.
fn transformed(cloud: &PointCloud, x: &Rigid) -> PointCloud {
    PointCloud {
        positions: cloud.positions.iter().map(|p| x.apply(p)).collect(),
        ..cloud.clone()
    }
}

/// Register the scan of the loop's `to` node onto the scan of its `from`
/// node (each in its own frame), starting from `guess`, the pose of `to`
/// in the frame of `from` (usually [`PoseGraph::relative`]). Returns the
/// refined measurement for [`PoseGraph::add_loop`] and the ICP outcome.
pub fn register_loop(
    from_scan: &PointCloud,
    to_scan: &PointCloud,
    guess: &Rigid,
    params: IcpParams,
) -> Option<(Rigid, IcpResult)> {
    let result = icp(&transformed(to_scan, guess), from_scan, params)?;
    Some((result.transform.compose(guess), result))
}

/// A registration without a usable initial guess, as when joining two
/// graphs recorded separately: `to_scan` starts at `guess` turned about the
/// vertical axis in `yaw_steps` equal steps. Each start is first refined by
/// a quick ICP on a subsample; the two that overlap most (see
/// [`overlap_fitness`]) are refined in full, and the better one wins.
/// Returns the measurement (pose of `to` in the frame of `from`), its
/// fitness and the ICP outcome.
pub fn register_with_yaw_search(
    from_scan: &PointCloud,
    to_scan: &PointCloud,
    guess: &Rigid,
    yaw_steps: usize,
    params: IcpParams,
    inlier_distance: f64,
) -> Option<(Rigid, f64, IcpResult)> {
    const COARSE_SAMPLE: usize = 1500;
    const COARSE_ITERATIONS: usize = 15;
    const REFINED: usize = 2;
    let steps = yaw_steps.max(1);
    // The same scan is scored at every start: find its floor once.
    let floor = scan_floor(to_scan);
    let fitness = |measurement: &Rigid| {
        overlap_fitness_above(from_scan, to_scan, measurement, inlier_distance, floor)
    };
    let coarse = IcpParams {
        sample: params.sample.min(COARSE_SAMPLE),
        max_iterations: params.max_iterations.min(COARSE_ITERATIONS),
        ..params
    };
    let mut starts: Vec<(Rigid, f64)> = (0..steps)
        .filter_map(|k| {
            let yaw = std::f64::consts::TAU * k as f64 / steps as f64;
            let start = guess.compose(&Rigid {
                rotation: exp_so3(&[0.0, 0.0, yaw]),
                translation: [0.0; 3],
            });
            let (measurement, _) = register_loop(from_scan, to_scan, &start, coarse)?;
            Some((measurement, fitness(&measurement)))
        })
        .collect();
    starts.sort_by(|a, b| b.1.total_cmp(&a.1));
    starts
        .into_iter()
        .take(REFINED)
        .filter_map(|(start, _)| {
            let (measurement, result) = register_loop(from_scan, to_scan, &start, params)?;
            Some((measurement, fitness(&measurement), result))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
}

impl PoseGraph {
    /// Add `other`'s nodes and edges, its poses moved by `transform` (into
    /// this graph's frame). Its ids are shifted past this graph's when they
    /// would clash, and its fixed nodes are freed (this graph keeps the
    /// gauge). Returns the index of its first node here.
    pub fn append(&mut self, other: &PoseGraph, transform: &Rigid) -> usize {
        let offset = self.nodes.len();
        let max_id = self.nodes.iter().map(|n| n.id).max();
        let min_other = other.nodes.iter().map(|n| n.id).min().unwrap_or(0);
        let clash = other
            .nodes
            .iter()
            .any(|n| self.nodes.iter().any(|m| m.id == n.id));
        let shift = match max_id {
            Some(max) if clash => max + 1 - min_other,
            _ => 0,
        };
        self.nodes.extend(other.nodes.iter().map(|n| Node {
            id: n.id + shift,
            pose: transform.compose(&n.pose),
            fixed: false,
        }));
        self.edges.extend(other.edges.iter().map(|e| Edge {
            from: e.from + offset,
            to: e.to + offset,
            ..e.clone()
        }));
        let plane_offset = self.planes.len();
        for plane in &other.planes {
            let index = self.add_plane(plane_in_world(&plane.coefficients, transform));
            self.planes[index].fixed = false;
        }
        self.plane_edges
            .extend(other.plane_edges.iter().map(|e| PlaneEdge {
                node: e.node + offset,
                plane: e.plane + plane_offset,
                ..e.clone()
            }));
        if !other.gravity_edges.is_empty() {
            if self.gravity_edges.is_empty() {
                let r = &transform.rotation;
                self.gravity =
                    std::array::from_fn(|i| (0..3).map(|k| r[i][k] * other.gravity[k]).sum());
            }
            self.gravity_edges
                .extend(other.gravity_edges.iter().map(|e| GravityEdge {
                    node: e.node + offset,
                    ..e.clone()
                }));
        }
        offset
    }
}

/// Tie keyframes to one floor plane from what each sees (`seen`: node
/// index and floor in its frame). This removes tilt and height drift where
/// the ground is flat overall. It asserts that the ground does not bend, so
/// on hills it flattens the path: odometry cannot hold a hill against it,
/// and neither can planes per stretch of path, which still assert zero
/// curvature within each (tried on KITTI 09). Returns the new plane's index,
/// or `None` when nothing is seen.
pub fn tie_to_floor(
    graph: &mut PoseGraph,
    seen: &[(usize, [f64; 4])],
    information: [[f64; 3]; 3],
) -> Option<usize> {
    let &(first, measurement) = seen.first()?;
    let plane = graph.add_plane(plane_in_world(&measurement, &graph.nodes[first].pose));
    graph
        .plane_edges
        .extend(seen.iter().map(|&(node, measurement)| PlaneEdge {
            node,
            plane,
            measurement,
            information,
        }));
    Some(plane)
}

/// The floor under a scan (in its frame): the plane with the most points
/// within `threshold` whose normal is at most `max_tilt` (radians) from `up`
/// and that lies below the sensor, refined by a least-squares fit to those
/// points. `None` when fewer than `min_points` support it. The normal points
/// up, so the offset is the sensor's height above the floor.
pub fn detect_floor(
    scan: &PointCloud,
    up: [f64; 3],
    max_tilt: f64,
    threshold: f64,
    min_points: usize,
) -> Option<[f64; 4]> {
    const SAMPLE: usize = 4000;
    const TRIALS: usize = 300;
    let up = unit3(up);
    let step = scan.len().div_ceil(SAMPLE).max(1);
    let points: Vec<[f64; 3]> = scan.positions.iter().step_by(step).copied().collect();
    if points.len() < 3 {
        return None;
    }
    let cos_max = max_tilt.cos();
    let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    let count = |n: &[f64; 3], d: f64| {
        points
            .iter()
            .filter(|p| (dot3(n, p) + d).abs() <= threshold)
            .count()
    };
    let mut best: Option<([f64; 3], f64, usize)> = None;
    for _ in 0..TRIALS {
        let (a, b, c) = (
            points[next(points.len())],
            points[next(points.len())],
            points[next(points.len())],
        );
        let normal = cross3(&sub3(&b, &a), &sub3(&c, &a));
        let length = dot3(&normal, &normal).sqrt();
        if length < 1e-12 {
            continue;
        }
        let mut n = normal.map(|x| x / length);
        if dot3(&n, &up) < 0.0 {
            n = n.map(|x| -x);
        }
        let d = -dot3(&n, &a);
        // Tilted too far, or above the sensor.
        if dot3(&n, &up) < cos_max || d <= 0.0 {
            continue;
        }
        let support = count(&n, d);
        if best.is_none_or(|(_, _, s)| support > s) {
            best = Some((n, d, support));
        }
    }
    let (n, d, _) = best?;
    // Least-squares refit: the direction of least spread of the inliers.
    let inliers: Vec<&[f64; 3]> = scan
        .positions
        .iter()
        .filter(|p| (dot3(&n, p) + d).abs() <= threshold)
        .collect();
    if inliers.len() < min_points {
        return None;
    }
    let k = inliers.len() as f64;
    let mean: [f64; 3] = std::array::from_fn(|i| inliers.iter().map(|p| p[i]).sum::<f64>() / k);
    let mut cov = [[0.0; 3]; 3];
    for p in &inliers {
        let q = sub3(p, &mean);
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += q[i] * q[j] / k;
            }
        }
    }
    let (values, vectors) = crate::icp::symmetric_eigen(cov);
    let smallest = (0..3).min_by(|&a, &b| values[a].total_cmp(&values[b]))?;
    let mut fitted = [
        vectors[0][smallest],
        vectors[1][smallest],
        vectors[2][smallest],
    ];
    if dot3(&fitted, &n) < 0.0 {
        fitted = fitted.map(|x| -x);
    }
    let fitted = unit3(fitted);
    Some([fitted[0], fitted[1], fitted[2], -dot3(&fitted, &mean)])
}

fn sub3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Fraction of `to_scan`'s structure (up to a few thousand points, evenly
/// spread) that lands within `max_distance` of a `from_scan` point once
/// moved by `measurement` (the pose of `to` in the frame of `from`). A loop
/// whose registration converged into the wrong place overlaps little.
///
/// Points of the floor under `to_scan` (a plane under the sensor, up +z)
/// do not count: open ground overlaps wherever two scans are put on it, so
/// with it a registration a street length off could pass. Without a floor,
/// or with too little else, every point counts.
pub fn overlap_fitness(
    from_scan: &PointCloud,
    to_scan: &PointCloud,
    measurement: &Rigid,
    max_distance: f64,
) -> f64 {
    overlap_fitness_above(
        from_scan,
        to_scan,
        measurement,
        max_distance,
        scan_floor(to_scan),
    )
}

/// The floor [`overlap_fitness`] leaves out of a scan, if it has one.
pub fn scan_floor(scan: &PointCloud) -> Option<[f64; 4]> {
    detect_floor(scan, [0.0, 0.0, 1.0], 20f64.to_radians(), 0.15, 100)
}

/// [`overlap_fitness`] with `to_scan`'s floor already found (see
/// [`scan_floor`]), for scoring the same scan many times.
pub fn overlap_fitness_above(
    from_scan: &PointCloud,
    to_scan: &PointCloud,
    measurement: &Rigid,
    max_distance: f64,
    floor: Option<[f64; 4]>,
) -> f64 {
    const SAMPLE: usize = 5000;
    const FLOOR_BAND: f64 = 0.3;
    const MIN_STRUCTURE: usize = 100;
    let Some(tree) = KdTree::new(&from_scan.positions) else {
        return 0.0;
    };
    let off_floor = |p: &&[f64; 3]| {
        floor.is_none_or(|f| (f[0] * p[0] + f[1] * p[1] + f[2] * p[2] + f[3]).abs() > FLOOR_BAND)
    };
    let mut points: Vec<&[f64; 3]> = to_scan.positions.iter().filter(off_floor).collect();
    if points.len() < MIN_STRUCTURE {
        points = to_scan.positions.iter().collect();
    }
    let step = points.len().div_ceil(SAMPLE).max(1);
    let mut guess = None;
    let (mut inside, mut total) = (0usize, 0usize);
    for p in points.into_iter().step_by(step) {
        let hit = tree.nearest(&measurement.apply(p), guess);
        guess = Some(hit);
        total += 1;
        inside += usize::from(hit.distance_sq <= max_distance * max_distance);
    }
    if total == 0 {
        0.0
    } else {
        inside as f64 / total as f64
    }
}

/// How far apart (metres) the graph may put two nodes that are really at
/// the same place: `max_distance`, plus `drift` (a fraction, e.g. 0.03 for
/// odometry that drifts 3 %) of the path travelled between them.
pub fn drift_budget(max_distance: f64, drift: f64, travel_between: f64) -> f64 {
    max_distance + drift * travel_between
}

/// A node pair worth checking for a loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopCandidate {
    pub from: usize,
    pub to: usize,
    /// Path length between them, through the nodes in order.
    pub travel: f64,
}

/// Node pairs `(earlier, later)` worth checking for a loop: at least
/// `min_travel` apart along the path through the nodes (in order), within
/// the [`drift_budget`] of each other as the graph has them now, and not
/// joined by an edge yet. A long way round may have drifted far, so a
/// revisit after a long path is looked for further away. Each node proposes
/// its most plausible earlier node (the smallest distance for its budget),
/// and along the path at most one proposal is kept every `spacing` metres.
pub fn loop_candidates(
    graph: &PoseGraph,
    max_distance: f64,
    drift: f64,
    min_travel: f64,
    spacing: f64,
) -> Vec<LoopCandidate> {
    let positions: Vec<[f64; 3]> = graph.nodes.iter().map(|n| n.pose.translation).collect();
    let Some(tree) = KdTree::new(&positions) else {
        return Vec::new();
    };
    let mut travel = vec![0.0; positions.len()];
    for i in 1..positions.len() {
        let (a, b) = (positions[i - 1], positions[i]);
        travel[i] = travel[i - 1] + dot3(&sub3(&b, &a), &sub3(&b, &a)).sqrt();
    }
    let joined: std::collections::HashSet<(usize, usize)> = graph
        .edges
        .iter()
        .map(|e| (e.from.min(e.to), e.from.max(e.to)))
        .collect();
    let mut out = Vec::new();
    let mut last_kept = f64::NEG_INFINITY;
    let mut near = Vec::new();
    for i in 0..positions.len() {
        if travel[i] - last_kept < spacing {
            continue;
        }
        near.clear();
        // Nothing earlier can be further than the budget for the whole path so far.
        tree.within(
            &positions[i],
            drift_budget(max_distance, drift, travel[i]),
            &mut near,
        );
        let plausibility = |j: usize| {
            let d = sub3(&positions[i], &positions[j]);
            dot3(&d, &d).sqrt() / drift_budget(max_distance, drift, travel[i] - travel[j])
        };
        let best = near
            .iter()
            .copied()
            .filter(|&j| {
                j < i
                    && travel[i] - travel[j] >= min_travel
                    && !joined.contains(&(j, i))
                    && plausibility(j) <= 1.0
            })
            .min_by(|&a, &b| plausibility(a).total_cmp(&plausibility(b)));
        if let Some(j) = best {
            out.push(LoopCandidate {
                from: j,
                to: i,
                travel: travel[i] - travel[j],
            });
            last_kept = travel[i];
        }
    }
    out
}

/// Every scan placed at its node's pose, in one cloud. `scans[i]` belongs
/// to node `i` (in that node's frame); nodes without a scan add nothing.
/// Intensity is kept when every scan has it.
pub fn assemble(graph: &PoseGraph, scans: &[Option<PointCloud>]) -> PointCloud {
    assemble_from(
        graph
            .nodes
            .iter()
            .zip(scans)
            .filter_map(|(n, s)| Some((&n.pose, s.as_ref()?))),
    )
}

/// Scans placed at poses, in one cloud (see [`assemble`]), for any set of
/// nodes: a part of the path, or the nodes near another session.
pub fn assemble_from<'a>(
    parts: impl IntoIterator<Item = (&'a Rigid, &'a PointCloud)>,
) -> PointCloud {
    let parts: Vec<(&Rigid, &PointCloud)> = parts.into_iter().collect();
    let intensity = |s: &PointCloud| match s.attribute(INTENSITY).map(|a| &a.values) {
        Some(AttributeValues::F32(v)) => Some(v.clone()),
        _ => None,
    };
    let with_intensity = parts.iter().all(|(_, s)| intensity(s).is_some());
    let mut out = PointCloud::default();
    let mut values = Vec::new();
    for (pose, scan) in parts {
        out.positions
            .extend(scan.positions.iter().map(|p| pose.apply(p)));
        if with_intensity {
            values.extend(intensity(scan).unwrap_or_default());
        }
    }
    if with_intensity && !values.is_empty() {
        out.attributes.push(Attribute {
            name: INTENSITY.into(),
            values: AttributeValues::F32(values),
        });
    }
    out
}

/// How far each point of the [`assemble`]d map moved between the poses in
/// `before` (one per node, e.g. as loaded) and the graph's poses now, in
/// the map's point order: the true shift of every point, where a
/// cloud-to-cloud distance would miss a slide along a wall.
pub fn correction(graph: &PoseGraph, scans: &[Option<PointCloud>], before: &[Rigid]) -> Vec<f32> {
    correction_from(
        graph
            .nodes
            .iter()
            .zip(scans)
            .zip(before)
            .filter_map(|((n, s), then)| Some((&n.pose, then, s.as_ref()?))),
    )
}

/// [`correction`] for any set of nodes, as `(pose now, pose before, scan)`.
pub fn correction_from<'a>(
    parts: impl IntoIterator<Item = (&'a Rigid, &'a Rigid, &'a PointCloud)>,
) -> Vec<f32> {
    let mut out = Vec::new();
    for (now, then, scan) in parts {
        out.extend(scan.positions.iter().map(|p| {
            let d = sub3(&now.apply(p), &then.apply(p));
            dot3(&d, &d).sqrt() as f32
        }));
    }
    out
}

#[cfg(test)]
mod tests;
