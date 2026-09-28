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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeKind {
    /// Between poses that follow each other.
    Odometry,
    /// Between poses far apart in time: a loop closure.
    Loop,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// The id in the source file (the frame index for trajectories).
    pub id: i64,
    pub pose: Rigid,
    /// Held in place by the optimiser.
    pub fixed: bool,
}

#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub struct Plane {
    /// Its vertex id in g2o files (distinct from the node ids).
    pub id: i64,
    pub coefficients: [f64; 4],
    pub fixed: bool,
}

/// A node's view of a plane: its coefficients in the node's frame, with a
/// 3x3 information on (normal tilt about two axes, offset).
#[derive(Debug, Clone, PartialEq)]
pub struct PlaneEdge {
    pub node: usize,
    pub plane: usize,
    pub measurement: [f64; 4],
    pub information: [[f64; 3]; 3],
}

/// A keyframe's measured up direction (a unit vector in its frame), e.g.
/// from an IMU's roll and pitch, with a 2x2 information on its tilt.
#[derive(Debug, Clone, PartialEq)]
pub struct GravityEdge {
    pub node: usize,
    pub up: [f64; 3],
    pub information: [[f64; 2]; 2],
}

#[derive(Debug, Clone, Default, PartialEq)]
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
fn exp_so3(w: &[f64; 3]) -> Mat3 {
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
fn log_so3(r: &Mat3) -> [f64; 3] {
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
    let present: Vec<(&Node, &PointCloud)> = graph
        .nodes
        .iter()
        .zip(scans)
        .filter_map(|(n, s)| Some((n, s.as_ref()?)))
        .collect();
    let intensity = |s: &PointCloud| match s.attribute(INTENSITY).map(|a| &a.values) {
        Some(AttributeValues::F32(v)) => Some(v.clone()),
        _ => None,
    };
    let with_intensity = present.iter().all(|(_, s)| intensity(s).is_some());
    let mut out = PointCloud::default();
    let mut values = Vec::new();
    for (node, scan) in present {
        out.positions
            .extend(scan.positions.iter().map(|p| node.pose.apply(p)));
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
    let mut out = Vec::new();
    for ((node, scan), then) in graph.nodes.iter().zip(scans).zip(before) {
        let Some(scan) = scan else { continue };
        out.extend(scan.positions.iter().map(|p| {
            let d = sub3(&node.pose.apply(p), &then.apply(p));
            dot3(&d, &d).sqrt() as f32
        }));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(t: [f64; 3], w: [f64; 3]) -> Rigid {
        Rigid {
            rotation: exp_so3(&w),
            translation: t,
        }
    }

    fn close(a: &Rigid, b: &Rigid, tolerance: f64) -> bool {
        let d = inverse(a).compose(b);
        let w = log_so3(&d.rotation);
        d.translation.iter().chain(&w).all(|x| x.abs() < tolerance)
    }

    #[test]
    fn so3_log_inverts_exp() {
        for w in [
            [0.0, 0.0, 0.0],
            [1e-9, -2e-9, 0.0],
            [0.3, -0.2, 0.9],
            [0.0, 0.0, std::f64::consts::PI - 1e-7],
            [2.0, -1.0, 0.5],
        ] {
            let back = log_so3(&exp_so3(&w));
            for i in 0..3 {
                assert!((back[i] - w[i]).abs() < 1e-6, "{w:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn block_cholesky_solves_a_dense_system() {
        // Three variables all connected: the factor fills in completely.
        let adjacency = vec![
            BTreeSet::from([1, 2]),
            BTreeSet::from([0, 2]),
            BTreeSet::from([0, 1]),
        ];
        let (position, columns) = symbolic(3, &adjacency);
        // A = M M^T + n I for a fixed M, 18x18.
        let n = 18;
        let entry = |i: usize, j: usize| ((i * 7 + j * 3) % 11) as f64 - 5.0;
        let a: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        (0..n).map(|k| entry(i, k) * entry(j, k)).sum::<f64>()
                            + if i == j { n as f64 } else { 0.0 }
                    })
                    .collect()
            })
            .collect();
        let mut m = BlockMatrix {
            diagonal: vec![ZERO6; 3],
            below: columns
                .iter()
                .map(|rows| rows.iter().map(|&r| (r, ZERO6)).collect())
                .collect(),
        };
        for vi in 0..3 {
            for vj in 0..3 {
                if position[vj] > position[vi] {
                    continue;
                }
                let block = m.block(position[vi], position[vj]);
                for i in 0..6 {
                    for j in 0..6 {
                        block[i][j] = a[vi * 6 + i][vj * 6 + j];
                    }
                }
            }
        }
        let b: Vec<f64> = (0..n).map(|i| i as f64 - 4.0).collect();
        let mut x = vec![[0.0; 6]; 3];
        for v in 0..3 {
            for i in 0..6 {
                x[position[v]][i] = b[v * 6 + i];
            }
        }
        assert!(factor(&mut m));
        substitute(&m, &mut x);
        for i in 0..n {
            let ax: f64 = (0..n).map(|j| a[i][j] * x[position[j / 6]][j % 6]).sum();
            assert!((ax - b[i]).abs() < 1e-8, "row {i}: {ax} != {}", b[i]);
        }
    }

    /// A square loop of `n` poses per side with noisy odometry.
    fn loop_graph(drift: f64) -> (PoseGraph, Vec<Rigid>) {
        let mut truth = Vec::new();
        for side in 0..4 {
            let yaw = side as f64 * std::f64::consts::FRAC_PI_2;
            for k in 0..10 {
                let s = k as f64;
                let corner = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]][side];
                let (c, sn) = (yaw.cos(), yaw.sin());
                truth.push(pose(
                    [corner[0] + c * s, corner[1] + sn * s, 0.1 * s],
                    [0.0, 0.0, yaw],
                ));
            }
        }
        let info = isotropic_information(0.05, 0.01);
        let mut graph = PoseGraph::from_poses(&truth, info);
        // Corrupt the odometry with a steady yaw and forward bias, then dead-reckon.
        for (k, e) in graph.edges.iter_mut().enumerate() {
            let bias = pose([drift, 0.0, 0.0], [0.0, 0.01 * drift, drift * 0.3]);
            e.measurement = e
                .measurement
                .compose(&bias)
                .compose(&pose([0.0, 0.001 * (k % 3) as f64, 0.0], [0.0; 3]));
        }
        for i in 1..graph.nodes.len() {
            let prev = graph.nodes[i - 1].pose;
            graph.nodes[i].pose = prev.compose(&graph.edges[i - 1].measurement);
        }
        let last = truth.len() - 1;
        graph.edges.push(Edge {
            from: last,
            to: 0,
            measurement: inverse(&truth[last]).compose(&truth[0]),
            information: info,
            kind: EdgeKind::Loop,
        });
        (graph, truth)
    }

    #[test]
    fn a_loop_closure_pulls_the_drifted_end_back() {
        let (mut graph, truth) = loop_graph(0.05);
        let end = graph.nodes.len() - 1;
        let before = inverse(&graph.nodes[end].pose).compose(&truth[end]);
        assert!(
            before.translation.iter().map(|x| x.abs()).sum::<f64>() > 2.0,
            "{before:?}"
        );

        let loop_error_before = *graph.edge_errors().last().unwrap();
        let report = optimize(&mut graph, &OptimizeParams::default()).unwrap();
        assert!(report.converged, "{report:?}");
        // The biased odometry cannot all be met, but the result is a minimum:
        // nudging any free pose in any direction raises the cost.
        assert!(report.final_cost < report.initial_cost * 0.05, "{report:?}");
        let params = OptimizeParams::default();
        let cost = graph.cost(&params);
        for i in (1..graph.nodes.len()).step_by(7) {
            for c in 0..6 {
                for h in [-1e-3, 1e-3] {
                    let mut nudged = graph.clone();
                    let mut d = [0.0; 6];
                    d[c] = h;
                    nudged.nodes[i].pose = retract(&nudged.nodes[i].pose, &d);
                    assert!(nudged.cost(&params) > cost - 1e-9, "node {i} axis {c}");
                }
            }
        }
        assert!(close(&graph.nodes[0].pose, &truth[0], 1e-12));
        let after = inverse(&graph.nodes[end].pose).compose(&truth[end]);
        let err: f64 = after.translation.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(err < 0.5, "end still {err} m off");
        // The loop's error is spread evenly over all edges.
        let errors = graph.edge_errors();
        let loop_error = errors[errors.len() - 1];
        assert!(
            loop_error < loop_error_before / 100.0,
            "{loop_error_before} -> {errors:?}"
        );
        let max = errors.iter().copied().fold(0.0, f64::max);
        assert!(loop_error > max / 2.0, "{errors:?}");
    }

    #[test]
    fn exact_measurements_are_already_optimal() {
        let (_, truth) = loop_graph(0.0);
        let mut graph = PoseGraph::from_poses(&truth, isotropic_information(0.1, 0.1));
        let report = optimize(&mut graph, &OptimizeParams::default()).unwrap();
        assert!(report.initial_cost < 1e-18, "{report:?}");
        for (n, t) in graph.nodes.iter().zip(&truth) {
            assert!(close(&n.pose, t, 1e-9));
        }
    }

    #[test]
    fn huber_kernel_limits_a_wrong_loop() {
        let (clean, truth) = loop_graph(0.0);
        let mut plain = clean.clone();
        // A bogus loop claiming poses 5 and 25 coincide.
        let bogus = Edge {
            from: 5,
            to: 25,
            measurement: Rigid::IDENTITY,
            information: isotropic_information(0.05, 0.01),
            kind: EdgeKind::Loop,
        };
        plain.edges.push(bogus.clone());
        let mut robust = plain.clone();
        optimize(&mut plain, &OptimizeParams::default()).unwrap();
        optimize(
            &mut robust,
            &OptimizeParams {
                loop_kernel: Some(1.0),
                ..OptimizeParams::default()
            },
        )
        .unwrap();
        let error = |g: &PoseGraph| -> f64 {
            g.nodes
                .iter()
                .zip(&truth)
                .map(|(n, t)| {
                    let d = inverse(&n.pose).compose(t).translation;
                    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
                })
                .fold(0.0, f64::max)
        };
        assert!(
            error(&robust) < error(&plain) / 3.0,
            "{} vs {}",
            error(&robust),
            error(&plain)
        );
    }

    #[test]
    fn g2o_round_trips() {
        let (graph, _) = loop_graph(0.02);
        let mut graph = graph;
        graph.nodes[3].fixed = true;
        let text = graph.to_g2o();
        let back = PoseGraph::from_g2o(&text).unwrap();
        assert_eq!(back.nodes.len(), graph.nodes.len());
        assert_eq!(back.edges.len(), graph.edges.len());
        assert!(back.nodes[0].fixed && back.nodes[3].fixed && !back.nodes[1].fixed);
        for (a, b) in back.nodes.iter().zip(&graph.nodes) {
            assert!(close(&a.pose, &b.pose, 1e-12));
        }
        for (a, b) in back.edges.iter().zip(&graph.edges) {
            assert_eq!((a.from, a.to, a.kind), (b.from, b.to, b.kind));
            assert!(close(&a.measurement, &b.measurement, 1e-12));
            assert_eq!(a.information, b.information);
        }
    }

    #[test]
    fn g2o_reads_the_reference_example() {
        let text = "\
# comment
VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1
VERTEX_SE3:QUAT 1 1 0 0 0 0 0.7071067811865476 0.7071067811865476
VERTEX_SE3:QUAT 7 1 1 0 0 0 1 0
EDGE_SE3:QUAT 0 1 1 0 0 0 0 0.7071067811865476 0.7071067811865476 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 4 0 0 4 0 4
EDGE_SE3:QUAT 1 7 1 0 0 0 0 0.7071067811865476 0.7071067811865476 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 4 0 0 4 0 4
VERTEX_SE2 9 0 0 0
FIX 1
";
        let graph = PoseGraph::from_g2o(text).unwrap();
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.nodes[2].id, 7);
        assert!(graph.nodes[1].fixed);
        assert_eq!(graph.edges[0].kind, EdgeKind::Odometry);
        assert_eq!(graph.edges[1].kind, EdgeKind::Loop);
        assert_eq!(graph.edges[1].information[3][3], 4.0);
        // Both edges agree with the vertices.
        for e in graph.edge_errors() {
            assert!(e < 1e-20, "{e}");
        }
        assert!(PoseGraph::from_g2o("EDGE_SE3:QUAT 0 1").is_err());
        assert!(PoseGraph::from_g2o("VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\nEDGE_SE3:QUAT 0 5 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1").is_err());
    }

    /// Points on three walls and a floor of a room, in the room's frame.
    fn room() -> PointCloud {
        let mut positions = Vec::new();
        for a in 0..40 {
            for b in 0..40 {
                let (u, v) = (a as f64 * 0.25, b as f64 * 0.1);
                positions.push([u, 0.0, v]);
                positions.push([0.0, u * 0.8, v]);
                positions.push([u, 8.0, v * 1.2]);
                positions.push([u, u * 0.2 + v, 0.0]);
            }
        }
        PointCloud {
            positions,
            ..PointCloud::default()
        }
    }

    #[test]
    fn register_loop_refines_a_rough_guess() {
        let truth = pose([0.8, -0.4, 0.1], [0.02, -0.01, 0.2]);
        let from_scan = room();
        // Seen from `to`, the room is the `from` scan moved by truth^-1.
        let to_scan = transformed(&from_scan, &inverse(&truth));
        let guess = truth.compose(&pose([0.15, 0.1, 0.0], [0.0, 0.0, 0.03]));
        let (measured, result) =
            register_loop(&from_scan, &to_scan, &guess, IcpParams::default()).unwrap();
        assert!(result.rms_final < 1e-3, "{result:?}");
        assert!(close(&measured, &truth, 1e-3));
    }

    #[test]
    fn assemble_places_scans_at_their_poses() {
        let poses = [
            pose([0.0; 3], [0.0; 3]),
            pose([10.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ];
        let graph = PoseGraph::from_poses(&poses, isotropic_information(1.0, 1.0));
        let scan = |i: f32| PointCloud {
            positions: vec![[1.0, 0.0, 0.0]],
            colors: None,
            attributes: vec![Attribute {
                name: INTENSITY.into(),
                values: AttributeValues::F32(vec![i]),
            }],
        };
        let map = assemble(&graph, &[Some(scan(1.0)), Some(scan(2.0))]);
        assert_eq!(map.len(), 2);
        let p = map.positions[1];
        assert!((p[0] - (10.0 + 1f64.cos())).abs() < 1e-12 && (p[1] - 1f64.sin()).abs() < 1e-12);
        assert_eq!(
            map.attribute(INTENSITY).unwrap().values,
            AttributeValues::F32(vec![1.0, 2.0])
        );
        // A scan without intensity drops the attribute; a missing scan adds nothing.
        let bare = PointCloud {
            positions: vec![[0.0; 3]],
            ..PointCloud::default()
        };
        let map = assemble(&graph, &[Some(bare), None]);
        assert_eq!(map.len(), 1);
        assert!(map.attribute(INTENSITY).is_none());
    }

    #[test]
    fn kitti_and_tum_text() {
        let poses = [
            pose([1.0, 2.0, 3.0], [0.0; 3]),
            pose([0.0; 3], [0.0, 0.0, 0.5]),
        ];
        let graph = PoseGraph::from_poses(&poses, isotropic_information(1.0, 1.0));
        let kitti = graph.to_kitti();
        assert_eq!(kitti.lines().next().unwrap(), "1 0 0 1 0 1 0 2 0 0 1 3");
        let tum = graph.to_tum(&[10.5, 11.5]);
        assert!(
            tum.starts_with(
                "10.5 1 2 3 0 0 0 1
"
            ),
            "{tum}"
        );
        assert!(
            graph
                .to_tum(&[])
                .lines()
                .nth(1)
                .unwrap()
                .starts_with("1 0 0 0 0 0 ")
        );
        use crate::trajectory::{Format, parse};
        let back = parse(&kitti, Format::Kitti).unwrap();
        assert_eq!(back.positions[0], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn loop_candidates_pair_the_ends_of_a_loop() {
        let (_, truth) = loop_graph(0.0);
        let mut graph = PoseGraph::from_poses(&truth, isotropic_information(0.1, 0.1));
        let candidates = loop_candidates(&graph, 3.0, 0.0, 20.0, 5.0);
        assert!(!candidates.is_empty());
        for c in &candidates {
            assert!(
                c.from < 3 && c.to >= 36 && c.travel > 30.0,
                "{candidates:?}"
            );
        }
        // Joined pairs are not proposed again; a short path is never a loop.
        let c = candidates[0];
        graph.add_loop(
            c.from,
            c.to,
            graph.relative(c.from, c.to),
            isotropic_information(0.1, 0.1),
        );
        let again = loop_candidates(&graph, 3.0, 0.0, 20.0, 5.0);
        assert!(!again.iter().any(|d| (d.from, d.to) == (c.from, c.to)));
        assert!(loop_candidates(&graph, 3.0, 0.0, 100.0, 5.0).is_empty());
    }

    #[test]
    fn the_search_widens_with_the_path_travelled() {
        // The end of a drifted loop is a few metres from the start after about 40 m.
        let (graph, _) = loop_graph(0.05);
        let end = graph.nodes.len() - 1;
        let apart = (0..3)
            .map(|j| {
                let d = sub3(
                    &graph.nodes[end].pose.translation,
                    &graph.nodes[j].pose.translation,
                );
                dot3(&d, &d).sqrt()
            })
            .fold(f64::INFINITY, f64::min);
        assert!(apart > 1.0, "{apart}");
        let reaches_start = |c: &[LoopCandidate]| c.iter().any(|c| c.from < 8 && c.to == end);
        assert!(!reaches_start(&loop_candidates(
            &graph, 1.0, 0.0, 20.0, 0.0
        )));
        // Drift of 20 % over the ~39 m path allows 1 + 7.8 m.
        assert!(apart < drift_budget(1.0, 0.2, 38.0), "{apart}");
        assert!(reaches_start(&loop_candidates(&graph, 1.0, 0.2, 20.0, 0.0)));
        assert!((drift_budget(3.0, 0.2, 40.0) - 11.0).abs() < 1e-12);
    }

    #[test]
    fn overlap_fitness_tells_a_good_registration_from_a_bad_one() {
        let truth = pose([0.8, -0.4, 0.1], [0.02, -0.01, 0.2]);
        let from_scan = room();
        let to_scan = transformed(&from_scan, &inverse(&truth));
        assert!(overlap_fitness(&from_scan, &to_scan, &truth, 0.05) > 0.99);
        let wrong = pose([0.0, 3.0, 0.0], [0.0; 3]).compose(&truth);
        assert!(overlap_fitness(&from_scan, &to_scan, &wrong, 0.05) < 0.5);
    }

    #[test]
    fn yaw_search_registers_without_a_guess() {
        let truth = pose([1.5, -0.8, 0.0], [0.0, 0.0, 1.9]);
        let from_scan = room();
        let to_scan = transformed(&from_scan, &inverse(&truth));
        // From the identity, plain ICP cannot turn 109 degrees.
        let (plain, _) =
            register_loop(&from_scan, &to_scan, &Rigid::IDENTITY, IcpParams::default()).unwrap();
        assert!(!close(&plain, &truth, 1e-2));
        let (measured, fitness, _) = register_with_yaw_search(
            &from_scan,
            &to_scan,
            &Rigid::IDENTITY,
            8,
            IcpParams::default(),
            0.05,
        )
        .unwrap();
        assert!(fitness > 0.99, "{fitness}");
        assert!(close(&measured, &truth, 1e-3));
    }

    #[test]
    fn append_moves_and_renumbers_the_other_graph() {
        let info = isotropic_information(1.0, 1.0);
        let mut a =
            PoseGraph::from_poses(&[Rigid::IDENTITY, pose([1.0, 0.0, 0.0], [0.0; 3])], info);
        let mut b =
            PoseGraph::from_poses(&[Rigid::IDENTITY, pose([0.0, 2.0, 0.0], [0.0; 3])], info);
        b.nodes[0].fixed = true;
        let t = pose([10.0, 0.0, 0.0], [0.0, 0.0, std::f64::consts::FRAC_PI_2]);
        let offset = a.append(&b, &t);
        assert_eq!(offset, 2);
        assert_eq!(
            a.nodes.iter().map(|n| n.id).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert!(a.nodes[0].fixed && !a.nodes[2].fixed);
        assert!(close(&a.nodes[3].pose, &t.compose(&b.nodes[1].pose), 1e-12));
        assert_eq!((a.edges[1].from, a.edges[1].to), (2, 3));
        // Moving both ends by the same transform keeps the edge satisfied.
        assert!(a.edge_errors()[1] < 1e-20);
        // Ids that do not clash are kept.
        let mut c = PoseGraph::from_poses(&[Rigid::IDENTITY], info);
        c.nodes[0].id = 100;
        a.append(&c, &Rigid::IDENTITY);
        assert_eq!(a.nodes[4].id, 100);
    }

    #[test]
    fn planes_move_between_frames() {
        let x = pose([1.0, -2.0, 0.5], [0.2, -0.1, 0.7]);
        let world = [0.0, 0.6, 0.8, -3.0];
        let local = plane_in_frame(&world, &x);
        let back = plane_in_world(&local, &x);
        for i in 0..4 {
            assert!((back[i] - world[i]).abs() < 1e-12);
        }
        // A point on the plane stays on it in the pose's frame.
        let on_world = [2.0, 0.0, 3.75];
        let on_local = inverse(&x).apply(&on_world);
        assert!(
            (local[0] * on_local[0] + local[1] * on_local[1] + local[2] * on_local[2] + local[3])
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn detect_floor_finds_the_ground_under_the_sensor() {
        // A slightly tilted floor 1.8 m below the sensor, a wall and a ceiling.
        let mut positions = Vec::new();
        for a in -30..=30 {
            for b in -30..=30 {
                let (x, y) = (a as f64 * 0.3, b as f64 * 0.3);
                positions.push([x, y, -1.8 + 0.02 * x]);
                positions.push([x, 6.0, 1.0 + b as f64 * 0.05]);
                positions.push([x, y, 1.2]);
            }
        }
        let scan = PointCloud {
            positions,
            ..PointCloud::default()
        };
        let floor = detect_floor(&scan, [0.0, 0.0, 1.0], 0.5, 0.05, 100).unwrap();
        let n = unit3([-0.02, 0.0, 1.0]);
        for i in 0..3 {
            assert!((floor[i] - n[i]).abs() < 1e-6, "{floor:?}");
        }
        assert!((floor[3] - 1.8 * n[2]).abs() < 1e-6, "{floor:?}");
        // Looking for a floor along another axis finds none below the sensor.
        assert!(detect_floor(&scan, [1.0, 0.0, 0.0], 0.3, 0.05, 100).is_none());
    }

    #[test]
    fn a_shared_floor_removes_height_drift() {
        // A straight 30-pose drive on flat ground; odometry pitches down a little each step.
        let truth: Vec<Rigid> = (0..30)
            .map(|k| pose([k as f64, 0.0, 0.0], [0.0; 3]))
            .collect();
        let info = isotropic_information(0.05, 0.01);
        let mut graph = PoseGraph::from_poses(&truth, info);
        for e in &mut graph.edges {
            e.measurement = e.measurement.compose(&pose([0.0; 3], [0.0, 0.004, 0.0]));
        }
        for i in 1..graph.nodes.len() {
            graph.nodes[i].pose = graph.nodes[i - 1]
                .pose
                .compose(&graph.edges[i - 1].measurement);
        }
        let height = |g: &PoseGraph| {
            g.nodes
                .iter()
                .map(|n| n.pose.translation[2].abs())
                .fold(0.0, f64::max)
        };
        assert!(height(&graph) > 1.0, "{}", height(&graph));
        // Every pose sees the floor 1.8 m below it.
        let floor = [0.0, 0.0, 1.0, 1.8];
        let seen: Vec<[f64; 4]> = truth.iter().map(|x| plane_in_frame(&floor, x)).collect();
        let plane = graph.add_plane(plane_in_world(&seen[0], &graph.nodes[0].pose));
        for (node, measurement) in seen.iter().enumerate() {
            graph.plane_edges.push(PlaneEdge {
                node,
                plane,
                measurement: *measurement,
                information: plane_information(0.01, 0.02),
            });
        }
        let report = optimize(&mut graph, &OptimizeParams::default()).unwrap();
        assert!(report.converged, "{report:?}");
        // What is left is the compromise with the biased odometry.
        assert!(height(&graph) < 0.1, "{}", height(&graph));
        let c = graph.planes[plane].coefficients;
        assert!(
            (c[2] - 1.0).abs() < 1e-4 && (c[3] - 1.8).abs() < 0.05,
            "{c:?}"
        );
    }

    #[test]
    fn planes_round_trip_through_g2o_and_append() {
        let info = isotropic_information(1.0, 1.0);
        let mut graph =
            PoseGraph::from_poses(&[Rigid::IDENTITY, pose([1.0, 0.0, 0.0], [0.0; 3])], info);
        let plane = graph.add_plane([0.0, 0.0, 1.0, 1.5]);
        assert_eq!(graph.planes[plane].id, 2);
        graph.plane_edges.push(PlaneEdge {
            node: 1,
            plane,
            measurement: [0.0, 0.0, 1.0, 1.5],
            information: plane_information(0.01, 0.02),
        });
        let text = graph.to_g2o();
        assert!(text.contains("VERTEX_PLANE 2 0 0 1 1.5"), "{text}");
        let back = PoseGraph::from_g2o(&text).unwrap();
        assert_eq!(back.planes, graph.planes);
        assert_eq!(back.plane_edges.len(), 1);
        assert_eq!(back.plane_edges[0].node, 1);
        assert!((back.plane_edges[0].information[2][2] - 2500.0).abs() < 1e-9);
        assert!(back.plane_edge_errors()[0] < 1e-20);

        // Appended somewhere else, the plane moves with its graph.
        let mut other = PoseGraph::from_poses(&[Rigid::IDENTITY], info);
        let t = pose([5.0, 0.0, 2.0], [0.0; 3]);
        other.append(&back, &t);
        assert_eq!(other.planes.len(), 1);
        assert!((other.planes[0].coefficients[3] - (1.5 - 2.0)).abs() < 1e-12);
        assert_eq!(other.plane_edges[0].node, 2);
        assert!(other.plane_edge_errors()[0] < 1e-20);
    }

    #[test]
    fn one_tilted_plane_fits_a_constant_grade() {
        let grade = 0.1f64.atan();
        let truth: Vec<Rigid> = (0..40)
            .map(|k| {
                let s = k as f64;
                pose([s * grade.cos(), 0.0, s * grade.sin()], [0.0, -grade, 0.0])
            })
            .collect();
        let seen: Vec<(usize, [f64; 4])> = (0..truth.len())
            .map(|i| (i, [0.0, 0.0, 1.0, 1.8]))
            .collect();
        let mut graph = PoseGraph::from_poses(&truth, isotropic_information(0.05, 0.005));
        let before = graph.cost(&OptimizeParams::default());
        tie_to_floor(&mut graph, &seen, plane_information(0.005, 0.02)).unwrap();
        let errors = graph.plane_edge_errors();
        let worst = errors.iter().copied().fold(0.0, f64::max);
        assert!(
            before < 1e-12 && worst < 1e-12,
            "{before} {worst} {:?}",
            graph.planes
        );
        optimize(&mut graph, &OptimizeParams::default()).unwrap();
        for (n, t) in graph.nodes.iter().zip(&truth) {
            assert!((n.pose.translation[2] - t.translation[2]).abs() < 1e-6);
        }
    }

    #[test]
    fn one_floor_flattens_hills() {
        // Height 5 sin(s / 60) over 300 m; the vehicle pitches with the road.
        let truth: Vec<Rigid> = (0..300)
            .map(|k| {
                let s = k as f64;
                let slope = (5.0 / 60.0 * (s / 60.0).cos()).atan();
                pose([s, 0.0, 5.0 * (s / 60.0).sin()], [0.0, -slope, 0.0])
            })
            .collect();
        let seen: Vec<(usize, [f64; 4])> = (0..truth.len())
            .map(|i| (i, [0.0, 0.0, 1.0, 1.8]))
            .collect();
        let mut graph = PoseGraph::from_poses(&truth, isotropic_information(0.1, 0.0175));
        let plane = tie_to_floor(&mut graph, &seen, plane_information(0.0087, 0.05)).unwrap();
        optimize(&mut graph, &OptimizeParams::default()).unwrap();
        // What the guide warns about: every pose ends up 1.8 m above one
        // plane, so the 10 m of hills are gone.
        let c = graph.planes[plane].coefficients;
        let off = graph
            .nodes
            .iter()
            .map(|n| (dot3(&[c[0], c[1], c[2]], &n.pose.translation) + c[3] - 1.8).abs())
            .fold(0.0, f64::max);
        assert!(off < 0.5, "{off}");
        assert!(tie_to_floor(&mut graph, &[], plane_information(0.01, 0.01)).is_none());
    }

    #[test]
    fn gravity_takes_out_tilt_drift_and_keeps_hills() {
        // Height 5 sin(s / 60) over 300 m; the vehicle pitches with the road.
        let truth: Vec<Rigid> = (0..300)
            .map(|k| {
                let s = k as f64;
                let slope = (5.0 / 60.0 * (s / 60.0).cos()).atan();
                pose([s, 0.0, 5.0 * (s / 60.0).sin()], [0.0, -slope, 0.0])
            })
            .collect();
        let mut graph = PoseGraph::from_poses(&truth, isotropic_information(0.05, 0.004));
        // Odometry that pitches up 0.05 degrees too much every metre.
        for e in &mut graph.edges {
            e.measurement = e
                .measurement
                .compose(&pose([0.0; 3], [0.0, -0.000_873, 0.0]));
        }
        for i in 1..graph.nodes.len() {
            graph.nodes[i].pose = graph.nodes[i - 1]
                .pose
                .compose(&graph.edges[i - 1].measurement);
        }
        let worst = |g: &PoseGraph| {
            g.nodes
                .iter()
                .zip(&truth)
                .map(|(n, t)| (n.pose.translation[2] - t.translation[2]).abs())
                .fold(0.0, f64::max)
        };
        let drifted = worst(&graph);
        assert!(drifted > 20.0, "{drifted}");
        // An IMU sees true up in each body frame.
        let ups: Vec<(usize, [f64; 3])> = truth
            .iter()
            .enumerate()
            .map(|(i, t)| (i, std::array::from_fn(|k| t.rotation[2][k])))
            .collect();
        assert!(tie_to_gravity(
            &mut graph,
            &ups,
            gravity_information(0.002),
            5
        ));
        assert!((graph.gravity[2] - 1.0).abs() < 1e-4, "{:?}", graph.gravity);
        let report = optimize(&mut graph, &OptimizeParams::default()).unwrap();
        assert!(report.converged, "{report:?}");
        let corrected = worst(&graph);
        assert!(corrected < 1.0, "{drifted} -> {corrected}");
        // The hills are kept: the top is still about 5 m up.
        let top = graph
            .nodes
            .iter()
            .map(|n| n.pose.translation[2])
            .fold(f64::MIN, f64::max);
        assert!((top - 5.0).abs() < 1.0, "{top}");
        assert!(!tie_to_gravity(
            &mut graph,
            &[],
            gravity_information(0.01),
            5
        ));
    }

    #[test]
    fn correction_is_each_points_shift() {
        let info = isotropic_information(1.0, 1.0);
        let mut graph =
            PoseGraph::from_poses(&[Rigid::IDENTITY, pose([5.0, 0.0, 0.0], [0.0; 3])], info);
        let before: Vec<Rigid> = graph.nodes.iter().map(|n| n.pose).collect();
        // Node 1 turns a quarter about its own origin.
        graph.nodes[1].pose = pose([5.0, 0.0, 0.0], [0.0, 0.0, std::f64::consts::FRAC_PI_2]);
        let scans = [
            Some(scan_of(&[[0.0; 3]])),
            Some(scan_of(&[[0.0; 3], [2.0, 0.0, 0.0]])),
        ];
        let moved = correction(&graph, &scans, &before);
        // Node 0's point stays; node 1's origin stays, its point 2 m out swings by 2 sqrt(2).
        assert_eq!(moved.len(), 3);
        assert!(moved[0] == 0.0 && moved[1] == 0.0);
        assert!(
            (moved[2] - 2.0 * std::f32::consts::SQRT_2).abs() < 1e-5,
            "{moved:?}"
        );
        assert_eq!(moved.len(), assemble(&graph, &scans).len());
    }

    fn scan_of(points: &[[f64; 3]]) -> PointCloud {
        PointCloud {
            positions: points.to_vec(),
            ..PointCloud::default()
        }
    }
}
