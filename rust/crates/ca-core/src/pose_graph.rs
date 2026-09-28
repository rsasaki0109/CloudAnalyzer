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

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};

use crate::icp::Rigid;
use crate::trajectory::{quaternion, rotation_matrix};

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

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PoseGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
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

fn inverse(x: &Rigid) -> Rigid {
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
    /// Read `VERTEX_SE3:QUAT`, `EDGE_SE3:QUAT` and `FIX` lines; other
    /// element types are skipped. Edges between consecutive ids are
    /// odometry, the rest loops.
    pub fn from_g2o(text: &str) -> Result<PoseGraph, PoseGraphError> {
        let mut graph = PoseGraph::default();
        let mut index = HashMap::new();
        let mut edges = Vec::new();
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
        for v in fixed {
            if let Some(&i) = index.get(&v) {
                graph.nodes[i].fixed = true;
            }
        }
        Ok(graph)
    }

    pub fn to_g2o(&self) -> String {
        let mut out = String::new();
        for node in &self.nodes {
            out += &format!("VERTEX_SE3:QUAT {} {}\n", node.id, pose_text(&node.pose));
        }
        for node in self.nodes.iter().filter(|n| n.fixed) {
            out += &format!("FIX {}\n", node.id);
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
        out
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
        PoseGraph { nodes, edges }
    }

    /// Relative pose of node `to` in the frame of node `from`, as the graph
    /// currently has it.
    pub fn relative(&self, from: usize, to: usize) -> Rigid {
        inverse(&self.nodes[from].pose).compose(&self.nodes[to].pose)
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
        self.edge_errors()
            .iter()
            .zip(&self.edges)
            .map(|(&chi2, e)| robust(chi2, params.kernel(e.kind)).0)
            .sum()
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
    if variables == 0 {
        return None;
    }
    let mut adjacency = vec![BTreeSet::new(); variables];
    for e in &graph.edges {
        let (a, b) = (variable[e.from], variable[e.to]);
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
        for e in &graph.edges {
            let (xi, xj) = (&graph.nodes[e.from].pose, &graph.nodes[e.to].pose);
            let z_inv = inverse(&e.measurement);
            let r = residual(xi, xj, &z_inv);
            let mut w = weight(&e.information);
            let s = robust(quadratic(&r, &w), params.kernel(e.kind)).1;
            w = w.map(|row| row.map(|x| x * s));
            let (ji, jj) = jacobians(xi, xj, &z_inv);
            let ends = [(variable[e.from], &ji), (variable[e.to], &jj)];
            for &(a, ja) in &ends {
                if a == usize::MAX {
                    continue;
                }
                let pa = position[a];
                let ga = at_w_e(ja, &w, &r);
                for i in 0..6 {
                    g[pa][i] += ga[i];
                }
                for &(b, jb) in &ends {
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
}
