//! A pose graph with a scan per node, for the interactive SLAM editor.

use ca_core::PointCloud;
use ca_core::icp::{IcpParams, Rigid};
use ca_core::pose_graph::{self, EdgeKind, OptimizeParams, PoseGraph};
use wasm_bindgen::prelude::*;

use crate::{Cloud, TrajectoryData};

#[wasm_bindgen]
pub struct PoseGraphSession {
    graph: PoseGraph,
    /// Per node, in the node's frame (voxel-thinned for registration).
    scans: Vec<Option<Scan>>,
    /// Per node, from a TUM trajectory (else empty).
    timestamps: Vec<f64>,
    /// Per node, its pose as loaded (for comparing maps before and after).
    initial: Vec<Rigid>,
    /// Per node, which of its scan's points are dynamic (see
    /// `detect_dynamic`); empty until detected.
    dynamic: Vec<Vec<bool>>,
}

use ca_core::loop_search::{LOOP_SAMPLE, LoopSettings};

/// Register `to_scan` onto `from_scan` from `guess` (the pose of `to` in
/// the frame of `from`). Returns `[rms before, rms after, iterations,
/// converged, fitness, 16 matrix entries, retried, discrepancy]`: the matrix
/// (row-major) is the measured pose of `to` in the frame of `from`, the
/// fitness the fraction of `to`'s points within the inlier distance of
/// `from`'s once placed by it, `retried` 1 when the result came from the
/// retry, and the discrepancy how far (metres) the measured position of
/// `to` lies from the guess. The retry: when the fitness is below
/// `retry_below`, the scans are registered again as if taken at the same
/// place, from `retry_headings` headings (0: no retry). After a long drift
/// the graph's relative pose of a revisit can be tens of metres off, while
/// the two scans were taken a few metres apart.
fn registration(
    from_scan: &PointCloud,
    to_scan: &PointCloud,
    guess: &Rigid,
    settings: &LoopSettings,
) -> Result<Vec<f64>, JsError> {
    let r = ca_core::loop_search::register_pair(from_scan, to_scan, guess, settings)
        .ok_or_else(|| JsError::new("ICP found too few matching points"))?;
    let mut out = vec![
        r.result.rms_initial,
        r.result.rms_final,
        r.result.iterations as f64,
        f64::from(u8::from(r.result.converged)),
        r.fitness,
    ];
    out.extend(r.measurement.to_matrix());
    out.push(f64::from(u8::from(r.retried)));
    out.push(r.discrepancy);
    Ok(out)
}

fn cloud_of(xyz: &[f64]) -> PointCloud {
    PointCloud {
        positions: xyz.as_chunks::<3>().0.to_vec(),
        ..PointCloud::default()
    }
}

/// [`PoseGraphSession::register_loop`] for scans sent to a pool worker:
/// `from` and `to` are interleaved xyz, `guess` a row-major 4x4.
#[wasm_bindgen(js_name = registerScans)]
#[allow(clippy::too_many_arguments)]
pub fn register_scans(
    from: &[f64],
    to: &[f64],
    guess: &[f64],
    max_iterations: usize,
    overlap: f64,
    inlier_distance: f64,
    retry_below: f64,
    retry_headings: usize,
) -> Result<Vec<f64>, JsError> {
    let guess: &[f64; 16] = guess
        .try_into()
        .map_err(|_| JsError::new("guess must have 16 entries"))?;
    let settings = LoopSettings {
        max_iterations,
        overlap,
        point_to_plane: true,
        inlier_distance,
        retry_below,
        retry_headings,
    };
    registration(
        &cloud_of(from),
        &cloud_of(to),
        &Rigid::from_matrix(guess),
        &settings,
    )
}

/// [`PoseGraphSession::detect_dynamic`] on a pool worker, for `count`
/// scans from `first` of a [`PoseGraphSession::dynamic_context`] (which
/// holds `window` more on either side). One flag per point of those
/// scans, scan after scan.
#[wasm_bindgen(js_name = dynamicScans)]
pub fn dynamic_scans(
    context: &[f64],
    first: usize,
    count: usize,
    window: usize,
    margin: f64,
    votes: usize,
) -> Result<Vec<u8>, JsError> {
    use ca_core::dynamic::{VisibilityParams, dynamic_points_of};
    let bad = || JsError::new("malformed dynamic context");
    let n = *context.first().ok_or_else(bad)? as usize;
    let poses_end = 1 + 16 * n;
    let counts = context.get(poses_end..poses_end + n).ok_or_else(bad)?;
    let poses: Vec<Rigid> = context[1..poses_end]
        .as_chunks::<16>()
        .0
        .iter()
        .map(Rigid::from_matrix)
        .collect();
    let mut at = poses_end + n;
    let mut clouds = Vec::with_capacity(n);
    for &c in counts {
        if c < 0.0 {
            clouds.push(None);
            continue;
        }
        let end = at + 3 * c as usize;
        clouds.push(Some(cloud_of(context.get(at..end).ok_or_else(bad)?)));
        at = end;
    }
    let scans: Vec<Option<&PointCloud>> = clouds.iter().map(Option::as_ref).collect();
    let params = VisibilityParams {
        window,
        margin,
        min_see_through: votes,
        ..VisibilityParams::default()
    };
    let flags = dynamic_points_of(&poses, &scans, &params, first..first + count);
    Ok(flags.into_iter().flatten().map(u8::from).collect())
}

/// A keyframe's scan as the session keeps it: positions in the scan's own
/// frame as `f32` (within a sensor's range that is finer than a tenth of a
/// millimetre) and intensity, half the memory of a [`PointCloud`]'s `f64`
/// positions. Joined drives hold tens of millions of scan points, and a
/// WebAssembly instance has 4 GB.
#[derive(Clone)]
struct Scan {
    positions: Vec<[f32; 3]>,
    intensity: Option<Vec<f32>>,
}

impl Scan {
    fn from_cloud(cloud: &PointCloud) -> Self {
        Scan {
            positions: cloud
                .positions
                .iter()
                .map(|p| p.map(|v| v as f32))
                .collect(),
            intensity: match cloud.attribute(ca_core::INTENSITY).map(|a| &a.values) {
                Some(ca_core::AttributeValues::F32(v)) => Some(v.clone()),
                _ => None,
            },
        }
    }

    fn len(&self) -> usize {
        self.positions.len()
    }

    fn point(&self, k: usize) -> [f64; 3] {
        self.positions[k].map(f64::from)
    }

    /// As a cloud, for the core's registration and plane fitting.
    fn to_cloud(&self) -> PointCloud {
        PointCloud {
            positions: self.positions.iter().map(|p| p.map(f64::from)).collect(),
            attributes: self
                .intensity
                .iter()
                .map(|v| ca_core::Attribute {
                    name: ca_core::INTENSITY.into(),
                    values: ca_core::AttributeValues::F32(v.clone()),
                })
                .collect(),
            ..PointCloud::default()
        }
    }
}

fn information(sigma_t: f64, sigma_r_deg: f64) -> [[f64; 6]; 6] {
    pose_graph::isotropic_information(sigma_t, sigma_r_deg.to_radians())
}

impl PoseGraphSession {
    fn scan(&self, index: usize) -> Result<PointCloud, JsError> {
        self.scans
            .get(index)
            .and_then(Option::as_ref)
            .map(Scan::to_cloud)
            .ok_or_else(|| {
                let id = self.graph.nodes.get(index).map_or(index as i64, |n| n.id);
                JsError::new(&format!("node {id} has no scan"))
            })
    }

    fn new(graph: PoseGraph, timestamps: Vec<f64>) -> PoseGraphSession {
        PoseGraphSession {
            scans: vec![None; graph.nodes.len()],
            initial: graph.nodes.iter().map(|n| n.pose).collect(),
            dynamic: Vec::new(),
            graph,
            timestamps,
        }
    }
}

#[wasm_bindgen]
impl PoseGraphSession {
    #[wasm_bindgen(js_name = fromG2o)]
    pub fn from_g2o(text: &str) -> Result<PoseGraphSession, JsError> {
        let graph = PoseGraph::from_g2o(text).map_err(|e| JsError::new(&e.0))?;
        Ok(PoseGraphSession::new(graph, Vec::new()))
    }

    /// A graph without nodes, to add them one at a time (see `addNode`).
    pub fn empty() -> PoseGraphSession {
        PoseGraphSession::new(
            PoseGraph::from_poses(&[], information(1.0, 1.0)),
            Vec::new(),
        )
    }

    /// Add a node at `pose` (row-major 4x4) with id `id`, tied to the last
    /// node by an odometry edge with the given standard deviations; its index.
    #[wasm_bindgen(js_name = addNode)]
    pub fn add_node(
        &mut self,
        pose: &[f64],
        id: f64,
        sigma_t: f64,
        sigma_r_deg: f64,
    ) -> Result<usize, JsError> {
        let m: &[f64; 16] = pose
            .try_into()
            .map_err(|_| JsError::new("a pose needs 16 numbers"))?;
        let pose = Rigid::from_matrix(m);
        let index = self.graph.nodes.len();
        if let Some(last) = self.graph.nodes.last() {
            self.graph.edges.push(pose_graph::Edge {
                from: index - 1,
                to: index,
                measurement: pose_graph::inverse(&last.pose).compose(&pose),
                information: information(sigma_t, sigma_r_deg),
                kind: EdgeKind::Odometry,
            });
        }
        self.graph.nodes.push(pose_graph::Node {
            id: id as i64,
            pose,
            fixed: index == 0,
        });
        self.scans.push(None);
        self.initial.push(pose);
        Ok(index)
    }

    /// An odometry chain through a trajectory's poses (which need
    /// orientations), with the given standard deviations.
    #[wasm_bindgen(js_name = fromTrajectory)]
    pub fn from_trajectory(
        trajectory: &TrajectoryData,
        sigma_t: f64,
        sigma_r_deg: f64,
    ) -> Result<PoseGraphSession, JsError> {
        let t = &trajectory.inner;
        let orientations = t
            .orientations
            .as_ref()
            .ok_or_else(|| JsError::new("the trajectory has no orientations"))?;
        let poses: Vec<Rigid> = t
            .positions
            .iter()
            .zip(orientations)
            .map(|(p, q)| Rigid {
                rotation: ca_core::trajectory::rotation_matrix(q),
                translation: *p,
            })
            .collect();
        let graph = PoseGraph::from_poses(&poses, information(sigma_t, sigma_r_deg));
        Ok(PoseGraphSession::new(graph, t.timestamps.clone()))
    }

    #[wasm_bindgen(getter, js_name = nodeCount)]
    pub fn node_count(&self) -> usize {
        self.graph.nodes.len()
    }

    /// The node ids (frame numbers for trajectories).
    #[wasm_bindgen(js_name = nodeIds)]
    pub fn node_ids(&self) -> Vec<f64> {
        self.graph.nodes.iter().map(|n| n.id as f64).collect()
    }

    /// Row-major 4x4 pose per node, concatenated.
    pub fn poses(&self) -> Vec<f64> {
        self.graph
            .nodes
            .iter()
            .flat_map(|n| n.pose.to_matrix())
            .collect()
    }

    /// `from, to` node indices per edge.
    #[wasm_bindgen(js_name = edgeEnds)]
    pub fn edge_ends(&self) -> Vec<u32> {
        self.graph
            .edges
            .iter()
            .flat_map(|e| [e.from as u32, e.to as u32])
            .collect()
    }

    /// 0 for odometry, 1 for loop edges.
    #[wasm_bindgen(js_name = edgeKinds)]
    pub fn edge_kinds(&self) -> Vec<u8> {
        self.graph
            .edges
            .iter()
            .map(|e| u8::from(e.kind == EdgeKind::Loop))
            .collect()
    }

    /// Squared error of every edge.
    #[wasm_bindgen(js_name = edgeErrors)]
    pub fn edge_errors(&self) -> Vec<f64> {
        self.graph.edge_errors()
    }

    /// Attach a scan to node `index`: moved by `extrinsic` (row-major 4x4,
    /// scan to node frame; empty for none) and voxel-thinned when `voxel`
    /// is positive. Returns the points kept.
    #[wasm_bindgen(js_name = setScan)]
    pub fn set_scan(
        &mut self,
        index: usize,
        cloud: &Cloud,
        voxel: f64,
        extrinsic: &[f64],
    ) -> Result<usize, JsError> {
        if index >= self.scans.len() {
            return Err(JsError::new("no such node"));
        }
        let mut scan = if voxel > 0.0 {
            let keep = ca_core::filter::voxel_subsample(&cloud.inner, voxel);
            cloud.inner.select(&keep)
        } else {
            cloud.inner.clone()
        };
        scan.colors = None;
        if !extrinsic.is_empty() {
            let m: &[f64; 16] = extrinsic
                .try_into()
                .map_err(|_| JsError::new("extrinsic must have 16 entries"))?;
            let e = Rigid::from_matrix(m);
            for p in &mut scan.positions {
                *p = e.apply(p);
            }
        }
        let n = scan.len();
        self.scans[index] = Some(Scan::from_cloud(&scan));
        Ok(n)
    }

    /// Up to `max_points` of node `index`'s scan (every n-th), in its frame.
    #[wasm_bindgen(js_name = scanPositions)]
    pub fn scan_positions(&self, index: usize, max_points: usize) -> Vec<f32> {
        let Some(Some(scan)) = self.scans.get(index) else {
            return Vec::new();
        };
        let step = scan.len().div_ceil(max_points.max(1)).max(1);
        scan.positions
            .iter()
            .step_by(step)
            .flatten()
            .copied()
            .collect()
    }

    /// Optimise; `loop_kernel` <= 0 means no robust kernel. Returns
    /// `[initial cost, final cost, iterations, converged]`.
    pub fn optimize(&mut self, loop_kernel: f64) -> Vec<f64> {
        let params = OptimizeParams {
            loop_kernel: (loop_kernel > 0.0).then_some(loop_kernel),
            ..OptimizeParams::default()
        };
        match pose_graph::optimize(&mut self.graph, &params) {
            Some(r) => vec![
                r.initial_cost,
                r.final_cost,
                r.iterations as f64,
                f64::from(u8::from(r.converged)),
            ],
            None => vec![0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Register node `to`'s scan onto node `from`'s from their current
    /// relative pose, without changing the graph (see [`register_scans`]
    /// for the result and the retry).
    #[wasm_bindgen(js_name = registerLoop)]
    #[allow(clippy::too_many_arguments)]
    pub fn register_loop(
        &self,
        from: usize,
        to: usize,
        max_iterations: usize,
        overlap: f64,
        point_to_plane: bool,
        inlier_distance: f64,
        retry_below: f64,
        retry_headings: usize,
    ) -> Result<Vec<f64>, JsError> {
        let n = self.graph.nodes.len();
        if from >= n || to >= n || from == to {
            return Err(JsError::new("pick two different nodes"));
        }
        let settings = LoopSettings {
            max_iterations,
            overlap,
            point_to_plane,
            inlier_distance,
            retry_below,
            retry_headings,
        };
        registration(
            &self.scan(from)?,
            &self.scan(to)?,
            &self.graph.relative(from, to),
            &settings,
        )
    }

    /// Node `index`'s scan (in its frame) as interleaved xyz, to register it
    /// on another worker with [`register_scans`].
    #[wasm_bindgen(js_name = scanXyz)]
    pub fn scan_xyz(&self, index: usize) -> Result<Vec<f64>, JsError> {
        Ok(self
            .scan(index)?
            .positions
            .iter()
            .flatten()
            .copied()
            .collect())
    }

    /// The pose of node `to` in the frame of node `from` (row-major 4x4).
    #[wasm_bindgen(js_name = relativePose)]
    pub fn relative_pose(&self, from: usize, to: usize) -> Result<Vec<f64>, JsError> {
        let n = self.graph.nodes.len();
        if from >= n || to >= n {
            return Err(JsError::new("no such node"));
        }
        Ok(self.graph.relative(from, to).to_matrix().to_vec())
    }

    /// Add a loop edge measuring `to` in the frame of `from` (row-major
    /// 4x4); returns its index.
    #[wasm_bindgen(js_name = addLoopEdge)]
    pub fn add_loop_edge(
        &mut self,
        from: usize,
        to: usize,
        measurement: &[f64],
        sigma_t: f64,
        sigma_r_deg: f64,
    ) -> Result<usize, JsError> {
        let n = self.graph.nodes.len();
        let m: &[f64; 16] = measurement
            .try_into()
            .map_err(|_| JsError::new("measurement must have 16 entries"))?;
        if from >= n || to >= n || from == to {
            return Err(JsError::new("pick two different nodes"));
        }
        Ok(self.graph.add_loop(
            from,
            to,
            Rigid::from_matrix(m),
            information(sigma_t, sigma_r_deg),
        ))
    }

    /// Join `other` (a graph recorded separately, with its scans) to this
    /// one. Node `b` of `other` is taken to stand near node `a` of this
    /// graph: its scan is registered onto `a`'s with a yaw search, which
    /// places the whole of `other`, and the registration becomes a loop
    /// edge between them. Fails, leaving this graph unchanged, when the
    /// best overlap is below `min_fitness`. Returns `[fitness, rms, edge
    /// index, index of other's first node]`.
    #[allow(clippy::too_many_arguments)]
    pub fn merge(
        &mut self,
        other: &PoseGraphSession,
        a: usize,
        b: usize,
        yaw_steps: usize,
        max_iterations: usize,
        overlap: f64,
        inlier_distance: f64,
        min_fitness: f64,
        sigma_t: f64,
        sigma_r_deg: f64,
    ) -> Result<Vec<f64>, JsError> {
        let a_scan = self
            .scans
            .get(a)
            .and_then(Option::as_ref)
            .map(Scan::to_cloud)
            .ok_or_else(|| JsError::new("the node picked here has no scan"))?;
        let b_scan = other
            .scans
            .get(b)
            .and_then(Option::as_ref)
            .map(Scan::to_cloud)
            .ok_or_else(|| JsError::new("the node picked in the other graph has no scan"))?;
        let params = IcpParams {
            max_iterations,
            overlap,
            sample: LOOP_SAMPLE,
            ..IcpParams::default()
        };
        let (measurement, fitness, result) = pose_graph::register_with_yaw_search(
            &a_scan,
            &b_scan,
            &Rigid::IDENTITY,
            yaw_steps,
            params,
            inlier_distance,
        )
        .ok_or_else(|| JsError::new("ICP found too few matching points"))?;
        if fitness < min_fitness {
            return Err(JsError::new(&format!(
                "the two scans overlap only {:.0} % at best: pick nodes at the same place",
                fitness * 100.0
            )));
        }
        // Place other so that its node b sits at a's pose times the measurement.
        let target = self.graph.nodes[a].pose.compose(&measurement);
        let transform = target.compose(&pose_graph::inverse(&other.graph.nodes[b].pose));
        let offset = self.graph.append(&other.graph, &transform);
        // The other graph as loaded, placed the same way.
        self.initial
            .extend(other.initial.iter().map(|pose| transform.compose(pose)));
        let edge = self.graph.add_loop(
            a,
            offset + b,
            measurement,
            information(sigma_t, sigma_r_deg),
        );
        self.scans.extend(other.scans.iter().cloned());
        // Joined scans have not been judged: detect again.
        self.dynamic.clear();
        if self.timestamps.is_empty() || other.timestamps.is_empty() {
            self.timestamps.clear();
        } else {
            self.timestamps.extend(&other.timestamps);
        }
        Ok(vec![fitness, result.rms_final, edge as f64, offset as f64])
    }

    /// Per node, 1 when the optimiser holds it in place.
    #[wasm_bindgen(js_name = fixedNodes)]
    pub fn fixed_nodes(&self) -> Vec<u8> {
        self.graph.nodes.iter().map(|n| u8::from(n.fixed)).collect()
    }

    #[wasm_bindgen(js_name = setFixed)]
    pub fn set_fixed(&mut self, index: usize, fixed: bool) -> Result<(), JsError> {
        let node = self
            .graph
            .nodes
            .get_mut(index)
            .ok_or_else(|| JsError::new("no such node"))?;
        node.fixed = fixed;
        Ok(())
    }

    /// Put node `index` at `pose` (row-major 4x4). With `carry`, the nodes
    /// after it (in order) make the same motion, so a whole stretch moves.
    #[wasm_bindgen(js_name = setNodePose)]
    pub fn set_node_pose(
        &mut self,
        index: usize,
        pose: &[f64],
        carry: bool,
    ) -> Result<(), JsError> {
        let m: &[f64; 16] = pose
            .try_into()
            .map_err(|_| JsError::new("pose must have 16 entries"))?;
        let old = self
            .graph
            .nodes
            .get(index)
            .ok_or_else(|| JsError::new("no such node"))?
            .pose;
        let new = Rigid::from_matrix(m);
        // The motion in world coordinates: new = motion * old.
        let motion = new.compose(&pose_graph::inverse(&old));
        let end = if carry {
            self.graph.nodes.len()
        } else {
            index + 1
        };
        for node in &mut self.graph.nodes[index..end] {
            node.pose = motion.compose(&node.pose);
        }
        Ok(())
    }

    #[wasm_bindgen(getter, js_name = gravityEdgeCount)]
    pub fn gravity_edge_count(&self) -> usize {
        self.graph.gravity_edges.len()
    }

    /// Tie nodes to gravity: `nodes[i]` measured `ups[3i..3i+3]` as its up
    /// direction (in its frame, e.g. from an IMU's roll and pitch), with
    /// standard deviation `sigma_deg`; with `calibrate`, the IMU's rotation
    /// into the scans' frame is estimated from the drive first (see
    /// `ca_core::pose_graph::set_gravity_calibrated`). Replaces earlier
    /// gravity edges. Returns `[edges added, spread as measured (degrees),
    /// spread with the estimated rotation (NaN when not used), sigma used]`.
    #[wasm_bindgen(js_name = setGravity)]
    pub fn set_gravity(
        &mut self,
        nodes: &[u32],
        ups: &[f64],
        sigma_deg: f64,
        calibrate: bool,
    ) -> Result<Vec<f64>, JsError> {
        if ups.len() != 3 * nodes.len() {
            return Err(JsError::new("one up vector per node expected"));
        }
        let n = self.graph.nodes.len();
        let measured: Vec<(usize, [f64; 3])> = nodes
            .iter()
            .zip(ups.as_chunks::<3>().0)
            .filter(|&(&i, _)| (i as usize) < n)
            .map(|(&i, up)| (i as usize, *up))
            .collect();
        let tie =
            pose_graph::set_gravity_calibrated(&mut self.graph, &measured, sigma_deg, calibrate)
                .ok_or_else(|| JsError::new("no usable up directions"))?;
        Ok(vec![
            tie.tied as f64,
            tie.spread,
            tie.mount.map_or(f64::NAN, |(_, spread)| spread),
            tie.sigma_deg,
        ])
    }

    /// Remove every gravity edge.
    #[wasm_bindgen(js_name = clearGravity)]
    pub fn clear_gravity(&mut self) {
        self.graph.gravity_edges.clear();
    }

    #[wasm_bindgen(getter, js_name = planeCount)]
    pub fn plane_count(&self) -> usize {
        self.graph.planes.len()
    }

    #[wasm_bindgen(getter, js_name = planeEdgeCount)]
    pub fn plane_edge_count(&self) -> usize {
        self.graph.plane_edges.len()
    }

    /// Tie every keyframe whose scan shows a floor to one new floor plane
    /// (see `ca_core::pose_graph::detect_floor` and `tie_to_floor`). `up` is the scans' up
    /// direction, or empty to pick the axis (+z, -y, +y) under which most
    /// of the first scans show a floor. Returns `[first new plane, planes,
    /// keyframes tied, up x, up y, up z]`; fails when no scan shows a floor.
    #[wasm_bindgen(js_name = addFloor)]
    pub fn add_floor(
        &mut self,
        up: &[f64],
        max_tilt_deg: f64,
        threshold: f64,
        min_points: usize,
        sigma_angle_deg: f64,
        sigma_offset: f64,
    ) -> Result<Vec<f64>, JsError> {
        let max_tilt = max_tilt_deg.to_radians();
        let floor = |scan: &Scan, up: [f64; 3]| {
            pose_graph::detect_floor(&scan.to_cloud(), up, max_tilt, threshold, min_points)
        };
        let scans: Vec<(usize, &Scan)> = self
            .scans
            .iter()
            .enumerate()
            .filter_map(|(i, s)| Some((i, s.as_ref()?)))
            .collect();
        let up: [f64; 3] = match up {
            [x, y, z] => [*x, *y, *z],
            [] => {
                let trial = &scans[..scans.len().min(10)];
                [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]
                    .into_iter()
                    .max_by_key(|&up| trial.iter().filter(|(_, s)| floor(s, up).is_some()).count())
                    .unwrap_or([0.0, 0.0, 1.0])
            }
            _ => return Err(JsError::new("up must have 3 entries")),
        };
        let seen: Vec<(usize, [f64; 4])> = scans
            .iter()
            .filter_map(|&(i, s)| Some((i, floor(s, up)?)))
            .collect();
        if seen.is_empty() {
            return Err(JsError::new(
                "no scan shows a floor: check the up axis and the tilt limit",
            ));
        }
        let information = pose_graph::plane_information(sigma_angle_deg.to_radians(), sigma_offset);
        let plane = pose_graph::tie_to_floor(&mut self.graph, &seen, information)
            .ok_or_else(|| JsError::new("no scan shows a floor"))?;
        Ok(vec![
            plane as f64,
            1.0,
            seen.len() as f64,
            up[0],
            up[1],
            up[2],
        ])
    }

    /// Remove plane `index` and every edge to it.
    #[wasm_bindgen(js_name = removePlane)]
    pub fn remove_plane(&mut self, index: usize) -> Result<(), JsError> {
        if index >= self.graph.planes.len() {
            return Err(JsError::new("no such plane"));
        }
        self.graph.planes.remove(index);
        self.graph.plane_edges.retain(|e| e.plane != index);
        for e in &mut self.graph.plane_edges {
            if e.plane > index {
                e.plane -= 1;
            }
        }
        Ok(())
    }

    /// Loop candidates as `earlier, later, path length between` triples
    /// (see `ca_core::pose_graph::loop_candidates`).
    #[wasm_bindgen(js_name = loopCandidates)]
    pub fn loop_candidates(
        &self,
        max_distance: f64,
        drift: f64,
        min_travel: f64,
        spacing: f64,
    ) -> Vec<f64> {
        pose_graph::loop_candidates(&self.graph, max_distance, drift, min_travel, spacing)
            .into_iter()
            .flat_map(|c| [c.from as f64, c.to as f64, c.travel])
            .collect()
    }

    /// Remove edge `index` (later edges shift down).
    #[wasm_bindgen(js_name = removeEdge)]
    pub fn remove_edge(&mut self, index: usize) -> Result<(), JsError> {
        if index >= self.graph.edges.len() {
            return Err(JsError::new("no such edge"));
        }
        self.graph.edges.remove(index);
        Ok(())
    }

    /// Edge `index` as `[from, to, kind (0 odometry, 1 loop), 16 matrix
    /// entries (row-major), 36 information entries]`, for [`Self::insert_edge`].
    #[wasm_bindgen(js_name = edgeData)]
    pub fn edge_data(&self, index: usize) -> Result<Vec<f64>, JsError> {
        let e = self
            .graph
            .edges
            .get(index)
            .ok_or_else(|| JsError::new("no such edge"))?;
        let mut out = vec![
            e.from as f64,
            e.to as f64,
            f64::from(u8::from(e.kind == EdgeKind::Loop)),
        ];
        out.extend(e.measurement.to_matrix());
        out.extend(e.information.iter().flatten());
        Ok(out)
    }

    /// Put an edge from [`Self::edge_data`] back at `index`.
    #[wasm_bindgen(js_name = insertEdge)]
    pub fn insert_edge(&mut self, index: usize, data: &[f64]) -> Result<(), JsError> {
        let n = self.graph.nodes.len();
        let (from, to) = (data.first().copied(), data.get(1).copied());
        let (Some(from), Some(to)) = (from, to) else {
            return Err(JsError::new("bad edge data"));
        };
        if data.len() != 3 + 16 + 36 || from as usize >= n || to as usize >= n {
            return Err(JsError::new("bad edge data"));
        }
        if index > self.graph.edges.len() {
            return Err(JsError::new("no such edge position"));
        }
        let matrix: &[f64; 16] = data[3..19].try_into().expect("16 entries");
        let information = std::array::from_fn(|i| std::array::from_fn(|j| data[19 + i * 6 + j]));
        self.graph.edges.insert(
            index,
            ca_core::pose_graph::Edge {
                from: from as usize,
                to: to as usize,
                measurement: Rigid::from_matrix(matrix),
                information,
                kind: if data[2] == 1.0 {
                    EdgeKind::Loop
                } else {
                    EdgeKind::Odometry
                },
            },
        );
        Ok(())
    }

    /// Replace every pose (row-major 4x4 each), e.g. to undo an optimisation.
    #[wasm_bindgen(js_name = setPoses)]
    pub fn set_poses(&mut self, poses: &[f64]) -> Result<(), JsError> {
        if poses.len() != 16 * self.graph.nodes.len() {
            return Err(JsError::new("one 4x4 matrix per node expected"));
        }
        for (node, m) in self.graph.nodes.iter_mut().zip(poses.as_chunks::<16>().0) {
            node.pose = Rigid::from_matrix(m);
        }
        Ok(())
    }

    /// The graph as text: `"g2o"`, `"kitti"` or `"tum"`.
    pub fn export(&self, format: &str) -> Result<String, JsError> {
        match format {
            "g2o" => Ok(self.graph.to_g2o()),
            "kitti" => Ok(self.graph.to_kitti()),
            "tum" => Ok(self.graph.to_tum(&self.timestamps)),
            other => Err(JsError::new(&format!("unknown pose format {other:?}"))),
        }
    }

    /// Every scan at its node's pose as one cloud, or with `initial` at its
    /// pose as loaded, thinned to one point per `voxel` (0 keeps every
    /// point); call [`Cloud::build_index`] before drawing it. With
    /// `correction`, each point carries how far it moved from its place as
    /// loaded (the `correction` attribute, metres). Only `nodes` (node
    /// indices) are included, or every node when it is empty.
    pub fn map(
        &self,
        initial: bool,
        correction: bool,
        nodes: &[u32],
        part: u8,
        voxel: f64,
    ) -> Result<Cloud, JsError> {
        // Only `nodes` (all when empty): a part of the path, or the stretch
        // near another session. `part` 1 keeps the static points only, 2 the
        // dynamic ones (after `detect_dynamic`).
        let n = self.graph.nodes.len();
        let chosen: Vec<usize> = if nodes.is_empty() {
            (0..n).collect()
        } else {
            nodes
                .iter()
                .map(|&i| i as usize)
                .filter(|&i| i < n)
                .collect()
        };
        if part != 0 && self.dynamic.len() != n {
            return Err(JsError::new("find the dynamic points first"));
        }
        let parts = || {
            chosen.iter().filter_map(|&i| {
                let scan = self.scans[i].as_ref()?;
                Some((i, &self.graph.nodes[i].pose, &self.initial[i], scan))
            })
        };
        // Every point, or (`part`) the static or the dynamic ones.
        let keep = |i: usize, k: usize| {
            part == 0 || self.dynamic[i].get(k).copied().unwrap_or(false) == (part == 2)
        };
        // Thinned to one point per `voxel` as it is assembled, exactly as
        // `voxel_subsample` would the whole map, which never has to exist.
        let mut thin = (voxel > 0.0).then(|| {
            let mut lo = [f64::INFINITY; 3];
            for (i, now, then, scan) in parts() {
                let pose = if initial { then } else { now };
                for k in (0..scan.len()).filter(|&k| keep(i, k)) {
                    let p = pose.apply(&scan.point(k));
                    for a in 0..3 {
                        lo[a] = lo[a].min(p[a]);
                    }
                }
            }
            ca_core::filter::VoxelFilter::new(lo, voxel, 1 << 16)
        });
        let mut map = ca_core::PointCloud::default();
        let mut intensity = Vec::new();
        let mut moved = Vec::new();
        let with_intensity = parts().all(|(.., scan)| scan.intensity.is_some());
        for (i, now, then, scan) in parts() {
            let pose = if initial { then } else { now };
            for k in 0..scan.len() {
                if !keep(i, k) {
                    continue;
                }
                let p = scan.point(k);
                let placed = pose.apply(&p);
                if let Some(filter) = &mut thin
                    && !filter.keep(&placed)
                {
                    continue;
                }
                map.positions.push(placed);
                if let (true, Some(v)) = (with_intensity, &scan.intensity) {
                    intensity.push(v[k]);
                }
                if correction && part == 0 {
                    let (a, b) = (now.apply(&p), then.apply(&p));
                    moved.push(
                        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2))
                            .sqrt() as f32,
                    );
                }
            }
        }
        if map.is_empty() {
            return Err(JsError::new("no points there"));
        }
        if with_intensity {
            map.attributes.push(ca_core::Attribute {
                name: ca_core::INTENSITY.into(),
                values: ca_core::AttributeValues::F32(intensity),
            });
        }
        if correction && part == 0 {
            map.attributes.push(ca_core::Attribute {
                name: "correction".into(),
                values: ca_core::AttributeValues::F32(moved),
            });
        }
        Ok(Cloud::unindexed(map))
    }

    /// Find the dynamic points of every scan at the current poses (see
    /// `ca_core::dynamic`): judged against the `window` keyframes on either
    /// side, see-through beyond `margin` metres (plus 2 % of the range),
    /// with at least `votes` votes. Returns `[dynamic points, all points]`.
    #[wasm_bindgen(js_name = detectDynamic)]
    pub fn detect_dynamic(&mut self, window: usize, margin: f64, votes: usize) -> Vec<f64> {
        use ca_core::dynamic::{VisibilityParams, dynamic_points};
        let params = VisibilityParams {
            window,
            margin,
            min_see_through: votes,
            ..VisibilityParams::default()
        };
        let poses: Vec<Rigid> = self.graph.nodes.iter().map(|n| n.pose).collect();
        // Only without a worker pool: then the graph is small enough to hold twice.
        let clouds: Vec<Option<PointCloud>> = self
            .scans
            .iter()
            .map(|s| s.as_ref().map(Scan::to_cloud))
            .collect();
        let scans: Vec<Option<&PointCloud>> = clouds.iter().map(Option::as_ref).collect();
        self.dynamic = dynamic_points(&poses, &scans, &params);
        let dynamic = self.dynamic.iter().flatten().filter(|&&d| d).count();
        let total: usize = self.dynamic.iter().map(Vec::len).sum();
        vec![dynamic as f64, total as f64]
    }

    /// Nodes `lo..hi` packed for [`dynamic_scans`] on a pool worker:
    /// `[count, poses (row-major 4x4 each), points per scan, xyz...]`; a
    /// node without a scan has -1 points.
    #[wasm_bindgen(js_name = dynamicContext)]
    pub fn dynamic_context(&self, lo: usize, hi: usize) -> Vec<f64> {
        let hi = hi.min(self.graph.nodes.len());
        let lo = lo.min(hi);
        let points: usize = (lo..hi)
            .filter_map(|i| self.scans.get(i)?.as_ref())
            .map(|s| s.positions.len())
            .sum();
        let mut out = Vec::with_capacity(1 + 17 * (hi - lo) + 3 * points);
        out.push((hi - lo) as f64);
        for node in &self.graph.nodes[lo..hi] {
            out.extend(node.pose.to_matrix());
        }
        for i in lo..hi {
            out.push(
                self.scans
                    .get(i)
                    .and_then(Option::as_ref)
                    .map_or(-1.0, |s| s.positions.len() as f64),
            );
        }
        for i in lo..hi {
            if let Some(scan) = self.scans.get(i).and_then(Option::as_ref) {
                out.extend(scan.positions.iter().flatten().map(|&v| f64::from(v)));
            }
        }
        out
    }

    /// Take the dynamic points found for nodes from `first` on (one flag
    /// per point, scan after scan, as [`dynamic_scans`] returns them).
    /// Returns how many of them are dynamic.
    #[wasm_bindgen(js_name = setDynamic)]
    pub fn set_dynamic(&mut self, first: usize, flags: &[u8]) -> Result<usize, JsError> {
        let n = self.graph.nodes.len();
        if self.dynamic.len() != n {
            self.dynamic = vec![Vec::new(); n];
        }
        let mut at = 0;
        for i in first..n {
            if at >= flags.len() {
                break;
            }
            let len = self
                .scans
                .get(i)
                .and_then(Option::as_ref)
                .map_or(0, |s| s.positions.len());
            let part = flags
                .get(at..at + len)
                .ok_or_else(|| JsError::new("dynamic flags do not match the scans"))?;
            self.dynamic[i] = part.iter().map(|&f| f != 0).collect();
            at += len;
        }
        Ok(flags.iter().filter(|&&f| f != 0).count())
    }
}
