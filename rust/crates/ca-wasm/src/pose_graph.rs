//! A pose graph with a scan per node, for the interactive SLAM editor.

use ca_core::PointCloud;
use ca_core::icp::{IcpMetric, IcpParams, Rigid};
use ca_core::pose_graph::{self, EdgeKind, OptimizeParams, PoseGraph};
use wasm_bindgen::prelude::*;

use crate::{Cloud, TrajectoryData};

#[wasm_bindgen]
pub struct PoseGraphSession {
    graph: PoseGraph,
    /// Per node, in the node's frame (voxel-thinned for registration).
    scans: Vec<Option<PointCloud>>,
    /// Per node, from a TUM trajectory (else empty).
    timestamps: Vec<f64>,
}

fn information(sigma_t: f64, sigma_r_deg: f64) -> [[f64; 6]; 6] {
    pose_graph::isotropic_information(sigma_t, sigma_r_deg.to_radians())
}

impl PoseGraphSession {
    fn new(graph: PoseGraph, timestamps: Vec<f64>) -> PoseGraphSession {
        PoseGraphSession {
            scans: vec![None; graph.nodes.len()],
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
        self.scans[index] = Some(scan);
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
            .flat_map(|p| p.map(|v| v as f32))
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
    /// relative pose, without changing the graph. Returns `[rms before, rms
    /// after, iterations, converged, fitness, 16 matrix entries]`: the
    /// matrix (row-major) is the measured pose of `to` in the frame of
    /// `from`, and the fitness the fraction of `to`'s points within
    /// `inlier_distance` of `from`'s once placed by it.
    #[wasm_bindgen(js_name = registerLoop)]
    pub fn register_loop(
        &self,
        from: usize,
        to: usize,
        max_iterations: usize,
        overlap: f64,
        point_to_plane: bool,
        inlier_distance: f64,
    ) -> Result<Vec<f64>, JsError> {
        let n = self.graph.nodes.len();
        if from >= n || to >= n || from == to {
            return Err(JsError::new("pick two different nodes"));
        }
        let scan = |i: usize| {
            self.scans[i].as_ref().ok_or_else(|| {
                JsError::new(&format!("node {} has no scan", self.graph.nodes[i].id))
            })
        };
        let params = IcpParams {
            metric: if point_to_plane {
                IcpMetric::PointToPlane
            } else {
                IcpMetric::PointToPoint
            },
            max_iterations,
            overlap,
            ..IcpParams::default()
        };
        let (from_scan, to_scan) = (scan(from)?, scan(to)?);
        let guess = self.graph.relative(from, to);
        let (measurement, result) =
            pose_graph::register_loop(from_scan, to_scan, &guess, params)
                .ok_or_else(|| JsError::new("ICP found too few matching points"))?;
        let fitness =
            pose_graph::overlap_fitness(from_scan, to_scan, &measurement, inlier_distance);
        let mut out = vec![
            result.rms_initial,
            result.rms_final,
            result.iterations as f64,
            f64::from(u8::from(result.converged)),
            fitness,
        ];
        out.extend(measurement.to_matrix());
        Ok(out)
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
            .ok_or_else(|| JsError::new("the node picked here has no scan"))?;
        let b_scan = other
            .scans
            .get(b)
            .and_then(Option::as_ref)
            .ok_or_else(|| JsError::new("the node picked in the other graph has no scan"))?;
        let params = IcpParams {
            max_iterations,
            overlap,
            ..IcpParams::default()
        };
        let (measurement, fitness, result) = pose_graph::register_with_yaw_search(
            a_scan,
            b_scan,
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
        let edge = self.graph.add_loop(
            a,
            offset + b,
            measurement,
            information(sigma_t, sigma_r_deg),
        );
        self.scans.extend(other.scans.iter().cloned());
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

    #[wasm_bindgen(getter, js_name = planeCount)]
    pub fn plane_count(&self) -> usize {
        self.graph.planes.len()
    }

    #[wasm_bindgen(getter, js_name = planeEdgeCount)]
    pub fn plane_edge_count(&self) -> usize {
        self.graph.plane_edges.len()
    }

    /// Tie every keyframe whose scan shows a floor to one new floor plane
    /// (see `ca_core::pose_graph::detect_floor`). `up` is the scans' up
    /// direction, or empty to pick the axis (+z, -y, +y) under which most
    /// of the first scans show a floor. Returns `[plane index, keyframes
    /// tied, up x, up y, up z]`; fails when no scan shows a floor.
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
        let floor = |scan: &PointCloud, up: [f64; 3]| {
            pose_graph::detect_floor(scan, up, max_tilt, threshold, min_points)
        };
        let scans: Vec<(usize, &PointCloud)> = self
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
        let Some(&(first, measurement)) = seen.first() else {
            return Err(JsError::new(
                "no scan shows a floor: check the up axis and the tilt limit",
            ));
        };
        let world = pose_graph::plane_in_world(&measurement, &self.graph.nodes[first].pose);
        let plane = self.graph.add_plane(world);
        let information = pose_graph::plane_information(sigma_angle_deg.to_radians(), sigma_offset);
        for &(node, measurement) in &seen {
            self.graph.plane_edges.push(pose_graph::PlaneEdge {
                node,
                plane,
                measurement,
                information,
            });
        }
        Ok(vec![plane as f64, seen.len() as f64, up[0], up[1], up[2]])
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

    /// Loop candidates as `earlier, later` node index pairs (see
    /// `ca_core::pose_graph::loop_candidates`).
    #[wasm_bindgen(js_name = loopCandidates)]
    pub fn loop_candidates(&self, max_distance: f64, min_travel: f64, spacing: f64) -> Vec<u32> {
        pose_graph::loop_candidates(&self.graph, max_distance, min_travel, spacing)
            .into_iter()
            .flat_map(|(a, b)| [a as u32, b as u32])
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

    /// Every scan at its node's pose as one cloud; call
    /// [`Cloud::build_index`] before drawing it.
    pub fn map(&self) -> Result<Cloud, JsError> {
        let map = pose_graph::assemble(&self.graph, &self.scans);
        if map.is_empty() {
            return Err(JsError::new("no scans are loaded"));
        }
        Ok(Cloud::unindexed(map))
    }
}
