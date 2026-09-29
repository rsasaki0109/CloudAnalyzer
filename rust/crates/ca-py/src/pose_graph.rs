//! A pose graph with its keyframes' scans, as the web app's pose graph
//! panel has it: loops found by ICP, IMU gravity, optimisation, dynamic
//! points by visibility, and the map.

use ca_core::icp::Rigid;
use ca_core::loop_search::{FindLoopsParams, find_loops};
use ca_core::pose_graph::{self, OptimizeParams, PoseGraph as Graph};
use ca_core::{Attribute, AttributeValues, INTENSITY, PointCloud};
use numpy::ndarray::Array3;
use numpy::{
    IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn rigid(m: &[[f64; 4]; 4]) -> Rigid {
    let mut flat = [0.0; 16];
    for r in 0..4 {
        flat[4 * r..4 * r + 4].copy_from_slice(&m[r]);
    }
    Rigid::from_matrix(&flat)
}

/// A pose graph and one scan per node (in the node's frame).
#[pyclass(module = "cloudanalyzer_core._core")]
pub struct PoseGraph {
    graph: Graph,
    scans: Vec<Option<PointCloud>>,
    /// Per node, its pose as loaded.
    initial: Vec<Rigid>,
    /// Per node, which scan points are dynamic (after `detect_dynamic`).
    dynamic: Vec<Vec<bool>>,
}

impl PoseGraph {
    fn wrap(graph: Graph) -> Self {
        let n = graph.nodes.len();
        PoseGraph {
            initial: graph.nodes.iter().map(|n| n.pose).collect(),
            graph,
            scans: vec![None; n],
            dynamic: Vec::new(),
        }
    }

    fn check(&self, index: usize) -> PyResult<()> {
        if index < self.graph.nodes.len() {
            Ok(())
        } else {
            Err(PyIndexError::new_err(format!("no node {index}")))
        }
    }
}

#[pymethods]
impl PoseGraph {
    /// A chain of odometry edges through ``poses`` ((N, 4, 4), sensor to
    /// world), each with standard deviations ``sigma_t`` (m) and ``sigma_r_deg``.
    #[staticmethod]
    #[pyo3(signature = (poses, sigma_t = 0.05, sigma_r_deg = 0.25))]
    fn from_poses(poses: PyReadonlyArray3<f64>, sigma_t: f64, sigma_r_deg: f64) -> PyResult<Self> {
        let shape = poses.shape();
        if shape[1] != 4 || shape[2] != 4 {
            return Err(PyValueError::new_err(format!(
                "expected (N, 4, 4) poses, got {shape:?}"
            )));
        }
        let view = poses.as_array();
        let list: Vec<Rigid> = view
            .outer_iter()
            .map(|m| rigid(&std::array::from_fn(|r| std::array::from_fn(|c| m[[r, c]]))))
            .collect();
        let information = pose_graph::isotropic_information(sigma_t, sigma_r_deg.to_radians());
        Ok(Self::wrap(Graph::from_poses(&list, information)))
    }

    /// A graph from g2o text (``VERTEX_SE3:QUAT``, ``EDGE_SE3:QUAT``, planes, ``FIX``).
    #[staticmethod]
    fn from_g2o(text: &str) -> PyResult<Self> {
        Graph::from_g2o(text)
            .map(Self::wrap)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.graph.nodes.len()
    }

    /// Vertex ids (g2o) or frame numbers, by node index.
    #[getter]
    fn node_ids(&self) -> Vec<i64> {
        self.graph.nodes.iter().map(|n| n.id).collect()
    }

    #[getter]
    fn loop_count(&self) -> usize {
        self.graph
            .edges
            .iter()
            .filter(|e| e.to != e.from + 1)
            .count()
    }

    #[getter]
    fn edge_count(&self) -> usize {
        self.graph.edges.len()
    }

    /// Node ``index``'s scan: ``positions`` (M, 3) in its frame and optional
    /// ``intensity`` (M,), thinned to one point per ``voxel`` (0 keeps all).
    /// Returns the points kept.
    #[pyo3(signature = (index, positions, intensity = None, voxel = 0.4))]
    fn set_scan(
        &mut self,
        py: Python<'_>,
        index: usize,
        positions: PyReadonlyArray2<f64>,
        intensity: Option<PyReadonlyArray1<f32>>,
        voxel: f64,
    ) -> PyResult<usize> {
        self.check(index)?;
        let mut cloud = PointCloud {
            positions: crate::points(&positions)?,
            ..PointCloud::default()
        };
        if let Some(values) = intensity {
            let values = values.as_slice()?.to_vec();
            if values.len() != cloud.positions.len() {
                return Err(PyValueError::new_err("one intensity per point expected"));
            }
            cloud.attributes.push(Attribute {
                name: INTENSITY.into(),
                values: AttributeValues::F32(values),
            });
        }
        let scan = py.detach(|| {
            if voxel > 0.0 {
                cloud.select(&ca_core::filter::voxel_subsample(&cloud, voxel))
            } else {
                cloud
            }
        });
        let n = scan.len();
        self.scans[index] = Some(scan);
        self.dynamic.clear();
        Ok(n)
    }

    /// Join ``other`` (a drive recorded separately, with its scans) to this
    /// graph: its node ``there`` is taken to stand near node ``here`` of this
    /// one, their scans are registered with a yaw search (``yaw_steps``
    /// headings), which places the whole of ``other``, and the registration
    /// becomes a loop edge. ``other``'s nodes follow this graph's, from
    /// ``offset``. Fails, leaving this graph as it was, below ``min_fitness``.
    /// Returns ``{"fitness", "rms", "offset"}``.
    #[pyo3(signature = (
        other, here, there, yaw_steps = 8, max_iterations = 50, overlap = 0.8, inlier_distance = 0.5,
        min_fitness = 0.5, sigma_t = 0.1, sigma_r_deg = 1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn join<'py>(
        &mut self,
        py: Python<'py>,
        other: PyRef<'_, PoseGraph>,
        here: usize,
        there: usize,
        yaw_steps: usize,
        max_iterations: usize,
        overlap: f64,
        inlier_distance: f64,
        min_fitness: f64,
        sigma_t: f64,
        sigma_r_deg: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.check(here)?;
        let a_scan = self.scans[here]
            .as_ref()
            .ok_or_else(|| PyValueError::new_err(format!("node {here} here has no scan")))?;
        let b_scan = other
            .scans
            .get(there)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                PyValueError::new_err(format!("node {there} of the other graph has no scan"))
            })?;
        let params = ca_core::icp::IcpParams {
            max_iterations,
            overlap,
            sample: ca_core::loop_search::LOOP_SAMPLE,
            ..ca_core::icp::IcpParams::default()
        };
        let (measurement, fitness, result) = py
            .detach(|| {
                pose_graph::register_with_yaw_search(
                    a_scan,
                    b_scan,
                    &Rigid::IDENTITY,
                    yaw_steps,
                    params,
                    inlier_distance,
                )
            })
            .ok_or_else(|| PyValueError::new_err("ICP found too few matching points"))?;
        if fitness < min_fitness {
            return Err(PyValueError::new_err(format!(
                "the two scans overlap only {:.0} % at best: pick nodes at the same place",
                fitness * 100.0
            )));
        }
        // Place other so that its node `there` sits at `here`'s pose times the measurement.
        let target = self.graph.nodes[here].pose.compose(&measurement);
        let transform = target.compose(&pose_graph::inverse(&other.graph.nodes[there].pose));
        let offset = self.graph.append(&other.graph, &transform);
        self.initial
            .extend(other.initial.iter().map(|pose| transform.compose(pose)));
        let information = pose_graph::isotropic_information(sigma_t, sigma_r_deg.to_radians());
        self.graph
            .add_loop(here, offset + there, measurement, information);
        self.scans.extend(other.scans.iter().cloned());
        self.dynamic.clear();
        let out = PyDict::new(py);
        out.set_item("fitness", fitness)?;
        out.set_item("rms", result.rms_final)?;
        out.set_item("offset", offset)?;
        Ok(out)
    }

    /// Find loops (candidates registered with ICP on all cores) and add them
    /// as edges, without optimising. Returns ``{"candidates", "implausible",
    /// "added": [(from, to, fitness, retried)]}``.
    #[pyo3(signature = (
        max_distance = 10.0, drift = 0.03, min_travel = 30.0, spacing = 5.0, min_fitness = 0.5,
        inlier_distance = 0.5, retry_headings = 8, max_iterations = 50, overlap = 0.8,
        sigma_t = 0.1, sigma_r_deg = 1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn find_loops<'py>(
        &mut self,
        py: Python<'py>,
        max_distance: f64,
        drift: f64,
        min_travel: f64,
        spacing: f64,
        min_fitness: f64,
        inlier_distance: f64,
        retry_headings: usize,
        max_iterations: usize,
        overlap: f64,
        sigma_t: f64,
        sigma_r_deg: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let params = FindLoopsParams {
            max_distance,
            drift,
            min_travel,
            spacing,
            min_fitness,
            inlier_distance,
            retry_headings,
            max_iterations,
            overlap,
            sigma_t,
            sigma_r: sigma_r_deg.to_radians(),
        };
        let (graph, scans) = (&mut self.graph, &self.scans);
        let found = py.detach(|| find_loops(graph, scans, &params));
        let out = PyDict::new(py);
        out.set_item("candidates", found.candidates)?;
        out.set_item("implausible", found.implausible)?;
        let added: Vec<(usize, usize, f64, bool)> = found
            .added
            .iter()
            .map(|l| (l.from, l.to, l.fitness, l.retried))
            .collect();
        out.set_item("added", added)?;
        Ok(out)
    }

    /// Tie ``nodes`` to the up directions they measured (``ups`` (K, 3), in
    /// each node's frame, e.g. from an IMU), with standard deviation
    /// ``sigma_deg``; replaces earlier ties. Returns the nodes tied.
    #[pyo3(signature = (nodes, ups, sigma_deg = 0.1))]
    fn set_gravity(
        &mut self,
        nodes: Vec<usize>,
        ups: PyReadonlyArray2<f64>,
        sigma_deg: f64,
    ) -> PyResult<usize> {
        let ups = crate::points(&ups)?;
        if ups.len() != nodes.len() {
            return Err(PyValueError::new_err("one up vector per node expected"));
        }
        let n = self.graph.nodes.len();
        let measured: Vec<(usize, [f64; 3])> =
            nodes.into_iter().zip(ups).filter(|&(i, _)| i < n).collect();
        if !pose_graph::set_gravity(&mut self.graph, &measured, sigma_deg.to_radians()) {
            return Err(PyValueError::new_err("no usable up directions"));
        }
        Ok(measured.len())
    }

    /// Optimise; ``loop_kernel`` > 0 puts a Huber kernel on loop edges.
    /// Returns ``{"initial_cost", "final_cost", "iterations", "converged"}``.
    #[pyo3(signature = (loop_kernel = 0.0))]
    fn optimize<'py>(&mut self, py: Python<'py>, loop_kernel: f64) -> PyResult<Bound<'py, PyDict>> {
        let params = OptimizeParams {
            loop_kernel: (loop_kernel > 0.0).then_some(loop_kernel),
            ..OptimizeParams::default()
        };
        let graph = &mut self.graph;
        let report = py
            .detach(|| pose_graph::optimize(graph, &params))
            .ok_or_else(|| PyValueError::new_err("nothing to optimise"))?;
        let out = PyDict::new(py);
        out.set_item("initial_cost", report.initial_cost)?;
        out.set_item("final_cost", report.final_cost)?;
        out.set_item("iterations", report.iterations)?;
        out.set_item("converged", report.converged)?;
        Ok(out)
    }

    /// The poses ((N, 4, 4)), as optimised or, with ``initial``, as loaded.
    #[pyo3(signature = (initial = false))]
    fn poses<'py>(&self, py: Python<'py>, initial: bool) -> Bound<'py, PyArray3<f64>> {
        let list: Vec<Rigid> = if initial {
            self.initial.clone()
        } else {
            self.graph.nodes.iter().map(|n| n.pose).collect()
        };
        let flat: Vec<f64> = list.iter().flat_map(|p| p.to_matrix()).collect();
        Array3::from_shape_vec((list.len(), 4, 4), flat)
            .expect("n x 4 x 4")
            .into_pyarray(py)
    }

    fn to_g2o(&self) -> String {
        self.graph.to_g2o()
    }

    fn to_kitti(&self) -> String {
        self.graph.to_kitti()
    }

    fn to_tum(&self, timestamps: Vec<f64>) -> String {
        self.graph.to_tum(&timestamps)
    }

    /// Find the dynamic points of every scan by visibility (see
    /// ``ca_core::dynamic``), on all cores. Returns ``(dynamic, total)``.
    #[pyo3(signature = (window = 10, margin = 0.5, votes = 3))]
    fn detect_dynamic(
        &mut self,
        py: Python<'_>,
        window: usize,
        margin: f64,
        votes: usize,
    ) -> (usize, usize) {
        use ca_core::dynamic::{VisibilityParams, dynamic_points_of};
        use rayon::prelude::*;
        let params = VisibilityParams {
            window,
            margin,
            min_see_through: votes,
            ..VisibilityParams::default()
        };
        let poses: Vec<Rigid> = self.graph.nodes.iter().map(|n| n.pose).collect();
        let scans: Vec<Option<&PointCloud>> = self.scans.iter().map(Option::as_ref).collect();
        let n = scans.len();
        let chunk = 32;
        self.dynamic = py.detach(|| {
            (0..n.div_ceil(chunk))
                .into_par_iter()
                .flat_map_iter(|k| {
                    dynamic_points_of(&poses, &scans, &params, k * chunk..((k + 1) * chunk).min(n))
                })
                .collect()
        });
        let dynamic = self.dynamic.iter().flatten().filter(|&&d| d).count();
        (dynamic, self.dynamic.iter().map(Vec::len).sum())
    }

    /// Every scan at its pose in one cloud, thinned to one point per
    /// ``voxel`` (0 keeps all): ``{"positions", "intensity"?, "correction"?}``.
    /// ``part`` "static" or "dynamic" keeps those points only (after
    /// ``detect_dynamic``); ``correction`` adds how far each point moved
    /// from its place as loaded.
    /// Only ``nodes`` (node indices) when given: a part of the drive.
    #[pyo3(signature = (voxel = 0.0, part = "all", initial = false, correction = false, nodes = None))]
    fn map<'py>(
        &self,
        py: Python<'py>,
        voxel: f64,
        part: &str,
        initial: bool,
        correction: bool,
        nodes: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let chosen: Vec<bool> = match &nodes {
            Some(list) => {
                let n = self.scans.len();
                let mut chosen = vec![false; n];
                for &i in list.iter().filter(|&&i| i < n) {
                    chosen[i] = true;
                }
                chosen
            }
            None => vec![true; self.scans.len()],
        };
        let want = match part {
            "all" => None,
            "static" => Some(false),
            "dynamic" => Some(true),
            other => {
                return Err(PyValueError::new_err(format!(
                    "part must be all, static or dynamic, not {other:?}"
                )));
            }
        };
        if want.is_some() && self.dynamic.len() != self.scans.len() {
            return Err(PyValueError::new_err("call detect_dynamic first"));
        }
        let (positions, intensity, moved) = py.detach(|| {
            let mut positions = Vec::new();
            let mut intensity = Vec::new();
            let mut moved = Vec::new();
            let with_intensity = self
                .scans
                .iter()
                .flatten()
                .all(|s| s.attribute(INTENSITY).is_some());
            for (i, scan) in self.scans.iter().enumerate() {
                let Some(scan) = scan.as_ref().filter(|_| chosen[i]) else {
                    continue;
                };
                let (now, then) = (&self.graph.nodes[i].pose, &self.initial[i]);
                let pose = if initial { then } else { now };
                let values = match scan.attribute(INTENSITY).map(|a| &a.values) {
                    Some(AttributeValues::F32(v)) => Some(v),
                    _ => None,
                };
                for (k, p) in scan.positions.iter().enumerate() {
                    if want.is_some_and(|w| self.dynamic[i].get(k).copied().unwrap_or(false) != w) {
                        continue;
                    }
                    positions.push(pose.apply(p));
                    if let (true, Some(v)) = (with_intensity, values) {
                        intensity.push(v[k]);
                    }
                    if correction {
                        let (a, b) = (now.apply(p), then.apply(p));
                        moved.push(
                            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2))
                                .sqrt() as f32,
                        );
                    }
                }
            }
            let cloud = PointCloud {
                positions,
                ..PointCloud::default()
            };
            let keep = if voxel > 0.0 {
                ca_core::filter::voxel_subsample(&cloud, voxel)
            } else {
                (0..cloud.len()).collect()
            };
            let pick = |v: &[f32]| {
                (v.len() == cloud.len()).then(|| keep.iter().map(|&k| v[k]).collect::<Vec<f32>>())
            };
            (
                keep.iter()
                    .map(|&k| cloud.positions[k])
                    .collect::<Vec<[f64; 3]>>(),
                pick(&intensity),
                pick(&moved),
            )
        });
        let out = PyDict::new(py);
        out.set_item("positions", crate::to_array2(py, positions))?;
        if let Some(v) = intensity {
            out.set_item("intensity", v.into_pyarray(py))?;
        }
        if let Some(v) = moved {
            out.set_item("correction", v.into_pyarray(py))?;
        }
        Ok(out)
    }
}
