//! Python bindings for the CloudAnalyzer Rust core (`cloudanalyzer_core`).
//!
//! Points cross the boundary as `(N, 3)` float64 NumPy arrays. Heavy work
//! releases the GIL and runs on all cores.

use ca_core::icp::{IcpMetric, IcpParams};
use ca_core::{AttributeValues, CLASSIFICATION, INTENSITY, PointCloud, TriangleMesh};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod bag;
mod copc_spatial;
mod odometry;
mod pose_graph;
mod vector_map;

/// Copy an `(N, 3)` array into points.
fn points(array: &PyReadonlyArray2<f64>) -> PyResult<Vec<[f64; 3]>> {
    let shape = array.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "expected an (N, 3) array, got shape {shape:?}"
        )));
    }
    let view = array.as_array();
    Ok(view
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect())
}

fn to_array2<'py, T: numpy::Element>(
    py: Python<'py>,
    rows: Vec<[T; 3]>,
) -> Bound<'py, PyArray2<T>> {
    let n = rows.len();
    let flat: Vec<T> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((n, 3), flat)
        .expect("n x 3 elements")
        .into_pyarray(py)
}

fn read_bytes(path: &str) -> PyResult<Vec<u8>> {
    std::fs::read(path).map_err(|e| PyIOError::new_err(format!("{path}: {e}")))
}

fn io_err(e: ca_core::IoError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Read a point cloud (PLY, PCD, LAS, LAZ, XYZ/TXT/CSV/PTS).
///
/// Returns a dict with ``positions`` ((N, 3) float64) and, when present,
/// ``colors`` ((N, 3) uint8), ``intensity`` ((N,) float32) and
/// ``classification`` ((N,) uint8). ``keep_every`` > 1 keeps every n-th point.
#[pyfunction]
#[pyo3(signature = (path, keep_every = 1))]
fn read<'py>(py: Python<'py>, path: &str, keep_every: usize) -> PyResult<Bound<'py, PyDict>> {
    let bytes = read_bytes(path)?;
    let cloud: PointCloud = py
        .detach(|| ca_core::io::read_thinned(path, &bytes, keep_every))
        .map_err(io_err)?;
    cloud_dict(py, cloud)
}

/// A cloud as ``{"positions", "colors"?, "intensity"?, "classification"?}``.
fn cloud_dict(py: Python<'_>, cloud: PointCloud) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    let PointCloud {
        positions,
        colors,
        attributes,
    } = cloud;
    out.set_item("positions", to_array2(py, positions))?;
    if let Some(colors) = colors {
        out.set_item("colors", to_array2(py, colors))?;
    }
    for a in attributes {
        let key = match a.name.as_str() {
            INTENSITY => "intensity",
            CLASSIFICATION => "classification",
            _ => continue,
        };
        match a.values {
            AttributeValues::F32(v) => out.set_item(key, v.into_pyarray(py))?,
            AttributeValues::U8(v) => out.set_item(key, v.into_pyarray(py))?,
        }
    }
    Ok(out)
}

/// Mesh vertices ((N, 3) float64) and triangle indices ((M, 3) uint32).
type MeshArrays<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<u32>>);

/// Read a triangle mesh (OBJ, STL, PLY with faces) as ``(vertices, triangles)``,
/// or ``None`` if the file is a point cloud.
#[pyfunction]
fn read_mesh<'py>(py: Python<'py>, path: &str) -> PyResult<Option<MeshArrays<'py>>> {
    let bytes = read_bytes(path)?;
    let mesh = py
        .detach(|| ca_core::read_mesh(path, &bytes))
        .map_err(io_err)?;
    Ok(mesh.map(|m| (to_array2(py, m.vertices), to_array2(py, m.triangles))))
}

/// Distance from every ``source`` point to its nearest ``target`` point
/// (cloud-to-cloud), multi-threaded.
#[pyfunction]
fn nearest_distances<'py>(
    py: Python<'py>,
    source: PyReadonlyArray2<f64>,
    target: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (source, target) = (points(&source)?, points(&target)?);
    let d = py
        .detach(|| ca_core::distance::cloud_to_cloud_par(&source, &target))
        .ok_or_else(|| PyValueError::new_err("target is empty"))?;
    Ok(d.into_pyarray(py))
}

fn mesh_from(
    vertices: &PyReadonlyArray2<f64>,
    triangles: &PyReadonlyArray2<u32>,
) -> PyResult<TriangleMesh> {
    let tri = triangles.as_array();
    if tri.ncols() != 3 {
        return Err(PyValueError::new_err("triangles must be an (M, 3) array"));
    }
    let mut mesh = TriangleMesh {
        vertices: points(vertices)?,
        triangles: tri.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect(),
    };
    mesh.validate();
    Ok(mesh)
}

/// Distance from every point to a triangle mesh (cloud-to-mesh). With
/// ``signed``, points behind the closest triangle's normal are negative.
#[pyfunction]
#[pyo3(signature = (points_, vertices, triangles, signed = true))]
fn cloud_to_mesh<'py>(
    py: Python<'py>,
    points_: PyReadonlyArray2<f64>,
    vertices: PyReadonlyArray2<f64>,
    triangles: PyReadonlyArray2<u32>,
    signed: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let pts = points(&points_)?;
    let mesh = mesh_from(&vertices, &triangles)?;
    let d = py
        .detach(|| ca_core::distance::cloud_to_mesh_par(&pts, &mesh, signed))
        .ok_or_else(|| PyValueError::new_err("mesh has no triangles"))?;
    Ok(d.into_pyarray(py))
}

/// Rigid ICP registration of ``moving`` onto ``reference``.
///
/// Returns a dict with ``transformation`` (4x4, maps moving onto reference),
/// ``rms_initial``, ``rms_final``, ``iterations`` and ``converged``.
#[pyfunction]
#[pyo3(signature = (moving, reference, max_iterations = 50, overlap = 1.0, match_centroids = false, point_to_plane = true))]
fn icp<'py>(
    py: Python<'py>,
    moving: PyReadonlyArray2<f64>,
    reference: PyReadonlyArray2<f64>,
    max_iterations: usize,
    overlap: f64,
    match_centroids: bool,
    point_to_plane: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let cloud = |p: Vec<[f64; 3]>| PointCloud {
        positions: p,
        colors: None,
        attributes: Vec::new(),
    };
    let (moving, reference) = (cloud(points(&moving)?), cloud(points(&reference)?));
    let params = IcpParams {
        metric: if point_to_plane {
            IcpMetric::PointToPlane
        } else {
            IcpMetric::PointToPoint
        },
        max_iterations,
        overlap,
        match_centroids,
        ..IcpParams::default()
    };
    let result = py
        .detach(|| ca_core::icp::icp(&moving, &reference, params))
        .ok_or_else(|| PyValueError::new_err("ICP needs non-empty clouds and at least 3 pairs"))?;
    let m = result.transform.to_matrix();
    let matrix = Array2::from_shape_vec((4, 4), m.to_vec())
        .expect("4 x 4")
        .into_pyarray(py);
    let out = PyDict::new(py);
    out.set_item("transformation", matrix)?;
    out.set_item("rms_initial", result.rms_initial)?;
    out.set_item("rms_final", result.rms_final)?;
    out.set_item("iterations", result.iterations)?;
    out.set_item("converged", result.converged)?;
    Ok(out)
}

fn cloud_of(p: Vec<[f64; 3]>) -> PointCloud {
    PointCloud {
        positions: p,
        colors: None,
        attributes: Vec::new(),
    }
}

fn indices<'py>(py: Python<'py>, keep: Vec<usize>) -> Bound<'py, PyArray1<i64>> {
    keep.into_iter()
        .map(|i| i as i64)
        .collect::<Vec<_>>()
        .into_pyarray(py)
}

/// Indices of one point per voxel of edge ``voxel`` (the first point of each).
#[pyfunction]
fn voxel_subsample<'py>(
    py: Python<'py>,
    points_: PyReadonlyArray2<f64>,
    voxel: f64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let cloud = cloud_of(points(&points_)?);
    let keep = py.detach(|| ca_core::filter::voxel_subsample(&cloud, voxel));
    Ok(indices(py, keep))
}

/// Indices of the points that survive statistical outlier removal.
#[pyfunction]
#[pyo3(signature = (points_, k = 8, ratio = 1.0))]
fn statistical_outliers<'py>(
    py: Python<'py>,
    points_: PyReadonlyArray2<f64>,
    k: usize,
    ratio: f64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let cloud = cloud_of(points(&points_)?);
    let keep = py.detach(|| ca_core::filter::statistical_outliers_par(&cloud, k, ratio));
    Ok(indices(py, keep))
}

/// Ground extraction with the Cloth Simulation Filter. Returns a boolean
/// array: ``True`` for ground points. ``rigidness`` is ``"flat"``,
/// ``"relief"`` or ``"steep"``; distances are in the cloud's units.
#[pyfunction]
#[pyo3(signature = (points_, cloth_resolution = 1.0, class_threshold = 0.5, rigidness = "relief", max_iterations = 500))]
fn ground_csf<'py>(
    py: Python<'py>,
    points_: PyReadonlyArray2<f64>,
    cloth_resolution: f64,
    class_threshold: f64,
    rigidness: &str,
    max_iterations: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    use ca_core::ground::{CsfParams, Rigidness, csf};
    let rigidness = match rigidness {
        "flat" => Rigidness::Flat,
        "relief" => Rigidness::Relief,
        "steep" => Rigidness::Steep,
        other => {
            return Err(PyValueError::new_err(format!(
                "rigidness must be flat, relief or steep, not {other:?}"
            )));
        }
    };
    let cloud = cloud_of(points(&points_)?);
    let params = CsfParams {
        cloth_resolution,
        class_threshold,
        rigidness,
        max_iterations,
        ..CsfParams::default()
    };
    let ground = py.detach(|| csf(&cloud, params)).ok_or_else(|| {
        PyValueError::new_err("need points and a positive, not too fine cloth resolution")
    })?;
    Ok(ground.into_pyarray(py))
}

/// The rotation of an IMU's frame into the scans' frame that its up
/// directions ``ups`` ((N, 3)) agree with best, seen through the keyframes'
/// ``rotations`` ((N, 3, 3), scans' frame to world); see
/// ``ca_core::pose_graph::calibrate_ups``.
#[pyfunction]
fn calibrate_ups<'py>(
    py: Python<'py>,
    rotations: numpy::PyReadonlyArray3<f64>,
    ups: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ups = points(&ups)?;
    let view = rotations.as_array();
    if view.shape()[0] != ups.len() || view.shape()[1] != 3 || view.shape()[2] != 3 {
        return Err(PyValueError::new_err("one (3, 3) rotation per up expected"));
    }
    let rotations: Vec<[[f64; 3]; 3]> = view
        .outer_iter()
        .map(|m| std::array::from_fn(|i| std::array::from_fn(|j| m[[i, j]])))
        .collect();
    let mount = py.detach(|| ca_core::pose_graph::calibrate_ups(&rotations, &ups));
    Ok(to_array2(py, mount.to_vec()))
}

/// The significant M3C2 changes grouped into objects (see
/// ``ca_core::m3c2::changed_objects``), largest first: ``(objects, labels)``
/// with one row per object ``[count, centroid xyz, min xyz, max xyz, mean
/// change]`` and each core point's object (or -1).
#[pyfunction]
#[pyo3(signature = (positions, change, significant, min_change = 0.3, link = 1.0, min_points = 8))]
#[allow(clippy::type_complexity)]
fn changed_objects<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray2<f64>,
    change: numpy::PyReadonlyArray1<f64>,
    significant: numpy::PyReadonlyArray1<bool>,
    min_change: f64,
    link: f64,
    min_points: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>)> {
    let positions = points(&positions)?;
    let (change, significant) = (
        change.as_slice()?.to_vec(),
        significant.as_slice()?.to_vec(),
    );
    if change.len() != positions.len() || significant.len() != positions.len() {
        return Err(PyValueError::new_err(
            "one change and one significance per point expected",
        ));
    }
    let (objects, labels) = py.detach(|| {
        ca_core::m3c2::changed_objects(
            &positions,
            &change,
            &significant,
            min_change,
            link,
            min_points,
        )
    });
    let rows: Vec<f64> = objects
        .iter()
        .flat_map(|o| {
            let mut row = vec![o.count as f64];
            row.extend(o.centroid);
            row.extend(o.min);
            row.extend(o.max);
            row.push(o.mean_change);
            row
        })
        .collect();
    let table = Array2::from_shape_vec((objects.len(), 11), rows).expect("k x 11");
    let labels: Vec<i64> = labels
        .iter()
        .map(|&l| {
            if l == ca_core::cluster::NOISE {
                -1
            } else {
                i64::from(l)
            }
        })
        .collect();
    Ok((table.into_pyarray(py), labels.into_pyarray(py)))
}

/// M3C2 change from ``cloud1`` to ``cloud2`` at the ``core`` points
/// (Lague et al. 2013), multi-threaded. Returns ``(distance, lod95,
/// significant, normals)``; distance and LoD95 are NaN where a cylinder
/// holds fewer than ``min_points`` points of either cloud.
#[pyfunction]
#[pyo3(signature = (core, cloud1, cloud2, normal_radius = 1.0, projection_radius = 0.5, max_depth = 2.0, min_points = 5, registration_error = 0.0))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn m3c2<'py>(
    py: Python<'py>,
    core: PyReadonlyArray2<f64>,
    cloud1: PyReadonlyArray2<f64>,
    cloud2: PyReadonlyArray2<f64>,
    normal_radius: f64,
    projection_radius: f64,
    max_depth: f64,
    min_points: usize,
    registration_error: f64,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<bool>>,
    Bound<'py, PyArray2<f64>>,
)> {
    use ca_core::m3c2::{M3c2Params, m3c2_par};
    let (core, cloud1, cloud2) = (points(&core)?, points(&cloud1)?, points(&cloud2)?);
    let params = M3c2Params {
        normal_radius,
        projection_radius,
        max_depth,
        min_points,
        registration_error,
    };
    let r = py
        .detach(|| m3c2_par(&core, &cloud1, &cloud2, params))
        .ok_or_else(|| PyValueError::new_err("need non-empty clouds and positive radii"))?;
    let normals = numpy::ndarray::Array2::from_shape_vec(
        (r.normals.len(), 3),
        r.normals.into_iter().flatten().collect(),
    )
    .expect("n x 3");
    Ok((
        r.distance.into_pyarray(py),
        r.lod95.into_pyarray(py),
        r.significant.into_pyarray(py),
        normals.into_pyarray(py),
    ))
}

/// Cross-section: the points within ``half_width`` (horizontally) of the
/// polyline ``line`` (an ``(M, 2)`` array of x, y vertices). Returns
/// ``(indices, distance_along)``.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn profile<'py>(
    py: Python<'py>,
    points_: PyReadonlyArray2<f64>,
    line: PyReadonlyArray2<f64>,
    half_width: f64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
    let points = points(&points_)?;
    let line = line.as_array();
    if line.ncols() != 2 {
        return Err(PyValueError::new_err("line must be an (M, 2) array"));
    }
    let line: Vec<[f64; 2]> = line.rows().into_iter().map(|r| [r[0], r[1]]).collect();
    let hits = py.detach(|| ca_core::profile::profile(&points, &line, half_width));
    let indices: Vec<i64> = hits.iter().map(|h| h.0 as i64).collect();
    let along: Vec<f64> = hits.iter().map(|h| h.1).collect();
    Ok((indices.into_pyarray(py), along.into_pyarray(py)))
}

/// Unit normals from the ``k`` nearest neighbours of each point (PCA),
/// multi-threaded. ``orientation`` is ``"up"`` (+z) or ``"outward"`` (away
/// from the bounding box centre). Rows are zero where no plane fits.
#[pyfunction]
#[pyo3(signature = (points_, k = 12, orientation = "up"))]
fn normals<'py>(
    py: Python<'py>,
    points_: PyReadonlyArray2<f64>,
    k: usize,
    orientation: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    use ca_core::normals::{Orientation, estimate_normals_par};
    let points = points(&points_)?;
    let orientation = match orientation {
        "up" => Orientation::Up,
        "outward" => {
            let cloud = cloud_of(points.clone());
            let b = cloud
                .bounds()
                .ok_or_else(|| PyValueError::new_err("no points"))?;
            Orientation::Away(b.center())
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "orientation must be up or outward, not {other:?}"
            )));
        }
    };
    let n = py.detach(|| estimate_normals_par(&points, k, orientation));
    let array =
        numpy::ndarray::Array2::from_shape_vec((n.len(), 3), n.into_flattened()).expect("n x 3");
    Ok(array.into_pyarray(py))
}

/// Node-by-node COPC reader; ``cloudanalyzer_core.read_copc`` drives it,
/// reading the byte ranges it asks for from a file or a URL.
#[pyclass(module = "cloudanalyzer_core._core")]
struct CopcReader {
    header: ca_core::io::copc::CopcHeader,
    selector: ca_core::io::copc::NodeSelector,
}

#[pymethods]
impl CopcReader {
    /// Whether these first bytes (at least 400) are a COPC file.
    #[staticmethod]
    fn is_copc(head: &[u8]) -> bool {
        ca_core::io::copc::CopcHeader::is_copc(head)
    }

    /// Bytes from the start of the file the reader needs.
    #[staticmethod]
    fn header_length(head: &[u8]) -> Option<usize> {
        ca_core::io::copc::CopcHeader::needed(head)
    }

    #[new]
    fn new(head: &[u8]) -> PyResult<Self> {
        let header = ca_core::io::copc::CopcHeader::parse(head).map_err(io_err)?;
        Ok(Self {
            selector: ca_core::io::copc::NodeSelector::new(header.root_page),
            header,
        })
    }

    #[getter]
    fn total_points(&self) -> u64 {
        self.header.total_points
    }

    /// Hierarchy pages still needed down to ``level``: ``(offset, size)``.
    fn pages_for(&self, level: i32) -> Vec<(u64, u64)> {
        self.selector.pages_for(level)
    }

    fn add_page(&mut self, offset: u64, page: &[u8]) {
        self.selector.add_page(offset, page);
    }

    /// Points in the nodes at ``level`` (once its pages are added).
    fn level_points(&self, level: i32) -> u64 {
        self.selector.level_points(level)
    }

    fn deeper_than(&self, level: i32) -> bool {
        self.selector.deeper_than(level)
    }

    /// Nodes down to ``level``: ``(offset, size, points)``.
    fn nodes_to(&self, level: i32) -> Vec<(u64, u64, u64)> {
        self.selector
            .nodes_to(level)
            .into_iter()
            .map(|e| (e.offset, e.byte_size as u64, e.point_count as u64))
            .collect()
    }

    /// Decode nodes (their compressed chunks and point counts) on all cores,
    /// as a dictionary like ``read``'s.
    fn decode<'py>(
        &self,
        py: Python<'py>,
        chunks: Vec<Vec<u8>>,
        counts: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if chunks.len() != counts.len() {
            return Err(PyValueError::new_err("one point count per chunk is needed"));
        }
        let nodes: Vec<(&[u8], usize)> = chunks.iter().map(|c| c.as_slice()).zip(counts).collect();
        let points = py
            .detach(|| self.header.decode_nodes_par(&nodes))
            .map_err(io_err)?;
        cloud_dict(py, points.into_cloud())
    }
}

/// A volume surface from Python: a float (constant height), a
/// ``(vertices, triangles)`` tuple (mesh), or an ``(N, 3)`` array (points).
enum PySurface {
    Constant(f64),
    Mesh(TriangleMesh),
    Points(Vec<[f64; 3]>),
}

impl PySurface {
    fn extract(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(z) = value.extract::<f64>() {
            return Ok(PySurface::Constant(z));
        }
        if let Ok((vertices, triangles)) =
            value.extract::<(PyReadonlyArray2<f64>, PyReadonlyArray2<u32>)>()
        {
            return Ok(PySurface::Mesh(mesh_from(&vertices, &triangles)?));
        }
        let array: PyReadonlyArray2<f64> = value.extract().map_err(|_| {
            PyValueError::new_err(
                "a surface is a float, a (vertices, triangles) tuple, or an (N, 3) array",
            )
        })?;
        Ok(PySurface::Points(points(&array)?))
    }

    fn surface(&self) -> ca_core::volume::Surface<'_> {
        use ca_core::volume::Surface;
        match self {
            PySurface::Constant(z) => Surface::Constant(*z),
            PySurface::Mesh(m) => Surface::Mesh(m),
            PySurface::Points(p) => Surface::Points(p),
        }
    }
}

/// Cut/fill volume between two surfaces on a grid of ``cell``-sized squares.
///
/// Each surface is a float (constant height), a ``(vertices, triangles)``
/// tuple, or an ``(N, 3)`` array. Returns a dict with ``added`` (fill),
/// ``removed`` (cut), ``net``, ``added_area``, ``removed_area``,
/// ``matched_cells``, ``total_cells`` and ``difference`` (``(ny, nx)``
/// after − before, NaN where undefined) with ``grid_min`` and ``cell``.
#[pyfunction]
#[pyo3(signature = (before, after, cell, height = "mean", fill_empty = false))]
fn volume<'py>(
    py: Python<'py>,
    before: &Bound<'py, PyAny>,
    after: &Bound<'py, PyAny>,
    cell: f64,
    height: &str,
    fill_empty: bool,
) -> PyResult<Bound<'py, PyDict>> {
    use ca_core::volume::{CellHeight, VolumeParams};
    let height = match height {
        "mean" => CellHeight::Mean,
        "min" => CellHeight::Min,
        "max" => CellHeight::Max,
        other => {
            return Err(PyValueError::new_err(format!(
                "height must be mean, min or max, not {other:?}"
            )));
        }
    };
    let (before, after) = (PySurface::extract(before)?, PySurface::extract(after)?);
    let params = VolumeParams {
        cell,
        height,
        fill_empty,
    };
    let r = py
        .detach(|| ca_core::volume::volume(before.surface(), after.surface(), params))
        .ok_or_else(|| {
            PyValueError::new_err("need a positive cell size and at least one non-constant surface")
        })?;
    let difference =
        Array2::from_shape_vec((r.grid.ny, r.grid.nx), r.difference()).expect("ny x nx cells");
    let out = PyDict::new(py);
    out.set_item("added", r.added)?;
    out.set_item("removed", r.removed)?;
    out.set_item("net", r.net())?;
    out.set_item("added_area", r.added_area)?;
    out.set_item("removed_area", r.removed_area)?;
    out.set_item("matched_cells", r.matched_cells)?;
    out.set_item("total_cells", r.total_cells)?;
    out.set_item("cell", r.grid.cell)?;
    out.set_item("grid_min", (r.grid.min[0], r.grid.min[1]))?;
    out.set_item("difference", difference.into_pyarray(py))?;
    Ok(out)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(read, m)?)?;
    m.add_function(wrap_pyfunction!(read_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_distances, m)?)?;
    m.add_function(wrap_pyfunction!(cloud_to_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(icp, m)?)?;
    m.add_function(wrap_pyfunction!(voxel_subsample, m)?)?;
    m.add_function(wrap_pyfunction!(statistical_outliers, m)?)?;
    m.add_function(wrap_pyfunction!(volume, m)?)?;
    m.add_function(wrap_pyfunction!(ground_csf, m)?)?;
    m.add_function(wrap_pyfunction!(m3c2, m)?)?;
    m.add_function(wrap_pyfunction!(changed_objects, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_ups, m)?)?;
    m.add_function(wrap_pyfunction!(profile, m)?)?;
    m.add_function(wrap_pyfunction!(normals, m)?)?;
    m.add_function(wrap_pyfunction!(vector_map::build_vector_map, m)?)?;
    m.add_function(wrap_pyfunction!(vector_map::measure_vector_map_signal, m)?)?;
    m.add_function(wrap_pyfunction!(
        vector_map::measure_vector_map_crosswalk,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        vector_map::measure_vector_map_signal_points,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        vector_map::connect_vector_map_junctions,
        m
    )?)?;
    m.add_class::<CopcReader>()?;
    m.add_class::<copc_spatial::CopcSpatialQuery>()?;
    m.add_class::<pose_graph::PoseGraph>()?;
    m.add_class::<odometry::LidarOdometry>()?;
    m.add_class::<bag::BagReader>()?;
    m.add_class::<bag::BagMessages>()?;
    m.add_class::<bag::Scan>()?;
    Ok(())
}
