"""Cross-representation geometry QA.

``ca geometry-evaluate`` runs the same Chamfer/AUC/F1 metric pipeline that
``ca evaluate`` uses, but it first normalizes the source artifact through a
representation-specific adapter so non-point-cloud inputs can be scored
against a reference scan without forcing the user to convert them by hand.

Currently supported source representations:

- ``point-cloud`` — anything :func:`ca.io.load_point_cloud` accepts. Passes
  straight through.
- ``gaussian-points`` — 3DGS-style PLY with ``opacity`` per vertex. The
  adapter extracts the Gaussian centers (xyz), applies the sigmoid to the
  stored opacity logit, and (optionally) filters splats whose rendered
  alpha falls below ``opacity_threshold``.
- ``mesh`` — triangle mesh (OBJ / STL / PLY-with-faces) loaded via Open3D
  and surface-sampled into a point cloud. Vertex-only sampling misses
  large flat faces; ``ca`` defaults to ``sample_points_uniformly`` with
  ``mesh_samples`` points (override with ``--mesh-method poisson_disk``
  when you need a more uniform spread at higher cost).
- ``auto`` — peek the source file. PLY with an ``opacity`` property →
  ``gaussian-points``; PLY with a ``face`` element / non-PLY mesh
  extension (``.obj``/``.stl``/``.glb``/``.gltf``) → ``mesh``; everything
  else → ``point-cloud``.

For ``gaussian-points`` the default ``splat_method="centers"`` keeps the
center-only proxy that the original adapter shipped — robust, cheap, and
catches most "did the splat reconstruction drift?" questions for
cross-representation regression tracking. Opt in to
``splat_method="ellipsoid"`` to use ``scale_*``/``rot_*`` and surface-sample
each splat's anisotropic ellipsoid, which gives a much closer proxy to the
rendered surface for thin/elongated splats at higher cost.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d

from ca.evaluate import evaluate as _evaluate_paths
from ca.io import load_point_cloud


REPRESENTATIONS: tuple[str, ...] = ("auto", "point-cloud", "gaussian-points", "mesh")
MESH_SAMPLE_METHODS: tuple[str, ...] = ("uniform", "poisson_disk")
MESH_FILE_EXTENSIONS: tuple[str, ...] = (".obj", ".stl", ".glb", ".gltf")
DEFAULT_MESH_SAMPLES: int = 100_000
SPLAT_METHODS: tuple[str, ...] = ("centers", "ellipsoid")
DEFAULT_SPLAT_SAMPLES: int = 8
_GAUSSIAN_SCALE_FIELDS: tuple[str, ...] = ("scale_0", "scale_1", "scale_2")
_GAUSSIAN_ROT_FIELDS: tuple[str, ...] = ("rot_0", "rot_1", "rot_2", "rot_3")


@dataclass(slots=True)
class GeometryLoadResult:
    """Outcome of loading a 3D source through a representation adapter."""

    point_cloud: o3d.geometry.PointCloud
    points: np.ndarray  # (N, 3)
    source_path: str
    representation_requested: str
    representation_detected: str
    original_count: int
    final_count: int
    applied_filters: list[str] = field(default_factory=list)


# --------------------------------------------------------------------- PLY


def _read_ply_header(path: Path) -> tuple[str, int, list[tuple[str, str]]]:
    """Return ``(format, vertex_count, vertex_properties)`` from a PLY header.

    Only handles vertex elements; faces and other elements are ignored —
    that's enough for Gaussian Splatting exports and unstructured point
    clouds, which is the scope of this adapter.
    """
    with path.open("rb") as f:
        magic = f.readline().rstrip(b"\r\n")
        if magic != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        ply_format: str | None = None
        vertex_count = 0
        properties: list[tuple[str, str]] = []
        in_vertex_element = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: PLY header ended without `end_header`")
            text = line.decode("ascii", errors="replace").strip()
            if text == "end_header":
                break
            tokens = text.split()
            if not tokens:
                continue
            if tokens[0] == "format" and len(tokens) >= 2:
                ply_format = tokens[1]
            elif tokens[0] == "element" and len(tokens) >= 3:
                in_vertex_element = tokens[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(tokens[2])
            elif tokens[0] == "property" and in_vertex_element:
                if len(tokens) < 3:
                    raise ValueError(f"{path}: malformed property line: {text!r}")
                if tokens[1] == "list":
                    # Lists are not used in 3DGS vertex blocks; ignore.
                    continue
                ply_type, name = tokens[1], tokens[-1]
                properties.append((ply_type, name))
    if ply_format is None:
        raise ValueError(f"{path}: PLY header missing `format` directive")
    if not properties:
        raise ValueError(f"{path}: PLY vertex element has no properties")
    return ply_format, vertex_count, properties


_BINARY_TYPE_TO_NUMPY = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _read_ply_vertices(path: Path, wanted: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read the named vertex properties from a PLY file.

    ``wanted`` lists property names (e.g. ``("x", "y", "z", "opacity")``);
    missing names are silently dropped from the result. Supports ASCII and
    binary little-endian (the two formats relevant to 3DGS exports and the
    test fixtures here). Big-endian and uncommon list properties are not
    supported; an explicit error is raised so users see why instead of
    getting silent corruption.

    The vertex block is read in one ``np.loadtxt`` (ASCII) or
    ``np.frombuffer`` (binary) call so 3DGS exports with millions of splats
    don't pay a Python-loop tax.
    """
    ply_format, vertex_count, properties = _read_ply_header(path)
    wanted_set = set(wanted)
    prop_names = [name for _, name in properties]
    num_props = len(properties)

    columns: dict[str, np.ndarray]
    if ply_format == "ascii":
        with path.open("r", encoding="ascii", errors="replace") as f:
            # Skip the header again (we already consumed it for layout).
            while True:
                line = f.readline()
                if not line:
                    raise ValueError(f"{path}: truncated PLY header")
                if line.strip() == "end_header":
                    break
            raw: np.ndarray
            if vertex_count == 0:
                raw = np.empty((0, num_props), dtype=np.float64)
            else:
                try:
                    raw = np.loadtxt(
                        f, max_rows=vertex_count, dtype=np.float64, ndmin=2
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"{path}: failed to parse ASCII vertex block: {exc}"
                    ) from exc
        if raw.shape[0] < vertex_count:
            raise ValueError(
                f"{path}: truncated vertex block "
                f"({raw.shape[0]}/{vertex_count} rows)"
            )
        if raw.shape[1] < num_props:
            raise ValueError(
                f"{path}: vertex row has {raw.shape[1]} fields, expected {num_props}"
            )
        columns = {name: raw[:, idx] for idx, name in enumerate(prop_names)}
    elif ply_format == "binary_little_endian":
        dtype_fields: list[tuple[str, str]] = []
        seen: set[str] = set()
        for ply_type, name in properties:
            np_code = _BINARY_TYPE_TO_NUMPY.get(ply_type)
            if np_code is None:
                raise ValueError(
                    f"{path}: unsupported PLY property type {ply_type!r}"
                )
            # numpy structured dtype requires unique field names; in the
            # unlikely case of a duplicate, suffix it so np.frombuffer still
            # parses the record layout — duplicates are dropped at the
            # ``columns`` assignment below anyway (last-wins).
            field_name = name
            if field_name in seen:
                suffix = 1
                while f"{name}__dup{suffix}" in seen:
                    suffix += 1
                field_name = f"{name}__dup{suffix}"
            seen.add(field_name)
            dtype_fields.append((field_name, "<" + np_code))
        record_dtype = np.dtype(dtype_fields)
        record_size = record_dtype.itemsize
        with path.open("rb") as fb:
            # Re-skip header on the binary handle.
            while True:
                raw_line = fb.readline()
                if not raw_line:
                    raise ValueError(f"{path}: truncated PLY header")
                if raw_line.rstrip(b"\r\n") == b"end_header":
                    break
            payload = fb.read(record_size * vertex_count)
        if len(payload) < record_size * vertex_count:
            raise ValueError(f"{path}: truncated vertex block")
        structured = np.frombuffer(payload, dtype=record_dtype, count=vertex_count)
        field_names = record_dtype.names
        assert field_names is not None  # structured dtype always has names
        columns = {}
        for field_name, name in zip(field_names, prop_names):
            columns[name] = np.asarray(structured[field_name], dtype=np.float64)
    else:
        raise ValueError(
            f"{path}: PLY format {ply_format!r} not supported "
            f"(supported: ascii, binary_little_endian)"
        )

    out: dict[str, np.ndarray] = {}
    for name in wanted_set:
        if name in columns:
            out[name] = np.asarray(columns[name], dtype=np.float64)
    return out


def _ply_has_property(path: Path, name: str) -> bool:
    try:
        _, _, properties = _read_ply_header(path)
    except ValueError:
        return False
    return any(prop_name == name for _, prop_name in properties)


def _ply_has_face_element(path: Path) -> bool:
    """Cheap scan: return True iff the PLY header declares an ``element face`` row.

    Avoids reusing :func:`_read_ply_header` because that helper only tracks
    vertex properties; we need to spot the ``face`` element header line
    itself, regardless of how many vertex properties precede it.
    """
    try:
        with path.open("rb") as f:
            magic = f.readline().rstrip(b"\r\n")
            if magic != b"ply":
                return False
            while True:
                line = f.readline()
                if not line:
                    return False
                text = line.decode("ascii", errors="replace").strip()
                if text == "end_header":
                    return False
                tokens = text.split()
                if (
                    len(tokens) >= 3
                    and tokens[0] == "element"
                    and tokens[1] == "face"
                ):
                    return True
    except OSError:
        return False


# --------------------------------------------------------------------- detect


def detect_representation(source_path: str) -> str:
    """Inspect ``source_path`` and pick a non-``auto`` representation.

    Heuristics:

    1. ``.obj``/``.stl``/``.glb``/``.gltf`` → ``mesh`` (those formats are
       mesh-only; treating them as point clouds would silently sample the
       vertex set, which is a much worse proxy than surface sampling).
    2. ``.ply`` with an ``opacity`` property → ``gaussian-points``.
       Opacity is the load-bearing identifier 3DGS exports share, so it
       takes precedence over face detection (a 3DGS PLY could in
       principle ship faces it doesn't use).
    3. ``.ply`` with an ``element face`` line → ``mesh``.
    4. Everything else → ``point-cloud``.
    """
    path = Path(source_path)
    suffix = path.suffix.lower()
    if suffix in MESH_FILE_EXTENSIONS:
        return "mesh"
    if suffix == ".ply":
        if _ply_has_property(path, "opacity"):
            return "gaussian-points"
        if _ply_has_face_element(path):
            return "mesh"
    return "point-cloud"


# --------------------------------------------------------------------- adapters


def _opacity_keep_mask(opacities_logit: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of splats whose sigmoid(opacity) ≥ threshold."""
    alphas = 1.0 / (1.0 + np.exp(-opacities_logit))
    return np.asarray(alphas >= threshold)


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Return ``n`` quasi-uniform points on the unit sphere (Fibonacci lattice).

    Deterministic for a given ``n``. ``n>=2`` is required so the lattice
    actually distributes points across both poles.
    """
    if n < 2:
        raise ValueError(f"_fibonacci_sphere requires n>=2; got {n}")
    indices = np.arange(n, dtype=np.float64)
    y = 1.0 - (indices / (n - 1)) * 2.0  # 1 → -1
    radius = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    # Golden angle spacing.
    theta = (math.pi * (3.0 - math.sqrt(5.0))) * indices
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.stack([x, y, z], axis=-1)


def _quaternions_to_rotmats(quats_wxyz: np.ndarray) -> np.ndarray:
    """Convert (N, 4) ``wxyz`` quaternions into (N, 3, 3) rotation matrices.

    3DGS PLY exports store ``rot_0..rot_3`` as ``(w, x, y, z)``. Quaternions
    are normalized defensively because some exporters drop precision in the
    log.
    """
    if quats_wxyz.ndim != 2 or quats_wxyz.shape[1] != 4:
        raise ValueError(
            f"_quaternions_to_rotmats expects (N, 4); got {quats_wxyz.shape}"
        )
    norm = np.linalg.norm(quats_wxyz, axis=1, keepdims=True)
    norm = np.where(norm > 0.0, norm, 1.0)
    q = quats_wxyz / norm
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    matrices = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    matrices[:, 0, 1] = 2.0 * (xy - wz)
    matrices[:, 0, 2] = 2.0 * (xz + wy)
    matrices[:, 1, 0] = 2.0 * (xy + wz)
    matrices[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    matrices[:, 1, 2] = 2.0 * (yz - wx)
    matrices[:, 2, 0] = 2.0 * (xz - wy)
    matrices[:, 2, 1] = 2.0 * (yz + wx)
    matrices[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return matrices


def _sample_splat_ellipsoids(
    centers: np.ndarray,
    scales_log: np.ndarray,
    quats_wxyz: np.ndarray,
    samples_per_splat: int,
) -> np.ndarray:
    """Surface-sample anisotropic ellipsoids from 3DGS splat parameters.

    Args:
        centers: (N, 3) splat centers.
        scales_log: (N, 3) per-axis log-scales (3DGS stores log-σ).
        quats_wxyz: (N, 4) per-splat rotation quaternion (w, x, y, z).
        samples_per_splat: ``K`` points per splat, drawn from a shared
            Fibonacci unit-sphere template scaled by ``exp(scales_log)`` and
            rotated by the quaternion. Total returned points are ``N*K``.

    Returns:
        (N*K, 3) ``float64`` array, with the K samples for splat ``i``
        stored at rows ``i*K..(i+1)*K``.
    """
    if samples_per_splat < 2:
        raise ValueError(
            f"splat_samples must be >= 2 for ellipsoid sampling; got {samples_per_splat}"
        )
    if centers.shape != scales_log.shape:
        raise ValueError(
            f"centers shape {centers.shape} != scales shape {scales_log.shape}"
        )
    if quats_wxyz.shape != (centers.shape[0], 4):
        raise ValueError(
            f"quats shape {quats_wxyz.shape} doesn't match centers count {centers.shape[0]}"
        )
    if centers.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    scales = np.exp(scales_log)  # (N, 3)
    rotmat = _quaternions_to_rotmats(quats_wxyz)  # (N, 3, 3)
    template = _fibonacci_sphere(samples_per_splat)  # (K, 3)
    # Broadcast scale per axis: (N, K, 3)
    scaled = template[np.newaxis, :, :] * scales[:, np.newaxis, :]
    # Rotate each (K, 3) batch by its rotmat: rotmat @ scaled^T → transpose.
    # einsum: (N, i, j) x (N, K, j) -> (N, K, i)
    rotated = np.einsum("nij,nkj->nki", rotmat, scaled)
    sampled = rotated + centers[:, np.newaxis, :]
    return np.asarray(sampled.reshape(-1, 3).astype(np.float64, copy=False))


def _load_gaussian_points(
    path: Path,
    opacity_threshold: float | None,
    *,
    splat_method: str = "centers",
    splat_samples: int = DEFAULT_SPLAT_SAMPLES,
) -> tuple[np.ndarray, list[str]]:
    if splat_method not in SPLAT_METHODS:
        raise ValueError(
            f"Unsupported splat_method {splat_method!r}. "
            f"Allowed: {', '.join(SPLAT_METHODS)}"
        )

    wanted: tuple[str, ...] = ("x", "y", "z", "opacity")
    if splat_method == "ellipsoid":
        wanted = wanted + _GAUSSIAN_SCALE_FIELDS + _GAUSSIAN_ROT_FIELDS
    fields = _read_ply_vertices(path, wanted)
    for axis in ("x", "y", "z"):
        if axis not in fields:
            raise ValueError(f"{path}: gaussian-points PLY missing `{axis}` property")

    centers = np.column_stack([fields["x"], fields["y"], fields["z"]]).astype(np.float64)
    applied: list[str] = []

    if splat_method == "ellipsoid":
        missing = [
            name for name in (*_GAUSSIAN_SCALE_FIELDS, *_GAUSSIAN_ROT_FIELDS)
            if name not in fields
        ]
        if missing:
            raise ValueError(
                f"{path}: splat_method='ellipsoid' but PLY is missing "
                f"required 3DGS properties: {', '.join(missing)}"
            )
        scales_log = np.column_stack(
            [fields[name] for name in _GAUSSIAN_SCALE_FIELDS]
        ).astype(np.float64)
        quats = np.column_stack(
            [fields[name] for name in _GAUSSIAN_ROT_FIELDS]
        ).astype(np.float64)
    else:
        scales_log = np.empty((centers.shape[0], 3), dtype=np.float64)
        quats = np.empty((centers.shape[0], 4), dtype=np.float64)

    if opacity_threshold is not None:
        if "opacity" not in fields:
            raise ValueError(
                f"{path}: opacity_threshold set but PLY has no `opacity` property"
            )
        keep = _opacity_keep_mask(fields["opacity"], opacity_threshold)
        centers = centers[keep]
        scales_log = scales_log[keep]
        quats = quats[keep]
        applied.append(f"opacity>={opacity_threshold:g} kept={int(keep.sum())}")

    if splat_method == "ellipsoid":
        points = _sample_splat_ellipsoids(
            centers, scales_log, quats, samples_per_splat=splat_samples
        )
        applied.append(
            f"splat_method=ellipsoid samples_per_splat={splat_samples} "
            f"splats={centers.shape[0]}"
        )
    else:
        points = centers

    return points, applied


def _load_mesh_points(
    path: Path,
    *,
    mesh_samples: int,
    mesh_method: str,
) -> tuple[np.ndarray, list[str]]:
    """Surface-sample a triangle mesh into a deterministic point cloud."""
    if mesh_samples <= 0:
        raise ValueError(
            f"mesh_samples must be positive; got {mesh_samples}"
        )
    if mesh_method not in MESH_SAMPLE_METHODS:
        raise ValueError(
            f"Unsupported mesh sampling method {mesh_method!r}. "
            f"Allowed: {', '.join(MESH_SAMPLE_METHODS)}"
        )

    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_vertices():
        raise ValueError(f"{path}: mesh has no vertices")
    if not mesh.has_triangles():
        raise ValueError(
            f"{path}: mesh has vertices but no triangles — "
            "use --representation point-cloud to score vertices directly"
        )

    # Open3D's sample_points_uniformly requires triangle normals for some
    # paths; make sure they exist so the function is deterministic across
    # Open3D versions.
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    applied: list[str] = [f"mesh_samples={mesh_samples}, method={mesh_method}"]
    # Open3D ≤0.19 doesn't accept an explicit seed; the sampler uses its own
    # RNG. Output isn't bit-reproducible across runs, but the *surface* is —
    # consumers compare the resulting point cloud against a reference scan
    # via Chamfer/AUC, which is robust to that variance.
    if mesh_method == "uniform":
        sampled = mesh.sample_points_uniformly(number_of_points=int(mesh_samples))
    else:
        sampled = mesh.sample_points_poisson_disk(number_of_points=int(mesh_samples))

    points = np.asarray(sampled.points, dtype=np.float64)
    if points.shape[0] == 0:
        raise ValueError(
            f"{path}: mesh sampling produced zero points "
            f"(method={mesh_method}, requested={mesh_samples})"
        )
    return points, applied


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Open3D voxel downsample preserving deterministic order."""
    if voxel_size <= 0 or points.shape[0] == 0:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd.points, dtype=np.float64)


# --------------------------------------------------------------------- load


def load_representation(
    source_path: str,
    representation: str = "auto",
    *,
    opacity_threshold: float | None = None,
    voxel_size: float | None = None,
    mesh_samples: int = DEFAULT_MESH_SAMPLES,
    mesh_method: str = "uniform",
    splat_method: str = "centers",
    splat_samples: int = DEFAULT_SPLAT_SAMPLES,
) -> GeometryLoadResult:
    """Load a 3D artifact and normalize it to a point cloud."""
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Unsupported representation: {representation!r}. "
            f"Allowed: {', '.join(REPRESENTATIONS)}"
        )
    if splat_method not in SPLAT_METHODS:
        raise ValueError(
            f"Unsupported splat_method: {splat_method!r}. "
            f"Allowed: {', '.join(SPLAT_METHODS)}"
        )

    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(source_path)

    detected = detect_representation(source_path)
    chosen = detected if representation == "auto" else representation

    applied: list[str] = []
    if chosen == "gaussian-points":
        points, opacity_filters = _load_gaussian_points(
            path,
            opacity_threshold,
            splat_method=splat_method,
            splat_samples=splat_samples,
        )
        applied.extend(opacity_filters)
    elif chosen == "mesh":
        points, mesh_filters = _load_mesh_points(
            path, mesh_samples=mesh_samples, mesh_method=mesh_method
        )
        applied.extend(mesh_filters)
        if opacity_threshold is not None:
            applied.append(
                "opacity_threshold ignored (representation=mesh)"
            )
    elif chosen == "point-cloud":
        pcd = load_point_cloud(source_path)
        points = np.asarray(pcd.points, dtype=np.float64)
        if opacity_threshold is not None:
            applied.append(
                f"opacity_threshold ignored (representation=point-cloud)"
            )
    else:  # pragma: no cover — guarded above
        raise ValueError(f"Unhandled representation: {chosen}")

    original_count = points.shape[0]
    if voxel_size is not None and voxel_size > 0:
        points = _voxel_downsample(points, voxel_size)
        applied.append(f"voxel={voxel_size:g} kept={points.shape[0]}")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    return GeometryLoadResult(
        point_cloud=cloud,
        points=points,
        source_path=str(path),
        representation_requested=representation,
        representation_detected=detected,
        original_count=original_count,
        final_count=points.shape[0],
        applied_filters=applied,
    )


# --------------------------------------------------------------------- evaluate


def evaluate_geometry(
    source_path: str,
    reference_path: str,
    *,
    representation: str = "auto",
    opacity_threshold: float | None = None,
    voxel_size: float | None = None,
    thresholds: list[float] | None = None,
    mesh_samples: int = DEFAULT_MESH_SAMPLES,
    mesh_method: str = "uniform",
    splat_method: str = "centers",
    splat_samples: int = DEFAULT_SPLAT_SAMPLES,
) -> dict:
    """Cross-representation geometry QA against a reference point cloud.

    Returns the same dict shape as :func:`ca.evaluate.evaluate` plus a
    ``representation`` block recording which adapter ran and what filters
    it applied. The reference is always loaded as a point cloud — the
    representation knob only describes the *source* artifact.
    """
    loaded = load_representation(
        source_path,
        representation=representation,
        opacity_threshold=opacity_threshold,
        voxel_size=voxel_size,
        mesh_samples=mesh_samples,
        mesh_method=mesh_method,
        splat_method=splat_method,
        splat_samples=splat_samples,
    )

    if loaded.points.shape[0] == 0:
        raise ValueError(
            f"{source_path}: no points left after representation adapter "
            f"({'; '.join(loaded.applied_filters) or 'no filters applied'})"
        )

    with tempfile.TemporaryDirectory(prefix="ca-geom-") as tmp_dir:
        tmp_source = Path(tmp_dir) / "source.pcd"
        if not o3d.io.write_point_cloud(str(tmp_source), loaded.point_cloud, write_ascii=False):
            raise RuntimeError(
                f"Failed to materialize adapted source to {tmp_source} for evaluation"
            )
        result = _evaluate_paths(str(tmp_source), reference_path, thresholds=thresholds)

    # Restore the user-facing source path so reports show the real input.
    result["source_path"] = loaded.source_path
    is_gaussian = (
        loaded.representation_detected == "gaussian-points"
        or representation == "gaussian-points"
    )
    is_mesh = (
        loaded.representation_detected == "mesh"
        or representation == "mesh"
    )
    result["representation"] = {
        "requested": loaded.representation_requested,
        "detected": loaded.representation_detected,
        "original_count": loaded.original_count,
        "final_count": loaded.final_count,
        "applied_filters": list(loaded.applied_filters),
        "voxel_size": float(voxel_size) if voxel_size else None,
        "opacity_threshold": (
            float(opacity_threshold) if opacity_threshold is not None else None
        ),
        "mesh_samples": int(mesh_samples) if is_mesh else None,
        "mesh_method": mesh_method if is_mesh else None,
        "splat_method": splat_method if is_gaussian else None,
        "splat_samples": (
            int(splat_samples) if is_gaussian and splat_method == "ellipsoid"
            else None
        ),
    }
    return result


# Re-export the few symbols a CLI / tests need.
__all__ = [
    "DEFAULT_MESH_SAMPLES",
    "DEFAULT_SPLAT_SAMPLES",
    "GeometryLoadResult",
    "MESH_FILE_EXTENSIONS",
    "MESH_SAMPLE_METHODS",
    "REPRESENTATIONS",
    "SPLAT_METHODS",
    "detect_representation",
    "evaluate_geometry",
    "load_representation",
]


# Tiny utility kept here (rather than ca.metrics) because it's PLY-shape
# specific and only the geometry adapter needs it.
def _sigmoid(x: float) -> float:  # pragma: no cover — kept for future docs
    return 1.0 / (1.0 + math.exp(-x))
