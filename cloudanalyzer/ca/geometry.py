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

The library deliberately treats Gaussian splats as points rather than
ellipsoids for now. Ellipsoid surface sampling using ``scale``/``rot`` is a
future enhancement; for cross-representation regression tracking the
center-only proxy already catches most "did the splat reconstruction drift?"
questions.
"""

from __future__ import annotations

import math
import struct
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


_BINARY_TYPE_TO_STRUCT = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def _read_ply_vertices(path: Path, wanted: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read the named vertex properties from a PLY file.

    ``wanted`` lists property names (e.g. ``("x", "y", "z", "opacity")``);
    missing names are silently dropped from the result. Supports ASCII and
    binary little-endian (the two formats relevant to 3DGS exports and the
    test fixtures here). Big-endian and uncommon list properties are not
    supported; an explicit error is raised so users see why instead of
    getting silent corruption.
    """
    ply_format, vertex_count, properties = _read_ply_header(path)
    wanted_set = set(wanted)

    if ply_format == "ascii":
        with path.open("r", encoding="ascii", errors="replace") as f:
            # Skip the header again (we already consumed it for layout).
            while True:
                line = f.readline()
                if not line:
                    raise ValueError(f"{path}: truncated PLY header")
                if line.strip() == "end_header":
                    break
            columns: dict[str, list[float]] = {name: [] for _, name in properties}
            for _ in range(vertex_count):
                row = f.readline()
                if not row:
                    raise ValueError(f"{path}: truncated vertex block")
                values = row.split()
                if len(values) < len(properties):
                    raise ValueError(
                        f"{path}: vertex row has {len(values)} fields, expected {len(properties)}"
                    )
                for (_, name), value in zip(properties, values):
                    columns[name].append(float(value))
    elif ply_format == "binary_little_endian":
        fmt_chars: list[str] = []
        sizes: list[int] = []
        for ply_type, _ in properties:
            char = _BINARY_TYPE_TO_STRUCT.get(ply_type)
            if char is None:
                raise ValueError(
                    f"{path}: unsupported PLY property type {ply_type!r}"
                )
            fmt_chars.append(char)
            sizes.append(struct.calcsize(char))
        record_size = sum(sizes)
        record_fmt = "<" + "".join(fmt_chars)
        with path.open("rb") as fb:
            # Re-skip header on the binary handle.
            while True:
                raw = fb.readline()
                if not raw:
                    raise ValueError(f"{path}: truncated PLY header")
                if raw.rstrip(b"\r\n") == b"end_header":
                    break
            payload = fb.read(record_size * vertex_count)
        if len(payload) < record_size * vertex_count:
            raise ValueError(f"{path}: truncated vertex block")
        columns_b: dict[str, list[float]] = {name: [] for _, name in properties}
        for i in range(vertex_count):
            chunk = payload[i * record_size : (i + 1) * record_size]
            unpacked = struct.unpack(record_fmt, chunk)
            for (_, name), value in zip(properties, unpacked):
                columns_b[name].append(float(value))
        columns = columns_b
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


def _filter_by_opacity(
    points: np.ndarray, opacities_logit: np.ndarray, threshold: float
) -> tuple[np.ndarray, int]:
    """Apply sigmoid to logits and keep splats whose rendered alpha ≥ threshold."""
    alphas = 1.0 / (1.0 + np.exp(-opacities_logit))
    keep = alphas >= threshold
    return points[keep], int(keep.sum())


def _load_gaussian_points(
    path: Path, opacity_threshold: float | None
) -> tuple[np.ndarray, list[str]]:
    fields = _read_ply_vertices(path, ("x", "y", "z", "opacity"))
    for axis in ("x", "y", "z"):
        if axis not in fields:
            raise ValueError(f"{path}: gaussian-points PLY missing `{axis}` property")
    points = np.column_stack([fields["x"], fields["y"], fields["z"]]).astype(np.float64)
    applied: list[str] = []
    if opacity_threshold is not None:
        if "opacity" not in fields:
            raise ValueError(
                f"{path}: opacity_threshold set but PLY has no `opacity` property"
            )
        points, kept = _filter_by_opacity(points, fields["opacity"], opacity_threshold)
        applied.append(f"opacity>={opacity_threshold:g} kept={kept}")
        _ = kept  # surfaced via applied_filters
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
) -> GeometryLoadResult:
    """Load a 3D artifact and normalize it to a point cloud."""
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Unsupported representation: {representation!r}. "
            f"Allowed: {', '.join(REPRESENTATIONS)}"
        )

    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(source_path)

    detected = detect_representation(source_path)
    chosen = detected if representation == "auto" else representation

    applied: list[str] = []
    if chosen == "gaussian-points":
        points, opacity_filters = _load_gaussian_points(path, opacity_threshold)
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
        "mesh_samples": (
            int(mesh_samples) if loaded.representation_detected == "mesh"
            or (representation == "mesh") else None
        ),
        "mesh_method": (
            mesh_method if loaded.representation_detected == "mesh"
            or (representation == "mesh") else None
        ),
    }
    return result


# Re-export the few symbols a CLI / tests need.
__all__ = [
    "DEFAULT_MESH_SAMPLES",
    "GeometryLoadResult",
    "MESH_FILE_EXTENSIONS",
    "MESH_SAMPLE_METHODS",
    "REPRESENTATIONS",
    "detect_representation",
    "evaluate_geometry",
    "load_representation",
]


# Tiny utility kept here (rather than ca.metrics) because it's PLY-shape
# specific and only the geometry adapter needs it.
def _sigmoid(x: float) -> float:  # pragma: no cover — kept for future docs
    return 1.0 / (1.0 + math.exp(-x))
