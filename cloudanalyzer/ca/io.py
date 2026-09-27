"""Point cloud I/O module."""

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import open3d as o3d

from ca._rust import core


SUPPORTED_EXTENSIONS = {".pcd", ".ply", ".las", ".laz", ".csv"}


def _validate_chunk_size(chunk_size: int) -> int:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    return int(chunk_size)


def _validate_bounds(
    bounds: tuple[float, float, float, float, float, float] | None,
) -> np.ndarray | None:
    if bounds is None:
        return None
    values = np.asarray(bounds, dtype=float)
    if values.shape != (6,):
        raise ValueError("bounds must be (min_x, min_y, min_z, max_x, max_y, max_z)")
    if not np.isfinite(values).all() or np.any(values[:3] > values[3:]):
        raise ValueError("bounds must be finite and min values must not exceed max values")
    return values


def _filter_chunk(
    points: np.ndarray,
    bounds: np.ndarray | None,
) -> np.ndarray:
    chunk = np.asarray(points, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != 3:
        raise ValueError(f"point chunk must be shape (N, 3); got {chunk.shape}")
    finite = np.isfinite(chunk).all(axis=1)
    if bounds is not None:
        finite &= np.all((chunk >= bounds[:3]) & (chunk <= bounds[3:]), axis=1)
    return chunk[finite]


def _yield_array_chunks(points: np.ndarray, chunk_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        if len(chunk):
            yield np.asarray(chunk, dtype=np.float64)


def _is_remote_copc_path(path: str) -> bool:
    parsed = urlparse(path)
    suffix = parsed.path.lower()
    return parsed.scheme in {"http", "https", "s3"} and suffix.endswith(
        (".copc.laz", ".laz")
    )


def _iter_remote_copc_chunks(
    path: str,
    chunk_size: int,
    bounds: np.ndarray | None,
) -> Iterator[np.ndarray]:
    """Read a remote COPC resource through optional PDAL.

    PDAL owns the HTTP Range/COPC spatial-index details.  The dependency is
    intentionally optional because its binary wheels are platform-specific.
    """
    try:
        import pdal
    except ImportError as exc:
        raise ValueError(
            "Remote COPC input requires the optional PDAL Python package; "
            "install PDAL and retry"
        ) from exc

    stages: list[dict[str, object]] = [{"type": "readers.copc", "filename": path}]
    if bounds is not None:
        stages.append(
            {
                "type": "filters.crop",
                "bounds": (
                    f"([{bounds[0]},{bounds[3]}],"
                    f"[{bounds[1]},{bounds[4]}],"
                    f"[{bounds[2]},{bounds[5]}])"
                ),
            }
        )

    pipeline = pdal.Pipeline(json.dumps(stages))
    pipeline.execute()
    arrays = getattr(pipeline, "arrays", [])
    yielded = False
    for array in arrays:
        names = {name.lower(): name for name in getattr(array.dtype, "names", ()) or ()}
        if not all(axis in names for axis in ("x", "y", "z")):
            raise ValueError("COPC data must expose X/Y/Z dimensions")
        points = np.column_stack(
            (array[names["x"]], array[names["y"]], array[names["z"]])
        )
        filtered = _filter_chunk(points, bounds)
        for chunk in _yield_array_chunks(filtered, chunk_size):
            yielded = True
            yield chunk
    if not yielded:
        raise ValueError(f"COPC resource is empty after filtering: {path}")


def _iter_csv_point_chunks(
    path: Path,
    chunk_size: int,
    bounds: np.ndarray | None,
) -> Iterator[np.ndarray]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        try:
            first_row = next(reader)
        except StopIteration:
            return

        try:
            [float(value) for value in first_row[:3]]
            has_header = False
        except (ValueError, TypeError):
            has_header = True

        if has_header:
            normalized = {name.strip().lower(): index for index, name in enumerate(first_row)}
            axis_indices = None
            for candidate in (("x", "y", "z"), ("x_m", "y_m", "z_m")):
                if all(axis in normalized for axis in candidate):
                    axis_indices = tuple(normalized[axis] for axis in candidate)
                    break
            if axis_indices is None:
                raise ValueError("CSV point cloud must contain x,y,z or x_m,y_m,z_m columns")
        else:
            axis_indices = (0, 1, 2)

        rows: list[list[float]] = []

        def flush() -> Iterator[np.ndarray]:
            nonlocal rows
            if rows:
                chunk = _filter_chunk(np.asarray(rows, dtype=np.float64), bounds)
                rows = []
                if len(chunk):
                    yield chunk

        pending_rows = [] if has_header else [first_row]
        for row in chain(pending_rows, reader):
            if len(row) <= max(axis_indices):
                continue
            try:
                xyz = [float(row[index]) for index in axis_indices]
            except ValueError:
                continue
            rows.append(xyz)
            if len(rows) >= chunk_size:
                yield from flush()
        yield from flush()


@dataclass(frozen=True, slots=True)
class PointChunkReader:
    """Lazy point-chunk reader for local point-cloud artifacts.

    LAS/LAZ is read with laspy's chunk iterator.  PCD/PLY remain compatible
    through Open3D and are split after loading because Open3D does not expose
    a portable streaming reader for those formats.  Remote HTTP(S)/S3
    ``.laz``/COPC paths use the optional PDAL adapter and preserve the same
    iterator contract.
    """

    path: str
    chunk_size: int = 100_000
    bounds: tuple[float, float, float, float, float, float] | None = None

    def __post_init__(self) -> None:
        _validate_chunk_size(self.chunk_size)
        _validate_bounds(self.bounds)

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter_point_chunks(
            self.path,
            chunk_size=self.chunk_size,
            bounds=self.bounds,
        )


@dataclass(slots=True)
class PointAccumulator:
    """Streaming count, centroid, covariance, and bounds accumulator."""

    count: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    m2: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    minimum: np.ndarray | None = None
    maximum: np.ndarray | None = None

    def update(self, points: np.ndarray) -> "PointAccumulator":
        chunk = _filter_chunk(points, None)
        if chunk.size == 0:
            return self

        batch_count = int(chunk.shape[0])
        batch_mean = np.mean(chunk, axis=0)
        centered = chunk - batch_mean
        batch_m2 = centered.T @ centered
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            total = self.count + batch_count
            delta = batch_mean - self.mean
            self.m2 += batch_m2 + np.outer(delta, delta) * (self.count * batch_count / total)
            self.mean += delta * (batch_count / total)
            self.count = total

        chunk_minimum = np.min(chunk, axis=0)
        chunk_maximum = np.max(chunk, axis=0)
        self.minimum = (
            chunk_minimum
            if self.minimum is None
            else np.minimum(self.minimum, chunk_minimum)
        )
        self.maximum = (
            chunk_maximum
            if self.maximum is None
            else np.maximum(self.maximum, chunk_maximum)
        )
        return self

    def finalize(self) -> dict[str, object]:
        covariance = (
            self.m2 / (self.count - 1)
            if self.count > 1
            else np.eye(3, dtype=np.float64) * 1e-6
        )
        return {
            "count": int(self.count),
            "mean": self.mean.copy(),
            "covariance": covariance,
            "minimum": self.minimum.copy() if self.minimum is not None else None,
            "maximum": self.maximum.copy() if self.maximum is not None else None,
        }


def iter_point_chunks(
    path: str,
    *,
    chunk_size: int = 100_000,
    bounds: tuple[float, float, float, float, float, float] | None = None,
) -> Iterator[np.ndarray]:
    """Yield finite XYZ chunks without requiring one in-memory point array.

    ``bounds`` is an inclusive axis-aligned box.  LAS/LAZ uses laspy's native
    chunk iterator; PCD/PLY are a compatibility fallback and may still load
    the complete file internally through Open3D.
    """
    chunk_size = _validate_chunk_size(chunk_size)
    validated_bounds = _validate_bounds(bounds)
    if _is_remote_copc_path(path):
        yield from _iter_remote_copc_chunks(path, chunk_size, validated_bounds)
        return

    point_path = Path(path)
    if not point_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = point_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    yielded = False
    if ext == ".csv":
        chunks = _iter_csv_point_chunks(point_path, chunk_size, validated_bounds)
    elif ext in {".las", ".laz"}:
        import laspy

        def las_chunks() -> Iterator[np.ndarray]:
            with laspy.open(str(point_path)) as reader:
                for las_chunk in reader.chunk_iterator(chunk_size):
                    chunk = _filter_chunk(
                        np.column_stack((las_chunk.x, las_chunk.y, las_chunk.z)),
                        validated_bounds,
                    )
                    if len(chunk):
                        yield chunk

        chunks = las_chunks()
    else:
        pcd = load_point_cloud(str(point_path))
        chunks = (
            _filter_chunk(chunk, validated_bounds)
            for chunk in _yield_array_chunks(np.asarray(pcd.points), chunk_size)
        )

    for chunk in chunks:
        if len(chunk):
            yielded = True
            yield chunk
    if not yielded:
        raise ValueError(f"Point cloud is empty after filtering: {path}")


def _load_csv_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    with path.open(newline="", encoding="utf-8") as f:
        sample = f.readline()
        if not sample:
            raise ValueError(f"Point cloud is empty: {path}")
        f.seek(0)
        first_fields = [field.strip() for field in sample.split(",")]
        has_header = False
        try:
            [float(value) for value in first_fields[:3]]
        except ValueError:
            has_header = True

        points: list[list[float]] = []
        if has_header:
            dict_reader = csv.DictReader(f)
            if dict_reader.fieldnames is None:
                raise ValueError(f"CSV point cloud has no header: {path}")
            normalized = {name.strip().lower(): name for name in dict_reader.fieldnames}
            axis_names = None
            for candidate in (("x", "y", "z"), ("x_m", "y_m", "z_m")):
                if all(axis in normalized for axis in candidate):
                    axis_names = tuple(normalized[axis] for axis in candidate)
                    break
            if axis_names is None:
                raise ValueError(
                    "CSV point cloud must contain x,y,z or x_m,y_m,z_m columns"
                )
            for dict_row in dict_reader:
                xyz = [float(dict_row[name]) for name in axis_names]
                if np.isfinite(xyz).all():
                    points.append(xyz)
        else:
            plain_reader = csv.reader(f)
            for row in plain_reader:
                if len(row) < 3:
                    continue
                xyz = [float(row[0]), float(row[1]), float(row[2])]
                if np.isfinite(xyz).all():
                    points.append(xyz)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    return pcd


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """Load point cloud from pcd / ply / las / laz / csv.

    Args:
        path: Path to point cloud file.

    Returns:
        open3d PointCloud object.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file format is not supported.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".csv":
        pcd = _load_csv_point_cloud(p)
    elif ext in {".las", ".laz"}:
        rust = core()
        if rust is not None:
            # Same scaled coordinates as laspy, without needing laspy/lazrs.
            xyz = rust.read(str(p))["positions"]
        else:
            import laspy
            las = laspy.read(str(p))
            xyz = np.vstack([las.x, las.y, las.z]).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
    else:
        pcd = o3d.io.read_point_cloud(str(p))

    if pcd.is_empty():
        raise ValueError(f"Point cloud is empty: {path}")

    return pcd


def save_point_cloud(path: str, pcd: o3d.geometry.PointCloud) -> None:
    """Save point cloud to pcd / ply / las / laz / csv.

    Args:
        path: Output file path.
        pcd: open3d PointCloud object.

    Raises:
        ValueError: If file format is not supported.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    p.parent.mkdir(parents=True, exist_ok=True)

    if ext == ".csv":
        xyz = np.asarray(pcd.points)
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z"])
            writer.writerows(xyz.tolist())
    elif ext in {".las", ".laz"}:
        import laspy
        xyz = np.asarray(pcd.points)
        header = laspy.LasHeader(point_format=0, version="1.4")
        header.offsets = xyz.min(axis=0)
        header.scales = np.full(3, 1e-6)
        las = laspy.LasData(header=header)
        las.x = xyz[:, 0]
        las.y = xyz[:, 1]
        las.z = xyz[:, 2]
        las.write(str(p))
    else:
        o3d.io.write_point_cloud(str(p), pcd)
