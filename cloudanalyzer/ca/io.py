"""Point cloud I/O module."""

import csv
from pathlib import Path

import numpy as np
import open3d as o3d


SUPPORTED_EXTENSIONS = {".pcd", ".ply", ".las", ".laz", ".csv"}


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
