"""Point cloud I/O module."""

from pathlib import Path

import numpy as np
import open3d as o3d


SUPPORTED_EXTENSIONS = {".pcd", ".ply", ".las", ".laz"}


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """Load point cloud from pcd / ply / las / laz.

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

    if ext in {".las", ".laz"}:
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
    """Save point cloud to pcd / ply / las / laz.

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

    if ext in {".las", ".laz"}:
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
