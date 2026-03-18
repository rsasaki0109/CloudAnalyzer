"""Point cloud I/O module."""

from pathlib import Path

import open3d as o3d


SUPPORTED_EXTENSIONS = {".pcd", ".ply", ".las"}


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """Load point cloud from pcd / ply / las.

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

    pcd = o3d.io.read_point_cloud(str(p))

    if pcd.is_empty():
        raise ValueError(f"Point cloud is empty: {path}")

    return pcd
