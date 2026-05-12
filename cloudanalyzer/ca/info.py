"""Point cloud info module."""

import numpy as np
import open3d as o3d

from ca.io import load_point_cloud
from ca.point_summary import axis_summary, require_points


def get_info(path: str) -> dict:
    """Get basic information about a point cloud file.

    Args:
        path: Path to point cloud file.

    Returns:
        Dict with point cloud metadata.
    """
    pcd = load_point_cloud(path)
    points = np.asarray(pcd.points)
    require_points(points, path)
    summary = axis_summary(points)

    info = {
        "path": path,
        "num_points": len(points),
        "has_colors": pcd.has_colors(),
        "has_normals": pcd.has_normals(),
        "centroid": [float(v) for v in points.mean(axis=0)],
    }
    info.update(summary)
    return info
