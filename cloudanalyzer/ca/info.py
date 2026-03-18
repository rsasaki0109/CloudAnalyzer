"""Point cloud info module."""

import numpy as np
import open3d as o3d

from ca.io import load_point_cloud


def get_info(path: str) -> dict:
    """Get basic information about a point cloud file.

    Args:
        path: Path to point cloud file.

    Returns:
        Dict with point cloud metadata.
    """
    pcd = load_point_cloud(path)
    points = np.asarray(pcd.points)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent = bbox_max - bbox_min

    info = {
        "path": path,
        "num_points": len(points),
        "has_colors": pcd.has_colors(),
        "has_normals": pcd.has_normals(),
        "bbox_min": [float(v) for v in bbox_min],
        "bbox_max": [float(v) for v in bbox_max],
        "extent": [float(v) for v in extent],
        "centroid": [float(v) for v in points.mean(axis=0)],
    }
    return info
