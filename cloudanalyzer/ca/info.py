"""Point cloud and ROS bag info module."""

from __future__ import annotations

import numpy as np
import open3d as o3d

from ca.core.bag_ingest import inspect_bag, is_bag_path
from ca.io import load_point_cloud
from ca.point_summary import axis_summary, require_points


def get_info(path: str) -> dict:
    """Get basic information about a point cloud or ROS bag file."""
    if is_bag_path(path):
        return inspect_bag(path, decode_sample=False)

    pcd = load_point_cloud(path)
    points = np.asarray(pcd.points)
    require_points(points, path)
    summary = axis_summary(points)

    info = {
        "path": path,
        "kind": "point_cloud",
        "num_points": len(points),
        "has_colors": pcd.has_colors(),
        "has_normals": pcd.has_normals(),
        "centroid": [float(v) for v in points.mean(axis=0)],
    }
    info.update(summary)
    return info
