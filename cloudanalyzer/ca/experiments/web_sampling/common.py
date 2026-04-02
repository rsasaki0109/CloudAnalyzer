"""Shared utilities for web sampling experiments."""

import open3d as o3d


def clone_point_cloud(point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Clone a point cloud so experiments don't share mutable geometry."""

    return o3d.geometry.PointCloud(point_cloud)
