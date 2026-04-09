"""Merge multiple point clouds into one."""

import open3d as o3d

from ca.io import load_point_cloud, save_point_cloud


def merge(paths: list[str], output: str) -> dict:
    """Merge multiple point clouds into a single file.

    Args:
        paths: List of input point cloud file paths.
        output: Output file path.

    Returns:
        Dict with per-file point counts and total.
    """
    merged = o3d.geometry.PointCloud()
    counts = []

    for path in paths:
        pcd = load_point_cloud(path)
        count = len(pcd.points)
        counts.append({"path": path, "points": count})
        merged += pcd

    save_point_cloud(output, merged)

    return {
        "inputs": counts,
        "total_points": len(merged.points),
        "output": output,
    }
