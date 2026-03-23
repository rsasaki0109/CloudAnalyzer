"""Detailed single point cloud statistics."""

import numpy as np
import open3d as o3d

from ca.io import load_point_cloud


def compute_stats(path: str) -> dict:
    """Compute detailed statistics for a single point cloud.

    Args:
        path: Path to point cloud file.

    Returns:
        Dict with density, spacing, and distribution stats.
    """
    pcd = load_point_cloud(path)
    points = np.asarray(pcd.points)
    num_points = len(points)

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent = bbox_max - bbox_min
    volume = float(np.prod(extent)) if np.all(extent > 0) else 0.0
    density = num_points / volume if volume > 0 else 0.0

    # Nearest neighbor distances (point spacing) — vectorized via Open3D
    nn_dists = np.asarray(pcd.compute_point_cloud_distance(pcd))
    # compute_point_cloud_distance returns 0 for self-match, so use KDTree k=2 approach
    # Actually Open3D's compute_point_cloud_distance finds nearest in *other* cloud
    # For self-spacing, we need a copy shifted by epsilon — or use KDTree k=2
    # Let's sample for large clouds to keep it fast
    if num_points > 500_000:
        sample_idx = np.random.default_rng(42).choice(num_points, size=500_000, replace=False)
        sample_pcd = pcd.select_by_index(sample_idx.tolist())
    else:
        sample_pcd = pcd

    tree = o3d.geometry.KDTreeFlann(pcd)
    sample_points = np.asarray(sample_pcd.points)
    nn_dists = np.zeros(len(sample_points))
    for i, pt in enumerate(sample_points):
        _, _, dist_sq = tree.search_knn_vector_3d(pt, 2)  # k=2: self + nearest
        nn_dists[i] = np.sqrt(dist_sq[1])

    return {
        "path": path,
        "num_points": num_points,
        "bbox_min": [float(v) for v in bbox_min],
        "bbox_max": [float(v) for v in bbox_max],
        "extent": [float(v) for v in extent],
        "volume": volume,
        "density": density,
        "spacing": {
            "mean": float(np.mean(nn_dists)),
            "median": float(np.median(nn_dists)),
            "min": float(np.min(nn_dists)),
            "max": float(np.max(nn_dists)),
            "std": float(np.std(nn_dists)),
        },
    }
