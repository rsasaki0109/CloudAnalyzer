"""Detailed single point cloud statistics."""

import numpy as np
from scipy.spatial import cKDTree

from ca.io import load_point_cloud
from ca.point_summary import axis_summary, require_points


def compute_stats(path: str) -> dict:
    """Compute detailed statistics for a single point cloud.

    Args:
        path: Path to point cloud file.

    Returns:
        Dict with density, spacing, and distribution stats.
    """
    pcd = load_point_cloud(path)
    points = np.asarray(pcd.points)
    require_points(points, path)
    num_points = len(points)

    summary = axis_summary(points)
    extent = np.asarray(summary["extent"], dtype=float)
    volume = float(np.prod(extent)) if np.all(extent > 0) else 0.0
    density = num_points / volume if volume > 0 else 0.0
    robust_extent = np.asarray(summary["robust_extent"], dtype=float)
    robust_volume = float(np.prod(robust_extent)) if np.all(robust_extent > 0) else 0.0
    robust_density = num_points / robust_volume if robust_volume > 0 else 0.0

    # Sample for large clouds to keep nearest-neighbor spacing computation bounded.
    if num_points > 500_000:
        sample_idx = np.random.default_rng(42).choice(num_points, size=500_000, replace=False)
        sample_points = points[sample_idx]
    else:
        sample_points = points

    tree = cKDTree(points)
    # k=2: a sample point is itself in the full cloud, so its 1-NN is itself
    # (distance 0); take the second-nearest as the spacing.
    distances, _ = tree.query(sample_points, k=2)
    nn_dists = np.asarray(distances[:, 1], dtype=np.float64)

    result = {
        "path": path,
        "num_points": num_points,
        "volume": volume,
        "density": density,
        "robust_volume": robust_volume,
        "robust_density": robust_density,
        "spacing_sample_points": int(len(sample_points)),
        "spacing": {
            "mean": float(np.mean(nn_dists)),
            "median": float(np.median(nn_dists)),
            "min": float(np.min(nn_dists)),
            "max": float(np.max(nn_dists)),
            "std": float(np.std(nn_dists)),
        },
    }
    result.update(summary)
    return result
