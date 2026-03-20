"""Distance metrics module."""

import numpy as np
import open3d as o3d


def compute_nn_distance(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> np.ndarray:
    """Compute nearest neighbor distances from source to target.

    Uses Open3D's vectorized compute_point_cloud_distance for performance.

    Args:
        source: Source point cloud.
        target: Target point cloud.

    Returns:
        Numpy array of nearest neighbor distances for each source point.
    """
    distances = np.asarray(source.compute_point_cloud_distance(target))
    return distances


def summarize(distances: np.ndarray) -> dict:
    """Compute summary statistics of distances.

    Args:
        distances: Array of distances.

    Returns:
        Dict with mean, median, max, min, std.
    """
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "max": float(np.max(distances)),
        "min": float(np.min(distances)),
        "std": float(np.std(distances)),
    }


def threshold_stats(distances: np.ndarray, threshold: float) -> dict:
    """Compute how many points exceed a distance threshold.

    Args:
        distances: Array of distances.
        threshold: Distance threshold.

    Returns:
        Dict with threshold, exceed_count, exceed_ratio, total.
    """
    total = len(distances)
    exceed = int(np.sum(distances > threshold))
    return {
        "threshold": threshold,
        "total": total,
        "exceed_count": exceed,
        "exceed_ratio": exceed / total if total > 0 else 0.0,
    }
