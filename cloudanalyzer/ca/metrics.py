"""Distance metrics module."""

from typing import Protocol, cast

import numpy as np
import open3d as o3d

from ca._rust import core


class _PreparedReference(Protocol):
    def distances(self, source: np.ndarray) -> np.ndarray: ...


def prepare_nn_reference(target: o3d.geometry.PointCloud) -> _PreparedReference | None:
    """Own a frozen reference tree when the optional core supports it.

    Callers must keep the reference coordinates fixed for the object's lifetime.
    Older extensions, disabled/missing cores and empty targets retain stateless
    dispatch. Real construction failures propagate to the caller.
    """
    rust = core()
    constructor = getattr(rust, "PreparedReference", None)
    if constructor is None or target.is_empty():
        return None
    return cast(_PreparedReference, constructor(np.asarray(target.points)))


def compute_nn_distance(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> np.ndarray:
    """Compute nearest neighbor distances from source to target.

    Uses the multi-threaded Rust core when ``cloudanalyzer_core`` is installed
    (exact nearest neighbours, identical results), otherwise Open3D's
    vectorized compute_point_cloud_distance.

    Args:
        source: Source point cloud.
        target: Target point cloud.

    Returns:
        Numpy array of nearest neighbor distances for each source point.
    """
    rust = core()
    if rust is not None and not target.is_empty():
        return np.asarray(
            rust.nearest_distances(np.asarray(source.points), np.asarray(target.points))
        )
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
