"""Shared point cloud summary helpers."""

from __future__ import annotations

import numpy as np


def require_points(points: np.ndarray, path: str) -> None:
    """Raise a clear error for empty point clouds."""
    if len(points) == 0:
        raise ValueError(f"Point cloud is empty: {path}")


def axis_summary(points: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> dict:
    """Return absolute and robust axis-aligned bounds for a point array."""
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    extent = bbox_max - bbox_min
    robust_min = np.percentile(points, low_percentile, axis=0)
    robust_max = np.percentile(points, high_percentile, axis=0)
    robust_extent = robust_max - robust_min
    outside = np.any((points < robust_min) | (points > robust_max), axis=1)
    outside_count = int(np.count_nonzero(outside))

    return {
        "bbox_min": [float(v) for v in bbox_min],
        "bbox_max": [float(v) for v in bbox_max],
        "extent": [float(v) for v in extent],
        "robust_bbox_min": [float(v) for v in robust_min],
        "robust_bbox_max": [float(v) for v in robust_max],
        "robust_extent": [float(v) for v in robust_extent],
        "robust_percentiles": [float(low_percentile), float(high_percentile)],
        "outside_robust_bbox_count": outside_count,
        "outside_robust_bbox_ratio": float(outside_count / len(points)),
        "axis_percentiles": {
            "p01": [float(v) for v in robust_min],
            "p50": [float(v) for v in np.percentile(points, 50.0, axis=0)],
            "p99": [float(v) for v in robust_max],
        },
    }
