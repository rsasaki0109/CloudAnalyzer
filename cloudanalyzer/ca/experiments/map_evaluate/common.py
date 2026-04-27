"""Shared request/result shapes and toy datasets for map evaluation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class MapEvaluateRequest:
    """Compare an estimated point cloud map to a reference (optional).

    - If `reference_points` is provided, strategies should produce GT-based metrics.
    - If `reference_points` is None, strategies should produce self-consistency metrics.
    """

    estimated_points: np.ndarray  # (N, 3)
    reference_points: np.ndarray | None = None  # (M, 3) when present
    downsample_voxel_size: float = 0.0
    # MapEval-like "accuracy_level": thresholds used for inlier ratios.
    thresholds_m: tuple[float, float, float, float, float] = (0.2, 0.1, 0.08, 0.05, 0.01)
    # A coarse voxel size useful for global-structure metrics / robust proxies.
    structure_voxel_size: float = 0.5


@dataclass(slots=True)
class MapEvaluateResult:
    """Common output for all strategies."""

    strategy: str
    design: str
    metrics: dict[str, float]
    artifacts: dict[str, Any]


def _require_xyz(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be shape (N, 3); got {arr.shape}")
    return arr


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Deterministic voxel downsample by centroid per voxel."""
    pts = _require_xyz(points, "points")
    if pts.shape[0] == 0 or voxel_size <= 0:
        return pts
    keys = np.floor(pts / voxel_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_sorted = keys[order]
    pts_sorted = pts[order]
    unique, start_idx = np.unique(keys_sorted, axis=0, return_index=True)
    # compute centroid for each group
    centroids = []
    for i, s in enumerate(start_idx):
        e = start_idx[i + 1] if i + 1 < len(start_idx) else pts_sorted.shape[0]
        centroids.append(pts_sorted[s:e].mean(axis=0))
    out = np.vstack(centroids) if centroids else np.zeros((0, 3), dtype=np.float64)
    # stable order by voxel key
    _ = unique  # keep correspondence obvious
    return out


@dataclass(slots=True)
class MapEvaluateDatasetCase:
    name: str
    description: str
    request: MapEvaluateRequest


def build_default_datasets() -> list[MapEvaluateDatasetCase]:
    """Small deterministic point-cloud scenarios for quick strategy comparison."""
    rng = np.random.default_rng(7)

    # Base reference: cube-ish cluster
    ref = rng.uniform([-5, -5, -1], [5, 5, 1], size=(400, 3))

    # Estimated: small noise + slight shift (simulates drift)
    est_drift = ref + np.array([0.15, -0.05, 0.0]) + rng.normal(0, 0.02, size=ref.shape)

    # Estimated: incomplete map (drop a region)
    mask = ref[:, 0] > -1.0
    est_incomplete = ref[mask] + rng.normal(0, 0.02, size=(mask.sum(), 3))

    # GT missing: self-consistency only dataset (two overlapping scans fused)
    scan_a = rng.uniform([-6, -6, -1], [6, 6, 1], size=(250, 3))
    scan_b = scan_a + rng.normal(0, 0.03, size=scan_a.shape)
    est_self = np.vstack([scan_a, scan_b])

    return [
        MapEvaluateDatasetCase(
            name="gt_drift",
            description="Estimated map has small rigid drift relative to reference.",
            request=MapEvaluateRequest(
                estimated_points=est_drift,
                reference_points=ref,
                downsample_voxel_size=0.0,
                thresholds_m=(0.2, 0.1, 0.08, 0.05, 0.01),
                structure_voxel_size=0.5,
            ),
        ),
        MapEvaluateDatasetCase(
            name="gt_incomplete",
            description="Estimated map misses a region; completeness should drop.",
            request=MapEvaluateRequest(
                estimated_points=est_incomplete,
                reference_points=ref,
                downsample_voxel_size=0.0,
                thresholds_m=(0.2, 0.1, 0.08, 0.05, 0.01),
                structure_voxel_size=0.5,
            ),
        ),
        MapEvaluateDatasetCase(
            name="no_gt_self_consistency",
            description="No GT; fused map from two noisy overlapping scans.",
            request=MapEvaluateRequest(
                estimated_points=est_self,
                reference_points=None,
                downsample_voxel_size=0.0,
                thresholds_m=(0.2, 0.1, 0.08, 0.05, 0.01),
                structure_voxel_size=0.5,
            ),
        ),
    ]

