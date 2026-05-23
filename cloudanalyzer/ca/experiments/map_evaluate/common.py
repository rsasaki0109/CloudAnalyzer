"""Re-exports for backward compatibility after the core promotion.

The map_evaluate request/result contract and helpers are now owned by
``ca.core.map_evaluate``. The experiment slice keeps this module as a thin
re-export so the remaining experimental strategies
(``voxel_entropy.VoxelEntropyMapEvaluateStrategy``) and any external callers
still work via the old import path.
"""

from __future__ import annotations

from ca.core.map_evaluate import (
    MapEvaluateRequest,
    MapEvaluateResult,
    _require_xyz,
    _require_transform_4x4,
    aligned_estimated_points,
    apply_transform,
    ensure_artifact_dir,
    voxel_downsample,
)

import numpy as np
from dataclasses import dataclass


@dataclass(slots=True)
class MapEvaluateDatasetCase:
    """Small fixture: human-readable name + the request to score."""

    name: str
    description: str
    request: MapEvaluateRequest


def build_default_datasets() -> list[MapEvaluateDatasetCase]:
    """Small deterministic point-cloud scenarios for quick strategy comparison."""
    rng = np.random.default_rng(7)

    ref = rng.uniform([-5, -5, -1], [5, 5, 1], size=(400, 3))

    est_drift = ref + np.array([0.15, -0.05, 0.0]) + rng.normal(0, 0.02, size=ref.shape)

    mask = ref[:, 0] > -1.0
    est_incomplete = ref[mask] + rng.normal(0, 0.02, size=(mask.sum(), 3))

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


__all__ = [
    "MapEvaluateDatasetCase",
    "MapEvaluateRequest",
    "MapEvaluateResult",
    "_require_xyz",
    "_require_transform_4x4",
    "aligned_estimated_points",
    "apply_transform",
    "build_default_datasets",
    "ensure_artifact_dir",
    "voxel_downsample",
]
