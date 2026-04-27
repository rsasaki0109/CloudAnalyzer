"""GT-based map evaluation via nearest-neighbor threshold ratios.

This is a lightweight, dependency-free proxy for MapEval-style AC/COM curves:
- "accuracy@t": fraction of estimated points within distance t of any GT point
- "completeness@t": fraction of GT points within distance t of any estimated point
- "chamfer": mean(est->gt) + mean(gt->est)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ca.experiments.map_evaluate.common import (
    MapEvaluateRequest,
    MapEvaluateResult,
    _require_xyz,
    voxel_downsample,
)


def _min_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute per-point min Euclidean distance from a to b (O(N*M)).

    Kept intentionally simple for toy datasets; real maps should switch to KD-tree.
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.full((a.shape[0],), np.inf, dtype=np.float64)
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))


@dataclass(slots=True)
class NNThresholdMapEvaluateStrategy:
    name: str = "nn_thresholds"
    design: str = "functional"

    def evaluate(self, request: MapEvaluateRequest) -> MapEvaluateResult:
        est = voxel_downsample(_require_xyz(request.estimated_points, "estimated_points"), request.downsample_voxel_size)
        if request.reference_points is None:
            raise ValueError("nn_thresholds requires reference_points (GT).")
        ref = voxel_downsample(_require_xyz(request.reference_points, "reference_points"), request.downsample_voxel_size)

        est_to_ref = _min_distances(est, ref)
        ref_to_est = _min_distances(ref, est)

        thresholds = tuple(float(x) for x in request.thresholds_m)
        metrics: dict[str, float] = {
            "n_est": float(est.shape[0]),
            "n_ref": float(ref.shape[0]),
            "mean_est_to_ref_m": float(np.mean(est_to_ref)) if est_to_ref.size else float("inf"),
            "mean_ref_to_est_m": float(np.mean(ref_to_est)) if ref_to_est.size else float("inf"),
            "chamfer_m": float(np.mean(est_to_ref)) + float(np.mean(ref_to_est)) if (est_to_ref.size and ref_to_est.size) else float("inf"),
        }

        for t in thresholds:
            metrics[f"accuracy@{t:.3f}m"] = float(np.mean(est_to_ref <= t)) if est_to_ref.size else 0.0
            metrics[f"completeness@{t:.3f}m"] = float(np.mean(ref_to_est <= t)) if ref_to_est.size else 0.0

        # One scalar summary (like F-score) using the first threshold.
        t0 = thresholds[0] if thresholds else 0.2
        acc = metrics.get(f"accuracy@{t0:.3f}m", 0.0)
        com = metrics.get(f"completeness@{t0:.3f}m", 0.0)
        fscore = (2 * acc * com / (acc + com)) if (acc + com) > 0 else 0.0
        metrics[f"fscore@{t0:.3f}m"] = float(fscore)

        return MapEvaluateResult(
            strategy=self.name,
            design=self.design,
            metrics=metrics,
            artifacts={
                "thresholds_m": thresholds,
                "downsample_voxel_size": float(request.downsample_voxel_size),
            },
        )

