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
    aligned_estimated_points,
    voxel_downsample,
)


def _min_distances_kdtree(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute per-point min Euclidean distance from a to b via Open3D KD-tree."""
    if a.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if b.shape[0] == 0:
        return np.full((a.shape[0],), np.inf, dtype=np.float64)

    import open3d as o3d

    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(np.asarray(b, dtype=np.float64))
    kdtree = o3d.geometry.KDTreeFlann(pcd_b)

    out = np.empty((a.shape[0],), dtype=np.float64)
    for i in range(a.shape[0]):
        # 1-NN search
        _k, _idx, dist2 = kdtree.search_knn_vector_3d(a[i], 1)
        out[i] = float(np.sqrt(dist2[0])) if dist2 else float("inf")
    return out


@dataclass(slots=True)
class NNThresholdMapEvaluateStrategy:
    name: str = "nn_thresholds"
    design: str = "functional"

    def evaluate(self, request: MapEvaluateRequest) -> MapEvaluateResult:
        est_aligned = aligned_estimated_points(request)
        est = voxel_downsample(_require_xyz(est_aligned, "estimated_points"), request.downsample_voxel_size)
        if request.reference_points is None:
            raise ValueError("nn_thresholds requires reference_points (GT).")
        ref = voxel_downsample(_require_xyz(request.reference_points, "reference_points"), request.downsample_voxel_size)

        est_to_ref = _min_distances_kdtree(est, ref)
        ref_to_est = _min_distances_kdtree(ref, est)

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
                "align_mode": request.align_mode,
            },
        )

