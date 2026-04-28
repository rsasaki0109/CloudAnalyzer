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
    ensure_artifact_dir,
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


def _error_colors(dist_m: np.ndarray, vmax_m: float) -> np.ndarray:
    """Map distances to RGB with a simple green->red ramp."""
    vmax = float(max(vmax_m, 1e-12))
    x = np.clip(dist_m / vmax, 0.0, 1.0)
    r = x
    g = 1.0 - x
    b = np.zeros_like(x)
    return np.column_stack([r, g, b]).astype(np.float64)


def _write_colored_ply(points: np.ndarray, colors: np.ndarray, path: str) -> None:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    o3d.io.write_point_cloud(path, pcd, write_ascii=False)


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

        artifacts: dict[str, object] = {
            "thresholds_m": thresholds,
            "downsample_voxel_size": float(request.downsample_voxel_size),
            "align_mode": request.align_mode,
        }

        # Optional: write MapEval-style raw/inlier error visualizations (estimated points).
        out_dir = ensure_artifact_dir(request, "map_evaluate/nn_thresholds")
        if out_dir is not None and est.shape[0] > 0:
            vmax = thresholds[0] if thresholds else 0.2
            raw_colors = _error_colors(est_to_ref, vmax_m=vmax)
            raw_path = str(out_dir / "estimated_error_raw.ply")
            _write_colored_ply(est, raw_colors, raw_path)

            inlier_mask = est_to_ref <= vmax
            inlier_points = est[inlier_mask]
            inlier_colors = raw_colors[inlier_mask]
            inlier_path = str(out_dir / f"estimated_error_inlier_{vmax:.3f}m.ply")
            _write_colored_ply(inlier_points, inlier_colors, inlier_path)

            artifacts["estimated_error_raw_ply"] = raw_path
            artifacts[f"estimated_error_inlier_{vmax:.3f}m_ply"] = inlier_path

        return MapEvaluateResult(
            strategy=self.name,
            design=self.design,
            metrics=metrics,
            artifacts=artifacts,
        )

