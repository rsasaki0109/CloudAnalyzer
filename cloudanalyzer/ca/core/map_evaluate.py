"""Stable, minimal interface for point-cloud map evaluation.

This slice started as ``ca/experiments/map_evaluate/`` with two
non-comparable strategies on separate metric-family lanes:

- ``nn_thresholds`` — reference-based (GT-aware) MapEval-style accuracy and
  completeness@τ via per-point nearest-neighbor distances (scipy cKDTree).
- ``voxel_entropy`` — reference-free self-consistency proxy via voxelized
  neighborhood occupancy entropy.

Because the two strategies do not compete head-to-head, "promoting to core"
means lifting the **request/result contract plus the adopted reference-based
strategy** here, leaving ``voxel_entropy`` in ``ca/experiments/map_evaluate``
as the orthogonal reference-free option. Callers that just want GT-based
metrics can now depend on ``ca.core.map_evaluate`` directly and the CLI
no longer reaches into ``ca.experiments``.

The reference-free lane remains experimental until we settle on a single
GT-free metric — anyone needing it for now imports from
``ca.experiments.map_evaluate.voxel_entropy`` explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


# ---------------------------------------------------------------- contract


@dataclass(slots=True)
class MapEvaluateRequest:
    """Compare an estimated point cloud map to a reference (optional).

    Strategies with ``reference_required=True`` raise on ``reference_points=None``;
    reference-free strategies treat the same field as unused.
    """

    estimated_points: np.ndarray  # (N, 3)
    reference_points: np.ndarray | None = None  # (M, 3) when present
    # MapEval-inspired: allow using an external coarse alignment.
    initial_transform_4x4: np.ndarray | None = None  # shape (4, 4)
    # "none" = use estimated_points as-is, "initial" = pre-apply initial_transform_4x4.
    align_mode: str = "none"
    downsample_voxel_size: float = 0.0
    # Optional output dir for strategy-specific artifacts (colored PLYs, etc.)
    artifact_dir: str | None = None
    # MapEval "accuracy_level" thresholds for inlier ratios.
    thresholds_m: tuple[float, ...] = (0.2, 0.1, 0.08, 0.05, 0.01)
    # A coarse voxel size useful for structure-aware proxies.
    structure_voxel_size: float = 0.5


@dataclass(slots=True)
class MapEvaluateResult:
    """Common output for all strategies.

    Classification fields (``metric_family``, ``reference_required``, ``mode``,
    ``sampling_policy``) describe *how* the metrics were computed so consumers
    (CI gates, reports, batch aggregators) can keep reference-based and
    reference-free metrics in separate lanes without parsing metric names.
    """

    strategy: str
    design: str
    metrics: dict[str, float]
    artifacts: dict[str, Any]
    metric_family: str = "unspecified"
    reference_required: bool = False
    mode: str = "exact"
    sampling_policy: dict[str, Any] = field(default_factory=dict)


class MapEvaluateStrategy(Protocol):
    """Protocol kept in core after promoting the reference-based strategy."""

    name: str
    design: str

    def evaluate(self, request: MapEvaluateRequest) -> MapEvaluateResult:
        """Evaluate a map and return classified metrics."""


# ---------------------------------------------------------------- helpers


def _require_xyz(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be shape (N, 3); got {arr.shape}")
    return arr


def _require_transform_4x4(matrix: np.ndarray, name: str) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"{name} must be shape (4, 4); got {mat.shape}")
    return mat


def apply_transform(points: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    pts = _require_xyz(points, "points")
    t = _require_transform_4x4(transform_4x4, "transform_4x4")
    if pts.shape[0] == 0:
        return pts
    hom = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    out = (hom @ t.T)[:, :3]
    return np.asarray(out, dtype=np.float64)


def aligned_estimated_points(request: MapEvaluateRequest) -> np.ndarray:
    est = _require_xyz(request.estimated_points, "estimated_points")
    mode = (request.align_mode or "none").strip().lower()
    if mode == "none":
        return est
    if mode == "initial":
        if request.initial_transform_4x4 is None:
            raise ValueError("align_mode='initial' requires initial_transform_4x4.")
        return apply_transform(est, request.initial_transform_4x4)
    raise ValueError(f"Unsupported align_mode: {request.align_mode!r}. Use 'none' or 'initial'.")


def ensure_artifact_dir(request: MapEvaluateRequest, subdir: str) -> Path | None:
    """Return an ensured artifact directory path or None if disabled."""
    if request.artifact_dir is None:
        return None
    root = Path(request.artifact_dir)
    out = root / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


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
    centroids: list[np.ndarray] = []
    for i, s in enumerate(start_idx):
        e = start_idx[i + 1] if i + 1 < len(start_idx) else pts_sorted.shape[0]
        centroids.append(pts_sorted[s:e].mean(axis=0))
    out = np.vstack(centroids) if centroids else np.zeros((0, 3), dtype=np.float64)
    _ = unique  # kept to mirror the index ordering for documentation
    return np.asarray(out)


# ------------------------------------------------------- adopted strategy


def _min_distances_kdtree(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-point min Euclidean distance from ``a`` to ``b`` via scipy cKDTree."""
    if a.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if b.shape[0] == 0:
        return np.full((a.shape[0],), np.inf, dtype=np.float64)

    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(b, dtype=np.float64))
    distances, _ = tree.query(np.asarray(a, dtype=np.float64), k=1)
    return np.asarray(distances, dtype=np.float64)


def _error_colors(dist_m: np.ndarray, vmax_m: float) -> np.ndarray:
    """Green→red ramp for distance visualization."""
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
    """MapEval-style accuracy@τ / completeness@τ via per-point NN distances.

    Promoted from ``ca/experiments/map_evaluate/nn_thresholds.py`` as the
    adopted reference-based map evaluation strategy. The reference-free
    ``voxel_entropy`` lane stays under ``ca.experiments`` because the two
    strategies do not compete on the same metric family.
    """

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
            "chamfer_m": (
                float(np.mean(est_to_ref)) + float(np.mean(ref_to_est))
                if (est_to_ref.size and ref_to_est.size)
                else float("inf")
            ),
        }

        for t in thresholds:
            metrics[f"accuracy@{t:.3f}m"] = float(np.mean(est_to_ref <= t)) if est_to_ref.size else 0.0
            metrics[f"completeness@{t:.3f}m"] = float(np.mean(ref_to_est <= t)) if ref_to_est.size else 0.0

        # One scalar summary (F-score) using the first threshold.
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

        # Optional: MapEval-style raw/inlier error visualization PLYs.
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

        downsample_voxel = float(request.downsample_voxel_size)
        mode = "voxelized" if downsample_voxel > 0 else "exact"
        sampling_policy: dict[str, object] = {
            "downsample_voxel_size_m": downsample_voxel,
            "thresholds_m": list(thresholds),
            "align_mode": request.align_mode,
            "nn_backend": "scipy_ckdtree",
        }

        return MapEvaluateResult(
            strategy=self.name,
            design=self.design,
            metrics=metrics,
            artifacts=artifacts,
            metric_family="reference_based_nn_thresholds",
            reference_required=True,
            mode=mode,
            sampling_policy=sampling_policy,
        )


def evaluate_map(
    request: MapEvaluateRequest,
    strategy: MapEvaluateStrategy | None = None,
) -> MapEvaluateResult:
    """Evaluate a map using the stabilized strategy (defaults to NNThreshold)."""
    eval_strategy = strategy or NNThresholdMapEvaluateStrategy()
    return eval_strategy.evaluate(request)


__all__ = [
    "MapEvaluateRequest",
    "MapEvaluateResult",
    "MapEvaluateStrategy",
    "NNThresholdMapEvaluateStrategy",
    "aligned_estimated_points",
    "apply_transform",
    "ensure_artifact_dir",
    "evaluate_map",
    "voxel_downsample",
]
