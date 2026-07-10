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


def _regularized_covariance(sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    sym = (np.asarray(sigma, dtype=np.float64) + np.asarray(sigma, dtype=np.float64).T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def wasserstein_distance_gaussian(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """L2 Wasserstein distance between 3D Gaussians (MapEval eq. 6)."""
    mu_diff = np.asarray(mu1, dtype=np.float64) - np.asarray(mu2, dtype=np.float64)
    s1 = _regularized_covariance(sigma1)
    s2 = _regularized_covariance(sigma2)
    # The Bures term needs the *symmetric positive-semidefinite* matrix
    # square root.  A Cholesky factor is not a matrix square root and its
    # trace gives incorrect distances for non-diagonal covariances.
    eigvals, eigvecs = np.linalg.eigh(s1)
    sqrt_s1 = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0.0))) @ eigvecs.T
    middle = _regularized_covariance(sqrt_s1 @ s2 @ sqrt_s1)
    middle_eigvals = np.linalg.eigvalsh(middle)
    trace_sqrt_middle = float(np.sqrt(np.maximum(middle_eigvals, 0.0)).sum())
    dist_sq = float(mu_diff @ mu_diff) + float(np.trace(s1 + s2)) - 2.0 * trace_sqrt_middle
    # Roundoff around identical distributions can leave a few ulps above zero.
    if abs(dist_sq) <= 1e-12 * max(float(np.trace(s1 + s2)), 1.0):
        dist_sq = 0.0
    return float(np.sqrt(max(0.0, dist_sq)))


@dataclass(slots=True)
class VoxelGaussian:
    mu: np.ndarray
    sigma: np.ndarray
    num_points: int


def build_voxel_gaussians(points: np.ndarray, voxel_size: float) -> dict[tuple[int, int, int], VoxelGaussian]:
    """Voxelize a point cloud into per-voxel Gaussian summaries."""
    pts = _require_xyz(points, "points")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0.")
    if pts.shape[0] == 0:
        return {}
    # Sort once and reduce contiguous groups.  This avoids one Python list and
    # one small ndarray allocation per input point on million-point maps.
    indices = np.floor(pts / voxel_size).astype(np.int64)
    unique, inverse, counts = np.unique(indices, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((unique.shape[0], 3), dtype=np.float64)
    np.add.at(sums, inverse, pts)
    means = sums / counts[:, None]
    centered = pts - means[inverse]
    covariance_sums = np.zeros((unique.shape[0], 3, 3), dtype=np.float64)
    np.add.at(covariance_sums, inverse, centered[:, :, None] * centered[:, None, :])
    voxels: dict[tuple[int, int, int], VoxelGaussian] = {}
    for raw_key, mu, covariance_sum, raw_count in zip(unique, means, covariance_sums, counts):
        count = int(raw_count)
        if count > 1:
            sigma = covariance_sum / (count - 1)
        else:
            sigma = np.eye(3, dtype=np.float64) * 1e-6
        key = tuple(int(value) for value in raw_key)
        voxels[key] = VoxelGaussian(mu=mu, sigma=sigma, num_points=count)
    return voxels


def _neighbor_indices(index: tuple[int, int, int], radius: int) -> list[tuple[int, int, int]]:
    ix, iy, iz = index
    neighbors: list[tuple[int, int, int]] = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbors.append((ix + dx, iy + dy, iz + dz))
    return neighbors


def compute_voxel_wasserstein_metrics(
    estimated_points: np.ndarray,
    reference_points: np.ndarray,
    *,
    voxel_size: float,
    min_voxel_points: int = 100,
    neighbor_radius: int = 5,
) -> dict[str, float]:
    """Compute MapEval-style AWD and SCS from voxelized Gaussian summaries."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0 for AWD/SCS.")
    est_map = build_voxel_gaussians(estimated_points, voxel_size)
    ref_map = build_voxel_gaussians(reference_points, voxel_size)

    wasserstein_distances: dict[tuple[int, int, int], float] = {}
    for index, est_voxel in est_map.items():
        ref_voxel = ref_map.get(index)
        if ref_voxel is None:
            continue
        if est_voxel.num_points < min_voxel_points or ref_voxel.num_points < min_voxel_points:
            continue
        wasserstein_distances[index] = wasserstein_distance_gaussian(
            est_voxel.mu,
            est_voxel.sigma,
            ref_voxel.mu,
            ref_voxel.sigma,
        )

    if not wasserstein_distances:
        return {
            "awd_m": float("nan"),
            "scs": float("nan"),
            "n_awd_voxels": 0.0,
            "n_scs_voxels": 0.0,
        }

    ws_values = list(wasserstein_distances.values())
    awd_m = float(np.mean(ws_values))

    scs_terms: list[float] = []
    for index in wasserstein_distances:
        neighbor_ws = [
            wasserstein_distances[neighbor]
            for neighbor in _neighbor_indices(index, neighbor_radius)
            if neighbor in wasserstein_distances
        ]
        if not neighbor_ws:
            continue
        mean_neighbor = float(np.mean(neighbor_ws))
        if mean_neighbor <= np.finfo(np.float64).eps:
            # A zero-error neighborhood is perfectly consistent.
            scs_terms.append(0.0 if np.allclose(neighbor_ws, 0.0) else float("nan"))
            continue
        std_neighbor = float(np.std(neighbor_ws))
        scs_terms.append(std_neighbor / mean_neighbor)

    scs = float(np.nanmean(scs_terms)) if scs_terms else float("nan")
    return {
        "awd_m": awd_m,
        "scs": scs,
        "n_awd_voxels": float(len(wasserstein_distances)),
        "n_scs_voxels": float(len(scs_terms)),
    }


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

        structure_voxel = float(request.structure_voxel_size)
        if structure_voxel > 0:
            voxel_metrics = compute_voxel_wasserstein_metrics(
                est,
                ref,
                voxel_size=structure_voxel,
            )
            # Keep the public metric mapping JSON/comparison friendly: an
            # unavailable metric is represented by its zero support count,
            # rather than a NaN value that is unequal to itself.
            metrics.update(
                {
                    key: value
                    for key, value in voxel_metrics.items()
                    if not key.startswith(("awd_", "scs")) or np.isfinite(value)
                }
            )
            metrics["n_awd_voxels"] = voxel_metrics["n_awd_voxels"]
            metrics["n_scs_voxels"] = voxel_metrics["n_scs_voxels"]

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
            "structure_voxel_size_m": structure_voxel,
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
    "VoxelGaussian",
    "aligned_estimated_points",
    "apply_transform",
    "build_voxel_gaussians",
    "compute_voxel_wasserstein_metrics",
    "ensure_artifact_dir",
    "evaluate_map",
    "voxel_downsample",
    "wasserstein_distance_gaussian",
]
