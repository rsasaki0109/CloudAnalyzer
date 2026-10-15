"""Scan-to-map matching diagnostics."""

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from ca.io import load_point_cloud, save_point_cloud
from ca.metrics import _PreparedReference, compute_nn_distance, prepare_nn_reference, summarize, threshold_stats
from ca.registration import _TargetNormalCache, register_detailed
from ca.visualization import colorize


def _matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _as_transform(matrix: list[float] | np.ndarray | None) -> np.ndarray:
    if matrix is None:
        return np.eye(4)
    arr = np.asarray(matrix, dtype=float)
    if arr.shape == (16,):
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        raise ValueError("initial_transform must be a 4x4 matrix or 16 row-major values")
    return arr


def _downsample_if_needed(
    point_cloud: o3d.geometry.PointCloud,
    voxel_size: float | None,
) -> o3d.geometry.PointCloud:
    if voxel_size is None or voxel_size <= 0:
        return o3d.geometry.PointCloud(point_cloud)
    return point_cloud.voxel_down_sample(float(voxel_size))


def _crop_target_around_source(
    target: o3d.geometry.PointCloud,
    source: o3d.geometry.PointCloud,
    margin: float | None,
) -> o3d.geometry.PointCloud:
    if margin is None or margin <= 0:
        return o3d.geometry.PointCloud(target)
    points = np.asarray(source.points)
    min_bound = points.min(axis=0) - margin
    max_bound = points.max(axis=0) + margin
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    cropped = target.crop(bbox)
    if cropped.is_empty():
        raise ValueError(
            "Map crop is empty. Increase --crop-margin or check the initial transform."
        )
    return cropped


def _distance_block(
    distances: np.ndarray,
    threshold: float | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"stats": summarize(distances)}
    if threshold is not None:
        result["threshold"] = threshold_stats(distances, threshold)
    return result


def _write_colored_cloud(
    point_cloud: o3d.geometry.PointCloud,
    distances: np.ndarray,
    path: Path,
) -> str:
    colored = o3d.geometry.PointCloud(point_cloud)
    colorize(colored, distances)
    save_point_cloud(str(path), colored)
    return str(path)


class _ScanMatchMapContext:
    """Lazy map snapshot and uncropped reference for one multi-frame run."""

    def __init__(self, map_path: str, voxel_size: float | None):
        self.map_path = map_path
        self.voxel_size = voxel_size
        self._raw = None
        self._cloud = None
        self._raw_points = 0
        self._reference: _PreparedReference | None = None
        self._reference_ready = False
        self._target_normals = _TargetNormalCache()

    def load(self) -> int:
        if self._raw is None and self._cloud is None:
            raw = load_point_cloud(self.map_path)
            self._raw_points = len(raw.points)
            self._raw = raw
        return self._raw_points

    def cloud(self) -> o3d.geometry.PointCloud:
        if self._cloud is None:
            self._cloud = _downsample_if_needed(self._raw, self.voxel_size)
            self._raw = None
        return self._cloud

    def prepare(self, target: o3d.geometry.PointCloud, margin: float | None) -> Any:
        # Positive/nonstandard crops depend on this frame's transformed scan.
        if margin is not None and not margin <= 0:
            return prepare_nn_reference(target)
        if not self._reference_ready:
            reference = prepare_nn_reference(target)
            self._reference = reference
            self._reference_ready = True
        return self._reference

    def run(self, **kwargs: Any) -> dict[str, Any]:
        return _run_scan_match_debug(**kwargs, map_context=self)

    def invalidate(self) -> None:
        # Retry reference failures on the next frame; do not retain a pending
        # raw load after failed preprocessing/transform validation.
        self._reference = None
        self._reference_ready = False
        self._target_normals.clear()
        self._raw = None
        if self._cloud is None:
            self._raw_points = 0

    def clear(self) -> None:
        self.invalidate()
        self._cloud = None
        self._raw_points = 0


def run_scan_match_debug(
    scan_path: str,
    map_path: str,
    method: str = "gicp",
    max_correspondence_distance: float = 1.0,
    initial_transform: list[float] | np.ndarray | None = None,
    scan_voxel_size: float | None = None,
    map_voxel_size: float | None = None,
    crop_margin: float | None = None,
    threshold: float | None = None,
    artifact_dir: str | None = None,
) -> dict[str, Any]:
    """Debug one scan-to-map registration attempt with a fresh map snapshot."""
    return _run_scan_match_debug(
        scan_path, map_path, method, max_correspondence_distance, initial_transform,
        scan_voxel_size, map_voxel_size, crop_margin, threshold, artifact_dir,
    )


def _run_scan_match_debug(
    scan_path: str,
    map_path: str,
    method: str = "gicp",
    max_correspondence_distance: float = 1.0,
    initial_transform: list[float] | np.ndarray | None = None,
    scan_voxel_size: float | None = None,
    map_voxel_size: float | None = None,
    crop_margin: float | None = None,
    threshold: float | None = None,
    artifact_dir: str | None = None,
    *,
    map_context: _ScanMatchMapContext | None = None,
) -> dict[str, Any]:
    """Debug one scan-to-map registration attempt.

    The scan is first transformed by `initial_transform`, then optionally matched
    against a cropped/downsampled map. The returned metrics compare nearest
    neighbor distance before and after registration.
    """

    scan_raw = load_point_cloud(scan_path)
    if map_context is None:
        map_raw = load_point_cloud(map_path)
        map_points_raw = len(map_raw.points)
    else:
        map_points_raw = map_context.load()
    initial = _as_transform(initial_transform)

    scan = _downsample_if_needed(scan_raw, scan_voxel_size)
    map_cloud = (
        _downsample_if_needed(map_raw, map_voxel_size)
        if map_context is None else map_context.cloud()
    )

    scan_initial = o3d.geometry.PointCloud(scan)
    scan_initial.transform(initial)
    map_working = _crop_target_around_source(map_cloud, scan_initial, crop_margin)

    # Registration leaves this private map's coordinates fixed.
    prepared_reference = (
        prepare_nn_reference(map_working)
        if map_context is None else map_context.prepare(map_working, crop_margin)
    )
    before_distances = (
        compute_nn_distance(scan_initial, map_working)
        if prepared_reference is None
        else np.asarray(prepared_reference.distances(np.asarray(scan_initial.points)))
    )
    before = _distance_block(before_distances, threshold)
    if artifact_dir is None:
        del before_distances
    registration_options: dict[str, Any] = {
        "method": method,
        "max_correspondence_distance": max_correspondence_distance,
    }
    if method.lower() == "icp" and artifact_dir is None:
        registration_options["_estimate_missing_normals"] = False
    if map_context is not None and (crop_margin is None or crop_margin <= 0):
        registration_options["_target_normals"] = map_context._target_normals
    registration = register_detailed(scan_initial, map_working, **registration_options)
    after_distances = (
        compute_nn_distance(registration.transformed, map_working)
        if prepared_reference is None
        else np.asarray(prepared_reference.distances(np.asarray(registration.transformed.points)))
    )
    del prepared_reference
    after = _distance_block(after_distances, threshold)
    if artifact_dir is None:
        del after_distances

    final_transform = registration.transformation @ initial
    before_stats = before["stats"]
    after_stats = after["stats"]

    artifacts: dict[str, str] = {}
    if artifact_dir is not None:
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts["scan_initial_error_ply"] = _write_colored_cloud(
            scan_initial,
            before_distances,
            out_dir / "scan_initial_error.ply",
        )
        del before_distances
        artifacts["scan_aligned_error_ply"] = _write_colored_cloud(
            registration.transformed,
            after_distances,
            out_dir / "scan_aligned_error.ply",
        )
        del after_distances
        artifacts["map_debug_ply"] = str(out_dir / "map_debug.ply")
        save_point_cloud(artifacts["map_debug_ply"], map_working)

    return {
        "scan": scan_path,
        "map": map_path,
        "method": method.lower(),
        "max_correspondence_distance": float(max_correspondence_distance),
        "preprocess": {
            "scan_points_raw": len(scan_raw.points),
            "scan_points_used": len(scan.points),
            "map_points_raw": map_points_raw,
            "map_points_used": len(map_working.points),
            "scan_voxel_size": scan_voxel_size,
            "map_voxel_size": map_voxel_size,
            "crop_margin": crop_margin,
        },
        "registration": {
            "fitness": registration.fitness,
            "inlier_rmse": registration.rmse,
            "delta_transform": _matrix_to_list(registration.transformation),
            "initial_transform": _matrix_to_list(initial),
            "final_transform": _matrix_to_list(final_transform),
        },
        "distance_before": before,
        "distance_after": after,
        "improvement": {
            "mean": float(before_stats["mean"] - after_stats["mean"]),
            "median": float(before_stats["median"] - after_stats["median"]),
            "max": float(before_stats["max"] - after_stats["max"]),
        },
        "artifacts": artifacts,
    }
