"""Scan-to-map matching diagnostics."""

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from ca.io import load_point_cloud, save_point_cloud
from ca.metrics import compute_nn_distance, summarize, threshold_stats
from ca.registration import register_detailed
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
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    threshold: float | None,
) -> dict[str, Any]:
    distances = compute_nn_distance(source, target)
    result: dict[str, Any] = {"stats": summarize(distances)}
    if threshold is not None:
        result["threshold"] = threshold_stats(distances, threshold)
    return result


def _write_colored_cloud(
    point_cloud: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    path: Path,
) -> str:
    distances = compute_nn_distance(point_cloud, target)
    colored = o3d.geometry.PointCloud(point_cloud)
    colorize(colored, distances)
    save_point_cloud(str(path), colored)
    return str(path)


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
    """Debug one scan-to-map registration attempt.

    The scan is first transformed by `initial_transform`, then optionally matched
    against a cropped/downsampled map. The returned metrics compare nearest
    neighbor distance before and after registration.
    """

    scan_raw = load_point_cloud(scan_path)
    map_raw = load_point_cloud(map_path)
    initial = _as_transform(initial_transform)

    scan = _downsample_if_needed(scan_raw, scan_voxel_size)
    map_cloud = _downsample_if_needed(map_raw, map_voxel_size)

    scan_initial = o3d.geometry.PointCloud(scan)
    scan_initial.transform(initial)
    map_working = _crop_target_around_source(map_cloud, scan_initial, crop_margin)

    before = _distance_block(scan_initial, map_working, threshold)
    registration = register_detailed(
        scan_initial,
        map_working,
        method=method,
        max_correspondence_distance=max_correspondence_distance,
    )
    after = _distance_block(registration.transformed, map_working, threshold)

    final_transform = registration.transformation @ initial
    before_stats = before["stats"]
    after_stats = after["stats"]

    artifacts: dict[str, str] = {}
    if artifact_dir is not None:
        out_dir = Path(artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts["scan_initial_error_ply"] = _write_colored_cloud(
            scan_initial,
            map_working,
            out_dir / "scan_initial_error.ply",
        )
        artifacts["scan_aligned_error_ply"] = _write_colored_cloud(
            registration.transformed,
            map_working,
            out_dir / "scan_aligned_error.ply",
        )
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
            "map_points_raw": len(map_raw.points),
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
