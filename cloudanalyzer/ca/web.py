"""Web-based 3D point cloud viewer using Three.js."""

import json
import re
import shutil
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from ca.core.web_sampling import reduce_point_cloud_for_web
from ca.core.web_progressive_loading import plan_progressive_loading_for_web
from ca.core.web_trajectory_sampling import reduce_trajectory_for_web
from ca.io import load_point_cloud
from ca.log import logger
from ca.metrics import compute_nn_distance
from ca.trajectory import evaluate_trajectory, load_trajectory

_MAX_TRAJECTORY_DISPLAY_POINTS = 4_000
_MIN_TRAJECTORY_DISPLAY_POINTS = 200
_MAX_PROGRESSIVE_INITIAL_POINTS = 120_000
_MIN_PROGRESSIVE_INITIAL_POINTS = 20_000
_MAX_PROGRESSIVE_CHUNK_POINTS = 60_000
_MIN_PROGRESSIVE_CHUNK_POINTS = 10_000

# The viewer page is a real file so it can be edited and checked as HTML.
_VIEWER_HTML = files("ca").joinpath("web_viewer.html").read_text(encoding="utf-8")


def _progressive_initial_budget(max_points: int, display_points: int) -> int:
    """Choose a bounded initial payload size for browser loading."""
    if display_points <= 0:
        return 0
    budget = max(
        _MIN_PROGRESSIVE_INITIAL_POINTS,
        min(max_points // 12, _MAX_PROGRESSIVE_INITIAL_POINTS),
    )
    return min(display_points, max(1, budget))


def _progressive_chunk_budget(max_points: int, display_points: int) -> int:
    """Choose a bounded deferred chunk size."""
    if display_points <= 0:
        return 0
    budget = max(
        _MIN_PROGRESSIVE_CHUNK_POINTS,
        min(max_points // 16, _MAX_PROGRESSIVE_CHUNK_POINTS),
    )
    return min(display_points, max(1, budget))


def _prepare_progressive_loading_payload(
    positions: np.ndarray,
    max_points: int,
    label: str,
    distances: np.ndarray | None = None,
    stream_name: str = "source",
) -> tuple[np.ndarray, np.ndarray | None, dict, dict[str, str]]:
    """Plan a small initial payload plus deferred chunk responses."""

    display_points = int(positions.shape[0])
    if display_points == 0:
        return (
            positions,
            distances,
            {
                "enabled": False,
                "strategy": None,
                "design": None,
                "initial_points": 0,
                "total_points": 0,
                "chunk_count": 0,
                "chunks": [],
            },
            {},
        )

    initial_points = _progressive_initial_budget(max_points, display_points)
    if display_points <= initial_points:
        return (
            positions,
            distances,
            {
                "enabled": False,
                "strategy": None,
                "design": None,
                "initial_points": display_points,
                "total_points": display_points,
                "chunk_count": 0,
                "chunks": [],
            },
            {},
        )

    plan = plan_progressive_loading_for_web(
        positions=positions,
        initial_points=initial_points,
        chunk_points=_progressive_chunk_budget(max_points, display_points),
        distances=distances,
        label=label,
    )
    chunk_payloads: dict[str, str] = {}
    chunk_descriptors = []
    for chunk_index, chunk in enumerate(plan.chunks):
        path = f"chunks/{stream_name}/{chunk_index}.json"
        chunk_payloads[path] = json.dumps(
            {
                "positions": chunk.positions.flatten().tolist(),
                "distances": (
                    chunk.distances.tolist() if chunk.distances is not None else None
                ),
            }
        )
        chunk_descriptors.append(
            {
                "path": path,
                "points": chunk.point_count,
            }
        )

    return (
        plan.initial_positions,
        plan.initial_distances,
        {
            "enabled": True,
            "strategy": plan.strategy,
            "design": plan.design,
            "initial_points": plan.initial_points,
            "total_points": plan.total_displayed_points,
            "chunk_count": len(plan.chunks),
            "chunks": chunk_descriptors,
        },
        chunk_payloads,
    )


def _scene_bounds(*arrays: np.ndarray) -> dict[str, list[float] | float]:
    """Compute full-scene bounds from already prepared display arrays."""

    valid_arrays = [array for array in arrays if array.size > 0]
    if not valid_arrays:
        return {
            "center": [0.0, 0.0, 0.0],
            "extent": 1.0,
        }
    stacked = np.vstack(valid_arrays)
    mins = np.min(stacked, axis=0)
    maxs = np.max(stacked, axis=0)
    center = (mins + maxs) / 2.0
    extent = float(np.max(maxs - mins))
    return {
        "center": center.tolist(),
        "extent": extent if extent > 0 else 1.0,
    }


def _make_handler(html: str, data_json: str, chunk_payloads: dict[str, str] | None = None):
    """Create a custom HTTP handler serving the viewer and data."""
    payloads = chunk_payloads or {}

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split('?', 1)[0]
            normalized = path.lstrip('/')
            if path == '/' or normalized == 'index.html':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())
            elif normalized == 'data.json':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(data_json.encode())
            elif normalized in payloads:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(payloads[normalized].encode())
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass  # Suppress default logging

    return Handler


def _downsample_for_web(
    pcd: o3d.geometry.PointCloud,
    max_points: int,
    label: str,
) -> o3d.geometry.PointCloud:
    """Reduce a point cloud for browser display via the stable core interface."""
    result = reduce_point_cloud_for_web(pcd, max_points=max_points, label=label)
    if result.reduced_points < result.original_points:
        logger.info(
            "Reduced %s to %d points for web display using %s",
            label,
            result.reduced_points,
            result.strategy,
        )
    return result.point_cloud


def _trajectory_display_budget(max_points: int, pose_count: int) -> int:
    """Pick a browser-safe trajectory budget without tying it too tightly to cloud density."""
    if pose_count <= 0:
        return 0
    return min(
        pose_count,
        max(_MIN_TRAJECTORY_DISPLAY_POINTS, min(max_points, _MAX_TRAJECTORY_DISPLAY_POINTS)),
    )


def _downsample_trajectory_for_web(
    positions: np.ndarray,
    timestamps: np.ndarray,
    max_points: int,
    label: str,
    preserve_indices: tuple[int, ...] = (),
):
    """Reduce trajectory overlays for browser display via the stable core interface."""
    result = reduce_trajectory_for_web(
        positions=positions,
        timestamps=timestamps,
        max_points=_trajectory_display_budget(max_points, positions.shape[0]),
        label=label,
        preserve_indices=preserve_indices,
    )
    if result.reduced_points < result.original_points:
        logger.info(
            "Reduced %s to %d poses for web display using %s",
            label,
            result.reduced_points,
            result.strategy,
        )
    return result


def _summarize_error_series(values: np.ndarray) -> dict[str, float]:
    """Summarize an error series for viewer metadata."""
    if values.size == 0:
        return {"rmse": 0.0, "max": 0.0, "mean": 0.0, "min": 0.0}
    return {
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
    }


def _optional_float(value) -> float | None:
    """Return finite float metadata for viewer JSON, or None."""

    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _nested_float(document: dict, *keys: str) -> float | None:
    """Return a finite float from nested report metadata, or None."""

    value: Any = document
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _optional_float(value)


def _optional_int(value) -> int | None:
    """Return integer metadata for viewer JSON, or None."""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _prepare_scan_match_debug_summary(scan_debug: dict) -> dict:
    """Extract compact scan-match debug metrics for the viewer panel."""

    preprocess = scan_debug.get("preprocess") if isinstance(scan_debug, dict) else {}
    summary = {
        "method": scan_debug.get("method"),
        "max_correspondence_distance": _optional_float(
            scan_debug.get("max_correspondence_distance")
        ),
        "registration_fitness": _nested_float(scan_debug, "registration", "fitness"),
        "registration_inlier_rmse_m": _nested_float(
            scan_debug, "registration", "inlier_rmse"
        ),
        "distance_before_mean_m": _nested_float(
            scan_debug, "distance_before", "stats", "mean"
        ),
        "distance_after_mean_m": _nested_float(
            scan_debug, "distance_after", "stats", "mean"
        ),
        "distance_before_median_m": _nested_float(
            scan_debug, "distance_before", "stats", "median"
        ),
        "distance_after_median_m": _nested_float(
            scan_debug, "distance_after", "stats", "median"
        ),
        "mean_improvement_m": _nested_float(scan_debug, "improvement", "mean"),
        "median_improvement_m": _nested_float(scan_debug, "improvement", "median"),
        "max_improvement_m": _nested_float(scan_debug, "improvement", "max"),
        "scan_points_used": _optional_int(preprocess.get("scan_points_used"))
        if isinstance(preprocess, dict)
        else None,
        "map_points_used": _optional_int(preprocess.get("map_points_used"))
        if isinstance(preprocess, dict)
        else None,
    }
    return {key: value for key, value in summary.items() if value is not None}


def _prepare_slam_debug_frames_for_viewer(
    report_path: str,
    trajectory_data: dict,
) -> tuple[list[dict], dict]:
    """Project selected SLAM-debug frames onto displayed trajectory timestamps."""

    path = Path(report_path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid SLAM debug report JSON: {path}: {exc.msg}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"SLAM debug report must be a JSON object: {path}")
    selected_frames = report.get("selected_frames", [])
    if not isinstance(selected_frames, list):
        raise ValueError(f"SLAM debug report selected_frames must be a list: {path}")

    timestamps = np.asarray(trajectory_data.get("timestamps") or [], dtype=float)
    frames: list[dict] = []
    for item in selected_frames:
        if not isinstance(item, dict):
            continue
        timestamp = _optional_float(item.get("timestamp_sec"))
        if timestamp is None or timestamps.size == 0:
            continue
        display_index = int(np.argmin(np.abs(timestamps - timestamp)))
        closest_delta = float(abs(timestamps[display_index] - timestamp))
        scan_debug = item.get("scan_match_debug_result")
        artifacts = scan_debug.get("artifacts") if isinstance(scan_debug, dict) else None
        scan_debug_summary = (
            _prepare_scan_match_debug_summary(scan_debug)
            if isinstance(scan_debug, dict)
            else None
        )
        diagnosis = item.get("diagnosis")
        frames.append(
            {
                "rank": int(item.get("rank", len(frames) + 1)),
                "scan_id": str(item.get("scan_id", "")),
                "timestamp_sec": timestamp,
                "display_index": display_index,
                "closest_display_time_delta_sec": closest_delta,
                "score": _optional_float(item.get("score")),
                "reasons": item.get("reasons") if isinstance(item.get("reasons"), list) else [],
                "scan_match_rmse_m": _optional_float(item.get("scan_match_rmse_m")),
                "scan_match_weighted_rmse": _optional_float(
                    item.get("scan_match_weighted_rmse")
                ),
                "prediction_delta_m": _optional_float(item.get("prediction_delta_m")),
                "initial_delta_m": _optional_float(item.get("initial_delta_m")),
                "diagnosis": diagnosis if isinstance(diagnosis, dict) else None,
                "scan_path": item.get("scan_path"),
                "scan_match_debug_command": item.get("scan_match_debug_command"),
                "scan_match_debug_summary": scan_debug_summary,
                "artifacts": artifacts if isinstance(artifacts, dict) else {},
            }
        )

    return frames, {
        "report_path": str(path),
        "total_frames": len(selected_frames),
        "displayed_frames": len(frames),
    }


def _safe_export_component(value: object, fallback: str) -> str:
    """Return a filesystem-safe path component for copied viewer assets."""

    text = str(value).strip() if value is not None else ""
    if not text:
        text = fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return (text or fallback)[:120]


def _resolve_slam_debug_artifact_path(path_value: str, report_dir: Path | None) -> Path | None:
    """Resolve an artifact path from a SLAM debug report to an existing file."""

    source = Path(path_value)
    candidates = [source]
    if not source.is_absolute() and report_dir is not None:
        candidates.insert(0, report_dir / source)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _copy_slam_debug_artifacts_for_export(data: dict, output_dir: Path) -> int:
    """Copy SLAM debug artifact files into a static viewer bundle."""

    trajectory = data.get("trajectory")
    if not isinstance(trajectory, dict):
        return 0
    frames = trajectory.get("slam_debug_frames")
    if not isinstance(frames, list):
        return 0
    summary = trajectory.get("slam_debug")
    report_path = (
        Path(summary["report_path"])
        if isinstance(summary, dict) and isinstance(summary.get("report_path"), str)
        else None
    )
    report_dir = report_path.parent if report_path is not None else None

    copied = 0
    asset_root = output_dir / "slam_debug_artifacts"
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        artifacts = frame.get("artifacts")
        if not isinstance(artifacts, dict):
            continue

        rank = frame.get("rank", frame_index + 1)
        try:
            rank_text = f"{int(rank):02d}"
        except (TypeError, ValueError):
            rank_text = f"{frame_index + 1:02d}"
        scan_id = _safe_export_component(frame.get("scan_id"), f"frame_{frame_index + 1}")
        frame_dir = asset_root / f"{rank_text}_{scan_id}"
        assets: dict[str, str] = {}
        for key, path_value in artifacts.items():
            if not isinstance(path_value, str) or not path_value:
                continue
            source = _resolve_slam_debug_artifact_path(path_value, report_dir)
            if source is None:
                continue
            safe_key = _safe_export_component(key, "artifact")
            safe_name = _safe_export_component(
                f"{safe_key}_{source.name}",
                f"{safe_key}{source.suffix}",
            )
            destination = frame_dir / safe_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            assets[str(key)] = destination.relative_to(output_dir).as_posix()
            copied += 1
        if assets:
            frame["artifact_assets"] = assets

    return copied


def _prepare_viewer_bundle(
    paths: list[str],
    max_points: int = 2_000_000,
    heatmap: bool = False,
    trajectory_path: str | None = None,
    trajectory_reference_path: str | None = None,
    trajectory_max_time_delta: float = 0.05,
    trajectory_align_origin: bool = False,
    trajectory_align_rigid: bool = False,
    slam_debug_report_path: str | None = None,
) -> tuple[dict, dict[str, str]]:
    """Prepare browser payload plus deferred chunk payloads."""
    if trajectory_reference_path is not None and trajectory_path is None:
        raise ValueError("--trajectory-reference requires --trajectory")
    if trajectory_path is None and (trajectory_align_origin or trajectory_align_rigid):
        raise ValueError("trajectory alignment options require --trajectory")
    if trajectory_path is None and slam_debug_report_path is not None:
        raise ValueError("--slam-debug-report requires --trajectory")
    if trajectory_path is not None and trajectory_reference_path is None and (
        trajectory_align_origin or trajectory_align_rigid
    ):
        raise ValueError(
            "trajectory alignment options require --trajectory-reference"
        )

    trajectory_data = None
    if trajectory_path is not None:
        trajectory_data = _prepare_trajectory_viewer_data(
            trajectory_path,
            trajectory_reference_path=trajectory_reference_path,
            max_points=max_points,
            max_time_delta=trajectory_max_time_delta,
            align_origin=trajectory_align_origin,
            align_rigid=trajectory_align_rigid,
        )
        if slam_debug_report_path is not None:
            frames, summary = _prepare_slam_debug_frames_for_viewer(
                slam_debug_report_path,
                trajectory_data,
            )
            trajectory_data["slam_debug"] = summary
            trajectory_data["slam_debug_frames"] = frames
    trajectory_positions = (
        np.asarray(trajectory_data["estimated_positions"], dtype=float).reshape(-1, 3)
        if trajectory_data and trajectory_data.get("estimated_positions")
        else np.zeros((0, 3), dtype=float)
    )
    trajectory_reference_positions = (
        np.asarray(trajectory_data["reference_positions"], dtype=float).reshape(-1, 3)
        if trajectory_data and trajectory_data.get("reference_positions")
        else np.zeros((0, 3), dtype=float)
    )

    if not paths:
        if heatmap:
            raise ValueError("--heatmap requires exactly 2 point cloud files")
        if trajectory_data is None:
            raise ValueError("At least one point cloud or --trajectory is required")
        data = {
            "positions": [],
            "filename": Path(trajectory_path).name if trajectory_path is not None else "",
            "viewer_mode": "trajectory",
            "display_points": 0,
            "initial_display_points": 0,
            "original_points": 0,
            "source_z_bounds": [
                float(np.min(trajectory_positions[:, 2])) if trajectory_positions.size else 0.0,
                float(np.max(trajectory_positions[:, 2])) if trajectory_positions.size else 0.0,
            ],
            "scene_bounds": _scene_bounds(
                trajectory_positions,
                trajectory_reference_positions,
            ),
            "progressive_loading": {
                "enabled": False,
                "source": None,
                "reference": None,
            },
            "trajectory": trajectory_data,
        }
        return data, {}

    if heatmap:
        if len(paths) != 2:
            raise ValueError("--heatmap requires exactly 2 point cloud files")

        source_path, target_path = paths
        source = load_point_cloud(source_path)
        target = load_point_cloud(target_path)
        logger.info("Loaded %d source points from %s", len(source.points), source_path)
        logger.info("Loaded %d reference points from %s", len(target.points), target_path)

        display_budget = max(1, max_points // 2)
        display_source = _downsample_for_web(source, display_budget, "source cloud")
        display_target = _downsample_for_web(target, display_budget, "reference cloud")
        source_positions = np.asarray(display_source.points, dtype=float)
        target_positions = np.asarray(display_target.points, dtype=float)
        distances = compute_nn_distance(display_source, target)
        (
            initial_source_positions,
            initial_source_distances,
            source_progressive,
            source_chunk_payloads,
        ) = _prepare_progressive_loading_payload(
            positions=source_positions,
            max_points=max_points,
            label="source cloud",
            distances=distances,
            stream_name="source",
        )
        (
            initial_reference_positions,
            _,
            reference_progressive,
            reference_chunk_payloads,
        ) = _prepare_progressive_loading_payload(
            positions=target_positions,
            max_points=max_points,
            label="reference cloud",
            stream_name="reference",
        )
        chunk_payloads = {**source_chunk_payloads, **reference_chunk_payloads}
        progressive_enabled = bool(
            source_progressive["enabled"] or reference_progressive["enabled"]
        )

        data = {
            "positions": initial_source_positions.flatten().tolist(),
            "reference_positions": initial_reference_positions.flatten().tolist(),
            "filename": Path(source_path).name,
            "viewer_mode": "heatmap",
            "source_filename": Path(source_path).name,
            "target_filename": Path(target_path).name,
            "display_points": int(source_positions.shape[0]),
            "initial_display_points": int(initial_source_positions.shape[0]),
            "original_points": len(source.points),
            "reference_points": len(target.points),
            "reference_display_points": int(target_positions.shape[0]),
            "reference_initial_display_points": int(initial_reference_positions.shape[0]),
            "distances": (
                initial_source_distances.tolist() if initial_source_distances is not None else None
            ),
            "distance_stats": {
                "mean": float(np.mean(distances)),
                "max": float(np.max(distances)),
                "min": float(np.min(distances)),
            },
            "source_z_bounds": [
                float(np.min(source_positions[:, 2])) if source_positions.size else 0.0,
                float(np.max(source_positions[:, 2])) if source_positions.size else 0.0,
            ],
            "scene_bounds": _scene_bounds(
                source_positions,
                target_positions,
                trajectory_positions,
                trajectory_reference_positions,
            ),
            "progressive_loading": {
                "enabled": progressive_enabled,
                "source": source_progressive if source_progressive["enabled"] else None,
                "reference": (
                    reference_progressive if reference_progressive["enabled"] else None
                ),
            },
        }
        if trajectory_data is not None:
            data["trajectory"] = trajectory_data
        return data, chunk_payloads

    merged = o3d.geometry.PointCloud()
    for path in paths:
        merged += load_point_cloud(path)

    total = len(merged.points)
    logger.info("Loaded %d points from %d file(s)", total, len(paths))
    display_cloud = _downsample_for_web(merged, max_points, "merged cloud")
    display_positions = np.asarray(display_cloud.points, dtype=float)
    (
        initial_positions,
        _,
        source_progressive,
        source_chunk_payloads,
    ) = _prepare_progressive_loading_payload(
        positions=display_positions,
        max_points=max_points,
        label="merged cloud",
        stream_name="source",
    )

    data = {
        "positions": initial_positions.flatten().tolist(),
        "filename": ", ".join(Path(p).name for p in paths),
        "viewer_mode": "standard",
        "display_points": int(display_positions.shape[0]),
        "initial_display_points": int(initial_positions.shape[0]),
        "original_points": total,
        "source_z_bounds": [
            float(np.min(display_positions[:, 2])) if display_positions.size else 0.0,
            float(np.max(display_positions[:, 2])) if display_positions.size else 0.0,
        ],
        "scene_bounds": _scene_bounds(
            display_positions,
            trajectory_positions,
            trajectory_reference_positions,
        ),
        "progressive_loading": {
            "enabled": bool(source_progressive["enabled"]),
            "source": source_progressive if source_progressive["enabled"] else None,
            "reference": None,
        },
    }
    if trajectory_data is not None:
        data["trajectory"] = trajectory_data
    return data, source_chunk_payloads


def _prepare_viewer_data(
    paths: list[str],
    max_points: int = 2_000_000,
    heatmap: bool = False,
    trajectory_path: str | None = None,
    trajectory_reference_path: str | None = None,
    trajectory_max_time_delta: float = 0.05,
    trajectory_align_origin: bool = False,
    trajectory_align_rigid: bool = False,
    slam_debug_report_path: str | None = None,
) -> dict:
    """Prepare browser payload for standard or heatmap web viewing."""
    data, _ = _prepare_viewer_bundle(
        paths,
        max_points=max_points,
        heatmap=heatmap,
        trajectory_path=trajectory_path,
        trajectory_reference_path=trajectory_reference_path,
        trajectory_max_time_delta=trajectory_max_time_delta,
        trajectory_align_origin=trajectory_align_origin,
        trajectory_align_rigid=trajectory_align_rigid,
        slam_debug_report_path=slam_debug_report_path,
    )
    return data


def _prepare_trajectory_viewer_data(
    trajectory_path: str,
    trajectory_reference_path: str | None = None,
    max_points: int = 2_000_000,
    max_time_delta: float = 0.05,
    align_origin: bool = False,
    align_rigid: bool = False,
) -> dict:
    """Prepare trajectory overlay payload for the web viewer."""
    if trajectory_reference_path is None:
        trajectory = load_trajectory(trajectory_path)
        positions = np.asarray(trajectory["positions"], dtype=float)
        timestamps = np.asarray(trajectory["timestamps"], dtype=float)
        reduced = _downsample_trajectory_for_web(
            positions=positions,
            timestamps=timestamps,
            max_points=max_points,
            label=f"trajectory {Path(trajectory_path).name}",
        )
        return {
            "mode": "single",
            "estimated_filename": Path(trajectory_path).name,
            "estimated_positions": reduced.positions.flatten().tolist(),
            "estimated_pose_count": int(positions.shape[0]),
            "displayed_estimated_pose_count": int(reduced.reduced_points),
            "reference_positions": None,
            "reference_pose_count": None,
            "displayed_reference_pose_count": None,
            "timestamps": (
                reduced.timestamps.tolist() if reduced.timestamps is not None else None
            ),
            "ate_errors": None,
            "rpe_timestamps": None,
            "rpe_errors": None,
            "worst_ate_index": None,
            "worst_ate_sample": None,
            "worst_rpe_index": None,
            "worst_rpe_segment": None,
            "sampling": {
                "strategy": reduced.strategy,
                "design": reduced.design,
                "display_budget": _trajectory_display_budget(max_points, positions.shape[0]),
                "reduction_ratio": reduced.reduction_ratio,
            },
        }

    result = evaluate_trajectory(
        trajectory_path,
        trajectory_reference_path,
        max_time_delta=max_time_delta,
        align_origin=align_origin,
        align_rigid=align_rigid,
    )
    matched = result["matched_trajectory"]
    estimated_positions = np.asarray(matched["estimated_positions"], dtype=float)
    reference_positions = np.asarray(matched["reference_positions"], dtype=float)
    timestamps = np.asarray(matched["timestamps"], dtype=float)
    full_ate_errors = np.asarray(matched["ate_errors"], dtype=float)
    full_rpe_errors = np.asarray(result["error_series"]["rpe_translation"], dtype=float)
    worst_ate_source_index = int(np.argmax(full_ate_errors)) if full_ate_errors.size > 0 else None
    worst_rpe_source_index = int(np.argmax(full_rpe_errors)) if full_rpe_errors.size > 0 else None

    preserve_indices = tuple(
        index
        for index in (
            worst_ate_source_index,
            worst_rpe_source_index,
            (
                worst_rpe_source_index + 1
                if worst_rpe_source_index is not None
                else None
            ),
        )
        if index is not None
    )
    reduced = _downsample_trajectory_for_web(
        positions=estimated_positions,
        timestamps=timestamps,
        max_points=max_points,
        label=f"matched trajectory {Path(trajectory_path).name}",
        preserve_indices=preserve_indices,
    )
    kept_indices = reduced.kept_indices
    displayed_reference_positions = reference_positions[kept_indices]
    displayed_timestamps = (
        reduced.timestamps
        if reduced.timestamps is not None
        else timestamps[kept_indices]
    )
    ate_errors = full_ate_errors[kept_indices]
    if reduced.positions.shape[0] >= 2:
        rpe_errors = np.linalg.norm(
            np.diff(reduced.positions, axis=0) - np.diff(displayed_reference_positions, axis=0),
            axis=1,
        )
        rpe_timestamps = (
            (displayed_timestamps[:-1] + displayed_timestamps[1:]) / 2.0
        )
    else:
        rpe_errors = np.zeros(0, dtype=float)
        rpe_timestamps = np.zeros(0, dtype=float)

    ate_summary = _summarize_error_series(ate_errors)
    rpe_summary = _summarize_error_series(rpe_errors)
    kept_lookup = {int(index): pos for pos, index in enumerate(kept_indices.tolist())}
    worst_ate_index = (
        kept_lookup.get(worst_ate_source_index)
        if worst_ate_source_index is not None
        else None
    )
    worst_rpe_index = int(np.argmax(rpe_errors)) if rpe_errors.size > 0 else None
    return {
        "mode": "paired",
        "estimated_filename": Path(trajectory_path).name,
        "reference_filename": Path(trajectory_reference_path).name,
        "estimated_positions": reduced.positions.flatten().tolist(),
        "reference_positions": displayed_reference_positions.flatten().tolist(),
        "estimated_pose_count": int(estimated_positions.shape[0]),
        "reference_pose_count": int(reference_positions.shape[0]),
        "displayed_estimated_pose_count": int(reduced.reduced_points),
        "displayed_reference_pose_count": int(displayed_reference_positions.shape[0]),
        "timestamps": displayed_timestamps.tolist(),
        "alignment": result["alignment"],
        "matching": {
            "matched_poses": result["matching"]["matched_poses"],
            "coverage_ratio": result["matching"]["coverage_ratio"],
        },
        "ate": {"rmse": ate_summary["rmse"], "max": ate_summary["max"]},
        "rpe": {"rmse": rpe_summary["rmse"], "max": rpe_summary["max"]},
        "drift": {
            "endpoint": result["drift"]["endpoint"],
        },
        "ate_errors": ate_errors.tolist(),
        "rpe_timestamps": rpe_timestamps.tolist(),
        "rpe_errors": rpe_errors.tolist(),
        "error_stats": ate_summary,
        "worst_ate_index": worst_ate_index,
        "worst_ate_sample": (
            result["worst_ate_samples"][0] if result["worst_ate_samples"] else None
        ),
        "worst_rpe_index": worst_rpe_index,
        "worst_rpe_segment": (
            {
                "start_timestamp": float(displayed_timestamps[worst_rpe_index]),
                "end_timestamp": float(displayed_timestamps[worst_rpe_index + 1]),
                "translation_error": float(rpe_errors[worst_rpe_index]),
            }
            if worst_rpe_index is not None
            else None
        ),
        "sampling": {
            "strategy": reduced.strategy,
            "design": reduced.design,
            "display_budget": _trajectory_display_budget(max_points, estimated_positions.shape[0]),
            "reduction_ratio": reduced.reduction_ratio,
            "preserve_indices": list(preserve_indices),
        },
    }


def serve(
    paths: list[str],
    port: int = 8080,
    max_points: int = 2_000_000,
    open_browser: bool = True,
    heatmap: bool = False,
    trajectory_path: str | None = None,
    trajectory_reference_path: str | None = None,
    trajectory_max_time_delta: float = 0.05,
    trajectory_align_origin: bool = False,
    trajectory_align_rigid: bool = False,
    slam_debug_report_path: str | None = None,
) -> None:
    """Start a web viewer for point cloud(s).

    Args:
        paths: Point cloud file paths.
        port: HTTP port.
        max_points: Max points to display (downsample if larger).
        open_browser: Auto-open browser.
        heatmap: If True, color the first point cloud by distance to the second.
        trajectory_path: Optional estimated trajectory to overlay.
        trajectory_reference_path: Optional reference trajectory for aligned overlay.
        trajectory_max_time_delta: Matching tolerance for paired trajectory overlays.
        trajectory_align_origin: Apply origin alignment to trajectory overlay.
        trajectory_align_rigid: Apply rigid alignment to trajectory overlay.
    """
    data, chunk_payloads = _prepare_viewer_bundle(
        paths,
        max_points=max_points,
        heatmap=heatmap,
        trajectory_path=trajectory_path,
        trajectory_reference_path=trajectory_reference_path,
        trajectory_max_time_delta=trajectory_max_time_delta,
        trajectory_align_origin=trajectory_align_origin,
        trajectory_align_rigid=trajectory_align_rigid,
        slam_debug_report_path=slam_debug_report_path,
    )
    data_json = json.dumps(data)

    handler = _make_handler(_VIEWER_HTML, data_json, chunk_payloads)
    server = HTTPServer(('0.0.0.0', port), handler)

    url = f"http://localhost:{port}"
    logger.info("Serving at %s", url)
    print(f"CloudAnalyzer Web Viewer: {url}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


def export_static_bundle(
    paths: list[str],
    output_dir: str,
    max_points: int = 2_000_000,
    heatmap: bool = False,
    trajectory_path: str | None = None,
    trajectory_reference_path: str | None = None,
    trajectory_max_time_delta: float = 0.05,
    trajectory_align_origin: bool = False,
    trajectory_align_rigid: bool = False,
    slam_debug_report_path: str | None = None,
) -> dict:
    """Write a static browser bundle that can be served from GitHub Pages or any static host."""

    data, chunk_payloads = _prepare_viewer_bundle(
        paths,
        max_points=max_points,
        heatmap=heatmap,
        trajectory_path=trajectory_path,
        trajectory_reference_path=trajectory_reference_path,
        trajectory_max_time_delta=trajectory_max_time_delta,
        trajectory_align_origin=trajectory_align_origin,
        trajectory_align_rigid=trajectory_align_rigid,
        slam_debug_report_path=slam_debug_report_path,
    )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    copied_slam_debug_artifacts = _copy_slam_debug_artifacts_for_export(data, root)
    (root / "index.html").write_text(_VIEWER_HTML)
    (root / "data.json").write_text(json.dumps(data, indent=2))

    for relative_path, payload in chunk_payloads.items():
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload)

    exported_files = 2 + len(chunk_payloads) + copied_slam_debug_artifacts
    return {
        "output_dir": str(root),
        "index_html": str(root / "index.html"),
        "data_json": str(root / "data.json"),
        "chunk_count": len(chunk_payloads),
        "slam_debug_artifact_count": copied_slam_debug_artifacts,
        "exported_files": exported_files,
        "viewer_mode": data["viewer_mode"],
        "display_points": data["display_points"],
    }
