"""SLAM run diagnostics helpers."""

from __future__ import annotations

import csv
import math
import shlex
from pathlib import Path
from typing import Any

from ca.scan_match_debug import run_scan_match_debug


def _as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _as_bool(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() in {"1", "true", "yes", "y"}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_scan_manifest(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = _read_csv_rows(path)
    base = path.parent
    manifest: dict[str, str] = {}
    for row in rows:
        scan_id = row.get("scan_id")
        points = row.get("points_csv")
        if not scan_id or not points:
            continue
        points_path = Path(points)
        if not points_path.is_absolute():
            points_path = base / points_path
        manifest[scan_id] = str(points_path)
    return manifest


def _read_trajectory_index(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    index: dict[str, dict[str, float]] = {}
    for row in _read_csv_rows(path):
        timestamp = row.get("timestamp_sec")
        if timestamp is None:
            continue
        parsed: dict[str, float] = {}
        for key in ("x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad"):
            value = _as_float(row, key)
            if value is not None:
                parsed[key] = value
        index[timestamp] = parsed
    return index


def _translation_matrix(x: float, y: float, z: float) -> list[float]:
    return [
        1.0,
        0.0,
        0.0,
        x,
        0.0,
        1.0,
        0.0,
        y,
        0.0,
        0.0,
        1.0,
        z,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def _matrix_csv(matrix: list[float]) -> str:
    return ",".join(f"{value:.12g}" for value in matrix)


def _score_row(row: dict[str, str], sort_by: str) -> float:
    failed = 1.0 if _as_bool(row, "scan_match_failed") else 0.0
    rmse = _as_float(row, "scan_match_rmse_m") or 0.0
    weighted_rmse = _as_float(row, "scan_match_weighted_rmse")
    cost_rmse = weighted_rmse if weighted_rmse is not None else rmse
    rejection = _as_float(row, "scan_match_correspondence_rejection_rate") or 0.0
    prediction_delta = _as_float(row, "prediction_delta_m") or 0.0
    initial_delta = _as_float(row, "scan_match_vs_initial_pose_delta_m") or 0.0
    retries = _as_float(row, "registration_retry_count") or 0.0
    consecutive_failures = _as_float(row, "consecutive_scan_match_failures") or 0.0
    low_quality = 1.0 if _as_bool(row, "scan_quality_low") else 0.0

    if sort_by == "rmse":
        return rmse
    if sort_by == "cost":
        return cost_rmse
    if sort_by == "rejection":
        return rejection
    if sort_by == "prediction-delta":
        return prediction_delta
    if sort_by == "initial-delta":
        return initial_delta
    if sort_by == "failure":
        return failed

    return (
        failed * 1_000_000.0
        + consecutive_failures * 10_000.0
        + low_quality * 1_000.0
        + retries * 100.0
        + cost_rmse * 10.0
        + rejection * 5.0
        + prediction_delta
        + initial_delta
    )


def _row_reasons(row: dict[str, str]) -> list[str]:
    reasons: list[str] = []
    if _as_bool(row, "scan_match_failed"):
        error = row.get("scan_match_error") or "unknown"
        reasons.append(f"scan_match_failed:{error}")
    if _as_bool(row, "scan_quality_low"):
        reason = row.get("scan_quality_reason") or "unknown"
        reasons.append(f"low_quality:{reason}")
    rmse = _as_float(row, "scan_match_rmse_m")
    if rmse is not None:
        reasons.append(f"rmse={rmse:.4g}")
    weighted_rmse = _as_float(row, "scan_match_weighted_rmse")
    if weighted_rmse is not None:
        reasons.append(f"weighted_rmse={weighted_rmse:.4g}")
    rejection = _as_float(row, "scan_match_correspondence_rejection_rate")
    if rejection is not None:
        reasons.append(f"rejection={rejection:.3g}")
    prediction_delta = _as_float(row, "prediction_delta_m")
    if prediction_delta is not None:
        reasons.append(f"prediction_delta={prediction_delta:.4g}m")
    initial_delta = _as_float(row, "scan_match_vs_initial_pose_delta_m")
    if initial_delta is not None:
        reasons.append(f"initial_delta={initial_delta:.4g}m")
    return reasons


def _frame_metrics(row: dict[str, str]) -> dict[str, Any]:
    return {
        "scan_match_failed": _as_bool(row, "scan_match_failed"),
        "scan_match_error": row.get("scan_match_error") or None,
        "scan_match_rmse_m": _as_float(row, "scan_match_rmse_m"),
        "scan_match_weighted_rmse": _as_float(row, "scan_match_weighted_rmse"),
        "scan_match_rejection_rate": _as_float(
            row, "scan_match_correspondence_rejection_rate"
        ),
        "prediction_delta_m": _as_float(row, "prediction_delta_m"),
        "initial_delta_m": _as_float(row, "scan_match_vs_initial_pose_delta_m"),
        "registration_retry_count": _as_float(row, "registration_retry_count"),
        "consecutive_scan_match_failures": _as_float(
            row, "consecutive_scan_match_failures"
        ),
        "scan_quality_low": _as_bool(row, "scan_quality_low"),
        "scan_quality_reason": row.get("scan_quality_reason") or None,
        "raw_points": _as_float(row, "raw_points"),
        "downsampled_points": _as_float(row, "downsampled_points"),
        "filtered_points": _as_float(row, "filtered_points"),
        "raw_range_min_m": _as_float(row, "raw_range_min_m"),
        "raw_range_max_m": _as_float(row, "raw_range_max_m"),
        "raw_range_mean_m": _as_float(row, "raw_range_mean_m"),
        "filtered_range_min_m": _as_float(row, "filtered_range_min_m"),
        "filtered_range_max_m": _as_float(row, "filtered_range_max_m"),
        "filtered_range_mean_m": _as_float(row, "filtered_range_mean_m"),
        "map_points": _as_float(row, "map_points"),
        "registration_map_points": _as_float(row, "registration_map_points"),
    }


def _nested_float(data: dict[str, Any] | None, *keys: str) -> float | None:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _diagnosis(
    label: str,
    confidence: str,
    suggested_action: str,
    signals: dict[str, Any],
    secondary_labels: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "label": label,
        "confidence": confidence,
        "suggested_action": suggested_action,
        "secondary_labels": secondary_labels or [],
        "signals": signals,
    }


def _diagnosis_candidate(
    label: str,
    confidence: str,
    suggested_action: str,
) -> dict[str, str]:
    return {
        "label": label,
        "confidence": confidence,
        "suggested_action": suggested_action,
    }


def _select_primary_diagnosis(
    candidates: list[dict[str, str]],
    signals: dict[str, Any],
) -> dict[str, Any]:
    if not candidates:
        return _diagnosis(
            "needs_review",
            "low",
            "Review GLIM metrics and colored artifacts manually.",
            signals,
        )
    primary = candidates[0]
    secondary_labels: list[str] = []
    for candidate in candidates[1:]:
        label = candidate["label"]
        if label not in secondary_labels:
            secondary_labels.append(label)
    return _diagnosis(
        primary["label"],
        primary["confidence"],
        primary["suggested_action"],
        signals,
        secondary_labels=secondary_labels,
    )


def diagnose_slam_frame(frame: dict[str, Any]) -> dict[str, Any]:
    """Classify a suspicious SLAM frame into an actionable debug bucket."""

    metrics = frame.get("glim_metrics") or {}
    debug = frame.get("scan_match_debug_result")

    rmse = metrics.get("scan_match_rmse_m")
    weighted_rmse = metrics.get("scan_match_weighted_rmse")
    cost_rmse = weighted_rmse if weighted_rmse is not None else rmse
    prediction_delta = metrics.get("prediction_delta_m")
    initial_delta = metrics.get("initial_delta_m")
    raw_points = metrics.get("raw_points")
    filtered_points = metrics.get("downsampled_points") or metrics.get("filtered_points")
    scan_quality_low = bool(metrics.get("scan_quality_low"))
    raw_range_mean = metrics.get("raw_range_mean_m")
    filtered_range_mean = metrics.get("filtered_range_mean_m")

    before_mean = _nested_float(debug, "distance_before", "stats", "mean")
    after_mean = _nested_float(debug, "distance_after", "stats", "mean")
    improvement = _nested_float(debug, "improvement", "mean")
    fitness = _nested_float(debug, "registration", "fitness")
    inlier_rmse = _nested_float(debug, "registration", "inlier_rmse")
    map_points_used = _nested_float(debug, "preprocess", "map_points_used")
    if map_points_used is None:
        map_points_used = metrics.get("registration_map_points") or metrics.get(
            "map_points"
        )
    scan_points_used = _nested_float(debug, "preprocess", "scan_points_used")

    filtered_ratio = None
    if raw_points not in (None, 0) and filtered_points is not None:
        filtered_ratio = float(filtered_points) / float(raw_points)

    signals = {
        "scan_match_rmse_m": rmse,
        "scan_match_weighted_rmse": weighted_rmse,
        "scan_match_cost_rmse": cost_rmse,
        "prediction_delta_m": prediction_delta,
        "initial_delta_m": initial_delta,
        "raw_points": raw_points,
        "filtered_points": filtered_points,
        "filtered_ratio": filtered_ratio,
        "raw_range_mean_m": raw_range_mean,
        "filtered_range_mean_m": filtered_range_mean,
        "raw_range_min_m": metrics.get("raw_range_min_m"),
        "raw_range_max_m": metrics.get("raw_range_max_m"),
        "filtered_range_min_m": metrics.get("filtered_range_min_m"),
        "filtered_range_max_m": metrics.get("filtered_range_max_m"),
        "before_mean": before_mean,
        "after_mean": after_mean,
        "improvement_mean": improvement,
        "fitness": fitness,
        "inlier_rmse": inlier_rmse,
        "map_points_used": map_points_used,
        "scan_points_used": scan_points_used,
    }

    candidates: list[dict[str, str]] = []

    if raw_points is not None and raw_points < 50:
        candidates.append(
            _diagnosis_candidate(
                "sparse_raw_scan",
                "high",
                "Inspect the source scan around this frame; too few raw points reached SLAM.",
            )
        )
    if (
        raw_points is not None
        and raw_points >= 1000
        and filtered_points is not None
        and filtered_points < 50
        and filtered_ratio is not None
        and filtered_ratio < 0.02
    ):
        candidates.append(
            _diagnosis_candidate(
                "filtering_too_aggressive",
                "high",
                "Inspect voxel/filter settings; the raw scan has points but too few survive downsampling.",
            )
        )
    if scan_quality_low or (filtered_points is not None and filtered_points < 50):
        candidates.append(
            _diagnosis_candidate(
                "scan_quality_issue",
                "high",
                "Inspect raw scan filtering and sensor data around this frame.",
            )
        )
    if map_points_used is not None and map_points_used < 100:
        candidates.append(
            _diagnosis_candidate(
                "map_too_sparse",
                "high",
                "Inspect keyframe insertion/local map selection and rerun with a wider local map.",
            )
        )
    if (
        improvement is not None
        and after_mean is not None
        and initial_delta is not None
        and improvement >= 0.5
        and after_mean <= 0.75
        and initial_delta >= 0.5
    ):
        candidates.append(
            _diagnosis_candidate(
                "bad_initial_guess",
                "high",
                "Inspect IMU/prediction seed, initial pose, and motion model around this timestamp.",
            )
        )
    if (
        debug
        and cost_rmse is not None
        and fitness is not None
        and inlier_rmse is not None
        and cost_rmse >= 1.5
        and fitness >= 0.95
        and inlier_rmse <= 0.25
    ):
        candidates.append(
            _diagnosis_candidate(
                "scan_match_cost_hotspot",
                "medium",
                (
                    "GLIM reports a high weighted scan-match cost while "
                    "CloudAnalyzer re-registration is stable; compare cost "
                    "scaling, residual definitions, and planar correspondence geometry."
                ),
            )
        )
    if (
        debug
        and cost_rmse is not None
        and improvement is not None
        and inlier_rmse is not None
        and cost_rmse >= 2.0
        and improvement < 0.1
        and inlier_rmse >= 0.5
    ):
        candidates.append(
            _diagnosis_candidate(
                "weak_geometry",
                "medium",
                "Inspect geometry degeneracy, correspondence radius, and voxel/normal constraints.",
            )
        )
    if (
        debug
        and rmse is not None
        and improvement is not None
        and rmse >= 2.0
        and improvement >= 0.25
        and (
            (prediction_delta is not None and prediction_delta >= 0.75)
            or (initial_delta is not None and initial_delta >= 0.75)
        )
    ):
        candidates.append(
            _diagnosis_candidate(
                "registration_local_minimum",
                "medium",
                (
                    "Compare GLIM scan-match result with CloudAnalyzer aligned artifacts; "
                    "try a wider initial search or alternate registration seed."
                ),
            )
        )
    if frame.get("scan_match_debug_error"):
        candidates.append(
            _diagnosis_candidate(
                "scan_match_debug_error",
                "medium",
                "Open the scan-match error and verify scan/map paths and formats.",
            )
        )
    return _select_primary_diagnosis(candidates, signals)


def render_slam_debug_markdown(result: dict[str, Any]) -> str:
    """Render a human-readable SLAM debug report."""

    lines = [
        "# SLAM Debug Report",
        "",
        f"- Metrics: `{result['metrics_csv']}`",
        f"- Frames: {result['total_frames']}",
        f"- Selected: {len(result['selected_frames'])}",
        f"- Sort: `{result['sort_by']}`",
    ]
    if result.get("map_path"):
        lines.append(f"- Map: `{result['map_path']}`")
    if result.get("trajectory_csv"):
        lines.append(f"- Trajectory: `{result['trajectory_csv']}`")
    lines.extend(["", "## Suspicious Frames", ""])

    for frame in result["selected_frames"]:
        lines.extend(
            [
                f"### #{frame['rank']:02d} `{frame['scan_id']}`",
                "",
                f"- Score: {frame['score']:.3f}",
                f"- Timestamp: {frame['timestamp_sec']}",
                f"- Scan: `{frame['scan_path']}`" if frame.get("scan_path") else "- Scan: n/a",
            ]
        )
        if frame.get("reasons"):
            lines.append(f"- Reasons: {', '.join(frame['reasons'])}")

        diagnosis = frame.get("diagnosis")
        if diagnosis:
            lines.extend(
                [
                    "- Diagnosis: "
                    f"`{diagnosis['label']}` ({diagnosis['confidence']})",
                    f"- Suggested action: {diagnosis['suggested_action']}",
                ]
            )
            secondary_labels = diagnosis.get("secondary_labels") or []
            if secondary_labels:
                lines.append(
                    "- Secondary labels: "
                    + ", ".join(f"`{label}`" for label in secondary_labels)
                )

        metrics = frame.get("glim_metrics", {})
        lines.extend(
            [
                "- GLIM metrics: "
                f"rmse={metrics.get('scan_match_rmse_m')}, "
                f"weighted_rmse={metrics.get('scan_match_weighted_rmse')}, "
                f"prediction_delta={metrics.get('prediction_delta_m')}, "
                f"initial_delta={metrics.get('initial_delta_m')}, "
                f"failed={metrics.get('scan_match_failed')}, "
                f"raw_points={metrics.get('raw_points')}, "
                f"downsampled_points={metrics.get('downsampled_points') or metrics.get('filtered_points')}, "
                f"raw_range_mean={metrics.get('raw_range_mean_m')}, "
                f"filtered_range_mean={metrics.get('filtered_range_mean_m')}",
            ]
        )

        debug = frame.get("scan_match_debug_result")
        if debug:
            registration = debug["registration"]
            before = debug["distance_before"]["stats"]
            after = debug["distance_after"]["stats"]
            improvement = debug["improvement"]
            lines.extend(
                [
                    "- CloudAnalyzer scan-match: "
                    f"fitness={registration['fitness']:.4f}, "
                    f"inlier_rmse={registration['inlier_rmse']:.4f}",
                    "- NN distance: "
                    f"before_mean={before['mean']:.4f}, "
                    f"after_mean={after['mean']:.4f}, "
                    f"improvement_mean={improvement['mean']:.4f}",
                ]
            )
            artifacts = debug.get("artifacts", {})
            if artifacts:
                lines.append("- Artifacts:")
                for name, path in artifacts.items():
                    lines.append(f"  - `{name}`: `{path}`")
        elif frame.get("scan_match_debug_error"):
            lines.append(f"- CloudAnalyzer scan-match error: `{frame['scan_match_debug_error']}`")

        if frame.get("scan_match_debug_command"):
            lines.extend(["", "```bash", frame["scan_match_debug_command"], "```"])
        lines.append("")

    commands = result.get("commands", {})
    if commands:
        lines.extend(["## Commands", ""])
        for key, command in commands.items():
            if isinstance(command, str):
                lines.extend([f"### {key}", "", "```bash", command, "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def write_slam_debug_markdown(result: dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_slam_debug_markdown(result), encoding="utf-8")


def analyze_slam_run(
    metrics_csv: str,
    scans_manifest_csv: str | None = None,
    trajectory_csv: str | None = None,
    map_path: str | None = None,
    top_k: int = 10,
    sort_by: str = "auto",
    artifact_dir: str | None = None,
    run_scan_match_debug_frames: bool = False,
    scan_match_method: str = "gicp",
    scan_match_max_correspondence_distance: float = 1.0,
    scan_match_scan_voxel_size: float | None = None,
    scan_match_map_voxel_size: float | None = None,
    scan_match_crop_margin: float | None = None,
    scan_match_threshold: float | None = None,
) -> dict[str, Any]:
    """Rank suspicious SLAM frames and emit CloudAnalyzer drill-down commands.

    The function is intentionally format-light: it consumes the `metrics.csv`
    emitted by `glim_mapping`, optionally joins scan paths from
    `scans_manifest.csv`, and produces commands for existing CloudAnalyzer
    viewers/debuggers.
    """

    metrics_path = Path(metrics_csv)
    manifest_path = Path(scans_manifest_csv) if scans_manifest_csv else None
    trajectory_path = Path(trajectory_csv) if trajectory_csv else None
    rows = _read_csv_rows(metrics_path)
    scans = _read_scan_manifest(manifest_path)
    trajectory = _read_trajectory_index(trajectory_path)
    top_k = max(1, int(top_k))
    sort_by = sort_by.lower()
    allowed = {
        "auto",
        "rmse",
        "cost",
        "rejection",
        "prediction-delta",
        "initial-delta",
        "failure",
    }
    if sort_by not in allowed:
        raise ValueError(f"sort_by must be one of: {', '.join(sorted(allowed))}")
    if run_scan_match_debug_frames and (manifest_path is None or map_path is None):
        raise ValueError(
            "run_scan_match_debug_frames requires scans_manifest_csv and map_path"
        )

    ranked = sorted(rows, key=lambda row: _score_row(row, sort_by), reverse=True)[:top_k]
    frames: list[dict[str, Any]] = []
    scan_debug_commands: list[str] = []
    out_base = Path(artifact_dir) if artifact_dir else None
    map_arg = str(Path(map_path)) if map_path else None

    for rank, row in enumerate(ranked, start=1):
        scan_id = row.get("scan_id", "")
        timestamp = row.get("timestamp_sec", "")
        scan_path = scans.get(scan_id)
        initial_x = _as_float(row, "initial_x_m")
        initial_y = _as_float(row, "initial_y_m")
        initial_z = _as_float(row, "initial_z_m")
        initial_matrix = (
            _translation_matrix(initial_x, initial_y, initial_z)
            if initial_x is not None and initial_y is not None and initial_z is not None
            else None
        )
        frame_artifact_dir = str(out_base / f"{rank:02d}_{scan_id}") if out_base else None

        command = None
        if scan_path is not None and map_arg is not None:
            parts = ["ca", "scan-match-debug", scan_path, map_arg]
            parts.extend(["--method", scan_match_method])
            parts.extend(
                [
                    "--max-correspondence-distance",
                    f"{scan_match_max_correspondence_distance:.12g}",
                ]
            )
            if initial_matrix is not None:
                parts.extend(["--initial-matrix", _matrix_csv(initial_matrix)])
            if scan_match_scan_voxel_size is not None:
                parts.extend(["--scan-voxel-size", f"{scan_match_scan_voxel_size:.12g}"])
            if scan_match_map_voxel_size is not None:
                parts.extend(["--map-voxel-size", f"{scan_match_map_voxel_size:.12g}"])
            if scan_match_crop_margin is not None:
                parts.extend(["--crop-margin", f"{scan_match_crop_margin:.12g}"])
            if scan_match_threshold is not None:
                parts.extend(["--threshold", f"{scan_match_threshold:.12g}"])
            if frame_artifact_dir is not None:
                parts.extend(["--artifact-dir", frame_artifact_dir])
            command = " ".join(shlex.quote(part) for part in parts)
            scan_debug_commands.append(command)

        scan_match_debug_result = None
        scan_match_debug_error = None
        if run_scan_match_debug_frames and scan_path is not None and map_arg is not None:
            try:
                scan_match_debug_result = run_scan_match_debug(
                    scan_path=scan_path,
                    map_path=map_arg,
                    method=scan_match_method,
                    max_correspondence_distance=scan_match_max_correspondence_distance,
                    initial_transform=initial_matrix,
                    scan_voxel_size=scan_match_scan_voxel_size,
                    map_voxel_size=scan_match_map_voxel_size,
                    crop_margin=scan_match_crop_margin,
                    threshold=scan_match_threshold,
                    artifact_dir=frame_artifact_dir,
                )
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                scan_match_debug_error = str(exc)

        frame = {
            "rank": rank,
            "score": _score_row(row, sort_by),
            "scan_id": scan_id,
            "timestamp_sec": _as_float(row, "timestamp_sec"),
            "scan_path": scan_path,
            "reasons": _row_reasons(row),
            "scan_match_failed": _as_bool(row, "scan_match_failed"),
            "scan_match_rmse_m": _as_float(row, "scan_match_rmse_m"),
            "scan_match_weighted_rmse": _as_float(row, "scan_match_weighted_rmse"),
            "scan_match_rejection_rate": _as_float(
                row, "scan_match_correspondence_rejection_rate"
            ),
            "prediction_delta_m": _as_float(row, "prediction_delta_m"),
            "initial_delta_m": _as_float(row, "scan_match_vs_initial_pose_delta_m"),
            "glim_metrics": _frame_metrics(row),
            "initial_pose_translation_m": [initial_x, initial_y, initial_z]
            if initial_matrix is not None
            else None,
            "final_pose": trajectory.get(timestamp),
            "scan_match_debug_command": command,
            "scan_match_debug_result": scan_match_debug_result,
            "scan_match_debug_error": scan_match_debug_error,
        }
        frame["diagnosis"] = diagnose_slam_frame(frame)
        frames.append(frame)

    commands: dict[str, Any] = {"scan_match_debug": scan_debug_commands}
    if map_arg is not None and trajectory_csv is not None:
        commands["web"] = " ".join(
            shlex.quote(part) for part in ["ca", "web", map_arg, "--trajectory", trajectory_csv]
        )
    if map_arg is not None and trajectory_csv is not None and artifact_dir is not None:
        commands["web_export"] = " ".join(
            shlex.quote(part)
            for part in [
                "ca",
                "web-export",
                map_arg,
                "--trajectory",
                trajectory_csv,
                "-o",
                str(Path(artifact_dir) / "web"),
            ]
        )

    return {
        "metrics_csv": str(metrics_path),
        "scans_manifest_csv": str(manifest_path) if manifest_path else None,
        "trajectory_csv": str(trajectory_path) if trajectory_path else None,
        "map_path": map_arg,
        "sort_by": sort_by,
        "total_frames": len(rows),
        "selected_frames": frames,
        "commands": commands,
        "scan_match_debug_ran": bool(run_scan_match_debug_frames),
    }
