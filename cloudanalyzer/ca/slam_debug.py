"""SLAM run diagnostics helpers."""

from __future__ import annotations

import csv
import math
import shlex
from pathlib import Path
from typing import Any


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
    rejection = _as_float(row, "scan_match_correspondence_rejection_rate") or 0.0
    prediction_delta = _as_float(row, "prediction_delta_m") or 0.0
    initial_delta = _as_float(row, "scan_match_vs_initial_pose_delta_m") or 0.0
    retries = _as_float(row, "registration_retry_count") or 0.0
    consecutive_failures = _as_float(row, "consecutive_scan_match_failures") or 0.0
    low_quality = 1.0 if _as_bool(row, "scan_quality_low") else 0.0

    if sort_by == "rmse":
        return rmse
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
        + rmse * 10.0
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


def analyze_slam_run(
    metrics_csv: str,
    scans_manifest_csv: str | None = None,
    trajectory_csv: str | None = None,
    map_path: str | None = None,
    top_k: int = 10,
    sort_by: str = "auto",
    artifact_dir: str | None = None,
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
    allowed = {"auto", "rmse", "rejection", "prediction-delta", "initial-delta", "failure"}
    if sort_by not in allowed:
        raise ValueError(f"sort_by must be one of: {', '.join(sorted(allowed))}")

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
            if initial_matrix is not None:
                parts.extend(["--initial-matrix", _matrix_csv(initial_matrix)])
            if frame_artifact_dir is not None:
                parts.extend(["--artifact-dir", frame_artifact_dir])
            command = " ".join(shlex.quote(part) for part in parts)
            scan_debug_commands.append(command)

        frames.append(
            {
                "rank": rank,
                "score": _score_row(row, sort_by),
                "scan_id": scan_id,
                "timestamp_sec": _as_float(row, "timestamp_sec"),
                "scan_path": scan_path,
                "reasons": _row_reasons(row),
                "scan_match_failed": _as_bool(row, "scan_match_failed"),
                "scan_match_rmse_m": _as_float(row, "scan_match_rmse_m"),
                "scan_match_rejection_rate": _as_float(
                    row, "scan_match_correspondence_rejection_rate"
                ),
                "prediction_delta_m": _as_float(row, "prediction_delta_m"),
                "initial_pose_translation_m": [initial_x, initial_y, initial_z]
                if initial_matrix is not None
                else None,
                "final_pose": trajectory.get(timestamp),
                "scan_match_debug_command": command,
            }
        )

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
    }

