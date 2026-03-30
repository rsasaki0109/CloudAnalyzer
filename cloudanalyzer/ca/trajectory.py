"""Trajectory evaluation utilities (ATE, translational RPE, drift)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


_TIMESTAMP_KEYS = ("timestamp", "time", "t")
SUPPORTED_TRAJECTORY_EXTENSIONS = {".csv", ".tum", ".txt"}


def _summary_stats(values: np.ndarray) -> dict:
    """Return summary stats for a 1D numeric array."""
    return {
        "count": int(values.size),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "std": float(np.std(values)),
    }


def _parse_csv_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load trajectory from CSV with either headers or raw columns."""
    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError("Trajectory file is empty")

    if any(char.isalpha() for char in lines[0]):
        reader = csv.DictReader(lines)
        if reader.fieldnames is None:
            raise ValueError("CSV trajectory header is missing")
        field_map = {name.strip().lower(): name for name in reader.fieldnames}
        timestamp_key = next((key for key in _TIMESTAMP_KEYS if key in field_map), None)
        if timestamp_key is None:
            raise ValueError("CSV trajectory must include a timestamp/time/t column")
        position_keys = ["x", "y", "z"]
        missing = [key for key in position_keys if key not in field_map]
        if missing:
            raise ValueError(f"CSV trajectory is missing required columns: {', '.join(missing)}")

        timestamps = []
        positions = []
        for row in reader:
            timestamps.append(float(row[field_map[timestamp_key]]))
            positions.append(
                [
                    float(row[field_map["x"]]),
                    float(row[field_map["y"]]),
                    float(row[field_map["z"]]),
                ]
            )
        return np.asarray(timestamps, dtype=float), np.asarray(positions, dtype=float)

    timestamps = []
    positions = []
    for csv_row in csv.reader(lines):
        if len(csv_row) < 4:
            raise ValueError("CSV trajectory rows must have at least 4 columns: timestamp,x,y,z")
        timestamps.append(float(csv_row[0]))
        positions.append([float(csv_row[1]), float(csv_row[2]), float(csv_row[3])])
    return np.asarray(timestamps, dtype=float), np.asarray(positions, dtype=float)


def _parse_tum_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load trajectory from whitespace-separated TUM-style format."""
    timestamps = []
    positions = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) not in {4, 8}:
            raise ValueError(
                "TUM trajectory rows must have 4 columns (timestamp x y z) "
                "or 8 columns (timestamp x y z qx qy qz qw)"
            )
        timestamps.append(float(parts[0]))
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not timestamps:
        raise ValueError("Trajectory file is empty")
    return np.asarray(timestamps, dtype=float), np.asarray(positions, dtype=float)


def load_trajectory(path: str) -> dict:
    """Load a trajectory from CSV or TUM-style text."""
    trajectory_path = Path(path)
    if not trajectory_path.exists():
        raise FileNotFoundError(path)

    suffix = trajectory_path.suffix.lower()
    if suffix == ".csv":
        timestamps, positions = _parse_csv_trajectory(trajectory_path)
        format_name = "csv"
    elif suffix in {".tum", ".txt"}:
        timestamps, positions = _parse_tum_trajectory(trajectory_path)
        format_name = "tum"
    else:
        raise ValueError("Unsupported trajectory format. Use .csv, .tum, or .txt")

    if timestamps.size < 2:
        raise ValueError("Trajectory must contain at least 2 poses")
    if positions.shape != (timestamps.size, 3):
        raise ValueError("Trajectory positions are malformed")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Trajectory timestamps must be strictly increasing")

    return {
        "path": path,
        "format": format_name,
        "timestamps": timestamps,
        "positions": positions,
        "num_poses": int(timestamps.size),
    }


def _interpolate_matches(
    estimated_times: np.ndarray,
    estimated_positions: np.ndarray,
    reference_times: np.ndarray,
    reference_positions: np.ndarray,
    max_time_delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match reference timestamps to estimated poses with linear interpolation."""
    matched_times: list[float] = []
    matched_estimated_positions: list[np.ndarray] = []
    matched_reference_positions: list[np.ndarray] = []
    matched_time_deltas: list[float] = []

    for reference_time, reference_position in zip(reference_times, reference_positions):
        insert_index = int(np.searchsorted(estimated_times, reference_time))

        if insert_index < estimated_times.size and abs(estimated_times[insert_index] - reference_time) <= 1e-9:
            matched_times.append(float(reference_time))
            matched_estimated_positions.append(estimated_positions[insert_index])
            matched_reference_positions.append(reference_position)
            matched_time_deltas.append(0.0)
            continue

        if 0 < insert_index < estimated_times.size:
            left_time = estimated_times[insert_index - 1]
            right_time = estimated_times[insert_index]
            left_delta = reference_time - left_time
            right_delta = right_time - reference_time
            if left_delta <= max_time_delta and right_delta <= max_time_delta:
                alpha = left_delta / (right_time - left_time)
                estimated_position = (
                    (1.0 - alpha) * estimated_positions[insert_index - 1]
                    + alpha * estimated_positions[insert_index]
                )
                matched_times.append(float(reference_time))
                matched_estimated_positions.append(estimated_position)
                matched_reference_positions.append(reference_position)
                matched_time_deltas.append(float(min(left_delta, right_delta)))
                continue

        if insert_index == 0 and abs(estimated_times[0] - reference_time) <= max_time_delta:
            matched_times.append(float(reference_time))
            matched_estimated_positions.append(estimated_positions[0])
            matched_reference_positions.append(reference_position)
            matched_time_deltas.append(float(abs(estimated_times[0] - reference_time)))
            continue

        if (
            insert_index == estimated_times.size
            and abs(reference_time - estimated_times[-1]) <= max_time_delta
        ):
            matched_times.append(float(reference_time))
            matched_estimated_positions.append(estimated_positions[-1])
            matched_reference_positions.append(reference_position)
            matched_time_deltas.append(float(abs(reference_time - estimated_times[-1])))

    return (
        np.asarray(matched_times, dtype=float),
        np.asarray(matched_estimated_positions, dtype=float),
        np.asarray(matched_reference_positions, dtype=float),
        np.asarray(matched_time_deltas, dtype=float),
    )


def _quality_gate(
    ate_rmse: float,
    rpe_rmse: float,
    endpoint_drift: float,
    coverage_ratio: float,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> dict | None:
    """Build optional trajectory quality gate metadata."""
    if max_ate is None and max_rpe is None and max_drift is None and min_coverage is None:
        return None

    reasons = []
    if max_ate is not None and ate_rmse > max_ate:
        reasons.append(f"ATE RMSE {ate_rmse:.4f} > max_ate {max_ate:.4f}")
    if max_rpe is not None and rpe_rmse > max_rpe:
        reasons.append(f"RPE RMSE {rpe_rmse:.4f} > max_rpe {max_rpe:.4f}")
    if max_drift is not None and endpoint_drift > max_drift:
        reasons.append(f"Endpoint Drift {endpoint_drift:.4f} > max_drift {max_drift:.4f}")
    if min_coverage is not None and coverage_ratio < min_coverage:
        reasons.append(
            f"Coverage {coverage_ratio:.1%} < min_coverage {min_coverage:.1%}"
        )
    return {
        "passed": not reasons,
        "max_ate": max_ate,
        "max_rpe": max_rpe,
        "max_drift": max_drift,
        "min_coverage": min_coverage,
        "reasons": reasons,
    }


def _apply_origin_alignment(
    estimated_positions: np.ndarray,
    reference_positions: np.ndarray,
    align_origin: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optionally align estimated trajectory by the initial translation offset."""
    if not align_origin:
        return estimated_positions, np.zeros(3, dtype=float), np.eye(3, dtype=float)

    translation = reference_positions[0] - estimated_positions[0]
    return estimated_positions + translation, translation, np.eye(3, dtype=float)


def _apply_rigid_alignment(
    estimated_positions: np.ndarray,
    reference_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align estimated trajectory to reference with a rigid transform."""
    estimated_centroid = np.mean(estimated_positions, axis=0)
    reference_centroid = np.mean(reference_positions, axis=0)
    estimated_centered = estimated_positions - estimated_centroid
    reference_centered = reference_positions - reference_centroid

    covariance = estimated_centered.T @ reference_centered
    u_matrix, _, v_transpose = np.linalg.svd(covariance)
    rotation = v_transpose.T @ u_matrix.T
    if np.linalg.det(rotation) < 0:
        v_transpose[-1, :] *= -1
        rotation = v_transpose.T @ u_matrix.T

    translation = reference_centroid - rotation @ estimated_centroid
    aligned = estimated_positions @ rotation.T + translation
    return aligned, translation, rotation


def _apply_alignment(
    estimated_positions: np.ndarray,
    reference_positions: np.ndarray,
    align_origin: bool,
    align_rigid: bool,
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray]:
    """Apply the requested alignment mode to the estimated trajectory."""
    if align_origin and align_rigid:
        raise ValueError("--align-origin and --align-rigid are mutually exclusive")
    if align_rigid:
        aligned, translation, rotation = _apply_rigid_alignment(
            estimated_positions,
            reference_positions,
        )
        return aligned, "rigid", translation, rotation
    if align_origin:
        aligned, translation, rotation = _apply_origin_alignment(
            estimated_positions,
            reference_positions,
            align_origin=True,
        )
        return aligned, "origin", translation, rotation
    return estimated_positions, "none", np.zeros(3, dtype=float), np.eye(3, dtype=float)


def evaluate_trajectory(
    estimated_path: str,
    reference_path: str,
    max_time_delta: float = 0.05,
    align_origin: bool = False,
    align_rigid: bool = False,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> dict:
    """Evaluate a trajectory against a reference trajectory."""
    if max_time_delta <= 0:
        raise ValueError("max_time_delta must be > 0")
    if min_coverage is not None and not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be between 0 and 1")

    estimated = load_trajectory(estimated_path)
    reference = load_trajectory(reference_path)

    matched_times, matched_estimated_positions, matched_reference_positions, matched_time_deltas = _interpolate_matches(
        estimated["timestamps"],
        estimated["positions"],
        reference["timestamps"],
        reference["positions"],
        max_time_delta=max_time_delta,
    )

    if matched_times.size < 2:
        raise ValueError("Need at least 2 matched poses within max_time_delta")

    aligned_estimated_positions, alignment_mode, alignment_translation, alignment_rotation = _apply_alignment(
        matched_estimated_positions,
        matched_reference_positions,
        align_origin=align_origin,
        align_rigid=align_rigid,
    )

    ate_errors = np.linalg.norm(aligned_estimated_positions - matched_reference_positions, axis=1)
    estimated_steps = np.diff(aligned_estimated_positions, axis=0)
    reference_steps = np.diff(matched_reference_positions, axis=0)
    rpe_errors = np.linalg.norm(estimated_steps - reference_steps, axis=1)

    reference_path_length = float(np.sum(np.linalg.norm(reference_steps, axis=1)))
    estimated_path_length = float(np.sum(np.linalg.norm(estimated_steps, axis=1)))
    endpoint_drift = float(
        np.linalg.norm(
            (aligned_estimated_positions[-1] - aligned_estimated_positions[0])
            - (matched_reference_positions[-1] - matched_reference_positions[0])
        )
    )
    drift_ratio = (
        float(endpoint_drift / reference_path_length)
        if reference_path_length > 0
        else None
    )

    matching = {
        "estimated_poses": estimated["num_poses"],
        "reference_poses": reference["num_poses"],
        "matched_poses": int(matched_times.size),
        "coverage_ratio": float(matched_times.size / reference["num_poses"]),
        "max_time_delta": float(max_time_delta),
        "mean_abs_time_delta": float(np.mean(matched_time_deltas)),
        "max_abs_time_delta": float(np.max(matched_time_deltas)),
        "reference_duration": float(reference["timestamps"][-1] - reference["timestamps"][0]),
        "matched_duration": float(matched_times[-1] - matched_times[0]),
    }
    matching["duration_coverage_ratio"] = (
        float(matching["matched_duration"] / matching["reference_duration"])
        if matching["reference_duration"] > 0
        else 1.0
    )

    worst_ate_indices = np.argsort(-ate_errors)[: min(5, ate_errors.size)]
    worst_rpe_indices = np.argsort(-rpe_errors)[: min(5, rpe_errors.size)]

    ate_stats = _summary_stats(ate_errors)
    rpe_stats = _summary_stats(rpe_errors)

    return {
        "estimated_path": estimated_path,
        "reference_path": reference_path,
        "alignment": {
            "mode": alignment_mode,
            "translation": alignment_translation.tolist(),
            "rotation_matrix": alignment_rotation.tolist(),
        },
        "matching": matching,
        "ate": ate_stats,
        "rpe_translation": rpe_stats,
        "drift": {
            "endpoint": endpoint_drift,
            "ratio_to_reference_path_length": drift_ratio,
            "reference_path_length": reference_path_length,
            "estimated_path_length": estimated_path_length,
            "path_length_ratio": (
                float(estimated_path_length / reference_path_length)
                if reference_path_length > 0
                else None
            ),
        },
        "worst_ate_samples": [
            {
                "timestamp": float(matched_times[index]),
                "position_error": float(ate_errors[index]),
                "time_delta": float(matched_time_deltas[index]),
            }
            for index in worst_ate_indices
        ],
        "worst_rpe_segments": [
            {
                "start_timestamp": float(matched_times[index]),
                "end_timestamp": float(matched_times[index + 1]),
                "translation_error": float(rpe_errors[index]),
            }
            for index in worst_rpe_indices
        ],
        "matched_trajectory": {
            "timestamps": matched_times.tolist(),
            "estimated_positions": aligned_estimated_positions.tolist(),
            "reference_positions": matched_reference_positions.tolist(),
            "ate_errors": ate_errors.tolist(),
        },
        "error_series": {
            "rpe_timestamps": (
                ((matched_times[:-1] + matched_times[1:]) / 2.0).tolist()
                if matched_times.size >= 2
                else []
            ),
            "rpe_translation": rpe_errors.tolist(),
        },
        "quality_gate": _quality_gate(
            ate_stats["rmse"],
            rpe_stats["rmse"],
            endpoint_drift,
            matching["coverage_ratio"],
            max_ate=max_ate,
            max_rpe=max_rpe,
            max_drift=max_drift,
            min_coverage=min_coverage,
        ),
    }


def plot_trajectory_overlay(result: dict, output_path: str) -> None:
    """Plot matched estimated/reference trajectories in the XY plane."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matched = result["matched_trajectory"]
    estimated_positions = np.asarray(matched["estimated_positions"], dtype=float)
    reference_positions = np.asarray(matched["reference_positions"], dtype=float)
    ate_errors = np.asarray(matched["ate_errors"], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(
        reference_positions[:, 0],
        reference_positions[:, 1],
        label="Reference",
        linewidth=2.2,
        color="#0f766e",
    )
    ax.plot(
        estimated_positions[:, 0],
        estimated_positions[:, 1],
        label="Estimated",
        linewidth=2.0,
        linestyle="--",
        color="#b45309",
    )
    ax.scatter(reference_positions[0, 0], reference_positions[0, 1], color="#0f766e", marker="o", s=60)
    ax.scatter(estimated_positions[0, 0], estimated_positions[0, 1], color="#b45309", marker="o", s=60)
    ax.scatter(reference_positions[-1, 0], reference_positions[-1, 1], color="#0f766e", marker="s", s=70)
    ax.scatter(estimated_positions[-1, 0], estimated_positions[-1, 1], color="#b45309", marker="s", s=70)

    worst_index = int(np.argmax(ate_errors))
    ax.scatter(
        reference_positions[worst_index, 0],
        reference_positions[worst_index, 1],
        color="#dc2626",
        marker="x",
        s=80,
        label="Worst ATE",
    )

    ax.set_title("Trajectory Overlay (XY)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def plot_trajectory_error_timeline(result: dict, output_path: str) -> None:
    """Plot ATE and translational RPE over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matched = result["matched_trajectory"]
    error_series = result["error_series"]
    timestamps = np.asarray(matched["timestamps"], dtype=float)
    ate_errors = np.asarray(matched["ate_errors"], dtype=float)
    rpe_timestamps = np.asarray(error_series["rpe_timestamps"], dtype=float)
    rpe_errors = np.asarray(error_series["rpe_translation"], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        timestamps,
        ate_errors,
        label=f"ATE (RMSE={result['ate']['rmse']:.4f})",
        linewidth=2.0,
        color="#2563eb",
    )
    if rpe_timestamps.size > 0:
        ax.plot(
            rpe_timestamps,
            rpe_errors,
            label=f"RPE (RMSE={result['rpe_translation']['rmse']:.4f})",
            linewidth=1.8,
            color="#dc2626",
        )

    ax.set_title("Trajectory Error Timeline")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=110)
    plt.close(fig)
