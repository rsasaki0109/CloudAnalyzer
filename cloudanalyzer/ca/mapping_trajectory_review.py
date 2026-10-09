"""Bounded, read-only localization of a saved mapping trajectory comparison."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ca import mapping_job as jobs
from ca.mapping_trajectory import MAX_BYTES, MAX_POSES, SCHEMA
from ca.posegraph_fix import read_trajectory
from ca.trajectory import load_trajectory

REVIEW_SCHEMA = "cloudanalyzer.mapping_trajectory_regions.v1"
PAGE_SIZE = 8


def _artifact(value: Any, limit: int | None = None) -> dict[str, Any]:
    if (not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}
            or not isinstance(value["path"], str) or not value["path"]
            or not isinstance(value["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
            or type(value["bytes"]) is not int or value["bytes"] < 0):
        raise ValueError("expected a file artifact with path, sha256 and bytes")
    if limit is not None and value["bytes"] > limit:
        raise ValueError("trajectory comparison input exceeds the 16 MiB limit")
    jobs._verify(value)
    return value


def _array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("comparison needs finite, consistently sized pose samples")
    return result


def _ids(value: Any) -> list[int]:
    if (not isinstance(value, list) or not 1 <= len(value) <= MAX_POSES
            or any(type(i) is not int or i < 0 for i in value) or value != sorted(set(value))):
        raise ValueError("comparison needs ordered unique original frame IDs")
    return value


def _rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(errors**2)))


def _read(report_file: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    _artifact(report_file, MAX_BYTES)
    report = json.loads(Path(report_file["path"]).read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise ValueError("unsupported mapping trajectory comparison schema")
    inputs, coverage = report["inputs"], report["coverage"]
    if not isinstance(inputs, dict):
        raise ValueError("comparison needs recorded input artifacts")
    if not {"job", "source", "pointcloud_map", "pointcloud_graph", "pointcloud_trajectory", "source_motion_trajectory", "reference"} <= set(inputs):
        raise ValueError("comparison is missing recorded input artifacts")
    for key, artifact in inputs.items():
        _artifact(artifact, MAX_BYTES if key in {"source_motion_trajectory", "pointcloud_graph", "pointcloud_trajectory", "reference"} else None)
    vertices = [line.split() for line in Path(inputs["pointcloud_graph"]["path"]).read_text().splitlines()]
    retained_ids = _ids(sorted(int(row[1]) for row in vertices if row and row[0] == "VERTEX_SE3:QUAT"))
    matched_ids = _ids(coverage["matched_original_frame_ids"])
    evaluated_ids = _ids(coverage["evaluated_original_frame_ids"])
    fitted_ids = _ids(coverage["alignment_original_frame_ids"])
    held_out = report["protocol"]["held_out_alignment"]
    if (type(held_out) is not bool or not set(matched_ids) <= set(retained_ids)
            or coverage["retained_poses"] != len(retained_ids) or coverage["matched_poses"] != len(matched_ids)
            or coverage["evaluated_poses"] != len(evaluated_ids) or coverage["alignment_fitted_poses"] != len(fitted_ids)
            or (fitted_ids + evaluated_ids != matched_ids if held_out else fitted_ids != matched_ids or evaluated_ids != matched_ids)):
        raise ValueError("comparison has inconsistent alignment/evaluation frame IDs")
    motion = load_trajectory(inputs["source_motion_trajectory"]["path"])
    if (motion["num_poses"] > MAX_POSES or retained_ids[-1] >= motion["num_poses"]
            or coverage["original_poses"] != motion["num_poses"]
            or not np.isclose(coverage["retained_pose_fraction"], len(matched_ids) / len(retained_ids), atol=1e-12, rtol=0)):
        raise ValueError("comparison frame IDs exceed original motion")
    corrected, _ = read_trajectory(Path(inputs["pointcloud_trajectory"]["path"]))
    if corrected.shape != (len(retained_ids), 4, 4) or not np.isfinite(corrected).all():
        raise ValueError("corrected trajectory and retained graph IDs disagree")
    indices = np.searchsorted(retained_ids, evaluated_ids)
    count = len(evaluated_ids)
    results = report["results"]
    times = _array(results["original"]["matched_trajectory"]["timestamps"], (count,))
    if not np.array_equal(times, motion["timestamps"][evaluated_ids]) or not np.all(np.diff(times) > 0):
        raise ValueError("comparison timestamps disagree with original frame IDs")
    truth = _array(results["original"]["matched_trajectory"]["reference_positions"], (count, 3))
    positions = {}
    for name, raw in (("original", motion["positions"][evaluated_ids]), ("corrected", corrected[indices, :3, 3])):
        result = results[name]
        samples = result["matched_trajectory"]
        if (not np.array_equal(_array(samples["timestamps"], (count,)), times)
                or not np.array_equal(_array(samples["reference_positions"], (count, 3)), truth)):
            raise ValueError("both estimates must use identical reference samples")
        positions[name] = _array(samples["estimated_positions"], (count, 3))
        rotation = _array(result["alignment"]["rotation_matrix"], (3, 3))
        translation = _array(result["alignment"]["translation"], (3,))
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-8, rtol=0)
                or not np.allclose(raw @ rotation.T + translation, positions[name], atol=1e-8, rtol=0)):
            raise ValueError("comparison samples disagree with recorded rigid alignment")
        errors = np.linalg.norm(positions[name] - truth, axis=1)
        if (not np.allclose(errors, _array(samples["ate_errors"], (count,)), atol=1e-9, rtol=0)
                or not np.isclose(_rmse(errors), result["ate"]["rmse"], atol=1e-9, rtol=0)):
            raise ValueError("comparison position errors disagree with saved samples")
    positions["reference"] = truth
    delta = results["corrected"]["ate"]["rmse"] - results["original"]["ate"]["rmse"]
    if not np.isclose(delta, report["change"]["ate_rmse_m_corrected_minus_original"], atol=1e-9, rtol=0):
        raise ValueError("comparison global change disagrees with saved samples")
    return report, corrected[:, :3, 3], indices, positions


def inspect_mapping_trajectory_comparison(
    report_file: dict[str, Any], window_poses: int = 12, ranking: str = "regression", offset: int = 0,
) -> dict[str, Any]:
    """Localize saved trajectory errors without loading clouds or spending attempts.

    Supply the exact report artifact returned by evaluate_mapping_trajectory.
    Verify the report and ALL recorded inputs before/after reading. Partition only
    evaluated retained poses into chronological, nonoverlapping windows of 2..64
    poses (last window may be shorter). Return eight windows per page, ordered by
    corrected-minus-original ATE RMSE (regression) or corrected ATE (corrected_ate).
    Exact original frame IDs, times and UNALIGNED corrected-map pose bounds locate
    review regions. Bounds cover sensor origins only, not scan returns or an
    authorized repair box. Reference uncertainty is uncalibrated; ranking is not
    a quality gate, a causal diagnosis or a repair/adoption decision.
    """
    if type(window_poses) is not int or not 2 <= window_poses <= 64:
        raise ValueError("window_poses must be an integer in [2, 64]")
    if ranking not in {"regression", "corrected_ate"}:
        raise ValueError("ranking must be regression or corrected_ate")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    try:
        report, map_positions, indices, positions = _read(report_file)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("malformed mapping trajectory comparison") from error
    coverage = report["coverage"]
    ids = coverage["evaluated_original_frame_ids"]
    times = report["results"]["original"]["matched_trajectory"]["timestamps"]
    distances = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(map_positions, axis=0), axis=1))))
    windows = []
    for start in range(0, len(ids), window_poses):
        end = min(start + window_poses, len(ids))
        chosen = indices[start:end]
        raw = map_positions[chosen]
        metrics: dict[str, dict[str, Any]] = {}
        for name in ("original", "corrected"):
            estimate, truth = positions[name][start:end], positions["reference"][start:end]
            metrics[name] = {"ate_rmse_m": _rmse(np.linalg.norm(estimate - truth, axis=1)),
                             "rpe_translation_rmse_m": _rmse(np.linalg.norm(np.diff(estimate, axis=0) - np.diff(truth, axis=0), axis=1)) if end - start > 1 else None}
        delta = metrics["corrected"]["ate_rmse_m"] - metrics["original"]["ate_rmse_m"]
        windows.append({"window_id": start // window_poses, "original_frame_ids": ids[start:end],
                        "timestamp_range_s": [times[start], times[end - 1]], "evaluated_poses": end - start,
                        "unevaluated_retained_poses_within_frame_span": int(chosen[-1] - chosen[0] + 1 - len(chosen)),
                        "corrected_graph_distance_range_m": [float(distances[chosen[0]]), float(distances[chosen[-1]])],
                        "evaluated_corrected_pose_bounds_xy": [float(raw[:, 0].min()), float(raw[:, 1].min()), float(raw[:, 0].max()), float(raw[:, 1].max())],
                        "results": metrics, "ate_rmse_m_corrected_minus_original": delta})
    metric = lambda window: window["ate_rmse_m_corrected_minus_original"] if ranking == "regression" else window["results"]["corrected"]["ate_rmse_m"]
    windows.sort(key=lambda window: (-metric(window), window["window_id"]))
    # Detect replacement or edits during parsing, including changes to large maps.
    for artifact in (report_file, *report["inputs"].values()):
        jobs._verify(artifact)
    return {"schema": REVIEW_SCHEMA, "report": report_file, "job": report["inputs"]["job"],
            "point_map": report["inputs"]["pointcloud_map"], "reference_provenance": report["reference_provenance"],
            "reference_independence": report["reference_independence"],
            "protocol": {"partition": "chronological_nonoverlapping_evaluated_retained_pose_windows", "window_poses": window_poses,
                         "ranking": ranking, "held_out_alignment": report["protocol"]["held_out_alignment"],
                         "alignment_prefix_fraction": report["protocol"]["alignment_prefix_fraction"],
                         "bounds_frame": "unaligned_corrected_point_map_sensor_origins_only",
                         "rpe_translation": "within_window_adjacent_evaluated_poses_variable_time_interval"},
            "coverage": {key: coverage[key] for key in ("original_poses", "retained_poses", "matched_poses", "evaluated_poses", "alignment_fitted_poses", "retained_pose_fraction")},
            "global_change": report["change"], "total_windows": len(windows), "offset": offset,
            "next_offset": offset + PAGE_SIZE if offset + PAGE_SIZE < len(windows) else None,
            "windows": windows[offset:offset + PAGE_SIZE],
            "guidance": "Review exact frames and source-supported map observations. Bounds are sensor-origin envelopes, not repair permissions. Error ranking cannot establish cause; density/HD-only repairs freeze motion and cannot fix trajectory error. Reference uncertainty and sensor correlation remain uncalibrated. No quality/adoption or attempts change."}
