"""Read-only, timestamp-bound comparison of mapping motion against a supplied reference."""
from __future__ import annotations

import json
import math
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

from ca import mapping_job as jobs
from ca.posegraph_fix import read_trajectory
from ca.trajectory import (
    _apply_rigid_alignment, _interpolate_matches, _interpolate_orientation_series, evaluate_trajectory, load_trajectory,
)

SCHEMA = "cloudanalyzer.mapping_trajectory_comparison.v1"
MAX_BYTES = 16 * 1024**2
MAX_POSES = 4096


def _provenance(value: dict[str, Any]) -> dict[str, Any]:
    keys = {"source", "license", "frame", "time_basis", "used_for_generation"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("reference provenance needs source, license, frame, time_basis and used_for_generation")
    for key in keys - {"used_for_generation"}:
        if not isinstance(value[key], str) or not 1 <= len(value[key].strip()) <= 2048:
            raise ValueError(f"reference provenance {key} needs a bounded nonempty string")
    if type(value["used_for_generation"]) is not bool:
        raise ValueError("used_for_generation must be an explicit boolean")
    return dict(value)


def _finite_trajectory(value: dict[str, Any]) -> None:
    if (value["num_poses"] > MAX_POSES or not np.isfinite(value["timestamps"]).all()
            or not np.isfinite(value["positions"]).all()):
        raise ValueError("trajectory needs at most 4096 finite poses and timestamps")


def _write(path: Path, times: np.ndarray, positions: np.ndarray, orientations: np.ndarray | None) -> None:
    columns = [times[:, None], positions]
    if orientations is not None:
        columns.append(orientations)
    np.savetxt(path, np.column_stack(columns), fmt="%.17g")


def evaluate_mapping_trajectory(
    job_dir: str, reference: str, reference_provenance: dict[str, Any], report_path: str,
    max_time_delta: float = .05,
    alignment_prefix_fraction: float = 1.,
) -> dict[str, Any]:
    """Compare original and corrected mapping poses on identical reference-supported timestamps.

    Supply a metre-frame TUM/CSV reference, its source/license/frame/time basis and
    explicit used_for_generation declaration, plus a NEW external report path.
    Requires saved hashed source_motion and original-frame graph IDs. Interpolate
    the reference only within its coverage and with both brackets within the time
    tolerance. Each estimate fits its own scale-free rigid alignment on the same
    matched poses. With alignment_prefix_fraction below 1, fit only that prefix
    and evaluate only the disjoint suffix; default 1 fits/evaluates all samples.
    This is trajectory shape evaluation, not map certification. Preserve input
    hashes, coverage and both ATE/RPE results. No job, map, attempt, selection or
    quality status changes. Caller declarations do not establish independence,
    surveyed map accuracy, georeferencing or road-use readiness.
    """
    provenance = _provenance(reference_provenance)
    if isinstance(max_time_delta, bool) or not math.isfinite(max_time_delta) or not 0 < max_time_delta <= 1:
        raise ValueError("max_time_delta must be finite and in (0, 1] seconds")
    if isinstance(alignment_prefix_fraction, bool) or not math.isfinite(alignment_prefix_fraction) or not 0 < alignment_prefix_fraction <= 1:
        raise ValueError("alignment_prefix_fraction must be finite and in (0, 1]")
    root, target = Path(job_dir).resolve(), Path(report_path).resolve()
    if target.exists():
        raise FileExistsError(f"report already exists: {target}")
    if target.is_relative_to(root) or any((parent / "job.json").is_file() for parent in (target.parent, *target.parent.parents)):
        raise ValueError("save the comparison outside the mapping job")
    snapshot = jobs._artifact(root / "job.json")
    job = jobs._load(root)
    pointcloud = job.get("pointcloud")
    if not pointcloud or not pointcloud.get("source_motion"):
        raise ValueError("comparison requires a job with hashed original source_motion")
    motion = pointcloud["source_motion"]
    inputs = {"job": snapshot, "source": job["source"],
              **{f"pointcloud_{k}": v for k, v in pointcloud["files"].items()},
              **{f"source_motion_{k}": v for k, v in motion.items()},
              "reference": jobs._artifact(reference)}
    for artifact in inputs.values():
        jobs._verify(artifact)
    for key in ("source_motion_trajectory", "pointcloud_graph", "pointcloud_trajectory", "reference"):
        if inputs[key]["bytes"] > MAX_BYTES:
            raise ValueError("trajectory/graph/reference exceeds the 16 MiB input limit")
    original = load_trajectory(motion["trajectory"]["path"])
    truth = load_trajectory(reference)
    _finite_trajectory(original); _finite_trajectory(truth)
    module = jobs.core()
    if module is None:
        raise RuntimeError("mapping trajectory comparison requires the native core")
    graph = module.PoseGraph.from_g2o(Path(inputs["pointcloud_graph"]["path"]).read_text())
    ids = list(graph.node_ids)
    if (len(ids) < 3 or len(ids) > MAX_POSES or ids != sorted(set(ids))
            or any(type(i) is not int or i < 0 or i >= original["num_poses"] for i in ids)):
        raise ValueError("graph needs 3..4096 ordered original-frame node IDs")
    corrected, _ = read_trajectory(Path(inputs["pointcloud_trajectory"]["path"]))
    graph_poses = np.asarray(graph.poses())
    if (corrected.shape != (len(ids), 4, 4) or not np.isfinite(corrected).all()
            or not np.allclose(corrected, graph_poses, atol=1e-10, rtol=0)):
        raise ValueError("corrected graph and frozen trajectory do not agree")
    rotations = corrected[:, :3, :3]
    if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-6, rtol=0)):
        raise ValueError("corrected poses need proper rigid rotations")
    times = original["timestamps"][ids]
    covered = times[(times >= truth["timestamps"][0]) & (times <= truth["timestamps"][-1])]
    matched, positions, _, deltas = _interpolate_matches(
        truth["timestamps"], truth["positions"], covered, np.zeros((len(covered), 3)), max_time_delta,
    )
    if len(matched) < 3:
        raise ValueError("need at least 3 reference-supported retained poses")
    # A line/point cannot constrain a full 3D rigid position alignment.
    if np.linalg.matrix_rank(positions - positions.mean(axis=0), tol=1e-8) < 2:
        raise ValueError("reference positions cannot constrain rigid alignment")
    selected = np.searchsorted(times, matched)
    original_ids = np.asarray(ids)[selected]
    quaternions = _interpolate_orientation_series(
        truth["timestamps"], truth.get("orientations"), matched, max_time_delta,
    )
    held_out = alignment_prefix_fraction < 1.
    fit_count = int(len(matched) * alignment_prefix_fraction) if held_out else len(matched)
    if held_out and (fit_count < 3 or len(matched) - fit_count < 3):
        raise ValueError("prefix alignment needs at least 3 fitting and 3 evaluated poses")
    estimates = {
        "original": (original["positions"][original_ids], None if original["orientations"] is None else original["orientations"][original_ids]),
        "corrected": (corrected[selected, :3, 3], Rotation.from_matrix(rotations[selected]).as_quat()),
    }
    evaluation_start = fit_count if held_out else 0
    if held_out:
        for values in (positions, *(pair[0] for pair in estimates.values())):
            if np.linalg.matrix_rank(values[:fit_count] - values[:fit_count].mean(axis=0), tol=1e-8) < 2:
                raise ValueError("alignment prefix cannot constrain rigid alignment")
    with tempfile.TemporaryDirectory(prefix="ca-mapping-trajectory-") as folder:
        staging = Path(folder)
        _write(staging / "reference.tum", matched[evaluation_start:], positions[evaluation_start:],
               None if quaternions is None else quaternions[evaluation_start:])
        results = {}
        for name in ("original", "corrected"):
            estimate_positions, estimate_quaternions = estimates[name]
            if held_out:
                fitted, translation, rotation = _apply_rigid_alignment(estimate_positions[:fit_count], positions[:fit_count])
                estimate_positions = estimate_positions @ rotation.T + translation
                if estimate_quaternions is not None:
                    estimate_quaternions = (Rotation.from_matrix(rotation) * Rotation.from_quat(estimate_quaternions)).as_quat()
            _write(staging / f"{name}.tum", matched[evaluation_start:], estimate_positions[evaluation_start:],
                   None if estimate_quaternions is None else estimate_quaternions[evaluation_start:])
            result = evaluate_trajectory(str(staging / f"{name}.tum"), str(staging / "reference.tum"), align_rigid=not held_out)
            if held_out:
                result["alignment"] = {"mode": "rigid_prefix", "translation": translation.tolist(),
                                       "rotation_matrix": rotation.tolist(),
                                       "fit_rmse_m": float(np.sqrt(np.mean(np.sum((fitted - positions[:fit_count])**2, axis=1))))}
            result["estimated_path"] = inputs["source_motion_trajectory" if name == "original" else "pointcloud_trajectory"]["path"]
            result["reference_path"] = inputs["reference"]["path"]
            # No calibrated thresholds were supplied; do not emit an empty passing gate.
            result.pop("quality_gate")
            results[name] = result
    known_reuse = [key for key, artifact in inputs.items() if key != "reference"
                   and artifact["sha256"] == inputs["reference"]["sha256"]]
    report = {
        "schema": SCHEMA, "inputs": inputs, "reference_provenance": provenance,
        "reference_independence": "not_independent" if provenance["used_for_generation"] or known_reuse else "caller_declared_not_used_for_generation",
        "reference_matches_recorded_inputs": known_reuse,
        "protocol": {"units": "metres_seconds", "max_time_delta_s": max_time_delta,
                     "matching": "reference_interpolated_at_retained_original_frame_timestamps_no_extrapolation",
                     "alignment": "separate_SE3_prefix_fits_disjoint_suffix_evaluation_no_scale" if held_out else "separate_SE3_fits_on_identical_matched_positions_no_scale",
                     "rpe_translation": "adjacent_matched_retained_poses_variable_time_interval",
                     "held_out_alignment": held_out, "alignment_prefix_fraction": alignment_prefix_fraction,
                     "native": jobs._native(),
                     "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "coverage": {"original_poses": original["num_poses"], "retained_poses": len(ids),
                     "matched_poses": len(matched), "retained_pose_fraction": len(matched) / len(ids),
                     "retained_duration_s": float(times[-1] - times[0]),
                     "matched_duration_s": float(matched[-1] - matched[0]),
                     "mean_nearest_reference_delta_s": float(np.mean(deltas)),
                     "max_nearest_reference_delta_s": float(np.max(deltas)),
                     "matched_original_frame_ids": original_ids.tolist(),
                     "alignment_fitted_poses": fit_count, "evaluated_poses": len(matched) - evaluation_start,
                     "alignment_original_frame_ids": original_ids[:fit_count].tolist(),
                     "evaluated_original_frame_ids": original_ids[evaluation_start:].tolist()},
        "results": results,
        "change": {"ate_rmse_m_corrected_minus_original": results["corrected"]["ate"]["rmse"] - results["original"]["ate"]["rmse"]},
        "scope": "Trajectory diagnostic only. Reference independence is caller-declared; consult alignment/evaluation split. Does not certify point/HD map accuracy or change job quality/adoption.",
    }
    # Detect interleaved changes before publishing anything; never replace a report.
    for artifact in inputs.values():
        jobs._verify(artifact)
    encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".ca-trajectory-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(encoded)
            stream.flush()
            os.link(temporary, target)
        finally:
            temporary.unlink()
    # Keep the agent response bounded; all matched samples remain in the hashed file.
    summary = {name: {key: value for key, value in result.items() if key not in {"matched_trajectory", "error_series"}}
               for name, result in results.items()}
    return {**report, "results": summary, "report": jobs._artifact(target)}
