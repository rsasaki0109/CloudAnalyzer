#!/usr/bin/env python3
"""Score a frozen NCLT IMU pilot against original body ground-truth CSV rows.

Evaluation is separate from estimation. No solver is invoked or retuned here.
All selected windows, including nonconverged results, remain in the comparison.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import nclt_body_frames as frames
from pilot_nclt_imu_window import SCHEMA, artifact, strictly_increasing


def reference_at(csv_path: Path, times: np.ndarray, tolerance_s=.05) -> np.ndarray:
    """Interpolate body origin and SO(3) using two nearby original GT rows."""
    strictly_increasing(times)
    stamps, poses = [], []
    with csv_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) != 7:
                raise ValueError("ground truth needs timestamp_us, xyz, roll/pitch/yaw")
            value = Decimal(row[0]) * 1000
            if not value.is_finite() or value != value.to_integral_value():
                raise ValueError("reference timestamp must be an exact integer nanosecond")
            stamps.append(int(value))
            xyzrpy = np.asarray(row[1:], dtype=np.float64)
            poses.append(frames.body_pose_z_up(xyzrpy[:3], xyzrpy[3:]))
    stamps = np.array(stamps, dtype=np.int64)
    strictly_increasing(stamps)
    poses = np.array(poses)
    result = []
    for t in times:
        right = int(np.searchsorted(stamps, t))
        if right < len(stamps) and stamps[right] == t:
            result.append(poses[right].copy())
            continue
        if right == 0 or right == len(stamps):
            raise ValueError("reference does not bracket scan time; no extrapolation")
        a, b = int(stamps[right - 1]), int(stamps[right])
        if max(int(t) - a, b - int(t)) * 1e-9 > tolerance_s:
            raise ValueError("reference bracket exceeds fixed timestamp tolerance")
        alpha = (int(t) - a) / (b - a)
        pose = np.eye(4)
        pose[:3, 3] = (1 - alpha) * poses[right - 1, :3, 3] + alpha * poses[right, :3, 3]
        pose[:3, :3] = Slerp([0., 1.], Rotation.from_matrix(poses[right - 1:right + 1, :3, :3]))([alpha]).as_matrix()[0]
        result.append(pose)
    return np.array(result)


def pose_errors(estimated: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    """SE3 relative errors and independent per-window rigid position-fit ATE."""
    if estimated.shape != reference.shape or estimated.ndim != 3 or estimated.shape[1:] != (4, 4) or len(estimated) < 2:
        raise ValueError("matched rigid-pose arrays (N,4,4), N >= 2, required")
    for series in (estimated, reference):
        for pose in series:
            frames.imu_pose_from_body_z_up(pose)  # Validate without repairing rotations.
    pe, pr = estimated[:, :3, 3], reference[:, :3, 3]
    re, rr = estimated[:, :3, :3], reference[:, :3, :3]
    # Rotation of the global evaluation gauge has no effect on relative poses.
    de = np.einsum("nji,nj->ni", re[:-1], np.diff(pe, axis=0))
    dr = np.einsum("nji,nj->ni", rr[:-1], np.diff(pr, axis=0))
    d_re = re[:-1].transpose(0, 2, 1) @ re[1:]
    d_rr = rr[:-1].transpose(0, 2, 1) @ rr[1:]
    rotation = np.linalg.norm(Rotation.from_matrix(d_rr.transpose(0, 2, 1) @ d_re).as_rotvec(), axis=1)
    ce, cr = pe.mean(axis=0), pr.mean(axis=0)
    u, _, vt = np.linalg.svd((pe - ce).T @ (pr - cr))
    correction = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    alignment = vt.T @ correction @ u.T
    aligned = (pe - ce) @ alignment.T + cr
    endpoint = re[0].T @ (pe[-1] - pe[0]) - rr[0].T @ (pr[-1] - pr[0])
    return {"ate_m": np.linalg.norm(aligned - pr, axis=1),
            "rpe_translation_m": np.linalg.norm(de - dr, axis=1),
            "rpe_rotation_deg": np.degrees(rotation),
            "endpoint_translation_m": np.array([np.linalg.norm(endpoint)])}


def summarize(errors: dict[str, np.ndarray]) -> dict:
    return {name: {"count": len(values), "rmse": float(np.sqrt(np.mean(values**2))),
                   "mean": float(np.mean(values)), "max": float(np.max(values))}
            for name, values in errors.items()}


def evaluate(pilot: Path, ground_truth: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    if output.resolve().is_relative_to(pilot.resolve()):
        raise ValueError("write evaluation outside the frozen estimation directory")
    receipt_path = pilot / "receipt.json"
    frozen = artifact(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["schema"] != SCHEMA or receipt["inputs"]["frame_module"] != artifact(Path(frames.__file__)):
        raise ValueError("unsupported pilot schema or changed frame convention")
    required = {"lidar.npz", *(w["name"] + ".npz" for w in receipt["windows"])}
    if not required.issubset(receipt["outputs"]):
        raise ValueError("pilot output manifest is incomplete")
    for name, expected in receipt["outputs"].items():
        if Path(name).name != name or artifact(pilot / name) != expected:
            raise ValueError("pilot output hash mismatch")
    truth_input = artifact(ground_truth)
    arrays = {}
    with np.load(pilot / "lidar.npz", allow_pickle=False) as baseline:
        all_times = baseline["timestamps_ns"].copy()
        baseline_poses = baseline["body_poses"].copy()
    # The same fixed bracket rule must support every scan, including those
    # outside fusion coverage. Never select windows using reference availability.
    truth = reference_at(ground_truth, all_times)
    pooled = {method: {} for method in ("lidar", "fused")}
    windows = []
    used = set()
    for i, window in enumerate(receipt["windows"]):
        start, stop, name = window["scan_start"], window["scan_stop"], window["name"]
        if Path(name).name != name or start < 0 or stop > len(all_times) or stop - start < 2:
            raise ValueError("invalid saved pilot window")
        indices = np.arange(start, stop)
        if used.intersection(indices):
            raise ValueError("pilot windows must be disjoint")
        used.update(indices)
        with np.load(pilot / f"{name}.npz", allow_pickle=False) as state:
            np.testing.assert_array_equal(state["timestamps_ns"], all_times[start:stop])
            np.testing.assert_array_equal(state["scan_indices"], indices)
            np.testing.assert_array_equal(state["lidar_body_poses"], baseline_poses[start:stop])
            estimates = {"lidar": state["lidar_body_poses"], "fused": state["fused_body_poses"]}
            metrics = {}
            for method, poses in estimates.items():
                errors = pose_errors(poses, truth[start:stop])
                metrics[method] = summarize(errors)
                for key, values in errors.items():
                    pooled[method].setdefault(key, []).append(values)
                    arrays[f"{i:03d}_{method}_{key}"] = values
        windows.append({"name": name, "scan_start": start, "scan_stop": stop,
                        "converged": window["converged"], "metrics": metrics})
    if not windows or len(used) != receipt["coverage"]["eligible_nodes"]:
        raise ValueError("incomplete pilot coverage")
    combined = {method: summarize({key: np.concatenate(values) for key, values in block.items()})
                for method, block in pooled.items()}
    result = {"schema": "cloudanalyzer.nclt_imu_pilot_evaluation.v1",
              "inputs": {"estimation_receipt": frozen, "ground_truth": truth_input,
                         "evaluator": artifact(Path(__file__)), "frame_module": artifact(Path(frames.__file__))},
              "reference_policy": "Original body rows -> body z up; xyz interpolation and SO3 SLERP, .05 s per bracket, no extrapolation",
              "primary_metrics": "Pooled within-window consecutive SE3 RPE; identical edges for both methods",
              "ate_policy": "Each window independently rigid-position aligned, no scale; short windows can hide drift",
              "scope": "Offline diagnostic pilot; correlated LiDAR priors, uncalibrated noise and first-order bias model; no default-fusion adoption or GLIM comparison",
              "coverage": receipt["coverage"], "metrics": combined, "windows": windows}
    if artifact(receipt_path) != frozen or artifact(ground_truth) != truth_input:
        raise ValueError("evaluation input changed")
    if any(artifact(pilot / name) != expected for name, expected in receipt["outputs"].items()):
        raise ValueError("pilot output changed during evaluation")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "errors.npz", **arrays)
    result["outputs"] = {"errors.npz": artifact(output / "errors.npz")}
    (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.pilot, args.ground_truth, args.out)["metrics"], indent=2))


if __name__ == "__main__":
    main()
