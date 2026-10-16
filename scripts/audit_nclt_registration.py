#!/usr/bin/env python3
"""Audit registrations using frozen NCLT frontend predictions, without retuning.

This replays the saved initial guesses, not the inertial optimizer. Every pose
bit and iteration count must match the earlier frontend recording. No reference
trajectory is read and no runtime or independent accuracy claim is made.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib
import json
import os
from pathlib import Path

import numpy as np

import pilot_nclt_imu_window as pilot
from replay_nclt_imu_frontend import SCHEMA as FRONTEND_SCHEMA

SCHEMA = "cloudanalyzer.nclt_registration_diagnostics.v1"
COLUMNS = ["arm", "scan_index", "timestamp_ns", "frontend_status", "termination", "iterations",
           "input_points", "registration_points", "threshold_m", "matched_points", "matched_fraction",
           "rmse_m", "max_residual_m"]


def distribution(values):
    values = np.asarray(values, dtype=float)
    return None if not len(values) else {
        "count": len(values), "min": float(values.min()), "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def summarize(rows):
    return {"scans": len(rows), "terminations": dict(sorted(Counter(r["termination"] for r in rows).items())),
            "iterations": sum(r["iterations"] for r in rows),
            "residual_reports": sum(r["matched_points"] is not None for r in rows),
            "matched_fraction": distribution([r["matched_fraction"] for r in rows if r["matched_fraction"] is not None]),
            "rmse_m": distribution([r["rmse_m"] for r in rows if r["rmse_m"] is not None]),
            "max_residual_m": distribution([r["max_residual_m"] for r in rows if r["max_residual_m"] is not None])}


def audit(lidar: Path, frontend: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"registration audit output already exists: {output}")
    if output.resolve().is_relative_to(frontend.resolve()):
        raise ValueError("write the audit outside the frozen frontend recording")
    if os.environ.get("RAYON_NUM_THREADS") != "1":
        raise ValueError("RAYON_NUM_THREADS=1 is required to compare the frozen pose bits")
    core = importlib.import_module("cloudanalyzer_core")
    native = Path(importlib.import_module("cloudanalyzer_core._core").__file__)
    receipt_path = frontend / "receipt.json"
    previous = json.loads(receipt_path.read_text(encoding="utf-8"))
    if previous["schema"] != FRONTEND_SCHEMA:
        raise ValueError("expected a frozen NCLT frontend recording")

    def inputs():
        artifacts = {}
        for name, expected in previous["outputs"].items():
            if Path(name).name != name:
                raise ValueError("invalid frozen output name")
            actual = pilot.artifact(frontend / name)
            if actual != expected:
                raise ValueError("frozen frontend output hash mismatch")
            artifacts[name] = actual
        return {"lidar": pilot.artifact(lidar), "frontend_receipt": pilot.artifact(receipt_path),
                "frontend_outputs": artifacts, "native_extension": pilot.artifact(native),
                "controller": pilot.artifact(Path(__file__)), "pilot_module": pilot.artifact(Path(pilot.__file__)),
                "frontend_module": pilot.artifact(Path(importlib.import_module("replay_nclt_imu_frontend").__file__))}

    frozen = inputs()
    if frozen["lidar"] != previous["inputs"]["lidar"]:
        raise ValueError("LiDAR input differs from the frozen frontend recording")
    with np.load(frontend / "trajectory.npz", allow_pickle=False) as data:
        saved = {key: data[key].copy() for key in data.files}
    scans = list(core.BagReader(str(lidar)).scans())
    times = np.array([s.stamp_ns for s in scans], dtype=np.int64)
    pilot.strictly_increasing(times)
    np.testing.assert_array_equal(times, saved["timestamps_ns"])
    if not 4 <= len(scans) <= 4096:
        raise ValueError("audit needs 4..4096 scans")
    for arm in ["baseline", "assisted"]:
        if (saved[f"{arm}_body_poses"].shape != (len(scans), 4, 4)
                or saved[f"{arm}_relative_predictions"].shape != (len(scans), 4, 4)
                or saved[f"{arm}_status"].shape != times.shape
                or saved[f"{arm}_icp_iterations"].shape != times.shape):
            raise ValueError("invalid frozen frontend array dimensions")
        status = saved[f"{arm}_status"]
        if status[0] != "lidar_start" or set(status[1:]) - {"imu_prediction", "lidar_gap", "lidar_bootstrap"}:
            raise ValueError("invalid frozen prediction availability")
    protocol = {"schema": SCHEMA, "inputs": frozen, "lidar_settings": previous["settings"]["lidar"],
                "diagnostics": True, "rayon_num_threads": 1, "required_parity": "Every pose bit and ICP iteration count",
                "residual_policy": "Unweighted Euclidean distances at the returned pose, pre-insertion map, ICP 27-voxel search and threshold",
                "scope": "Saved prediction replay on an already examined cohort; no IMU optimization, ground-truth scoring, runtime measurement or calibrated quality threshold"}
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    rows, arms, arrays = [], {}, {"timestamps_ns": times}
    for arm in ["baseline", "assisted"]:
        odometry = core.LidarOdometry(**previous["settings"]["lidar"], diagnostics=True)
        poses, arm_rows = [], []
        for index, scan in enumerate(scans):
            status = str(saved[f"{arm}_status"][index])
            prediction = saved[f"{arm}_relative_predictions"][index] if arm == "assisted" and status == "imu_prediction" else None
            pose = odometry.register(scan.positions(), relative_prediction=prediction)
            expected = saved[f"{arm}_body_poses"][index]
            if not np.array_equal(pose.view(np.uint64), expected.view(np.uint64)):
                raise ValueError(f"pose bits changed: {arm} scan {index}")
            report = odometry.last_report()
            if report["iterations"] != int(saved[f"{arm}_icp_iterations"][index]):
                raise ValueError(f"ICP iteration count changed: {arm} scan {index}")
            stats = report.pop("correspondences")
            row = {"arm": arm, "scan_index": index, "timestamp_ns": int(times[index]),
                   "frontend_status": status, **report,
                   **{key: None if stats is None else stats[key] for key in COLUMNS[-4:]}}
            arm_rows.append(row)
            poses.append(pose)
            if index and index % 50 == 0:
                print(f"{arm}: audited scan {index}/{len(scans)}", flush=True)
        rows.extend(arm_rows)
        arrays[f"{arm}_body_poses"] = np.asarray(poses)
        arms[arm] = {"all": summarize(arm_rows), "by_frontend_status": {
            status: summarize([r for r in arm_rows if r["frontend_status"] == status])
            for status in sorted({r["frontend_status"] for r in arm_rows})},
            "pose_bits_unchanged": len(scans), "iteration_counts_unchanged": len(scans)}
    if inputs() != frozen:
        raise ValueError("registration audit input changed")
    with (output / "registrations.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "poses.npz", **arrays)
    result = {**protocol, "arms": arms,
              "outputs": {p.name: pilot.artifact(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["lidar", "frontend", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.lidar, args.frontend, args.out)["arms"], indent=2))


if __name__ == "__main__":
    main()
