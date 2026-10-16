#!/usr/bin/env python3
"""Ordered bounded-window replay of frozen NCLT LiDAR poses and original IMU.

No ground-truth input. Latest-state outputs are emitted once per scan; original
IMU gaps reset the inertial window and emit an explicit LiDAR-only fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

import nclt_body_frames as frames
import pilot_nclt_imu_window as pilot

SCHEMA = "cloudanalyzer.nclt_imu_replay.v1"
MAX_STATES = 4


def frame_source_matches(captured: dict) -> bool:
    raw = Path(frames.__file__).read_bytes().replace(b"\r\n", b"\n")
    return captured["sha256"] in {hashlib.sha256(b).hexdigest() for b in (raw, raw.replace(b"\n", b"\r\n"))}


def advance(core, window, times, previous_pose, observed_pose, factor, config):
    """One chronological update; only the current/previous LiDAR observations."""
    dt = (int(times[1]) - int(times[0])) * 1e-9
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("replay update timestamps must increase")
    velocity = (observed_pose[:3, 3] - previous_pose[:3, 3]) / dt
    marginalization = None
    if window is None:
        biases = np.zeros((2, 6))
        window = core.ImuWindow(times, np.array([previous_pose, observed_pose]),
                                np.tile(velocity, (2, 1)), biases, gravity=config["gravity"])
        window.add_imu_factor(0, factor, gyro_bias_random_walk=config["gyro_bias_random_walk"],
                              accel_bias_random_walk=config["accel_bias_random_walk"])
        window.add_state_prior(0, previous_pose, velocity, biases[0], sigmas=config["state_sigmas"])
        index = 1
    else:
        states = window.states()
        if len(states["timestamps_ns"]) == MAX_STATES:
            marginalization = window.marginalize_oldest()
        window.append_state(int(times[1]), observed_pose, velocity, states["biases"][-1], factor,
                             gyro_bias_random_walk=config["gyro_bias_random_walk"],
                             accel_bias_random_walk=config["accel_bias_random_walk"])
        index = len(window.states()["timestamps_ns"]) - 1
    window.add_pose_prior(index, observed_pose, rotation_sigma=config["pose_rotation_sigma"],
                           position_sigma=config["pose_position_sigma"])
    result = window.optimize(**config["optimizer"])
    return window, result, marginalization


def replay(pilot_dir: Path, imu: Path, euler: Path, output: Path, *, imu_topic="/ms25/raw") -> dict:
    if output.exists():
        raise FileExistsError(f"replay output already exists: {output}")
    if output.resolve().is_relative_to(pilot_dir.resolve()) or (imu.is_dir() and output.resolve().is_relative_to(imu.resolve())):
        raise ValueError("write replay outside frozen inputs")
    baseline_path, source_receipt_path = pilot_dir / "lidar.npz", pilot_dir / "receipt.json"
    receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    if receipt["schema"] != pilot.SCHEMA or receipt["interpreted_scan_frame"] != frames.FRAME_ID:
        raise ValueError("expected a frozen body-z-up NCLT pilot baseline")
    if not frame_source_matches(receipt["inputs"]["frame_module"]):
        raise ValueError("baseline frame convention changed")
    if pilot.artifact(baseline_path) != receipt["outputs"]["lidar.npz"]:
        raise ValueError("frozen LiDAR baseline hash mismatch")
    core = importlib.import_module("cloudanalyzer_core")
    native = Path(importlib.import_module("cloudanalyzer_core._core").__file__)
    config = pilot.settings()
    inputs = {"baseline_receipt": pilot.artifact(source_receipt_path), "lidar_poses": pilot.artifact(baseline_path),
              "imu": pilot.recording_artifacts(imu), "euler": pilot.artifact(euler),
              "controller": pilot.artifact(Path(__file__)), "frame_module": pilot.artifact(Path(frames.__file__)),
              "pilot_module": pilot.artifact(Path(pilot.__file__)), "native_extension": pilot.artifact(native)}
    with np.load(baseline_path, allow_pickle=False) as data:
        times, body = data["timestamps_ns"].copy(), data["body_poses"].copy()
    pilot.strictly_increasing(times)
    if len(times) > 4096 or body.shape != (len(times), 4, 4):
        raise ValueError("expected at most 4096 matched body poses")
    # No stored offline pilot velocities or states are loaded.
    poses = np.array([frames.imu_pose_from_body_z_up(p) for p in body])
    reader = core.BagReader(str(imu))
    raw = reader.imu_raw(topic=imu_topic)
    if set(raw["frame_ids"]) != {"nclt_ms25_native"}:
        raise ValueError("raw IMU vectors must use the nclt_ms25_native frame")
    rejected = set(pilot.gap_edges(times, raw["timestamps_ns"], config["max_gap_s"]))
    gravity, attitude = pilot.initial_gravity(euler, int(times[0]))
    config["gravity"] = gravity.tolist()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "inputs": inputs, "settings": config, "max_states": MAX_STATES,
              "imu_topic": imu_topic, "initial_attitude": attitude,
              "scope": "Latest-state pose/IMU replay, fixed diagnostic weights, full trajectory including LiDAR fallbacks; no LiDAR frontend feedback, calibrated covariance or real-time speed claim",
              "updates": [], "marginalizations": [], "gap_resets": sorted(rejected)}
    (output / "protocol.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fused = body.copy()
    velocities, biases = np.zeros((len(times), 3)), np.zeros((len(times), 6))
    status = np.full(len(times), "lidar_gap", dtype="<U16")
    status[0] = "lidar_start"
    active_sizes = np.zeros(len(times), dtype=np.int64)
    window = None
    inverse_mount = np.linalg.inv(frames.BODY_UP_FROM_IMU)
    for i in range(1, len(times)):
        if i - 1 in rejected:
            # No integration, marginal-prior carry or uncertainty bridge across gaps.
            window = None
            continue
        integration_bias = np.zeros(6) if window is None else window.states()["biases"][-1]
        factor = reader.imu_factor(int(times[i - 1]), int(times[i]), bias=integration_bias,
                                   topic=imu_topic, gyro_noise_density=config["gyro_noise_density"],
                                   accel_noise_density=config["accel_noise_density"], max_gap_s=config["max_gap_s"])
        window, result, marginalized = advance(core, window, times[i - 1:i + 1], poses[i - 1], poses[i], factor, config)
        fused[i] = result["poses"][-1] @ inverse_mount
        velocities[i], biases[i] = result["velocities"][-1], result["biases"][-1]
        status[i] = "fused"
        active_sizes[i] = len(result["timestamps_ns"])
        report["updates"].append({"scan_index": i, "active_states": int(active_sizes[i]),
                                  **{k: result[k] for k in ["initial_cost", "final_cost", "iterations", "converged", "termination"]},
                                  "max_abs_bias_correction": result["max_abs_bias_correction"].tolist()})
        if marginalized is not None:
            report["marginalizations"].append({"scan_index": i, **marginalized})
        if i % 50 == 0:
            print(f"scan {i}/{len(times)}, active states {active_sizes[i]}", flush=True)
    np.savez_compressed(output / "trajectory.npz", timestamps_ns=times, lidar_body_poses=body,
                        replay_body_poses=fused, velocities=velocities, biases=biases,
                        status=status, active_sizes=active_sizes)
    pilot.write_tum(output / "trajectory.tum", times, fused)
    report["coverage"] = {"scans": len(times), "all_edges": len(times) - 1,
                          "fused_updates": len(report["updates"]), "gap_fallbacks": len(rejected), "startup_fallbacks": 1,
                          "marginalizations": len(report["marginalizations"]), "max_active_states": int(active_sizes.max()),
                          "nonconverged_updates": sum(not u["converged"] for u in report["updates"]),
                          "output_policy": "All scans, latest output once per scan; fallback velocities/biases are placeholders, not IMU estimates"}
    after = {"baseline_receipt": pilot.artifact(source_receipt_path), "lidar_poses": pilot.artifact(baseline_path),
             "imu": pilot.recording_artifacts(imu), "euler": pilot.artifact(euler),
             "controller": pilot.artifact(Path(__file__)), "frame_module": pilot.artifact(Path(frames.__file__)),
             "pilot_module": pilot.artifact(Path(pilot.__file__)), "native_extension": pilot.artifact(native)}
    if after != inputs:
        raise ValueError("replay input changed")
    report["outputs"] = {p.name: pilot.artifact(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--imu", type=Path, required=True)
    parser.add_argument("--euler", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imu-topic", default="/ms25/raw")
    args = parser.parse_args()
    print(json.dumps(replay(args.pilot, args.imu, args.euler, args.out, imu_topic=args.imu_topic)["coverage"]))


if __name__ == "__main__":
    main()
