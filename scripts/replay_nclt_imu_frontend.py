#!/usr/bin/env python3
"""Experimental causal IMU predictions for NCLT body-frame LiDAR registration.

Both arms optimize the same four-state diagnostic inertial window. Only the
assisted arm supplies its next relative prediction to ICP. The map and emitted
trajectory always use ICP's poses. No reference trajectory is read here.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

import nclt_body_frames as frames
import pilot_nclt_imu_window as pilot
import replay_nclt_imu_window as window_replay

SCHEMA = "cloudanalyzer.nclt_imu_frontend.v1"


class Frontend:
    """Ordered experiment state, with explicit gap resets and startup fallback."""

    def __init__(self, core, series, times, config, *, use_prediction):
        pilot.strictly_increasing(times)
        self.core, self.series, self.times, self.config = core, series, times.copy(), config
        self.use_prediction = use_prediction
        self.gaps = set(pilot.gap_edges(times, series.timestamps_ns(), config["max_gap_s"]))
        self.odometry = core.LidarOdometry(**config["lidar"])
        self.window = None
        self.inverse_mount = np.linalg.inv(frames.BODY_UP_FROM_IMU)
        self.poses, self.backend_poses, self.velocities, self.biases = [], [], [], []
        self.predictions, self.status, self.active_sizes, self.iterations = [], [], [], []
        self.updates, self.marginalizations = [], []

    def register(self, index, points):
        if index != len(self.poses) or not 0 <= index < len(self.times):
            raise ValueError("frontend scans must be consumed once in timestamp order")
        factor, delta = None, None
        status = "lidar_start" if index == 0 else "lidar_gap"
        if index and index - 1 not in self.gaps:
            states = None if self.window is None else self.window.states()
            bias = np.zeros(6) if states is None else states["biases"][-1]
            factor = self.series.factor(int(self.times[index - 1]), int(self.times[index]), bias=bias,
                                        gyro_noise_density=self.config["gyro_noise_density"],
                                        accel_noise_density=self.config["accel_noise_density"],
                                        max_gap_s=self.config["max_gap_s"])
            status = "lidar_bootstrap"
            if states is not None:
                # This runs before registering the current points. Only the previous
                # optimized state and this interval's held IMU samples are used.
                initial = states["poses"][-1]
                predicted = factor.predict(initial, states["velocities"][-1],
                                           gravity=self.config["gravity"], bias=bias)
                previous_body = initial @ self.inverse_mount
                predicted_body = predicted["pose"] @ self.inverse_mount
                delta = np.linalg.inv(previous_body) @ predicted_body
                status = "imu_prediction"
        else:
            self.window = None
        pose = self.odometry.register(points, relative_prediction=delta if self.use_prediction else None)
        self.iterations.append(self.odometry.last_iterations())
        backend, velocity, bias, active = pose.copy(), np.zeros(3), np.zeros(6), 0
        if factor is not None:
            observed = frames.imu_pose_from_body_z_up(pose)
            previous = frames.imu_pose_from_body_z_up(self.poses[-1])
            self.window, result, marginalized = window_replay.advance(
                self.core, self.window, self.times[index - 1:index + 1], previous, observed, factor, self.config)
            backend = result["poses"][-1] @ self.inverse_mount
            velocity, bias = result["velocities"][-1], result["biases"][-1]
            active = len(result["timestamps_ns"])
            self.updates.append({"scan_index": index, "active_states": active,
                                 **{k: result[k] for k in ["initial_cost", "final_cost", "iterations", "converged", "termination"]},
                                 "max_abs_bias_correction": result["max_abs_bias_correction"].tolist()})
            if marginalized is not None:
                self.marginalizations.append({"scan_index": index, **marginalized})
        self.poses.append(pose.copy())
        self.backend_poses.append(backend.copy())
        self.velocities.append(velocity.copy())
        self.biases.append(bias.copy())
        self.predictions.append(np.eye(4) if delta is None else delta.copy())
        self.status.append(status)
        self.active_sizes.append(active)
        return pose

    def coverage(self):
        return {"scans": len(self.poses), "all_edges": len(self.poses) - 1,
                "prediction_available": self.status.count("imu_prediction"),
                "prediction_used": self.status.count("imu_prediction") if self.use_prediction else 0,
                "gap_fallbacks": self.status.count("lidar_gap"),
                "bootstrap_fallbacks": self.status.count("lidar_bootstrap"), "startup_fallbacks": 1,
                "max_active_states": max(self.active_sizes), "marginalizations": len(self.marginalizations),
                "nonconverged_updates": sum(not u["converged"] for u in self.updates),
                "icp_iterations": sum(self.iterations)}


def replay(lidar: Path, imu: Path, euler: Path, output: Path, *, imu_topic="/ms25/raw") -> dict:
    if output.exists():
        raise FileExistsError(f"frontend replay output already exists: {output}")
    if imu.is_dir() and output.resolve().is_relative_to(imu.resolve()):
        raise ValueError("write replay outside the input recording")
    core = importlib.import_module("cloudanalyzer_core")
    native = Path(importlib.import_module("cloudanalyzer_core._core").__file__)

    def inputs():
        return {"lidar": pilot.artifact(lidar), "imu": pilot.recording_artifacts(imu), "euler": pilot.artifact(euler),
                "controller": pilot.artifact(Path(__file__)), "frame_module": pilot.artifact(Path(frames.__file__)),
                "pilot_module": pilot.artifact(Path(pilot.__file__)),
                "window_module": pilot.artifact(Path(window_replay.__file__)), "native_extension": pilot.artifact(native)}

    frozen = inputs()
    scans = list(core.BagReader(str(lidar)).scans())
    if not 4 <= len(scans) <= 4096:
        raise ValueError("frontend replay needs 4..4096 synchronized body-frame scans")
    if len({s.frame_id for s in scans}) != 1 or scans[0].frame_id not in {"velodyne", frames.FRAME_ID}:
        raise ValueError("verify NCLT synchronized body-z-up scan provenance")
    times = np.array([s.stamp_ns for s in scans], dtype=np.int64)
    series = core.BagReader(str(imu)).imu_series(topic=imu_topic)
    if set(series.frame_ids()) != {"nclt_ms25_native"}:
        raise ValueError("raw IMU vectors must use the nclt_ms25_native frame")
    config = pilot.settings()
    gravity, attitude = pilot.initial_gravity(euler, int(times[0]))
    config["gravity"] = gravity.tolist()
    config["max_states"] = window_replay.MAX_STATES
    report = {"schema": SCHEMA, "inputs": frozen, "settings": config, "imu_topic": imu_topic,
              "initial_attitude": attitude, "scan_frame_id": scans[0].frame_id,
              "interpreted_scan_frame": frames.FRAME_ID,
              "scope": "Opt-in frontend prediction diagnostic; ICP poses build the map and are emitted once per scan. Fixed uncalibrated weights; no default adoption or real-time speed claim.",
              "arms": {}}
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    arrays = {"timestamps_ns": times}
    for name, feedback in [("baseline", False), ("assisted", True)]:
        frontend = Frontend(core, series, times, config, use_prediction=feedback)
        for index, scan in enumerate(scans):
            frontend.register(index, scan.positions())
            if index and index % 50 == 0:
                print(f"{name}: scan {index}/{len(scans)}", flush=True)
        for key, value in [("body_poses", frontend.poses), ("backend_body_poses", frontend.backend_poses),
                           ("velocities", frontend.velocities), ("biases", frontend.biases),
                           ("relative_predictions", frontend.predictions), ("status", frontend.status),
                           ("active_sizes", frontend.active_sizes), ("icp_iterations", frontend.iterations)]:
            arrays[f"{name}_{key}"] = np.asarray(value)
        report["arms"][name] = {"coverage": frontend.coverage(), "gap_edges": sorted(frontend.gaps),
                                "updates": frontend.updates, "marginalizations": frontend.marginalizations}
        pilot.write_tum(output / f"{name}.tum", times, np.asarray(frontend.poses))
    if inputs() != frozen:
        raise ValueError("frontend replay input changed")
    np.savez_compressed(output / "trajectory.npz", **arrays)
    report["outputs"] = {p.name: pilot.artifact(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["lidar", "imu", "euler", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--imu-topic", default="/ms25/raw")
    args = parser.parse_args()
    result = replay(args.lidar, args.imu, args.euler, args.out, imu_topic=args.imu_topic)
    print(json.dumps({k: v["coverage"] for k, v in result["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
