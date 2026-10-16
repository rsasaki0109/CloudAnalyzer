#!/usr/bin/env python3
"""Bounded offline NCLT LiDAR-pose/raw-IMU pilot, with no reference input.

Inputs must be velodyne_sync-derived body-z-up scans and original ms25 IMU
vectors in native axes. See docs/slam-nclt-imu-pilot.md before interpreting
results. The fixed noise model is diagnostic, not a sensor calibration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import nclt_body_frames as frames

SCHEMA = "cloudanalyzer.nclt_imu_pilot.v1"


def artifact(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def settings() -> dict:
    """Fixed before scoring; changing these requires a distinct experiment."""
    return {
        "lidar": {"min_range": 1.5, "max_range": 80., "map_voxel": 1.,
                  "map_points": 20, "deskew": False},
        "min_states": 4, "max_states": 16, "max_gap_s": .05,
        "gyro_noise_density": .02, "accel_noise_density": .15,
        "gyro_bias_random_walk": .005, "accel_bias_random_walk": .05,
        "pose_rotation_sigma": .02, "pose_position_sigma": .1,
        "state_sigmas": [.02] * 3 + [.5] * 3 + [.05] * 3 + [.02] * 3 + [.5] * 3,
        "optimizer": {"max_iterations": 80, "initial_damping": 1e-3,
                      "gradient_tolerance": 1e-8, "step_tolerance": 1e-9,
                      "cost_tolerance": 1e-10},
    }


def strictly_increasing(times: np.ndarray) -> None:
    if (times.ndim != 1 or len(times) < 2 or times.dtype != np.dtype("int64")
            or any(int(b) <= int(a) for a, b in zip(times[:-1], times[1:]))):
        raise ValueError("need at least two strictly increasing int64 header timestamps")


def gap_edges(scan_times: np.ndarray, imu_times: np.ndarray, max_gap_s: float) -> list[int]:
    """Test original held-sample intervals before clipping to scan endpoints."""
    strictly_increasing(scan_times)
    strictly_increasing(imu_times)
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive and finite")
    if imu_times[0] > scan_times[0] or imu_times[-1] < scan_times[-1]:
        raise ValueError("raw IMU must bracket every LiDAR header time; no extrapolation")
    gaps = [i for i, (a, b) in enumerate(zip(imu_times[:-1], imu_times[1:]))
            if (int(b) - int(a)) * 1e-9 > max_gap_s]
    return [i for i, (a, b) in enumerate(zip(scan_times[:-1], scan_times[1:]))
            if any(imu_times[k] < b and imu_times[k + 1] > a for k in gaps)]


def select_windows(count: int, rejected_edges: list[int], *, minimum=4, maximum=16) -> list[tuple[int, int]]:
    """Disjoint [start, stop) chunks of valid runs; small tails stay excluded."""
    if not 2 <= minimum <= maximum <= 32 or count < 2:
        raise ValueError("need 2..32 state bounds and at least two scans")
    if (rejected_edges != sorted(set(rejected_edges))
            or any(i < 0 or i >= count - 1 for i in rejected_edges)):
        raise ValueError("rejected edges must be sorted unique valid scan indices")
    windows = []
    start = 0
    for stop in [i + 1 for i in rejected_edges] + [count]:
        for left in range(start, stop, maximum):
            right = min(left + maximum, stop)
            if right - left >= minimum:
                windows.append((left, right))
        start = stop
    return windows


def initial_gravity(euler_csv: Path, first_scan_ns: int, max_age_s=.05) -> tuple[np.ndarray, dict]:
    """Last available sensor roll/pitch only; no reference or heading alignment."""
    previous = None
    last_time = None
    with euler_csv.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) != 4:
                raise ValueError("ms25_euler.csv needs integer timestamp_us, roll, pitch, heading")
            stamp = int(row[0]) * 1000
            values = np.asarray(row[1:], dtype=np.float64)
            if not np.isfinite(values).all() or (last_time is not None and stamp <= last_time):
                raise ValueError("Euler readings must be finite and strictly increasing")
            last_time = stamp
            if stamp > first_scan_ns:
                break
            previous = (stamp, values)
    if previous is None or (first_scan_ns - previous[0]) * 1e-9 > max_age_s:
        raise ValueError("no sufficiently recent Euler reading at or before first scan")
    stamp, values = previous
    gravity = -9.81 * frames.body_up_from_rpy(*values[:2])
    return gravity, {"timestamp_ns": stamp, "roll_rad": float(values[0]),
                     "pitch_rad": float(values[1]), "age_s": (first_scan_ns - stamp) * 1e-9}


def initial_imu_states(times: np.ndarray, body_poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    strictly_increasing(times)
    if len(times) < 3 or body_poses.shape != (len(times), 4, 4):
        raise ValueError("need at least three matched body poses for velocity initialization")
    poses = np.array([frames.imu_pose_from_body_z_up(pose) for pose in body_poses])
    # Subtract integer epochs before converting to seconds. Differentiating the
    # IMU origin, rather than the axle, includes rotational lever-arm velocity.
    seconds = np.array([int(t) - int(times[0]) for t in times], dtype=np.float64) * 1e-9
    velocities = np.gradient(poses[:, :3, 3], seconds, axis=0, edge_order=2)
    return poses, velocities


def write_tum(path: Path, times: np.ndarray, poses: np.ndarray) -> None:
    quaternions = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for t, pose, q in zip(times, poses, quaternions, strict=True):
            seconds, nanos = divmod(int(t), 10**9)
            values = " ".join(f"{x:.17g}" for x in np.r_[pose[:3, 3], q])
            stream.write(f"{seconds}.{nanos:09d} {values}\n")


def recording_artifacts(path: Path) -> dict:
    if path.is_file():
        return {path.name: artifact(path)}
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError("IMU recording folder is empty")
    return {str(p.relative_to(path)).replace("\\", "/"): artifact(p) for p in files}


def run(lidar: Path, imu: Path, euler: Path, output: Path, *, imu_topic="/ms25/raw") -> dict:
    """Create one new experiment; never read ground truth or overwrite output."""
    if output.exists():
        raise FileExistsError(f"pilot output already exists: {output}")
    if imu.is_dir() and output.resolve().is_relative_to(imu.resolve()):
        raise ValueError("write pilot output outside the input recording")
    core = importlib.import_module("cloudanalyzer_core")
    native = Path(importlib.import_module("cloudanalyzer_core._core").__file__)
    config = settings()
    inputs = {"lidar": artifact(lidar), "imu": recording_artifacts(imu), "euler": artifact(euler),
              "controller": artifact(Path(__file__)), "frame_module": artifact(Path(frames.__file__)),
              "native_extension": artifact(native)}
    scans = list(core.BagReader(str(lidar)).scans())
    if not 4 <= len(scans) <= 4096:
        raise ValueError("pilot needs 4..4096 body-coordinate synchronized scans")
    if len({s.frame_id for s in scans}) != 1 or scans[0].frame_id not in {"velodyne", frames.FRAME_ID}:
        raise ValueError("unexpected NCLT scan frame; verify synchronized body-z-up provenance")
    times = np.array([s.stamp_ns for s in scans], dtype=np.int64)
    reader = core.BagReader(str(imu))
    raw = reader.imu_raw(topic=imu_topic)
    if set(raw["frame_ids"]) != {"nclt_ms25_native"}:
        raise ValueError("raw IMU vectors must use the nclt_ms25_native frame")
    rejected = gap_edges(times, raw["timestamps_ns"], config["max_gap_s"])
    selected = select_windows(len(times), rejected, minimum=config["min_states"], maximum=config["max_states"])
    if not selected:
        raise ValueError("no complete windows under the fixed sampling-gap policy")
    gravity, attitude = initial_gravity(euler, int(times[0]))
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "inputs": inputs, "settings": config, "imu_topic": imu_topic,
              "rayon_num_threads_environment": os.getenv("RAYON_NUM_THREADS"),
              "scan_frame_id": scans[0].frame_id, "interpreted_scan_frame": frames.FRAME_ID,
              "initial_attitude": attitude, "gravity_world_m_s2": gravity.tolist(),
              "scope": "Bounded offline diagnostic fusion, no ground-truth initialization or tuning; no continuous LIO, calibrated covariance or speed claim",
              "outputs": {}, "windows": []}
    (output / "protocol.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    odometry = core.LidarOdometry(**config["lidar"])
    body_poses = np.array([odometry.register(s.positions()) for s in scans])
    poses, velocities = initial_imu_states(times, body_poses)
    np.savez_compressed(output / "lidar.npz", timestamps_ns=times, body_poses=body_poses,
                        imu_poses=poses, imu_velocities=velocities)
    write_tum(output / "lidar.tum", times, body_poses)
    body_from_imu_inverse = np.linalg.inv(frames.BODY_UP_FROM_IMU)
    for number, (start, stop) in enumerate(selected):
        node_times = times[start:stop]
        biases = np.zeros((stop - start, 6))
        window = core.ImuWindow(node_times, poses[start:stop], velocities[start:stop], biases, gravity=gravity)
        for j, (a, b) in enumerate(zip(node_times[:-1], node_times[1:])):
            factor = reader.imu_factor(int(a), int(b), topic=imu_topic,
                                       gyro_noise_density=config["gyro_noise_density"],
                                       accel_noise_density=config["accel_noise_density"],
                                       max_gap_s=config["max_gap_s"])
            window.add_imu_factor(j, factor, gyro_bias_random_walk=config["gyro_bias_random_walk"],
                                  accel_bias_random_walk=config["accel_bias_random_walk"])
        window.add_state_prior(0, poses[start], velocities[start], biases[0], sigmas=config["state_sigmas"])
        for j in range(1, stop - start):
            window.add_pose_prior(j, poses[start + j], rotation_sigma=config["pose_rotation_sigma"],
                                  position_sigma=config["pose_position_sigma"])
        result = window.optimize(**config["optimizer"])
        fused_body = result["poses"] @ body_from_imu_inverse
        name = f"window-{number:03d}"
        np.savez_compressed(output / f"{name}.npz", timestamps_ns=node_times,
                            scan_indices=np.arange(start, stop, dtype=np.int64),
                            lidar_body_poses=body_poses[start:stop], fused_body_poses=fused_body,
                            fused_imu_poses=result["poses"], velocities=result["velocities"],
                            biases=result["biases"], cost_history=result["cost_history"])
        write_tum(output / f"{name}.tum", node_times, fused_body)
        report["windows"].append({"name": name, "scan_start": start, "scan_stop": stop,
                                  **{k: result[k] for k in ["initial_cost", "final_cost", "iterations",
                                      "accepted_steps", "rejected_steps", "converged", "termination"]},
                                  "max_abs_bias_correction": result["max_abs_bias_correction"].tolist()})
        print(f"{name}: scans {start}..{stop - 1}, {result['termination']}", flush=True)
    nodes = sum(b - a for a, b in selected)
    edges = nodes - len(selected)
    report["coverage"] = {"scans": len(scans), "points": sum(len(s) for s in scans),
                          "all_scan_edges": len(times) - 1, "rejected_gap_edges": rejected,
                          "windows": len(selected), "eligible_nodes": nodes, "eligible_edges": edges,
                          "unused_valid_edges": len(times) - 1 - len(rejected) - edges,
                          "excluded_scan_nodes": len(times) - nodes,
                          "eligible_duration_s": sum((int(times[b - 1]) - int(times[a])) * 1e-9 for a, b in selected),
                          "full_duration_s": (int(times[-1]) - int(times[0])) * 1e-9,
                          "nonconverged_windows": sum(not w["converged"] for w in report["windows"])}
    # Check inputs again so a changing recording/controller cannot get a valid receipt.
    after = {"lidar": artifact(lidar), "imu": recording_artifacts(imu), "euler": artifact(euler),
             "controller": artifact(Path(__file__)), "frame_module": artifact(Path(frames.__file__)),
             "native_extension": artifact(native)}
    if after != inputs:
        raise ValueError("pilot inputs changed during estimation")
    report["outputs"] = {p.name: artifact(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lidar", type=Path, required=True)
    parser.add_argument("--imu", type=Path, required=True)
    parser.add_argument("--euler", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imu-topic", default="/ms25/raw")
    args = parser.parse_args()
    report = run(args.lidar, args.imu, args.euler, args.out, imu_topic=args.imu_topic)
    print(json.dumps(report["coverage"]))


if __name__ == "__main__":
    main()
