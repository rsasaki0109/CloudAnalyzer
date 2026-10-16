#!/usr/bin/env python3
"""Fixed-prefix metric LiDAR/IMU startup audit; no reference trajectory or timing."""
from __future__ import annotations

import argparse
from collections import Counter
from importlib import import_module
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import cloudanalyzer_core as core
import cloudanalyzer_core._core as native
import audit_flatwall_raw_point_time as fields

SCANS = list(range(1, 65))
LIDAR = dict(min_range=1.5, max_range=80.0, map_voxel=1.0, map_points=20, diagnostics=True)
PARAMS = dict(gyro_bias=(0.0,) * 3, accel_bias=(0.0,) * 3, gravity_magnitude=9.80665,
              max_gravity_error=1.0, max_rotation_rms=0.05, max_velocity_rms=0.2,
              max_gyro_bias_change=0.2, max_iterations=15, max_gap_s=0.05, min_singular_ratio=1e-6)


def reference(bag, stamps, poses):
    types = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([bag]) as reader:
        connections = [c for c in reader.connections if c.topic == "/livox/imu"]
        if len(connections) != 1:
            raise ValueError("expected one IMU connection")
        messages = [types.deserialize_ros1(payload, c.msgtype) for c, _, payload in reader.messages(connections=connections)]
    times = np.array([int(m.header.stamp.sec) * 10**9 + int(m.header.stamp.nanosec) for m in messages], dtype=np.int64)
    gyro = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for m in messages])
    accel = np.array([[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z] for m in messages]) * 9.80665
    segments = []
    for start, end in zip(stamps[:-1], stamps[1:]):
        if start < times[0] or end > times[-1]:
            raise ValueError("uncovered startup reference")
        bounds = np.r_[start, times[(times > start) & (times < end)], end]
        indices = np.searchsorted(times, bounds[:-1], side="right") - 1
        segments.append((np.diff(bounds).astype(np.float64) * 1e-9, gyro[indices], accel[indices]))

    def integrated(bias, positions=False):
        result = []
        for durations, angular, force in segments:
            r, v, p = np.eye(3), np.zeros(3), np.zeros(3)
            for dt, w, a in zip(durations, angular, force):
                if positions:
                    world_a = r @ (a - PARAMS["accel_bias"])
                    p += v * dt + 0.5 * world_a * dt**2
                    v += world_a * dt
                r = r @ Rotation.from_rotvec((w - bias) * dt).as_matrix()
            result.append((r, v, p))
        return result

    def residual(bias):
        return np.concatenate([Rotation.from_matrix(delta[0].T @ poses[i, :3, :3].T @ poses[i + 1, :3, :3]).as_rotvec()
                               for i, delta in enumerate(integrated(bias))])

    fit = least_squares(residual, PARAMS["gyro_bias"], xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=100)
    deltas = integrated(fit.x, positions=True)
    dt = np.array([int(b) - int(a) for a, b in zip(stamps[:-1], stamps[1:])], dtype=np.float64) * 1e-9
    world_v = np.array([poses[i, :3, :3] @ delta[1] for i, delta in enumerate(deltas)])
    world_p = np.array([poses[i, :3, :3] @ delta[2] for i, delta in enumerate(deltas)])
    # Acceleration bias is fixed; the real recording uses an identity extrinsic.
    position = poses[:, :3, 3] - poses[0, :3, 3]
    average = (np.diff(position, axis=0) - world_p) / dt[:, None]
    y = np.diff(average, axis=0) - world_v[:-1]
    alpha = 0.5 * (dt[:-1] + dt[1:])
    gravity = np.sum(alpha[:, None] * y, axis=0) / np.sum(alpha**2)
    velocity = average - 0.5 * dt[:, None] * gravity
    velocity = np.vstack([velocity, velocity[-1] + gravity * dt[-1] + world_v[-1]])
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    diagnostics = {"gravity_error": abs(float(np.linalg.norm(gravity)) - PARAMS["gravity_magnitude"]),
                   "rotation_rms": float(np.sqrt(np.mean(np.sum(fit.fun.reshape(-1, 3)**2, axis=1)))),
                   "velocity_rms": float(np.sqrt(np.mean(np.sum((y - alpha[:, None] * gravity)**2, axis=1)))),
                   "gyro_singular_ratio": float(singular[-1] / singular[0]),
                   "max_gyro_bias_change": float(np.max(np.abs(fit.x - PARAMS["gyro_bias"]))),
                   "optimizer_success": bool(fit.success)}
    failed = [key for key in ["gravity_error", "rotation_rms", "velocity_rms", "gyro_bias_change"]
              if diagnostics["max_gyro_bias_change" if key == "gyro_bias_change" else key] > PARAMS["max_" + key]]
    if diagnostics["gyro_singular_ratio"] < PARAMS["min_singular_ratio"]:
        failed.append("gyro_rank")
    if not fit.success:
        failed.append("optimizer")
    return {"accepted": not failed, "failed_limits": failed, "gravity": gravity.tolist(),
            "bias": np.r_[fit.x, PARAMS["accel_bias"]].tolist(), "velocities": velocity.tolist(), "diagnostics": diagnostics}


def audit(bag, output, *, baseline=False, comparison=None):
    bag, output = bag.resolve(), output.resolve()
    comparison = None if comparison is None else comparison.resolve()
    if os.environ.get("RAYON_NUM_THREADS") != "4":
        raise ValueError("use a fresh process with RAYON_NUM_THREADS=4")
    if output.exists() or (comparison is not None and (comparison == output or comparison in output.parents)):
        raise ValueError("use a fresh output outside prior artifacts")

    def inputs():
        result = {"recording": fields.artifact(bag), "controller": fields.artifact(Path(__file__)),
                  "helpers": fields.artifact(Path(fields.__file__)), "native": fields.artifact(Path(native.__file__)),
                  "numpy_version": version("numpy"), "scipy_version": version("scipy"), "rosbags_version": version("rosbags"),
                  "rotation_backend": fields.artifact(Path(import_module("scipy.spatial.transform._rotation").__file__)),
                  "optimizer_source": fields.artifact(Path(inspect.getfile(least_squares))),
                  "independent_reader": fields.artifact(Path(inspect.getfile(AnyReader))),
                  "standard_types": fields.artifact(Path(import_module("rosbags.typesys.stores.ros1_noetic").__file__))}
        if comparison is not None:
            result["comparison_protocol"] = fields.artifact(comparison / "protocol.json")
            result["comparison_receipt"] = fields.artifact(comparison / "receipt.json")
            result["startup_poses"] = fields.artifact(comparison / "startup-poses.npz")
        return result

    frozen = inputs()
    if frozen["recording"]["sha256"] != fields.BAG_SHA256:
        raise ValueError("expected original flatwall recording")
    prior = None
    if comparison is not None:
        prior = json.loads((comparison / "receipt.json").read_text())
        assert prior["baseline"]
        assert prior["outputs"]["startup-poses.npz"] == frozen["startup_poses"]
        for key in prior["inputs"].keys() - {"native"}:
            assert prior["inputs"][key] == frozen[key], key
    protocol = {"schema": "cloudanalyzer.flatwall_imu_initialization.v1", "inputs": frozen,
                "baseline": baseline, "scan_indices": SCANS, "lidar": LIDAR, "parameters": PARAMS,
                "reference_time": "maximum checked integer point timestamp", "acceleration_scale": 9.80665,
                "imu_from_lidar": np.eye(4).tolist(), "startup_pose_source": "Pure LiDAR ICP, deskew=False, no IMU prediction; immutable baseline poses reused by new native.",
                "oracle": "Standard ROS IMU decode, SciPy exact held-gyro bias integration/finite-difference least squares and NumPy displacement-triple gravity/velocity equations.",
                "numerical_tolerances": {"gyro_bias": 2e-5, "gravity": 2e-5, "velocity": 2e-5},
                "scope": "Fixed-prefix startup consistency only. No stationary assumption, reference poses, physical calibration/accuracy, runtime or GPU result; acceptance limits are not relaxed."}
    output.mkdir(parents=True, exist_ok=False)
    fields.save(output / "protocol.json", protocol)
    reader = core.BagReader(str(bag))
    if baseline:
        odometry = core.LidarOdometry(deskew=False, **LIDAR)
        stamps, poses, reports = [], [], []
        for index, scan in enumerate(reader.scans("/livox/points", raw_time_field="t")):
            if index not in SCANS:
                if index > SCANS[-1]:
                    break
                continue
            stamps.append(int(scan.point_times_ns(unit="ns", origin="header").max()))
            poses.append(odometry.register(scan.positions()))
            reports.append(odometry.last_report())
        stamps, poses = np.array(stamps, dtype=np.int64), np.asarray(poses)
        assert len(stamps) == 64 and np.all(np.diff(stamps) > 0)
        np.savez_compressed(output / "startup-poses.npz", timestamps_ns=stamps, lidar_poses=poses)
        fields.save(output / "lidar-diagnostics.json", reports)
        oracle = reference(bag, stamps, poses)
        fields.save(output / "independent.json", oracle)
        common = {"scans": len(stamps), "first_reference_ns": int(stamps[0]), "last_reference_ns": int(stamps[-1]),
                  "termination_counts": dict(Counter(r["termination"] for r in reports)),
                  "icp_iterations": sum(r["iterations"] for r in reports), "oracle": oracle}
        native_result = None
    else:
        with np.load(comparison / "startup-poses.npz", allow_pickle=False) as arrays:
            stamps, poses = arrays["timestamps_ns"].copy(), arrays["lidar_poses"].copy()
        common = prior["common_results"]
        oracle = common["oracle"]
        series = reader.imu_series("/livox/imu", acceleration_scale=9.80665)
        original = fields.fields_digest({"times": stamps, "poses": poses})
        results, errors = [], []
        for parallel in [False, True]:
            try:
                result = series.initialize_lidar(stamps, poses, imu_from_lidar=np.eye(4), parallel=parallel, **PARAMS)
            except ValueError as error:
                errors.append(str(error))
            else:
                results.append(result)
        assert fields.fields_digest({"times": stamps, "poses": poses}) == original
        if oracle["accepted"]:
            assert len(results) == 2 and not errors, errors
            a, b = results
            for key in ["timestamps_ns", "imu_poses", "velocities", "gravity", "bias"]:
                np.testing.assert_array_equal(a[key].view(np.uint64), b[key].view(np.uint64))
            np.testing.assert_array_equal(a["timestamps_ns"], stamps)
            np.testing.assert_allclose(a["imu_poses"], poses, rtol=0, atol=1e-12)
            for key, tolerance in [("gravity", 2e-5), ("velocities", 2e-5), ("bias", 2e-5)]:
                np.testing.assert_allclose(a[key], oracle[key], rtol=0, atol=tolerance)
            native_result = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in a.items()}
            native_result["accepted"] = True
            native_result["serial_parallel_bits_match"] = True
        else:
            assert not results and len(errors) == 2 and errors[0] == errors[1], errors
            native_result = {"accepted": False, "error": errors[0], "serial_parallel_rejection_matches": True}
    assert inputs() == frozen
    receipt = {**protocol, "protocol": fields.artifact(output / "protocol.json"), "common_results": common,
               "native_result": native_result, "outputs": {p.name: fields.artifact(p) for p in sorted(output.iterdir()) if p.is_file()},
               "no_reference_poses_used": True}
    fields.save(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["bag", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--comparison", type=Path)
    args = parser.parse_args()
    result = audit(args.bag, args.out, baseline=args.baseline, comparison=args.comparison)
    print(json.dumps({"oracle": {k: v for k, v in result["common_results"]["oracle"].items() if k != "velocities"},
                      "native": result["native_result"]}, indent=2))


if __name__ == "__main__":
    main()
