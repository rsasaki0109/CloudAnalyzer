#!/usr/bin/env python3
"""Check nominal IMU deskew/ICP/map integration on complete real scans."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib import import_module
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

import cloudanalyzer_core as core
import cloudanalyzer_core._core as native
import audit_flatwall_raw_point_time as fields

GRAVITY = np.array([0.0, 0.0, -9.80665])
SETTINGS = dict(min_range=1.5, max_range=80.0, map_voxel=1.0, map_points=20, diagnostics=True)


def reference_motion(bag):
    """Standard ROS decoding and fixed-left nominal dynamics, independent of core."""
    types = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([bag]) as reader:
        connections = [c for c in reader.connections if c.topic == "/livox/imu"]
        if len(connections) != 1:
            raise ValueError("expected one IMU connection")
        messages = [types.deserialize_ros1(payload, c.msgtype) for c, _, payload in reader.messages(connections=connections)]
    times = np.array([int(m.header.stamp.sec) * 10**9 + int(m.header.stamp.nanosec) for m in messages], dtype=np.int64)
    gyro = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for m in messages])
    raw_accel = np.array([[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z] for m in messages])
    if len(times) != 3755 or not np.all(np.diff(times) > 0):
        raise ValueError("unexpected IMU samples")
    accel = raw_accel * 9.80665
    rotations = np.empty((len(times), 3, 3))
    positions = np.empty((len(times), 3))
    velocities = np.empty_like(positions)
    rotations[0], positions[0], velocities[0] = np.eye(3), np.zeros(3), np.zeros(3)
    for k, dt in enumerate(np.diff(times).astype(np.float64) * 1e-9):
        a = rotations[k] @ accel[k] + GRAVITY
        positions[k + 1] = positions[k] + velocities[k] * dt + 0.5 * a * dt**2
        velocities[k + 1] = velocities[k] + a * dt
        rotations[k + 1] = rotations[k] @ Rotation.from_rotvec(gyro[k] * dt).as_matrix()

    def states_at(stamps):
        if np.any(stamps < times[0]) or np.any(stamps > times[-1]):
            raise ValueError("oracle cannot extrapolate")
        k = np.minimum(np.searchsorted(times, stamps, side="right") - 1, len(times) - 2)
        dt = (stamps - times[k]).astype(np.float64) * 1e-9
        r = rotations[k] @ Rotation.from_rotvec(gyro[k] * dt[:, None]).as_matrix()
        a = np.einsum("nij,nj->ni", rotations[k], accel[k]) + GRAVITY
        p = positions[k] + velocities[k] * dt[:, None] + 0.5 * a * dt[:, None]**2
        return r, p

    return times, gyro, raw_accel, states_at


def insertion_points(compensated):
    """Independent ordered range filtering and first-per-half-voxel selection."""
    ranges = np.sqrt(np.sum(compensated * compensated, axis=1))
    near = compensated[(ranges >= SETTINGS["min_range"]) & (ranges <= SETTINGS["max_range"])]
    keys = np.floor(near / (0.5 * SETTINGS["map_voxel"])).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return near[np.sort(first)]


def audit(bag, output, component):
    bag, output, component = bag.resolve(), output.resolve(), component.resolve()
    if os.environ.get("RAYON_NUM_THREADS") != "4":
        raise ValueError("use a fresh process with RAYON_NUM_THREADS=4")
    if output.exists() or component == output or component in output.parents:
        raise ValueError("use a fresh output outside the frozen component audit")

    def inputs():
        return {"recording": fields.artifact(bag), "controller": fields.artifact(Path(__file__)),
                "helpers": fields.artifact(Path(fields.__file__)), "native": fields.artifact(Path(native.__file__)),
                "component_protocol": fields.artifact(component / "protocol.json"),
                "component_receipt": fields.artifact(component / "receipt.json"),
                "numpy_version": version("numpy"), "scipy_version": version("scipy"),
                "rosbags_version": version("rosbags"),
                "rotation_backend": fields.artifact(Path(import_module("scipy.spatial.transform._rotation").__file__)),
                "independent_reader": fields.artifact(Path(inspect.getfile(AnyReader))),
                "standard_types": fields.artifact(Path(import_module("rosbags.typesys.stores.ros1_noetic").__file__))}

    frozen = inputs()
    component_result = json.loads((component / "receipt.json").read_text(encoding="utf-8"))
    if frozen["recording"]["sha256"] != fields.BAG_SHA256:
        raise ValueError("expected original flatwall_01.bag from Zenodo record 7641866")
    if component_result["inputs"]["native"] != frozen["native"] or not component_result["native_results"]["serial_parallel_bits_match"]:
        raise ValueError("first complete the frozen deskew component audit with this native binary")
    protocol = {"schema": "cloudanalyzer.flatwall_imu_odometry.v1", "inputs": frozen,
                "settings": SETTINGS, "parallel_compensation": True, "rayon_threads_requested": 4,
                "declared_state": "first IMU header, identity pose, zero world velocity/bias, gravity [0,0,-9.80665], acceleration scale9.80665, identity IMU-from-LiDAR",
                "reference_time": "maximum integer point timestamp; reject the entire first unbracketed scan; accept all other scans in source order",
                "coordinate_frame": "first accepted reference LiDAR frame; world_from_first_lidar aligns it to the nominal IMU world",
                "checks": {"point_oracle_atol": 2e-8, "relative_prediction_atol": 1e-8, "placed_map_atol": 2e-8,
                           "insertion_selection": "Independent range filter and first-per-half-voxel selection on native-compensated coordinates, after their complete independent coordinate check; use final ICP pose for placement."},
                "scope": "Integration/numerical audit on real measurements with arbitrary declared initial state; no physical initialization/calibration fit, reference poses, accuracy, runtime or GPU result."}
    output.mkdir(parents=True, exist_ok=False)
    fields.save(output / "protocol.json", protocol)
    times, gyro, accel, states_at = reference_motion(bag)
    reader = core.BagReader(str(bag))
    raw = reader.imu_raw("/livox/imu")
    np.testing.assert_array_equal(raw["timestamps_ns"], times)
    np.testing.assert_array_equal(raw["angular_velocity_rad_s"].view(np.uint64), gyro.view(np.uint64))
    np.testing.assert_array_equal(raw["linear_acceleration_m_s2"].view(np.uint64), accel.view(np.uint64))
    if fields.fields_digest(raw) != component_result["common_results"]["raw_imu_sha256"]:
        raise ValueError("raw IMU changed")
    series = reader.imu_series("/livox/imu", acceleration_scale=9.80665)
    motion = series.trajectory(int(times[0]), int(times[-1]), np.eye(4), np.zeros(3), gravity=tuple(GRAVITY), bias=(0.0,) * 6)
    odometry = core.ImuLidarOdometry(motion, np.eye(4), parallel=True, **SETTINGS)
    rows, poses, predictions, map_chunks, references, skipped = [], [], [], [], [], []
    previous_r = previous_p = None
    errors = {"deskew": 0.0, "relative_prediction": 0.0, "placed_map": 0.0, "world_alignment": 0.0}
    deskew_digest = hashlib.sha256()
    for index, scan in enumerate(reader.scans("/livox/points", raw_time_field="t")):
        stamps, xyz = scan.point_times_ns(unit="ns", origin="header"), scan.positions()
        reference = int(stamps.max())
        if stamps.min() < times[0] or reference > times[-1]:
            try:
                odometry.register(xyz, stamps, reference)
            except ValueError as error:
                if "cover point timestamp at index 0" not in str(error):
                    raise
            else:
                raise ValueError("uncovered scan was accepted")
            assert odometry.poses().shape == (0, 4, 4) and odometry.placed().shape == (0, 3)
            assert odometry.reference_timestamps_ns().size == 0
            assert odometry.world_from_first_lidar is odometry.last_report() is None
            skipped.append(index)
            continue
        r, p = states_at(stamps)
        ref_r, ref_p = states_at(np.array([reference], dtype=np.int64))
        ref_r, ref_p = ref_r[0], ref_p[0]
        expected = (np.einsum("nij,nj->ni", r, xyz) + p - ref_p) @ ref_r
        compensated = motion.deskew(xyz, stamps, reference, imu_from_lidar=np.eye(4), parallel=True)
        np.testing.assert_allclose(compensated, expected, rtol=0, atol=2e-8)
        errors["deskew"] = max(errors["deskew"], float(np.max(np.abs(compensated - expected))))
        deskew_digest.update(index.to_bytes(4, "little") + compensated.astype("<f8").tobytes())
        original = fields.fields_digest({"points": xyz, "times": stamps})
        pose = odometry.register(xyz, stamps, reference)
        assert fields.fields_digest({"points": xyz, "times": stamps}) == original
        assert np.isfinite(pose).all()
        np.testing.assert_allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-6, rtol=0)
        np.testing.assert_array_equal(pose[3], [0, 0, 0, 1])
        prediction = odometry.last_prediction()
        if previous_r is None:
            np.testing.assert_array_equal(pose, np.eye(4))
            assert prediction is None
            anchor = np.eye(4)
            anchor[:3, :3], anchor[:3, 3] = ref_r, ref_p
            actual = odometry.world_from_first_lidar
            np.testing.assert_allclose(actual, anchor, rtol=0, atol=1e-8)
            errors["world_alignment"] = float(np.max(np.abs(actual - anchor)))
        else:
            expected_prediction = np.eye(4)
            expected_prediction[:3, :3] = previous_r.T @ ref_r
            expected_prediction[:3, 3] = previous_r.T @ (ref_p - previous_p)
            np.testing.assert_allclose(prediction, expected_prediction, rtol=0, atol=1e-8)
            errors["relative_prediction"] = max(errors["relative_prediction"], float(np.max(np.abs(prediction - expected_prediction))))
        source = insertion_points(compensated)
        expected_map = source @ pose[:3, :3].T + pose[:3, 3]
        placed = odometry.placed()
        np.testing.assert_allclose(placed, expected_map, rtol=0, atol=2e-8)
        if len(placed):
            errors["placed_map"] = max(errors["placed_map"], float(np.max(np.abs(placed - expected_map))))
        report = odometry.last_report()
        assert report["input_points"] == len(xyz)
        rows.append({"scan_index": index, "reference_ns": reference, "input_points": len(xyz),
                     "placed_points": len(placed), "registration": report})
        references.append(reference)
        poses.append(pose)
        predictions.append(np.eye(4) if prediction is None else prediction)
        map_chunks.append(placed)
        previous_r, previous_p = ref_r, ref_p
        if index % 50 == 0:
            print(f"registered {index}/185 covered scans", flush=True)
    assert len(rows) == 185 and sum(row["input_points"] for row in rows) == 4_440_000 and skipped == [0]
    assert deskew_digest.hexdigest() == component_result["native_results"]["deskew_sha256"]
    np.testing.assert_array_equal(odometry.reference_timestamps_ns(), references)
    np.testing.assert_array_equal(odometry.poses(), poses)
    assert inputs() == frozen
    fields.save(output / "per-scan.json", rows)
    np.savez_compressed(output / "trajectory.npz", timestamps_ns=odometry.reference_timestamps_ns(),
                        poses=odometry.poses(), relative_predictions=np.asarray(predictions),
                        world_from_first_lidar=odometry.world_from_first_lidar)
    np.save(output / "map_points.npy", np.concatenate(map_chunks), allow_pickle=False)
    result = {"registered_scans": len(rows), "registered_points": sum(row["input_points"] for row in rows),
              "skipped_scan_indices": skipped, "placed_map_points": sum(row["placed_points"] for row in rows),
              "icp_iterations": sum(row["registration"]["iterations"] for row in rows),
              "termination_counts": dict(Counter(row["registration"]["termination"] for row in rows)),
              "max_abs_errors": errors, "compensated_points_sha256": deskew_digest.hexdigest(),
              "first_rejection_atomic": True, "reference_timestamps_exact": True, "caller_arrays_unchanged": True}
    receipt = {**protocol, "protocol": fields.artifact(output / "protocol.json"), "results": result,
               "outputs": {name: fields.artifact(output / name) for name in ["per-scan.json", "trajectory.npz", "map_points.npy"]},
               "no_reference_poses_used": True}
    fields.save(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["bag", "out", "component-audit"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.bag, args.out, args.component_audit)
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
