#!/usr/bin/env python3
"""Declared-state deskew component audit on original recordings; no accuracy score."""
from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

import cloudanalyzer_core as core
import cloudanalyzer_core._core as native
import audit_flatwall_raw_point_time as fields

GRAVITY = np.array([0., 0., -9.80665])
CHECK_SCANS = [1, 2, 9, 17, 49, 93, 139, 185]


def audit(bag, output, *, baseline=False, comparison=None, controls=()):
    if os.environ.get("RAYON_NUM_THREADS") != "4":
        raise ValueError("run a fresh process with RAYON_NUM_THREADS=4")
    output = output.resolve()
    comparison = comparison.resolve() if comparison is not None else None
    if output.exists() or (comparison is not None and comparison in output.parents):
        raise ValueError("use a fresh output outside the frozen baseline")
    recordings = [("flatwall", bag.resolve(), "/livox/points")] + [(label, path.resolve(), topic) for label, path, topic in controls]
    labels = [label for label, path, topic in recordings]
    if len(set(labels)) != len(labels) or any(not re.fullmatch(r"[a-z0-9-]+", label) for label in labels):
        raise ValueError("unique lowercase recording labels required")

    def inputs():
        result = {"recordings": {label: {**fields.artifact(path), "topic": topic} for label, path, topic in recordings},
                  "controller": fields.artifact(Path(__file__)), "field_audit_helpers": fields.artifact(Path(fields.__file__)),
                  "native": fields.artifact(Path(native.__file__)), "numpy_version": version("numpy"),
                  "scipy_version": version("scipy"), "rosbags_version": version("rosbags"), "rayon_threads_requested": 4,
                  "rotation_backend": fields.artifact(Path(import_module("scipy.spatial.transform._rotation").__file__)),
                  "independent_reader": fields.artifact(Path(inspect.getfile(AnyReader))),
                  "standard_type_sources": [fields.artifact(Path(import_module("rosbags.typesys.stores." + name).__file__)) for name in ["ros1_noetic", "ros2_humble"]]}
        if comparison is not None:
            result["comparison_protocol"] = fields.artifact(comparison / "protocol.json")
            result["comparison_receipt"] = fields.artifact(comparison / "receipt.json")
        return result

    frozen = inputs()
    if frozen["recordings"]["flatwall"]["sha256"] != fields.BAG_SHA256:
        raise ValueError("expected the original flatwall_01.bag from Zenodo record 7641866")
    prior = None
    if comparison is not None:
        prior = json.loads((comparison / "receipt.json").read_text(encoding="utf-8"))
        if not prior["baseline"] or prior["protocol"] != frozen["comparison_protocol"]:
            raise ValueError("invalid or changed baseline protocol")
        for key in frozen.keys() - {"native", "comparison_protocol", "comparison_receipt"}:
            if prior["inputs"][key] != frozen[key]:
                raise ValueError(f"baseline input changed: {key}")
    protocol = {"schema": "cloudanalyzer.flatwall_imu_deskew.v1", "inputs": frozen, "baseline": baseline,
                "declared_policy": {"point_time": "UINT32 t, ns offsets from integer PointCloud2 Header", "initial_state": "first IMU header, identity IMU-to-world pose, zero world velocity", "gravity_world": GRAVITY.tolist(), "bias": [0.] * 6, "acceleration_scale": 9.80665, "imu_from_lidar": np.eye(4).tolist(), "reference": "maximum point timestamp", "max_gap_s": .05},
                "independent_model": "standard ROS1 IMU decode, SciPy held-gyro exponentials and NumPy fixed-left acceleration recurrence; full rigid point transform",
                "comparison": {"all_covered_points_tolerance": 2e-8, "state_rotation_tolerance": 1e-10, "state_velocity_position_tolerance": 1e-8, "native_preintegration_scan_indices": CHECK_SCANS, "point_indices": "0, N//2, N-1 and reference", "parallel": "four-thread indexed map must match serial output bits"},
                "coverage": "reject the entire first scan rather than extrapolate 632 uncovered points; retain every other point in source order",
                "scope": "Real measurements with arbitrary declared numerical initial state/gravity. No physical initialization or calibration fitted, reference archive access, LiDAR fusion, trajectory accuracy or runtime claim."}
    output.mkdir(parents=True, exist_ok=False)
    fields.save(output / "protocol.json", protocol)
    legacy = {}
    for label, path, topic in recordings:
        timing = label == "flatwall"
        reader = core.BagReader(str(path))
        ordinary = iter(reader.scans(topic))
        retained = iter(reader.scans(topic, raw_time_field="t")) if timing else None
        digest, time_digest, count, points = hashlib.sha256(), hashlib.sha256(), 0, 0
        with AnyReader([path]) as independent:
            connections = [c for c in independent.connections if c.topic == topic]
            if len(connections) != 1 or connections[0].msgtype != "sensor_msgs/msg/PointCloud2":
                raise ValueError("expected one standard PointCloud2 connection")
            types = get_typestore(Stores.ROS2_HUMBLE if independent.is2 else Stores.ROS1_NOETIC)
            for connection, recorded, payload in independent.messages(connections=connections):
                message = types.deserialize_cdr(payload, connection.msgtype) if independent.is2 else types.deserialize_ros1(payload, connection.msgtype)
                expected, raw_values, layout = fields.reference_fields(message, timing=timing)
                scan = next(ordinary)
                actual = fields.fields_digest(fields.scan_fields(scan))
                if actual != fields.fields_digest(expected) or scan.raw_time() is not None:
                    raise ValueError("legacy fields differ from independent decoding")
                digest.update(bytes.fromhex(actual))
                count += 1
                points += len(scan)
                if timing:
                    extra = next(retained)
                    if fields.fields_digest(fields.scan_fields(extra)) != actual or extra.raw_time_datatype != 6:
                        raise ValueError("retained fields changed")
                    np.testing.assert_array_equal(extra.raw_time().view(np.uint64), raw_values.view(np.uint64))
                    point_ns = extra.point_times_ns(unit="ns", origin="header")
                    expected_ns = np.fromiter((expected["stamp_ns"] + int(v) for v in raw_values), dtype=np.int64, count=len(raw_values))
                    np.testing.assert_array_equal(point_ns, expected_ns)
                    time_digest.update(point_ns.astype("<i8").tobytes())
        if next(ordinary, None) is not None or (retained is not None and next(retained, None) is not None):
            raise ValueError("extra native scans")
        legacy[label] = {"scans": count, "finite_points": points, "legacy_fields_sha256": digest.hexdigest(), "point_times_ns_sha256": time_digest.hexdigest() if timing else None}

    reader = core.BagReader(str(bag))
    raw = reader.imu_raw("/livox/imu")
    series = reader.imu_series("/livox/imu", acceleration_scale=9.80665)
    messages = []
    with AnyReader([bag]) as independent:
        connections = [c for c in independent.connections if c.topic == "/livox/imu"]
        if len(connections) != 1:
            raise ValueError("expected one IMU topic")
        types = get_typestore(Stores.ROS1_NOETIC)
        messages = [types.deserialize_ros1(payload, c.msgtype) for c, recorded, payload in independent.messages(connections=connections)]
    times = np.array([int(m.header.stamp.sec) * 10**9 + int(m.header.stamp.nanosec) for m in messages], dtype=np.int64)
    gyro = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for m in messages])
    acceleration = np.array([[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z] for m in messages])
    expected_raw = {"angular_velocity_rad_s": gyro, "linear_acceleration_m_s2": acceleration,
                    "orientation_xyzw": np.array([[m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w] for m in messages])}
    for name, attr in [("orientation_covariance", "orientation_covariance"), ("angular_velocity_covariance", "angular_velocity_covariance"), ("linear_acceleration_covariance", "linear_acceleration_covariance")]:
        expected_raw[name] = np.array([getattr(m, attr) for m in messages]).reshape(-1, 3, 3)
    np.testing.assert_array_equal(times, raw["timestamps_ns"])
    np.testing.assert_array_equal(times, series.timestamps_ns())
    for name, array in expected_raw.items():
        np.testing.assert_array_equal(array.view(np.uint64), raw[name].view(np.uint64))
    if raw["frame_ids"] != [m.header.frame_id for m in messages] or len(set(raw["frame_ids"])) != 1 or len(times) != 3755 or not np.all(np.diff(times) > 0) or np.diff(times).max() > 50_000_000:
        raise ValueError("unexpected IMU frames/order/coverage")
    acceleration *= 9.80665
    rotations = np.empty((len(times), 3, 3))
    velocities = np.empty((len(times), 3))
    positions = np.empty_like(velocities)
    rotations[0], velocities[0], positions[0] = np.eye(3), np.zeros(3), np.zeros(3)
    for k, dt in enumerate(np.diff(times).astype(np.float64) * 1e-9):
        a = rotations[k] @ acceleration[k] + GRAVITY
        positions[k + 1] = positions[k] + velocities[k] * dt + .5 * a * dt**2
        velocities[k + 1] = velocities[k] + a * dt
        rotations[k + 1] = rotations[k] @ Rotation.from_rotvec(gyro[k] * dt).as_matrix()

    def states_at(stamps):
        k = np.minimum(np.searchsorted(times, stamps, side="right") - 1, len(times) - 2)
        dt = (stamps - times[k]).astype(np.float64) * 1e-9
        r = rotations[k] @ Rotation.from_rotvec(gyro[k] * dt[:, None]).as_matrix()
        a = np.einsum("nij,nj->ni", rotations[k], acceleration[k]) + GRAVITY
        p = positions[k] + velocities[k] * dt[:, None] + .5 * a * dt[:, None]**2
        return r, p

    trajectory = None if baseline else series.trajectory(int(times[0]), int(times[-1]), np.eye(4), np.zeros(3), gravity=tuple(GRAVITY), bias=(0.,) * 6)
    errors = {"state_rotation": 0., "state_velocity": 0., "state_position": 0., "point_coordinates": 0.}
    if trajectory is not None:
        np.testing.assert_array_equal(trajectory.timestamps_ns(), times)
        for k, stamp in enumerate(times):
            state = trajectory.state(int(stamp))
            errors["state_rotation"] = max(errors["state_rotation"], float(np.max(np.abs(state["pose"][:3, :3] - rotations[k]))))
            errors["state_velocity"] = max(errors["state_velocity"], float(np.max(np.abs(state["velocity"] - velocities[k]))))
            errors["state_position"] = max(errors["state_position"], float(np.max(np.abs(state["pose"][:3, 3] - positions[k]))))
    oracle_digest, native_digest, control_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    scans = points = decreasing = 0
    skipped = []
    for scan_index, scan in enumerate(reader.scans("/livox/points", raw_time_field="t")):
        stamps = scan.point_times_ns(unit="ns", origin="header")
        xyz = scan.positions()
        reference = int(stamps.max())
        if stamps.min() < times[0] or stamps.max() > times[-1]:
            skipped.append(scan_index)
            if trajectory is not None:
                try:
                    trajectory.deskew(xyz, stamps, reference, imu_from_lidar=np.eye(4))
                except ValueError as error:
                    if "cover point timestamp at index 0" not in str(error):
                        raise
                else:
                    raise ValueError("unbracketed first scan was extrapolated")
            continue
        r, p = states_at(stamps)
        ref_r, ref_p = states_at(np.array([reference], dtype=np.int64))
        expected = (np.einsum("nij,nj->ni", r, xyz) + p - ref_p[0]) @ ref_r[0]
        oracle_digest.update(scan_index.to_bytes(4, "little") + expected.astype("<f8").tobytes())
        if scan_index in CHECK_SCANS:
            for stamp in [int(stamps[0]), int(stamps[len(stamps) // 2]), int(stamps[-1]), reference]:
                m = series.preintegrate(int(times[0]), stamp, gyro_noise_density=.02, accel_noise_density=.15, bias=(0.,) * 6)
                dt = (stamp - int(times[0])) * 1e-9
                q, x = states_at(np.array([stamp], dtype=np.int64))
                np.testing.assert_allclose(m["delta_rotation"], q[0], rtol=0, atol=1e-10)
                np.testing.assert_allclose(m["delta_position"] + .5 * GRAVITY * dt**2, x[0], rtol=0, atol=1e-8)
                control_digest.update(bytes.fromhex(fields.fields_digest(m)))
        if trajectory is not None:
            before_xyz = xyz.tobytes()
            serial = trajectory.deskew(xyz, stamps, reference, imu_from_lidar=np.eye(4))
            parallel = trajectory.deskew(xyz, stamps, reference, imu_from_lidar=np.eye(4), parallel=True)
            np.testing.assert_array_equal(serial.view(np.uint64), parallel.view(np.uint64))
            np.testing.assert_allclose(serial, expected, rtol=0, atol=2e-8)
            errors["point_coordinates"] = max(errors["point_coordinates"], float(np.max(np.abs(serial - expected))))
            if xyz.tobytes() != before_xyz:
                raise ValueError("deskew changed caller points")
            native_digest.update(scan_index.to_bytes(4, "little") + serial.astype("<f8").tobytes())
        scans += 1
        points += len(xyz)
        decreasing += int(np.sum(np.diff(stamps) < 0))
    common = {"legacy_inputs": legacy, "raw_imu_sha256": fields.fields_digest(raw), "imu_samples": len(times), "covered_scans": scans, "covered_points": points, "skipped_scan_indices": skipped, "decreasing_point_time_steps": decreasing, "independent_deskew_sha256": oracle_digest.hexdigest(), "legacy_preintegration_sha256": control_digest.hexdigest()}
    if scans != 185 or points != 4_440_000 or skipped != [0] or decreasing != 11 or inputs() != frozen:
        raise ValueError("coverage/provenance changed")
    if prior is not None and common != prior["common_results"]:
        raise ValueError("legacy inputs, preintegration or independent reference changed")
    if errors["state_rotation"] > 1e-10 or max(errors["state_velocity"], errors["state_position"]) > 1e-8:
        raise ValueError("native trajectory differs from independent states")
    receipt = {**protocol, "protocol": fields.artifact(output / "protocol.json"), "common_results": common,
               "native_results": None if baseline else {"max_abs_errors": errors, "deskew_sha256": native_digest.hexdigest(), "serial_parallel_bits_match": True, "first_unbracketed_scan_rejected": True, "caller_points_unchanged": True},
               "legacy_fields_match_old_native": comparison is not None, "ground_truth_archive_read": False}
    fields.save(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--comparison", type=Path)
    parser.add_argument("--control", nargs=3, action="append", default=[], metavar=("LABEL", "BAG", "TOPIC"))
    args = parser.parse_args()
    result = audit(args.bag, args.out, baseline=args.baseline, comparison=args.comparison, controls=[(label, Path(path), topic) for label, path, topic in args.control])
    print(json.dumps({"common": result["common_results"], "native": result["native_results"]}, indent=2))


if __name__ == "__main__":
    main()
