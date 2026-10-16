#!/usr/bin/env python3
"""Gyro-only real-recording component and startup consistency audit; no timing."""

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
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import cloudanalyzer_core as core
import cloudanalyzer_core._core as native
import audit_flatwall_imu_initialization as startup
import audit_flatwall_raw_point_time as fields

CASES = [("legacy", "omitted", None), ("explicit-0.1875", "explicit-0.1875", 0.1875)]
LIDAR = dict(
    min_range=1.5,
    max_range=80.0,
    map_voxel=1.0,
    map_points=20,
    deskew=False,
    diagnostics=True,
)


def audit(bag, output, previous):
    bag, output, previous = bag.resolve(), output.resolve(), previous.resolve()
    if output.exists() or output.is_relative_to(previous):
        raise ValueError("use a fresh output outside previous evidence")
    if (
        os.environ.get("RAYON_NUM_THREADS") != "4"
        or os.environ.get("OPENBLAS_NUM_THREADS") != "1"
    ):
        raise ValueError(
            "use fresh RAYON_NUM_THREADS=4 and OPENBLAS_NUM_THREADS=1 processes"
        )

    def inputs():
        return {
            "recording": fields.artifact(bag),
            "controller": fields.artifact(Path(__file__)),
            "native": fields.artifact(Path(native.__file__)),
            "helpers": fields.artifact(Path(fields.__file__)),
            "startup_controller": fields.artifact(Path(startup.__file__)),
            "numpy_version": version("numpy"),
            "scipy_version": version("scipy"),
            "rosbags_version": version("rosbags"),
            "rotation_backend": fields.artifact(
                Path(import_module("scipy.spatial.transform._rotation").__file__)
            ),
            "optimizer_source": fields.artifact(Path(inspect.getfile(least_squares))),
            "independent_reader": fields.artifact(Path(inspect.getfile(AnyReader))),
            "standard_types": fields.artifact(
                Path(import_module("rosbags.typesys.stores.ros1_noetic").__file__)
            ),
            "previous": {
                name: {
                    filename: fields.artifact(previous / old / filename)
                    for filename in [
                        "startup-poses.npz",
                        "lidar-diagnostics.json",
                        "independent.json",
                        "receipt.json",
                    ]
                }
                for name, old, _ in CASES
            },
        }

    frozen = inputs()
    assert frozen["recording"]["sha256"] == fields.BAG_SHA256
    extrinsics = {"identity": np.eye(4), "lever-arm-control": np.eye(4)}
    extrinsics["lever-arm-control"][:3, :3] = Rotation.from_rotvec(
        [0.2, -0.1, 0.3]
    ).as_matrix()
    extrinsics["lever-arm-control"][:3, 3] = [0.5, -0.3, 0.2]
    protocol = {
        "schema": "cloudanalyzer.flatwall_imu_rotation.v1",
        "inputs": frozen,
        "gyro_bias_rad_s": [0.0] * 3,
        "max_gap_s": 0.05,
        "point_time": "UINT32 t in ns from exact integer PointCloud2 header",
        "reference_time": "maximum point timestamp; same previous 64 references for startup",
        "extrinsics": {name: matrix.tolist() for name, matrix in extrinsics.items()},
        "component_tolerances": {"rotation": 1e-10, "coordinates": 2e-8},
        "startup": {
            "scan_indices": list(range(1, 65)),
            "source_cases": [name for name, _, _ in CASES],
            "lidar": LIDAR,
            "initializer_parameters": startup.PARAMS,
            "acceleration_scale": 9.80665,
            "imu_from_lidar": np.eye(4).tolist(),
            "native_oracle_tolerance": 2e-5,
        },
        "model": "Held body gyro, identity start gauge, fixed IMU origin within each scan, including full LiDAR lever arm. Acceleration and reported attitude are ignored by compensation. No IMU-origin translation is modeled.",
        "scope": "Numerical component agreement and same-prefix initialization consistency only. Zero bias and identity real extrinsic are declared, not calibrated. No timing, GPU execution, reference poses, physical accuracy, changed limits/defaults or state adoption.",
    }
    output.mkdir(parents=True, exist_ok=False)
    fields.save(output / "protocol.json", protocol)
    types = get_typestore(Stores.ROS1_NOETIC)
    with AnyReader([bag]) as independent:
        connections = [c for c in independent.connections if c.topic == "/livox/imu"]
        assert len(connections) == 1 and connections[0].msgtype == "sensor_msgs/msg/Imu"
        messages = [
            types.deserialize_ros1(payload, c.msgtype)
            for c, _, payload in independent.messages(connections=connections)
        ]
    times = np.array(
        [
            int(m.header.stamp.sec) * 10**9 + int(m.header.stamp.nanosec)
            for m in messages
        ],
        dtype=np.int64,
    )
    gyro = np.array(
        [
            [m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z]
            for m in messages
        ]
    )
    assert (
        len(times) == 3755
        and np.all(np.diff(times) > 0)
        and np.diff(times).max() <= 50_000_000
    )
    assert len({m.header.frame_id for m in messages}) == 1
    rotations = np.empty((len(times), 3, 3))
    rotations[0] = np.eye(3)
    for k, dt in enumerate(np.diff(times).astype(np.float64) * 1e-9):
        rotations[k + 1] = rotations[k] @ Rotation.from_rotvec(gyro[k] * dt).as_matrix()

    def at(stamps):
        k = np.minimum(np.searchsorted(times, stamps, side="right") - 1, len(times) - 2)
        dt = (stamps - times[k]).astype(np.float64) * 1e-9
        return rotations[k] @ Rotation.from_rotvec(gyro[k] * dt[:, None]).as_matrix()

    reader = core.BagReader(str(bag))
    series = reader.imu_series("/livox/imu", acceleration_scale=9.80665)
    np.testing.assert_array_equal(series.timestamps_ns(), times)
    trajectory = series.rotation_trajectory(
        int(times[0]), int(times[-1]), gyro_bias=(0.0,) * 3
    )
    np.testing.assert_array_equal(trajectory.timestamps_ns(), times)
    rotation_error = max(
        float(np.max(np.abs(trajectory.rotation(int(t)) - r)))
        for t, r in zip(times, rotations)
    )
    assert rotation_error < 1e-10
    old_cases, odometries, raw_controls, poses, stamps_out, reports = (
        {},
        {},
        {},
        {},
        {},
        {},
    )
    for name, old, voxel in CASES:
        with np.load(
            previous / old / "startup-poses.npz", allow_pickle=False
        ) as arrays:
            old_cases[name] = {key: arrays[key].copy() for key in arrays.files}
        old_cases[name]["reports"] = json.loads(
            (previous / old / "lidar-diagnostics.json").read_text()
        )
        odometries[name] = core.LidarOdometry(**LIDAR, registration_voxel=voxel)
        raw_controls[name] = core.LidarOdometry(**LIDAR, registration_voxel=voxel)
        poses[name], stamps_out[name], reports[name] = [], [], []
    digests = {
        name: {kind: hashlib.sha256() for kind in ["oracle", "native"]}
        for name in extrinsics
    }
    coordinate_errors = {name: 0.0 for name in extrinsics}
    skipped, scan_count, point_count, decreasing = [], 0, 0, 0
    native_scans = iter(reader.scans("/livox/points", raw_time_field="t"))
    with AnyReader([bag]) as independent:
        connections = [c for c in independent.connections if c.topic == "/livox/points"]
        assert (
            len(connections) == 1
            and connections[0].msgtype == "sensor_msgs/msg/PointCloud2"
        )
        for index, (connection, _, payload) in enumerate(
            independent.messages(connections=connections)
        ):
            message = types.deserialize_ros1(payload, connection.msgtype)
            expected_fields, raw_times, _ = fields.reference_fields(
                message, timing=True
            )
            scan = next(native_scans)
            assert fields.fields_digest(
                fields.scan_fields(scan)
            ) == fields.fields_digest(expected_fields)
            xyz = scan.positions()
            stamps = scan.point_times_ns(unit="ns", origin="header")
            expected_times = np.fromiter(
                (expected_fields["stamp_ns"] + int(t) for t in raw_times),
                dtype=np.int64,
            )
            np.testing.assert_array_equal(stamps, expected_times)
            reference = int(stamps.max())
            if stamps.min() < times[0] or reference > times[-1]:
                skipped.append(index)
                for parallel in [False, True]:
                    try:
                        trajectory.deskew(
                            xyz,
                            stamps,
                            reference,
                            imu_from_lidar=np.eye(4),
                            parallel=parallel,
                        )
                    except ValueError as error:
                        assert "cover point timestamp at index 0" in str(error)
                    else:
                        raise AssertionError("uncovered first scan was extrapolated")
                continue
            point_rotations, ref_rotation = (
                at(stamps),
                at(np.array([reference], dtype=np.int64))[0],
            )
            original = xyz.tobytes(), stamps.tobytes()
            identity_points = None
            for label, extrinsic in extrinsics.items():
                imu = (
                    expected_fields["positions"] @ extrinsic[:3, :3].T
                    + extrinsic[:3, 3]
                )
                gauge = np.einsum("nij,nj->ni", point_rotations, imu)
                expected = (gauge @ ref_rotation - extrinsic[:3, 3]) @ extrinsic[:3, :3]
                serial = trajectory.deskew(
                    xyz, stamps, reference, imu_from_lidar=extrinsic
                )
                parallel = trajectory.deskew(
                    xyz, stamps, reference, imu_from_lidar=extrinsic, parallel=True
                )
                np.testing.assert_allclose(serial, expected, rtol=0, atol=2e-8)
                np.testing.assert_array_equal(
                    serial.view(np.uint64), parallel.view(np.uint64)
                )
                np.testing.assert_array_equal(
                    serial[stamps == reference].view(np.uint64),
                    xyz[stamps == reference].view(np.uint64),
                )
                coordinate_errors[label] = max(
                    coordinate_errors[label], float(np.max(np.abs(serial - expected)))
                )
                for kind, array in [("oracle", expected), ("native", serial)]:
                    digests[label][kind].update(
                        index.to_bytes(4, "little") + array.astype("<f8").tobytes()
                    )
                if label == "identity":
                    identity_points = serial
            assert (xyz.tobytes(), stamps.tobytes()) == original
            if index in range(1, 65):
                for name, _, _ in CASES:
                    row = index - 1
                    assert reference == old_cases[name]["timestamps_ns"][row]
                    raw_pose = raw_controls[name].register(xyz)
                    np.testing.assert_array_equal(
                        raw_pose.view(np.uint64),
                        old_cases[name]["lidar_poses"][row].view(np.uint64),
                    )
                    assert (
                        raw_controls[name].last_report()
                        == old_cases[name]["reports"][row]
                    )
                    poses[name].append(odometries[name].register(identity_points))
                    stamps_out[name].append(reference)
                    reports[name].append(odometries[name].last_report())
            scan_count += 1
            point_count += len(xyz)
            decreasing += int(np.count_nonzero(np.diff(stamps) < 0))
    assert (
        next(native_scans, None) is None
        and skipped == [0]
        and scan_count == 185
        and point_count == 4_440_000
    )
    startup_results = []
    for name, old, voxel in CASES:
        destination = output / name
        destination.mkdir(exist_ok=False)
        pose_array, stamp_array = (
            np.asarray(poses[name]),
            np.asarray(stamps_out[name], dtype=np.int64),
        )
        np.savez_compressed(
            destination / "startup-poses.npz",
            timestamps_ns=stamp_array,
            lidar_poses=pose_array,
        )
        fields.save(destination / "lidar-diagnostics.json", reports[name])
        oracle = startup.reference(bag, stamp_array, pose_array)
        fields.save(destination / "independent.json", oracle)
        outcomes = []
        for parallel in [False, True]:
            try:
                value = series.initialize_lidar(
                    stamp_array,
                    pose_array,
                    imu_from_lidar=np.eye(4),
                    parallel=parallel,
                    **startup.PARAMS,
                )
            except ValueError as error:
                assert not oracle["accepted"] and "IMU initialization rejected:" in str(
                    error
                )
                assert all(limit in str(error) for limit in oracle["failed_limits"])
                outcomes.append({"accepted": False, "error": str(error)})
            else:
                assert oracle["accepted"]
                for key in ["gravity", "velocities", "bias"]:
                    np.testing.assert_allclose(
                        value[key], oracle[key], rtol=0, atol=2e-5
                    )
                outcomes.append(
                    {
                        "accepted": True,
                        **{
                            k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in value.items()
                        },
                    }
                )
        assert outcomes[0] == outcomes[1]
        fields.save(destination / "native.json", outcomes[0])
        previous_result = json.loads((previous / old / "independent.json").read_text())
        counts = [r["registration_points"] for r in reports[name]]
        result = {
            "case": name,
            "registration_voxel": voxel,
            "scans": len(stamp_array),
            "previous_undeskewed": {
                k: previous_result[k]
                for k in ["accepted", "failed_limits", "diagnostics"]
            },
            "rotation_compensated": {
                k: oracle[k] for k in ["accepted", "failed_limits", "diagnostics"]
            },
            "native": outcomes[0] if not oracle["accepted"] else {"accepted": True},
            "serial_parallel_outcomes_match": True,
            "previous_raw_pose_bits_and_reports_match": True,
            "termination_counts": dict(
                Counter(r["termination"] for r in reports[name])
            ),
            "registration_points": {
                "min": min(counts),
                "median": float(np.median(counts)),
                "max": max(counts),
            },
            "poses_sha256": fields.fields_digest({"poses": pose_array}),
            "outputs": {
                p.name: fields.artifact(p) for p in sorted(destination.iterdir())
            },
        }
        fields.save(destination / "receipt.json", result)
        startup_results.append(result)
    assert inputs() == frozen
    receipt = {
        **protocol,
        "protocol": fields.artifact(output / "protocol.json"),
        "component": {
            "scans": scan_count,
            "points_per_extrinsic": point_count,
            "skipped_scans": skipped,
            "decreasing_point_steps_retained": decreasing,
            "rotation_error": rotation_error,
            "max_coordinate_errors": coordinate_errors,
            "digests": {
                label: {k: d.hexdigest() for k, d in group.items()}
                for label, group in digests.items()
            },
            "serial_parallel_all_bits_match": True,
            "reference_points_all_bits_unchanged": True,
            "caller_buffers_unchanged": True,
        },
        "startup_results": startup_results,
        "no_state_adopted": True,
        "no_reference_poses_used": True,
    }
    fields.save(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["bag", "out", "previous"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.bag, args.out, args.previous)
    print(
        json.dumps(
            {
                "component": result["component"],
                "startup_results": [
                    {k: v for k, v in r.items() if k != "outputs"}
                    for r in result["startup_results"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
