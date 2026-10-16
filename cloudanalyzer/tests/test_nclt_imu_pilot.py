"""Pilot protocol boundaries and independent kinematic/evaluation checks.

The stationary serialization fixture is a correctness test, not real accuracy.
"""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

EPOCH = 1_700_000_000_000_000_123


@pytest.fixture
def pilot(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("pilot_nclt_imu_window")


def test_missing_sample_gap_is_checked_before_endpoint_clipping(pilot):
    imu = EPOCH + np.array([0, 50, 101, 151, 201], dtype=np.int64) * 1_000_000
    # The 51 ms source gap contributes only 1 ms to the second scan edge.
    scans = EPOCH + np.array([0, 100, 150, 201], dtype=np.int64) * 1_000_000
    assert pilot.gap_edges(scans, imu, .05) == [0, 1]
    # A gap ending exactly at the first scan lies outside all integrated edges.
    assert pilot.gap_edges(imu[2:], imu, .05) == []
    with pytest.raises(ValueError, match="bracket"):
        pilot.gap_edges(scans, imu[1:], .05)
    with pytest.raises(ValueError, match="increasing"):
        pilot.gap_edges(scans[::-1], imu, .05)


def test_window_selection_keeps_disjoint_valid_runs_and_exposes_short_tails(pilot):
    windows = pilot.select_windows(40, [2, 3, 13, 34])
    assert windows == [(4, 14), (14, 30), (30, 35), (35, 40)]
    indices = [i for a, b in windows for i in range(a, b)]
    assert len(indices) == len(set(indices))
    for a, b in windows:
        assert not set(range(a, b - 1)).intersection([2, 3, 13, 34])
    assert pilot.select_windows(4, [0, 1, 2]) == []
    for bad in [[1, 1], [2, 1], [-1], [39]]:
        with pytest.raises(ValueError, match="rejected edges"):
            pilot.select_windows(40, bad)


def test_offline_velocity_includes_rotating_imu_lever_arm_at_exact_epoch(pilot):
    t = np.arange(11) * .001
    times = EPOCH + np.arange(11, dtype=np.int64) * 1_000_000
    body = np.tile(np.eye(4), (len(t), 1, 1))
    body[:, :3, :3] = Rotation.from_euler("z", (1.5 * t)[:, None]).as_matrix()
    body[:, :3, 3] = np.column_stack([2 * t + .3 * t**2, -.2 * t, .1 * t**2])
    imu, velocities = pilot.initial_imu_states(times, body)
    offset = np.array([-.11, .18, .71])  # Table 4 in the scan's z-up axes.
    rotated_offset = body[:, :3, :3] @ offset
    expected_v = np.column_stack([2 + .6 * t, np.full(len(t), -.2), .2 * t])
    expected_v += np.cross([0., 0., 1.5], rotated_offset)
    np.testing.assert_allclose(imu[:, :3, 3], body[:, :3, 3] + rotated_offset, atol=1e-14)
    np.testing.assert_allclose(velocities, expected_v, atol=3e-7)
    assert np.max(np.abs(velocities[:, 1] + .2)) > .15


def test_gravity_uses_only_causal_sensor_roll_pitch_and_preserves_native_axes(pilot, tmp_path):
    path = tmp_path / "ms25_euler.csv"
    first = EPOCH // 1000 * 1000
    path.write_text(f"{first // 1000 - 10000},0.2,-0.3,1.7\n{first // 1000 + 1000},1.,1.,2.\n")
    gravity, reading = pilot.initial_gravity(path, EPOCH)
    native = Rotation.from_euler("xyz", [.2, -.3, 1.7]).as_matrix()
    expected = 9.81 * np.diag([1., -1., -1.]) @ native.T @ [0., 0., 1.]
    np.testing.assert_allclose(gravity, expected, atol=1e-12)
    assert reading["timestamp_ns"] < EPOCH
    with pytest.raises(ValueError, match="recent"):
        pilot.initial_gravity(path, EPOCH + 10**9)
    with pytest.raises(ValueError, match="recent"):
        pilot.initial_gravity(path, first - 20_000_000)


def test_reference_has_no_extrapolation_and_uses_body_origin(pilot, tmp_path):
    evaluate = importlib.import_module("evaluate_nclt_imu_pilot")
    path = tmp_path / "truth.csv"
    path.write_text("100000000,1,2,3,0.1,-0.2,0.3\n100050000,2,4,6,0.2,-0.1,0.5\n")
    times = np.array([100_000_000_000, 100_025_000_000], dtype=np.int64)
    actual = evaluate.reference_at(path, times)
    flip = np.diag([1., -1., -1.])
    native = Rotation.from_euler("xyz", [[.1, -.2, .3], [.2, -.1, .5]])
    np.testing.assert_allclose(actual[1, :3, :3], flip @ Slerp([0., 1.], native)(.5).as_matrix() @ flip)
    np.testing.assert_allclose(actual[1, :3, 3], flip @ [1.5, 3., 4.5])
    with pytest.raises(ValueError, match="extrapolation"):
        evaluate.reference_at(path, times + 10**9)
    with pytest.raises(ValueError, match="tolerance"):
        evaluate.reference_at(path, times, tolerance_s=.01)


def test_metrics_match_explicit_se3_error_and_ignore_global_rigid_gauge(pilot):
    evaluate = importlib.import_module("evaluate_nclt_imu_pilot")
    truth = np.tile(np.eye(4), (4, 1, 1))
    truth[:, :3, :3] = Rotation.from_euler("xyz", [[0., .2, .3], [.1, .2, .4], [.3, .2, .5], [.1, .4, .6]]).as_matrix()
    truth[:, :3, 3] = [[0., 0., 0.], [1., .2, -.1], [2., -.3, .4], [3., 1., .2]]
    gauge = np.eye(4)
    gauge[:3, :3] = Rotation.from_euler("xyz", [.6, -.2, 1.]).as_matrix()
    gauge[:3, 3] = [7., -3., 4.]
    estimated = gauge @ truth
    for values in evaluate.pose_errors(estimated, truth).values():
        np.testing.assert_allclose(values, 0., atol=3e-14)
    estimated[-1, :3, 3] += [.2, -.3, .1]
    estimated[-1, :3, :3] @= Rotation.from_rotvec([.1, .2, -.1]).as_matrix()
    errors = evaluate.pose_errors(estimated, truth)
    relative_truth = np.linalg.inv(truth[-2]) @ truth[-1]
    relative_est = np.linalg.inv(estimated[-2]) @ estimated[-1]
    error = np.linalg.inv(relative_truth) @ relative_est
    assert errors["rpe_translation_m"][-1] == pytest.approx(np.linalg.norm(error[:3, 3]))
    assert errors["rpe_rotation_deg"][-1] == pytest.approx(np.degrees(Rotation.from_matrix(error[:3, :3]).magnitude()))
    with pytest.raises(ValueError, match="rigid transform"):
        malformed = estimated.copy()
        malformed[1, 0, 0] += .2
        evaluate.pose_errors(malformed, truth)


def _stationary_recording(tmp_path):
    pytest.importorskip("cloudanalyzer_core")
    pytest.importorskip("rosbags")
    from rosbags.rosbag2 import StoragePlugin, Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS2_HUMBLE)
    cls = lambda name: ts.get_msgdef(name).cls  # noqa: E731
    points = np.random.default_rng(4).uniform([2., -5., -1.], [20., 5., 4.], (80, 3)).astype("<f4")
    fields = [cls("sensor_msgs/msg/PointField")(name=name, offset=4 * i, datatype=7, count=1)
              for i, name in enumerate("xyz")]
    folder = tmp_path / "recording"
    with Writer(folder, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        lidar = writer.add_connection("/points", "sensor_msgs/msg/PointCloud2", typestore=ts)
        imu = writer.add_connection("/ms25/raw", "sensor_msgs/msg/Imu", typestore=ts)
        for k in range(12):
            stamp = EPOCH + k * 10_000_000
            def header(frame):
                return cls("std_msgs/msg/Header")(
                    stamp=cls("builtin_interfaces/msg/Time")(sec=stamp // 10**9, nanosec=stamp % 10**9), frame_id=frame)
            message = cls("sensor_msgs/msg/Imu")(
                header=header("nclt_ms25_native"), orientation=cls("geometry_msgs/msg/Quaternion")(x=0., y=0., z=0., w=0.),
                orientation_covariance=np.array([-1.] + [0.] * 8),
                angular_velocity=cls("geometry_msgs/msg/Vector3")(x=0., y=0., z=0.), angular_velocity_covariance=np.zeros(9),
                linear_acceleration=cls("geometry_msgs/msg/Vector3")(x=0., y=0., z=-9.81), linear_acceleration_covariance=np.zeros(9))
            writer.write(imu, stamp, ts.serialize_cdr(message, imu.msgtype))
            if k in [1, 4, 7, 10]:
                message = cls("sensor_msgs/msg/PointCloud2")(
                    header=header("nclt_body_z_up"), height=1, width=len(points), fields=fields,
                    is_bigendian=False, point_step=12, row_step=12 * len(points),
                    data=np.frombuffer(points.tobytes(), dtype=np.uint8), is_dense=True)
                writer.write(lidar, stamp, ts.serialize_cdr(message, lidar.msgtype))
    euler = tmp_path / "euler.csv"
    euler.write_text(f"{EPOCH // 1000},0,0,0\n")
    return next(folder.glob("*.mcap")), euler


def test_real_native_pipeline_freezes_outputs_and_evaluation_cannot_change_estimates(pilot, tmp_path):
    bag, euler = _stationary_recording(tmp_path)
    output = tmp_path / "estimate"
    report = pilot.run(bag, bag, euler, output)
    assert report["coverage"]["eligible_nodes"] == 4
    assert report["coverage"]["eligible_edges"] == 3
    assert report["coverage"]["nonconverged_windows"] == 0
    assert "ground_truth" not in report["inputs"]
    with pytest.raises(FileExistsError):
        pilot.run(bag, bag, euler, output)
    evaluate = importlib.import_module("evaluate_nclt_imu_pilot")
    truth = tmp_path / "truth.csv"
    truth.write_text("".join(f"{EPOCH // 1000 + k * 5000},0,0,0,0,0,0\n" for k in range(24)))
    before = {p.name: pilot.artifact(p) for p in output.iterdir()}
    metrics = evaluate.evaluate(output, truth, tmp_path / "score")
    assert metrics["metrics"]["fused"]["rpe_translation_m"]["rmse"] < 1e-8
    assert {p.name: pilot.artifact(p) for p in output.iterdir()} == before
    assert json.loads((output / "receipt.json").read_text())["schema"] == pilot.SCHEMA
    with pytest.raises(FileExistsError):
        evaluate.evaluate(output, truth, tmp_path / "score")
    with pytest.raises(ValueError, match="outside"):
        evaluate.evaluate(output, truth, output / "score")
    with (output / "window-000.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate.evaluate(output, truth, tmp_path / "tampered-score")
