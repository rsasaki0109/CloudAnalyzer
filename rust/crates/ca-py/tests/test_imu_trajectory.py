"""Declared nominal-motion and coordinate tests; no trajectory accuracy score."""

import numpy as np
import pytest

core = pytest.importorskip("cloudanalyzer_core")
pytest.importorskip("rosbags")
from test_imu_preintegration import EPOCH, _oracle, _write  # noqa: E402


def data(tmp_path, format="cdr"):
    times = EPOCH + np.arange(21, dtype=np.int64) * 10_000_000
    k = np.arange(len(times))
    gyro = np.column_stack([0.2 + 0.01 * k, -0.1 * np.cos(k), 0.3 * np.sin(k)])
    accel = np.column_stack([0.05 * k, -0.02 * k, 1.0 + 0.001 * k])
    series = _write(tmp_path, times, gyro, accel, format=format).imu_series(
        acceleration_scale=9.80665
    )
    return series, times, gyro, accel


def configuration():
    from scipy.spatial.transform import Rotation

    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([0.3, -0.2, 0.1]).as_matrix()
    pose[:3, 3] = [10.0, -20.0, 3.0]
    velocity = np.array([1.0, -2.0, 0.5])
    gravity = np.array([-0.1, 0.3, -9.7])
    bias = np.array([0.01, -0.02, 0.03, 0.1, -0.2, 0.05])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    extrinsic[:3, 3] = [0.5, -0.3, 0.2]
    return pose, velocity, gravity, bias, extrinsic


def reference_state(times, gyro, accel, start, stamp, pose, velocity, gravity, bias):
    if stamp == start:
        return pose.copy(), velocity.copy()
    bounds = np.r_[
        np.int64(start), times[(times > start) & (times < stamp)], np.int64(stamp)
    ]
    held = np.searchsorted(times, bounds, side="right") - 1
    dr, dv, dp = _oracle(bounds, gyro[held], accel[held] * 9.80665, bias)
    dt = (stamp - start) * 1e-9
    result = pose.copy()
    result[:3, :3] = pose[:3, :3] @ dr
    result[:3, 3] += velocity * dt + 0.5 * gravity * dt**2 + pose[:3, :3] @ dp
    return result, velocity + gravity * dt + pose[:3, :3] @ dv


@pytest.mark.parametrize("format", ["ros1", "cdr", "cdr-be", "sqlite"])
@pytest.mark.parametrize("reference_offset", [13, 93_456_789, 199_999_979])
def test_states_and_full_extrinsic_deskew_match_independent_oracle(
    tmp_path, format, reference_offset
):
    pytest.importorskip("scipy")
    series, times, gyro, accel = data(tmp_path, format)
    pose, velocity, gravity, bias, extrinsic = configuration()
    start, end = int(times[0]) + 13, int(times[-1]) - 21
    trajectory = series.trajectory(
        start, end, pose, velocity, gravity=tuple(gravity), bias=tuple(bias)
    )
    stamps = np.array(
        [end, start, EPOCH + 43_456_789, start, EPOCH + 93_456_789], dtype=np.int64
    )
    positions = (
        np.arange(30, dtype=np.float64).reshape(5, 6)[:, ::2] / 3
    )  # Noncontiguous points.
    original = positions.copy()
    reference = EPOCH + reference_offset
    expected = []
    ref_pose, _ = reference_state(
        times, gyro, accel, start, reference, pose, velocity, gravity, bias
    )
    ref_lidar = ref_pose @ extrinsic
    for stamp, point in zip(stamps, positions):
        expected_pose, expected_velocity = reference_state(
            times, gyro, accel, start, int(stamp), pose, velocity, gravity, bias
        )
        state = trajectory.state(int(stamp))
        np.testing.assert_allclose(state["pose"], expected_pose, rtol=0, atol=3e-11)
        np.testing.assert_allclose(
            state["velocity"], expected_velocity, rtol=0, atol=3e-11
        )
        lidar_pose = expected_pose @ extrinsic
        expected.append(
            ref_lidar[:3, :3].T
            @ (lidar_pose[:3, :3] @ point + lidar_pose[:3, 3] - ref_lidar[:3, 3])
        )
    serial = trajectory.deskew(positions, stamps, reference, imu_from_lidar=extrinsic)
    parallel = trajectory.deskew(
        positions, stamps, reference, imu_from_lidar=extrinsic, parallel=True
    )
    np.testing.assert_allclose(serial, expected, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(serial.view(np.uint64), parallel.view(np.uint64))
    np.testing.assert_array_equal(positions, original)
    serial[:] = -123
    np.testing.assert_array_equal(
        trajectory.deskew(positions, stamps, reference, imu_from_lidar=extrinsic).view(
            np.uint64
        ),
        parallel.view(np.uint64),
    )


@pytest.mark.parametrize("format", ["ros1", "cdr", "cdr-be", "sqlite"])
def test_owned_snapshot_copied_metadata_and_world_origin_invariance(tmp_path, format):
    pytest.importorskip("scipy")
    series, times, gyro, accel = data(tmp_path, format)
    pose, velocity, gravity, bias, extrinsic = configuration()
    start, end = int(times[0]), int(times[-1])
    kwargs = dict(gravity=tuple(gravity), bias=tuple(bias))
    trajectory = series.trajectory(start, end, pose, velocity, **kwargs)
    shifted_pose = pose.copy()
    shifted_pose[:3, 3] += [1e12, -2e12, 3e12]
    shifted = series.trajectory(start, end, shifted_pose, velocity, **kwargs)
    del series
    for path in [tmp_path / "imu.bag", tmp_path / "imu"]:
        if path.exists():
            target = tmp_path / ("moved_" + path.name)
            assert path.resolve().is_relative_to(
                tmp_path.resolve()
            ) and target.resolve().is_relative_to(tmp_path.resolve())
            path.rename(target)
    assert isinstance(trajectory, core.ImuTrajectory) and len(trajectory) == 21
    assert trajectory.start_ns == start and trajectory.end_ns == end
    assert (
        trajectory.frame_id == "imu/body" and trajectory.acceleration_scale == 9.80665
    )
    np.testing.assert_array_equal(trajectory.timestamps_ns(), times)
    trajectory.timestamps_ns()[:] = -1
    state = trajectory.state(start)
    state["pose"][:] = -1
    state["velocity"][:] = -1
    np.testing.assert_array_equal(trajectory.state(start)["pose"], pose)
    np.testing.assert_array_equal(trajectory.state(start)["velocity"], velocity)
    for name, value in [
        ("start_ns", 0),
        ("end_ns", 0),
        ("frame_id", "changed"),
        ("acceleration_scale", 1.0),
    ]:
        with pytest.raises(AttributeError):
            setattr(trajectory, name, value)
    stamps = times[::-3]  # Noncontiguous, decreasing integer timestamps.
    points = np.arange(len(stamps) * 3, dtype=np.float64).reshape(-1, 3)
    np.testing.assert_array_equal(
        trajectory.deskew(points, stamps, end, imu_from_lidar=extrinsic).view(
            np.uint64
        ),
        shifted.deskew(points, stamps, end, imu_from_lidar=extrinsic).view(np.uint64),
    )
    assert trajectory.deskew(
        np.empty((0, 3)), np.empty(0, dtype=np.int64), start, imu_from_lidar=extrinsic
    ).shape == (0, 3)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "problem",
    [
        "point-shape",
        "count",
        "time-dtype",
        "point-nonfinite",
        "time-range",
        "reference-range",
        "transform-shape",
        "transform-row",
        "transform-rotation",
        "transform-nonfinite",
    ],
)
def test_invalid_deskew_inputs_rejected(tmp_path, parallel, problem):
    series, times, gyro, accel = data(tmp_path)
    trajectory = series.trajectory(
        int(times[0]),
        int(times[-1]),
        np.eye(4),
        np.zeros(3),
        gravity=(0.0, 0.0, 0.0),
        bias=(0.0,) * 6,
    )
    points, stamps, extrinsic = np.zeros((3, 3)), times[:3].copy(), np.eye(4)
    reference = int(times[-1])
    if problem == "point-shape":
        points = np.zeros((3, 4))
    elif problem == "count":
        stamps = stamps[:2]
    elif problem == "time-dtype":
        stamps = stamps.astype(np.float64)
    elif problem == "point-nonfinite":
        points[1, 0] = np.nan
    elif problem == "time-range":
        stamps[1] = times[0] - 1
    elif problem == "reference-range":
        reference += 1
    elif problem == "transform-shape":
        extrinsic = np.eye(3)
    elif problem == "transform-row":
        extrinsic[3, 0] = 1.0
    elif problem == "transform-rotation":
        extrinsic[0, 0] = -1.0
    elif problem == "transform-nonfinite":
        extrinsic[0, 3] = np.inf
    with pytest.raises((ValueError, TypeError)):
        trajectory.deskew(
            points, stamps, reference, imu_from_lidar=extrinsic, parallel=parallel
        )


@pytest.mark.parametrize(
    "problem",
    [
        "order",
        "coverage",
        "gap",
        "pose-shape",
        "pose-row",
        "pose-rotation",
        "velocity",
        "gravity",
        "bias",
    ],
)
def test_invalid_builder_inputs_rejected(tmp_path, problem):
    series, times, gyro, accel = data(tmp_path)
    start, end = int(times[0]), int(times[-1])
    pose, velocity = np.eye(4), np.zeros(3)
    kwargs = dict(gravity=(0.0, 0.0, 0.0), bias=(0.0,) * 6)
    if problem == "order":
        end = start
    elif problem == "coverage":
        start -= 1
    elif problem == "gap":
        kwargs["max_gap_s"] = 0.001
    elif problem == "pose-shape":
        pose = np.eye(3)
    elif problem == "pose-row":
        pose[3, 0] = 1.0
    elif problem == "pose-rotation":
        pose[0, 0] = 2.0
    elif problem == "velocity":
        velocity[0] = np.inf
    elif problem == "gravity":
        kwargs["gravity"] = (np.nan, 0.0, 0.0)
    elif problem == "bias":
        kwargs["bias"] = (np.inf,) * 6
    with pytest.raises(ValueError):
        series.trajectory(start, end, pose, velocity, **kwargs)
