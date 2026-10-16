"""Independent startup coordinate/acceptance tests, not trajectory benchmarks."""

import numpy as np
import pytest

core = pytest.importorskip("cloudanalyzer_core")
pytest.importorskip("rosbags")
pytest.importorskip("scipy")
from scipy.spatial.transform import Rotation  # noqa: E402
from test_imu_preintegration import EPOCH, _write  # noqa: E402


def fixture(tmp_path, format="cdr", scale=9.80665, gyro_bias=(0.015, -0.02, 0.01)):
    times = EPOCH + np.arange(121, dtype=np.int64) * 5_000_000
    k = np.arange(len(times))
    bias = np.r_[gyro_bias, [0.1, -0.2, 0.05]]
    angular = np.column_stack(
        [0.3 + 0.1 * np.sin(k / 7), -0.2 * np.cos(k / 9), 0.4 + 0.1 * np.cos(k / 13)]
    )
    force = np.column_stack(
        [0.3 * np.sin(k / 5), 0.2 * np.cos(k / 7), 9.80665 + 0.1 * np.sin(k / 13)]
    )
    covariance = np.zeros((len(times), 3, 9))
    covariance[:, 0, 0] = -1  # No sensor attitude is available or used.
    reader = _write(
        tmp_path,
        times,
        angular + bias[:3],
        (force + bias[3:]) / scale,
        format=format,
        covariances=covariance,
    )
    series = reader.imu_series(acceleration_scale=scale)
    gravity = np.array([-0.2, 0.35, -9.78])
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([0.3, -0.2, 0.1]).as_matrix()
    pose[:3, 3] = [10.0, -20.0, 3.0]
    velocity = np.array([1.2, -0.4, 0.3])
    poses, velocities = [pose.copy()], [velocity.copy()]
    for i, dt in enumerate(np.diff(times).astype(np.float64) * 1e-9):
        a = pose[:3, :3] @ force[i] + gravity
        pose[:3, 3] += velocity * dt + 0.5 * a * dt**2
        velocity += a * dt
        pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(angular[i] * dt).as_matrix()
        poses.append(pose.copy())
        velocities.append(velocity.copy())
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_rotvec([-0.2, 0.1, 0.3]).as_matrix()
    extrinsic[:3, 3] = [0.5, -0.3, 0.2]
    selected = np.arange(0, len(times), 20)
    imu_poses, velocity = np.array(poses)[selected], np.array(velocities)[selected]
    lidar = imu_poses @ extrinsic
    settings = dict(
        imu_from_lidar=extrinsic,
        gyro_bias=(0.0,) * 3,
        accel_bias=tuple(bias[3:]),
        gravity_magnitude=float(np.linalg.norm(gravity)),
        max_gravity_error=1e-6,
        max_rotation_rms=1e-8,
        max_velocity_rms=1e-7,
        max_gyro_bias_change=0.3,
    )
    return series, times[selected], lidar, imu_poses, velocity, gravity, bias, settings


@pytest.mark.parametrize("format", ["ros1", "cdr", "cdr-be", "sqlite"])
@pytest.mark.parametrize("scale", [1.0, 9.80665])
@pytest.mark.parametrize("parallel", [False, True])
def test_moving_start_full_extrinsics_gravity_velocity_and_bias_match_independent_dynamics(
    tmp_path, format, scale, parallel
):
    series, times, lidar, poses, velocity, gravity, bias, settings = fixture(
        tmp_path, format, scale
    )
    # Ascending logical order with negative physical strides, plus strided columns.
    tview = np.repeat(times[::-1], 2)[::-2]
    pview = np.repeat(np.repeat(lidar[::-1], 2, axis=0), 2, axis=2)[::-2, :, ::2]
    before_times, before_poses = tview.copy(), pview.copy()
    out = series.initialize_lidar(tview, pview, parallel=parallel, **settings)
    np.testing.assert_allclose(out["bias"], bias, rtol=0, atol=1e-9)
    np.testing.assert_allclose(out["gravity"], gravity, rtol=0, atol=1e-8)
    np.testing.assert_allclose(out["velocities"], velocity, rtol=0, atol=1e-8)
    np.testing.assert_allclose(out["imu_poses"], poses, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(out["timestamps_ns"], times)
    np.testing.assert_array_equal(tview, before_times)
    np.testing.assert_array_equal(pview, before_poses)
    assert out["accel_bias_fixed"] is True and out["frame_id"] == "imu/body"
    assert out["acceleration_scale"] == scale
    assert out["gyro_singular_ratio"] > 0.99
    assert out["rotation_rms"] < 1e-9 and out["velocity_rms"] < 1e-8
    other = series.initialize_lidar(times, lidar, parallel=not parallel, **settings)
    for key in ["timestamps_ns", "imu_poses", "velocities", "gravity", "bias"]:
        np.testing.assert_array_equal(
            out[key].view(np.uint64), other[key].view(np.uint64)
        )
        original = out[key].copy()
        out[key][...] = -123
        np.testing.assert_array_equal(
            series.initialize_lidar(times, lidar, **settings)[key], original
        )


def test_large_gyro_bias_is_reintegrated_and_three_references_are_sufficient(tmp_path):
    series, times, lidar, _, velocity, gravity, bias, settings = fixture(
        tmp_path, gyro_bias=(0.12, -0.09, 0.1)
    )
    out = series.initialize_lidar(times[::3], lidar[::3], **settings)
    assert len(out["timestamps_ns"]) == 3 and out["gyro_iterations"] >= 2
    np.testing.assert_allclose(out["bias"], bias, atol=1e-9, rtol=0)
    np.testing.assert_allclose(out["gravity"], gravity, atol=1e-8, rtol=0)
    np.testing.assert_allclose(out["velocities"], velocity[::3], atol=1e-8, rtol=0)


@pytest.mark.parametrize("parallel", [False, True])
def test_world_origin_is_removed_before_lever_arm_and_series_owns_recording(
    tmp_path, parallel
):
    times = EPOCH + np.arange(65, dtype=np.int64) * 7_812_500
    reader = _write(
        tmp_path,
        times,
        np.tile([0.01, -0.02, 0.015], (len(times), 1)),
        np.tile([0.0, 0.0, 9.80665], (len(times), 1)),
    )
    series = reader.imu_series()
    del reader
    source, target = tmp_path / "imu", tmp_path / "moved_imu"
    assert source.resolve().is_relative_to(
        tmp_path.resolve()
    ) and target.resolve().is_relative_to(tmp_path.resolve())
    source.rename(target)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix()
    extrinsic[:3, 3] = [0.5, -0.25, 0.125]
    poses = np.tile(np.eye(4), (5, 1, 1))
    poses[:, :3, 3] = np.arange(5)[:, None] * [0.125, 0.0625, -0.03125]
    lidar = poses @ extrinsic
    shifted = lidar.copy()
    shifted[:, :3, 3] += [1e12, -2e12, 3e12]
    settings = dict(
        imu_from_lidar=extrinsic,
        gyro_bias=(0.0,) * 3,
        accel_bias=(0.0,) * 3,
        gravity_magnitude=9.80665,
        max_gravity_error=1e-7,
        max_rotation_rms=1e-8,
        max_velocity_rms=1e-7,
        max_gyro_bias_change=0.2,
        parallel=parallel,
    )
    a = series.initialize_lidar(times[::16], lidar, **settings)
    b = series.initialize_lidar(times[::16], shifted, **settings)
    for key in ["gravity", "velocities", "bias"]:
        np.testing.assert_array_equal(a[key].view(np.uint64), b[key].view(np.uint64))
    np.testing.assert_allclose(
        a["velocities"], [[1.0, 0.5, -0.25]] * 5, atol=1e-9, rtol=0
    )


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "bad",
    [
        "count",
        "two_poses",
        "shape",
        "homogeneous",
        "reflection",
        "nonfinite",
        "duplicate",
        "reverse",
        "uncovered",
        "float_times",
        "float_poses",
        "extrinsic_shape",
        "extrinsic_row",
        "extrinsic_reflection",
    ],
)
def test_invalid_inputs_leave_all_caller_arrays_and_imu_snapshot_unchanged(
    tmp_path, parallel, bad
):
    series, times, lidar, _, _, _, _, settings = fixture(tmp_path)
    settings = dict(settings)
    if bad == "count":
        times = times[:-1]
    elif bad == "two_poses":
        times, lidar = times[:2], lidar[:2]
    elif bad == "shape":
        lidar = lidar[:, :3, :3]
    elif bad == "homogeneous":
        lidar[2, 3, 0] = 0.1
    elif bad == "reflection":
        lidar[2, :3, 0] *= -1
    elif bad == "nonfinite":
        lidar[1, 0, 3] = np.nan
    elif bad == "duplicate":
        times[2] = times[1]
    elif bad == "reverse":
        times = times[::-1]
    elif bad == "uncovered":
        times[0] -= 1
    elif bad == "float_times":
        times = times.astype(np.float64)
    elif bad == "float_poses":
        lidar = lidar.astype(np.float32)
    elif bad == "extrinsic_shape":
        settings["imu_from_lidar"] = np.eye(3)
    elif bad == "extrinsic_row":
        settings["imu_from_lidar"][3, 0] = 0.01
    elif bad == "extrinsic_reflection":
        settings["imu_from_lidar"][:3, 0] *= -1
    before = times.copy(), lidar.copy(), series.timestamps_ns(), series.frame_ids()
    with pytest.raises((ValueError, TypeError)):
        series.initialize_lidar(times, lidar, parallel=parallel, **settings)
    for a, b in zip([times, lidar, series.timestamps_ns()], before[:3]):
        np.testing.assert_array_equal(a, b)
    assert series.frame_ids() == before[3]


@pytest.mark.parametrize(
    "key,value",
    [
        ("gravity_magnitude", 0.0),
        ("gravity_magnitude", np.nan),
        ("max_gravity_error", -1.0),
        ("max_rotation_rms", -1.0),
        ("max_velocity_rms", np.inf),
        ("max_gyro_bias_change", 0.0),
        ("max_iterations", 0),
        ("max_iterations", 101),
        ("max_gap_s", 0.0),
        ("min_singular_ratio", 0.0),
        ("min_singular_ratio", 1.1),
        ("gyro_bias", (0.0, np.nan, 0.0)),
        ("accel_bias", (np.inf, 0.0, 0.0)),
    ],
)
def test_limits_and_fixed_bias_must_be_finite_and_valid(tmp_path, key, value):
    series, times, lidar, _, _, _, _, settings = fixture(tmp_path)
    settings[key] = value
    with pytest.raises(ValueError):
        series.initialize_lidar(times, lidar, **settings)


@pytest.mark.parametrize(
    "bad", ["gravity", "rotation", "velocity", "bias_limit", "iterations"]
)
def test_inconsistent_estimates_are_not_returned_or_silently_normalized(tmp_path, bad):
    series, times, lidar, _, _, _, _, settings = fixture(tmp_path)
    if bad == "gravity":
        settings["gravity_magnitude"] = 12.0
    elif bad == "rotation":
        lidar[2, :3, :3] = (
            lidar[2, :3, :3] @ Rotation.from_rotvec([0.03, -0.02, 0.01]).as_matrix()
        )
    elif bad == "velocity":
        lidar[3, 0, 3] += 0.1
    elif bad == "bias_limit":
        settings["max_gyro_bias_change"] = 1e-12
    else:
        settings["max_iterations"] = 1
    with pytest.raises(ValueError):
        series.initialize_lidar(times, lidar, **settings)
