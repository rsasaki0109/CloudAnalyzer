"""Numerical/serialization checks, not a real-world odometry accuracy benchmark."""

import numpy as np
import pytest

core = pytest.importorskip("cloudanalyzer_core")
pytest.importorskip("rosbags")
from rosbags.rosbag1 import Writer as Writer1  # noqa: E402
from rosbags.rosbag2 import StoragePlugin, Writer as Writer2  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402

EPOCH = 1_700_000_000_000_000_123
KIND = "sensor_msgs/msg/Imu"


def _write(tmp_path, times, gyro, accel, *, format="cdr", covariances=None, frames=None):
    ros1 = format == "ros1"
    ts = get_typestore(Stores.ROS1_NOETIC if ros1 else Stores.ROS2_HUMBLE)
    cls = lambda name: ts.get_msgdef(name).cls  # noqa: E731
    messages = []
    for i, t in enumerate(times):
        stamp = cls("builtin_interfaces/msg/Time")(sec=int(t // 10**9), nanosec=int(t % 10**9))
        header = cls("std_msgs/msg/Header")(
            stamp=stamp, frame_id=frames[i] if frames else "imu/body",
            **({"seq": i} if ros1 else {}),
        )
        cov = covariances[i] if covariances is not None else np.zeros((3, 9))
        messages.append(cls(KIND)(
            header=header, orientation=cls("geometry_msgs/msg/Quaternion")(x=0., y=0., z=0., w=1.),
            orientation_covariance=np.asarray(cov[0], dtype=np.float64),
            angular_velocity=cls("geometry_msgs/msg/Vector3")(**dict(zip("xyz", gyro[i]))),
            angular_velocity_covariance=np.asarray(cov[1], dtype=np.float64),
            linear_acceleration=cls("geometry_msgs/msg/Vector3")(**dict(zip("xyz", accel[i]))),
            linear_acceleration_covariance=np.asarray(cov[2], dtype=np.float64),
        ))
    if ros1:
        path = tmp_path / "imu.bag"
        with Writer1(path) as writer:
            connection = writer.add_connection("/imu", KIND, typestore=ts)
            for i, message in enumerate(messages):
                writer.write(connection, EPOCH + i * 10_000_000, ts.serialize_ros1(message, KIND))
    else:
        path = tmp_path / "imu"
        storage = StoragePlugin.SQLITE3 if format == "sqlite" else StoragePlugin.MCAP
        with Writer2(path, version=9, storage_plugin=storage) as writer:
            connection = writer.add_connection("/imu", KIND, typestore=ts)
            for i, message in enumerate(messages):
                writer.write(connection, EPOCH + i * 10_000_000,
                             ts.serialize_cdr(message, KIND, little_endian=format != "cdr-be"))
    return core.BagReader(str(path))


@pytest.mark.parametrize("format", ["ros1", "cdr", "cdr-be", "sqlite"])
def test_raw_imu_matches_rosbags_fields_and_preserves_exact_header_times(tmp_path, format):
    times = EPOCH + np.arange(4, dtype=np.int64) * 100
    gyro = np.array([[0.1, -0.2, 0.3]] * 4)
    accel = np.array([[1., 2., 9.81]] * 4)
    cov = np.zeros((4, 3, 9))
    cov[0, :, 0] = -1.0
    cov[2] = np.tile(np.diag([0.01, 0.02, 0.03]).ravel(), (3, 1))
    cov[3, :, 0] = -2.0
    reader = _write(tmp_path, times, gyro, accel, format=format, covariances=cov)
    raw = reader.imu_raw()
    assert raw["timestamps_ns"].dtype == np.int64
    np.testing.assert_array_equal(raw["timestamps_ns"], times)
    assert raw["frame_ids"] == ["imu/body"] * 4
    np.testing.assert_array_equal(raw["orientation_xyzw"], [[0., 0., 0., 1.]] * 4)
    np.testing.assert_array_equal(raw["angular_velocity_rad_s"], gyro)
    np.testing.assert_array_equal(raw["linear_acceleration_m_s2"], accel)
    for i, key in enumerate(["orientation_covariance", "angular_velocity_covariance", "linear_acceleration_covariance"]):
        np.testing.assert_array_equal(raw[key], cov[:, i].reshape(4, 3, 3))
    for key in ["orientation_status", "angular_velocity_status", "linear_acceleration_status"]:
        assert raw[key] == ["missing", "unknown", "known", "invalid"]
    # Legacy output remains a separate gravity-direction contract.
    stamps, ups, old_accel = reader.imu()
    np.testing.assert_array_equal(stamps, raw["timestamps_s"])
    assert np.isnan(ups[0]).all()
    np.testing.assert_array_equal(ups[1:], [[0., 0., 1.]] * 3)
    np.testing.assert_array_equal(old_accel, accel)


def _oracle(times, gyro, accel, bias):
    """SciPy rotations and explicit nominal dynamics, independent of Rust Jacobians."""
    from scipy.spatial.transform import Rotation

    rotation = np.eye(3)
    velocity = np.zeros(3)
    position = np.zeros(3)
    for i, duration in enumerate(np.diff(times).astype(float) * 1e-9):
        specific_force = rotation @ (accel[i] - bias[3:])
        position += velocity * duration + specific_force * duration**2 / 2
        velocity += specific_force * duration
        rotation = rotation @ Rotation.from_rotvec((gyro[i] - bias[:3]) * duration).as_matrix()
    return rotation, velocity, position


def test_nominal_bias_and_full_covariance_match_independent_numerical_oracle(tmp_path):
    from scipy.spatial.transform import Rotation

    times = EPOCH + np.array([0, 7, 19, 24, 39, 48, 60], dtype=np.int64) * 1_000_000
    k = np.arange(len(times))
    gyro = np.column_stack([np.sin(k), 0.3 * np.cos(k), 0.4 + 0.1 * k])
    accel = np.column_stack([0.2 * k, np.cos(k), 9.8 + 0.1 * k])
    bias = np.array([0.03, -0.02, 0.01, 0.1, 0.2, -0.05])
    reader = _write(tmp_path, times, gyro, accel)
    out = reader.preintegrate_imu(int(times[0]), int(times[-1]),
                                gyro_noise_density=0.02, accel_noise_density=0.15,
                                bias=tuple(bias))
    nominal = _oracle(times, gyro, accel, bias)
    for name, expected in zip(["delta_rotation", "delta_velocity", "delta_position"], nominal):
        np.testing.assert_allclose(out[name], expected, rtol=1e-12, atol=1e-14)
    assert out["delta_time_s"] == pytest.approx(0.06)
    assert out["integration_steps"] == 6
    assert out["frame_id"] == "imu/body"

    def error(state):
        return np.r_[Rotation.from_matrix(nominal[0].T @ state[0]).as_rotvec(),
                     state[1] - nominal[1], state[2] - nominal[2]]

    # At this time scale, smaller perturbations lose off-diagonal covariance
    # accuracy to cancellation. A 1e-3..1e-5 step sweep verifies convergence.
    eps = 1e-4
    numerical_covariance = np.zeros((9, 9))
    for i, dt in enumerate(np.diff(times).astype(float) * 1e-9):
        for j in range(6):
            plus_g, minus_g, plus_a, minus_a = [x.copy() for x in [gyro, gyro, accel, accel]]
            plus, minus = (plus_g, minus_g) if j < 3 else (plus_a, minus_a)
            plus[i, j % 3] += eps
            minus[i, j % 3] -= eps
            derivative = (error(_oracle(times, plus_g, plus_a, bias))
                          - error(_oracle(times, minus_g, minus_a, bias))) / (2 * eps)
            density = 0.02 if j < 3 else 0.15
            numerical_covariance += np.outer(derivative, derivative) * density**2 / dt
    np.testing.assert_allclose(out["covariance"], numerical_covariance, rtol=2e-6, atol=1e-13)
    np.testing.assert_allclose(out["covariance"], out["covariance"].T, rtol=0, atol=1e-18)
    assert np.linalg.eigvalsh(out["covariance"]).min() > -1e-16

    for j in range(6):
        plus = bias.copy()
        minus = bias.copy()
        plus[j] += eps
        minus[j] -= eps
        derivative = (error(_oracle(times, gyro, accel, plus))
                      - error(_oracle(times, gyro, accel, minus))) / (2 * eps)
        np.testing.assert_allclose(out["bias_jacobian"][:, j], derivative, rtol=1e-5, atol=2e-10)


def test_clipped_nanosecond_window_does_not_require_orientation_or_final_measurement(tmp_path):
    times = EPOCH + np.array([0, 100, 200], dtype=np.int64)
    cov = np.zeros((3, 3, 9))
    cov[:, 0, 0] = -1.0
    cov[2, 1:, 0] = -1.0
    reader = _write(tmp_path, times, np.zeros((3, 3)), np.array([[2., 0., 0.]] * 3), covariances=cov)
    out = reader.preintegrate_imu(int(times[0] + 50), int(times[-1]), gyro_noise_density=0., accel_noise_density=0.)
    np.testing.assert_allclose(out["delta_velocity"], [300e-9, 0., 0.], rtol=1e-14, atol=1e-25)
    np.testing.assert_allclose(out["delta_position"], [150e-9**2, 0., 0.], rtol=1e-14, atol=1e-30)
    np.testing.assert_array_equal(out["covariance"], np.zeros((9, 9)))


@pytest.mark.parametrize("problem", ["missing", "invalid", "gap", "duplicate", "frame", "nan", "overflow"])
def test_preintegration_rejects_unusable_data(tmp_path, problem):
    times = EPOCH + np.array([0, 10, 20], dtype=np.int64) * 1_000_000
    gyro = np.zeros((3, 3))
    accel = np.array([[0., 0., 9.81]] * 3)
    cov = np.zeros((3, 3, 9))
    frames = ["imu"] * 3
    if problem == "missing":
        cov[0, 1, 0] = -1.
    elif problem == "invalid":
        cov[0, 2, [0, 1, 3, 4, 8]] = [1., 2., 2., 1., 1.]
    elif problem == "gap":
        times[-1] += 100_000_000
    elif problem == "duplicate":
        times[1] = times[0]
    elif problem == "frame":
        frames[1] = "other"
    elif problem == "nan":
        gyro[0, 0] = np.nan
    else:
        accel[:, 0] = 1e308
    reader = _write(tmp_path, times, gyro, accel, covariances=cov, frames=frames)
    kwargs = {"gyro_noise_density": 0.01, "accel_noise_density": 0.1}
    if problem == "overflow":
        kwargs["accel_noise_density"] = 1e308
    with pytest.raises(ValueError):
        reader.preintegrate_imu(int(times[0]), int(times[-1]), **kwargs)


def test_no_imu_returns_shaped_empty_arrays_and_no_preintegration(tmp_path):
    path = tmp_path / "empty.bag"
    with Writer1(path):
        pass
    reader = core.BagReader(str(path))
    raw = reader.imu_raw()
    assert raw["timestamps_ns"].shape == (0,)
    assert raw["orientation_xyzw"].shape == (0, 4)
    assert raw["angular_velocity_rad_s"].shape == (0, 3)
    assert raw["linear_acceleration_covariance"].shape == (0, 3, 3)
    assert raw["frame_ids"] == []
    with pytest.raises(ValueError, match="bracket"):
        reader.preintegrate_imu(0, 100, gyro_noise_density=0.01, accel_noise_density=0.1)
    with pytest.raises(ValueError, match="topic not found"):
        reader.imu_raw("/missing")
