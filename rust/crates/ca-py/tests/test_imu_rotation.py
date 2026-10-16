"""Gyro-only reference-time transforms, not a physical trajectory accuracy score."""

import numpy as np
import pytest

core = pytest.importorskip("cloudanalyzer_core")
pytest.importorskip("rosbags")
from test_imu_preintegration import EPOCH, _write  # noqa: E402


def data(tmp_path, format="cdr", *, gyro_problem=None):
    times = EPOCH + np.arange(21, dtype=np.int64) * 10_000_000
    k = np.arange(len(times))
    gyro = np.column_stack([0.2 + 0.01 * k, -0.1 * np.cos(k), 0.3 * np.sin(k)])
    cov = np.zeros((len(times), 3, 9))
    cov[:, 0, 0] = np.nan  # Invalid reported orientation, deliberately ignored.
    cov[:, 2, 0] = -1.0  # Missing acceleration with nonfinite values.
    if gyro_problem == "left-missing":
        cov[1, 1, 0] = -1.0
    elif gyro_problem == "left-invalid":
        cov[1, 1, 0] = -2.0
    elif gyro_problem == "closing-missing":
        cov[-1, 1, 0] = -1.0
    return (
        _write(
            tmp_path,
            times,
            gyro,
            np.full_like(gyro, np.nan),
            format=format,
            covariances=cov,
        ),
        times,
        gyro,
    )


def rotation_at(times, gyro, bias, start, stamp):
    from scipy.spatial.transform import Rotation

    rotation = np.eye(3)
    for k in range(len(times) - 1):
        begin, end = max(start, int(times[k])), min(stamp, int(times[k + 1]))
        if end > begin:
            rotation = (
                rotation
                @ Rotation.from_rotvec(
                    (gyro[k] - bias) * ((end - begin) * 1e-9)
                ).as_matrix()
            )
    return rotation


@pytest.mark.parametrize("format", ["ros1", "cdr", "cdr-be", "sqlite"])
@pytest.mark.parametrize("reference_offset", [13, 93_456_789, 199_999_979])
def test_gyro_only_full_extrinsic_and_strided_inputs_match_scipy(
    tmp_path, format, reference_offset
):
    from scipy.spatial.transform import Rotation

    reader, times, gyro = data(tmp_path, format)
    series = reader.imu_series(acceleration_scale=9.80665)
    bias = np.array([0.01, -0.02, 0.03])
    start, end = int(times[0]) + 13, int(times[-1]) - 21
    trajectory = series.rotation_trajectory(start, end, gyro_bias=tuple(bias))
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    extrinsic[:3, 3] = [0.5, -0.3, 0.2]
    stamps = np.array([end, start, EPOCH + 43_456_789, start, EPOCH + 93_456_789])
    # Both arrays noncontiguous, with decreasing/repeated exact integer times.
    storage = np.repeat(stamps, 2)
    stamps = storage[::2]
    points = np.arange(30, dtype=np.float64).reshape(5, 6)[:, ::2] / 3
    points = np.asfortranarray(points)
    before = points.copy(), stamps.copy(), extrinsic.copy()
    reference = EPOCH + reference_offset
    ref = rotation_at(times, gyro, bias, start, reference)
    expected = []
    for stamp, point in zip(stamps, points):
        rotation = rotation_at(times, gyro, bias, start, int(stamp))
        np.testing.assert_allclose(
            trajectory.rotation(int(stamp)), rotation, rtol=0, atol=1e-12
        )
        expected.append(
            extrinsic[:3, :3].T
            @ (
                ref.T @ rotation @ (extrinsic[:3, :3] @ point + extrinsic[:3, 3])
                - extrinsic[:3, 3]
            )
        )
    serial = trajectory.deskew(points, stamps, reference, imu_from_lidar=extrinsic)
    parallel = trajectory.deskew(
        points, stamps, reference, imu_from_lidar=extrinsic, parallel=True
    )
    np.testing.assert_allclose(serial, expected, rtol=0, atol=1e-11)
    np.testing.assert_array_equal(serial.view(np.uint64), parallel.view(np.uint64))
    for actual, original in zip([points, stamps, extrinsic], before):
        np.testing.assert_array_equal(actual, original)
    serial[:] = -123
    np.testing.assert_array_equal(
        trajectory.deskew(points, stamps, reference, imu_from_lidar=extrinsic).view(
            np.uint64
        ),
        parallel.view(np.uint64),
    )
    # Acceleration scale never enters this separate gyro model.
    other = reader.imu_series(acceleration_scale=1.0).rotation_trajectory(
        start, end, gyro_bias=tuple(bias)
    )
    np.testing.assert_array_equal(other.rotation(end), trajectory.rotation(end))
    with pytest.raises(ValueError, match="acceleration is missing/invalid"):
        series.trajectory(
            start, end, np.eye(4), np.zeros(3), gravity=(0.0,) * 3, bias=(0.0,) * 6
        )


@pytest.mark.parametrize("format", ["ros1", "cdr", "cdr-be", "sqlite"])
def test_owned_copied_immutable_snapshot_and_identity_reference_bits(tmp_path, format):
    reader, times, gyro = data(tmp_path, format, gyro_problem="closing-missing")
    series = reader.imu_series()
    trajectory = series.rotation_trajectory(
        int(times[0]), int(times[-1]), gyro_bias=(0.0,) * 3
    )
    del reader, series
    for path in [tmp_path / "imu.bag", tmp_path / "imu"]:
        if path.exists():
            target = tmp_path / ("moved_" + path.name)
            assert path.resolve().is_relative_to(
                tmp_path.resolve()
            ) and target.resolve().is_relative_to(tmp_path.resolve())
            path.rename(target)
    assert isinstance(trajectory, core.ImuRotationTrajectory) and len(trajectory) == 21
    assert trajectory.start_ns == times[0] and trajectory.end_ns == times[-1]
    assert trajectory.frame_id == "imu/body" and trajectory.gyro_bias == (0.0,) * 3
    np.testing.assert_array_equal(trajectory.timestamps_ns(), times)
    trajectory.timestamps_ns()[:] = -1
    trajectory.rotation(int(times[0]))[:] = -1
    np.testing.assert_array_equal(trajectory.rotation(int(times[0])), np.eye(3))
    for name, value in [
        ("start_ns", 0),
        ("end_ns", 0),
        ("frame_id", "other"),
        ("gyro_bias", (1.0,) * 3),
    ]:
        with pytest.raises(AttributeError):
            setattr(trajectory, name, value)
    assert not hasattr(trajectory, "state")
    points = np.array([[-0.0, 1.0, 2.0]])
    np.testing.assert_array_equal(
        trajectory.deskew(
            points, times[:1], int(times[0]), imu_from_lidar=np.eye(4)
        ).view(np.uint64),
        points.view(np.uint64),
    )
    assert trajectory.deskew(
        np.empty((0, 3)),
        np.empty(0, dtype=np.int64),
        int(times[0]),
        imu_from_lidar=np.eye(4),
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
        "result-nonfinite",
    ],
)
def test_invalid_deskew_inputs_rejected(tmp_path, parallel, problem):
    reader, times, _ = data(tmp_path)
    trajectory = reader.imu_series().rotation_trajectory(
        int(times[0]), int(times[-1]), gyro_bias=(0.0,) * 3
    )
    points, stamps, transform = np.zeros((3, 3)), times[:3].copy(), np.eye(4)
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
        transform = np.eye(3)
    elif problem == "transform-row":
        transform[3, 0] = 1.0
    elif problem == "transform-rotation":
        transform[0, 0] = -1.0
    elif problem == "transform-nonfinite":
        transform[0, 3] = np.inf
    elif problem == "result-nonfinite":
        points[1] = np.finfo(float).max
    with pytest.raises((ValueError, TypeError)):
        trajectory.deskew(
            points, stamps, reference, imu_from_lidar=transform, parallel=parallel
        )


@pytest.mark.parametrize(
    "problem",
    [
        "left-missing",
        "left-invalid",
        "before",
        "after",
        "empty",
        "gap",
        "bias",
        "nonfinite-gap",
    ],
)
def test_constructor_validation_and_coverage(tmp_path, problem):
    reader, times, _ = data(tmp_path, gyro_problem=problem)
    start, end = int(times[0]), int(times[-1])
    bias, gap = (0.0,) * 3, 0.05
    if problem == "before":
        start -= 1
    elif problem == "after":
        end += 1
    elif problem == "empty":
        end = start
    elif problem == "gap":
        gap = 0.001
    elif problem == "bias":
        bias = (np.inf,) * 3
    elif problem == "nonfinite-gap":
        gap = np.nan
    series = reader.imu_series()
    with pytest.raises(ValueError):
        series.rotation_trajectory(start, end, gyro_bias=bias, max_gap_s=gap)


def test_rotation_query_coverage_and_first_indexed_error(tmp_path):
    reader, times, _ = data(tmp_path)
    trajectory = reader.imu_series().rotation_trajectory(
        int(times[0]), int(times[-1]), gyro_bias=(0.0,) * 3
    )
    for stamp in [int(times[0]) - 1, int(times[-1]) + 1]:
        with pytest.raises(ValueError, match="cover"):
            trajectory.rotation(stamp)
    stamps = times[:3].copy()
    stamps[1] = times[0] - 1
    stamps[2] = times[-1] + 1
    errors = []
    for parallel in [False, True]:
        with pytest.raises(ValueError, match="index 1") as caught:
            trajectory.deskew(
                np.zeros((3, 3)),
                stamps,
                int(times[-1]),
                imu_from_lidar=np.eye(4),
                parallel=parallel,
            )
        errors.append(str(caught.value))
    assert errors[0] == errors[1]
