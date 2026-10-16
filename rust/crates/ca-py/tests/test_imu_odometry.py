"""Reference coordinates and state ownership; fixtures are not accuracy benchmarks."""

import numpy as np
import pytest

core = pytest.importorskip("cloudanalyzer_core")
pytest.importorskip("rosbags")
from test_imu_trajectory import configuration, data, reference_state  # noqa: E402
from test_imu_preintegration import EPOCH  # noqa: E402


def setup(tmp_path, format="cdr", *, shift=None):
    series, times, gyro, accel = data(tmp_path, format)
    pose, velocity, gravity, bias, extrinsic = configuration()
    if shift is not None:
        pose[:3, 3] += shift
    start, end = int(times[0]) + 13, int(times[-1]) - 21
    motion = series.trajectory(
        start, end, pose, velocity, gravity=tuple(gravity), bias=tuple(bias)
    )
    model_extrinsic = extrinsic.copy()

    def lidar_pose(stamp):
        state, _ = reference_state(
            times, gyro, accel, start, stamp, pose, velocity, gravity, bias
        )
        return state @ model_extrinsic

    return motion, extrinsic, lidar_pose


def thin(points, voxel, min_range=0.0, max_range=80.0):
    seen, result = set(), []
    for p in points:
        if not min_range <= np.linalg.norm(p) <= max_range:
            continue
        key = tuple(np.floor(p / voxel).astype(np.int64))
        if key not in seen:
            seen.add(key)
            result.append(p)
    return np.asarray(result, dtype=np.float64).reshape(-1, 3)


@pytest.mark.parametrize("format", ["ros1", "cdr", "cdr-be", "sqlite"])
@pytest.mark.parametrize("parallel", [False, True])
def test_deskew_icp_reference_pose_and_placed_map_use_the_same_coordinates(
    tmp_path, format, parallel
):
    motion, extrinsic, lidar_pose = setup(tmp_path, format)
    settings = dict(min_range=0.0, map_voxel=0.03, diagnostics=True)
    odometry = core.ImuLidarOdometry(motion, extrinsic, parallel=parallel, **settings)
    control = core.LidarOdometry(deskew=False, **settings)
    landmarks = np.random.default_rng(7105).uniform(-8.0, 8.0, (240, 3)) + [5, 2, 1]
    references = EPOCH + np.array([50_000_000, 100_000_000, 150_000_000, 199_999_979])
    first = lidar_pose(int(references[0]))
    assert odometry.world_from_first_lidar is None
    assert odometry.last_prediction() is odometry.last_report() is None
    assert odometry.placed().shape == (0, 3)
    assert odometry.poses().shape == (0, 4, 4)
    assert odometry.last_iterations() == 0
    previous = None
    for reference in references:
        reference = int(reference)
        stamps = (
            reference - 40_000_000 + np.arange(len(landmarks)) * 927_371 % 40_000_001
        )
        stamps[-1] = reference
        source = []
        for point, stamp in zip(landmarks, stamps):
            pose = lidar_pose(int(stamp))
            source.append((point - pose[:3, 3]) @ pose[:3, :3])
        # Negative-stride/noncontiguous buffers with correspondingly ordered times.
        source = np.repeat(np.asarray(source), 2, axis=1)[::-1, ::2]
        stamps = stamps[::-1]
        original_points, original_times = source.copy(), stamps.copy()
        absolute = lidar_pose(reference)
        expected = (landmarks[::-1] - absolute[:3, 3]) @ absolute[:3, :3]
        prediction = None if previous is None else np.linalg.inv(previous) @ absolute
        found = odometry.register(source, stamps, reference)
        manual = control.register(expected, relative_prediction=prediction)
        np.testing.assert_allclose(
            found, np.linalg.inv(first) @ absolute, atol=3e-9, rtol=0
        )
        np.testing.assert_allclose(found, manual, atol=3e-9, rtol=0)
        local = thin(expected, 0.015)
        placed = local @ found[:3, :3].T + found[:3, 3]
        np.testing.assert_allclose(odometry.placed(), placed, atol=3e-9, rtol=0)
        np.testing.assert_array_equal(source, original_points)
        np.testing.assert_array_equal(stamps, original_times)
        if previous is not None:
            np.testing.assert_allclose(
                odometry.last_prediction(), prediction, atol=1e-11, rtol=0
            )
        assert odometry.last_report()["input_points"] == len(source)
        previous = absolute
    np.testing.assert_array_equal(odometry.reference_timestamps_ns(), references)
    np.testing.assert_allclose(
        odometry.world_from_first_lidar, first, atol=1e-11, rtol=0
    )
    # Every exported array is independently owned and writable.
    for getter in [
        odometry.poses,
        odometry.placed,
        odometry.reference_timestamps_ns,
        odometry.last_prediction,
        lambda: odometry.world_from_first_lidar,
    ]:
        original = getter().copy()
        getter()[...] = -123
        np.testing.assert_array_equal(getter(), original)


@pytest.mark.parametrize("format", ["ros1", "cdr", "cdr-be", "sqlite"])
def test_shared_snapshot_lifetime_large_origin_and_empty_prediction_steps(
    tmp_path, format
):
    motion, extrinsic, lidar_pose = setup(tmp_path, format)
    initial = motion.state(motion.start_ns)
    (tmp_path / "shifted").mkdir()
    series, times, _, _ = data(tmp_path / "shifted", format)
    pose, _, gravity, bias, _ = configuration()
    pose[:3, 3] += [1e12, -2e12, 3e12]
    shifted = series.trajectory(
        motion.start_ns,
        motion.end_ns,
        pose,
        initial["velocity"],
        gravity=tuple(gravity),
        bias=tuple(bias),
    )
    a = core.ImuLidarOdometry(motion, extrinsic)
    b = core.ImuLidarOdometry(shifted, extrinsic)
    extrinsic[:] = 99
    del motion, shifted, series
    for base in [tmp_path, tmp_path / "shifted"]:
        for name in ["imu.bag", "imu"]:
            source = base / name
            if source.exists():
                target = base / ("moved_" + name)
                assert source.resolve().is_relative_to(tmp_path.resolve())
                assert target.resolve().is_relative_to(tmp_path.resolve())
                source.rename(target)
    empty, stamps = np.empty((0, 3)), np.empty(0, dtype=np.int64)
    references = [EPOCH + 50_000_000, EPOCH + 100_000_000, EPOCH + 199_999_979]
    first = lidar_pose(references[0])
    for reference in references:
        pa, pb = (
            a.register(empty, stamps, reference),
            b.register(empty, stamps, reference),
        )
        np.testing.assert_array_equal(pa.view(np.uint64), pb.view(np.uint64))
        np.testing.assert_allclose(
            pa, np.linalg.inv(first) @ lidar_pose(reference), atol=1e-11, rtol=0
        )
        assert a.placed().shape == (0, 3)
        assert a.last_report()["termination"] == "empty_map"
        assert a.last_iterations() == 0
    np.testing.assert_array_equal(a.reference_timestamps_ns(), references)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize(
    "bad",
    [
        "point_shape",
        "time_count",
        "nan_point",
        "early_point",
        "late_point",
        "early_reference",
        "late_reference",
        "duplicate_reference",
        "reverse_reference",
        "float_times",
    ],
)
def test_rejected_scan_cannot_change_map_pose_history_or_diagnostics(
    tmp_path, parallel, bad
):
    motion, extrinsic, _ = setup(tmp_path)
    settings = dict(min_range=0.0, map_voxel=0.1, diagnostics=True, parallel=parallel)
    a = core.ImuLidarOdometry(motion, extrinsic, **settings)
    control = core.ImuLidarOdometry(motion, extrinsic, **settings)
    points = np.random.default_rng(810).uniform(-6.0, 6.0, (120, 3))
    for reference in [EPOCH + 50_000_000, EPOCH + 100_000_000]:
        stamps = np.full(len(points), reference, dtype=np.int64)
        a.register(points, stamps, reference)
        control.register(points, stamps, reference)
    reference = EPOCH + 150_000_000
    stamps = np.full(len(points), reference, dtype=np.int64)
    candidate = points.copy()
    if bad == "point_shape":
        candidate = candidate[:, :2]
    elif bad == "time_count":
        stamps = stamps[:-1]
    elif bad == "nan_point":
        candidate[3, 1] = np.nan
    elif bad == "early_point":
        stamps[7] = EPOCH
    elif bad == "late_point":
        stamps[7] = motion.end_ns + 1
    elif bad == "early_reference":
        reference = EPOCH
    elif bad == "late_reference":
        reference = motion.end_ns + 1
    elif bad == "duplicate_reference":
        reference = EPOCH + 100_000_000
    elif bad == "reverse_reference":
        reference = EPOCH + 75_000_000
    elif bad == "float_times":
        stamps = stamps.astype(np.float64)
    getters = [
        a.poses,
        a.placed,
        a.reference_timestamps_ns,
        a.last_prediction,
        lambda: a.world_from_first_lidar,
    ]
    arrays = [getter().copy() for getter in getters]
    report = a.last_report()
    with pytest.raises((ValueError, TypeError)):
        a.register(candidate, stamps, reference)
    for getter, before in zip(getters, arrays):
        np.testing.assert_array_equal(getter(), before)
    assert a.last_report() == report
    # Also check the map/threshold state through the next valid registration.
    stamps = np.full(len(points), EPOCH + 150_000_000, dtype=np.int64)
    np.testing.assert_allclose(
        a.register(points + [0.01, -0.02, 0.03], stamps, EPOCH + 150_000_000),
        control.register(points + [0.01, -0.02, 0.03], stamps, EPOCH + 150_000_000),
        atol=1e-10,
        rtol=0,
    )


@pytest.mark.parametrize(
    "bad",
    [
        "shape",
        "homogeneous",
        "reflection",
        "nonrigid",
        "nan_transform",
        "min_range",
        "min_nan",
        "max_range",
        "max_inf",
        "voxel_zero",
        "voxel_nan",
        "map_points",
    ],
)
def test_constructor_rejects_invalid_calibration_and_settings(tmp_path, bad):
    motion, extrinsic, _ = setup(tmp_path)
    kwargs = {}
    if bad == "shape":
        extrinsic = extrinsic[:3]
    elif bad == "homogeneous":
        extrinsic[3, 0] = 0.01
    elif bad == "reflection":
        extrinsic[:3, 0] *= -1
    elif bad == "nonrigid":
        extrinsic[0, 0] += 0.1
    elif bad == "nan_transform":
        extrinsic[1, 3] = np.nan
    else:
        kwargs = {
            "min_range": {"min_range": -1},
            "min_nan": {"min_range": np.nan},
            "max_range": {"max_range": 1.5},
            "max_inf": {"max_range": np.inf},
            "voxel_zero": {"map_voxel": 0},
            "voxel_nan": {"map_voxel": np.nan},
            "map_points": {"map_points": 0},
        }[bad]
    with pytest.raises(ValueError):
        core.ImuLidarOdometry(motion, extrinsic, **kwargs)


def test_rejected_first_scan_leaves_no_reference_gauge_or_map(tmp_path):
    motion, extrinsic, _ = setup(tmp_path)
    odometry = core.ImuLidarOdometry(motion, extrinsic, diagnostics=True)
    with pytest.raises(ValueError, match="point timestamp"):
        odometry.register(
            np.ones((1, 3)), np.array([EPOCH], dtype=np.int64), EPOCH + 50_000_000
        )
    assert (
        odometry.world_from_first_lidar
        is odometry.last_prediction()
        is odometry.last_report()
        is None
    )
    assert odometry.poses().shape == (0, 4, 4)
    assert odometry.placed().shape == (0, 3)
    assert odometry.reference_timestamps_ns().size == 0


def test_range_filter_and_first_per_voxel_thinning_use_compensated_coordinates(
    tmp_path,
):
    motion, extrinsic, lidar_pose = setup(tmp_path)
    odometry = core.ImuLidarOdometry(
        motion, extrinsic, min_range=1.5, max_range=4.0, map_voxel=0.2
    )
    reference = motion.end_ns
    at_reference = np.array(
        [
            [1.49, 0.02, 0.03],
            [1.51, 0.02, 0.03],
            [1.5101, 0.02, 0.03],
            [3.99, 0.02, 0.03],
            [4.01, 0.02, 0.03],
            [2.02, 1.02, 0.03],
            [2.04, 1.02, 0.03],
        ]
    )
    world = lidar_pose(reference)
    landmarks = at_reference @ world[:3, :3].T + world[:3, 3]
    previous = lidar_pose(motion.start_ns)
    source = (landmarks - previous[:3, 3]) @ previous[:3, :3]
    stamps = np.full(len(source), motion.start_ns, dtype=np.int64)
    np.testing.assert_array_equal(
        odometry.register(source, stamps, reference), np.eye(4)
    )
    expected = thin(at_reference, 0.1, 1.5, 4.0)
    assert len(expected) == 3
    np.testing.assert_allclose(odometry.placed(), expected, atol=3e-11, rtol=0)
