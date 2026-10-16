"""Sampling and insertion contracts; numerical fixtures are not accuracy benchmarks."""

import numpy as np
import pytest

core = pytest.importorskip("cloudanalyzer_core")


def points():
    return np.array(
        [
            [
                3 * (i % 4) - 6 + 0.03125 + 0.09375 * j,
                3 * (i // 4) + 2.03125,
                i % 3 + 0.03125,
            ]
            for i in range(12)
            for j in range(4)
        ],
        dtype=np.float64,
    )


def thin(source, voxel, minimum=0.0):
    seen, out = set(), []
    for point in source:
        if not minimum <= np.linalg.norm(point) <= 80.0:
            continue
        cell = tuple(np.floor(point / voxel).astype(np.int64))
        if cell not in seen:
            out.append(point)
            seen.add(cell)
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


def selected(source, voxel, minimum=0.0):
    return (
        thin(thin(source, 0.5, minimum), 1.5)
        if voxel is None
        else thin(source, voxel, minimum)
    )


@pytest.mark.parametrize("voxel", [None, 0.75, 0.125, 0.0625])
@pytest.mark.parametrize("strided", [False, True])
def test_independent_source_density_counts_and_final_residuals_use_unchanged_insertion(
    voxel, strided
):
    source = np.vstack([[0.0, 0.0, 0.0], points(), [100.0, 0.0, 0.0]])
    if strided:
        source = np.repeat(source, 2, axis=1)[::-1, ::2]
    original = source.copy()
    odometry = core.LidarOdometry(
        min_range=1.0, diagnostics=True, registration_voxel=voxel
    )
    assert odometry.registration_voxel == voxel
    np.testing.assert_array_equal(odometry.register(source), np.eye(4))
    target, registered = thin(source, 0.5, 1.0), selected(source, voxel, 1.0)
    assert len(registered) > len(target) if voxel == 0.0625 else len(registered) > 0
    assert odometry.last_report()["registration_points"] == len(registered)
    pose = odometry.register(source)
    report = odometry.last_report()
    world = registered @ pose[:3, :3].T + pose[:3, 3]
    distances = []
    for point in world:
        allowed = np.all(np.abs(np.floor(target) - np.floor(point)) <= 1, axis=1)
        if np.any(allowed):
            distance = np.min(np.sum((target[allowed] - point) ** 2, axis=1))
            if distance <= report["threshold_m"] ** 2:
                distances.append(distance)
    stats = report["correspondences"]
    assert report["registration_points"] == len(registered)
    assert stats["matched_points"] == len(distances) == len(registered)
    np.testing.assert_allclose(
        stats["rmse_m"], np.sqrt(np.mean(distances)), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        stats["max_residual_m"], np.sqrt(max(distances)), rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("deskew", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_omitted_and_none_keep_complete_default_pose_bits_and_reports(deskew, strided):
    omitted = core.LidarOdometry(min_range=0.0, deskew=deskew, diagnostics=True)
    explicit = core.LidarOdometry(
        min_range=0.0, deskew=deskew, diagnostics=True, registration_voxel=None
    )
    for index in range(5):
        scan = points() - [index * 0.0625, 0.0, 0.0]
        times = np.linspace(0.0, 1.0, len(scan), dtype=np.float32)
        if strided:
            scan = np.repeat(scan, 2, axis=1)[::-1, ::2]
            times = times[::-1].copy()  # Existing times require contiguous storage.
        np.testing.assert_array_equal(
            omitted.register(scan, times).view(np.uint64),
            explicit.register(scan, times).view(np.uint64),
        )
        assert omitted.last_report() == explicit.last_report()
    np.testing.assert_array_equal(
        omitted.poses().view(np.uint64), explicit.poses().view(np.uint64)
    )
    assert omitted.registration_voxel is explicit.registration_voxel is None


@pytest.fixture
def motion(tmp_path):
    pytest.importorskip("rosbags")
    pytest.importorskip("scipy")
    from test_imu_odometry import setup

    return setup(tmp_path)


@pytest.mark.parametrize("voxel", [None, 0.75, 0.125, 0.0625])
@pytest.mark.parametrize("parallel", [False, True])
def test_imu_frontend_samples_compensated_points_but_preserves_insertion(
    motion, voxel, parallel
):
    from test_imu_preintegration import EPOCH

    trajectory, extrinsic, lidar_pose = motion
    reference = EPOCH + 100_000_000
    expected = points()
    reference_pose = lidar_pose(reference)
    landmarks = expected @ reference_pose[:3, :3].T + reference_pose[:3, 3]
    stamps = reference - 30_000_000 + np.arange(len(expected), dtype=np.int64) * 625_000
    stamps[-1] = reference
    source = np.array(
        [
            (point - lidar_pose(int(stamp))[:3, 3]) @ lidar_pose(int(stamp))[:3, :3]
            for point, stamp in zip(landmarks, stamps)
        ]
    )
    source, stamps = np.repeat(source, 2, axis=1)[::-1, ::2], stamps[::-1]
    original, original_times = source.copy(), stamps.copy()
    expected = expected[::-1]
    odometry = core.ImuLidarOdometry(
        trajectory,
        extrinsic,
        min_range=0.0,
        diagnostics=True,
        parallel=parallel,
        registration_voxel=voxel,
    )
    assert odometry.registration_voxel == voxel
    np.testing.assert_array_equal(
        odometry.register(source, stamps, reference), np.eye(4)
    )
    assert odometry.last_report()["registration_points"] == len(
        selected(expected, voxel)
    )
    np.testing.assert_allclose(
        odometry.placed(), thin(expected, 0.5), rtol=0, atol=1e-10
    )
    np.testing.assert_array_equal(odometry.reference_timestamps_ns(), [reference])
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(stamps, original_times)


@pytest.mark.parametrize("voxel", [0.0, -0.0, -0.25, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("imu", [False, True])
def test_invalid_registration_voxel_is_rejected(motion, voxel, imu):
    trajectory, extrinsic, _ = motion
    args = (trajectory, extrinsic) if imu else ()
    constructor = core.ImuLidarOdometry if imu else core.LidarOdometry
    with pytest.raises(
        ValueError, match="registration_voxel must be finite and positive"
    ):
        constructor(*args, registration_voxel=voxel)


@pytest.mark.parametrize("imu", [False, True])
def test_python_registration_voxel_is_read_only(motion, imu):
    trajectory, extrinsic, _ = motion
    odometry = (
        core.ImuLidarOdometry(trajectory, extrinsic, registration_voxel=0.25)
        if imu
        else core.LidarOdometry(registration_voxel=0.25)
    )
    with pytest.raises(AttributeError):
        odometry.registration_voxel = 0.5
    assert odometry.registration_voxel == 0.25
