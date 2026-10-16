"""Numerical reporting checks; these fixtures are not trajectory benchmarks."""
import numpy as np
import pytest

core = pytest.importorskip("cloudanalyzer_core")


def scene():
    return np.array([[3. * (i % 4), 3. * (i // 4), float(i % 3)] for i in range(12)])


def test_reports_are_optional_copies_and_invalid_predictions_are_atomic():
    odometry = core.LidarOdometry(min_range=0., diagnostics=True)
    assert odometry.last_report() is None
    odometry.register(scene())
    first = odometry.last_report()
    assert first == {"termination": "empty_map", "iterations": 0, "input_points": 12,
                     "registration_points": 12, "threshold_m": 6., "correspondences": None}
    with pytest.raises(ValueError):
        odometry.register(scene(), relative_prediction=np.eye(4) * 2.)
    assert odometry.last_report() == first
    odometry.register(scene())
    report = odometry.last_report()
    assert report["termination"] == "converged"
    assert report["iterations"] == odometry.last_iterations() == 1
    assert report["correspondences"] == {"matched_points": 12, "matched_fraction": 1.,
                                         "rmse_m": 0., "max_residual_m": 0.}
    report["termination"] = "changed"
    report["correspondences"]["matched_points"] = -1
    assert odometry.last_report()["termination"] == "converged"
    assert odometry.last_report()["correspondences"]["matched_points"] == 12
    odometry.register(np.empty((0, 3)))
    assert odometry.last_report()["termination"] == "insufficient_points"
    assert odometry.last_report()["correspondences"] is None


def test_final_residuals_use_the_returned_pose_and_map_before_insertion():
    points = scene()
    odometry = core.LidarOdometry(min_range=0., diagnostics=True)
    odometry.register(points)
    scan = points.copy()
    scan[:, 0] += .10 + .003 * np.arange(len(scan))
    scan[:, 1] -= .02 * (np.arange(len(scan)) % 3)
    scan = np.vstack([scan, [35., 0., 0.]])
    pose = odometry.register(scan)
    report = odometry.last_report()
    stats = report["correspondences"]
    placed = scan @ pose[:3, :3].T + pose[:3, 3]
    distances = []
    for q in placed:
        # Independent brute-force search of the same 27-voxel stencil.
        allowed = np.all(np.abs(np.floor(points) - np.floor(q)) <= 1, axis=1)
        if np.any(allowed):
            distance_sq = np.min(np.sum((points[allowed] - q) ** 2, axis=1))
            if distance_sq <= report["threshold_m"] ** 2:
                distances.append(distance_sq)
    assert report["registration_points"] == len(scan)
    assert stats["matched_points"] == len(distances) == 12
    assert stats["matched_fraction"] == 12 / 13
    assert stats["rmse_m"] > 1e-3  # Inserting these points first would give zero.
    np.testing.assert_allclose(stats["rmse_m"], np.sqrt(np.mean(distances)), atol=1e-14, rtol=0.)
    np.testing.assert_allclose(stats["max_residual_m"], np.sqrt(max(distances)), atol=1e-14, rtol=0.)


def test_no_matches_and_degenerate_matches_have_explicit_failed_solve_status():
    for collinear in [False, True]:
        points = np.array([[3. * i, 0., 0.] for i in range(12)]) if collinear else scene()
        odometry = core.LidarOdometry(min_range=0., diagnostics=True)
        odometry.register(points)
        np.testing.assert_array_equal(odometry.register(points if collinear else points + [50., 0., 0.]), np.eye(4))
        report = odometry.last_report()
        assert report["termination"] == "singular_system"
        assert report["iterations"] == 0
        stats = report["correspondences"]
        assert stats["matched_points"] == (12 if collinear else 0)
        assert stats["rmse_m"] == (0. if collinear else None)
        assert stats["max_residual_m"] == (0. if collinear else None)


def test_enabling_diagnostics_preserves_pose_bits_and_prediction_behavior():
    plain = core.LidarOdometry(min_range=0.)
    measured = core.LidarOdometry(min_range=0., diagnostics=True)
    points = scene()
    for i in range(6):
        prediction = None
        if i:
            prediction = np.eye(4)
            prediction[:3, 3] = [.1, -.02, 0.]
        scan = points - [i * .1, -i * .02, 0.]
        np.testing.assert_array_equal(plain.register(scan, relative_prediction=prediction).view(np.uint64),
                                      measured.register(scan, relative_prediction=prediction).view(np.uint64))
        a, b = plain.last_report(), measured.last_report()
        assert a["correspondences"] is None
        b["correspondences"] = None
        assert a == b
