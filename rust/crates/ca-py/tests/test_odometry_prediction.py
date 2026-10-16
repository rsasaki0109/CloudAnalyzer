"""Rigid-frame semantics, initialization and atomic validation of ICP prediction."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

core = pytest.importorskip("cloudanalyzer_core")


def test_prediction_seeds_icp_across_a_motion_outside_the_initial_search_radius():
    points = np.random.default_rng(918).uniform(-10., 10., (120, 3))
    motion = np.eye(4)
    motion[:3, :3] = Rotation.from_rotvec([.1, -.2, .35]).as_matrix()
    motion[:3, 3] = [3., -2., .7]
    scan = (points - motion[:3, 3]) @ motion[:3, :3]
    odometry = core.LidarOdometry(min_range=0., map_voxel=.02)
    np.testing.assert_array_equal(odometry.register(points), np.eye(4))
    found = odometry.register(scan, relative_prediction=motion)
    np.testing.assert_allclose(found, motion, atol=1e-12, rtol=0.)
    assert odometry.last_iterations() == 1
    # A third, empty scan exposes multiplication order without ICP correction.
    delta = np.eye(4)
    delta[:3, 3] = [.7, -.2, .3]
    np.testing.assert_allclose(odometry.register(np.empty((0, 3)), relative_prediction=delta), motion @ delta, atol=1e-12)


def test_prediction_validation_is_atomic_and_none_preserves_default_behavior():
    points = np.random.default_rng(352).uniform(-6., 6., (200, 3))
    odometry, control = core.LidarOdometry(), core.LidarOdometry()
    with pytest.raises(ValueError, match="previously"):
        odometry.register(points, relative_prediction=np.eye(4))
    assert odometry.poses().shape == (0, 4, 4)
    odometry.register(points)
    control.register(points)
    before = odometry.poses().copy()
    bad_inputs = [np.eye(3), np.eye(4) * 2]
    for i, j, value in [(0, 0, -1.), (0, 1, .1), (1, 3, np.nan), (2, 2, np.inf), (3, 0, .01)]:
        bad = np.eye(4)
        bad[i, j] = value
        bad_inputs.append(bad)
    for bad in bad_inputs:
        with pytest.raises(ValueError):
            odometry.register(points, relative_prediction=bad)
        np.testing.assert_array_equal(odometry.poses(), before)
        assert odometry.last_iterations() == 0
    np.testing.assert_array_equal(odometry.register(points, relative_prediction=None), control.register(points))
