"""Independent registration oracle controls, without importing native code."""
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import audit_flatwall_gicp_registration as audit  # noqa: E402


def test_brute_force_match_threshold_and_weights():
    s = np.array([[0., 0., 0.], [5., 0., 0.]])
    t = np.array([[1., 0., 0.], [-1., 0., 0.]])
    assert not len(audit.nearest(s, t, np.eye(4), 1.)[0])
    si, ti, d2, w = audit.nearest(s, t, np.eye(4), 2., 2.)
    np.testing.assert_array_equal(si, [0])
    np.testing.assert_array_equal(ti, [0])
    np.testing.assert_array_equal(d2, [1.])
    np.testing.assert_allclose(w, [(4./5.)**2])


def test_frozen_cost_closed_form_and_rotation_center():
    source, target = np.array([[1., 2., 3.]]), np.zeros((1, 3))
    cost, _ = audit.frozen_cost(source, target, np.array([np.diag([.5, 1., 2.])]), np.eye(4), np.array([2.]))
    assert cost == 10.5
    pose = np.eye(4)
    pose[:3, 3] = [8., -3., 1.]
    result = audit.retract(pose, np.array([7., -1., 2.]), np.array([.2, -.1, .3, 0., 0., np.pi/2]), 1.)
    np.testing.assert_allclose(result[:3, 3], [9.2, -.1, 1.3], atol=1e-14)


def test_reference_recovers_rigid_control_and_records_frozen_trials():
    s = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]])
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_rotvec([.03, -.02, .04]).as_matrix()
    truth[:3, 3] = [.1, -.05, .07]
    t = s @ truth[:3, :3].T + truth[:3, 3]
    shape = np.tile(np.eye(3), (4, 1, 1))
    result = audit.reference(s, t, shape, shape, np.eye(4), np.zeros(3),
                             translation_tolerance=1e-10, rotation_tolerance=1e-10)
    assert result["termination"] == "converged_step"
    np.testing.assert_allclose(result["pose"], truth, atol=1e-9)
    assert all(row["cost_after"] < row["cost_before"] for row in result["history"] if row["accepted_scale"] is not None)


def test_empty_and_rank_deficient_reference_remain_explicit():
    empty, shapes = np.empty((0, 3)), np.empty((0, 3, 3))
    result = audit.reference(empty, empty, shapes, shapes, np.eye(4), np.zeros(3))
    assert result["termination"] == "insufficient_correspondences"
    assert result["final_cost"] == 0.
    line = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    shape = np.tile(np.eye(3), (3, 1, 1))
    result = audit.reference(line, line, shape, shape, np.eye(4), np.zeros(3))
    assert result["termination"] == "singular_system"
