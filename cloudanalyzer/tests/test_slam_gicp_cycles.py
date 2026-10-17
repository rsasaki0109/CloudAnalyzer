"""Independent cycle witness controls, separate from estimator convergence."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import audit_flatwall_gicp_cycles as cycles


def state(phase):
    pose = np.eye(4)
    pose[0, 3] = phase*.01
    return dict(pose_before=pose, source_indices=np.array([0, 1, 2]), target_indices=np.array([0, phase+1, 2]),
                weights=np.ones(3), combined_shapes=np.tile(np.diag([.1, 1., 2.]), (3, 1, 1)))


@pytest.mark.parametrize("period", [2, 3, 5, 7, 8])
def test_every_phase_and_two_returns_are_required(period):
    history = [state(i % period) for i in range(3*period)]
    assert cycles.cycle_at(history[:-1], np.zeros(3)) is None
    result = cycles.cycle_at(history, np.zeros(3))
    assert result["period"] == period and result["compared_state_pairs"] == 2*period
    assert result["first_iteration"] == 1 and result["last_iteration"] == 3*period
    history[period]["weights"][0] = .5
    assert cycles.cycle_at(history, np.zeros(3)) is None


def test_bound_and_single_phase_are_not_enough():
    assert cycles.cycle_at([state(i % 9) for i in range(40)], np.zeros(3)) is None
    history = [state(i % 3) for i in range(9)]
    history[1]["target_indices"][0] = 99
    assert cycles.cycle_at(history, np.zeros(3)) is None


@pytest.mark.parametrize("change", ["drift", "shape", "weight", "support", "slow", "stationary"])
def test_repeating_ids_or_poses_alone_are_insufficient(change):
    history = [state(i % 3) for i in range(9)]
    for i, row in enumerate(history):
        if change == "drift": row["pose_before"][1, 3] = i*.001
        if change == "shape": row["combined_shapes"][0, 0, 0] += i*.01
        if change == "weight": row["weights"][0] -= i*.01
        if change == "support": row["source_indices"][0] = 10+i
        if change in ["slow", "stationary"]:
            row["pose_before"][0, 3] = i*1e-12 if change == "slow" else 0.
    assert cycles.cycle_at(history, np.zeros(3)) is None


def test_centered_motion_is_source_and_world_frame_invariant():
    from scipy.spatial.transform import Rotation
    a, b = np.eye(4), np.eye(4)
    a[:3, :3] = Rotation.from_rotvec([.1, -.2, .3]).as_matrix()
    a[:3, 3], b[:3, 3], center = [2., -1., .5], [-.5, .8, 1.], np.array([.4, -.7, 2.])
    expected = cycles.centered_distance(a, b, center)
    frame = np.eye(4)
    frame[:3, :3] = Rotation.from_rotvec([-.2, .3, .4]).as_matrix()
    frame[:3, 3] = [1e6, -2e6, 3e6]
    np.testing.assert_allclose(cycles.centered_distance(a@frame, b@frame, center), expected, atol=2e-9)
    np.testing.assert_allclose(cycles.centered_distance(frame@a, frame@b, frame[:3, :3]@center+frame[:3, 3]), expected, atol=2e-9)


def test_shape_comparison_is_relative_and_world_basis_invariant():
    from scipy.spatial.transform import Rotation
    a, b = state(1), state(1)
    b["combined_shapes"][0, 0, 0] += 1e-10
    expected = cycles.differences(a, b, np.zeros(3))[2]
    q = Rotation.from_rotvec([.3, -.2, .1]).as_matrix()
    for row in [a, b]: row["combined_shapes"] = q @ (row["combined_shapes"]*1e30) @ q.T
    assert cycles.differences(a, b, np.zeros(3))[2] == pytest.approx(expected, abs=1e-16)


@pytest.mark.parametrize("options", [dict(max_period=1), dict(max_period=17), dict(confirmations=1),
    dict(confirmations=5), dict(translation_tolerance=0.), dict(rotation_tolerance=np.nan),
    dict(relative_shape_tolerance=np.inf), dict(weight_tolerance=-1.), dict(translation_tolerance=1e-5)])
def test_invalid_configuration_rejects(options):
    with pytest.raises(ValueError): cycles.cycle_at([], np.zeros(3), **options)
