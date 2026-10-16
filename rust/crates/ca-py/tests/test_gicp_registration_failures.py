"""Numerical stopping controls at the representability boundary."""
import numpy as np
import pytest

core = pytest.importorskip("cloudanalyzer_core")


def test_no_descent_preserves_pose_when_full_step_is_below_world_ulp():
    source = np.concatenate([np.eye(3), -np.eye(3)])
    pose = np.eye(4)
    pose[:3, 3] = 1e6
    target = source + pose[:3, 3]
    target[0, 0] = np.nextafter(target[0, 0], np.inf)
    shapes = np.tile(np.eye(3), (len(source), 1, 1))
    result = core.gicp_register(source, target, shapes, shapes, pose, pose[:3, 3],
                               max_correspondence_distance=.5, translation_tolerance=1e-20,
                               rotation_tolerance=1e-20, capture_history=True)
    assert result["termination"] == "no_descent"
    assert result["accepted_steps"] == 0 and not result["converged"]
    np.testing.assert_array_equal(result["pose"], pose)
    step = result["history"][0]
    assert np.linalg.norm(step["step"][:3]) > 1e-20
    np.testing.assert_array_equal(step["trial_scales"], [1., .5, .25, .125, .0625, .03125])
    assert step["trial_costs"] == [step["cost_before"]]*6
    assert step["accepted_scale"] is None and step["cost_after"] is None


@pytest.mark.parametrize("parallel", [False, True])
def test_frozen_cost_retains_first_error_index_even_at_zero_weight(parallel):
    source, target = np.zeros((1025, 3)), np.zeros((1025, 3))
    shapes = np.tile(np.eye(3), (1025, 1, 1))
    weights = np.ones(1025)
    shapes[7, 2, 2], weights[7] = -1., 0.
    source[333, 0] = np.nan
    with pytest.raises(ValueError, match="pair 7: combined shape"):
        core.gicp_frozen_cost(source, target, shapes, np.eye(4), weights=weights, parallel=parallel)
