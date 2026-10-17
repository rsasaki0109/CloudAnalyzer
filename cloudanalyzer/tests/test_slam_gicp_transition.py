"""Closed-form checks for fixed-support GICP transition accounting."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2]/"scripts/audit_flatwall_gicp_transition.py"
spec = importlib.util.spec_from_file_location("gicp_transition", SCRIPT)
transition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transition)


def ids(values):
    return np.asarray(values, dtype=np.int64)


def rotation():
    pose = np.eye(4)
    pose[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    return pose


def inputs():
    source = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    target = np.array([[0., 0., 0.], [2.5, 0., 0.]])
    cs, ct = np.tile(np.eye(3), (4, 1, 1)), np.tile(np.eye(3), (2, 1, 1))
    before, after = np.eye(4), np.eye(4)
    after[0, 3] = .25
    return [source, target, cs, ct, before, after, ids([0, 1]), ids([0, 0]), ids([1, 2]), ids([1, 1])]


def test_anisotropic_cube_matches_all_eight_closed_form_costs():
    source, target = np.array([[1., 0., 0.]]), np.array([[0., 1., 0.], [2., 0., 0.]])
    cs = np.diag([.01, 1., 2.])[None]
    ct = np.array([np.diag([.3, .4, .5]), np.diag([.7, .2, .8])])
    result = transition.transition_costs(source, target, cs, ct, np.eye(4), rotation(), ids([0]), ids([0]), ids([0]), ids([1]))
    expected = np.array([[[.5*(1/.31+1/1.4), 0.], [.5*(1/1.3+1/.41), 0.]],
                         [[.5/.71, .5*(4/.71+1/1.2)], [.5/1.7, .5*(4/1.7+1/.21)]]])
    np.testing.assert_allclose(result["cube"], expected, rtol=1e-14)
    np.testing.assert_allclose(result["common_delta"], expected[1, 1, 1]-expected[0, 0, 0])
    np.testing.assert_allclose(result["assignment_metric_interaction"],
                               (expected[1, 1, 1]-expected[0, 1, 1])-(expected[1, 0, 1]-expected[0, 0, 1]))
    assert abs(result["accounting_error"]) < 1e-14


def test_source_shape_refresh_can_reverse_frozen_cost_improvement():
    source, target = np.array([[1., 0., 0.]]), np.zeros((1, 3))
    cs, ct = np.diag([.001, 1., 1.])[None], np.diag([.01, .0001, 1.])[None]
    result = transition.transition_costs(source, target, cs, ct, np.eye(4), rotation(), ids([0]), ids([0]), ids([0]), ids([0]))
    np.testing.assert_allclose(result["baseline_cost"], .5/.011)
    np.testing.assert_allclose(result["frozen_after_cost"], .5/1.0001)
    np.testing.assert_allclose(result["refreshed_after_cost"], .5/.0011)
    assert result["motion_delta"] < 0. and result["source_metric_delta"] > 0.
    assert result["full_delta"] > 0. and result["assignment_delta"] == 0.


def test_added_removed_support_is_accounted_separately():
    result = transition.transition_costs(*inputs())
    np.testing.assert_array_equal(result["common_source_indices"], [1])
    np.testing.assert_array_equal(result["added_source_indices"], [2])
    np.testing.assert_array_equal(result["removed_source_indices"], [0])
    assert result["baseline_cost"] == .25
    assert result["added_cost"] == .015625 and result["removed_cost"] == 0.
    assert result["refreshed_after_cost"] == .40625
    assert result["full_delta"] == .15625 and result["accounting_error"] == 0.
    assert result["source_metric_delta"] == result["assignment_delta"] == 0.


@pytest.mark.parametrize("old,new", [([0], [2]), ([], [2]), ([0], []), ([], [])])
def test_empty_common_support_has_zero_cube_and_explicit_full_cost(old, new):
    values = inputs()
    values[6:] = [ids(old), ids([0]*len(old)), ids(new), ids([1]*len(new))]
    result = transition.transition_costs(*values)
    assert result["common_source_indices"].size == 0
    np.testing.assert_array_equal(result["cube"], np.zeros((2, 2, 2)))
    assert result["common_combined_shapes"].shape == (2, 2, 0, 3, 3)
    assert result["full_delta"] == result["added_cost"]-result["removed_cost"]
    assert result["accounting_error"] == 0.


def test_common_ids_keep_original_source_order_inputs_and_outputs_do_not_alias():
    values = inputs()
    values[6:] = [ids([2, 0, 1]), ids([1, 0, 0]), ids([1, 2, 0]), ids([1, 1, 0])]
    values[2][0, 0, 1] = 2e-14  # accepted canonical asymmetry
    before = [array.copy() for array in values]
    result = transition.transition_costs(*values)
    np.testing.assert_array_equal(result["common_source_indices"], [0, 1, 2])
    np.testing.assert_array_equal(result["common_target_indices"], [[0, 0, 1], [0, 1, 1]])
    result["common_combined_shapes"][:] = 0.
    for original, expected in zip(values, before):
        np.testing.assert_array_equal(original, expected)
    repeat = transition.transition_costs(*values)
    assert repeat["common_combined_shapes"].any()


@pytest.mark.parametrize("kind", ["point_nan", "shape_inf", "shape_zero", "shape_asymmetric", "pose_scale", "pose_reflection",
    "pose_bottom", "negative_source", "large_source", "large_target", "duplicate_source", "noninteger", "id_lengths", "shape_rows", "overflow"])
def test_invalid_geometry_poses_ids_and_arithmetic_fail_explicitly(kind):
    values = inputs()
    if kind == "point_nan": values[0][3, 0] = np.nan  # even unmatched input rows
    elif kind == "shape_inf": values[3][0, 0, 0] = np.inf
    elif kind == "shape_zero": values[2][3] = 0.
    elif kind == "shape_asymmetric": values[3][0, 0, 1] = .1
    elif kind == "pose_scale": values[4][0, 0] = 2.
    elif kind == "pose_reflection": values[5][0, 0] = -1.
    elif kind == "pose_bottom": values[5][3, 0] = .1
    elif kind == "negative_source": values[6][0] = -1
    elif kind == "large_source": values[6][0] = 4
    elif kind == "large_target": values[9][0] = 2
    elif kind == "duplicate_source": values[8][:] = 1
    elif kind == "noninteger": values[6] = np.array([0., 1.])
    elif kind == "id_lengths": values[7] = ids([0])
    elif kind == "shape_rows": values[2] = values[2][:2]
    else: values[0][1, 0] = 1e308
    with pytest.raises(ValueError):
        transition.transition_costs(*values)
