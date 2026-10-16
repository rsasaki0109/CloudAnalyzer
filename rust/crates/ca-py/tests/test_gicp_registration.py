"""Iterative GICP contracts and independent NumPy/SciPy controls."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

core = pytest.importorskip("cloudanalyzer_core")
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
import audit_flatwall_gicp_registration as reference  # noqa: E402


def data():
    source = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.], [2., 3., 4.]])
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_rotvec([.03, -.02, .04]).as_matrix()
    truth[:3, 3] = [.1, -.05, .07]
    target = source @ truth[:3, :3].T + truth[:3, 3]
    cs = np.tile(np.diag([.01, .7, 1.]), (len(source), 1, 1))
    ct = np.tile(np.diag([.8, .02, 1.]), (len(source), 1, 1))
    return source, target, cs, ct, np.eye(4), np.array([.3, -.2, .1]), truth


def call(values, **kwargs):
    defaults = dict(max_correspondence_distance=2., capture_history=True)
    defaults.update(kwargs)
    return core.gicp_register(*values[:6], **defaults)


@pytest.mark.parametrize("robust_scale", [None, .1, 2.])
def test_all_iterations_match_independent_solver_and_recover_rigid_pose(robust_scale):
    values = data()
    kwargs = dict(robust_scale=robust_scale, translation_tolerance=1e-10, rotation_tolerance=1e-10)
    actual = call(values, **kwargs)
    expected = reference.reference(*values[:6], max_correspondence_distance=2., **kwargs)
    reference.compare(actual, expected)
    reference.equal_bits(actual, call(values, parallel=True, **kwargs))
    assert actual["termination"] == "converged_step"
    np.testing.assert_allclose(actual["pose"], values[6], atol=1e-9)
    for step in actual["history"]:
        if step["accepted_scale"] is not None:
            assert step["cost_after"] < step["cost_before"]
        for scale, cost in zip(step["trial_scales"], step["trial_costs"]):
            pose = reference.retract(step["pose_before"], values[5], step["step"], scale)
            si, ti = step["source_indices"], step["target_indices"]
            frozen = core.gicp_frozen_cost(values[0][si], values[1][ti], step["combined_shapes"], pose, weights=step["weights"])
            np.testing.assert_allclose(cost, frozen["cost"], atol=1e-12)


def test_candidate_cost_holds_anisotropic_metric_fixed_and_agrees_with_derivative():
    values = data()
    source, target, cs, ct, pose, center = values[:6]
    weights = np.array([0., .2, 1., 2., .4])
    linear = core.gicp_linearization(*values[:6], weights=weights)
    candidate = reference.retract(pose, center, np.array([.02, -.01, .03, .2, -.1, .3]), 1.)
    frozen = core.gicp_frozen_cost(source, target, linear["combined_shapes"], candidate, weights=weights)
    expected, mahal = reference.frozen_cost(source, target, linear["combined_shapes"], candidate, weights)
    np.testing.assert_allclose(frozen["cost"], expected, rtol=1e-13)
    np.testing.assert_allclose(frozen["mahalanobis_squared"], mahal, rtol=1e-13)
    refreshed = core.gicp_linearization(source, target, cs, ct, candidate, center, weights=weights)
    assert abs(refreshed["cost"]-frozen["cost"]) > .01
    reference.equal_bits(frozen, core.gicp_frozen_cost(source, target, linear["combined_shapes"], candidate, weights=weights, parallel=True))
    eps = 1e-6
    gradient = []
    for axis in range(6):
        delta = np.zeros(6)
        delta[axis] = eps
        costs = [core.gicp_frozen_cost(source, target, linear["combined_shapes"], reference.retract(pose, center, delta, sign), weights=weights)["cost"] for sign in [1., -1.]]
        gradient.append((costs[0]-costs[1])/(2*eps))
    np.testing.assert_allclose(gradient, -linear["right_hand_side"], atol=2e-8)


@pytest.mark.parametrize("origin", [np.zeros(3), np.array([1e6, -2e6, 3e6])])
def test_world_basis_and_origin_changes_preserve_registration(origin):
    values = list(data())
    baseline = call(values)
    q = Rotation.from_rotvec([.2, -.3, .4]).as_matrix()
    frame = np.eye(4)
    frame[:3, :3], frame[:3, 3] = q, origin
    values[1] = values[1] @ q.T + origin
    values[3] = q @ values[3] @ q.T
    values[4] = frame @ values[4]
    values[5] = q @ values[5] + origin
    actual = call(values)
    assert actual["termination"] == baseline["termination"]
    np.testing.assert_allclose(actual["pose"], frame @ baseline["pose"], atol=2e-7)


def test_source_axes_and_lever_arm_change_preserve_world_alignment():
    values = list(data())
    baseline = call(values)
    frame = np.eye(4)
    frame[:3, :3] = Rotation.from_rotvec([-.3, .2, .1]).as_matrix()
    frame[:3, 3] = [2., -1., .5]
    values[0] = values[0] @ frame[:3, :3].T + frame[:3, 3]
    values[2] = frame[:3, :3] @ values[2] @ frame[:3, :3].T
    values[4] = values[4] @ np.linalg.inv(frame)
    actual = call(values)
    np.testing.assert_allclose(actual["pose"] @ frame, baseline["pose"], atol=1e-10)


@pytest.mark.parametrize("case", ["empty_source", "empty_target", "no_matches", "two_matches", "line", "coincident", "limit", "nonfinite"])
def test_explicit_termination_and_last_accepted_pose(case):
    values = list(data())
    kwargs = {}
    status = "insufficient_correspondences"
    if case == "empty_source":
        values[0], values[2] = np.empty((0, 3)), np.empty((0, 3, 3))
    elif case == "empty_target":
        values[1], values[3] = np.empty((0, 3)), np.empty((0, 3, 3))
    elif case == "no_matches":
        values[1] += 100.
    elif case == "two_matches":
        values[0], values[2] = values[0][:2], values[2][:2]
    elif case in ["line", "coincident"]:
        values[0] = values[1] = np.zeros((5, 3))
        if case == "line":
            values[0][:, 0] = np.arange(5.)
        status = "singular_system"
    elif case == "limit":
        kwargs["max_iterations"], status = 1, "max_iterations"
    else:
        values[0] *= 1e307
        status = "nonfinite"
    result = call(values, **kwargs)
    assert result["termination"] == status
    assert not result["converged"]
    if case != "limit":
        np.testing.assert_array_equal(result["pose"], values[4])
        assert result["accepted_steps"] == 0
    else:
        assert result["accepted_steps"] == 1
        np.testing.assert_array_equal(result["pose"], result["history"][-1]["pose_after"])


@pytest.mark.parametrize("name,value", [("max_correspondence_distance", 0.), ("max_correspondence_distance", np.inf),
    ("max_correspondence_distance", 1e308), ("max_correspondence_distance", 1e-300), ("max_iterations", 0),
    ("translation_tolerance", -1.), ("rotation_tolerance", np.nan), ("robust_scale", 0.),
    ("robust_scale", -1.), ("robust_scale", np.inf), ("robust_scale", 1e308), ("robust_scale", 1e-300)])
def test_invalid_parameters_are_rejected(name, value):
    with pytest.raises(ValueError):
        call(data(), **{name: value})


@pytest.mark.parametrize("part", [0, 1, 2, 3, 4, 5])
def test_invalid_inputs_reject_even_unmatched_shapes(part):
    values = list(data())
    if part in [0, 1]:
        values[part][2, 1] = np.nan
    elif part in [2, 3]:
        values[part][2, 0, 0] = -1.
    elif part == 4:
        values[4][0, 0] = -1.
    else:
        values[5][0] = np.inf
    with pytest.raises(ValueError):
        call(values, max_correspondence_distance=1e-6)


def test_owned_strided_inputs_outputs_and_optional_history():
    values = data()
    inputs = [v.copy() for v in values[:6]]
    actual = call(values)
    strided = [v[::-1].copy()[::-1] for v in values[:6]]
    reference.equal_bits(actual, call(strided))
    plain = call(values, capture_history=False)
    assert plain["history"] == []
    reference.equal_bits({k:v for k,v in actual.items() if k != "history"}, {k:v for k,v in plain.items() if k != "history"})
    for a, b in zip(values, inputs):
        np.testing.assert_array_equal(a, b)
    actual["pose"][:] = 99.
    actual["history"][0]["combined_shapes"][:] = 99.
    reference.equal_bits(plain, call(values, capture_history=False))


def test_frozen_metric_empty_invalid_counts_pose_and_zero_weight_shape():
    empty, shapes = np.empty((0, 3)), np.empty((0, 3, 3))
    result = core.gicp_frozen_cost(empty, empty, shapes, np.eye(4))
    assert result["cost"] == 0. and result["mahalanobis_squared"].shape == (0,)
    source, target, cs, _, pose = data()[:5]
    with pytest.raises(ValueError):
        core.gicp_frozen_cost(source, target[:-1], cs, pose)
    with pytest.raises(ValueError):
        core.gicp_frozen_cost(source, target, cs, pose, weights=np.zeros(4))
    cs[0, 0, 0] = 0.
    with pytest.raises(ValueError, match="pair 0"):
        core.gicp_frozen_cost(source, target, cs, pose, weights=np.zeros(5))
    pose[3, 0] = 1.
    with pytest.raises(ValueError, match="homogeneous"):
        core.gicp_frozen_cost(empty, empty, shapes, pose)


def test_strict_threshold_and_deterministic_ties():
    values = list(data())
    values[0], values[1] = np.array([[0., 0., 0.]]), np.array([[1., 0., 0.], [-1., 0., 0.]])
    values[2], values[3] = values[2][:1], values[3][:2]
    assert not len(call(values, max_correspondence_distance=1.)["weights"])
    result = call(values)
    assert result["squared_distances"].tolist() == [1.]
    reference.equal_bits(result, call(values, parallel=True))
