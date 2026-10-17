"""Transported-shape gradients and unchanged frozen-factor outputs."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

core = pytest.importorskip("cloudanalyzer_core")
sys.path.insert(0, str(Path(__file__).resolve().parents[4]/"scripts"))
import audit_flatwall_gicp_registration as registration
import gicp_transported_reference as independent


def data(n=5):
    source = np.array([[.3, -.5, .8], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.], [2., 3., 4.]])[:n]
    target = source+[.2, -.1, .05]
    cs = np.tile(np.diag([.1, .7, 1.]), (len(source), 1, 1))
    ct = np.tile(np.diag([.8, .02, 1.]), (len(source), 1, 1))
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec([.1, -.2, .03]).as_matrix()
    return [source, target, cs, ct, pose, np.array([.2, -.1, .4])]


def test_closed_form_nine_over_twenty_two_and_frozen_serialization():
    source, target = np.zeros((1, 3)), np.array([[-1., -1., 0.]])
    args = (source, target, np.diag([.1, 1., 2.])[None], np.eye(3)[None], np.eye(4), np.zeros(3))
    actual = core.gicp_transported_linearization(*args)
    frozen = core.gicp_linearization(*args)
    registration.equal_bits({k: actual[k] for k in frozen}, frozen)
    np.testing.assert_allclose(actual["metric_gradient_rotation"], [0., 0., 9./22.], atol=1e-15)
    np.testing.assert_array_equal(actual["objective_right_hand_side"], -actual["objective_gradient"])


@pytest.mark.parametrize("axis", range(6))
def test_full_objective_gradient_matches_independent_central_differences(axis):
    v = data()
    actual = core.gicp_transported_linearization(*v)
    metric = independent.metric_gradient(v[0], v[1], v[2], v[4], actual["combined_shapes"])
    np.testing.assert_allclose(actual["metric_gradient_rotation"], metric, atol=2e-13, rtol=1e-12)
    delta = np.zeros(6)
    delta[axis] = 1e-5
    costs = []
    for sign in [1., -1.]:
        pose = registration.retract(v[4], v[5], delta, sign)
        combined = v[3]+pose[:3, :3]@v[2]@pose[:3, :3].T
        costs.append(registration.frozen_cost(v[0], v[1], combined, pose, np.ones(len(v[0])))[0])
    assert (costs[0]-costs[1])/2e-5 == pytest.approx(actual["objective_gradient"][axis], abs=2e-8, rel=1e-8)


@pytest.mark.parametrize("source_frame", [False, True])
@pytest.mark.parametrize("origin", [np.zeros(3), np.array([1e6, -2e6, 3e6])])
def test_world_and_source_basis_origins_and_lever_arm_contract(source_frame, origin):
    v = data()
    baseline = core.gicp_transported_linearization(*v)
    q = Rotation.from_rotvec([.2, -.3, .4]).as_matrix()
    frame = np.eye(4)
    frame[:3, :3], frame[:3, 3] = q, origin
    if source_frame:
        v[0], v[2], v[4] = v[0]@q.T+origin, q@v[2]@q.T, v[4]@np.linalg.inv(frame)
        expected = baseline["objective_gradient"]
        metric = baseline["metric_gradient_rotation"]
    else:
        v[1], v[3], v[4], v[5] = v[1]@q.T+origin, q@v[3]@q.T, frame@v[4], q@v[5]+origin
        expected = np.concatenate([q@baseline["objective_gradient"][:3], q@baseline["objective_gradient"][3:]])
        metric = q@baseline["metric_gradient_rotation"]
    actual = core.gicp_transported_linearization(*v)
    np.testing.assert_allclose(actual["objective_gradient"], expected, atol=2e-7, rtol=1e-9)
    np.testing.assert_allclose(actual["metric_gradient_rotation"], metric, atol=2e-7, rtol=1e-9)
    assert actual["cost"] == pytest.approx(baseline["cost"], abs=2e-8)


def test_parallel_bits_strides_ownership_empty_and_isotropic_source():
    v = data()
    v[0], v[1], v[2], v[3] = [np.tile(a, (207,)+(1,)*(a.ndim-1))[::-1] for a in v[:4]]
    expected = core.gicp_transported_linearization(*v)
    registration.equal_bits(expected, core.gicp_transported_linearization(*v, parallel=True))
    frozen = core.gicp_linearization(*v)
    registration.equal_bits({k: expected[k] for k in frozen}, frozen)
    expected["metric_gradient_rotation"][:] = 999.
    expected["combined_shapes"][:] = 0.
    again = core.gicp_transported_linearization(*v)
    assert not np.any(again["metric_gradient_rotation"] == 999.)
    empty = [np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3, 3)), np.empty((0, 3, 3)), np.eye(4), np.zeros(3)]
    np.testing.assert_array_equal(core.gicp_transported_linearization(*empty)["objective_gradient"], np.zeros(6))
    v = data()
    v[2] = np.tile(np.eye(3), (len(v[0]), 1, 1))
    np.testing.assert_allclose(core.gicp_transported_linearization(*v)["metric_gradient_rotation"], np.zeros(3), atol=2e-14)


@pytest.mark.parametrize("part", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("parallel", [False, True])
def test_original_validation_and_first_errors_are_retained(part, parallel):
    v = data()
    if part in [0, 1]: v[part][2, 0] = np.nan
    elif part in [2, 3]: v[part][2, 0, 0] = -1.
    elif part == 4: v[4][0, 0] = 2.
    else: v[5][1] = np.inf
    with pytest.raises(ValueError) as expected: core.gicp_linearization(*v, parallel=parallel)
    with pytest.raises(ValueError) as actual: core.gicp_transported_linearization(*v, parallel=parallel)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("parallel", [False, True])
def test_new_gradient_overflow_does_not_change_frozen_factor(parallel):
    source = np.zeros((1025, 3))
    target = np.tile([-1., -1., 0.], (1025, 1))
    cs = np.tile(np.eye(3), (1025, 1, 1))
    ct = cs.copy()
    for index in [7, 333]:
        target[index] = [-1e100, -1e200, 0.]
        cs[index] = ct[index] = np.diag([.5e-100, .5e100, .5])
    args = (source, target, cs, ct, np.eye(4), np.zeros(3))
    assert np.isfinite(core.gicp_linearization(*args)["cost"])
    with pytest.raises(ValueError, match="pair 7: nonfinite transported metric gradient"):
        core.gicp_transported_linearization(*args, parallel=parallel)


@pytest.mark.parametrize("keyword", ["weights", "robust_scale"])
def test_external_weights_are_not_silently_differentiated(keyword):
    with pytest.raises(TypeError): core.gicp_transported_linearization(*data(), **{keyword: 1.})
