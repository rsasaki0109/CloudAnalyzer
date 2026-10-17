"""Independent derivative and rejected-policy controls, not performance data."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import audit_flatwall_gicp_registration as registration
import gicp_transported_reference as rematched
import gicp_transported_fixed_reference as fixed


def test_closed_form_metric_gradient():
    cs, ct = np.diag([.1, 1., 2.])[None], np.eye(3)[None]
    gradient = rematched.metric_gradient(np.zeros((1, 3)), np.array([[-1., -1., 0.]]), cs, np.eye(4), cs+ct)
    np.testing.assert_allclose(gradient, [0., 0., 9./22.], atol=1e-15)


@pytest.mark.parametrize("variant", [rematched, fixed])
def test_armijo_requires_same_literal_support_and_sufficient_decrease(variant):
    ids = np.array([0, 2, 3], dtype=np.uint64)
    assert variant.acceptable(ids, ids.copy(), 10., 9., 1., -100.)
    assert not variant.acceptable(ids, ids[:-1], 10., 1., 1., -100.)
    assert not variant.acceptable(ids, ids[::-1], 10., 1., 1., -100.)
    assert not variant.acceptable(ids, ids, 10., 9.999, 1., -100.)
    assert not variant.acceptable(ids, ids, 10., np.nan, 1., -100.)


@pytest.mark.parametrize("variant", [rematched, fixed])
def test_supported_isotropic_motion_keeps_full_direction_stopping_and_inputs(variant):
    source = np.concatenate([np.eye(3)*2., -np.eye(3)*2.])
    target = source+[.1, -.03, .04]
    shapes = np.tile(np.eye(3), (len(source), 1, 1))
    inputs = [source, target, shapes, shapes.copy(), np.eye(4), np.zeros(3)]
    before = [a.copy() for a in inputs]
    actual = variant.reference(*inputs)
    assert actual["termination"] == "converged_step" and actual["accepted_steps"] >= 1
    np.testing.assert_allclose(actual["pose"][:3, 3], [.1, -.03, .04], atol=1e-12)
    assert actual["history"][-1]["accepted_scale"] is None
    for a, b in zip(inputs, before): np.testing.assert_array_equal(a, b)
    for row in actual["history"]:
        if row["accepted_scale"] is not None:
            assert row["cost_after"] < row["cost_before"]
            assert row["trial_support_equal"][-1]
    with pytest.raises(ValueError, match="unit weights"): variant.reference(*inputs, robust_scale=1.)
