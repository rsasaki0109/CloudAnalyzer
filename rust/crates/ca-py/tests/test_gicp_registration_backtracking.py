"""Thin geometry control requiring backtracking and explicit descent failure."""
from pathlib import Path
import sys

import numpy as np
import pytest

core = pytest.importorskip("cloudanalyzer_core")
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
import audit_flatwall_gicp_registration as reference  # noqa: E402


def test_shortened_step_is_accepted_without_claiming_convergence():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(7, 3))*[1., .01, .01]
    target = source + rng.normal(size=(7, 3))*.5
    normals = rng.normal(size=(2, 7, 3))
    normals /= np.linalg.norm(normals, axis=2)[..., None]
    shapes = np.eye(3) - .999*np.einsum("kni,knj->knij", normals, normals)
    args = (source, target, shapes[0], shapes[1], np.eye(4), np.zeros(3))
    kwargs = dict(max_correspondence_distance=3., max_iterations=3,
                  translation_tolerance=1e-12, rotation_tolerance=1e-12)
    actual = core.gicp_register(*args, capture_history=True, **kwargs)
    reference.compare(actual, reference.reference(*args, **kwargs))
    reference.equal_bits(actual, core.gicp_register(*args, capture_history=True, parallel=True, **kwargs))
    assert [step["accepted_scale"] for step in actual["history"]] == [1., .0625, None]
    shortened = actual["history"][1]
    assert all(cost >= shortened["cost_before"] for cost in shortened["trial_costs"][:-1])
    assert shortened["trial_costs"][-1] == shortened["cost_after"] < shortened["cost_before"]
    assert actual["termination"] == "no_descent" and not actual["converged"]
    assert actual["accepted_steps"] == 2
    np.testing.assert_array_equal(actual["pose"], shortened["pose_after"])
