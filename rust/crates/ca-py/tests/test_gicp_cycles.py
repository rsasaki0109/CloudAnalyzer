"""Explicit cycle-guard API and unchanged convergence/error semantics."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pytest

core = pytest.importorskip("cloudanalyzer_core")
sys.path.insert(0, str(Path(__file__).resolve().parents[4]/"scripts"))
import audit_flatwall_gicp_registration as reference


def data():
    source = np.concatenate([np.eye(3)*2., -np.eye(3)*2.])
    shapes = np.tile(np.diag([.1, 1., 2.]), (len(source), 1, 1))
    return [source, source+[.1, -.03, .04], shapes, shapes.copy(), np.eye(4), np.zeros(3)]


@pytest.mark.parametrize("robust_scale", [None, .1, 2.])
@pytest.mark.parametrize("capture_history", [False, True])
def test_guard_preserves_full_results_on_convergence_and_shared_target(robust_scale, capture_history):
    values = data()
    params = dict(max_correspondence_distance=1., robust_scale=robust_scale, capture_history=capture_history)
    baseline = core.gicp_register(*values, **params)
    guard = core.GicpCycleGuard()
    assert baseline["termination"] == "converged_step" and "cycle" not in baseline
    reference.equal_bits(core.gicp_register(*values, cycle_guard=None, **params), baseline)
    for parallel in [False, True]:
        reference.equal_bits(core.gicp_register(*values, cycle_guard=guard, parallel=parallel, **params), baseline)
    target = core.GicpTarget(values[1], values[3])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = pool.map(lambda _: target.register(values[0], values[2], values[4], values[5], cycle_guard=guard, **params), range(8))
        for result in results: reference.equal_bits(result, baseline)
    with pytest.raises(AttributeError): guard.max_period = 16


@pytest.mark.parametrize("case", ["empty_source", "empty_target", "two", "line", "limit", "nonfinite"])
def test_existing_failure_order_and_last_pose_are_preserved(case):
    values, params = data(), dict(max_correspondence_distance=1., capture_history=True)
    if case == "empty_source": values[0], values[2] = np.empty((0, 3)), np.empty((0, 3, 3))
    elif case == "empty_target": values[1], values[3] = np.empty((0, 3)), np.empty((0, 3, 3))
    elif case == "two": values[0], values[2] = values[0][:2], values[2][:2]
    elif case == "line":
        values[0][:] = values[1][:] = 0.
        values[0][:, 0] = values[1][:, 0] = np.arange(6.)
    elif case == "limit": params["max_iterations"] = 1
    else: values[0] *= 1e307
    expected = core.gicp_register(*values, **params)
    actual = core.gicp_register(*values, cycle_guard=core.GicpCycleGuard(), **params)
    reference.equal_bits(actual, expected)
    assert not actual["converged"] and "cycle" not in actual


def test_no_descent_is_not_reclassified_and_short_trials_do_not_converge():
    source = np.concatenate([np.eye(3), -np.eye(3)])
    pose = np.eye(4)
    pose[:3, 3] = 1e6
    target = source+pose[:3, 3]
    target[0, 0] = np.nextafter(target[0, 0], np.inf)
    shapes = np.tile(np.eye(3), (len(source), 1, 1))
    params = dict(max_correspondence_distance=.5, translation_tolerance=1e-20,
                  rotation_tolerance=1e-20, capture_history=True)
    guard = core.GicpCycleGuard(translation_tolerance=1e-24, rotation_tolerance=1e-24)
    expected = core.gicp_register(source, target, shapes, shapes, pose, pose[:3, 3], **params)
    actual = core.gicp_register(source, target, shapes, shapes, pose, pose[:3, 3], cycle_guard=guard, **params)
    reference.equal_bits(actual, expected)
    assert actual["termination"] == "no_descent" and not actual["converged"]


@pytest.mark.parametrize("params", [dict(max_period=1), dict(max_period=17), dict(confirmations=1),
    dict(confirmations=5), dict(translation_tolerance=0.), dict(rotation_tolerance=np.nan),
    dict(relative_shape_tolerance=np.inf), dict(weight_tolerance=-1.)])
def test_invalid_guard_config_rejected_at_construction(params):
    with pytest.raises(ValueError): core.GicpCycleGuard(**params)


def test_common_validation_precedes_guard_then_source_validation():
    values = data()
    values[0][0, 0] = np.nan
    guard = core.GicpCycleGuard()
    params = dict(max_correspondence_distance=1., translation_tolerance=1e-12, cycle_guard=guard)
    with pytest.raises(ValueError, match="cycle pose tolerances"):
        core.gicp_register(*values, **params)
    target = core.GicpTarget(values[1], values[3])
    with pytest.raises(ValueError, match="cycle pose tolerances"):
        target.register(values[0], values[2], values[4], values[5], **params)
    values[4][0, 3] = np.nan
    with pytest.raises(ValueError, match="initial_pose"):
        core.gicp_register(*values, **params)
