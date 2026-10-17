"""Immutable target ownership, validation and exact registration equivalence."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import gc
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

core = pytest.importorskip("cloudanalyzer_core")
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
import audit_flatwall_gicp_registration as reference  # noqa: E402


def data():
    source = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.], [2., 3., 4.]])
    q = Rotation.from_rotvec([.03, -.02, .04]).as_matrix()
    target = source @ q.T + [.1, -.05, .07]
    cs = np.tile(np.diag([.01, .7, 1.]), (len(source), 1, 1))
    ct = np.tile(np.diag([.8, .02, 1.]), (len(target), 1, 1))
    ct[0, 0, 1] = 2e-14  # valid asymmetric input must follow original canonicalization
    return [source, target, cs, ct, np.eye(4), np.array([.3, -.2, .1])]


def options(**kwargs):
    return dict(max_correspondence_distance=2., capture_history=True, **kwargs)


def register(target, values, **kwargs):
    return target.register(values[0], values[2], values[4], values[5], **options(**kwargs))


@pytest.mark.parametrize("robust_scale", [None, .1, 2.])
@pytest.mark.parametrize("history", [False, True])
def test_all_results_history_and_reuse_equal_standalone(robust_scale, history):
    v = data()
    params = dict(max_correspondence_distance=2., capture_history=history, robust_scale=robust_scale)
    expected = core.gicp_register(*v, **params)
    target = core.GicpTarget(v[1], v[3])
    assert len(target) == len(v[1])
    for parallel in [False, True, False]:
        actual = target.register(v[0], v[2], v[4], v[5], parallel=parallel, **params)
        reference.equal_bits(actual, expected)


def test_constructor_copies_strided_inputs_and_results_own_storage():
    v = data()
    v[1], v[3] = v[1][::-1], v[3][::-1]
    v[0], v[2] = v[0][::-1], v[2][::-1]
    expected = core.gicp_register(*v, **options())
    target = core.GicpTarget(v[1], v[3])
    v[1][:], v[3][:] = np.nan, 0.
    del v[1], v[2]  # original target and target_shapes (after first deletion)
    gc.collect()
    source, cs, pose, center = v
    first = target.register(source, cs, pose, center, **options())
    reference.equal_bits(first, expected)
    first["pose"][:] = 0.
    first["history"][0]["combined_shapes"][:] = 0.
    first["matched_target_indices"][:] = 999
    reference.equal_bits(target.register(source, cs, pose, center, **options()), expected)
    with pytest.raises(AttributeError):
        target.points = np.zeros((1, 3))


def test_shared_read_only_calls_release_gil_and_keep_full_outputs():
    v = data()
    target = core.GicpTarget(v[1], v[3])
    expected = register(target, v)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda i: register(target, v, parallel=bool(i % 2)), range(12)):
            reference.equal_bits(result, expected)


@pytest.mark.parametrize("case", ["empty_source", "empty_target", "no_matches", "two", "line", "coincident", "limit", "nonfinite", "ties", "strict"])
def test_all_termination_and_tie_cases_preserve_full_outputs(case):
    v, params = data(), options()
    if case == "empty_source":
        v[0], v[2] = np.empty((0, 3)), np.empty((0, 3, 3))
    elif case == "empty_target":
        v[1], v[3] = np.empty((0, 3)), np.empty((0, 3, 3))
    elif case == "no_matches":
        v[1] += 100.
    elif case == "two":
        v[0], v[2] = v[0][:2], v[2][:2]
    elif case in ["line", "coincident"]:
        v[0], v[1] = np.zeros((5, 3)), np.zeros((5, 3))
        if case == "line": v[0][:, 0] = v[1][:, 0] = np.arange(5.)
    elif case == "limit":
        params["max_iterations"] = 1
    elif case == "nonfinite":
        v[0] *= 1e307
    else:
        v[0], v[1] = np.zeros((1, 3)), np.array([[1., 0., 0.], [-1., 0., 0.]])
        v[2], v[3] = np.eye(3)[None], np.tile(np.eye(3), (2, 1, 1))
        if case == "strict": params["max_correspondence_distance"] = 1.
    expected = core.gicp_register(*v, **params)
    target = core.GicpTarget(v[1], v[3])
    for parallel in [False, True]:
        reference.equal_bits(target.register(v[0], v[2], v[4], v[5], parallel=parallel, **params), expected)


@pytest.mark.parametrize("kind", ["point_nan", "point_inf", "shape_nan", "zero", "indefinite", "asymmetric"])
def test_constructor_validates_unmatched_rows_and_first_error(kind):
    v = data()
    if kind.startswith("point"):
        v[1][2, 0] = np.nan if kind == "point_nan" else np.inf
    elif kind == "shape_nan": v[3][2, 0, 0] = np.nan
    elif kind == "zero": v[3][2] = 0.
    elif kind == "indefinite": v[3][2, 0, 0] = -1.
    else: v[3][2, 0, 1] = .1
    v[1][4, 0] = np.nan
    with pytest.raises(ValueError, match="pair 2: invalid target point or shape"):
        core.GicpTarget(v[1], v[3])


@pytest.mark.parametrize("name,value", [("max_correspondence_distance", 0.), ("max_correspondence_distance", np.inf),
    ("max_correspondence_distance", 1e308), ("max_correspondence_distance", 1e-300), ("max_iterations", 0),
    ("translation_tolerance", -1.), ("rotation_tolerance", np.nan), ("robust_scale", 0.),
    ("robust_scale", -1.), ("robust_scale", np.inf), ("robust_scale", 1e308), ("robust_scale", 1e-300)])
def test_per_call_parameter_validation_matches_standalone(name, value):
    v, params = data(), options()
    params[name] = value
    target = core.GicpTarget(v[1], v[3])
    with pytest.raises(ValueError) as standalone:
        core.gicp_register(*v, **params)
    with pytest.raises(ValueError) as prepared:
        target.register(v[0], v[2], v[4], v[5], **params)
    assert str(prepared.value) == str(standalone.value)


def test_per_call_source_errors_pose_errors_and_shapes():
    v = data()
    target = core.GicpTarget(v[1], v[3])
    with pytest.raises(ValueError, match="target_shapes needs shape"):
        core.GicpTarget(v[1], v[3][:2])
    with pytest.raises(ValueError, match="source_shapes needs shape"):
        target.register(v[0], v[2][:2], v[4], v[5], **options())
    with pytest.raises(ValueError, match="center_world needs shape"):
        target.register(v[0], v[2], v[4], np.zeros(2), **options())
    v[2][1] = 0.
    with pytest.raises(ValueError, match="pair 1: invalid source point or shape"):
        register(target, v)
    v[4][0, 0] = 2.
    with pytest.raises(ValueError, match="initial_pose must be proper rigid"):
        register(target, v)
