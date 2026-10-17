"""Analytic geometry, exact neighborhoods, invariance and finite differences."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("plane_objective", ROOT/"scripts/flatwall_plane_objective.py")
plane = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plane)


def fixture():
    x, y = np.meshgrid(np.linspace(-2, 2, 7), np.linspace(-2, 2, 7))
    target = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    source = target+[.4, -.2, .1]
    normal = np.tile([0., 0., 1.], (len(source), 1))
    return source, target, normal, np.linspace(.5, 1., len(source)), np.array([.2, -.3, .4])


def test_plane_reveals_three_null_directions_on_the_same_point_pairs():
    q, p, n, w, c = fixture()
    result = plane.systems(q, p, n, w, c)
    h = result["plane_normal_matrix"]
    np.testing.assert_array_equal(h[[0, 1, 5]], np.zeros((3, 6)))
    assert result["plane_geometry"]["weak_directions_below_1e_3"] == 3
    assert result["point_geometry"]["weak_directions_below_1e_3"] == 0
    np.testing.assert_allclose(result["plane_residual"], .1)
    assert result["plane_cost"] < result["point_cost"]
    assert result["plane_right_hand_side"][2] < 0


def test_centered_left_rotation_jacobian_and_gradient_match_finite_differences():
    q, p, n, w, c = fixture()
    result = plane.systems(q, p, n, w, c)
    def residual(delta):
        moved = (q-c)@Rotation.from_rotvec(delta[3:]).as_matrix().T+c+delta[:3]
        return np.sum(n*(moved-p), axis=1)
    numeric = []
    eps = 1e-6
    for i in range(6):
        delta = np.zeros(6)
        delta[i] = eps
        numeric.append((residual(delta)-residual(-delta))/(2*eps))
    jacobian = np.array(numeric).T
    np.testing.assert_allclose(result["plane_jacobian"], jacobian, rtol=0, atol=5e-10)
    np.testing.assert_allclose(result["plane_right_hand_side"], -jacobian.T@(w*residual(np.zeros(6))), atol=5e-9)
    np.testing.assert_allclose(result["plane_normal_matrix"], jacobian.T@(w[:, None]*jacobian), atol=2e-8)


def test_normals_sign_and_world_origin_do_not_change_the_system():
    q, p, n, w, c = fixture()
    original = plane.systems(q, p, n, w, c)
    shifted = plane.systems(q+[100, -200, 300], p+[100, -200, 300], -n, w, c+[100, -200, 300])
    for key in ["plane_normal_matrix", "plane_right_hand_side", "point_normal_matrix", "point_right_hand_side"]:
        np.testing.assert_allclose(shifted[key], original[key], rtol=0, atol=2e-10)


def test_rotating_world_coordinates_rotates_translation_and_rotation_blocks():
    q, p, n, w, c = fixture()
    r = Rotation.from_rotvec([.2, -.3, .4]).as_matrix()
    block = np.zeros((6, 6))
    block[:3, :3] = block[3:, 3:] = r
    original = plane.systems(q, p, n, w, c)
    rotated = plane.systems(q@r.T, p@r.T, n@r.T, w, c@r.T)
    np.testing.assert_allclose(rotated["plane_normal_matrix"], block@original["plane_normal_matrix"]@block.T, atol=2e-12)
    np.testing.assert_allclose(rotated["plane_right_hand_side"], block@original["plane_right_hand_side"], atol=2e-12)
    np.testing.assert_allclose(rotated["plane_geometry"]["eigenvalues"], original["plane_geometry"]["eigenvalues"], atol=2e-12)


def test_parallel_radius_neighbors_match_exhaustive_distance_and_tie_order():
    x, y = np.meshgrid(np.linspace(-.5, .5, 5), np.linspace(-.5, .5, 5))
    cloud = plane.canonical_map(np.vstack([np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)]),
                                         [1, 0, 0], [-1, 0, 0], [1+1e-12, 0, 0]]))
    before = cloud.copy()
    targets = np.array([[0, 0, 0], [.5, 0, 0], [4, 4, 4]])
    serial = plane.neighborhoods(cloud, targets, workers=1)
    parallel = plane.neighborhoods(cloud, targets, workers=4)
    np.testing.assert_array_equal(serial, parallel)
    for i, target in enumerate(targets):
        ordered = sorted((float(np.sum((p-target)**2)), j) for j, p in enumerate(cloud) if np.sum((p-target)**2) <= 1.)[:12]
        expected = [j for _, j in ordered]+[-1]*(12-len(ordered))
        np.testing.assert_array_equal(serial[i], expected)
    np.testing.assert_array_equal(cloud, before)
    geometry = plane.normal_geometry(cloud, serial)
    assert geometry["labels"].tolist() == ["accepted", "accepted", "insufficient_neighbors"]
    np.testing.assert_allclose(geometry["normals"][:2], [[0, 0, 1], [0, 0, 1]], atol=1e-12)


@pytest.mark.parametrize("kind", ["line", "point", "volume", "few"])
def test_degenerate_and_nonplanar_neighborhoods_keep_explicit_rejections(kind):
    if kind == "line": cloud = np.column_stack([np.linspace(-.5, .5, 12), np.zeros((12, 2))])
    elif kind == "point": cloud = np.zeros((12, 3))
    elif kind == "volume": cloud = np.vstack([np.eye(3)*.2, -np.eye(3)*.2])
    else: cloud = np.zeros((5, 3))
    cloud = plane.canonical_map(cloud)
    result = plane.normal_geometry(cloud, plane.neighborhoods(cloud, np.zeros((1, 3))))
    assert not result["accepted"][0]
    assert result["labels"][0] == ("not_planar" if kind == "volume" else "insufficient_neighbors" if kind == "few" else "line_or_point")


def test_zero_support_never_creates_an_observability_claim():
    result = plane.systems(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.zeros(3))
    assert result["plane_geometry"] is result["point_geometry"] is result["radius_m"] is None
    assert result["weight_sum"] == result["plane_cost"] == 0
    np.testing.assert_array_equal(result["plane_normal_matrix"], np.zeros((6, 6)))


@pytest.mark.parametrize("kind", ["normal", "negative", "nan", "shape"])
def test_invalid_fixed_match_inputs_are_rejected(kind):
    q, p, n, w, c = fixture()
    if kind == "normal": n *= 2
    elif kind == "negative": w[0] = -1
    elif kind == "nan": q[0, 0] = np.nan
    else: w = w[:-1]
    with pytest.raises(ValueError): plane.systems(q, p, n, w, c)
