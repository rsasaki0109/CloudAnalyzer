"""f64 surface-shape preparation, exact neighborhoods and support contracts."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

core = pytest.importorskip("cloudanalyzer_core")
sys.path.insert(0, str(Path(__file__).resolve().parents[4]/"scripts"))
import audit_flatwall_gicp_shapes as audit  # noqa: E402


def plane():
    return np.array([[x*.3, y*.2, 2.] for x in range(9) for y in range(7)])


@pytest.mark.parametrize("n", [0, 1, 4, 5, 9, 129, 1025])
def test_exact_neighbor_ranks_and_pca_agree_with_independent_reference(n):
    points = np.random.default_rng(451).normal(size=(n, 3))
    actual = core.gicp_shapes(points, capture_neighbors=True)
    audit.check_shapes(points, actual)
    audit.equal_bits(actual, core.gicp_shapes(points, capture_neighbors=True, parallel=True))
    plain = core.gicp_shapes(points)
    assert plain["neighbor_indices"].shape == plain["neighbor_squared_distances"].shape == (n, 0)
    for name in ["shapes", "supported", "scatter_eigenvalues", "neighbor_counts"]:
        audit.equal_bits(plain[name], actual[name])


@pytest.mark.parametrize("neighbors", [5, 7, 20, sys.maxsize])
def test_plane_shape_and_clipped_neighborhood_size(neighbors):
    points = plane()
    result = core.gicp_shapes(points, neighbors=neighbors, capture_neighbors=True)
    assert result["supported"].all()
    np.testing.assert_allclose(result["shapes"], np.tile(np.diag([1., 1., .001]), (len(points), 1, 1)), atol=1e-14)
    np.testing.assert_array_equal(result["neighbor_counts"], min(neighbors, len(points)))
    assert result["neighbor_indices"].shape == (len(points), min(neighbors, len(points)))


@pytest.mark.parametrize("points", [np.empty((0, 3)), np.zeros((4, 3)), np.zeros((27, 3)),
    np.array([[i, 0., 0.] for i in range(25)])])
def test_unsupported_geometry_returns_zero_shapes_without_fallback(points):
    result = core.gicp_shapes(points, capture_neighbors=True)
    assert not result["supported"].any()
    np.testing.assert_array_equal(result["shapes"], np.zeros((len(points), 3, 3)))
    stats = audit.check_shapes(points, result)
    assert stats["unsupported"] == len(points)
    if len(points):
        with pytest.raises(ValueError, match="source point or shape"):
            core.gicp_register(points, points, result["shapes"], result["shapes"], np.eye(4), np.zeros(3), max_correspondence_distance=1.)


def test_repeated_minimum_eigenspace_is_explicitly_ambiguous_and_supported():
    points = np.concatenate([np.eye(3), -np.eye(3)])
    result = core.gicp_shapes(points, capture_neighbors=True)
    stats = audit.check_shapes(points, result)
    assert stats["ambiguous_minimum_eigenspaces"] == len(points)
    assert result["supported"].all()
    np.testing.assert_allclose(np.linalg.eigvalsh(result["shapes"]), np.tile([.001, 1., 1.], (len(points), 1)), atol=1e-14)


@pytest.mark.parametrize("origin", [np.zeros(3), np.array([1e6, -2e6, 3e6])])
def test_rotated_world_plane_and_large_origin_keep_f64_shapes(origin):
    points = plane()
    q = Rotation.from_rotvec([.31, -.17, .43]).as_matrix()
    expected = q @ np.diag([1., 1., .001]) @ q.T
    result = core.gicp_shapes(points @ q.T + origin)
    assert result["supported"].all()
    np.testing.assert_allclose(result["shapes"], np.tile(expected, (len(points), 1, 1)), atol=2e-7 if origin.any() else 2e-13)
    if not origin.any():
        assert np.max(np.abs(result["shapes"].astype(np.float32).astype(float)-expected)) > 1e-9


@pytest.mark.parametrize("scale", [1e-100, 1e-4, 1e4, 1e100])
def test_scaled_eigensolve_preserves_shape_across_representable_scatter_scales(scale):
    points = np.random.default_rng(912).normal(size=(9, 3))
    baseline = core.gicp_shapes(points)
    actual = core.gicp_shapes(points*scale)
    np.testing.assert_array_equal(actual["supported"], baseline["supported"])
    np.testing.assert_allclose(actual["shapes"], baseline["shapes"], atol=2e-12)
    np.testing.assert_allclose(actual["scatter_eigenvalues"]/(scale*scale), baseline["scatter_eigenvalues"], atol=2e-12)


@pytest.mark.parametrize("neighbors", [0, 1, 4])
def test_invalid_neighbor_count_is_rejected(neighbors):
    with pytest.raises(ValueError, match="at least five"):
        core.gicp_shapes(np.empty((0, 3)), neighbors=neighbors)


@pytest.mark.parametrize("parallel", [False, True])
def test_first_nonfinite_point_and_arithmetic_failure_are_deterministic(parallel):
    points = np.zeros((1025, 3))
    points[7, 0], points[333, 1] = np.nan, np.inf
    with pytest.raises(ValueError, match="point 7: point must be finite"):
        core.gicp_shapes(points, parallel=parallel)
    points = np.array([[0., 0., 0.], [np.finfo(float).max]*3, [-np.finfo(float).max]*3, [1.]*3, [2.]*3])
    with pytest.raises(ValueError, match="point 0: nonfinite"):
        core.gicp_shapes(points, parallel=parallel, capture_neighbors=True)


def test_strided_inputs_and_all_outputs_are_owned():
    points = plane()
    original = points.copy()
    expected = core.gicp_shapes(points, capture_neighbors=True)
    audit.equal_bits(expected, core.gicp_shapes(points[::-1].copy()[::-1], capture_neighbors=True))
    np.testing.assert_array_equal(points, original)
    result = core.gicp_shapes(points, capture_neighbors=True)
    for value in result.values():
        value[...] = 0
    audit.equal_bits(expected, core.gicp_shapes(points, capture_neighbors=True))
    with pytest.raises(ValueError):
        core.gicp_shapes(np.zeros((4, 4)))
