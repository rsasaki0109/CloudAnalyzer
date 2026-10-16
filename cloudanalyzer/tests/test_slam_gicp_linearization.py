"""Closed-form checks for the independent GICP audit, without native code."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("gicp_audit", SCRIPTS / "audit_flatwall_gicp_linearization.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_independent_anisotropic_system_matches_closed_form():
    source, target = np.array([[2., 3., 4.]]), np.array([[1., 1., 1.]])
    shape = np.array([np.diag([.25, .5, 1.])])
    result, radius = audit.reference(source, target, shape, shape, np.eye(4), source[0], np.array([2.]))
    np.testing.assert_array_equal(result["normal_matrix"], np.diag([4., 2., 1., 0., 0., 0.]))
    np.testing.assert_array_equal(result["right_hand_side"], [-4., -4., -3., 0., 0., 0.])
    assert result["cost"] == 10.5 and result["mahalanobis_squared"][0] == 10.5
    assert radius == 0.


def test_local_plane_shapes_use_declared_anisotropy_and_keep_query_order():
    points = np.array([[x, y, 2.] for x in range(4) for y in range(4)], dtype=float)
    shapes, valid = audit.shapes(points, points[[13, 1, 7]])
    assert valid.all()
    np.testing.assert_allclose(shapes, np.tile(np.diag([1., 1., .001]), (3, 1, 1)), atol=1e-15)


@pytest.mark.parametrize("points", [np.zeros((4, 3)), np.array([[i, 0., 0.] for i in range(20)])])
def test_missing_or_line_neighborhoods_are_unsupported_without_identity_fallback(points):
    shapes, valid = audit.shapes(points, points[:2])
    assert not valid.any()
    np.testing.assert_array_equal(shapes, np.zeros((2, 3, 3)))


def test_empty_reference_has_zero_system_and_no_radius():
    result, radius = audit.reference(np.empty((0, 3)), np.empty((0, 3)),
                                     np.empty((0, 3, 3)), np.empty((0, 3, 3)),
                                     np.eye(4), np.zeros(3), np.empty(0))
    assert result["cost"] == result["weight_sum"] == 0.
    assert radius is None
    np.testing.assert_array_equal(result["normal_matrix"], np.zeros((6, 6)))


def test_bit_comparison_detects_signed_zero_changes():
    with pytest.raises(AssertionError):
        audit.equal_bits({"value": np.array([0.])}, {"value": np.array([-0.])})
