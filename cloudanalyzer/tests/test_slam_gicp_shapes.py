"""Independent PCA oracle controls without importing the native extension."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
import audit_flatwall_gicp_shapes as audit  # noqa: E402


def test_plane_population_scatter_and_normalized_shape():
    points = np.array([[x, y, 2.] for x in range(4) for y in range(4)], dtype=float)
    ids = np.tile(np.arange(len(points)), (len(points), 1))
    shapes, good, values, scatter = audit.pca_reference(points, ids)
    assert good.all()
    np.testing.assert_allclose(values, np.tile([0., 1.25, 1.25], (len(points), 1)), atol=1e-15)
    np.testing.assert_allclose(scatter, np.tile(np.diag([1.25, 1.25, 0.]), (len(points), 1, 1)), atol=1e-15)
    np.testing.assert_allclose(shapes, np.tile(np.diag([1., 1., .001]), (len(points), 1, 1)), atol=1e-15)


@pytest.mark.parametrize("points", [np.empty((0, 3)), np.zeros((4, 3)), np.zeros((8, 3)),
    np.array([[i, 0., 0.] for i in range(8)])])
def test_unsupported_reference_has_no_identity_fallback(points):
    ids = np.tile(np.arange(len(points)), (len(points), 1))
    shapes, good, _, _ = audit.pca_reference(points, ids)
    assert not good.any()
    np.testing.assert_array_equal(shapes, np.zeros((len(points), 3, 3)))


def test_recursive_bit_check_handles_boolean_arrays_and_signed_zero():
    audit.equal_bits(dict(support=np.array([True, False, True])), dict(support=np.array([True, False, True])))
    with pytest.raises(AssertionError):
        audit.equal_bits(np.array([0.]), np.array([-0.]))
