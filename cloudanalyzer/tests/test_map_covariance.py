"""Grouped covariance preserves point-order f64 reductions and edge behavior."""

import numpy as np
import pytest

import ca.core.map_evaluate as maps


def scatter(centered, inverse, cells):
    result = np.zeros((cells, 3, 3), dtype=np.float64)
    np.add.at(result, inverse, centered[:, :, None] * centered[:, None, :])
    return result


@pytest.mark.parametrize("scale", [1e-150, 1., 1e140])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed"])
def test_covariance_bits_and_input_order(scale, layout):
    rng = np.random.default_rng(421)
    storage = rng.normal(size=(2050, 6)) * scale
    centered = storage[:, ::2] if layout == "strided" else storage[:, :3].copy()
    if layout == "reversed":
        centered = centered[::-1]
    inverse = rng.integers(0, 31, len(centered), dtype=np.intp)
    before = centered.copy()
    expected = scatter(centered, inverse, 33)
    observed = maps._voxel_covariance_sums(centered, inverse, 33)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(centered.view(np.uint64), before.view(np.uint64))


def test_cancellation_signed_zero_subnormals_and_six_components(monkeypatch):
    centered = np.array([[1e16, 1., -1.], [1., 1., 1.], [-1e16, 1., 1.],
                         [-0., 0., -0.], [np.nextafter(0., 1.), -1., 1.]])
    inverse = np.zeros(len(centered), dtype=np.intp)
    expected = scatter(centered, inverse, 1)
    calls = []
    original = np.bincount
    def traced(*args, **kwargs):
        calls.append(kwargs['weights'].shape)
        return original(*args, **kwargs)
    monkeypatch.setattr(np, 'bincount', traced)
    observed = maps._voxel_covariance_sums(centered, inverse, 1)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))
    assert calls == [(len(centered),)] * 6


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1e200, -1e200])
def test_nonfinite_and_extreme_scatter_fallback(value, monkeypatch):
    centered = np.array([[value, 1., 0.], [-value, -1., 0.]])
    inverse = np.zeros(2, dtype=np.intp)
    def forbidden(*args, **kwargs):
        raise AssertionError('unsafe input used bincount')
    monkeypatch.setattr(np, 'bincount', forbidden)
    with np.errstate(all='ignore'):
        expected = scatter(centered, inverse, 1)
        observed = maps._voxel_covariance_sums(centered, inverse, 1)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))


def test_overflow_error_is_preserved():
    with np.errstate(over='raise', invalid='raise'):
        with pytest.raises(FloatingPointError):
            maps._voxel_covariance_sums(np.full((2, 3), 1e200), np.zeros(2, dtype=np.intp), 1)


def test_empty_covariance():
    result = maps._voxel_covariance_sums(np.empty((0, 3)), np.empty(0, dtype=np.intp), 2)
    np.testing.assert_array_equal(result, np.zeros((2, 3, 3)))


@pytest.mark.parametrize("size", [.5, 3., 100.])
def test_resident_gaussians_match_legacy_scatter(size, monkeypatch):
    rng = np.random.default_rng(521)
    points = rng.uniform(-4., 4., size=(2200, 3))
    # Include repeated points, a singleton and exact boundary coordinates.
    points = np.concatenate([points, points[:40], [[1000., -1000., 0.], [0., 0., 0.]]])
    optimized = maps.build_voxel_gaussians(points, size)
    monkeypatch.setattr(maps, '_voxel_covariance_sums', scatter)
    legacy = maps.build_voxel_gaussians(points, size)
    assert list(optimized) == list(legacy)
    for a, b in zip(optimized.values(), legacy.values(), strict=True):
        assert a.num_points == b.num_points
        np.testing.assert_array_equal(a.mu.view(np.uint64), b.mu.view(np.uint64))
        np.testing.assert_array_equal(a.sigma.view(np.uint64), b.sigma.view(np.uint64))
