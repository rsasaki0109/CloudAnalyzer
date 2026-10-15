"""Scalar voxel codes preserve integer grouping, order and Gaussian outputs."""

import numpy as np
import pytest

import ca.core.map_evaluate as maps


def axis_grouping(indices):
    return np.unique(indices, axis=0, return_inverse=True, return_counts=True)


@pytest.mark.parametrize('origin', [0, -(1 << 62), (1 << 62), np.iinfo(np.int64).min])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'reversed'])
def test_integer_grouping_and_point_order(origin, layout):
    rng = np.random.default_rng(721)
    storage = rng.integers(0, 17, size=(1300, 6), dtype=np.int64) + origin
    indices = storage[:, ::2] if layout == 'strided' else storage[:, :3].copy()
    if layout == 'reversed':
        indices = indices[::-1]
    before = indices.copy()
    observed, inverse, counts = maps._unique_voxel_indices(indices)
    expected = axis_grouping(indices)
    for a, b in zip((observed, inverse, counts), expected, strict=True):
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype
    np.testing.assert_array_equal(observed[inverse], indices)
    np.testing.assert_array_equal(indices, before)


@pytest.mark.parametrize('points', [0, 1, 63, 64, 65])
def test_small_input_boundary_and_dispatch(points, monkeypatch):
    indices = np.arange(points * 3, dtype=np.int64).reshape(points, 3)
    expected = axis_grouping(indices)
    calls = []
    original = np.unique
    def traced(values, **kwargs):
        calls.append((values.ndim, kwargs.get('axis')))
        return original(values, **kwargs)
    monkeypatch.setattr(np, 'unique', traced)
    observed = maps._unique_voxel_indices(indices)
    for a, b in zip(observed, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    assert calls == ([(2, 0)] if points < 64 else [(1, None)])


@pytest.mark.parametrize('case', ['largest_safe', 'one_above', 'full_signed_span', 'product_overflow'])
def test_integer_capacity_guards(case, monkeypatch):
    maximum = np.iinfo(np.int64).max
    rows = {
        'largest_safe': [[0, 0, 0], [maximum - 1, 0, 0]],
        'one_above': [[0, 0, 0], [maximum, 0, 0]],
        'full_signed_span': [[np.iinfo(np.int64).min, 0, 0], [maximum, 0, 0]],
        'product_overflow': [[-2_000_000, -2_000_000, -2_000_000], [2_000_000, 2_000_000, 2_000_000]],
    }
    indices = np.tile(np.array(rows[case], dtype=np.int64), (40, 1))
    expected = axis_grouping(indices)
    calls = []
    original = np.unique
    def traced(values, **kwargs):
        calls.append(values.ndim)
        return original(values, **kwargs)
    monkeypatch.setattr(np, 'unique', traced)
    with np.errstate(all='raise'):
        observed = maps._unique_voxel_indices(indices)
    for a, b in zip(observed, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    assert calls == ([1] if case == 'largest_safe' else [2])


def test_constant_axes_and_near_maximum_keys():
    maximum = np.iinfo(np.int64).max
    indices = np.array([[maximum - (i % 11), -7, 5] for i in range(90)], dtype=np.int64)
    for a, b in zip(maps._unique_voxel_indices(indices), axis_grouping(indices), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('voxel_size', [.5, 3., 100.])
def test_gaussians_preserve_all_f64_bits(voxel_size, monkeypatch):
    rng = np.random.default_rng(821)
    points = rng.uniform(-5., 5., size=(2600, 3))
    points = np.concatenate([points, points[:80], [[1000., -1000., 0.], [0., 0., 0.]]])
    optimized = maps.build_voxel_gaussians(points, voxel_size)
    monkeypatch.setattr(maps, '_unique_voxel_indices', axis_grouping)
    legacy = maps.build_voxel_gaussians(points, voxel_size)
    assert list(optimized) == list(legacy)
    for a, b in zip(optimized.values(), legacy.values(), strict=True):
        assert a.num_points == b.num_points
        np.testing.assert_array_equal(a.mu.view(np.uint64), b.mu.view(np.uint64))
        np.testing.assert_array_equal(a.sigma.view(np.uint64), b.sigma.view(np.uint64))
