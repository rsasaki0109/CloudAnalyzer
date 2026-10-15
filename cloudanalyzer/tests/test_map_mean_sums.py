"""Coordinate reductions retain scatter bits and exceptional-value behavior."""

import warnings

import numpy as np
import pytest

import ca.core.map_evaluate as maps


def scatter(points, inverse, cells):
    result = np.zeros((cells, 3), dtype=np.float64)
    np.add.at(result, inverse, points)
    return result


@pytest.mark.parametrize('scale', [1e-320, 1e-150, 1., 1e140])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'reversed'])
def test_coordinate_bits_point_order_and_unchanged_inputs(scale, layout):
    rng = np.random.default_rng(623)
    storage = rng.normal(size=(2050, 6)) * scale
    points = storage[:, ::2] if layout == 'strided' else storage[:, :3].copy()
    if layout == 'reversed':
        points = points[::-1]
    inverse = rng.integers(0, 31, len(points), dtype=np.intp)
    before, before_inverse = points.copy(), inverse.copy()
    with np.errstate(all='raise'):
        observed = maps._voxel_coordinate_sums(points, inverse, 33)
        expected = scatter(points, inverse, 33)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(points.view(np.uint64), before.view(np.uint64))
    np.testing.assert_array_equal(inverse, before_inverse)


@pytest.mark.parametrize('count', [0, 1, 63, 64, 65])
def test_small_input_boundary_and_three_reductions(count, monkeypatch):
    points = np.arange(count * 3, dtype=np.float64).reshape(count, 3)
    inverse = np.zeros(count, dtype=np.intp)
    original, calls = np.bincount, []
    def traced(*args, **kwargs):
        calls.append(kwargs['weights'].shape)
        return original(*args, **kwargs)
    monkeypatch.setattr(np, 'bincount', traced)
    observed = maps._voxel_coordinate_sums(points, inverse, 2)
    np.testing.assert_array_equal(observed.view(np.uint64), scatter(points, inverse, 2).view(np.uint64))
    assert calls == ([(count,)] * 3 if count >= 64 else [])


def test_cancellation_zero_and_subnormal_point_order():
    points = np.zeros((64, 3))
    points[:5] = [[1e16, 1., -1.], [1., 1., 1.], [-1e16, -1., 1.],
                  [-0., 0., -0.], [np.nextafter(0., 1.), -0., 0.]]
    inverse = np.zeros(len(points), dtype=np.intp)
    np.testing.assert_array_equal(maps._voxel_coordinate_sums(points, inverse, 1).view(np.uint64),
                                  scatter(points, inverse, 1).view(np.uint64))


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, 1e307, -1e307])
def test_unsafe_values_keep_scatter_bits_warnings_and_policy(value, monkeypatch):
    points = np.ones((64, 3))
    points[:2, 0] = [value, -value]
    inverse = np.zeros(len(points), dtype=np.intp)
    def forbidden(*args, **kwargs):
        raise AssertionError('unsafe input used bincount')
    monkeypatch.setattr(np, 'bincount', forbidden)
    with np.errstate(all='warn'):
        policy = np.geterr().copy()
        with warnings.catch_warnings(record=True) as old_warnings:
            warnings.simplefilter('always')
            expected = scatter(points, inverse, 1)
        with warnings.catch_warnings(record=True) as new_warnings:
            warnings.simplefilter('always')
            observed = maps._voxel_coordinate_sums(points, inverse, 1)
        assert np.geterr() == policy
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))
    assert [(item.category, str(item.message)) for item in new_warnings] == [(item.category, str(item.message)) for item in old_warnings]


@pytest.mark.parametrize('kind', ['overflow', 'invalid'])
def test_scatter_exceptions_are_preserved(kind):
    points = np.full((64, 3), 1e308 if kind == 'overflow' else np.inf)
    if kind == 'invalid':
        points[1] = -np.inf
    inverse = np.zeros(len(points), dtype=np.intp)
    with np.errstate(all='raise'):
        with pytest.raises(FloatingPointError):
            scatter(points, inverse, 1)
        with pytest.raises(FloatingPointError):
            maps._voxel_coordinate_sums(points, inverse, 1)


@pytest.mark.parametrize('size', [.5, 3., 100.])
def test_resident_gaussians_keep_full_bits(size, monkeypatch):
    rng = np.random.default_rng(624)
    points = rng.uniform(-4., 4., size=(2200, 3))
    points = np.concatenate([points, points[:40], [[1000., -1000., 0.], [0., 0., 0.]]])
    observed = maps.build_voxel_gaussians(points, size)
    monkeypatch.setattr(maps, '_voxel_coordinate_sums', scatter)
    expected = maps.build_voxel_gaussians(points, size)
    assert list(observed) == list(expected)
    for a, b in zip(observed.values(), expected.values(), strict=True):
        assert a.num_points == b.num_points
        np.testing.assert_array_equal(a.mu.view(np.uint64), b.mu.view(np.uint64))
        np.testing.assert_array_equal(a.sigma.view(np.uint64), b.sigma.view(np.uint64))


def test_complete_resident_report_matches_scatter(monkeypatch):
    rng = np.random.default_rng(625)
    points = rng.uniform(-2., 2., size=(700, 3))
    reference = points + [.03, -.01, .02]
    request = maps.MapEvaluateRequest(estimated_points=points, reference_points=reference,
        map_eval_protocol=maps.MapEvalProtocol(voxel_size_m=1., min_voxel_points=4, neighbor_radius=2))
    observed = maps.evaluate_map(request)
    monkeypatch.setattr(maps, '_voxel_coordinate_sums', scatter)
    expected = maps.evaluate_map(request)
    assert observed == expected
