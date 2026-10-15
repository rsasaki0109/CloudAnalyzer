"""Downsampling retains legacy centroid bits while batching singleton means."""

from dataclasses import asdict

import numpy as np
import pytest

import ca.core.map_evaluate as maps


def legacy_centroids(points, starts):
    return np.vstack([points[start:(starts[i + 1] if i + 1 < len(starts) else len(points))].mean(axis=0)
                      for i, start in enumerate(starts)]) if len(starts) else np.empty((0, 3))


def legacy_downsample(points, size):
    points = maps._require_xyz(points, 'points')
    if not len(points) or size <= 0:
        return points
    keys = np.floor(points / size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    _, starts = np.unique(keys[order], axis=0, return_index=True)
    return legacy_centroids(points[order], starts)


@pytest.mark.parametrize('size', [.2, 1., 10.])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'reversed'])
def test_centroid_bits_order_and_input_snapshot(size, layout):
    rng = np.random.default_rng(1121)
    storage = rng.uniform(-5., 5., size=(2400, 6))
    points = storage[:, ::2] if layout == 'strided' else storage[:, :3].copy()
    if layout == 'reversed':
        points = points[::-1]
    before = points.copy()
    expected = legacy_downsample(points, size)
    observed = maps.voxel_downsample(points, size)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(points.view(np.uint64), before.view(np.uint64))
    assert observed.dtype == np.float64 and observed.flags.c_contiguous


@pytest.mark.parametrize('count', [65_535, 65_536, 65_537])
def test_singleton_batch_boundary_and_edge_bits(count, monkeypatch):
    values = np.array([[0., -0., np.nextafter(0., 1.)],
                       [np.finfo(np.float64).max, -np.finfo(np.float64).max, -np.nextafter(0., 1.)],
                       [np.finfo(np.float64).tiny, -1., 1.]])
    points = np.resize(values, (count, 3))
    starts = np.arange(count, dtype=np.intp)
    expected = legacy_centroids(points, starts)
    calls = []
    original = np.mean
    def traced(batch, **kwargs):
        calls.append((batch.shape, kwargs))
        return original(batch, **kwargs)
    monkeypatch.setattr(np, 'mean', traced)
    with np.errstate(all='raise'):
        observed = maps._sorted_voxel_centroids(points, starts)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))
    assert calls == [((min(count, 65_536), 1, 3), dict(axis=1))] + ([((1, 1, 3), dict(axis=1))] if count > 65_536 else [])


def test_mixed_groups_keep_cancellation_order():
    points = np.array([[1e16, 1., -1.], [1., 1., 1.], [-1e16, 1., 1.],
                       [-0., 0., -0.], [.2, .3, .4], [.4, .1, .8], [9., 8., 7.]])
    starts = np.array([0, 3, 4, 6])
    expected = legacy_centroids(points, starts)
    observed = maps._sorted_voxel_centroids(points, starts)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_groups_keep_legacy_mean_calls(value, monkeypatch):
    points = np.array([[value, 0., 0.], [1., 2., 3.], [4., 5., 6.]])
    starts = np.arange(3)
    expected = legacy_centroids(points, starts)
    def forbidden(*args, **kwargs):
        raise AssertionError('nonfinite input used batched mean')
    monkeypatch.setattr(np, 'mean', forbidden)
    observed = maps._sorted_voxel_centroids(points, starts)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))


def test_multi_point_overflow_exception_is_preserved():
    points = np.full((2, 3), np.finfo(np.float64).max)
    with np.errstate(over='raise'):
        with pytest.raises(FloatingPointError):
            maps._sorted_voxel_centroids(points, np.array([0]))


@pytest.mark.parametrize('size', [0., -1., -np.inf])
def test_disabled_downsample_retains_original_array(size):
    points = np.arange(12, dtype=np.float64).reshape(4, 3)
    assert maps.voxel_downsample(points, size) is points


def test_empty_points_and_single_point():
    points = np.empty((0, 3))
    assert maps.voxel_downsample(points, .2) is points
    point = np.array([[-0., 0., .1]])
    np.testing.assert_array_equal(maps.voxel_downsample(point, .2).view(np.uint64), legacy_downsample(point, .2).view(np.uint64))


@pytest.mark.parametrize('size', [np.nan, np.inf])
def test_legacy_nonfinite_voxel_size_behavior(size):
    points = np.arange(30, dtype=np.float64).reshape(10, 3)
    with np.errstate(invalid='ignore'):
        expected = legacy_downsample(points, size)
        observed = maps.voxel_downsample(points, size)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))


def test_complete_resident_report_parity(monkeypatch):
    rng = np.random.default_rng(1221)
    reference = rng.uniform(-3., 3., size=(2400, 3))
    estimated = reference + rng.normal(0., .01, reference.shape)
    request = maps.MapEvaluateRequest(estimated_points=estimated, reference_points=reference, downsample_voxel_size=.2,
                 map_eval_protocol=maps.MapEvalProtocol(voxel_size_m=1., min_voxel_points=2, neighbor_radius=2))
    observed = maps.evaluate_map(request)
    monkeypatch.setattr(maps, 'voxel_downsample', legacy_downsample)
    expected = maps.evaluate_map(request)
    assert asdict(observed) == asdict(expected)
