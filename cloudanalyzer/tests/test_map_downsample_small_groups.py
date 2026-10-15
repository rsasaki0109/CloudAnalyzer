"""Small-group batches preserve mean bits, arithmetic policy and output order."""

from dataclasses import asdict

import numpy as np
import pytest

import ca.core.map_evaluate as maps


def legacy_centroids(points, starts):
    return np.vstack([points[start:(starts[i + 1] if i + 1 < len(starts) else len(points))].mean(axis=0)
                      for i, start in enumerate(starts)])


@pytest.mark.parametrize('count', range(2, 9))
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'reversed'])
def test_all_small_group_bits_and_input_snapshot(count, layout):
    rng = np.random.default_rng(1421)
    storage = rng.normal(size=(4096 * count, 6))
    points = storage[:, ::2] if layout == 'strided' else storage[:, :3].copy()
    if layout == 'reversed':
        points = points[::-1]
    starts = np.arange(0, len(points), count)
    before = points.copy()
    expected = legacy_centroids(points, starts)
    observed = maps._sorted_voxel_centroids(points, starts)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(points.view(np.uint64), before.view(np.uint64))


@pytest.mark.parametrize('groups', [4095, 4096, 4097])
def test_group_count_guard_and_dispatch(groups, monkeypatch):
    points = np.arange(groups * 6, dtype=np.float64).reshape(groups * 2, 3)
    starts = np.arange(0, len(points), 2)
    expected = legacy_centroids(points, starts)
    calls = []
    original = np.mean
    def traced(batch, **kwargs):
        calls.append(batch.shape)
        return original(batch, **kwargs)
    monkeypatch.setattr(np, 'mean', traced)
    observed = maps._sorted_voxel_centroids(points, starts)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))
    assert calls == ([(groups, 2, 3)] if groups >= 4096 else [])


@pytest.mark.parametrize('groups', [32767, 32768, 32769])
def test_total_gathered_points_batch_bound(groups, monkeypatch):
    points = np.arange(groups * 6, dtype=np.float64).reshape(groups * 2, 3)
    starts = np.arange(0, len(points), 2)
    expected = legacy_centroids(points, starts)
    calls = []
    original = np.mean
    def traced(batch, **kwargs):
        calls.append(batch.shape)
        return original(batch, **kwargs)
    monkeypatch.setattr(np, 'mean', traced)
    observed = maps._sorted_voxel_centroids(points, starts)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))
    assert all(groups * count <= 65536 and axes == 3 for groups, count, axes in calls)
    assert calls == [(min(groups, 32768), 2, 3)] + ([(1, 2, 3)] if groups > 32768 else [])


def test_mixed_lengths_cancellation_signed_zero_and_large_fallback():
    rng = np.random.default_rng(1521)
    counts = np.tile(np.array([1, 3, 2, 8, 9, 4, 7, 6, 5]), 600)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    points = rng.normal(size=(int(counts.sum()), 3))
    for start, count in zip(starts, counts):
        points[start, 0] = -0.
        if count >= 3:
            points[start:start + 3, 1] = [1e16, 1., -1e16]
    expected = legacy_centroids(points, starts)
    observed = maps._sorted_voxel_centroids(points, starts)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize('scale', [1e-320, 1e-150, 1e140])
def test_tiny_and_large_finite_mean_bits(scale):
    rng = np.random.default_rng(1621)
    points = rng.normal(size=(4096 * 5, 3)) * scale
    starts = np.arange(0, len(points), 5)
    expected = legacy_centroids(points, starts)
    observed = maps._sorted_voxel_centroids(points, starts)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize('under', ['raise', 'warn'])
def test_underflow_reporting_keeps_ordered_mean_path(under, monkeypatch):
    points = np.arange(4096 * 6, dtype=np.float64).reshape(4096 * 2, 3)
    starts = np.arange(0, len(points), 2)
    expected = legacy_centroids(points, starts)
    def forbidden(*args, **kwargs):
        raise AssertionError('underflow-reporting input used batched mean')
    monkeypatch.setattr(np, 'mean', forbidden)
    before = np.geterr()
    with np.errstate(under=under):
        observed = maps._sorted_voxel_centroids(points, starts)
    assert np.geterr() == before
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))


def test_extreme_finite_coordinates_keep_ordered_overflow(monkeypatch):
    points = np.full((4096 * 2, 3), np.finfo(np.float64).max)
    starts = np.arange(0, len(points), 2)
    def forbidden(*args, **kwargs):
        raise AssertionError('extreme coordinates used batched mean')
    monkeypatch.setattr(np, 'mean', forbidden)
    with np.errstate(over='raise'):
        with pytest.raises(FloatingPointError):
            maps._sorted_voxel_centroids(points, starts)


def test_many_large_groups_do_not_enable_small_group_batches(monkeypatch):
    points = np.ones((4096 * 9, 3))
    starts = np.arange(0, len(points), 9)
    def forbidden(*args, **kwargs):
        raise AssertionError('large groups used batched mean')
    monkeypatch.setattr(np, 'mean', forbidden)
    np.testing.assert_array_equal(maps._sorted_voxel_centroids(points, starts), np.ones((4096, 3)))


def test_complete_resident_report_preserves_centroid_order(monkeypatch):
    rng = np.random.default_rng(1721)
    centers = np.column_stack((np.arange(4300) * .5, np.zeros(4300), np.zeros(4300)))
    reference = np.repeat(centers, 3, axis=0) + rng.uniform(.05, .15, size=(12900, 3))
    estimated = reference + .01
    request = maps.MapEvaluateRequest(estimated_points=estimated, reference_points=reference, downsample_voxel_size=.2,
                 map_eval_protocol=maps.MapEvalProtocol(voxel_size_m=3., min_voxel_points=1, neighbor_radius=2))
    observed = maps.evaluate_map(request)
    monkeypatch.setattr(maps, '_sorted_voxel_centroids', legacy_centroids)
    expected = maps.evaluate_map(request)
    assert asdict(observed) == asdict(expected)
