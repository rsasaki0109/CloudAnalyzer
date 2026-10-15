"""Expanded equal-size groups keep scalar bits and a fixed gather budget."""

import weakref

import numpy as np
import pytest

import ca.core.map_evaluate as maps
from ca.io import PointAccumulator


def assert_bits(a, b):
    np.testing.assert_array_equal(np.asarray(a).view(np.uint64), np.asarray(b).view(np.uint64))


@pytest.mark.parametrize('count', [33, 63, 64, 65, 127, 128])
@pytest.mark.parametrize('scale', [1e-150, 1., 1e100, 1e150])
def test_expanded_moments_and_merges_keep_scalar_bits(count, scale):
    rng = np.random.default_rng(406)
    points = rng.uniform(-.9, .9, size=(128 * count, 3)) * scale
    counts = np.full(128, count, dtype=np.int64)
    starts = np.arange(128) * count
    moments = maps._stream_group_moments(points, starts, counts)
    assert moments is not None and moments.selected.all()
    for slot, start in enumerate(starts):
        batch = points[start:start + count]
        expected = PointAccumulator().update(batch)
        for name in ['mean', 'm2', 'minimum', 'maximum']:
            assert_bits(getattr(moments, name)[slot], getattr(expected, name))
        observed = PointAccumulator()
        for _ in range(2):
            maps._update_stream_voxel(observed, batch, moments, slot)
        expected.update(batch)
        assert observed.count == expected.count
        for name in ['mean', 'm2', 'minimum', 'maximum']:
            assert_bits(getattr(observed, name), getattr(expected, name))


@pytest.mark.parametrize('groups', [255, 256, 257, 511, 512, 513, 1024])
def test_point_budget_tail_and_scratch_release(groups, monkeypatch):
    rng = np.random.default_rng(492)
    points = rng.normal(size=(groups * 128, 3))
    counts = np.full(groups, 128, dtype=np.int64)
    starts = np.arange(groups) * 128
    refs, sizes = [], []
    original = np.mean
    def traced(values, *args, **kwargs):
        if values.ndim == 3:
            assert all(reference() is None for reference in refs)
            sizes.append(values.shape[0] * values.shape[1])
            refs.append(weakref.ref(values))
        result = original(values, *args, **kwargs)
        if values.ndim == 3:
            refs.append(weakref.ref(result))
        return result
    monkeypatch.setattr(np, 'mean', traced)
    moments = maps._stream_group_moments(points, starts, counts)
    assert moments is not None and moments.selected.all()
    assert all(size <= 32768 for size in sizes)
    assert sum(sizes) == len(points)
    assert len(sizes) == (groups + 255) // 256
    assert all(reference() is None for reference in refs)
    for slot in [0, min(255, groups - 1), groups - 1]:
        scalar = PointAccumulator().update(points[starts[slot]:starts[slot] + 128])
        for name in ['mean', 'm2', 'minimum', 'maximum']:
            assert_bits(getattr(moments, name)[slot], getattr(scalar, name))


@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'reversed'])
def test_many_eight_group_cohorts(layout):
    counts = np.repeat(np.arange(1, 17, dtype=np.int64), 8)
    rng = np.random.default_rng(642)
    storage = rng.normal(size=(sum(counts), 6))
    points = storage[:, ::2] if layout == 'strided' else storage[:, :3].copy()
    if layout == 'reversed':
        points = points[::-1]
    starts = np.cumsum(counts) - counts
    moments = maps._stream_group_moments(points, starts, counts)
    assert moments is not None and moments.selected.all()
    for slot, count in enumerate(counts):
        scalar = PointAccumulator().update(points[starts[slot]:starts[slot] + count])
        for name in ['mean', 'm2', 'minimum', 'maximum']:
            assert_bits(getattr(moments, name)[slot], getattr(scalar, name))


@pytest.mark.parametrize('count', [33, 128])
def test_large_signed_zero_and_subnormal_groups(count):
    tiny = np.nextafter(0., 1.)
    pattern = np.array([[0., -0., tiny], [-0., 0., -tiny], [tiny, -tiny, -0.]])
    points = np.tile(pattern, (128 * count // 3 + 1, 1))[:128 * count].copy()
    counts = np.full(128, count, dtype=np.int64)
    starts = np.arange(128) * count
    moments = maps._stream_group_moments(points, starts, counts)
    assert moments is not None
    for slot, start in enumerate(starts):
        scalar = PointAccumulator().update(points[start:start + count])
        for name in ['mean', 'm2', 'minimum', 'maximum']:
            assert_bits(getattr(moments, name)[slot], getattr(scalar, name))
