"""Verified finite scalar groups borrow input while preserving moment bits."""

import weakref

import numpy as np
import pytest

import ca.core.map_evaluate as maps
import ca.io as io
from ca.io import PointAccumulator


def assert_bits(a, b):
    np.testing.assert_array_equal(np.asarray(a).view(np.uint64), np.asarray(b).view(np.uint64))


@pytest.mark.parametrize('count', [0, 1, 2, 3, 32, 128, 129, 255, 256, 257, 1024, 4096])
@pytest.mark.parametrize('scale', [1e-150, 1., 1e100])
def test_borrowed_readonly_offset_groups_keep_all_state_bits(count, scale):
    rng = np.random.default_rng(283)
    storage = rng.uniform(-.9, .9, size=(count + 21, 3)) * scale
    points = storage[17:17 + count]
    original = points.copy()
    points.flags.writeable = False
    observed, expected = PointAccumulator(), PointAccumulator()
    for _ in range(2):
        assert observed._update_finite(points) is observed
        assert expected.update(points) is expected
        assert observed.count == expected.count
        for name in ['mean', 'm2', 'minimum', 'maximum']:
            a, b = getattr(observed, name), getattr(expected, name)
            if a is None or b is None:
                assert a is b
            else:
                assert_bits(a, b)
                assert not np.shares_memory(a, points)
    assert_bits(points, original)
    for name in ['mean', 'covariance']:
        assert_bits(observed.finalize()[name], expected.finalize()[name])


def varied_groups():
    counts = np.arange(1, 129, dtype=np.int64)
    keys = np.repeat(np.arange(128), counts)
    rng = np.random.default_rng(814)
    points = rng.uniform(.1, .2, size=(len(keys), 3))
    points[:, 0] += 2 * keys
    return points


def scalar_builder(points):
    keys, inverse = np.unique(np.floor(points).astype(np.int64), axis=0, return_inverse=True)
    result = {}
    for index, key in enumerate(keys):
        summary = PointAccumulator().update(points[inverse == index]).finalize()
        result[tuple(int(value) for value in key)] = maps.VoxelGaussian(summary['mean'], summary['covariance'], summary['count'])
    return result


def assert_gaussians(observed, expected):
    assert list(observed) == list(expected)
    for key in expected:
        assert observed[key].num_points == expected[key].num_points
        assert_bits(observed[key].mu, expected[key].mu)
        assert_bits(observed[key].sigma, expected[key].sigma)


@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'reversed', 'float32'])
def test_global_verification_replaces_per_group_filters(layout, monkeypatch):
    points = varied_groups()
    if layout == 'strided':
        storage = np.empty((len(points), 6))
        storage[:, ::2] = points
        points = storage[:, ::2]
    elif layout == 'reversed':
        points = points[::-1]
    elif layout == 'float32':
        points = points.astype(np.float32)
    expected = scalar_builder(np.asarray(points, dtype=np.float64))
    original = io._filter_chunk
    calls = []
    def traced(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)
    monkeypatch.setattr(io, '_filter_chunk', traced)
    observed = maps.build_voxel_gaussians_from_chunks([points], 1.)
    # Each size occurs once, so no moment cohort qualifies; all 128 cells use
    # the verified scalar path instead of independently filtering/copying.
    assert calls == []
    assert_gaussians(observed, expected)


@pytest.mark.parametrize('guard', ['small', 'nonfinite', 'error_policy', 'magnitude'])
def test_unsafe_or_insufficient_chunks_keep_public_filter(guard, monkeypatch):
    points = varied_groups()
    if guard == 'small':
        points = points[:1023]
    elif guard == 'nonfinite':
        points[0, 0] = np.nan
    elif guard == 'magnitude':
        points[0, 0] = 1e151
    settings = dict(invalid='ignore') if guard in ['nonfinite', 'magnitude'] else {}
    if guard == 'error_policy':
        settings = dict(over='raise')
    original = io._filter_chunk
    calls = []
    def traced(*args, **kwargs):
        calls.append(len(args[0]))
        return original(*args, **kwargs)
    monkeypatch.setattr(io, '_filter_chunk', traced)
    with np.errstate(**settings):
        expected = scalar_builder(points)
        calls.clear()
        observed = maps.build_voxel_gaussians_from_chunks([points], 1.)
    assert calls and len(calls) == len(observed)
    assert_gaussians(observed, expected)


def test_verified_path_does_not_retain_gathered_points(monkeypatch):
    references = []
    original = PointAccumulator._update_finite
    def traced(self, points):
        references.append(weakref.ref(points))
        assert points.flags.c_contiguous and np.isfinite(points).all()
        return original(self, points)
    monkeypatch.setattr(PointAccumulator, '_update_finite', traced)
    points = varied_groups()
    def chunks():
        yield points
        assert all(reference() is None for reference in references)
        yield points
        assert all(reference() is None for reference in references)
    observed = maps.build_voxel_gaussians_from_chunks(chunks(), 1.)
    assert sum(voxel.num_points for voxel in observed.values()) == 2 * len(points)


def test_verified_backend_failure_does_not_consume_next_chunk(monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError('verified backend failed')
    monkeypatch.setattr(PointAccumulator, '_update_finite', failed)
    def chunks():
        yield varied_groups()
        raise AssertionError('next chunk consumed')
    with pytest.raises(RuntimeError, match='verified backend failed'):
        maps.build_voxel_gaussians_from_chunks(chunks(), 1.)
