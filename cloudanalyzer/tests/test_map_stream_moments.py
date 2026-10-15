"""Bounded same-size streaming reductions retain the scalar moment bits."""

from dataclasses import asdict
import weakref

import numpy as np
import pytest

import ca.core.map_evaluate as maps
from ca.io import PointAccumulator


class ScalarAccumulator(PointAccumulator):
    def update(self, points):
        chunk = np.asarray(points, dtype=np.float64)
        chunk = chunk[np.isfinite(chunk).all(axis=1)]
        if not len(chunk):
            return self
        count = len(chunk)
        mean = np.mean(chunk, axis=0)
        centered = chunk - mean
        m2 = centered.T @ centered
        if self.count == 0:
            self.count, self.mean, self.m2 = count, mean, m2
        else:
            total = self.count + count
            delta = mean - self.mean
            self.m2 += m2 + np.outer(delta, delta) * (self.count * count / total)
            self.mean += delta * (count / total)
            self.count = total
        minimum, maximum = np.min(chunk, axis=0), np.max(chunk, axis=0)
        self.minimum = minimum if self.minimum is None else np.minimum(self.minimum, minimum)
        self.maximum = maximum if self.maximum is None else np.maximum(self.maximum, maximum)
        return self


def assert_bits(a, b):
    np.testing.assert_array_equal(np.asarray(a).view(np.uint64), np.asarray(b).view(np.uint64))


def assert_accumulators(a, b):
    assert a.count == b.count
    for name in ['mean', 'm2', 'minimum', 'maximum']:
        first, second = getattr(a, name), getattr(b, name)
        if first is None or second is None:
            assert first is second
        else:
            assert_bits(first, second)


def scalar_builder(chunks, size):
    accumulators = {}
    for chunk in chunks:
        points = maps._require_xyz(np.asarray(chunk, dtype=np.float64), 'point chunk')
        if not len(points):
            continue
        unique, inverse = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_inverse=True)
        for index, key_values in enumerate(unique):
            key = tuple(int(value) for value in key_values)
            accumulator = accumulators.get(key)
            if accumulator is None:
                accumulator = ScalarAccumulator()
                accumulators[key] = accumulator
            accumulator.update(points[inverse == index])
    result = {}
    for key, accumulator in accumulators.items():
        summary = accumulator.finalize()
        result[key] = maps.VoxelGaussian(summary['mean'], summary['covariance'], summary['count'])
    return result


@pytest.mark.parametrize('count', [1, 2, 3, 4, 7, 8, 16, 32])
@pytest.mark.parametrize('scale', [1e-150, 1., 1e100])
def test_all_batch_fields_and_ordered_merges_match_scalar_bits(count, scale):
    rng = np.random.default_rng(782)
    points = rng.normal(size=(128 * count, 3)) * scale
    counts = np.full(128, count, dtype=np.int64)
    starts = np.arange(128) * count
    moments = maps._stream_group_moments(points, starts, counts)
    assert moments is not None and moments.selected.all()
    for slot, start in enumerate(starts):
        batch = points[start:start + count]
        expected = ScalarAccumulator().update(batch)
        for name in ['mean', 'm2', 'minimum', 'maximum']:
            assert_bits(getattr(moments, name)[slot], getattr(expected, name))
        observed = PointAccumulator()
        maps._update_stream_voxel(observed, batch, moments, slot)
        assert_accumulators(observed, expected)
        maps._update_stream_voxel(observed, batch, moments, slot)
        expected.update(batch)
        assert_accumulators(observed, expected)
    # Initial means/moments/bounds must own their data, including after merging.
    for name in ['mean', 'm2', 'minimum', 'maximum']:
        assert not np.shares_memory(getattr(observed, name), getattr(moments, name))


@pytest.mark.parametrize('count', [1, 2, 3, 8, 32])
def test_signed_zeros_and_subnormals(count):
    tiny = np.nextafter(0., 1.)
    pattern = np.array([[0., -0., tiny], [-0., 0., -tiny], [-0., -0., 0.]])
    points = np.tile(pattern, (128 * count // 3 + 1, 1))[:128 * count].copy()
    counts = np.full(128, count, dtype=np.int64)
    starts = np.arange(128) * count
    moments = maps._stream_group_moments(points, starts, counts)
    assert moments is not None
    for slot, start in enumerate(starts):
        expected = ScalarAccumulator().update(points[start:start + count])
        for name in ['mean', 'm2', 'minimum', 'maximum']:
            assert_bits(getattr(moments, name)[slot], getattr(expected, name))


@pytest.mark.parametrize('selected', [127, 128])
def test_minimum_selected_groups(selected):
    counts = np.array([1] * selected + [33] * (129 - selected), dtype=np.int64)
    starts = np.cumsum(counts) - counts
    points = np.arange(sum(counts) * 3, dtype=np.float64).reshape(-1, 3)
    moments = maps._stream_group_moments(points, starts, counts)
    if selected == 127:
        assert moments is None
    else:
        assert moments is not None and np.count_nonzero(moments.selected) == 128


def test_same_size_and_width_boundaries_leave_scalar_groups():
    counts = np.array([1] * 7 + [2] * 8 + [128] * 128 + [129] * 128, dtype=np.int64)
    starts = np.cumsum(counts) - counts
    points = np.zeros((sum(counts), 3))
    moments = maps._stream_group_moments(points, starts, counts)
    assert moments is not None
    np.testing.assert_array_equal(moments.selected, (counts == 2) | (counts == 128))


@pytest.mark.parametrize('size', [1023, 1024])
def test_chunk_point_boundary(size):
    points = np.ones((size, 3))
    counts = np.ones(128, dtype=np.int64)
    assert maps._stream_moments_batch_allowed(points, counts) is (size == 1024)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, 1e151, -1e151])
def test_unsafe_points_retain_scalar_path(value):
    points = np.ones((1024, 3))
    points[0, 0] = value
    assert not maps._stream_moments_batch_allowed(points, np.ones(128, dtype=np.int64))


@pytest.mark.parametrize('settings', [{'under': 'warn'}, {'under': 'raise'}, {'over': 'raise'},
                                     {'over': 'ignore'}, {'invalid': 'raise'}, {'divide': 'ignore'}])
def test_explicit_error_policy_retains_scalar_path(settings):
    with np.errstate(**settings):
        assert not maps._stream_moments_batch_allowed(np.ones((1024, 3)), np.ones(128, dtype=np.int64))


@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'reversed'])
def test_public_accumulator_refactor_keeps_every_field(layout):
    rng = np.random.default_rng(608)
    storage = rng.normal(size=(280, 6))
    points = storage[:, ::2] if layout == 'strided' else storage[:, :3].copy()
    if layout == 'reversed':
        points = points[::-1]
    observed, expected = PointAccumulator(), ScalarAccumulator()
    for batch in [np.empty((0, 3)), *np.array_split(points, 7), np.array([[np.nan, 0., 0.]])]:
        assert observed.update(batch) is observed
        expected.update(batch)
        assert_accumulators(observed, expected)
    for name in ['mean', 'covariance', 'minimum', 'maximum']:
        assert_bits(observed.finalize()[name], expected.finalize()[name])


def test_window_buffers_released_before_next_window_and_chunk(monkeypatch):
    refs = []
    original = maps._stream_group_moments
    def traced(*args):
        assert all(reference() is None for reference in refs)
        result = original(*args)
        assert result is not None
        refs.extend(weakref.ref(getattr(result, name)) for name in ['selected', 'mean', 'm2', 'minimum', 'maximum'])
        assert len(result.selected) <= 1024
        return result
    monkeypatch.setattr(maps, '_stream_group_moments', traced)
    points = np.zeros((3072, 3))
    points[:, 0] = np.arange(3072) + .1
    def chunks():
        yield points
        assert all(reference() is None for reference in refs)
        yield points
        assert all(reference() is None for reference in refs)
    result = maps.build_voxel_gaussians_from_chunks(chunks(), 1.)
    assert len(result) == 3072 and all(voxel.num_points == 2 for voxel in result.values())


def test_batch_backend_failure_does_not_consume_next_chunk(monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError('moment backend failed')
    monkeypatch.setattr(np, 'mean', failed)
    points = np.zeros((1024, 3))
    points[:, 0] = np.arange(1024) + .1
    def chunks():
        yield points
        raise AssertionError('next chunk consumed')
    with pytest.raises(RuntimeError, match='moment backend failed'):
        maps.build_voxel_gaussians_from_chunks(chunks(), 1.)


@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'reversed'])
def test_complete_streaming_report_and_all_gaussian_bits(layout, monkeypatch):
    rng = np.random.default_rng(793)
    points = np.repeat(np.column_stack((np.arange(1536), np.zeros((1536, 2)))), 2, axis=0)
    points += rng.uniform(.1, .2, points.shape)
    estimated = points + np.array([.005, .002, .001])
    if layout == 'strided':
        reference_storage, estimated_storage = np.empty((len(points), 6)), np.empty((len(points), 6))
        reference_storage[:, ::2], estimated_storage[:, ::2] = points, estimated
        points, estimated = reference_storage[:, ::2], estimated_storage[:, ::2]
    elif layout == 'reversed':
        points, estimated = points[::-1], estimated[::-1]
    for values in [points, estimated]:
        observed = maps.build_voxel_gaussians_from_chunks([values, values], 1.)
        expected = scalar_builder([values, values], 1.)
        assert list(observed) == list(expected)
        for key in expected:
            assert observed[key].num_points == expected[key].num_points
            assert_bits(observed[key].mu, expected[key].mu)
            assert_bits(observed[key].sigma, expected[key].sigma)
    protocol = maps.MapEvalProtocol(voxel_size_m=1., min_voxel_points=1, neighbor_radius=2)
    observed = maps.evaluate_map_streaming([estimated, estimated], [points, points], protocol=protocol)
    monkeypatch.setattr(maps, 'build_voxel_gaussians_from_chunks', scalar_builder)
    expected = maps.evaluate_map_streaming([estimated, estimated], [points, points], protocol=protocol)
    assert asdict(observed) == asdict(expected)
