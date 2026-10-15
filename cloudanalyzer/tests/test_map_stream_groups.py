"""Stable chunk groups preserve update sequences and streaming Gaussian bits."""

from dataclasses import asdict
import hashlib
import weakref

import numpy as np
import pytest

import ca.core.map_evaluate as maps
from ca.io import PointAccumulator


def legacy_gaussians(chunks, size):
    if size <= 0:
        raise ValueError('voxel_size must be > 0.')
    accumulators = {}
    for chunk in chunks:
        pts = maps._require_xyz(np.asarray(chunk, dtype=np.float64), 'point chunk')
        if not len(pts):
            continue
        indices = np.floor(pts / size).astype(np.int64)
        unique, inverse = np.unique(indices, axis=0, return_inverse=True)
        for index, raw_key in enumerate(unique):
            key = tuple(int(value) for value in raw_key)
            accumulators.setdefault(key, PointAccumulator()).update(pts[inverse == index])
    result = {}
    for key, accumulator in accumulators.items():
        summary = accumulator.finalize()
        result[key] = maps.VoxelGaussian(summary['mean'], summary['covariance'], summary['count'])
    return result


def assert_gaussians(a, b):
    assert list(a) == list(b)
    for first, second in zip(a.values(), b.values(), strict=True):
        assert first.num_points == second.num_points
        np.testing.assert_array_equal(first.mu.view(np.uint64), second.mu.view(np.uint64))
        np.testing.assert_array_equal(first.sigma.view(np.uint64), second.sigma.view(np.uint64))


@pytest.mark.parametrize('chunk_size', [1, 7, 64, 101])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'reversed'])
def test_ordered_batches_and_all_gaussian_bits(chunk_size, layout, monkeypatch):
    rng = np.random.default_rng(921)
    storage = rng.uniform(-2., 2., size=(350, 6))
    points = storage[:, ::2] if layout == 'strided' else storage[:, :3].copy()
    if layout == 'reversed':
        points = points[::-1]
    before = points.copy()
    batches = []
    original = PointAccumulator.update
    def traced(self, values):
        batches.append((len(values), hashlib.sha256(np.asarray(values, dtype='<f8').tobytes()).hexdigest()))
        return original(self, values)
    monkeypatch.setattr(PointAccumulator, 'update', traced)
    def chunks():
        yield np.empty((0, 3))
        for start in range(0, len(points), chunk_size):
            yield points[start:start + chunk_size]
            yield np.empty((0, 3))
    legacy = legacy_gaussians(chunks(), 1.)
    expected_batches = batches.copy()
    batches.clear()
    observed = maps.build_voxel_gaussians_from_chunks(chunks(), 1.)
    assert batches == expected_batches
    assert_gaussians(observed, legacy)
    np.testing.assert_array_equal(points.view(np.uint64), before.view(np.uint64))


def test_wide_integer_range_preserves_axis_fallback():
    points = np.tile(np.array([[-1e18, -1e18, -1e18], [1e18, 1e18, 1e18]]), (40, 1))
    assert_gaussians(maps.build_voxel_gaussians_from_chunks([points], 1.), legacy_gaussians([points], 1.))


def test_nonfinite_filtering_and_zero_count_keys():
    points = np.array([[np.nan, 0., 0.], [np.inf, 1., 0.], [-np.inf, 1., 0.],
                       [0., 0., 0.], [.1, .1, .1]])
    with np.errstate(invalid='ignore'):
        assert_gaussians(maps.build_voxel_gaussians_from_chunks([points], 1.), legacy_gaussians([points], 1.))


def test_chunk_scratch_released_before_next_pull(monkeypatch):
    references = []
    original = np.argsort
    def traced(*args, **kwargs):
        result = original(*args, **kwargs)
        references.append(weakref.ref(result))
        return result
    monkeypatch.setattr(np, 'argsort', traced)
    def chunks():
        yield np.arange(300, dtype=np.float64).reshape(100, 3)
        assert references and all(reference() is None for reference in references)
        yield np.zeros((2, 3))
        assert all(reference() is None for reference in references)
    result = maps.build_voxel_gaussians_from_chunks(chunks(), 3.)
    assert sum(voxel.num_points for voxel in result.values()) == 102


def test_single_pass_iterator_and_failure_propagation():
    class Once:
        def __init__(self):
            self.iterations = 0
        def __iter__(self):
            self.iterations += 1
            assert self.iterations == 1
            yield np.zeros((65, 3))
            raise RuntimeError('source failure')
    source = Once()
    with pytest.raises(RuntimeError, match='source failure'):
        maps.build_voxel_gaussians_from_chunks(source, 1.)
    assert source.iterations == 1


@pytest.mark.parametrize('size', [0., -1.])
def test_invalid_size_does_not_consume_iterator(size):
    def forbidden():
        raise AssertionError('invalid size consumed input')
        yield
    with pytest.raises(ValueError, match='voxel_size'):
        maps.build_voxel_gaussians_from_chunks(forbidden(), size)


def test_invalid_chunk_shape():
    with pytest.raises(ValueError, match='shape'):
        maps.build_voxel_gaussians_from_chunks([np.zeros((2, 4))], 1.)


def test_complete_streaming_report_parity(monkeypatch):
    rng = np.random.default_rng(1021)
    reference = rng.uniform(-1.8, 1.8, size=(1400, 3))
    estimated = reference + rng.normal(0., .01, reference.shape)
    protocol = maps.MapEvalProtocol(voxel_size_m=1., min_voxel_points=2, neighbor_radius=2)
    def chunks(points, size):
        return (points[start:start + size] for start in range(0, len(points), size))
    observed = maps.evaluate_map_streaming(chunks(estimated, 101), chunks(reference, 97), protocol=protocol)
    monkeypatch.setattr(maps, 'build_voxel_gaussians_from_chunks', legacy_gaussians)
    expected = maps.evaluate_map_streaming(chunks(estimated, 101), chunks(reference, 97), protocol=protocol)
    assert asdict(observed) == asdict(expected)


def test_empty_chunks_are_unavailable():
    result = maps.evaluate_map_streaming([np.empty((0, 3))], [])
    assert np.isnan(result.metrics['awd_m']) and np.isnan(result.metrics['scs'])
    assert result.metrics['n_awd_voxels'] == result.metrics['n_scs_voxels'] == 0.
