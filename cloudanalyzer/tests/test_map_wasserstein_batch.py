"""Bounded Gaussian batches retain scalar distance bits and report behavior."""

from dataclasses import asdict
from pathlib import Path
import warnings

import numpy as np
import pytest

import ca.core.map_evaluate as maps


def scalar(pairs):
    return np.asarray([maps.wasserstein_distance_gaussian(a.mu, a.sigma, b.mu, b.sigma)
                       for a, b in pairs], dtype=np.float64)


def pairs(count, scale=1., layout='contiguous'):
    rng = np.random.default_rng(701)
    means = rng.normal(size=(count, 2, 3)) * scale
    factors = rng.normal(size=(count, 2, 3, 3))
    covariances = (factors @ factors.transpose(0, 1, 3, 2)) * scale
    def shaped(value):
        if layout == 'strided':
            storage = np.zeros((*value.shape[:-1], 2 * value.shape[-1]))
            storage[..., ::2] = value
            return storage[..., ::2]
        if layout == 'reversed':
            return value[..., ::-1].copy()[..., ::-1]
        return value.copy()
    return [(maps.VoxelGaussian(shaped(mu[0]), shaped(sigma[0]), 100),
             maps.VoxelGaussian(shaped(mu[1]), shaped(sigma[1]), 100))
            for mu, sigma in zip(means, covariances, strict=True)]


@pytest.mark.parametrize('scale', [1e-300, 1e-20, 1e-6, 1., 1e30])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'reversed'])
def test_batch_distance_bits_and_unchanged_arrays(scale, layout):
    values = pairs(256, scale, layout)
    before = [(a.mu.copy(), a.sigma.copy(), b.mu.copy(), b.sigma.copy()) for a, b in values]
    expected = scalar(values)
    observed = maps._wasserstein_distances_batch(values)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))
    for (a, b), snapshots in zip(values, before, strict=True):
        for array, snapshot in zip([a.mu, a.sigma, b.mu, b.sigma], snapshots, strict=True):
            np.testing.assert_array_equal(array.view(np.uint64), snapshot.view(np.uint64))


@pytest.mark.parametrize('count', [0, 1, 127, 128, 129, 255, 256, 257, 512, 513])
def test_support_order_and_batch_boundaries(count, monkeypatch):
    values = pairs(count)
    keys = [(int(index), -int(index), 2) for index in np.random.default_rng(702).permutation(count)]
    estimated = {key: a for key, (a, b) in zip(keys, values, strict=True)}
    reference = {key: b for key, (a, b) in reversed(list(zip(keys, values, strict=True)))}
    # Unsupported and missing-reference cells must not enter any batch.
    estimated[(10000, 0, 0)] = maps.VoxelGaussian(np.zeros(3), np.eye(3), 99)
    reference[(10000, 0, 0)] = values[0][1] if values else estimated[(10000, 0, 0)]
    estimated[(10001, 0, 0)] = maps.VoxelGaussian(np.zeros(3), np.eye(3), 100)
    estimated[(10002, 0, 0)] = maps.VoxelGaussian(np.zeros(3), np.eye(3), 100)
    reference[(10002, 0, 0)] = maps.VoxelGaussian(np.zeros(3), np.eye(3), 99)
    original, batches = maps._wasserstein_distances_batch, []
    def traced(block):
        batches.append(len(block))
        return original(block)
    monkeypatch.setattr(maps, '_wasserstein_distances_batch', traced)
    observed = list(maps._voxel_wasserstein_distances(estimated, reference, 100))
    assert [key for key, _ in observed] == keys
    np.testing.assert_array_equal(np.asarray([value for _, value in observed]).view(np.uint64), scalar(values).view(np.uint64))
    assert batches == [256] * (count // 256) + ([count % 256] if count % 256 else [])


@pytest.mark.parametrize('kind', ['identity', 'near_identity', 'singular', 'indefinite', 'asymmetric'])
def test_identity_regularization_and_rank_deficient_bits(kind, monkeypatch):
    values = pairs(128)
    for a, b in values:
        if kind in ['identity', 'near_identity']:
            b.mu, b.sigma = a.mu.copy(), a.sigma.copy()
            if kind == 'near_identity':
                b.mu += 1e-14
                b.sigma += 1e-14
        elif kind == 'singular':
            a.sigma = np.outer(a.mu, a.mu)
            b.sigma = np.zeros((3, 3))
        elif kind == 'indefinite':
            a.sigma -= 10 * np.eye(3)
        else:
            a.sigma[0, 1] += 3.
    expected = scalar(values)
    if kind in ['identity', 'near_identity']:
        def forbidden(*args, **kwargs):
            raise AssertionError('identity unnecessarily decomposed')
        monkeypatch.setattr(np.linalg, 'eigh', forbidden)
    np.testing.assert_array_equal(maps._wasserstein_distances_batch(values).view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, 1e101, -1e101])
def test_nonfinite_and_extreme_batches_keep_scalar_warnings(value, monkeypatch):
    values = pairs(128)
    values[17][0].mu[0] = value
    def forbidden(*args, **kwargs):
        raise AssertionError('unsafe batch decomposed')
    monkeypatch.setattr(maps, '_regularized_covariances_batch', forbidden)
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter('always')
        expected = scalar(values)
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter('always')
        observed = maps._wasserstein_distances_batch(values)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))
    assert [(item.category, str(item.message)) for item in new] == [(item.category, str(item.message)) for item in old]


@pytest.mark.parametrize('option,value', [('under', 'raise'), ('under', 'warn'), ('over', 'raise'),
                                       ('invalid', 'raise'), ('divide', 'raise'), ('over', 'ignore')])
def test_nondefault_error_policy_uses_ordered_scalar(option, value, monkeypatch):
    values = pairs(128)
    def forbidden(*args, **kwargs):
        raise AssertionError('explicit error policy used batch')
    monkeypatch.setattr(maps, '_regularized_covariances_batch', forbidden)
    with np.errstate(**{option: value}):
        policy = np.geterr().copy()
        expected, observed = scalar(values), maps._wasserstein_distances_batch(values)
        assert np.geterr() == policy
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize('kind', ['float32', 'lists'])
def test_nonstandard_arrays_retain_scalar_coercion(kind, monkeypatch):
    values = pairs(128)
    a = values[17][0]
    a.mu = a.mu.astype(np.float32) if kind == 'float32' else a.mu.tolist()
    a.sigma = a.sigma.astype(np.float32) if kind == 'float32' else a.sigma.tolist()
    def forbidden(*args, **kwargs):
        raise AssertionError('nonstandard arrays used batch')
    monkeypatch.setattr(maps, '_regularized_covariances_batch', forbidden)
    np.testing.assert_array_equal(maps._wasserstein_distances_batch(values).view(np.uint64), scalar(values).view(np.uint64))


def test_backend_errors_propagate(monkeypatch):
    def failing(*args, **kwargs):
        raise np.linalg.LinAlgError('Gaussian eigendecomposition failed')
    monkeypatch.setattr(np.linalg, 'eigh', failing)
    with pytest.raises(np.linalg.LinAlgError, match='Gaussian eigendecomposition failed'):
        maps._wasserstein_distances_batch(pairs(128))


def test_resident_streaming_reports_and_complete_plys_match_scalar(tmp_path, monkeypatch):
    rng = np.random.default_rng(703)
    points = np.concatenate([np.asarray(key) + rng.uniform(.1, .8, (8, 3))
                             for key in np.ndindex(7, 7, 7)])
    reference = points + [.02, -.01, .03]
    protocol = maps.MapEvalProtocol(voxel_size_m=1., min_voxel_points=4, neighbor_radius=2)
    request = maps.MapEvaluateRequest(points, reference, map_eval_protocol=protocol, artifact_dir=str(tmp_path))
    observed = maps.evaluate_map(request)
    artifacts = {key: Path(path).read_bytes() for key, path in observed.artifacts.items() if key.endswith('_ply')}
    def chunks(array):
        return (array[start:start+101] for start in range(0, len(array), 101))
    observed_stream = maps.evaluate_map_streaming(chunks(points), chunks(reference), protocol=protocol)
    monkeypatch.setattr(maps, '_wasserstein_distances_batch', scalar)
    expected = maps.evaluate_map(request)
    expected_stream = maps.evaluate_map_streaming(chunks(points), chunks(reference), protocol=protocol)
    assert asdict(observed) == asdict(expected)
    assert asdict(observed_stream) == asdict(expected_stream)
    assert observed.metrics['n_awd_voxels'] == 343
    for key, content in artifacts.items():
        assert Path(expected.artifacts[key]).read_bytes() == content
