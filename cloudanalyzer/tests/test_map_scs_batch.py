"""Bounded SCS statistics retain scalar arithmetic, order and error policies."""

from dataclasses import asdict
import itertools
from pathlib import Path
import warnings

import numpy as np
import pytest

import ca.core.map_evaluate as maps


def term(row):
    mean = float(np.mean(row))
    if mean <= np.finfo(np.float64).eps:
        return 0. if np.allclose(row, 0.) else float('nan')
    return float(np.std(row)) / mean


def scalar(distances, radius):
    return [term(row) for row in maps._scs_neighbor_values(distances, radius) if row]


@pytest.mark.parametrize('width', [1, 2, 7, 8, 15, 16, 31, 32, 127, 128, 255, 256])
@pytest.mark.parametrize('scale', [1e-320, 1e-14, 1., 1e140])
def test_equal_length_rows_keep_mean_std_bits(width, scale):
    rng = np.random.default_rng(801)
    rows = (rng.uniform(.1, 1., size=(17, width)) * scale).tolist()
    expected = np.asarray([term(row) for row in rows])
    observed = maps._scs_terms_batch(rows)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize('count', [0, 1, 15, 16, 17])
def test_small_batch_and_mixed_sizes_keep_row_order(count):
    rows = [[float(index), 1., -0.][:index % 3 + 1] for index in range(count)]
    before = [row.copy() for row in rows]
    np.testing.assert_array_equal(maps._scs_terms_batch(rows).view(np.uint64), np.asarray([term(row) for row in rows]).view(np.uint64))
    assert rows == before


@pytest.mark.parametrize('count', [63, 64, 65, 1023, 1024, 1025])
@pytest.mark.parametrize('radius', [1, 2, 5])
def test_actual_neighborhood_bits_and_center_order(count, radius):
    order = np.random.default_rng(802).permutation(count)
    distances = {(int(index), 0, 0): float(index + 1) for index in order}
    expected = np.asarray(scalar(distances, radius))
    observed = np.asarray(list(maps._scs_terms(distances, radius)))
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize('width', [63, 64, 65, 65535, 65536, 65537])
def test_value_budget_and_large_single_neighborhood(width, monkeypatch):
    distances = {(index, 0, 0): 1. for index in range(64)}
    rows = [[1.] * width for _ in range(18)]
    monkeypatch.setattr(maps, '_scs_neighbor_values', lambda *_: iter(rows))
    original, batches = maps._scs_terms_batch, []
    def traced(block):
        batches.append((len(block), sum(len(row) for row in block)))
        return original(block)
    monkeypatch.setattr(maps, '_scs_terms_batch', traced)
    assert list(maps._scs_terms(distances, 2)) == [0.] * len(rows)
    assert all(centers <= 1024 and values <= 65536 for centers, values in batches)
    assert bool(batches) == (width <= 65536)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf, -1., 1e200])
def test_unsafe_distances_keep_scalar_warning_order(value, monkeypatch):
    distances = {(index, 0, 0): float(index + 1) for index in range(64)}
    distances[(17, 0, 0)] = float(value)
    def forbidden(*args, **kwargs):
        raise AssertionError('unsafe values used batched statistics')
    monkeypatch.setattr(maps, '_scs_terms_batch', forbidden)
    with warnings.catch_warnings(record=True) as old:
        warnings.simplefilter('always')
        expected = np.asarray(scalar(distances, 2))
    with warnings.catch_warnings(record=True) as new:
        warnings.simplefilter('always')
        observed = np.asarray(list(maps._scs_terms(distances, 2)))
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))
    assert [(item.category, str(item.message)) for item in new] == [(item.category, str(item.message)) for item in old]


@pytest.mark.parametrize('option,value', [('under', 'raise'), ('under', 'warn'), ('over', 'raise'),
                                       ('invalid', 'raise'), ('divide', 'raise'), ('over', 'ignore')])
def test_explicit_error_policy_retains_scalar_calls(option, value, monkeypatch):
    distances = {(index, 0, 0): float(index + 1) for index in range(64)}
    def forbidden(*args, **kwargs):
        raise AssertionError('explicit error policy used batch')
    monkeypatch.setattr(maps, '_scs_terms_batch', forbidden)
    with np.errstate(**{option: value}):
        policy = np.geterr().copy()
        expected = np.asarray(scalar(distances, 2))
        observed = np.asarray(list(maps._scs_terms(distances, 2)))
        assert np.geterr() == policy
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))


def test_numpy_scalar_values_retain_previous_dtype_coercion(monkeypatch):
    distances = {(index, 0, 0): np.float32(index + 1) for index in range(64)}
    def forbidden(*args, **kwargs):
        raise AssertionError('nonstandard values used batch')
    monkeypatch.setattr(maps, '_scs_terms_batch', forbidden)
    np.testing.assert_array_equal(np.asarray(list(maps._scs_terms(distances, 2))).view(np.uint64), np.asarray(scalar(distances, 2)).view(np.uint64))


def test_zero_tiny_and_mixed_mean_branches():
    rows = [[-0., 0., 0.], [1e-20] * 3, [0., 1., 0.], [np.nextafter(0., 1.)] * 3] * 4
    expected = np.asarray([term(row) for row in rows])
    np.testing.assert_array_equal(maps._scs_terms_batch(rows).view(np.uint64), expected.view(np.uint64))


def test_isolated_cells_skip_terms_and_query_errors_propagate(monkeypatch):
    distances = {(index * 100, 0, 0): 1. for index in range(64)}
    assert list(maps._scs_terms(distances, 2)) == []
    def failing(*args, **kwargs):
        raise RuntimeError('SCS query failed')
    monkeypatch.setattr(maps, '_scs_neighbor_values', failing)
    with pytest.raises(RuntimeError, match='SCS query failed'):
        list(maps._scs_terms(distances, 2))


def test_resident_streaming_reports_and_plys_keep_scalar_bits(tmp_path, monkeypatch):
    rng = np.random.default_rng(803)
    points = np.concatenate([np.asarray(key) + rng.uniform(.1, .8, (8, 3))
                             for key in itertools.product(range(7), repeat=3)])
    reference = points + [.02, -.01, .03]
    protocol = maps.MapEvalProtocol(voxel_size_m=1., min_voxel_points=4, neighbor_radius=2)
    request = maps.MapEvaluateRequest(points, reference, map_eval_protocol=protocol, artifact_dir=str(tmp_path))
    observed = maps.evaluate_map(request)
    artifacts = {key: Path(path).read_bytes() for key, path in observed.artifacts.items() if key.endswith('_ply')}
    def chunks(array):
        return (array[start:start+101] for start in range(0, len(array), 101))
    observed_stream = maps.evaluate_map_streaming(chunks(points), chunks(reference), protocol=protocol)
    monkeypatch.setattr(maps, '_scs_terms', scalar)
    expected = maps.evaluate_map(request)
    expected_stream = maps.evaluate_map_streaming(chunks(points), chunks(reference), protocol=protocol)
    assert asdict(observed) == asdict(expected)
    assert asdict(observed_stream) == asdict(expected_stream)
    assert observed.metrics['n_scs_voxels'] == 343
    for key, data in artifacts.items():
        assert Path(expected.artifacts[key]).read_bytes() == data
