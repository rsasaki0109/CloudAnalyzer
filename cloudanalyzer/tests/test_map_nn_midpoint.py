"""Large-map midpoint trees retain the existing exact distance contract."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

import ca.core.map_evaluate as maps


def balanced_nn(a, b):
    if not len(a):
        return np.zeros(0, dtype=np.float64)
    if not len(b):
        return np.full(len(a), np.inf, dtype=np.float64)
    return cKDTree(np.asarray(b, dtype=np.float64)).query(np.asarray(a, dtype=np.float64), k=1,
        workers=-1 if len(a) >= maps._PARALLEL_NN_MIN_POINTS else 1)[0]


@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'reversed'])
@pytest.mark.parametrize('scale', [1e-100, 1., 1e100])
def test_midpoint_distances_keep_bits_ties_and_input_ownership(layout, scale, monkeypatch):
    rng = np.random.default_rng(901)
    source, target = rng.normal(size=(513, 3)) * scale, rng.normal(size=(257, 3)) * scale
    target[1] = target[0]
    target[2:4] = [[-scale, 0., 0.], [scale, 0., 0.]]
    source[0] = 0.
    if layout == 'strided':
        a, b = np.zeros((len(source), 6)), np.zeros((len(target), 6))
        a[:, ::2], b[:, ::2] = source, target
        source, target = a[:, ::2], b[:, ::2]
    elif layout == 'reversed':
        source, target = source[::-1], target[::-1]
    before = source.copy(), target.copy()
    monkeypatch.setattr(maps, '_MIDPOINT_NN_MIN_POINTS', 1)
    monkeypatch.setattr(maps, '_PARALLEL_NN_MIN_POINTS', 1)
    for a, b in [(source, target), (target, source)]:
        observed = maps._min_distances_kdtree(a, b)
        np.testing.assert_array_equal(observed.view(np.uint64), balanced_nn(a, b).view(np.uint64))
        observed[:] = -1.
        np.testing.assert_array_equal(maps._min_distances_kdtree(a, b).view(np.uint64), balanced_nn(a, b).view(np.uint64))
    for array, snapshot in zip([source, target], before, strict=True):
        np.testing.assert_array_equal(array.view(np.uint64), snapshot.view(np.uint64))


@pytest.mark.parametrize('query_count,target_count', [(0, 65536), (65536, 0), (1, 65536), (65536, 1),
    (65535, 65536), (65536, 65535), (65536, 65536), (65537, 65537)])
def test_actual_two_map_boundary_and_existing_worker_policy(query_count, target_count, monkeypatch):
    import scipy.spatial
    calls = []
    class TracedTree:
        def __init__(self, points, **kwargs):
            calls.append(('tree', points.shape, kwargs))
        def query(self, points, **kwargs):
            calls.append(('query', points.shape, kwargs))
            return np.zeros(len(points)), np.zeros(len(points), dtype=np.intp)
    monkeypatch.setattr(scipy.spatial, 'cKDTree', TracedTree)
    observed = maps._min_distances_kdtree(np.zeros((query_count, 3)), np.zeros((target_count, 3)))
    if not query_count or not target_count:
        assert calls == []
        assert observed.shape == (query_count,)
        assert not target_count or observed.size == 0
    else:
        assert calls == [('tree', (target_count, 3), {'balanced_tree': False} if min(query_count, target_count) >= 65536 else {}),
            ('query', (query_count, 3), {'k': 1, 'workers': -1 if query_count >= 65536 else 1})]


@pytest.mark.parametrize('kind', ['duplicates', 'line', 'plane', 'anisotropic', 'uneven'])
def test_degenerate_and_uneven_maps_match_balanced(kind, monkeypatch):
    rng = np.random.default_rng(902)
    target = rng.normal(size=(500, 3))
    source = rng.normal(size=(300, 3))
    if kind == 'duplicates':
        target[:] = [1., 2., 3.]
    elif kind == 'line':
        target[:, 1:] = 0.
    elif kind == 'plane':
        target[:, 2] = 0.
    elif kind == 'anisotropic':
        target *= [1e-100, 1., 1e100]
        source *= [1e-100, 1., 1e100]
    else:
        target[:490] *= 1e-12
        target[490:] *= 1e12
    monkeypatch.setattr(maps, '_MIDPOINT_NN_MIN_POINTS', 1)
    for a, b in [(source, target), (target, source)]:
        np.testing.assert_array_equal(maps._min_distances_kdtree(a, b).view(np.uint64), balanced_nn(a, b).view(np.uint64))


@pytest.mark.parametrize('side', ['source', 'target'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_errors_propagate(side, value, monkeypatch):
    a, b = np.zeros((64, 3)), np.ones((64, 3))
    (a if side == 'source' else b)[0, 0] = value
    monkeypatch.setattr(maps, '_MIDPOINT_NN_MIN_POINTS', 1)
    with pytest.raises(ValueError):
        maps._min_distances_kdtree(a, b)


def test_complete_report_and_ply_bytes_match_balanced(tmp_path, monkeypatch):
    rng = np.random.default_rng(903)
    source, target = rng.uniform(-1., 1., (500, 3)), rng.uniform(-1., 1., (480, 3))
    request = maps.MapEvaluateRequest(source, target, artifact_dir=str(tmp_path),
        map_eval_protocol=maps.MapEvalProtocol(voxel_size_m=.5, min_voxel_points=2, neighbor_radius=2))
    monkeypatch.setattr(maps, '_MIDPOINT_NN_MIN_POINTS', 1)
    observed = maps.evaluate_map(request)
    outputs = {key: Path(path).read_bytes() for key, path in observed.artifacts.items() if key.endswith('_ply')}
    monkeypatch.setattr(maps, '_min_distances_kdtree', balanced_nn)
    expected = maps.evaluate_map(request)
    assert asdict(observed) == asdict(expected)
    for key, content in outputs.items():
        assert Path(expected.artifacts[key]).read_bytes() == content
