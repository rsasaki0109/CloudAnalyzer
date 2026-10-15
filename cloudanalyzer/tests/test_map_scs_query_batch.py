"""Batched spatial queries retain integer neighborhoods and bounded hit lists."""

from dataclasses import asdict
from pathlib import Path
import weakref

import numpy as np
import pytest
from scipy.spatial import cKDTree

import ca.core.map_evaluate as maps


def scalar_neighbors(distances, radius):
    prepared = maps._scs_spatial_index(distances, radius)
    if prepared is None:
        return [[distances[key] for key in maps._neighbor_indices(center, radius) if key in distances]
                for center in distances]
    tree, keys, origin = prepared
    return [[distances[keys[slot]] for slot in tree.query_ball_point(
        tuple(center[axis] - origin[axis] for axis in range(3)), float(radius), p=np.inf, return_sorted=True)
        if keys[slot] != center] for center in distances]


@pytest.mark.parametrize('count', [1023, 1024, 1025, 32768, 32769])
def test_batch_threshold_hit_budget_and_scalar_fallback(count, monkeypatch):
    import scipy.spatial
    calls = []
    class TracedTree:
        def __init__(self, points):
            self.points = points
        def query_ball_point(self, query, radius, **kwargs):
            query = np.asarray(query)
            calls.append((query.shape, kwargs))
            if query.ndim == 1:
                return [int(query[0])]
            return np.asarray([[int(row[0])] for row in query], dtype=object)
    monkeypatch.setattr(scipy.spatial, 'cKDTree', TracedTree)
    distances = {(index, 0, 0): float(index) for index in range(count)}
    assert list(maps._scs_neighbor_values(distances, 5)) == [[] for _ in distances]
    size = min(256, 65536 // count)
    if count < 1024 or size < 2:
        assert calls == [((3,), {'p': np.inf, 'return_sorted': True})] * count
    else:
        assert len(calls) == (count + size - 1) // size
        for shape, options in calls:
            assert shape[1] == 3 and shape[0] <= size and shape[0] * count <= 65536
            assert options == {'p': np.inf, 'return_sorted': True, 'workers': 1}


@pytest.mark.parametrize('radius', [1, 2, 5])
@pytest.mark.parametrize('offset', [0, 1 << 52, -(1 << 63), 1 << 100])
def test_actual_integer_values_and_center_neighbor_orders(radius, offset):
    cells = list(np.ndindex(11, 10, 10))
    np.random.default_rng(1001).shuffle(cells)
    distances = {tuple(int(axis) + offset for axis in key): float(index) for index, key in enumerate(cells)}
    assert list(maps._scs_neighbor_values(distances, radius)) == scalar_neighbors(distances, radius)


def test_dense_cube_corners_exact_boundary_and_exclusion():
    cells = list(np.ndindex(11, 10, 10))
    cells.reverse()
    distances = {key: float(index) for index, key in enumerate(cells)}
    assert list(maps._scs_neighbor_values(distances, 10)) == scalar_neighbors(distances, 10)
    assert all(len(row) == 1099 for row in maps._scs_neighbor_values(distances, 10))


def test_exact_local_span_and_unsafe_span_fallback():
    distances = {(index, 0, 0): float(index) for index in range(1024)}
    distances[(1 << 52, 0, 0)] = 2.
    assert list(maps._scs_neighbor_values(distances, 5)) == scalar_neighbors(distances, 5)
    distances[((1 << 52) + 1, 0, 0)] = 3.
    assert list(maps._scs_neighbor_values(distances, 5)) == scalar_neighbors(distances, 5)


def test_previous_batch_arrays_released_before_next_query(monkeypatch):
    import scipy.spatial
    references = []
    class TracedTree:
        def __init__(self, points):
            self.tree = cKDTree(points)
        def query_ball_point(self, query, *args, **kwargs):
            assert all(reference() is None for reference in references)
            result = self.tree.query_ball_point(query, *args, **kwargs)
            references.extend([weakref.ref(query), weakref.ref(result)])
            return result
    monkeypatch.setattr(scipy.spatial, 'cKDTree', TracedTree)
    distances = {(index * 100, 0, 0): float(index) for index in range(1025)}
    assert list(maps._scs_neighbor_values(distances, 5)) == [[] for _ in distances]
    assert all(reference() is None for reference in references)


def test_query_backend_errors_propagate(monkeypatch):
    import scipy.spatial
    class FailingTree:
        def __init__(self, points):
            pass
        def query_ball_point(self, *args, **kwargs):
            raise RuntimeError('batched SCS query failed')
    monkeypatch.setattr(scipy.spatial, 'cKDTree', FailingTree)
    with pytest.raises(RuntimeError, match='batched SCS query failed'):
        list(maps._scs_neighbor_values({(index, 0, 0): float(index) for index in range(1024)}, 5))


def test_complete_resident_streaming_reports_and_plys_match_scalar(tmp_path, monkeypatch):
    rng = np.random.default_rng(1002)
    source = np.concatenate([np.asarray(key) + rng.uniform(.1, .8, (4, 3)) for key in np.ndindex(11, 10, 10)])
    reference = source + [.02, -.01, .03]
    protocol = maps.MapEvalProtocol(voxel_size_m=1., min_voxel_points=2, neighbor_radius=2)
    request = maps.MapEvaluateRequest(source, reference, map_eval_protocol=protocol, artifact_dir=str(tmp_path))
    observed = maps.evaluate_map(request)
    outputs = {key: Path(path).read_bytes() for key, path in observed.artifacts.items() if key.endswith('_ply')}
    def chunks(array):
        return (array[start:start+1001] for start in range(0, len(array), 1001))
    observed_stream = maps.evaluate_map_streaming(chunks(source), chunks(reference), protocol=protocol)
    monkeypatch.setattr(maps, '_SCS_QUERY_BATCH_MIN_VOXELS', 1 << 60)
    expected = maps.evaluate_map(request)
    expected_stream = maps.evaluate_map_streaming(chunks(source), chunks(reference), protocol=protocol)
    assert asdict(observed) == asdict(expected)
    assert asdict(observed_stream) == asdict(expected_stream)
    assert observed.metrics['n_awd_voxels'] == 1100
    for key, content in outputs.items():
        assert Path(expected.artifacts[key]).read_bytes() == content
