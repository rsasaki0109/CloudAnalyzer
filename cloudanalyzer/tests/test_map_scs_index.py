"""Spatial SCS lookup preserves integer neighborhoods and reduction order."""

from dataclasses import asdict
from contextlib import nullcontext
import itertools
from pathlib import Path

import numpy as np
import pytest

import ca.core.map_evaluate as maps


def legacy_neighbors(distances, radius):
    return [[distances[key] for key in maps._neighbor_indices(center, radius) if key in distances]
            for center in distances]


@pytest.mark.parametrize("radius", [1, 2, 5])
@pytest.mark.parametrize("offset", [0, 1 << 52, -(1 << 63), 1 << 100])
def test_exact_neighbor_values_and_orders(radius, offset):
    rng = np.random.default_rng(341)
    cells = [tuple(int(axis) + offset for axis in row) for row in rng.integers(-12, 13, size=(210, 3))]
    distances = {key: float(index + 1) for index, key in enumerate(cells)}
    observed = list(maps._scs_neighbor_values(distances, radius))
    assert observed == legacy_neighbors(distances, radius)
    assert (maps._scs_spatial_index(distances, radius) is not None) == (radius > 1)


@pytest.mark.parametrize("count", [63, 64, 65])
def test_actual_index_boundary(count):
    distances = {(index, 0, 0): float(index) for index in range(count)}
    assert (maps._scs_spatial_index(distances, 5) is not None) == (count >= 64)
    assert list(maps._scs_neighbor_values(distances, 5)) == legacy_neighbors(distances, 5)


def test_cube_corners_boundary_and_center_exclusion():
    keys = list(itertools.product(range(-2, 3), repeat=3))
    keys.reverse()
    distances = {key: float(index) for index, key in enumerate(keys)}
    observed = list(maps._scs_neighbor_values(distances, 2))
    assert observed == legacy_neighbors(distances, 2)
    center = list(distances).index((0, 0, 0))
    assert len(observed[center]) == 124


def test_large_local_span_keeps_legacy_integer_lookup():
    distances = {(index, 0, 0): float(index) for index in range(64)}
    distances[(1 << 52, 0, 0)] = 91.
    assert maps._scs_spatial_index(distances, 5) is not None
    assert list(maps._scs_neighbor_values(distances, 5)) == legacy_neighbors(distances, 5)
    distances[((1 << 52) + 1, 0, 0)] = 92.
    assert maps._scs_spatial_index(distances, 5) is None
    assert list(maps._scs_neighbor_values(distances, 5)) == legacy_neighbors(distances, 5)


def test_non_python_integer_keys_and_radius_retain_legacy_behavior():
    distances = {(np.int64(index), 0, 0): float(index) for index in range(70)}
    assert maps._scs_spatial_index(distances, 5) is None
    assert list(maps._scs_neighbor_values(distances, 5)) == legacy_neighbors(distances, 5)
    distances = {(index, 0, 0): float(index) for index in range(70)}
    assert maps._scs_spatial_index(distances, np.int64(5)) is None
    assert list(maps._scs_neighbor_values(distances, np.int64(5))) == legacy_neighbors(distances, np.int64(5))
    with pytest.raises(TypeError):
        list(maps._scs_neighbor_values(distances, 5.))


@pytest.mark.parametrize("values", ["zero", "tiny", "mixed", "nan"])
def test_full_structural_reduction_bits_preserve_center_order(monkeypatch, values):
    keys = list(itertools.product(range(5), range(4), range(4)))
    rng = np.random.default_rng(342)
    rng.shuffle(keys)
    count = len(keys)
    ws = np.zeros(count) if values == "zero" else np.geomspace(1e-20, 1e20, count)
    if values == "tiny":
        ws = np.linspace(0., 1e-20, count)
    if values == "nan":
        ws[::17] = np.nan
    estimated = {key: maps.VoxelGaussian(np.array([value, 0., 0.]), np.eye(3), 100)
                 for key, value in zip(keys, ws, strict=True)}
    reference = {key: maps.VoxelGaussian(np.zeros(3), np.eye(3), 100) for key in reversed(keys)}
    monkeypatch.setattr(maps, "wasserstein_distance_gaussian", lambda mu, *_: float(mu[0]))
    index = maps._scs_spatial_index
    monkeypatch.setattr(maps, "_scs_spatial_index", lambda *_: None)
    with pytest.warns(RuntimeWarning, match="Mean of empty slice") if values == "nan" else nullcontext():
        expected = maps._compute_voxel_wasserstein_metrics_from_maps(estimated, reference, min_voxel_points=100, neighbor_radius=2)
    monkeypatch.setattr(maps, "_scs_spatial_index", index)
    with pytest.warns(RuntimeWarning, match="Mean of empty slice") if values == "nan" else nullcontext():
        observed = maps._compute_voxel_wasserstein_metrics_from_maps(estimated, reference, min_voxel_points=100, neighbor_radius=2)
    for key in expected:
        assert np.float64(observed[key]).view(np.uint64) == np.float64(expected[key]).view(np.uint64)


def test_sparse_neighbors_do_not_retain_all_pairs(monkeypatch):
    import scipy.spatial
    original = scipy.spatial.cKDTree
    counts = []

    class TracedTree:
        def __init__(self, points):
            self.tree = original(points)

        def query_ball_point(self, query, *args, **kwargs):
            assert np.asarray(query).shape == (3,)
            hits = self.tree.query_ball_point(query, *args, **kwargs)
            counts.append(len(hits))
            return hits

    monkeypatch.setattr(scipy.spatial, "cKDTree", TracedTree)
    distances = {(index * 100, 0, 0): float(index) for index in range(100)}
    assert list(maps._scs_neighbor_values(distances, 5)) == [[] for _ in distances]
    assert counts == [1] * len(distances)


def test_index_query_errors_propagate(monkeypatch):
    import scipy.spatial

    class FailingTree:
        def __init__(self, points):
            pass

        def query_ball_point(self, *args, **kwargs):
            raise RuntimeError("SCS index query failed")

    monkeypatch.setattr(scipy.spatial, "cKDTree", FailingTree)
    with pytest.raises(RuntimeError, match="SCS index query failed"):
        list(maps._scs_neighbor_values({(index, 0, 0): float(index) for index in range(65)}, 5))


def test_resident_and_chunked_results_and_artifacts_match_legacy(tmp_path, monkeypatch):
    rng = np.random.default_rng(343)
    keys = list(itertools.product(range(4), repeat=3))
    source = np.concatenate([np.asarray(key) + rng.uniform(.1, .9, size=(8, 3)) for key in keys])
    reference = source + [.02, -.01, .03]
    request = maps.MapEvaluateRequest(source, reference, artifact_dir=str(tmp_path),
        map_eval_protocol=maps.MapEvalProtocol(voxel_size_m=1., min_voxel_points=3, neighbor_radius=2))
    index = maps._scs_spatial_index
    monkeypatch.setattr(maps, "_scs_spatial_index", lambda *_: None)
    expected = maps.evaluate_map(request)
    expected_files = {key: Path(path).read_bytes() for key, path in expected.artifacts.items() if key.endswith("_ply")}
    expected_stream = maps.evaluate_map_streaming([source], [reference], protocol=request.map_eval_protocol)
    monkeypatch.setattr(maps, "_scs_spatial_index", index)
    observed = maps.evaluate_map(request)
    stream = maps.evaluate_map_streaming([source], [reference], protocol=request.map_eval_protocol)
    assert asdict(observed) == asdict(expected) and asdict(stream) == asdict(expected_stream)
    assert observed.metrics["n_awd_voxels"] == 64
    for key, data in expected_files.items():
        assert Path(observed.artifacts[key]).read_bytes() == data
