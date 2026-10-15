"""Parallel SciPy map queries keep exact distances and complete result values."""

from dataclasses import asdict
import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

import ca.core.map_evaluate as maps


def serial_nn(a, b):
    if not len(a):
        return np.zeros(0, dtype=np.float64)
    if not len(b):
        return np.full(len(a), np.inf, dtype=np.float64)
    return cKDTree(np.asarray(b, dtype=np.float64)).query(np.asarray(a, dtype=np.float64), k=1, workers=1)[0]


@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed"])
@pytest.mark.parametrize("scale", [1e-100, 1., 1e100])
def test_parallel_distance_bits_and_input_ownership(monkeypatch, layout, scale):
    rng = np.random.default_rng(74)
    source, target = rng.normal(size=(513, 3)) * scale, rng.normal(size=(257, 3)) * scale
    # Duplicate hits and equal-distance ties must retain their distance bits.
    target[1] = target[0]
    target[2:4] = [[-scale, 0., 0.], [scale, 0., 0.]]
    source[0] = [0., 0., 0.]
    if layout == "strided":
        source_storage, target_storage = np.zeros((len(source), 6)), np.zeros((len(target), 6))
        source_storage[:, ::2], target_storage[:, ::2] = source, target
        source, target = source_storage[:, ::2], target_storage[:, ::2]
    elif layout == "reversed":
        source, target = source[::-1], target[::-1]
    snapshots = [np.array(points) for points in [source, target]]
    monkeypatch.setattr(maps, "_PARALLEL_NN_MIN_POINTS", 1)
    expected = serial_nn(source, target)
    observed = maps._min_distances_kdtree(source, target)
    np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))
    for points, snapshot in zip([source, target], snapshots, strict=True):
        np.testing.assert_array_equal(points.view(np.uint64), snapshot.view(np.uint64))
    observed[:] = -1
    np.testing.assert_array_equal(maps._min_distances_kdtree(source, target).view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("count", [0, 1, 65_535, 65_536, 65_537])
def test_actual_parallel_boundary_and_dispatch(monkeypatch, count):
    import scipy.spatial

    source = np.column_stack([np.arange(count, dtype=np.float64), np.zeros((count, 2))])
    target = np.array([[0., 0., 0.], [17., 0., 0.], [100_000., 0., 0.]])
    calls = []

    class TracedTree:
        def __init__(self, points):
            self.tree = cKDTree(points)

        def query(self, points, **kwargs):
            calls.append(kwargs)
            return self.tree.query(points, **kwargs)

    monkeypatch.setattr(scipy.spatial, "cKDTree", TracedTree)
    result = maps._min_distances_kdtree(source, target)
    np.testing.assert_array_equal(result.view(np.uint64), serial_nn(source, target).view(np.uint64))
    assert calls == ([{"k": 1, "workers": -1 if count >= 65_536 else 1}] if count else [])


def test_empty_reference_skips_tree_and_preserves_infinities(monkeypatch):
    import scipy.spatial

    def unexpected_tree(*args):
        pytest.fail("empty reference constructed a tree")

    monkeypatch.setattr(scipy.spatial, "cKDTree", unexpected_tree)
    result = maps._min_distances_kdtree(np.ones((65_536, 3)), np.empty((0, 3)))
    assert result.dtype == np.float64 and np.isposinf(result).all()


@pytest.mark.parametrize("bad_side", ["source", "target"])
def test_nonfinite_input_errors_propagate(monkeypatch, bad_side):
    monkeypatch.setattr(maps, "_PARALLEL_NN_MIN_POINTS", 1)
    source, target = np.zeros((3, 3)), np.ones((3, 3))
    (source if bad_side == "source" else target)[0, 0] = np.nan
    with pytest.raises(ValueError):
        maps._min_distances_kdtree(source, target)


@pytest.mark.parametrize("transform", [False, True])
@pytest.mark.parametrize("downsample", [0., .3])
def test_complete_map_results_and_ply_bytes_match_serial(tmp_path, monkeypatch, transform, downsample):
    rng = np.random.default_rng(217)
    source, target = rng.uniform(-1., 1., size=(220, 3)), rng.uniform(-1., 1., size=(180, 3))
    matrix = np.eye(4)
    matrix[:3, 3] = [.1, -.2, .03]
    request = maps.MapEvaluateRequest(estimated_points=source, reference_points=target,
        align_mode="initial" if transform else "none", initial_transform_4x4=matrix,
        downsample_voxel_size=downsample, thresholds_m=(.3, .1, .1), artifact_dir=str(tmp_path),
        map_eval_protocol=maps.MapEvalProtocol(voxel_size_m=.5, min_voxel_points=1, neighbor_radius=1,
                                              accuracy_level_m=(.3, .1, .1)))
    original = maps._min_distances_kdtree
    monkeypatch.setattr(maps, "_min_distances_kdtree", serial_nn)
    expected = maps.evaluate_map(request)
    expected_files = {key: Path(path).read_bytes() for key, path in expected.artifacts.items() if key.endswith("_ply")}
    calls = []

    def parallel_nn(a, b):
        calls.append((len(a), len(b)))
        return original(a, b)

    monkeypatch.setattr(maps, "_min_distances_kdtree", parallel_nn)
    monkeypatch.setattr(maps, "_PARALLEL_NN_MIN_POINTS", 1)
    observed = maps.evaluate_map(request)
    assert asdict(observed) == asdict(expected) and len(calls) == 2
    for key, data in expected_files.items():
        with open(observed.artifacts[key], "rb") as file:
            assert file.read() == data


def test_separate_calls_keep_both_directions_independent(monkeypatch):
    monkeypatch.setattr(maps, "_PARALLEL_NN_MIN_POINTS", 1)
    grid = np.array(list(itertools.product([0., 1., 2.], repeat=3)))
    source = grid[:17]
    for target in [grid, grid[::-1], grid[:3], grid + .01]:
        for a, b in [(source, target), (target, source)]:
            np.testing.assert_array_equal(maps._min_distances_kdtree(a, b).view(np.uint64), serial_nn(a, b).view(np.uint64))
