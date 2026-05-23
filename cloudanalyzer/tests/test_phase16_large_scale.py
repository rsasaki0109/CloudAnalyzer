"""Regression guard: NN paths must stay vectorized.

Phase 16 vectorized `_min_distances_kdtree` (map_evaluate.nn_thresholds) and
`compute_stats` spacing computation by switching from per-point Open3D KD-tree
loops to batched `scipy.spatial.cKDTree.query`. A regression to per-point Python
loops would push 100k-point runs from a few hundred ms to tens of seconds, so a
loose wall-clock ceiling is enough to catch it.
"""

import time

import numpy as np
import open3d as o3d
import pytest

from ca.experiments.map_evaluate.nn_thresholds import _min_distances_kdtree
from ca.stats import compute_stats


def _make_cloud(num_points: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-50.0, 50.0, size=(num_points, 3))


def test_min_distances_kdtree_handles_100k_points_quickly() -> None:
    a = _make_cloud(100_000, seed=1)
    b = _make_cloud(100_000, seed=2)

    start = time.perf_counter()
    distances = _min_distances_kdtree(a, b)
    elapsed = time.perf_counter() - start

    assert distances.shape == (100_000,)
    assert np.all(np.isfinite(distances))
    assert elapsed < 5.0, f"100k-point NN took {elapsed:.2f}s (regression to per-point loop?)"


def test_min_distances_kdtree_matches_brute_force_on_small_input() -> None:
    a = _make_cloud(50, seed=10)
    b = _make_cloud(80, seed=11)

    fast = _min_distances_kdtree(a, b)

    # Brute force reference.
    diffs = a[:, None, :] - b[None, :, :]
    brute = np.sqrt((diffs ** 2).sum(axis=-1)).min(axis=1)

    np.testing.assert_allclose(fast, brute, rtol=0, atol=1e-9)


def test_min_distances_kdtree_empty_inputs() -> None:
    a = _make_cloud(0)
    b = _make_cloud(10, seed=3)
    assert _min_distances_kdtree(a, b).shape == (0,)

    a = _make_cloud(10, seed=4)
    b = _make_cloud(0)
    out = _min_distances_kdtree(a, b)
    assert out.shape == (10,)
    assert np.all(np.isinf(out))


def test_compute_stats_handles_100k_points_quickly(tmp_path) -> None:
    points = _make_cloud(100_000, seed=5)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    path = tmp_path / "large.pcd"
    o3d.io.write_point_cloud(str(path), pcd)

    start = time.perf_counter()
    result = compute_stats(str(path))
    elapsed = time.perf_counter() - start

    assert result["num_points"] == 100_000
    assert result["spacing"]["mean"] > 0
    assert elapsed < 5.0, f"compute_stats on 100k took {elapsed:.2f}s (regression to per-point loop?)"
