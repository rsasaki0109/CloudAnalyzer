"""Regression guard: voxel-based ground evaluation stays vectorized.

Phase 18 replaced ``_voxel_keys`` and its set-intersection callers with
``np.unique(axis=0)`` + ``np.intersect1d`` on a void-view. The previous code
built a Python ``set`` of tuples via ``{(int(row[0]), int(row[1]),
int(row[2])) for row in indices}`` four times per evaluation, which dominated
runtime on city-scale ground segmentation QA.

The wall-clock ceilings here are loose — they catch regressions to per-row
Python loops (which would push the test into many seconds) without flaking on
slow CI workers.
"""

from __future__ import annotations

import time

import numpy as np

from ca.core.ground_evaluate import (
    GroundEvaluateRequest,
    VoxelConfusionGroundEvaluateStrategy,
    _voxel_intersection_size,
    _voxel_keys,
    evaluate_ground,
)


# --------------------------------------------------------- equivalence guards


def _set_voxel_keys(points: np.ndarray, voxel_size: float) -> set[tuple[int, int, int]]:
    """Naive reference: the original set-of-tuples implementation."""
    if points.shape[0] == 0:
        return set()
    indices = np.floor(points / voxel_size).astype(np.int64)
    return {(int(row[0]), int(row[1]), int(row[2])) for row in indices}


def test_voxel_keys_matches_set_reference() -> None:
    rng = np.random.default_rng(101)
    points = rng.uniform(-3.0, 3.0, size=(5_000, 3))

    keys = _voxel_keys(points, voxel_size=0.5)
    expected = _set_voxel_keys(points, voxel_size=0.5)

    assert keys.dtype == np.int64
    assert keys.shape[1] == 3
    actual = {(int(r[0]), int(r[1]), int(r[2])) for r in keys}
    assert actual == expected


def test_voxel_keys_empty_input() -> None:
    keys = _voxel_keys(np.empty((0, 3), dtype=np.float64), voxel_size=0.5)
    assert keys.shape == (0, 3)
    assert keys.dtype == np.int64


def test_voxel_intersection_size_matches_set() -> None:
    rng = np.random.default_rng(102)
    a_pts = rng.uniform(-2.0, 2.0, size=(2_000, 3))
    b_pts = rng.uniform(-2.0, 2.0, size=(2_000, 3))

    a = _voxel_keys(a_pts, voxel_size=0.4)
    b = _voxel_keys(b_pts, voxel_size=0.4)

    expected = len(_set_voxel_keys(a_pts, 0.4) & _set_voxel_keys(b_pts, 0.4))
    assert _voxel_intersection_size(a, b) == expected


def test_voxel_intersection_size_handles_empty() -> None:
    a = _voxel_keys(np.random.default_rng(1).uniform(-1, 1, size=(50, 3)), 0.5)
    empty = np.empty((0, 3), dtype=np.int64)
    assert _voxel_intersection_size(a, empty) == 0
    assert _voxel_intersection_size(empty, a) == 0
    assert _voxel_intersection_size(empty, empty) == 0


def test_voxel_intersection_size_identical_inputs() -> None:
    rng = np.random.default_rng(103)
    pts = rng.uniform(-1.0, 1.0, size=(300, 3))
    keys = _voxel_keys(pts, voxel_size=0.2)
    assert _voxel_intersection_size(keys, keys) == keys.shape[0]


# ------------------------------------------------------------- wall-clock guard


def test_voxel_confusion_ground_evaluate_under_wall_clock() -> None:
    """City-scale ground evaluation must finish well under a second.

    With the original set-of-tuples comprehensions, this case ran for many
    seconds because four Python loops walked every voxel index.
    """
    rng = np.random.default_rng(7)
    n = 200_000
    estimated_ground = rng.uniform(-50.0, 50.0, size=(n, 3))
    estimated_nonground = rng.uniform(-50.0, 50.0, size=(n, 3))
    reference_ground = rng.uniform(-50.0, 50.0, size=(n, 3))
    reference_nonground = rng.uniform(-50.0, 50.0, size=(n, 3))

    request = GroundEvaluateRequest(
        estimated_ground=estimated_ground,
        estimated_nonground=estimated_nonground,
        reference_ground=reference_ground,
        reference_nonground=reference_nonground,
        voxel_size=0.5,
    )

    start = time.perf_counter()
    result = evaluate_ground(request, strategy=VoxelConfusionGroundEvaluateStrategy())
    elapsed = time.perf_counter() - start

    assert result.strategy == "voxel_confusion"
    # Sanity: confusion counts populated.
    assert result.tp >= 0 and result.fp >= 0 and result.fn >= 0 and result.tn >= 0
    assert elapsed < 5.0, (
        f"voxel_confusion ground_evaluate on 4x200k points took {elapsed:.2f}s "
        "— regression to per-row Python set comprehension?"
    )


def test_voxel_confusion_matches_naive_reference() -> None:
    """The vectorized strategy must give the same confusion counts as the
    original set-based implementation."""
    rng = np.random.default_rng(99)
    estimated_ground = rng.uniform(-5.0, 5.0, size=(1_000, 3))
    estimated_nonground = rng.uniform(-5.0, 5.0, size=(1_000, 3))
    reference_ground = rng.uniform(-5.0, 5.0, size=(1_000, 3))
    reference_nonground = rng.uniform(-5.0, 5.0, size=(1_000, 3))

    request = GroundEvaluateRequest(
        estimated_ground=estimated_ground,
        estimated_nonground=estimated_nonground,
        reference_ground=reference_ground,
        reference_nonground=reference_nonground,
        voxel_size=0.5,
    )
    result = evaluate_ground(request, strategy=VoxelConfusionGroundEvaluateStrategy())

    # Naive reference via the original set-of-tuples path.
    eg = _set_voxel_keys(estimated_ground, 0.5)
    en = _set_voxel_keys(estimated_nonground, 0.5)
    rg = _set_voxel_keys(reference_ground, 0.5)
    rn = _set_voxel_keys(reference_nonground, 0.5)

    assert result.tp == len(eg & rg)
    assert result.fp == len(eg & rn)
    assert result.fn == len(en & rg)
    assert result.tn == len(en & rn)
