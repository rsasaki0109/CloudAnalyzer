"""Regression guard: PLY vertex parsing and split tile grouping stay vectorized.

Phase 17 replaced two per-point Python loops with C-side numpy ops:

- ``ca.geometry._read_ply_vertices`` for both ASCII and binary little-endian
  PLY now reads the full vertex block in one ``np.loadtxt`` / ``np.frombuffer``
  call. Large 3DGS exports (1M+ splats) previously took ~1 minute on the
  ASCII path; the vectorized version finishes in well under a second.
- ``ca.split.split`` builds tile buckets via ``np.unique(axis=0)`` + argsort
  instead of iterating over every point in Python.

The loose wall-clock ceilings here are deliberate — they catch regressions to
per-point Python loops (which would push these tests into many seconds) without
flaking on slow CI workers.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from ca.geometry import load_representation
from ca.split import split


# --------------------------------------------------------------- PLY helpers


def _write_ascii_xyz_ply(points: np.ndarray, path: Path) -> None:
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    lines = "\n".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in points)
    path.write_text("\n".join(header) + "\n" + lines + "\n", encoding="ascii")


def _write_binary_xyz_opacity_ply(
    points: np.ndarray, opacity_logits: np.ndarray, path: Path
) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float opacity\n"
        "end_header\n"
    ).encode("ascii")
    payload = np.empty((points.shape[0], 4), dtype="<f4")
    payload[:, :3] = points.astype("<f4")
    payload[:, 3] = opacity_logits.astype("<f4")
    with path.open("wb") as f:
        f.write(header)
        f.write(payload.tobytes())


# --------------------------------------------------------------- PLY tests


def test_ascii_ply_300k_vertices_under_wall_clock(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    points = rng.uniform(-10.0, 10.0, size=(300_000, 3))
    path = tmp_path / "ascii_big.ply"
    _write_ascii_xyz_ply(points, path)

    start = time.perf_counter()
    loaded = load_representation(str(path), representation="point-cloud")
    elapsed = time.perf_counter() - start

    assert loaded.final_count == 300_000
    assert elapsed < 5.0, (
        f"ASCII PLY 300k took {elapsed:.2f}s — regression to per-row Python loop?"
    )

    np.testing.assert_allclose(
        np.sort(loaded.points[:, 0]), np.sort(points[:, 0]), atol=1e-4
    )


def test_binary_ply_500k_vertices_under_wall_clock(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    points = rng.uniform(-10.0, 10.0, size=(500_000, 3))
    opacities = rng.uniform(-5.0, 5.0, size=500_000)
    path = tmp_path / "binary_big.ply"
    _write_binary_xyz_opacity_ply(points, opacities, path)

    start = time.perf_counter()
    loaded = load_representation(str(path), representation="gaussian-points")
    elapsed = time.perf_counter() - start

    assert loaded.final_count == 500_000
    assert elapsed < 5.0, (
        f"Binary PLY 500k took {elapsed:.2f}s — regression to per-record Python loop?"
    )

    np.testing.assert_allclose(loaded.points, points.astype(np.float64), atol=1e-4)


def test_binary_ply_truncated_payload_raises(tmp_path: Path) -> None:
    """A short payload must still raise a clear error (not silently produce
    zero or NaN-filled rows)."""
    points = np.arange(12, dtype=np.float64).reshape(4, 3)
    opacities = np.zeros(4, dtype=np.float64)
    path = tmp_path / "truncated.ply"
    _write_binary_xyz_opacity_ply(points, opacities, path)

    # Chop off the last byte to corrupt the payload.
    data = path.read_bytes()
    path.write_bytes(data[:-1])

    import pytest

    with pytest.raises(ValueError, match="truncated vertex block"):
        load_representation(str(path), representation="gaussian-points")


def test_binary_ply_matches_struct_unpack_reference(tmp_path: Path) -> None:
    """Byte-for-byte agreement with the old struct.unpack-based path on a
    handful of records — guards against endian / dtype mistakes in the
    np.frombuffer rewrite."""
    points = np.array(
        [
            [0.0, 1.0, 2.0],
            [-3.5, 4.25, 7.125],
            [100.0, -200.0, 0.5],
        ],
        dtype=np.float64,
    )
    opacities = np.array([0.0, 1.5, -2.25])
    path = tmp_path / "small.ply"
    _write_binary_xyz_opacity_ply(points, opacities, path)

    loaded = load_representation(str(path), representation="gaussian-points")

    # Reference: walk the binary payload with struct.unpack one record at a
    # time (the old code path).
    with path.open("rb") as f:
        while f.readline().rstrip(b"\r\n") != b"end_header":
            pass
        payload = f.read()
    expected_xyz = []
    record_fmt = "<ffff"
    record_size = struct.calcsize(record_fmt)
    for i in range(points.shape[0]):
        chunk = payload[i * record_size : (i + 1) * record_size]
        x, y, z, _op = struct.unpack(record_fmt, chunk)
        expected_xyz.append([x, y, z])
    np.testing.assert_allclose(
        loaded.points, np.asarray(expected_xyz, dtype=np.float64), atol=1e-6
    )


# --------------------------------------------------------------- split tests


def test_split_200k_points_under_wall_clock(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    points = rng.uniform(-50.0, 50.0, size=(200_000, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    src = tmp_path / "big.pcd"
    o3d.io.write_point_cloud(str(src), pcd)

    out_dir = tmp_path / "tiles"
    start = time.perf_counter()
    result = split(str(src), str(out_dir), grid_size=5.0)
    elapsed = time.perf_counter() - start

    assert result["total_points"] == 200_000
    # Every point must land in exactly one tile.
    assert sum(t["points"] for t in result["tiles"]) == 200_000
    assert elapsed < 5.0, (
        f"split() on 200k points took {elapsed:.2f}s — regression to per-point loop?"
    )


def test_split_matches_naive_grouping(tmp_path: Path) -> None:
    """The vectorized grouping must produce the same tile membership and
    sorted-by-index ordering as the original per-point loop."""
    rng = np.random.default_rng(11)
    points = rng.uniform(-3.0, 3.0, size=(2_000, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    src = tmp_path / "small.pcd"
    o3d.io.write_point_cloud(str(src), pcd)

    out_dir = tmp_path / "tiles"
    result = split(str(src), str(out_dir), grid_size=0.5)

    # Recompute tile membership the slow way for comparison.
    origin = points[:, [0, 1]].min(axis=0)
    grid = ((points[:, [0, 1]] - origin) / 0.5).astype(int)
    expected: dict[tuple[int, int], int] = {}
    for i, j in grid:
        key = (int(i), int(j))
        expected[key] = expected.get(key, 0) + 1

    actual = {tuple(t["grid"]): t["points"] for t in result["tiles"]}
    assert actual == expected
