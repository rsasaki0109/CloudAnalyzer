"""The optional Rust core must agree exactly with the Open3D implementations."""

import os

import numpy as np
import open3d as o3d
import pytest

cloudanalyzer_core = pytest.importorskip("cloudanalyzer_core")

from ca import _rust  # noqa: E402
from ca.metrics import compute_nn_distance  # noqa: E402


def _cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


@pytest.fixture
def without_rust(monkeypatch):
    monkeypatch.setenv("CA_DISABLE_RUST_CORE", "1")
    _rust._load.cache_clear()
    yield
    monkeypatch.delenv("CA_DISABLE_RUST_CORE")
    _rust._load.cache_clear()


@pytest.mark.skipif(
    os.environ.get("CA_DISABLE_RUST_CORE", "").strip() not in {"", "0"},
    reason="the Rust core is disabled for this run",
)
def test_core_is_used_by_default():
    _rust._load.cache_clear()
    assert _rust.core() is cloudanalyzer_core


def test_disable_switch(without_rust):
    assert _rust.core() is None


def test_nn_distance_matches_open3d(without_rust):
    rng = np.random.default_rng(0)
    source = _cloud(rng.uniform(-5, 5, (20_000, 3)) + [368_000, 3_955_000, 40])
    target = _cloud(rng.uniform(-5, 5, (30_000, 3)) + [368_000, 3_955_000, 40])
    expected = compute_nn_distance(source, target)  # Open3D (core disabled)
    rust = cloudanalyzer_core.nearest_distances(np.asarray(source.points), np.asarray(target.points))
    np.testing.assert_allclose(rust, expected, rtol=0, atol=1e-9)


def test_icp_recovers_a_shift():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 40, (30_000, 2))
    reference = np.c_[xy, 2 * np.sin(xy[:, 0] / 5) * np.cos(xy[:, 1] / 4)]
    moving = reference[::3] + [0.3, -0.2, 0.1]
    result = cloudanalyzer_core.icp(moving, reference)
    assert result["converged"]
    np.testing.assert_allclose(result["transformation"][:3, 3], [-0.3, 0.2, -0.1], atol=1e-3)


def test_read_matches_open3d_for_ply(tmp_path):
    points = np.random.default_rng(2).uniform(-1, 1, (500, 3))
    path = tmp_path / "cloud.ply"
    o3d.io.write_point_cloud(str(path), _cloud(points))
    data = cloudanalyzer_core.read(str(path))
    np.testing.assert_allclose(data["positions"], np.asarray(o3d.io.read_point_cloud(str(path)).points))


def test_las_and_laz_match_laspy(tmp_path):
    laspy = pytest.importorskip("laspy")
    rng = np.random.default_rng(3)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [368_000.0, 3_955_000.0, 0.0]
    las = laspy.LasData(header)
    n = 5_000
    las.x = 368_000 + rng.uniform(0, 100, n)
    las.y = 3_955_000 + rng.uniform(0, 100, n)
    las.z = rng.uniform(0, 30, n)
    las.intensity = rng.integers(0, 65535, n)
    las.classification = rng.integers(0, 20, n)
    for ext in ("las", "laz"):
        path = tmp_path / f"cloud.{ext}"
        try:
            las.write(str(path))
        except Exception as exc:  # no LAZ backend installed
            pytest.skip(f"cannot write .{ext}: {exc}")
        back = laspy.read(str(path))
        data = cloudanalyzer_core.read(str(path))
        np.testing.assert_array_equal(data["positions"], np.vstack([back.x, back.y, back.z]).T)
        np.testing.assert_array_equal(data["intensity"], np.asarray(back.intensity, dtype=np.float32))
        np.testing.assert_array_equal(data["classification"], np.asarray(back.classification))


def test_filters_return_indices():
    grid = np.stack(np.meshgrid(np.arange(40) * 0.1, np.arange(40) * 0.1), -1).reshape(-1, 2)
    points = np.c_[grid, np.zeros(len(grid))]
    points = np.vstack([points, [[50.0, 50.0, 50.0]]])
    keep = cloudanalyzer_core.statistical_outliers(points, k=8, ratio=1.0)
    assert len(keep) == 1600 and keep.max() < 1600
    assert len(cloudanalyzer_core.voxel_subsample(points[:1600], 0.5)) == 64


def test_volume_against_constant_and_mesh():
    x, y = (a.ravel() for a in np.meshgrid(np.arange(60) * 0.1, np.arange(60) * 0.1))
    mound = (x >= 0.95) & (x < 2.95) & (y >= 0.95) & (y < 2.95)
    after = np.c_[x, y, np.where(mound, 1.0, 0.0)]
    flat = cloudanalyzer_core.volume(0.0, after, cell=0.5)
    assert flat["added"] == pytest.approx(4.0)
    assert flat["removed"] == 0.0
    # As CloudCompare, the first point is a cell centre: 0..5.9 m in 0.5 m cells is 13 cells.
    assert flat["difference"].shape == (13, 13)
    plane = (
        np.array([[0, 0, 0], [6, 0, 0], [6, 6, 0], [0, 6, 0]], dtype=float),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
    )
    assert cloudanalyzer_core.volume(plane, after, cell=0.5)["added"] == pytest.approx(4.0)


def test_ground_csf_separates_a_box():
    x, y = (a.ravel() for a in np.meshgrid(np.arange(80) * 0.25, np.arange(80) * 0.25))
    box = (x >= 8) & (x < 12) & (y >= 8) & (y < 12)
    points = np.c_[x, y, np.where(box, 3.0, 0.0)]
    ground = cloudanalyzer_core.ground_csf(points)
    assert ground.dtype == bool
    np.testing.assert_array_equal(ground, ~box)


def test_m3c2_measures_a_rise():
    x, y = (a.ravel() for a in np.meshgrid(np.arange(100) * 0.1, np.arange(100) * 0.1))
    rng = np.random.default_rng(0)
    before = np.c_[x, y, 0.01 * rng.standard_normal(x.size)]
    after = np.c_[x, y, 0.25 + 0.01 * rng.standard_normal(x.size)]
    core = np.array([[5.0, 5.0, 0.0], [2.0, 7.0, 0.0], [50.0, 50.0, 0.0]])
    distance, lod95, significant, normals = cloudanalyzer_core.m3c2(core, before, after)
    np.testing.assert_allclose(distance[:2], 0.25, atol=0.01)
    assert np.all(lod95[:2] < 0.02) and significant[:2].all()
    np.testing.assert_allclose(normals[:2], [[0, 0, 1]] * 2, atol=0.01)
    assert np.isnan(distance[2]) and not significant[2]


def test_profile_along_a_polyline():
    x, y = (a.ravel() for a in np.meshgrid(np.arange(100) * 0.1, np.arange(100) * 0.1))
    points = np.c_[x, y, np.zeros_like(x)]
    idx, along = cloudanalyzer_core.profile(points, np.array([[1.0, 1.0], [5.0, 1.0], [5.0, 4.0]]), 0.01)
    assert len(idx) == 71
    assert along.max() == pytest.approx(7.0)
    np.testing.assert_allclose(points[idx, 1][along < 4 - 1e-9], 1.0, atol=1e-9)


def test_normals_of_a_tilted_plane():
    x, y = (a.ravel() for a in np.meshgrid(np.arange(50) * 0.1, np.arange(50) * 0.1))
    points = np.c_[x, y, 0.5 * x]
    normals = cloudanalyzer_core.normals(points, k=10)
    assert normals.shape == (2500, 3) and normals.dtype == np.float32
    np.testing.assert_allclose(normals, np.tile([-0.5, 0, 1] / np.sqrt(1.25), (2500, 1)), atol=1e-4)
