"""Tests for MME (Mean Map Entropy) computation."""

import numpy as np
import pytest

from ca.mme import compute_mme


def _make_pcd_file(tmp_path, points: np.ndarray) -> str:
    """Write a small PCD file from a numpy array."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    path = str(tmp_path / "cloud.pcd")
    o3d.io.write_point_cloud(path, pcd)
    return path


def test_compute_mme_basic(tmp_path):
    """Output dict has required keys and correct types."""
    rng = np.random.default_rng(0)
    pts = rng.random((200, 3))
    path = _make_pcd_file(tmp_path, pts)

    result = compute_mme(path)

    assert isinstance(result["mme"], float)
    assert isinstance(result["k_neighbors"], int)
    assert isinstance(result["num_points"], int)
    assert isinstance(result["num_points_used"], int)
    assert isinstance(result["sampled"], bool)
    assert result["path"] == path


def test_mme_lower_for_structured(tmp_path):
    """Grid cloud has lower MME than random cloud."""
    # Structured: flat plane with tiny noise (very low entropy)
    xs = np.linspace(0, 5, 20)
    ys = np.linspace(0, 5, 20)
    grid_pts = np.array([[x, y, 0.0] for x in xs for y in ys])
    grid_pts = np.tile(grid_pts[:100], (3, 1)) + np.random.default_rng(1).normal(0, 0.001, (300, 3))
    grid_path_str = str(tmp_path / "grid.pcd")
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(grid_pts)
    o3d.io.write_point_cloud(grid_path_str, pcd)

    # Random: uniform random (high entropy)
    rand_pts = np.random.default_rng(42).uniform(-5, 5, (300, 3))
    rand_dir = tmp_path / "rand"
    rand_dir.mkdir()
    rand_path = _make_pcd_file(rand_dir, rand_pts)

    grid_mme = compute_mme(grid_path_str)["mme"]
    rand_mme = compute_mme(rand_path)["mme"]

    assert grid_mme < rand_mme, f"Grid MME {grid_mme:.4f} should be < random MME {rand_mme:.4f}"


def test_mme_sampling(tmp_path):
    """Cloud larger than max_points triggers sampling."""
    rng = np.random.default_rng(0)
    pts = rng.random((1000, 3))
    path = _make_pcd_file(tmp_path, pts)

    result = compute_mme(path, max_points=500)

    assert result["sampled"] is True
    assert result["num_points"] == 1000
    assert result["num_points_used"] == 500


def test_mme_no_sampling(tmp_path):
    """Cloud smaller than max_points is not sampled."""
    rng = np.random.default_rng(0)
    pts = rng.random((200, 3))
    path = _make_pcd_file(tmp_path, pts)

    result = compute_mme(path, max_points=500)

    assert result["sampled"] is False
    assert result["num_points_used"] == result["num_points"]


def test_mme_custom_k(tmp_path):
    """Custom k_neighbors produces a valid float result."""
    rng = np.random.default_rng(0)
    pts = rng.random((200, 3))
    path = _make_pcd_file(tmp_path, pts)

    result = compute_mme(path, k_neighbors=5)

    assert result["k_neighbors"] == 5
    assert np.isfinite(result["mme"])


def test_mme_invalid_k(tmp_path):
    """k_neighbors < 4 raises ValueError."""
    rng = np.random.default_rng(0)
    pts = rng.random((200, 3))
    path = _make_pcd_file(tmp_path, pts)

    with pytest.raises(ValueError, match="k_neighbors"):
        compute_mme(path, k_neighbors=3)
