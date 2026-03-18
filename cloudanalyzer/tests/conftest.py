"""Shared fixtures for CloudAnalyzer tests."""

import numpy as np
import open3d as o3d
import pytest


@pytest.fixture
def simple_pcd():
    """A small point cloud with 100 points in a unit cube."""
    pcd = o3d.geometry.PointCloud()
    rng = np.random.default_rng(42)
    pcd.points = o3d.utility.Vector3dVector(rng.random((100, 3)))
    return pcd


@pytest.fixture
def shifted_pcd(simple_pcd):
    """Same shape as simple_pcd but shifted by (0.1, 0, 0)."""
    points = np.asarray(simple_pcd.points).copy()
    points[:, 0] += 0.1
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


@pytest.fixture
def identical_pcd(simple_pcd):
    """Exact copy of simple_pcd."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(simple_pcd.points).copy())
    return pcd


@pytest.fixture
def sample_pcd_file(tmp_path, simple_pcd):
    """Write a PCD file and return its path."""
    path = tmp_path / "sample.pcd"
    o3d.io.write_point_cloud(str(path), simple_pcd)
    return str(path)


@pytest.fixture
def source_and_target_files(tmp_path, simple_pcd, shifted_pcd):
    """Write source and target PCD files, return (source_path, target_path)."""
    src = tmp_path / "source.pcd"
    tgt = tmp_path / "target.pcd"
    o3d.io.write_point_cloud(str(src), simple_pcd)
    o3d.io.write_point_cloud(str(tgt), shifted_pcd)
    return str(src), str(tgt)
