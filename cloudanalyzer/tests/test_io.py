"""Tests for ca.io module."""

import numpy as np
import pytest
import open3d as o3d

from ca.io import load_point_cloud, save_point_cloud, SUPPORTED_EXTENSIONS


class TestLoadPointCloud:
    def test_load_pcd(self, sample_pcd_file):
        pcd = load_point_cloud(sample_pcd_file)
        assert isinstance(pcd, o3d.geometry.PointCloud)
        assert len(pcd.points) == 100

    def test_load_ply(self, tmp_path, simple_pcd):
        path = tmp_path / "test.ply"
        o3d.io.write_point_cloud(str(path), simple_pcd)
        pcd = load_point_cloud(str(path))
        assert len(pcd.points) == 100

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_point_cloud("/nonexistent/file.pcd")

    def test_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported format"):
            load_point_cloud(str(bad_file))

    def test_empty_point_cloud(self, tmp_path):
        # Open3D can't write an empty PCD, so create a minimal valid file manually
        path = tmp_path / "empty.pcd"
        path.write_text(
            "# .PCD v0.7 - Point Cloud Data file format\n"
            "VERSION 0.7\n"
            "FIELDS x y z\n"
            "SIZE 4 4 4\n"
            "TYPE F F F\n"
            "COUNT 1 1 1\n"
            "WIDTH 0\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            "POINTS 0\n"
            "DATA ascii\n"
        )
        with pytest.raises(ValueError, match="empty"):
            load_point_cloud(str(path))

    def test_supported_extensions(self):
        assert ".pcd" in SUPPORTED_EXTENSIONS
        assert ".ply" in SUPPORTED_EXTENSIONS
        assert ".las" in SUPPORTED_EXTENSIONS
        assert ".laz" in SUPPORTED_EXTENSIONS


class TestSavePointCloud:
    def test_laz_roundtrip_preserves_coordinates(self, tmp_path, simple_pcd):
        path = str(tmp_path / "test.laz")
        save_point_cloud(path, simple_pcd)
        loaded = load_point_cloud(path)
        np.testing.assert_allclose(
            np.asarray(loaded.points),
            np.asarray(simple_pcd.points),
            atol=1e-5,
        )

    def test_las_roundtrip_preserves_coordinates(self, tmp_path, simple_pcd):
        path = str(tmp_path / "test.las")
        save_point_cloud(path, simple_pcd)
        loaded = load_point_cloud(path)
        np.testing.assert_allclose(
            np.asarray(loaded.points),
            np.asarray(simple_pcd.points),
            atol=1e-5,
        )

    def test_unsupported_format_raises(self, tmp_path, simple_pcd):
        with pytest.raises(ValueError, match="Unsupported format"):
            save_point_cloud(str(tmp_path / "out.xyz"), simple_pcd)
