"""Tests for ca.normals module."""

import open3d as o3d
import pytest

from ca.normals import estimate_normals


class TestEstimateNormals:
    def test_normals_estimated(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "with_normals.ply")
        result = estimate_normals(sample_pcd_file, output)
        assert result["num_points"] == 100
        pcd = o3d.io.read_point_cloud(output)
        assert pcd.has_normals()

    def test_custom_params(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "normals.ply")
        result = estimate_normals(sample_pcd_file, output, radius=1.0, max_nn=50)
        assert result["radius"] == 1.0
        assert result["max_nn"] == 50

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            estimate_normals("/no/file.pcd", str(tmp_path / "out.ply"))
