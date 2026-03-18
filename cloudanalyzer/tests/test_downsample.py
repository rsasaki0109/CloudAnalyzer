"""Tests for ca.downsample module."""

import open3d as o3d

from ca.downsample import downsample


class TestDownsample:
    def test_reduces_points(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "down.pcd")
        result = downsample(sample_pcd_file, voxel_size=0.3, output=output)
        assert result["downsampled_points"] <= result["original_points"]
        assert result["original_points"] == 100
        assert result["reduction_ratio"] >= 0.0
        assert result["voxel_size"] == 0.3

    def test_output_file_created(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "down.pcd")
        downsample(sample_pcd_file, voxel_size=0.3, output=output)
        pcd = o3d.io.read_point_cloud(output)
        assert not pcd.is_empty()

    def test_tiny_voxel_preserves_most(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "down.pcd")
        result = downsample(sample_pcd_file, voxel_size=0.001, output=output)
        # Very small voxel should keep most points
        assert result["downsampled_points"] >= result["original_points"] * 0.8
