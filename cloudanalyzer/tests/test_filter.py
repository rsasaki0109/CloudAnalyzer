"""Tests for ca.filter module."""

import open3d as o3d

from ca.filter import filter_outliers


class TestFilterOutliers:
    def test_basic_filter(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "filtered.pcd")
        result = filter_outliers(sample_pcd_file, output)
        assert result["filtered_points"] <= result["original_points"]
        assert result["removed_points"] >= 0
        assert result["filtered_points"] + result["removed_points"] == result["original_points"]

    def test_output_file(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "filtered.pcd")
        result = filter_outliers(sample_pcd_file, output)
        pcd = o3d.io.read_point_cloud(output)
        assert len(pcd.points) == result["filtered_points"]

    def test_strict_filter(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "filtered.pcd")
        result = filter_outliers(sample_pcd_file, output, std_ratio=0.1)
        # Very strict filter should remove more points
        assert result["removed_points"] > 0

    def test_custom_params(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "filtered.pcd")
        result = filter_outliers(sample_pcd_file, output, nb_neighbors=10, std_ratio=3.0)
        assert result["nb_neighbors"] == 10
        assert result["std_ratio"] == 3.0
