"""Tests for ca.crop module."""

import open3d as o3d

from ca.crop import crop


class TestCrop:
    def test_crop_keeps_subset(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "cropped.pcd")
        result = crop(sample_pcd_file, output, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5)
        assert result["cropped_points"] <= result["original_points"]
        assert result["cropped_points"] > 0
        assert result["original_points"] == 100

    def test_crop_all_inside(self, sample_pcd_file, tmp_path):
        # Points are in [0,1) range, so this should keep all
        output = str(tmp_path / "cropped.pcd")
        result = crop(sample_pcd_file, output, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0)
        assert result["cropped_points"] == 100

    def test_crop_none_inside(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "cropped.pcd")
        result = crop(sample_pcd_file, output, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0)
        assert result["cropped_points"] == 0

    def test_output_file_valid(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "cropped.pcd")
        result = crop(sample_pcd_file, output, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5)
        if result["cropped_points"] > 0:
            pcd = o3d.io.read_point_cloud(output)
            assert len(pcd.points) == result["cropped_points"]
