"""Tests for ca.density_map module."""

from pathlib import Path

import pytest

from ca.density_map import density_map


class TestDensityMap:
    def test_basic(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "density.png")
        result = density_map(sample_pcd_file, output)
        assert Path(output).exists()
        assert result["num_points"] == 100
        assert result["projection_axis"] == "z"
        assert result["max_density"] > 0

    def test_x_axis(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "density_x.png")
        result = density_map(sample_pcd_file, output, axis="x")
        assert result["projection_axis"] == "x"

    def test_y_axis(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "density_y.png")
        result = density_map(sample_pcd_file, output, axis="y")
        assert result["projection_axis"] == "y"

    def test_custom_resolution(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "density.png")
        result = density_map(sample_pcd_file, output, resolution=0.1)
        assert result["resolution"] == 0.1

    def test_invalid_axis(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "density.png")
        with pytest.raises(ValueError, match="Invalid axis"):
            density_map(sample_pcd_file, output, axis="w")
