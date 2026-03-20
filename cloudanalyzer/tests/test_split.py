"""Tests for ca.split module."""

import pytest
from pathlib import Path

from ca.split import split


class TestSplit:
    def test_basic_split(self, sample_pcd_file, tmp_path):
        output_dir = str(tmp_path / "tiles")
        result = split(sample_pcd_file, output_dir, grid_size=0.5)
        assert result["total_points"] == 100
        assert result["num_tiles"] > 0
        assert len(list(Path(output_dir).glob("tile_*.pcd"))) == result["num_tiles"]

    def test_large_grid_single_tile(self, sample_pcd_file, tmp_path):
        output_dir = str(tmp_path / "tiles")
        result = split(sample_pcd_file, output_dir, grid_size=100.0)
        assert result["num_tiles"] == 1

    def test_small_grid_many_tiles(self, sample_pcd_file, tmp_path):
        output_dir = str(tmp_path / "tiles")
        result = split(sample_pcd_file, output_dir, grid_size=0.1)
        assert result["num_tiles"] > 1

    def test_total_points_preserved(self, sample_pcd_file, tmp_path):
        output_dir = str(tmp_path / "tiles")
        result = split(sample_pcd_file, output_dir, grid_size=0.3)
        total = sum(t["points"] for t in result["tiles"])
        assert total == 100

    def test_invalid_axis(self, sample_pcd_file, tmp_path):
        with pytest.raises(ValueError, match="Invalid axis"):
            split(sample_pcd_file, str(tmp_path / "tiles"), grid_size=1.0, axis="abc")

    def test_xz_axis(self, sample_pcd_file, tmp_path):
        output_dir = str(tmp_path / "tiles")
        result = split(sample_pcd_file, output_dir, grid_size=0.5, axis="xz")
        assert result["axis"] == "xz"
        assert result["num_tiles"] > 0
