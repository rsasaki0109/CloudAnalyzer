"""Tests for ca.convert module."""

import open3d as o3d
import pytest

from ca.convert import convert


class TestConvert:
    def test_pcd_to_ply(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "out.ply")
        result = convert(sample_pcd_file, output)
        assert result["num_points"] == 100
        assert result["input_format"] == ".pcd"
        assert result["output_format"] == ".ply"
        pcd = o3d.io.read_point_cloud(output)
        assert len(pcd.points) == 100

    def test_ply_to_pcd(self, tmp_path, simple_pcd):
        ply_path = str(tmp_path / "input.ply")
        o3d.io.write_point_cloud(ply_path, simple_pcd)
        output = str(tmp_path / "out.pcd")
        result = convert(ply_path, output)
        assert result["output_format"] == ".pcd"

    def test_unsupported_output_format(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "out.xyz")
        with pytest.raises(ValueError, match="Unsupported output format"):
            convert(sample_pcd_file, output)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            convert("/no/file.pcd", str(tmp_path / "out.ply"))

    def test_creates_parent_dirs(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "sub" / "dir" / "out.ply")
        convert(sample_pcd_file, output)
        pcd = o3d.io.read_point_cloud(output)
        assert len(pcd.points) == 100
