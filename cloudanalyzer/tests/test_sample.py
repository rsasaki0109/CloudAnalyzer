"""Tests for ca.sample module."""

import open3d as o3d
import pytest

from ca.sample import random_sample


class TestRandomSample:
    def test_basic_sample(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "sampled.pcd")
        result = random_sample(sample_pcd_file, output, 50)
        assert result["sampled_points"] == 50
        assert result["original_points"] == 100

    def test_output_file(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "sampled.pcd")
        random_sample(sample_pcd_file, output, 30)
        pcd = o3d.io.read_point_cloud(output)
        assert len(pcd.points) == 30

    def test_sample_all(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "sampled.pcd")
        result = random_sample(sample_pcd_file, output, 100)
        assert result["sampled_points"] == 100

    def test_too_many_raises(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "sampled.pcd")
        with pytest.raises(ValueError, match="Requested 200"):
            random_sample(sample_pcd_file, output, 200)
