"""Tests for ca.merge module."""

import open3d as o3d

from ca.merge import merge


class TestMerge:
    def test_merge_two_files(self, source_and_target_files, tmp_path):
        src, tgt = source_and_target_files
        output = str(tmp_path / "merged.pcd")
        result = merge([src, tgt], output)
        assert result["total_points"] == 200
        assert len(result["inputs"]) == 2

    def test_output_file_created(self, source_and_target_files, tmp_path):
        src, tgt = source_and_target_files
        output = str(tmp_path / "merged.pcd")
        merge([src, tgt], output)
        pcd = o3d.io.read_point_cloud(output)
        assert len(pcd.points) == 200

    def test_merge_single_file(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "merged.pcd")
        result = merge([sample_pcd_file], output)
        assert result["total_points"] == 100
