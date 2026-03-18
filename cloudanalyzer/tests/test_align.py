"""Tests for ca.align module."""

import pytest

from ca.align import align


class TestAlign:
    def test_align_two_files(self, source_and_target_files, tmp_path):
        src, tgt = source_and_target_files
        output = str(tmp_path / "aligned.pcd")
        result = align([src, tgt], output)
        assert result["total_points"] == 200
        assert result["num_inputs"] == 2
        assert len(result["steps"]) == 1
        assert result["steps"][0]["fitness"] >= 0

    def test_align_three_files(self, source_and_target_files, sample_pcd_file, tmp_path):
        src, tgt = source_and_target_files
        output = str(tmp_path / "aligned.pcd")
        result = align([src, tgt, sample_pcd_file], output)
        assert result["num_inputs"] == 3
        assert len(result["steps"]) == 2

    def test_too_few_files(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "aligned.pcd")
        with pytest.raises(ValueError, match="At least 2"):
            align([sample_pcd_file], output)

    def test_icp_method(self, source_and_target_files, tmp_path):
        src, tgt = source_and_target_files
        output = str(tmp_path / "aligned.pcd")
        result = align([src, tgt], output, method="icp")
        assert result["method"] == "icp"
