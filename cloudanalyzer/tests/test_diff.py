"""Tests for ca.diff module."""

import pytest

from ca.diff import run_diff


class TestRunDiff:
    def test_basic_diff(self, source_and_target_files):
        src, tgt = source_and_target_files
        result = run_diff(src, tgt)
        assert result["source_points"] == 100
        assert result["target_points"] == 100
        assert "distance_stats" in result
        assert result["distance_stats"]["mean"] > 0

    def test_same_file(self, sample_pcd_file):
        result = run_diff(sample_pcd_file, sample_pcd_file)
        assert result["distance_stats"]["mean"] == pytest.approx(0.0, abs=1e-10)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            run_diff("/no/such/file.pcd", "/no/such/other.pcd")
