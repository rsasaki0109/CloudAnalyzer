"""Tests for ca.stats module."""

import pytest

from ca.stats import compute_stats


class TestComputeStats:
    def test_basic_stats(self, sample_pcd_file):
        result = compute_stats(sample_pcd_file)
        assert result["num_points"] == 100
        assert result["volume"] > 0
        assert result["density"] > 0

    def test_spacing_keys(self, sample_pcd_file):
        result = compute_stats(sample_pcd_file)
        spacing = result["spacing"]
        assert set(spacing.keys()) == {"mean", "median", "min", "max", "std"}

    def test_spacing_positive(self, sample_pcd_file):
        result = compute_stats(sample_pcd_file)
        spacing = result["spacing"]
        assert spacing["mean"] > 0
        assert spacing["min"] >= 0

    def test_bbox(self, sample_pcd_file):
        result = compute_stats(sample_pcd_file)
        for lo, hi in zip(result["bbox_min"], result["bbox_max"]):
            assert lo <= hi

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            compute_stats("/no/file.pcd")
