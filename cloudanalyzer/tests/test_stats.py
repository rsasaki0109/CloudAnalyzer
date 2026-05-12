"""Tests for ca.stats module."""

import pytest

from ca.stats import compute_stats


class TestComputeStats:
    def test_basic_stats(self, sample_pcd_file):
        result = compute_stats(sample_pcd_file)
        assert result["num_points"] == 100
        assert result["volume"] > 0
        assert result["density"] > 0
        assert result["robust_volume"] > 0
        assert result["robust_density"] > 0

    def test_spacing_keys(self, sample_pcd_file):
        result = compute_stats(sample_pcd_file)
        spacing = result["spacing"]
        assert set(spacing.keys()) == {"mean", "median", "min", "max", "std"}

    def test_spacing_positive(self, sample_pcd_file):
        result = compute_stats(sample_pcd_file)
        spacing = result["spacing"]
        assert spacing["mean"] > 0
        assert spacing["min"] >= 0
        assert result["spacing_sample_points"] == result["num_points"]

    def test_bbox(self, sample_pcd_file):
        result = compute_stats(sample_pcd_file)
        for lo, hi in zip(result["bbox_min"], result["bbox_max"]):
            assert lo <= hi
        for lo, hi in zip(result["robust_bbox_min"], result["robust_bbox_max"]):
            assert lo <= hi

    def test_robust_outlier_keys(self, sample_pcd_file):
        result = compute_stats(sample_pcd_file)
        assert result["outside_robust_bbox_count"] >= 0
        assert 0.0 <= result["outside_robust_bbox_ratio"] <= 1.0
        assert set(result["axis_percentiles"].keys()) == {"p01", "p50", "p99"}

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            compute_stats("/no/file.pcd")
