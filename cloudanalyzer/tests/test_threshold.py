"""Tests for threshold functionality across metrics, compare, diff."""

import numpy as np
import pytest

from ca.metrics import threshold_stats


class TestThresholdStats:
    def test_all_below(self):
        distances = np.array([0.1, 0.2, 0.3])
        result = threshold_stats(distances, 1.0)
        assert result["exceed_count"] == 0
        assert result["exceed_ratio"] == 0.0
        assert result["total"] == 3

    def test_all_above(self):
        distances = np.array([2.0, 3.0, 4.0])
        result = threshold_stats(distances, 1.0)
        assert result["exceed_count"] == 3
        assert result["exceed_ratio"] == pytest.approx(1.0)

    def test_partial(self):
        distances = np.array([0.5, 1.5, 2.5, 0.3])
        result = threshold_stats(distances, 1.0)
        assert result["exceed_count"] == 2
        assert result["exceed_ratio"] == pytest.approx(0.5)

    def test_exact_threshold_not_exceeded(self):
        distances = np.array([1.0])
        result = threshold_stats(distances, 1.0)
        assert result["exceed_count"] == 0


class TestCompareWithThreshold:
    def test_threshold_in_result(self, source_and_target_files):
        from ca.compare import run_compare
        src, tgt = source_and_target_files
        result = run_compare(src, tgt, method=None, threshold=0.05)
        assert "threshold" in result
        assert result["threshold"]["threshold"] == 0.05

    def test_no_threshold(self, source_and_target_files):
        from ca.compare import run_compare
        src, tgt = source_and_target_files
        result = run_compare(src, tgt, method=None)
        assert "threshold" not in result


class TestDiffWithThreshold:
    def test_threshold_in_result(self, source_and_target_files):
        from ca.diff import run_diff
        src, tgt = source_and_target_files
        result = run_diff(src, tgt, threshold=0.05)
        assert "threshold" in result
        assert result["threshold"]["exceed_count"] >= 0
