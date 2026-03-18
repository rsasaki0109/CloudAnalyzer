"""Tests for ca.metrics module."""

import numpy as np
import pytest

from ca.metrics import compute_nn_distance, summarize


class TestComputeNNDistance:
    def test_identical_clouds_zero_distance(self, simple_pcd, identical_pcd):
        distances = compute_nn_distance(simple_pcd, identical_pcd)
        assert len(distances) == 100
        np.testing.assert_allclose(distances, 0.0, atol=1e-10)

    def test_shifted_clouds_positive_distance(self, simple_pcd, shifted_pcd):
        distances = compute_nn_distance(simple_pcd, shifted_pcd)
        assert len(distances) == 100
        assert np.all(distances >= 0.0)
        assert np.mean(distances) > 0.0

    def test_returns_ndarray(self, simple_pcd, shifted_pcd):
        distances = compute_nn_distance(simple_pcd, shifted_pcd)
        assert isinstance(distances, np.ndarray)


class TestSummarize:
    def test_keys(self):
        distances = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = summarize(distances)
        assert set(stats.keys()) == {"mean", "median", "max", "min", "std"}

    def test_values(self):
        distances = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = summarize(distances)
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["median"] == pytest.approx(3.0)
        assert stats["max"] == pytest.approx(5.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["std"] == pytest.approx(np.std(distances))

    def test_single_value(self):
        distances = np.array([42.0])
        stats = summarize(distances)
        assert stats["mean"] == pytest.approx(42.0)
        assert stats["std"] == pytest.approx(0.0)

    def test_all_float(self):
        stats = summarize(np.array([1.0, 2.0]))
        for v in stats.values():
            assert isinstance(v, float)
