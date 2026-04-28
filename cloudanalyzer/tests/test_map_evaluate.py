"""Tests for experimental map_evaluate strategies."""

from __future__ import annotations

import numpy as np
import pytest

from ca.experiments.map_evaluate.common import MapEvaluateRequest
from ca.experiments.map_evaluate.nn_thresholds import NNThresholdMapEvaluateStrategy
from ca.experiments.map_evaluate.voxel_entropy import VoxelEntropyMapEvaluateStrategy


class TestNNThresholdMapEvaluate:
    def test_identical_maps_near_zero_chamfer(self):
        rng = np.random.default_rng(0)
        pts = rng.random((80, 3))
        req = MapEvaluateRequest(
            estimated_points=pts,
            reference_points=np.asarray(pts, dtype=np.float64).copy(),
            thresholds_m=(0.2, 0.1, 0.05),
        )
        out = NNThresholdMapEvaluateStrategy().evaluate(req)
        assert out.strategy == "nn_thresholds"
        assert out.metrics["chamfer_m"] == pytest.approx(0.0, abs=1e-9)
        assert out.metrics[f"accuracy@{0.2:.3f}m"] == pytest.approx(1.0)
        assert out.metrics[f"completeness@{0.2:.3f}m"] == pytest.approx(1.0)

    def test_shifted_map_has_positive_chamfer(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(size=(120, 3)) * 0.05
        est = ref + np.array([0.3, 0.0, 0.0])
        req = MapEvaluateRequest(
            estimated_points=est,
            reference_points=ref,
            thresholds_m=(0.5, 0.2),
        )
        out = NNThresholdMapEvaluateStrategy().evaluate(req)
        assert out.metrics["chamfer_m"] > 0.01

    def test_requires_reference(self):
        req = MapEvaluateRequest(
            estimated_points=np.zeros((10, 3)),
            reference_points=None,
        )
        with pytest.raises(ValueError, match="reference"):
            NNThresholdMapEvaluateStrategy().evaluate(req)


class TestVoxelEntropyMapEvaluate:
    def test_no_reference_returns_metrics(self):
        rng = np.random.default_rng(2)
        pts = rng.uniform(-2.0, 2.0, size=(300, 3))
        req = MapEvaluateRequest(
            estimated_points=pts,
            reference_points=None,
            structure_voxel_size=0.25,
            thresholds_m=(0.2,),
        )
        out = VoxelEntropyMapEvaluateStrategy().evaluate(req)
        assert out.strategy == "voxel_entropy"
        assert "mean_neighbor_entropy_bits" in out.metrics

    def test_rejects_reference_points(self):
        rng = np.random.default_rng(3)
        pts = rng.random((50, 3))
        req = MapEvaluateRequest(
            estimated_points=pts,
            reference_points=pts.copy(),
            structure_voxel_size=0.2,
        )
        with pytest.raises(ValueError, match="GT-free"):
            VoxelEntropyMapEvaluateStrategy().evaluate(req)
