"""Tests for stable web sampling core interfaces."""

import numpy as np
import open3d as o3d
import pytest

from ca.core import (
    RandomBudgetWebSamplingStrategy,
    WebSampleRequest,
    WebSampleResult,
    reduce_point_cloud_for_web,
)


def _make_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(float))
    return point_cloud


def _make_ordered_cloud(count: int) -> o3d.geometry.PointCloud:
    points = np.column_stack(
        [
            np.arange(count, dtype=float),
            np.arange(count, dtype=float) * 0.1,
            np.arange(count, dtype=float) * 0.01,
        ]
    )
    return _make_point_cloud(points)


class DummyStrategy:
    name = "dummy"
    design = "test"

    def __init__(self):
        self.requests = []

    def reduce(self, request: WebSampleRequest) -> WebSampleResult:
        self.requests.append(request)
        return WebSampleResult(
            point_cloud=request.point_cloud,
            strategy=self.name,
            design=self.design,
            original_points=len(request.point_cloud.points),
            reduced_points=len(request.point_cloud.points),
            metadata={"label": request.label},
        )


class TestWebSampleRequest:
    def test_rejects_non_positive_budget(self, simple_pcd):
        with pytest.raises(ValueError):
            WebSampleRequest(point_cloud=simple_pcd, max_points=0)

        with pytest.raises(ValueError):
            WebSampleRequest(point_cloud=simple_pcd, max_points=-3)

    def test_defaults_label(self, simple_pcd):
        request = WebSampleRequest(point_cloud=simple_pcd, max_points=10)
        assert request.label == "point cloud"


class TestWebSampleResult:
    def test_reduction_ratio_for_regular_case(self, simple_pcd):
        result = WebSampleResult(
            point_cloud=simple_pcd,
            strategy="dummy",
            design="test",
            original_points=100,
            reduced_points=25,
        )
        assert result.reduction_ratio == 0.75

    def test_reduction_ratio_for_zero_original_points(self):
        result = WebSampleResult(
            point_cloud=o3d.geometry.PointCloud(),
            strategy="dummy",
            design="test",
            original_points=0,
            reduced_points=0,
        )
        assert result.reduction_ratio == 0.0


class TestRandomBudgetWebSamplingStrategy:
    def test_passthrough_when_point_count_is_within_budget(self, simple_pcd):
        strategy = RandomBudgetWebSamplingStrategy(seed=11)
        request = WebSampleRequest(point_cloud=simple_pcd, max_points=200, label="sample")

        result = strategy.reduce(request)

        assert result.point_cloud is simple_pcd
        assert result.reduced_points == 100
        assert result.metadata == {"label": "sample", "seed": 11}

    def test_reduction_is_deterministic_for_same_seed(self):
        point_cloud = _make_ordered_cloud(120)
        request = WebSampleRequest(point_cloud=point_cloud, max_points=20, label="ordered")

        result_a = RandomBudgetWebSamplingStrategy(seed=5).reduce(request)
        result_b = RandomBudgetWebSamplingStrategy(seed=5).reduce(request)

        points_a = np.asarray(result_a.point_cloud.points)
        points_b = np.asarray(result_b.point_cloud.points)
        assert np.array_equal(points_a, points_b)
        assert result_a.metadata["selected_points"] == 20

    def test_reduction_changes_with_different_seed(self):
        point_cloud = _make_ordered_cloud(120)
        request = WebSampleRequest(point_cloud=point_cloud, max_points=20, label="ordered")

        result_a = RandomBudgetWebSamplingStrategy(seed=5).reduce(request)
        result_b = RandomBudgetWebSamplingStrategy(seed=6).reduce(request)

        points_a = np.asarray(result_a.point_cloud.points)
        points_b = np.asarray(result_b.point_cloud.points)
        assert not np.array_equal(points_a, points_b)


class TestReducePointCloudForWeb:
    def test_uses_injected_strategy(self, simple_pcd):
        strategy = DummyStrategy()

        result = reduce_point_cloud_for_web(
            simple_pcd,
            max_points=50,
            label="custom",
            strategy=strategy,
        )

        assert result.strategy == "dummy"
        assert len(strategy.requests) == 1
        assert strategy.requests[0].max_points == 50
        assert strategy.requests[0].label == "custom"

    def test_uses_default_strategy(self):
        point_cloud = _make_ordered_cloud(64)

        result = reduce_point_cloud_for_web(point_cloud, max_points=16, label="ordered")

        assert result.strategy == "random_budget"
        assert result.reduced_points == 16
        assert result.metadata["label"] == "ordered"

    def test_invalid_budget_bubbles_up_from_request(self, simple_pcd):
        with pytest.raises(ValueError):
            reduce_point_cloud_for_web(simple_pcd, max_points=0)
