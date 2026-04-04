"""Tests for experimental web sampling strategies."""

import numpy as np
import open3d as o3d

from ca.core.web_sampling import WebSampleRequest
from ca.experiments.web_sampling import get_web_sampling_strategies
from ca.experiments.web_sampling.common import clone_point_cloud
from ca.experiments.web_sampling.functional_voxel import (
    FunctionalVoxelSamplingStrategy,
    _grow_voxel_size,
    reduce_with_functional_voxels,
)
from ca.experiments.web_sampling.object_random import (
    RandomBudgetConfig,
    RandomBudgetSamplingStrategy,
)
from ca.experiments.web_sampling.pipeline_hybrid import (
    HybridPipelineSamplingStrategy,
    PipelineState,
)


def _make_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(float))
    return point_cloud


def _make_grid_cloud(width: int = 16, height: int = 12, depth: int = 4) -> o3d.geometry.PointCloud:
    x = np.linspace(0.0, 2.0, width)
    y = np.linspace(0.0, 1.5, height)
    z = np.linspace(0.0, 0.6, depth)
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return _make_point_cloud(points)


class TestCommonHelpers:
    def test_clone_point_cloud_returns_independent_geometry(self):
        original = _make_grid_cloud()
        cloned = clone_point_cloud(original)

        assert cloned is not original
        assert np.array_equal(np.asarray(cloned.points), np.asarray(original.points))


class TestStrategyRegistry:
    def test_returns_three_named_strategies(self):
        strategies = get_web_sampling_strategies()
        names = [strategy.name for strategy in strategies]
        designs = [strategy.design for strategy in strategies]

        assert names == ["functional_voxel", "random_budget", "hybrid_pipeline"]
        assert designs == ["functional", "oop", "pipeline"]

    def test_all_registered_strategies_respect_shared_contract(self):
        point_cloud = _make_grid_cloud()
        request = WebSampleRequest(point_cloud=point_cloud, max_points=60, label="grid")

        for strategy in get_web_sampling_strategies():
            result = strategy.reduce(request)
            assert result.strategy == strategy.name
            assert result.design == strategy.design
            assert result.reduced_points <= 60
            assert result.original_points == len(point_cloud.points)


class TestFunctionalVoxelSampling:
    def test_grow_voxel_size_for_no_progress(self):
        grown = _grow_voxel_size(
            current_voxel_size=0.1,
            reduced_points=100,
            previous_points=100,
            growth_factor=1.35,
        )
        assert grown == 0.1 * 1.35 * 1.25

    def test_grow_voxel_size_for_progress(self):
        grown = _grow_voxel_size(
            current_voxel_size=0.1,
            reduced_points=80,
            previous_points=100,
            growth_factor=1.35,
        )
        assert grown == 0.1 * 1.35

    def test_reduce_with_functional_voxels_records_iterations(self):
        point_cloud = _make_grid_cloud(width=18, height=18, depth=4)
        request = WebSampleRequest(point_cloud=point_cloud, max_points=90, label="grid")

        result = reduce_with_functional_voxels(
            request,
            initial_voxel_size=0.02,
            growth_factor=1.4,
        )

        assert result.reduced_points <= 90
        assert result.metadata["iterations"] > 0
        assert result.metadata["applied_voxel_size"] > 0.0
        assert result.metadata["initial_voxel_size"] == 0.02

    def test_strategy_adapter_uses_constructor_parameters(self):
        point_cloud = _make_grid_cloud(width=18, height=18, depth=4)
        request = WebSampleRequest(point_cloud=point_cloud, max_points=80, label="grid")

        result = FunctionalVoxelSamplingStrategy(
            initial_voxel_size=0.03,
            growth_factor=1.5,
        ).reduce(request)

        assert result.metadata["initial_voxel_size"] == 0.03
        assert result.reduced_points <= 80


class TestRandomBudgetSampling:
    def test_select_indices_returns_sorted_unique_indices(self):
        strategy = RandomBudgetSamplingStrategy(RandomBudgetConfig(seed=13))
        indices = strategy._select_indices(total_points=50, max_points=12)

        assert indices == sorted(indices)
        assert len(indices) == 12
        assert len(set(indices)) == 12

    def test_reduce_uses_config_seed(self):
        point_cloud = _make_grid_cloud(width=20, height=10, depth=1)
        request = WebSampleRequest(point_cloud=point_cloud, max_points=25, label="grid")
        config = RandomBudgetConfig(seed=123)

        result = RandomBudgetSamplingStrategy(config).reduce(request)

        assert result.metadata["seed"] == 123
        assert result.metadata["selected_points"] == 25
        assert result.reduced_points == 25


class TestHybridPipelineSampling:
    def test_estimate_voxel_stage_handles_degenerate_cloud(self):
        points = np.zeros((30, 3), dtype=float)
        state = PipelineState(
            current_cloud=_make_point_cloud(points),
            original_points=30,
            max_points=10,
            label="degenerate",
        )

        updated = HybridPipelineSamplingStrategy()._estimate_voxel_stage(state)

        assert updated.metadata["estimated_voxel_size"] > 0.0

    def test_voxel_stage_is_passthrough_under_budget(self):
        point_cloud = _make_grid_cloud(width=4, height=4, depth=1)
        state = PipelineState(
            current_cloud=point_cloud,
            original_points=len(point_cloud.points),
            max_points=32,
            label="small",
            metadata={"estimated_voxel_size": 0.1},
        )

        updated = HybridPipelineSamplingStrategy()._voxel_stage(state)

        assert updated.current_cloud is point_cloud
        assert updated.metadata["voxel_pass_points"] == len(point_cloud.points)

    def test_trim_stage_reduces_to_exact_budget(self):
        point_cloud = _make_grid_cloud(width=12, height=10, depth=1)
        total_points = len(point_cloud.points)
        state = PipelineState(
            current_cloud=point_cloud,
            original_points=total_points,
            max_points=25,
            label="trim",
        )

        updated = HybridPipelineSamplingStrategy()._trim_stage(state)

        assert len(updated.current_cloud.points) == 25
        assert updated.metadata["trimmed_points"] == total_points - 25

    def test_full_reduce_records_stage_metadata(self):
        point_cloud = _make_grid_cloud(width=20, height=20, depth=2)
        request = WebSampleRequest(point_cloud=point_cloud, max_points=100, label="grid")

        result = HybridPipelineSamplingStrategy().reduce(request)

        assert result.reduced_points <= 100
        assert result.metadata["stage_count"] == 3
        assert "estimated_voxel_size" in result.metadata
        assert "voxel_pass_points" in result.metadata
