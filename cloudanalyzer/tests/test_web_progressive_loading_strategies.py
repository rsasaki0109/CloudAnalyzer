"""Tests for experimental progressive-loading strategies."""

import numpy as np

from ca.core.web_progressive_loading import WebProgressiveLoadingRequest
from ca.experiments.web_progressive_loading import get_strategies
from ca.experiments.web_progressive_loading.common import (
    build_result_from_order,
    chunk_size_std,
    coverage_p95,
    normalize_positions,
    progressive_prefix_points,
    split_indices_into_chunks,
)
from ca.experiments.web_progressive_loading.distance_shells import DistanceShellsStrategy
from ca.experiments.web_progressive_loading.grid_tiles import GridTilesStrategy
from ca.experiments.web_progressive_loading.spatial_shuffle import SpatialShuffleStrategy


def _make_positions(count: int = 240) -> np.ndarray:
    return np.column_stack(
        [
            np.linspace(0.0, 20.0, count),
            np.sin(np.linspace(0.0, 6.0 * np.pi, count)),
            np.cos(np.linspace(0.0, 2.0 * np.pi, count)) * 0.5,
        ]
    )


class TestCommonHelpers:
    def test_normalize_positions_scales_to_unit_cube(self):
        normalized = normalize_positions(_make_positions(20))
        assert normalized.shape == (20, 3)
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)

    def test_split_indices_and_build_result(self):
        positions = _make_positions(30)
        request = WebProgressiveLoadingRequest(
            positions=positions,
            initial_points=8,
            chunk_points=7,
        )
        chunks = split_indices_into_chunks(np.arange(8, 30), chunk_points=7)
        result = build_result_from_order(
            request=request,
            ordered_indices=np.arange(30),
            strategy="demo",
            design="test",
            metadata={},
        )
        assert len(chunks) == 4
        assert result.initial_points == 8
        assert sum(chunk.point_count for chunk in result.chunks) == 22

    def test_progressive_prefix_and_coverage(self):
        positions = _make_positions(60)
        request = WebProgressiveLoadingRequest(
            positions=positions,
            initial_points=10,
            chunk_points=15,
        )
        result = build_result_from_order(
            request=request,
            ordered_indices=np.arange(60),
            strategy="demo",
            design="test",
            metadata={},
        )
        prefixes = progressive_prefix_points(result)
        assert len(prefixes) == 1 + len(result.chunks)
        assert coverage_p95(positions, prefixes[0]) >= 0.0
        assert chunk_size_std(result) >= 0.0


class TestStrategies:
    def test_registry_returns_three_strategies(self):
        assert [strategy.name for strategy in get_strategies()] == [
            "grid_tiles",
            "spatial_shuffle",
            "distance_shells",
        ]

    def test_each_strategy_preserves_all_points_without_duplicates(self):
        positions = _make_positions(150)
        request = WebProgressiveLoadingRequest(
            positions=positions,
            initial_points=24,
            chunk_points=30,
        )
        for strategy in get_strategies():
            result = strategy.plan(request)
            rebuilt = np.vstack([result.initial_positions, *[chunk.positions for chunk in result.chunks]])
            assert rebuilt.shape == positions.shape

    def test_grid_tiles_reports_tile_metadata(self):
        result = GridTilesStrategy().plan(
            WebProgressiveLoadingRequest(
                positions=_make_positions(180),
                initial_points=20,
                chunk_points=32,
            )
        )
        assert result.metadata["grid_side"] >= 1
        assert result.metadata["tile_count"] >= 1

    def test_spatial_shuffle_chunks_respect_budget(self):
        result = SpatialShuffleStrategy().plan(
            WebProgressiveLoadingRequest(
                positions=_make_positions(170),
                initial_points=18,
                chunk_points=25,
            )
        )
        assert all(chunk.point_count <= 25 for chunk in result.chunks)

    def test_distance_shells_reports_shell_metadata(self):
        result = DistanceShellsStrategy(shell_count=6).plan(
            WebProgressiveLoadingRequest(
                positions=_make_positions(200),
                initial_points=22,
                chunk_points=28,
            )
        )
        assert result.metadata["shell_count"] >= 1
