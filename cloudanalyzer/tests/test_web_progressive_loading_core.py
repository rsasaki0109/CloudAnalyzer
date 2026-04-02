"""Tests for stable progressive-loading core interfaces."""

import numpy as np
import pytest

from ca.core.web_progressive_loading import (
    DistanceShellsWebProgressiveLoadingStrategy,
    WebProgressiveLoadingChunk,
    WebProgressiveLoadingRequest,
    plan_progressive_loading_for_web,
)


def _make_positions(count: int = 120) -> np.ndarray:
    return np.column_stack(
        [
            np.linspace(0.0, 12.0, count),
            np.sin(np.linspace(0.0, 3.0 * np.pi, count)),
            np.zeros(count),
        ]
    )


class TestWebProgressiveLoadingRequest:
    def test_rejects_invalid_shapes(self):
        with pytest.raises(ValueError):
            WebProgressiveLoadingRequest(
                positions=np.array([1.0, 2.0, 3.0]),
                initial_points=10,
                chunk_points=5,
            )

    def test_rejects_misaligned_distances(self):
        with pytest.raises(ValueError):
            WebProgressiveLoadingRequest(
                positions=_make_positions(10),
                initial_points=4,
                chunk_points=3,
                distances=np.ones(9),
            )


class TestWebProgressiveLoadingChunk:
    def test_point_count_property(self):
        chunk = WebProgressiveLoadingChunk(positions=_make_positions(7))
        assert chunk.point_count == 7


class TestStablePlanner:
    def test_default_planner_uses_distance_shells(self):
        positions = _make_positions(180)

        result = plan_progressive_loading_for_web(
            positions=positions,
            initial_points=24,
            chunk_points=40,
        )

        assert result.strategy == "distance_shells"
        assert result.design == "radial"
        assert result.initial_points == 24
        assert result.total_displayed_points == 180
        assert all(chunk.point_count <= 40 for chunk in result.chunks)

    def test_direct_strategy_plan_covers_all_points(self):
        positions = _make_positions(90)
        result = DistanceShellsWebProgressiveLoadingStrategy().plan(
            WebProgressiveLoadingRequest(
                positions=positions,
                initial_points=15,
                chunk_points=20,
            )
        )

        rebuilt = np.vstack([result.initial_positions, *[chunk.positions for chunk in result.chunks]])
        assert rebuilt.shape == positions.shape
