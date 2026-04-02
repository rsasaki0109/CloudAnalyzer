"""Tests for experimental trajectory simplification strategies."""

import numpy as np

from ca.core.web_trajectory_sampling import WebTrajectorySamplingRequest
from ca.experiments.web_trajectory_sampling import get_strategies
from ca.experiments.web_trajectory_sampling.common import (
    allocate_evenly_spaced_indices,
    normalize_preserve_indices,
    path_length,
    reconstruct_positions,
    shrink_sorted_indices,
)
from ca.experiments.web_trajectory_sampling.distance_accumulator import DistanceAccumulatorStrategy
from ca.experiments.web_trajectory_sampling.turn_aware import TurnAwareStrategy
from ca.experiments.web_trajectory_sampling.uniform_stride import (
    UniformStrideStrategy,
    reduce_with_uniform_stride,
)


def _make_switchback(points_per_segment: int = 40) -> tuple[np.ndarray, np.ndarray]:
    positions = []
    current = np.array([0.0, 0.0, 0.0])
    for step in range(6):
        next_point = np.array([current[0] + 2.5, 1.2 if step % 2 == 0 else -1.2, 0.0])
        t = np.linspace(0.0, 1.0, points_per_segment, endpoint=True)
        segment = current[None, :] + (next_point - current)[None, :] * t[:, None]
        if positions:
            segment = segment[1:]
        positions.append(segment)
        current = next_point
    merged = np.vstack(positions)
    timestamps = np.linspace(0.0, 10.0, merged.shape[0])
    return timestamps, merged


class TestCommonHelpers:
    def test_normalize_preserve_indices_adds_endpoints(self):
        normalized = normalize_preserve_indices(10, (3, 3, -1, 8, 11))
        assert normalized == (0, 3, 8, 9)

    def test_allocate_evenly_spaced_indices_respects_budget_and_preserve(self):
        indices = allocate_evenly_spaced_indices(50, budget=6, preserve_indices=(10, 25))
        assert indices.size <= 6
        assert 0 in indices.tolist()
        assert 10 in indices.tolist()
        assert 25 in indices.tolist()
        assert 49 in indices.tolist()

    def test_shrink_sorted_indices_keeps_preserve_indices(self):
        indices = np.array([0, 2, 4, 6, 8, 10, 12])
        shrunk = shrink_sorted_indices(indices, budget=4, preserve_indices=(4, 10))
        assert shrunk.size == 4
        assert 4 in shrunk.tolist()
        assert 10 in shrunk.tolist()

    def test_path_length_and_reconstruction(self):
        timestamps = np.array([0.0, 1.0, 2.0, 3.0])
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        sampled_positions = positions[[0, 3]]
        sampled_timestamps = timestamps[[0, 3]]
        reconstructed = reconstruct_positions(timestamps, sampled_timestamps, sampled_positions)

        assert path_length(positions) == 3.0
        assert np.allclose(reconstructed, positions)


class TestStrategies:
    def test_registry_returns_three_strategies(self):
        names = [strategy.name for strategy in get_strategies()]
        assert names == ["uniform_stride", "distance_accumulator", "turn_aware"]

    def test_uniform_stride_reducer_respects_budget_and_preserve(self):
        timestamps, positions = _make_switchback()
        result = reduce_with_uniform_stride(
            WebTrajectorySamplingRequest(
                positions=positions,
                timestamps=timestamps,
                max_points=12,
                preserve_indices=(20, 100),
            )
        )
        assert result.reduced_points <= 12
        assert 20 in result.kept_indices.tolist()
        assert 100 in result.kept_indices.tolist()

    def test_distance_accumulator_records_threshold_and_trim(self):
        timestamps, positions = _make_switchback(points_per_segment=60)
        result = DistanceAccumulatorStrategy().reduce(
            WebTrajectorySamplingRequest(
                positions=positions,
                timestamps=timestamps,
                max_points=14,
                preserve_indices=(50, 180),
            )
        )
        assert result.reduced_points <= 14
        assert "distance_threshold" in result.metadata
        assert "trimmed_points" in result.metadata
        assert 50 in result.kept_indices.tolist()
        assert 180 in result.kept_indices.tolist()

    def test_turn_aware_prioritizes_turn_points(self):
        timestamps, positions = _make_switchback(points_per_segment=50)
        request = WebTrajectorySamplingRequest(
            positions=positions,
            timestamps=timestamps,
            max_points=16,
            preserve_indices=(40, 160),
        )
        result = TurnAwareStrategy(turn_ratio=0.5).reduce(request)

        assert result.reduced_points <= 16
        assert result.metadata["turn_points"] > 0
        assert 40 in result.kept_indices.tolist()
        assert 160 in result.kept_indices.tolist()

    def test_all_strategies_keep_endpoints(self):
        timestamps, positions = _make_switchback()
        request = WebTrajectorySamplingRequest(
            positions=positions,
            timestamps=timestamps,
            max_points=10,
        )
        for strategy in get_strategies():
            result = strategy.reduce(request)
            assert result.kept_indices[0] == 0
            assert result.kept_indices[-1] == positions.shape[0] - 1
