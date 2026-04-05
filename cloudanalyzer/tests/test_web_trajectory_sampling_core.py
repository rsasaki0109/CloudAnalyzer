"""Tests for stable trajectory sampling core interfaces."""

import numpy as np
import pytest

from ca.core import (
    TurnAwareWebTrajectorySamplingStrategy,
    WebTrajectorySamplingRequest,
    WebTrajectorySamplingResult,
    reduce_trajectory_for_web,
)


class DummyTrajectoryStrategy:
    name = "dummy"
    design = "test"

    def __init__(self):
        self.requests = []

    def reduce(self, request: WebTrajectorySamplingRequest) -> WebTrajectorySamplingResult:
        self.requests.append(request)
        keep_indices = np.arange(request.positions.shape[0], dtype=int)
        return WebTrajectorySamplingResult(
            positions=request.positions,
            timestamps=request.timestamps,
            kept_indices=keep_indices,
            strategy=self.name,
            design=self.design,
            original_points=request.positions.shape[0],
            reduced_points=request.positions.shape[0],
            metadata={"label": request.label},
        )


class TestWebTrajectorySamplingRequest:
    def test_rejects_invalid_position_shape(self):
        with pytest.raises(ValueError):
            WebTrajectorySamplingRequest(positions=np.array([1.0, 2.0, 3.0]), max_points=10)

    def test_rejects_invalid_budget(self):
        with pytest.raises(ValueError):
            WebTrajectorySamplingRequest(positions=np.zeros((3, 3)), max_points=0)

    def test_rejects_timestamp_mismatch_and_non_monotonic_input(self):
        with pytest.raises(ValueError):
            WebTrajectorySamplingRequest(
                positions=np.zeros((3, 3)),
                timestamps=np.array([0.0, 1.0]),
                max_points=2,
            )
        with pytest.raises(ValueError):
            WebTrajectorySamplingRequest(
                positions=np.zeros((3, 3)),
                timestamps=np.array([0.0, 0.5, 0.4]),
                max_points=2,
            )


class TestWebTrajectorySamplingResult:
    def test_reduction_ratio(self):
        result = WebTrajectorySamplingResult(
            positions=np.zeros((4, 3)),
            timestamps=np.arange(4, dtype=float),
            kept_indices=np.arange(4, dtype=int),
            strategy="dummy",
            design="test",
            original_points=10,
            reduced_points=4,
        )
        assert result.reduction_ratio == 0.6


class TestReduceTrajectoryForWeb:
    def test_uses_injected_strategy(self):
        strategy = DummyTrajectoryStrategy()
        positions = np.arange(30, dtype=float).reshape(10, 3)
        timestamps = np.arange(10, dtype=float)

        result = reduce_trajectory_for_web(
            positions=positions,
            timestamps=timestamps,
            max_points=4,
            label="custom",
            strategy=strategy,
        )

        assert result.strategy == "dummy"
        assert len(strategy.requests) == 1
        assert strategy.requests[0].label == "custom"
        assert strategy.requests[0].max_points == 4

    def test_default_strategy_preserves_requested_indices(self):
        positions = np.column_stack([np.arange(20, dtype=float), np.zeros(20), np.zeros(20)])
        timestamps = np.arange(20, dtype=float)

        result = reduce_trajectory_for_web(
            positions=positions,
            timestamps=timestamps,
            max_points=5,
            preserve_indices=(7, 11),
        )

        assert result.strategy == "turn_aware"
        assert 7 in result.kept_indices.tolist()
        assert 11 in result.kept_indices.tolist()
        assert result.reduced_points <= 5

    def test_stable_strategy_passthrough_within_budget(self):
        positions = np.column_stack([np.arange(5, dtype=float), np.zeros(5), np.zeros(5)])
        timestamps = np.arange(5, dtype=float)

        result = TurnAwareWebTrajectorySamplingStrategy().reduce(
            WebTrajectorySamplingRequest(
                positions=positions,
                timestamps=timestamps,
                max_points=10,
            )
        )

        assert np.array_equal(result.positions, positions)
        assert np.array_equal(result.timestamps, timestamps)
        assert np.array_equal(result.kept_indices, np.arange(5, dtype=int))
