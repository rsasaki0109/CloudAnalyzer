"""Pipeline-oriented trajectory simplification using travelled-distance accumulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ca.core.web_trajectory_sampling import WebTrajectorySamplingRequest, WebTrajectorySamplingResult
from ca.experiments.web_trajectory_sampling.common import (
    normalize_preserve_indices,
    path_length,
    shrink_sorted_indices,
)


@dataclass(slots=True)
class DistanceAccumulatorState:
    """Mutable state passed between reduction stages."""

    request: WebTrajectorySamplingRequest
    effective_budget: int
    preserve_indices: tuple[int, ...]
    selected_indices: list[int] = field(default_factory=list)
    metadata: dict[str, float | int | str] = field(default_factory=dict)


class DistanceAccumulatorStrategy:
    """Simplify trajectories by sampling after traveled-distance thresholds."""

    name = "distance_accumulator"
    design = "pipeline"

    def __init__(self):
        self.stages = (
            self._seed_stage,
            self._accumulate_stage,
            self._trim_stage,
        )

    def _seed_stage(self, state: DistanceAccumulatorState) -> DistanceAccumulatorState:
        state.selected_indices = [0]
        state.metadata["stage_count"] = len(self.stages)
        return state

    def _accumulate_stage(self, state: DistanceAccumulatorState) -> DistanceAccumulatorState:
        positions = state.request.positions
        total_points = positions.shape[0]
        if total_points <= state.effective_budget:
            state.selected_indices = list(range(total_points))
            state.metadata["distance_threshold"] = 0.0
            return state

        total_length = path_length(positions)
        threshold = total_length / max(state.effective_budget - 1, 1)
        state.metadata["distance_threshold"] = float(threshold)
        selected = set(state.selected_indices)
        preserve = set(state.preserve_indices)
        accumulated = 0.0
        last_kept = 0

        for index in range(1, total_points - 1):
            step = float(np.linalg.norm(positions[index] - positions[index - 1]))
            accumulated += step
            if index in preserve:
                selected.add(index)
                last_kept = index
                accumulated = 0.0
                continue
            if accumulated >= threshold:
                selected.add(index)
                last_kept = index
                accumulated = 0.0

        if last_kept != total_points - 1:
            selected.add(total_points - 1)

        state.selected_indices = sorted(selected)
        return state

    def _trim_stage(self, state: DistanceAccumulatorState) -> DistanceAccumulatorState:
        indices = np.asarray(state.selected_indices, dtype=int)
        shrunk = shrink_sorted_indices(
            indices=indices,
            budget=state.effective_budget,
            preserve_indices=state.preserve_indices,
        )
        state.selected_indices = shrunk.tolist()
        state.metadata["trimmed_points"] = int(indices.size - shrunk.size)
        return state

    def reduce(self, request: WebTrajectorySamplingRequest) -> WebTrajectorySamplingResult:
        original_points = request.positions.shape[0]
        preserve = normalize_preserve_indices(original_points, request.preserve_indices)
        state = DistanceAccumulatorState(
            request=request,
            effective_budget=max(request.max_points, len(preserve)),
            preserve_indices=preserve,
            metadata={"label": request.label, "preserved_points": len(preserve)},
        )
        for stage in self.stages:
            state = stage(state)

        keep_indices = np.asarray(state.selected_indices, dtype=int)
        timestamps = request.timestamps[keep_indices] if request.timestamps is not None else None
        return WebTrajectorySamplingResult(
            positions=request.positions[keep_indices],
            timestamps=timestamps,
            kept_indices=keep_indices,
            strategy=self.name,
            design=self.design,
            original_points=original_points,
            reduced_points=int(keep_indices.size),
            metadata=state.metadata,
        )
