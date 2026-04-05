"""Class-oriented trajectory simplification that protects high-curvature points."""

from __future__ import annotations

import numpy as np

from ca.core.web_trajectory_sampling import WebTrajectorySamplingRequest, WebTrajectorySamplingResult
from ca.experiments.web_trajectory_sampling.common import (
    allocate_evenly_spaced_indices,
    normalize_preserve_indices,
)


class TurnAwareStrategy:
    """Simplify trajectories by prioritizing turning points before filling gaps evenly."""

    name = "turn_aware"
    design = "oop"

    def __init__(self, turn_ratio: float = 0.45):
        self.turn_ratio = turn_ratio

    def _turn_scores(self, positions: np.ndarray) -> np.ndarray:
        total_points = positions.shape[0]
        scores = np.zeros(total_points, dtype=float)
        if total_points < 3:
            return scores

        previous_vectors = positions[1:-1] - positions[:-2]
        next_vectors = positions[2:] - positions[1:-1]
        previous_norm = np.linalg.norm(previous_vectors, axis=1)
        next_norm = np.linalg.norm(next_vectors, axis=1)
        valid = (previous_norm > 1e-9) & (next_norm > 1e-9)
        if not np.any(valid):
            return scores

        cosines = np.ones(total_points - 2, dtype=float)
        cosines[valid] = np.clip(
            np.sum(previous_vectors[valid] * next_vectors[valid], axis=1)
            / (previous_norm[valid] * next_norm[valid]),
            -1.0,
            1.0,
        )
        angles = np.arccos(cosines)
        local_scale = np.minimum(previous_norm, next_norm)
        scores[1:-1] = angles * local_scale
        return scores

    def reduce(self, request: WebTrajectorySamplingRequest) -> WebTrajectorySamplingResult:
        original_points = request.positions.shape[0]
        preserve = normalize_preserve_indices(original_points, request.preserve_indices)
        effective_budget = max(request.max_points, len(preserve))
        if original_points <= effective_budget:
            return WebTrajectorySamplingResult(
                positions=request.positions,
                timestamps=request.timestamps,
                kept_indices=np.arange(original_points, dtype=int),
                strategy=self.name,
                design=self.design,
                original_points=original_points,
                reduced_points=original_points,
                metadata={
                    "label": request.label,
                    "preserved_points": len(preserve),
                    "turn_points": 0,
                    "effective_budget": effective_budget,
                },
            )

        scores = self._turn_scores(request.positions)
        remaining_budget = max(effective_budget - len(preserve), 0)
        turn_budget = min(
            max(int(np.ceil(remaining_budget * self.turn_ratio)), 0),
            max(original_points - len(preserve), 0),
        )
        preserved_set = set(preserve)
        turn_points: list[int] = []
        for index in np.argsort(scores)[::-1]:
            candidate = int(index)
            if candidate in preserved_set or scores[candidate] <= 0:
                continue
            turn_points.append(candidate)
            if len(turn_points) >= turn_budget:
                break

        keep_indices = allocate_evenly_spaced_indices(
            total_points=original_points,
            budget=effective_budget,
            preserve_indices=tuple(sorted(preserved_set | set(turn_points))),
        )
        timestamps = request.timestamps[keep_indices] if request.timestamps is not None else None
        return WebTrajectorySamplingResult(
            positions=request.positions[keep_indices],
            timestamps=timestamps,
            kept_indices=keep_indices,
            strategy=self.name,
            design=self.design,
            original_points=original_points,
            reduced_points=int(keep_indices.size),
            metadata={
                "label": request.label,
                "preserved_points": len(preserve),
                "turn_points": len(turn_points),
                "effective_budget": effective_budget,
            },
        )
