"""Stable minimal interface for web trajectory simplification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(slots=True)
class WebTrajectorySamplingRequest:
    """Shared input contract for browser-oriented trajectory reduction."""

    positions: np.ndarray
    max_points: int
    timestamps: np.ndarray | None = None
    label: str = "trajectory"
    preserve_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.max_points <= 0:
            raise ValueError(f"max_points must be positive, got {self.max_points}")
        if self.timestamps is not None:
            self.timestamps = np.asarray(self.timestamps, dtype=float)
            if self.timestamps.ndim != 1 or self.timestamps.shape[0] != self.positions.shape[0]:
                raise ValueError("timestamps must have shape (N,) matching positions")
            if self.timestamps.size > 1 and np.any(np.diff(self.timestamps) <= 0):
                raise ValueError("timestamps must be strictly increasing")
        self.preserve_indices = tuple(int(index) for index in self.preserve_indices)


@dataclass(slots=True)
class WebTrajectorySamplingResult:
    """Shared output contract for browser-oriented trajectory reduction."""

    positions: np.ndarray
    kept_indices: np.ndarray
    strategy: str
    design: str
    original_points: int
    reduced_points: int
    timestamps: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reduction_ratio(self) -> float:
        if self.original_points == 0:
            return 0.0
        return 1.0 - (self.reduced_points / self.original_points)


class WebTrajectorySamplingStrategy(Protocol):
    """Stable protocol retained after comparing concrete implementations."""

    name: str
    design: str

    def reduce(self, request: WebTrajectorySamplingRequest) -> WebTrajectorySamplingResult:
        """Reduce a trajectory for browser display."""


class TurnAwareWebTrajectorySamplingStrategy:
    """Stable reducer selected after experiment comparison."""

    name = "turn_aware"
    design = "oop"

    def __init__(self, turn_ratio: float = 0.45):
        self.turn_ratio = turn_ratio

    def _normalize_preserve(self, total_points: int, preserve_indices: tuple[int, ...]) -> tuple[int, ...]:
        if total_points <= 0:
            return ()
        normalized = {0, total_points - 1}
        normalized.update(index for index in preserve_indices if 0 <= index < total_points)
        return tuple(sorted(normalized))

    def _allocate_evenly(self, total_points: int, budget: int, preserve_indices: tuple[int, ...]) -> np.ndarray:
        if total_points <= 0:
            return np.zeros(0, dtype=int)
        preserve = list(self._normalize_preserve(total_points, preserve_indices))
        if total_points <= budget:
            return np.arange(total_points, dtype=int)
        if len(preserve) >= budget:
            return np.asarray(preserve[:budget], dtype=int)

        extras = budget - len(preserve)
        spans = []
        available_total = 0
        for left, right in zip(preserve[:-1], preserve[1:]):
            available = max(right - left - 1, 0)
            spans.append((left, right, available))
            available_total += available

        selected = set(preserve)
        assigned = []
        remainders = []
        for left, right, available in spans:
            if available_total == 0 or available == 0:
                quota = 0
                remainder = 0.0
            else:
                exact = extras * (available / available_total)
                quota = min(available, int(np.floor(exact)))
                remainder = exact - quota
            assigned.append(quota)
            remainders.append(remainder)

        remaining = extras - sum(assigned)
        for idx in np.argsort(remainders)[::-1]:
            if remaining <= 0:
                break
            left, right, available = spans[idx]
            if assigned[idx] < available:
                assigned[idx] += 1
                remaining -= 1

        for (left, right, _), quota in zip(spans, assigned):
            if quota <= 0:
                continue
            samples = np.linspace(left, right, num=quota + 2, dtype=int)[1:-1]
            selected.update(int(sample) for sample in samples if left < sample < right)

        if len(selected) < budget:
            for candidate in np.linspace(0, total_points - 1, num=budget, dtype=int):
                selected.add(int(candidate))
                if len(selected) >= budget:
                    break

        return np.asarray(sorted(selected), dtype=int)

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
        preserve = self._normalize_preserve(original_points, request.preserve_indices)
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
                    "effective_budget": effective_budget,
                    "preserved_points": len(preserve),
                    "turn_points": 0,
                },
            )

        scores = self._turn_scores(request.positions)
        remaining_budget = max(effective_budget - len(preserve), 0)
        turn_budget = min(
            max(int(np.ceil(remaining_budget * self.turn_ratio)), 0),
            max(original_points - len(preserve), 0),
        )
        candidate_indices = np.argsort(scores)[::-1]
        turn_points: list[int] = []
        preserved_set = set(preserve)
        for index in candidate_indices:
            candidate = int(index)
            if candidate in preserved_set or scores[candidate] <= 0:
                continue
            turn_points.append(candidate)
            if len(turn_points) >= turn_budget:
                break

        keep_indices = self._allocate_evenly(
            total_points=original_points,
            budget=effective_budget,
            preserve_indices=tuple(sorted(preserved_set | set(turn_points))),
        )
        positions = request.positions[keep_indices]
        timestamps = request.timestamps[keep_indices] if request.timestamps is not None else None
        return WebTrajectorySamplingResult(
            positions=positions,
            timestamps=timestamps,
            kept_indices=keep_indices,
            strategy=self.name,
            design=self.design,
            original_points=original_points,
            reduced_points=int(keep_indices.size),
            metadata={
                "label": request.label,
                "effective_budget": effective_budget,
                "preserved_points": len(preserve),
                "turn_points": len(turn_points),
            },
        )


def reduce_trajectory_for_web(
    positions: np.ndarray,
    max_points: int,
    timestamps: np.ndarray | None = None,
    label: str = "trajectory",
    preserve_indices: tuple[int, ...] = (),
    strategy: WebTrajectorySamplingStrategy | None = None,
) -> WebTrajectorySamplingResult:
    """Reduce a trajectory for browser display using the stable strategy."""

    reducer = strategy or TurnAwareWebTrajectorySamplingStrategy()
    request = WebTrajectorySamplingRequest(
        positions=positions,
        timestamps=timestamps,
        max_points=max_points,
        label=label,
        preserve_indices=preserve_indices,
    )
    return reducer.reduce(request)
