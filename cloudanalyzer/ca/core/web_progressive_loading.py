"""Stable minimal interface for browser progressive point loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(slots=True)
class WebProgressiveLoadingRequest:
    """Shared input contract for progressive browser point loading."""

    positions: np.ndarray
    initial_points: int
    chunk_points: int
    distances: np.ndarray | None = None
    label: str = "point cloud"

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        if self.initial_points <= 0:
            raise ValueError("initial_points must be positive")
        if self.chunk_points <= 0:
            raise ValueError("chunk_points must be positive")
        if self.distances is not None:
            self.distances = np.asarray(self.distances, dtype=float)
            if self.distances.ndim != 1 or self.distances.shape[0] != self.positions.shape[0]:
                raise ValueError("distances must have shape (N,) matching positions")


@dataclass(slots=True)
class WebProgressiveLoadingChunk:
    """One deferred chunk appended after the initial point payload."""

    positions: np.ndarray
    distances: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def point_count(self) -> int:
        return int(self.positions.shape[0])


@dataclass(slots=True)
class WebProgressiveLoadingResult:
    """Shared output contract for progressive browser point loading."""

    initial_positions: np.ndarray
    initial_distances: np.ndarray | None
    chunks: tuple[WebProgressiveLoadingChunk, ...]
    strategy: str
    design: str
    original_points: int
    initial_points: int
    chunk_points: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def streamed_points(self) -> int:
        return int(sum(chunk.point_count for chunk in self.chunks))

    @property
    def total_displayed_points(self) -> int:
        return self.initial_points + self.streamed_points


class WebProgressiveLoadingStrategy(Protocol):
    """Stable protocol retained after comparing concrete implementations."""

    name: str
    design: str

    def plan(self, request: WebProgressiveLoadingRequest) -> WebProgressiveLoadingResult:
        """Plan an initial payload and deferred chunks."""


def _normalize_positions(positions: np.ndarray) -> np.ndarray:
    """Normalize positions into a unit cube for stable grid math."""

    if positions.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    mins = np.min(positions, axis=0)
    spans = np.max(positions, axis=0) - mins
    spans[spans < 1e-9] = 1.0
    return np.asarray((positions - mins) / spans, dtype=float)


def _grid_side(point_count: int, initial_points: int) -> int:
    target_cells = max(min(initial_points, point_count), 1)
    return max(1, int(np.ceil(np.cbrt(target_cells))))


def _build_result_from_groups(
    request: WebProgressiveLoadingRequest,
    groups: list[np.ndarray],
    strategy: str,
    design: str,
    metadata: dict[str, Any],
) -> WebProgressiveLoadingResult:
    total_points = request.positions.shape[0]
    if total_points == 0:
        return WebProgressiveLoadingResult(
            initial_positions=np.zeros((0, 3), dtype=float),
            initial_distances=np.zeros(0, dtype=float) if request.distances is not None else None,
            chunks=(),
            strategy=strategy,
            design=design,
            original_points=0,
            initial_points=0,
            chunk_points=request.chunk_points,
            metadata=metadata,
        )

    initial_budget = min(request.initial_points, total_points)
    initial_indices: list[int] = []
    level = 0
    while len(initial_indices) < initial_budget:
        added = False
        for group in groups:
            if level < group.size:
                initial_indices.append(int(group[level]))
                added = True
                if len(initial_indices) >= initial_budget:
                    break
        if not added:
            break
        level += 1

    initial_index_set = set(initial_indices)
    chunk_payloads: list[np.ndarray] = []
    current: list[int] = []
    for group in groups:
        for index in group.tolist():
            point_index = int(index)
            if point_index in initial_index_set:
                continue
            current.append(point_index)
            if len(current) >= request.chunk_points:
                chunk_payloads.append(np.asarray(current, dtype=int))
                current = []
    if current:
        chunk_payloads.append(np.asarray(current, dtype=int))

    initial_index_array = np.asarray(initial_indices, dtype=int)
    chunks = tuple(
        WebProgressiveLoadingChunk(
            positions=request.positions[indices],
            distances=request.distances[indices] if request.distances is not None else None,
            metadata={"chunk_index": chunk_index},
        )
        for chunk_index, indices in enumerate(chunk_payloads)
    )
    return WebProgressiveLoadingResult(
        initial_positions=request.positions[initial_index_array],
        initial_distances=(
            request.distances[initial_index_array] if request.distances is not None else None
        ),
        chunks=chunks,
        strategy=strategy,
        design=design,
        original_points=total_points,
        initial_points=int(initial_index_array.size),
        chunk_points=request.chunk_points,
        metadata=metadata,
    )


class DistanceShellsWebProgressiveLoadingStrategy:
    """Stable reducer selected after experiment comparison."""

    name = "distance_shells"
    design = "radial"

    def __init__(self, shell_count: int = 8):
        self.shell_count = shell_count

    def plan(self, request: WebProgressiveLoadingRequest) -> WebProgressiveLoadingResult:
        positions = request.positions
        if positions.shape[0] == 0:
            return _build_result_from_groups(
                request=request,
                groups=[],
                strategy=self.name,
                design=self.design,
                metadata={"shell_count": 0},
            )

        center = np.mean(positions, axis=0)
        radii = np.linalg.norm(positions - center[None, :], axis=1)
        shell_edges = np.quantile(radii, np.linspace(0.0, 1.0, self.shell_count + 1))
        shell_ids = np.searchsorted(shell_edges[1:-1], radii, side="right")

        groups: list[np.ndarray] = []
        for shell_id in range(self.shell_count):
            group = np.flatnonzero(shell_ids == shell_id)
            if group.size == 0:
                continue
            group_order = np.argsort(radii[group], kind="stable")
            groups.append(group[group_order])

        return _build_result_from_groups(
            request=request,
            groups=groups,
            strategy=self.name,
            design=self.design,
            metadata={"shell_count": len(groups)},
        )


def plan_progressive_loading_for_web(
    positions: np.ndarray,
    initial_points: int,
    chunk_points: int,
    distances: np.ndarray | None = None,
    label: str = "point cloud",
    strategy: WebProgressiveLoadingStrategy | None = None,
) -> WebProgressiveLoadingResult:
    """Plan progressive browser loading with the stable strategy."""

    planner = strategy or DistanceShellsWebProgressiveLoadingStrategy()
    request = WebProgressiveLoadingRequest(
        positions=positions,
        initial_points=initial_points,
        chunk_points=chunk_points,
        distances=distances,
        label=label,
    )
    return planner.plan(request)
