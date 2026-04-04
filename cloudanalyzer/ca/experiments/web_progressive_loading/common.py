"""Shared helpers for progressive loading experiments."""

from __future__ import annotations

import numpy as np

from ca.core.web_progressive_loading import (
    WebProgressiveLoadingChunk,
    WebProgressiveLoadingRequest,
    WebProgressiveLoadingResult,
)


def normalize_positions(positions: np.ndarray) -> np.ndarray:
    """Normalize positions into a unit cube."""

    values = np.asarray(positions, dtype=float)
    if values.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    mins = np.min(values, axis=0)
    spans = np.max(values, axis=0) - mins
    spans[spans < 1e-9] = 1.0
    return np.asarray((values - mins) / spans, dtype=float)


def split_indices_into_chunks(indices: np.ndarray, chunk_points: int) -> tuple[np.ndarray, ...]:
    """Split indices into stable chunk slices."""

    if indices.size == 0:
        return ()
    return tuple(
        indices[start : start + chunk_points]
        for start in range(0, indices.size, chunk_points)
    )


def build_result_from_order(
    request: WebProgressiveLoadingRequest,
    ordered_indices: np.ndarray,
    strategy: str,
    design: str,
    metadata: dict,
) -> WebProgressiveLoadingResult:
    """Build a comparable result from an ordered index stream."""

    indices = np.asarray(ordered_indices, dtype=int)
    if indices.size == 0:
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

    initial_size = min(request.initial_points, indices.size)
    initial_indices = indices[:initial_size]
    chunk_index_groups = split_indices_into_chunks(indices[initial_size:], request.chunk_points)
    chunks = tuple(
        WebProgressiveLoadingChunk(
            positions=request.positions[group],
            distances=request.distances[group] if request.distances is not None else None,
            metadata={"chunk_index": chunk_index},
        )
        for chunk_index, group in enumerate(chunk_index_groups)
    )
    return WebProgressiveLoadingResult(
        initial_positions=request.positions[initial_indices],
        initial_distances=(
            request.distances[initial_indices] if request.distances is not None else None
        ),
        chunks=chunks,
        strategy=strategy,
        design=design,
        original_points=int(request.positions.shape[0]),
        initial_points=int(initial_indices.size),
        chunk_points=request.chunk_points,
        metadata=metadata,
    )


def progressive_prefix_points(result: WebProgressiveLoadingResult) -> list[np.ndarray]:
    """Materialize progressive prefixes for evaluation."""

    prefixes = [result.initial_positions]
    loaded = [result.initial_positions]
    for chunk in result.chunks:
        loaded.append(chunk.positions)
        prefixes.append(np.vstack(loaded))
    return prefixes


def coverage_p95(full_positions: np.ndarray, sampled_positions: np.ndarray) -> float:
    """Compute nearest-neighbor coverage p95 exactly for small synthetic datasets."""

    if full_positions.shape[0] == 0 or sampled_positions.shape[0] == 0:
        return 0.0
    deltas = full_positions[:, None, :] - sampled_positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    nearest = np.min(distances, axis=1)
    return float(np.quantile(nearest, 0.95))


def chunk_size_std(result: WebProgressiveLoadingResult) -> float:
    """Measure chunk size balance."""

    if not result.chunks:
        return 0.0
    return float(np.std([chunk.point_count for chunk in result.chunks]))
