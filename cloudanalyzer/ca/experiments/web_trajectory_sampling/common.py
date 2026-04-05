"""Shared helpers for trajectory simplification experiments."""

from __future__ import annotations

import numpy as np


def normalize_preserve_indices(total_points: int, preserve_indices: tuple[int, ...]) -> tuple[int, ...]:
    """Normalize requested anchor indices and always keep endpoints."""

    if total_points <= 0:
        return ()
    normalized = {0, total_points - 1}
    normalized.update(index for index in preserve_indices if 0 <= index < total_points)
    return tuple(sorted(normalized))


def allocate_evenly_spaced_indices(
    total_points: int,
    budget: int,
    preserve_indices: tuple[int, ...] = (),
) -> np.ndarray:
    """Allocate evenly spaced indices while keeping anchor indices."""

    if total_points <= 0:
        return np.zeros(0, dtype=int)

    preserve = list(normalize_preserve_indices(total_points, preserve_indices))
    effective_budget = max(budget, len(preserve))
    if total_points <= effective_budget:
        return np.arange(total_points, dtype=int)
    if len(preserve) >= effective_budget:
        return np.asarray(preserve[:effective_budget], dtype=int)

    extras = effective_budget - len(preserve)
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
        indices = np.linspace(left, right, num=quota + 2, dtype=int)[1:-1]
        selected.update(int(index) for index in indices if left < index < right)

    if len(selected) < effective_budget:
        for candidate in np.linspace(0, total_points - 1, num=effective_budget, dtype=int):
            selected.add(int(candidate))
            if len(selected) >= effective_budget:
                break

    return np.asarray(sorted(selected), dtype=int)


def shrink_sorted_indices(
    indices: np.ndarray,
    budget: int,
    preserve_indices: tuple[int, ...] = (),
) -> np.ndarray:
    """Shrink a sorted index list while keeping anchor indices."""

    sorted_indices = np.asarray(indices, dtype=int)
    if sorted_indices.size <= budget:
        return sorted_indices

    position_of = {int(index): pos for pos, index in enumerate(sorted_indices.tolist())}
    preserve_positions = {0, sorted_indices.size - 1}
    preserve_positions.update(
        position_of[index] for index in preserve_indices if index in position_of
    )
    keep_positions = allocate_evenly_spaced_indices(
        total_points=sorted_indices.size,
        budget=budget,
        preserve_indices=tuple(sorted(preserve_positions)),
    )
    return np.asarray(sorted_indices[keep_positions], dtype=int)


def path_length(positions: np.ndarray) -> float:
    """Compute cumulative Euclidean path length."""

    if positions.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def reconstruct_positions(
    original_timestamps: np.ndarray,
    sampled_timestamps: np.ndarray,
    sampled_positions: np.ndarray,
) -> np.ndarray:
    """Interpolate simplified positions back to original timestamps."""

    if sampled_timestamps.size == 0:
        return np.zeros((0, 3), dtype=float)
    if sampled_timestamps.size == 1:
        return np.repeat(sampled_positions[:1], repeats=original_timestamps.size, axis=0)

    reconstructed = np.column_stack(
        [
            np.interp(original_timestamps, sampled_timestamps, sampled_positions[:, axis])
            for axis in range(3)
        ]
    )
    return reconstructed
