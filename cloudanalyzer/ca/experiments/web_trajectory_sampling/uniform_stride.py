"""Function-first trajectory simplification by uniform stride allocation."""

from __future__ import annotations

from ca.core.web_trajectory_sampling import WebTrajectorySamplingRequest, WebTrajectorySamplingResult
from ca.experiments.web_trajectory_sampling.common import allocate_evenly_spaced_indices


def reduce_with_uniform_stride(request: WebTrajectorySamplingRequest) -> WebTrajectorySamplingResult:
    """Reduce a trajectory by allocating evenly spaced samples along the index domain."""

    original_points = request.positions.shape[0]
    keep_indices = allocate_evenly_spaced_indices(
        total_points=original_points,
        budget=request.max_points,
        preserve_indices=request.preserve_indices,
    )
    timestamps = request.timestamps[keep_indices] if request.timestamps is not None else None
    return WebTrajectorySamplingResult(
        positions=request.positions[keep_indices],
        timestamps=timestamps,
        kept_indices=keep_indices,
        strategy="uniform_stride",
        design="functional",
        original_points=original_points,
        reduced_points=int(keep_indices.size),
        metadata={
            "label": request.label,
            "preserved_points": len(request.preserve_indices),
        },
    )


class UniformStrideStrategy:
    """Thin adapter over the functional stride implementation."""

    name = "uniform_stride"
    design = "functional"

    def reduce(self, request: WebTrajectorySamplingRequest) -> WebTrajectorySamplingResult:
        return reduce_with_uniform_stride(request)
