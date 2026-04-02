"""Stable core interfaces extracted from experiments."""

from ca.core.web_sampling import (
    RandomBudgetWebSamplingStrategy,
    WebSampleRequest,
    WebSampleResult,
    WebSamplingStrategy,
    reduce_point_cloud_for_web,
)
from ca.core.web_progressive_loading import (
    DistanceShellsWebProgressiveLoadingStrategy,
    WebProgressiveLoadingChunk,
    WebProgressiveLoadingRequest,
    WebProgressiveLoadingResult,
    WebProgressiveLoadingStrategy,
    plan_progressive_loading_for_web,
)
from ca.core.web_trajectory_sampling import (
    TurnAwareWebTrajectorySamplingStrategy,
    WebTrajectorySamplingRequest,
    WebTrajectorySamplingResult,
    WebTrajectorySamplingStrategy,
    reduce_trajectory_for_web,
)

__all__ = [
    "DistanceShellsWebProgressiveLoadingStrategy",
    "RandomBudgetWebSamplingStrategy",
    "TurnAwareWebTrajectorySamplingStrategy",
    "WebProgressiveLoadingChunk",
    "WebProgressiveLoadingRequest",
    "WebProgressiveLoadingResult",
    "WebProgressiveLoadingStrategy",
    "WebSampleRequest",
    "WebSampleResult",
    "WebSamplingStrategy",
    "WebTrajectorySamplingRequest",
    "WebTrajectorySamplingResult",
    "WebTrajectorySamplingStrategy",
    "plan_progressive_loading_for_web",
    "reduce_point_cloud_for_web",
    "reduce_trajectory_for_web",
]
