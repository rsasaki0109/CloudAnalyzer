"""Stable core interfaces extracted from experiments."""

from ca.core.checks import (
    CheckOutputs,
    CheckSpec,
    CheckSuite,
    load_check_suite,
    run_check_suite,
)
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
    "CheckOutputs",
    "CheckSpec",
    "CheckSuite",
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
    "load_check_suite",
    "plan_progressive_loading_for_web",
    "reduce_point_cloud_for_web",
    "reduce_trajectory_for_web",
    "run_check_suite",
]
