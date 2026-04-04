"""Experimental strategy set for web display point-cloud reduction."""

from ca.experiments.web_sampling.functional_voxel import FunctionalVoxelSamplingStrategy
from ca.experiments.web_sampling.object_random import RandomBudgetSamplingStrategy
from ca.experiments.web_sampling.pipeline_hybrid import HybridPipelineSamplingStrategy


def get_web_sampling_strategies():
    """Return the concrete implementations to compare."""

    return [
        FunctionalVoxelSamplingStrategy(),
        RandomBudgetSamplingStrategy(),
        HybridPipelineSamplingStrategy(),
    ]


__all__ = [
    "FunctionalVoxelSamplingStrategy",
    "RandomBudgetSamplingStrategy",
    "HybridPipelineSamplingStrategy",
    "get_web_sampling_strategies",
]
