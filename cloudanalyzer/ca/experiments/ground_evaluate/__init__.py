"""Experimental strategy set for ground segmentation evaluation."""

from ca.experiments.ground_evaluate.voxel_confusion import VoxelConfusionExperimentalStrategy
from ca.experiments.ground_evaluate.nearest_neighbor import NearestNeighborGroundEvaluateStrategy
from ca.experiments.ground_evaluate.height_band import HeightBandGroundEvaluateStrategy


def get_ground_evaluate_strategies():
    """Return the concrete implementations to compare."""

    return [
        VoxelConfusionExperimentalStrategy(),
        NearestNeighborGroundEvaluateStrategy(),
        HeightBandGroundEvaluateStrategy(),
    ]


__all__ = [
    "HeightBandGroundEvaluateStrategy",
    "NearestNeighborGroundEvaluateStrategy",
    "VoxelConfusionExperimentalStrategy",
    "get_ground_evaluate_strategies",
]
