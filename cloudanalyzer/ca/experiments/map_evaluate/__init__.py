"""Experimental point-cloud map evaluation strategies (MapEval-inspired).

This slice is intentionally experimental: it compares multiple designs for map-to-map
evaluation (GT available) and map self-consistency (no GT).
"""

from __future__ import annotations

from ca.experiments.map_evaluate.nn_thresholds import NNThresholdMapEvaluateStrategy
from ca.experiments.map_evaluate.voxel_entropy import VoxelEntropyMapEvaluateStrategy


def get_map_evaluate_strategies():
    return [
        NNThresholdMapEvaluateStrategy(),
        VoxelEntropyMapEvaluateStrategy(),
    ]

