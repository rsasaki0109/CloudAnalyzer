"""Backward-compatible re-export of the now-promoted nn_thresholds strategy.

The MapEval-style nearest-neighbor threshold strategy has been promoted to
``ca.core.map_evaluate`` as ``NNThresholdMapEvaluateStrategy``. This module
stays as a thin re-export so any existing imports (and the experimental
``map_evaluate`` registry) keep working.
"""

from __future__ import annotations

from ca.core.map_evaluate import (
    NNThresholdMapEvaluateStrategy,
    _min_distances_kdtree,
)

__all__ = ["NNThresholdMapEvaluateStrategy", "_min_distances_kdtree"]
