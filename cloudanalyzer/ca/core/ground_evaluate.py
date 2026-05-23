"""Stable, minimal interface for ground segmentation evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(slots=True)
class GroundEvaluateRequest:
    """Input contract shared by all ground segmentation evaluation strategies."""

    estimated_ground: np.ndarray
    estimated_nonground: np.ndarray
    reference_ground: np.ndarray
    reference_nonground: np.ndarray
    voxel_size: float = 0.2


@dataclass(slots=True)
class GroundEvaluateResult:
    """Evaluation output shared by all strategies."""

    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    iou: float
    accuracy: float
    strategy: str
    design: str
    metadata: dict[str, Any] = field(default_factory=dict)


class GroundEvaluateStrategy(Protocol):
    """Protocol kept in core after comparing concrete ground evaluation strategies."""

    name: str
    design: str

    def evaluate(self, request: GroundEvaluateRequest) -> GroundEvaluateResult:
        """Evaluate ground segmentation quality."""


def _voxel_keys(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Compute unique voxel grid keys for an Nx3 point array.

    Returns an (M, 3) int64 array of unique (i, j, k) voxel indices, sorted
    lexicographically. The original implementation built a ``set`` of Python
    tuples via a per-row generator, which dominated runtime on city-scale
    inputs (millions of points × 4 calls per evaluation).
    """
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64)
    indices = np.floor(points / voxel_size).astype(np.int64)
    return np.unique(indices, axis=0)


def _voxel_intersection_size(a: np.ndarray, b: np.ndarray) -> int:
    """Count voxel keys present in both ``a`` and ``b``.

    Both inputs are expected to be the unique (M, 3) int64 output of
    :func:`_voxel_keys`. Packs each row into a single ``np.void`` of the row's
    byte length so ``np.intersect1d`` can run in C.
    """
    if a.size == 0 or b.size == 0:
        return 0
    item_size = a.dtype.itemsize * a.shape[1]
    void_dtype = np.dtype((np.void, item_size))
    a_view = np.ascontiguousarray(a).view(void_dtype).ravel()
    b_view = np.ascontiguousarray(b).view(void_dtype).ravel()
    return int(np.intersect1d(a_view, b_view, assume_unique=True).size)


def confusion_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """Compute precision, recall, F1, IoU, accuracy from confusion counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
    }


class VoxelConfusionGroundEvaluateStrategy:
    """Stable ground evaluation strategy selected after experiment comparison."""

    name = "voxel_confusion"
    design = "functional"

    def evaluate(self, request: GroundEvaluateRequest) -> GroundEvaluateResult:
        est_ground_voxels = _voxel_keys(request.estimated_ground, request.voxel_size)
        est_nonground_voxels = _voxel_keys(request.estimated_nonground, request.voxel_size)
        ref_ground_voxels = _voxel_keys(request.reference_ground, request.voxel_size)
        ref_nonground_voxels = _voxel_keys(request.reference_nonground, request.voxel_size)

        tp = _voxel_intersection_size(est_ground_voxels, ref_ground_voxels)
        fp = _voxel_intersection_size(est_ground_voxels, ref_nonground_voxels)
        fn = _voxel_intersection_size(est_nonground_voxels, ref_ground_voxels)
        tn = _voxel_intersection_size(est_nonground_voxels, ref_nonground_voxels)
        metrics = confusion_metrics(tp, fp, fn, tn)

        return GroundEvaluateResult(
            tp=tp, fp=fp, fn=fn, tn=tn,
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            iou=metrics["iou"],
            accuracy=metrics["accuracy"],
            strategy=self.name,
            design=self.design,
        )


def evaluate_ground(
    request: GroundEvaluateRequest,
    strategy: GroundEvaluateStrategy | None = None,
) -> GroundEvaluateResult:
    """Evaluate ground segmentation using the stabilized strategy."""
    eval_strategy = strategy or VoxelConfusionGroundEvaluateStrategy()
    return eval_strategy.evaluate(request)
