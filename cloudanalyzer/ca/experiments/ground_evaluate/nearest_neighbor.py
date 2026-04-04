"""OOP nearest-neighbor ground evaluation strategy."""

from __future__ import annotations

import numpy as np

from ca.core.ground_evaluate import (
    GroundEvaluateRequest,
    GroundEvaluateResult,
    confusion_metrics,
)


class _PointClassifier:
    """Classify reference points by nearest-neighbor lookup against estimated sets."""

    def __init__(self, estimated_ground: np.ndarray, estimated_nonground: np.ndarray) -> None:
        self._all_points = np.vstack([estimated_ground, estimated_nonground])
        self._labels = np.concatenate([
            np.ones(estimated_ground.shape[0], dtype=np.int8),
            np.zeros(estimated_nonground.shape[0], dtype=np.int8),
        ])

    def classify(self, query_points: np.ndarray) -> np.ndarray:
        """Return per-point ground label (1=ground, 0=nonground) via nearest neighbor."""
        if self._all_points.shape[0] == 0:
            return np.zeros(query_points.shape[0], dtype=np.int8)
        # Brute-force nearest neighbor (fine for experiment-sized datasets)
        labels = np.empty(query_points.shape[0], dtype=np.int8)
        for i, point in enumerate(query_points):
            dists = np.sum((self._all_points - point) ** 2, axis=1)
            labels[i] = self._labels[int(np.argmin(dists))]
        return labels


class NearestNeighborGroundEvaluateStrategy:
    """Evaluate by classifying each reference point via nearest estimated neighbor."""

    name = "nearest_neighbor"
    design = "oop"

    def evaluate(self, request: GroundEvaluateRequest) -> GroundEvaluateResult:
        classifier = _PointClassifier(request.estimated_ground, request.estimated_nonground)

        # Classify reference ground points
        ref_ground_preds = classifier.classify(request.reference_ground)
        tp = int(np.sum(ref_ground_preds == 1))
        fn = int(np.sum(ref_ground_preds == 0))

        # Classify reference nonground points
        ref_nonground_preds = classifier.classify(request.reference_nonground)
        fp = int(np.sum(ref_nonground_preds == 1))
        tn = int(np.sum(ref_nonground_preds == 0))

        metrics = confusion_metrics(tp, fp, fn, tn)
        return GroundEvaluateResult(
            tp=tp, fp=fp, fn=fn, tn=tn,
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            iou=metrics["iou"],
            accuracy=metrics["accuracy"],
            strategy=self.name, design=self.design,
            metadata={"matching": "nearest_neighbor_per_reference_point"},
        )
