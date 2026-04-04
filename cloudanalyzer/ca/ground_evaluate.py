"""Ground segmentation evaluation: precision, recall, F1, IoU."""

from __future__ import annotations

import numpy as np

from ca.core.ground_evaluate import GroundEvaluateRequest, evaluate_ground
from ca.io import load_point_cloud


def load_ground_points(path: str) -> np.ndarray:
    """Load point cloud and return Nx3 positions."""
    pcd = load_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float64)


def evaluate_ground_segmentation(
    estimated_ground_path: str,
    estimated_nonground_path: str,
    reference_ground_path: str,
    reference_nonground_path: str,
    voxel_size: float = 0.2,
    min_precision: float | None = None,
    min_recall: float | None = None,
    min_f1: float | None = None,
    min_iou: float | None = None,
) -> dict:
    """Evaluate ground segmentation quality via the stabilized core strategy."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    est_ground = load_ground_points(estimated_ground_path)
    est_nonground = load_ground_points(estimated_nonground_path)
    ref_ground = load_ground_points(reference_ground_path)
    ref_nonground = load_ground_points(reference_nonground_path)

    request = GroundEvaluateRequest(
        estimated_ground=est_ground,
        estimated_nonground=est_nonground,
        reference_ground=ref_ground,
        reference_nonground=ref_nonground,
        voxel_size=voxel_size,
    )
    result = evaluate_ground(request)

    # Quality gate
    gate_reasons: list[str] = []
    if min_precision is not None and result.precision < min_precision:
        gate_reasons.append(f"Precision {result.precision:.4f} < min_precision {min_precision:.4f}")
    if min_recall is not None and result.recall < min_recall:
        gate_reasons.append(f"Recall {result.recall:.4f} < min_recall {min_recall:.4f}")
    if min_f1 is not None and result.f1 < min_f1:
        gate_reasons.append(f"F1 {result.f1:.4f} < min_f1 {min_f1:.4f}")
    if min_iou is not None and result.iou < min_iou:
        gate_reasons.append(f"IoU {result.iou:.4f} < min_iou {min_iou:.4f}")

    has_gate = any(v is not None for v in (min_precision, min_recall, min_f1, min_iou))
    quality_gate = (
        {
            "passed": not gate_reasons,
            "min_precision": min_precision,
            "min_recall": min_recall,
            "min_f1": min_f1,
            "min_iou": min_iou,
            "reasons": gate_reasons,
        }
        if has_gate
        else None
    )

    return {
        "estimated_ground_path": estimated_ground_path,
        "estimated_nonground_path": estimated_nonground_path,
        "reference_ground_path": reference_ground_path,
        "reference_nonground_path": reference_nonground_path,
        "voxel_size": voxel_size,
        "counts": {
            "estimated_ground_points": int(est_ground.shape[0]),
            "estimated_nonground_points": int(est_nonground.shape[0]),
            "reference_ground_points": int(ref_ground.shape[0]),
            "reference_nonground_points": int(ref_nonground.shape[0]),
        },
        "confusion_matrix": {
            "tp": result.tp, "fp": result.fp, "fn": result.fn, "tn": result.tn,
        },
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "iou": result.iou,
        "accuracy": result.accuracy,
        "quality_gate": quality_gate,
    }
