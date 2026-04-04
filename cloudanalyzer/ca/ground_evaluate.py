"""Ground segmentation evaluation: precision, recall, F1, IoU."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from ca.io import load_point_cloud


def _points_array(pcd) -> np.ndarray:
    """Extract Nx3 numpy array from an Open3D point cloud."""
    return np.asarray(pcd.points, dtype=np.float64)


def load_ground_points(path: str) -> np.ndarray:
    """Load point cloud and return Nx3 positions."""
    pcd = load_point_cloud(path)
    return _points_array(pcd)


def _voxel_keys(points: np.ndarray, voxel_size: float) -> set[tuple[int, int, int]]:
    """Compute voxel grid keys for an Nx3 point array."""
    indices = np.floor(points / voxel_size).astype(np.int64)
    return {(int(row[0]), int(row[1]), int(row[2])) for row in indices}


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
    """Evaluate ground segmentation quality via voxel-based confusion matrix.

    Each point cloud is voxelized at the given resolution. For each occupied voxel,
    the label (ground or non-ground) is compared between the estimation and reference.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    est_ground = load_ground_points(estimated_ground_path)
    est_nonground = load_ground_points(estimated_nonground_path)
    ref_ground = load_ground_points(reference_ground_path)
    ref_nonground = load_ground_points(reference_nonground_path)

    est_ground_voxels = _voxel_keys(est_ground, voxel_size)
    est_nonground_voxels = _voxel_keys(est_nonground, voxel_size)
    ref_ground_voxels = _voxel_keys(ref_ground, voxel_size)
    ref_nonground_voxels = _voxel_keys(ref_nonground, voxel_size)

    # Confusion matrix over voxels
    tp = len(est_ground_voxels & ref_ground_voxels)
    fp = len(est_ground_voxels & ref_nonground_voxels)
    fn = len(est_nonground_voxels & ref_ground_voxels)
    tn = len(est_nonground_voxels & ref_nonground_voxels)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    # Quality gate
    gate_reasons: list[str] = []
    if min_precision is not None and precision < min_precision:
        gate_reasons.append(f"Precision {precision:.4f} < min_precision {min_precision:.4f}")
    if min_recall is not None and recall < min_recall:
        gate_reasons.append(f"Recall {recall:.4f} < min_recall {min_recall:.4f}")
    if min_f1 is not None and f1 < min_f1:
        gate_reasons.append(f"F1 {f1:.4f} < min_f1 {min_f1:.4f}")
    if min_iou is not None and iou < min_iou:
        gate_reasons.append(f"IoU {iou:.4f} < min_iou {min_iou:.4f}")

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
        "voxel_counts": {
            "estimated_ground_voxels": len(est_ground_voxels),
            "estimated_nonground_voxels": len(est_nonground_voxels),
            "reference_ground_voxels": len(ref_ground_voxels),
            "reference_nonground_voxels": len(ref_nonground_voxels),
        },
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "quality_gate": quality_gate,
    }
