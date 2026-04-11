"""3D object detection evaluation (supports axis-aligned and oriented boxes, AP/mAP)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ca.object_eval import (
    BoxSequence,
    frame_map,
    greedy_match_boxes,
    load_box_sequence,
    ordered_frame_ids,
    sequence_counts,
)


def _has_yaw(sequence: BoxSequence) -> bool:
    return any(box.yaw != 0.0 for frame in sequence.frames for box in frame.boxes)


def _normalize_iou_thresholds(iou_thresholds: list[float] | tuple[float, ...] | None) -> tuple[float, ...]:
    if iou_thresholds is None:
        return (0.25, 0.5)
    normalized = tuple(sorted({float(value) for value in iou_thresholds}))
    if not normalized:
        raise ValueError("iou_thresholds must contain at least one threshold")
    for value in normalized:
        if not 0.0 < value <= 1.0:
            raise ValueError("iou_thresholds must be within (0, 1]")
    return normalized


def _resolve_primary_threshold(
    iou_thresholds: tuple[float, ...],
    primary_iou_threshold: float | None,
) -> float:
    if primary_iou_threshold is None:
        return 0.5 if 0.5 in iou_thresholds else iou_thresholds[0]
    threshold = float(primary_iou_threshold)
    if threshold not in iou_thresholds:
        raise ValueError("primary_iou_threshold must be one of iou_thresholds")
    return threshold


def _average_precision(tp_flags: list[int], fp_flags: list[int], num_reference: int) -> float:
    if num_reference <= 0:
        return 0.0
    if not tp_flags:
        return 0.0

    tp = np.cumsum(np.asarray(tp_flags, dtype=np.float64))
    fp = np.cumsum(np.asarray(fp_flags, dtype=np.float64))
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / float(num_reference)

    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_points = np.concatenate(([0.0], recall, [1.0]))
    precision_points = np.concatenate(([precision_envelope[0]], precision_envelope, [0.0]))
    return float(np.sum((recall_points[1:] - recall_points[:-1]) * precision_points[:-1]))


def _threshold_result(
    *,
    reference_frames: dict[str, Any],
    estimated_frames: dict[str, Any],
    frame_ids: list[str],
    iou_threshold: float,
) -> dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_ious: list[float] = []
    matched_center_distances: list[float] = []
    per_class_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "reference_boxes": 0,
            "estimated_boxes": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }
    )

    for frame_id in frame_ids:
        reference_boxes = reference_frames.get(frame_id)
        estimated_boxes = estimated_frames.get(frame_id)
        reference_frame_boxes = tuple(reference_boxes.boxes) if reference_boxes is not None else tuple()
        estimated_frame_boxes = tuple(estimated_boxes.boxes) if estimated_boxes is not None else tuple()
        for box in reference_frame_boxes:
            per_class_counts[box.label]["reference_boxes"] += 1
        for box in estimated_frame_boxes:
            per_class_counts[box.label]["estimated_boxes"] += 1

        matches, unmatched_reference, unmatched_estimated = greedy_match_boxes(
            reference_frame_boxes,
            estimated_frame_boxes,
            iou_threshold=iou_threshold,
        )
        total_tp += len(matches)
        total_fp += len(unmatched_estimated)
        total_fn += len(unmatched_reference)
        for match in matches:
            label = match["reference_box"].label
            per_class_counts[label]["true_positives"] += 1
            matched_ious.append(match["iou"])
            matched_center_distances.append(match["center_distance"])
        for ref_index in unmatched_reference:
            per_class_counts[reference_frame_boxes[ref_index].label]["false_negatives"] += 1
        for est_index in unmatched_estimated:
            per_class_counts[estimated_frame_boxes[est_index].label]["false_positives"] += 1

    precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) else 0.0
    recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    per_class_ap: dict[str, float | None] = {}
    per_class_summary: dict[str, dict[str, Any]] = {}
    reference_classes = {
        box.label for frame in reference_frames.values() for box in frame.boxes
    }
    observed_classes = {
        box.label for frame in reference_frames.values() for box in frame.boxes
    } | {
        box.label for frame in estimated_frames.values() for box in frame.boxes
    }
    for label in sorted(observed_classes):
        prediction_candidates: list[tuple[str, Box3D]] = []
        reference_boxes_by_frame: dict[str, list[Box3D]] = defaultdict(list)
        num_reference = 0
        for frame_id in frame_ids:
            reference_frame = reference_frames.get(frame_id)
            estimated_frame = estimated_frames.get(frame_id)
            if reference_frame is not None:
                for box in reference_frame.boxes:
                    if box.label == label:
                        reference_boxes_by_frame[frame_id].append(box)
                        num_reference += 1
            if estimated_frame is not None:
                for box in estimated_frame.boxes:
                    if box.label == label:
                        prediction_candidates.append((frame_id, box))

        prediction_candidates.sort(
            key=lambda item: (-item[1].score, item[0], item[1].index)
        )
        matched_reference: dict[str, set[int]] = defaultdict(set)
        tp_flags: list[int] = []
        fp_flags: list[int] = []
        for frame_id, prediction in prediction_candidates:
            best_iou = 0.0
            best_ref_index: int | None = None
            for ref_index, reference_box in enumerate(reference_boxes_by_frame.get(frame_id, [])):
                if ref_index in matched_reference[frame_id]:
                    continue
                iou = greedy_match_boxes(
                    [reference_box],
                    [prediction],
                    iou_threshold=iou_threshold,
                )[0]
                candidate_iou = iou[0]["iou"] if iou else 0.0
                if candidate_iou > best_iou:
                    best_iou = candidate_iou
                    best_ref_index = ref_index
            if best_ref_index is not None and best_iou >= iou_threshold:
                matched_reference[frame_id].add(best_ref_index)
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        ap = _average_precision(tp_flags, fp_flags, num_reference) if label in reference_classes else None
        per_class_ap[label] = ap
        counts = per_class_counts[label]
        class_precision = (
            counts["true_positives"] / (counts["true_positives"] + counts["false_positives"])
            if (counts["true_positives"] + counts["false_positives"])
            else 0.0
        )
        class_recall = (
            counts["true_positives"] / (counts["true_positives"] + counts["false_negatives"])
            if (counts["true_positives"] + counts["false_negatives"])
            else 0.0
        )
        class_f1 = (
            2.0 * class_precision * class_recall / (class_precision + class_recall)
            if (class_precision + class_recall)
            else 0.0
        )
        per_class_summary[label] = {
            **counts,
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1": float(class_f1),
            "ap": None if ap is None else float(ap),
        }

    reference_class_aps = [ap for label, ap in per_class_ap.items() if label in reference_classes and ap is not None]
    mean_ap = float(sum(reference_class_aps) / len(reference_class_aps)) if reference_class_aps else 0.0
    return {
        "iou_threshold": float(iou_threshold),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0,
        "mean_center_distance": (
            float(sum(matched_center_distances) / len(matched_center_distances))
            if matched_center_distances
            else 0.0
        ),
        "matched_detections": len(matched_ious),
        "per_class": per_class_summary,
        "per_class_ap": per_class_ap,
        "map": mean_ap,
    }


def _quality_gate(
    *,
    map_value: float,
    precision: float,
    recall: float,
    f1: float,
    min_map: float | None = None,
    min_precision: float | None = None,
    min_recall: float | None = None,
    min_f1: float | None = None,
) -> dict[str, Any] | None:
    if all(value is None for value in (min_map, min_precision, min_recall, min_f1)):
        return None
    reasons: list[str] = []
    if min_map is not None and map_value < min_map:
        reasons.append(f"mAP {map_value:.4f} < min_map {min_map:.4f}")
    if min_precision is not None and precision < min_precision:
        reasons.append(f"Precision {precision:.4f} < min_precision {min_precision:.4f}")
    if min_recall is not None and recall < min_recall:
        reasons.append(f"Recall {recall:.4f} < min_recall {min_recall:.4f}")
    if min_f1 is not None and f1 < min_f1:
        reasons.append(f"F1 {f1:.4f} < min_f1 {min_f1:.4f}")
    return {
        "passed": not reasons,
        "min_map": min_map,
        "min_precision": min_precision,
        "min_recall": min_recall,
        "min_f1": min_f1,
        "reasons": reasons,
    }


def evaluate_detection(
    estimated_path: str,
    reference_path: str,
    *,
    iou_thresholds: list[float] | tuple[float, ...] | None = None,
    primary_iou_threshold: float | None = None,
    min_map: float | None = None,
    min_precision: float | None = None,
    min_recall: float | None = None,
    min_f1: float | None = None,
) -> dict[str, Any]:
    """Evaluate one sequence of 3D detections against reference annotations."""

    thresholds = _normalize_iou_thresholds(iou_thresholds)
    primary_threshold = _resolve_primary_threshold(thresholds, primary_iou_threshold)

    estimated = load_box_sequence(estimated_path)
    reference = load_box_sequence(reference_path)
    estimated_frames = frame_map(estimated)
    reference_frames = frame_map(reference)
    frame_ids = ordered_frame_ids(reference, estimated)

    threshold_results = [
        _threshold_result(
            reference_frames=reference_frames,
            estimated_frames=estimated_frames,
            frame_ids=frame_ids,
            iou_threshold=threshold,
        )
        for threshold in thresholds
    ]
    primary_result = next(result for result in threshold_results if result["iou_threshold"] == primary_threshold)
    per_class: dict[str, dict[str, Any]] = {}
    for label in sorted(primary_result["per_class"]):
        ap_by_threshold = {
            f"{result['iou_threshold']:.2f}": result["per_class_ap"].get(label)
            for result in threshold_results
        }
        valid_aps = [value for value in ap_by_threshold.values() if value is not None]
        per_class[label] = {
            **primary_result["per_class"][label],
            "ap_by_threshold": ap_by_threshold,
            "mean_ap": float(sum(valid_aps) / len(valid_aps)) if valid_aps else None,
        }

    mean_ap = float(sum(result["map"] for result in threshold_results) / len(threshold_results))
    best_threshold = max(threshold_results, key=lambda result: result["f1"])
    counts_estimated = sequence_counts(estimated)
    counts_reference = sequence_counts(reference)
    quality_gate = _quality_gate(
        map_value=mean_ap,
        precision=primary_result["precision"],
        recall=primary_result["recall"],
        f1=primary_result["f1"],
        min_map=min_map,
        min_precision=min_precision,
        min_recall=min_recall,
        min_f1=min_f1,
    )

    return {
        "estimated_path": estimated_path,
        "reference_path": reference_path,
        "matching_policy": {
            "geometry": "oriented_3d_boxes" if _has_yaw(estimated) or _has_yaw(reference) else "axis_aligned_3d_boxes",
            "class_aware": True,
            "yaw_ignored": not (_has_yaw(estimated) or _has_yaw(reference)),
            "iou_thresholds": list(thresholds),
            "primary_iou_threshold": float(primary_threshold),
        },
        "counts": {
            "estimated_frames": counts_estimated["frames"],
            "reference_frames": counts_reference["frames"],
            "shared_frames": len(set(estimated_frames) & set(reference_frames)),
            "estimated_boxes": counts_estimated["boxes"],
            "reference_boxes": counts_reference["boxes"],
        },
        "mAP": mean_ap,
        "best_threshold": {
            "iou_threshold": best_threshold["iou_threshold"],
            "f1": best_threshold["f1"],
            "precision": best_threshold["precision"],
            "recall": best_threshold["recall"],
        },
        "primary_threshold_result": primary_result,
        "threshold_results": threshold_results,
        "per_class": per_class,
        "quality_gate": quality_gate,
    }
