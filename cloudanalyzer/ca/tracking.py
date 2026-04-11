"""3D multi-object tracking evaluation (box matching + ID switches)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ca.object_eval import BoxSequence, frame_map, greedy_match_boxes, load_box_sequence, ordered_frame_ids, sequence_counts


def _has_yaw(sequence: BoxSequence) -> bool:
    return any(box.yaw != 0.0 for frame in sequence.frames for box in frame.boxes)


def _quality_gate(
    *,
    mota: float,
    recall: float,
    id_switches: int,
    min_mota: float | None = None,
    min_recall: float | None = None,
    max_id_switches: int | None = None,
) -> dict[str, Any] | None:
    if all(value is None for value in (min_mota, min_recall, max_id_switches)):
        return None
    reasons: list[str] = []
    if min_mota is not None and mota < min_mota:
        reasons.append(f"MOTA {mota:.4f} < min_mota {min_mota:.4f}")
    if min_recall is not None and recall < min_recall:
        reasons.append(f"Recall {recall:.4f} < min_recall {min_recall:.4f}")
    if max_id_switches is not None and id_switches > max_id_switches:
        reasons.append(
            f"ID switches {id_switches} > max_id_switches {max_id_switches}"
        )
    return {
        "passed": not reasons,
        "min_mota": min_mota,
        "min_recall": min_recall,
        "max_id_switches": max_id_switches,
        "reasons": reasons,
    }


def evaluate_tracking(
    estimated_path: str,
    reference_path: str,
    *,
    iou_threshold: float = 0.5,
    min_mota: float | None = None,
    min_recall: float | None = None,
    max_id_switches: int | None = None,
) -> dict[str, Any]:
    """Evaluate one 3D tracking sequence against reference annotations."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be within (0, 1]")

    estimated = load_box_sequence(estimated_path, require_track_ids=True)
    reference = load_box_sequence(reference_path, require_track_ids=True)
    estimated_frames = frame_map(estimated)
    reference_frames = frame_map(reference)
    frame_ids = ordered_frame_ids(reference, estimated)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_ious: list[float] = []
    matched_center_distances: list[float] = []
    id_switches = 0
    last_estimated_track_for_reference: dict[str, str] = {}
    track_visibility: dict[str, list[bool]] = defaultdict(list)
    reference_track_totals: dict[str, int] = defaultdict(int)
    reference_track_matches: dict[str, int] = defaultdict(int)
    per_class: dict[str, dict[str, int | float]] = defaultdict(
        lambda: {
            "reference_detections": 0,
            "estimated_detections": 0,
            "matched_detections": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "id_switches": 0,
        }
    )
    matched_samples: list[dict[str, Any]] = []

    for frame_id in frame_ids:
        reference_frame = reference_frames.get(frame_id)
        estimated_frame = estimated_frames.get(frame_id)
        reference_boxes = tuple(reference_frame.boxes) if reference_frame is not None else tuple()
        estimated_boxes = tuple(estimated_frame.boxes) if estimated_frame is not None else tuple()

        for box in reference_boxes:
            reference_track_totals[str(box.track_id)] += 1
            per_class[box.label]["reference_detections"] += 1
        for box in estimated_boxes:
            per_class[box.label]["estimated_detections"] += 1

        matches, unmatched_reference, unmatched_estimated = greedy_match_boxes(
            reference_boxes,
            estimated_boxes,
            iou_threshold=iou_threshold,
        )
        total_tp += len(matches)
        total_fp += len(unmatched_estimated)
        total_fn += len(unmatched_reference)

        matched_reference_tracks = {str(match["reference_box"].track_id) for match in matches}
        for reference_box in reference_boxes:
            track_visibility[str(reference_box.track_id)].append(
                str(reference_box.track_id) in matched_reference_tracks
            )

        for match in matches:
            reference_box = match["reference_box"]
            estimated_box = match["estimated_box"]
            reference_track_id = str(reference_box.track_id)
            estimated_track_id = str(estimated_box.track_id)
            previous_track = last_estimated_track_for_reference.get(reference_track_id)
            if previous_track is not None and previous_track != estimated_track_id:
                id_switches += 1
                per_class[reference_box.label]["id_switches"] += 1
            last_estimated_track_for_reference[reference_track_id] = estimated_track_id
            reference_track_matches[reference_track_id] += 1
            per_class[reference_box.label]["matched_detections"] += 1
            matched_ious.append(match["iou"])
            matched_center_distances.append(match["center_distance"])
            if len(matched_samples) < 10:
                matched_samples.append(
                    {
                        "frame_id": frame_id,
                        "label": reference_box.label,
                        "reference_track_id": reference_track_id,
                        "estimated_track_id": estimated_track_id,
                        "iou": match["iou"],
                        "center_distance": match["center_distance"],
                    }
                )

        for ref_index in unmatched_reference:
            per_class[reference_boxes[ref_index].label]["false_negatives"] += 1
        for est_index in unmatched_estimated:
            per_class[estimated_boxes[est_index].label]["false_positives"] += 1

    precision = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) else 0.0
    recall = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    total_reference = counts_reference = sequence_counts(reference)["boxes"]
    mota = (
        float(1.0 - ((total_fn + total_fp + id_switches) / total_reference))
        if total_reference > 0
        else 0.0
    )
    track_fragmentations = 0
    mostly_tracked = 0
    for track_id, visibility in track_visibility.items():
        segments = 0
        previous_visible = False
        for visible in visibility:
            if visible and not previous_visible:
                segments += 1
            previous_visible = visible
        track_fragmentations += max(segments - 1, 0)
        if reference_track_totals[track_id] > 0 and (
            reference_track_matches[track_id] / reference_track_totals[track_id]
        ) >= 0.8:
            mostly_tracked += 1

    per_class_summary: dict[str, dict[str, Any]] = {}
    for label, counts in sorted(per_class.items()):
        class_precision = (
            counts["matched_detections"] / (counts["matched_detections"] + counts["false_positives"])
            if (counts["matched_detections"] + counts["false_positives"])
            else 0.0
        )
        class_recall = (
            counts["matched_detections"] / (counts["matched_detections"] + counts["false_negatives"])
            if (counts["matched_detections"] + counts["false_negatives"])
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
        }

    quality_gate = _quality_gate(
        mota=mota,
        recall=recall,
        id_switches=id_switches,
        min_mota=min_mota,
        min_recall=min_recall,
        max_id_switches=max_id_switches,
    )
    estimated_counts = sequence_counts(estimated)
    reference_counts = sequence_counts(reference)

    return {
        "estimated_path": estimated_path,
        "reference_path": reference_path,
        "matching_policy": {
            "geometry": "oriented_3d_boxes" if _has_yaw(estimated) or _has_yaw(reference) else "axis_aligned_3d_boxes",
            "class_aware": True,
            "yaw_ignored": not (_has_yaw(estimated) or _has_yaw(reference)),
            "iou_threshold": float(iou_threshold),
        },
        "counts": {
            "estimated_frames": estimated_counts["frames"],
            "reference_frames": reference_counts["frames"],
            "shared_frames": len(set(estimated_frames) & set(reference_frames)),
            "estimated_detections": estimated_counts["boxes"],
            "reference_detections": reference_counts["boxes"],
            "estimated_tracks": estimated_counts["tracks"],
            "reference_tracks": reference_counts["tracks"],
            "matched_detections": total_tp,
        },
        "detection": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "tracking": {
            "mota": mota,
            "id_switches": id_switches,
            "track_fragmentations": track_fragmentations,
            "mostly_tracked_ratio": (
                float(mostly_tracked / reference_counts["tracks"])
                if reference_counts["tracks"] > 0
                else 0.0
            ),
            "mean_iou": float(sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0,
            "mean_center_distance": (
                float(sum(matched_center_distances) / len(matched_center_distances))
                if matched_center_distances
                else 0.0
            ),
        },
        "per_class": per_class_summary,
        "matched_samples": matched_samples,
        "quality_gate": quality_gate,
    }
