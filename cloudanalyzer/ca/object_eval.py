"""Shared helpers for 3D object detection and tracking evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Box3D:
    """3D bounding box with optional yaw rotation for detection/tracking evaluation."""

    frame_id: str
    label: str
    center: np.ndarray
    size: np.ndarray
    yaw: float
    score: float
    track_id: str | None
    index: int


@dataclass(frozen=True, slots=True)
class FrameBoxes:
    """Box collection for one frame."""

    frame_id: str
    timestamp: float | None
    boxes: tuple[Box3D, ...]


@dataclass(frozen=True, slots=True)
class BoxSequence:
    """Loaded sequence of per-frame boxes."""

    path: str
    frames: tuple[FrameBoxes, ...]


def _require_mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _vector3(value: object, context: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{context} must be a 3-element array")
    try:
        vector = np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must contain numeric values") from exc
    return vector


def _box_label(raw_box: dict[str, Any]) -> str:
    label = raw_box.get("label", raw_box.get("class", raw_box.get("category")))
    if label is None:
        raise ValueError("box must have a 'label', 'class', or 'category' field")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("box.label must be a non-empty string")
    return label.strip()


def _frame_id(raw_frame: dict[str, Any], index: int) -> str:
    value = raw_frame.get("frame_id", raw_frame.get("id", raw_frame.get("name", raw_frame.get("timestamp"))))
    if value is None:
        return f"{index:06d}"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError("frame_id must be a non-empty string, number, or omitted")


def load_box_sequence(path: str, *, require_track_ids: bool = False) -> BoxSequence:
    """Load a detection/tracking box sequence from JSON."""

    sequence_path = Path(path)
    if not sequence_path.exists():
        raise FileNotFoundError(path)
    if sequence_path.suffix.lower() != ".json":
        raise ValueError("Object evaluation inputs must be JSON files")

    try:
        raw = json.loads(sequence_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {sequence_path}: {exc.msg}") from exc

    if isinstance(raw, list):
        raw_frames = raw
    else:
        root = _require_mapping(raw, "root")
        raw_frames = root.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("Object evaluation JSON must contain a non-empty frames list")

    frames: list[FrameBoxes] = []
    seen_frame_ids: set[str] = set()
    for frame_index, raw_frame in enumerate(raw_frames):
        frame = _require_mapping(raw_frame, f"frames[{frame_index}]")
        frame_id = _frame_id(frame, frame_index)
        if frame_id in seen_frame_ids:
            raise ValueError(f"Duplicate frame_id found: {frame_id}")
        seen_frame_ids.add(frame_id)

        raw_boxes = frame.get("boxes", [])
        if not isinstance(raw_boxes, list):
            raise ValueError(f"frames[{frame_index}].boxes must be a list")

        timestamp_value = frame.get("timestamp")
        timestamp = float(timestamp_value) if isinstance(timestamp_value, (int, float)) else None
        boxes: list[Box3D] = []
        for box_index, raw_box in enumerate(raw_boxes):
            box = _require_mapping(raw_box, f"frames[{frame_index}].boxes[{box_index}]")
            center = _vector3(box.get("center", box.get("position")), "box.center")
            size = _vector3(box.get("size", box.get("dimensions", box.get("extent"))), "box.size")
            if np.any(size <= 0):
                raise ValueError("box.size must be > 0 on every axis")
            score_value = box.get("score", 1.0)
            try:
                score = float(score_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("box.score must be numeric") from exc
            yaw_value = box.get("yaw", box.get("rotation_y", 0.0))
            try:
                yaw = float(yaw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError("box.yaw must be numeric") from exc
            track_id: str | None = None
            if "track_id" in box and box["track_id"] is not None:
                track_id = str(box["track_id"])
            elif require_track_ids:
                raise ValueError("Tracking evaluation requires track_id on every box")
            boxes.append(
                Box3D(
                    frame_id=frame_id,
                    label=_box_label(box),
                    center=center,
                    size=size,
                    yaw=yaw,
                    score=score,
                    track_id=track_id,
                    index=box_index,
                )
            )

        frames.append(
            FrameBoxes(
                frame_id=frame_id,
                timestamp=timestamp,
                boxes=tuple(boxes),
            )
        )

    return BoxSequence(path=path, frames=tuple(frames))


def sequence_counts(sequence: BoxSequence) -> dict[str, int]:
    """Return simple frame/box/track counts for one sequence."""

    track_ids = {
        box.track_id
        for frame in sequence.frames
        for box in frame.boxes
        if box.track_id is not None
    }
    return {
        "frames": len(sequence.frames),
        "boxes": sum(len(frame.boxes) for frame in sequence.frames),
        "tracks": len(track_ids),
    }


def frame_map(sequence: BoxSequence) -> dict[str, FrameBoxes]:
    """Return frame_id -> FrameBoxes lookup."""

    return {frame.frame_id: frame for frame in sequence.frames}


def ordered_frame_ids(reference: BoxSequence, estimated: BoxSequence) -> list[str]:
    """Return stable frame ordering based on reference-first sequence order."""

    ordered = [frame.frame_id for frame in reference.frames]
    seen = set(ordered)
    for frame in estimated.frames:
        if frame.frame_id not in seen:
            ordered.append(frame.frame_id)
            seen.add(frame.frame_id)
    return ordered


def _rotated_corners_2d(center_xy: np.ndarray, size_xy: np.ndarray, yaw: float) -> np.ndarray:
    """Return 4x2 BEV corner vertices for a box rotated by yaw around Z."""
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    half = size_xy / 2.0
    dx, dy = half[0], half[1]
    corners_local = np.array([
        [-dx, -dy],
        [dx, -dy],
        [dx, dy],
        [-dx, dy],
    ])
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    return (corners_local @ rotation.T) + center_xy


def _sutherland_hodgman_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Clip a convex polygon (subject) against another convex polygon (clip)."""
    output = subject.copy()
    num_clip = len(clip)
    for i in range(num_clip):
        if len(output) == 0:
            return output
        edge_start = clip[i]
        edge_end = clip[(i + 1) % num_clip]
        edge_vec = edge_end - edge_start
        new_output: list[np.ndarray] = []
        for j in range(len(output)):
            current = output[j]
            previous = output[j - 1]
            cross_current = float(np.cross(edge_vec, current - edge_start))
            cross_previous = float(np.cross(edge_vec, previous - edge_start))
            if cross_current >= 0.0:
                if cross_previous < 0.0:
                    denom = cross_current - cross_previous
                    if abs(denom) > 1e-12:
                        t = cross_current / denom
                        new_output.append(current + t * (previous - current))
                new_output.append(current)
            elif cross_previous >= 0.0:
                denom = cross_previous - cross_current
                if abs(denom) > 1e-12:
                    t = cross_previous / denom
                    new_output.append(previous + t * (current - previous))
        output = np.array(new_output) if new_output else np.empty((0, 2))
    return output


def _polygon_area(vertices: np.ndarray) -> float:
    """Compute area of a convex polygon using the shoelace formula."""
    if len(vertices) < 3:
        return 0.0
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _aabb_iou_3d(left: Box3D, right: Box3D) -> float:
    """Compute axis-aligned 3D IoU (fast path for yaw=0)."""
    left_min = left.center - left.size / 2.0
    left_max = left.center + left.size / 2.0
    right_min = right.center - right.size / 2.0
    right_max = right.center + right.size / 2.0
    intersection_min = np.maximum(left_min, right_min)
    intersection_max = np.minimum(left_max, right_max)
    overlap = np.maximum(intersection_max - intersection_min, 0.0)
    intersection_volume = float(np.prod(overlap))
    if intersection_volume <= 0.0:
        return 0.0
    left_volume = float(np.prod(left.size))
    right_volume = float(np.prod(right.size))
    union_volume = left_volume + right_volume - intersection_volume
    if union_volume <= 0.0:
        return 0.0
    return float(intersection_volume / union_volume)


def box_iou_3d(left: Box3D, right: Box3D) -> float:
    """Compute 3D IoU supporting oriented boxes via BEV polygon intersection."""
    if left.yaw == 0.0 and right.yaw == 0.0:
        return _aabb_iou_3d(left, right)

    # BEV polygon intersection for XY plane
    left_corners = _rotated_corners_2d(left.center[:2], left.size[:2], left.yaw)
    right_corners = _rotated_corners_2d(right.center[:2], right.size[:2], right.yaw)
    intersection_poly = _sutherland_hodgman_clip(left_corners, right_corners)
    intersection_area = _polygon_area(intersection_poly)
    if intersection_area <= 0.0:
        return 0.0

    # Height overlap along Z axis
    left_z_min = left.center[2] - left.size[2] / 2.0
    left_z_max = left.center[2] + left.size[2] / 2.0
    right_z_min = right.center[2] - right.size[2] / 2.0
    right_z_max = right.center[2] + right.size[2] / 2.0
    z_overlap = max(0.0, min(left_z_max, right_z_max) - max(left_z_min, right_z_min))
    if z_overlap <= 0.0:
        return 0.0

    intersection_volume = intersection_area * z_overlap
    left_volume = float(np.prod(left.size))
    right_volume = float(np.prod(right.size))
    union_volume = left_volume + right_volume - intersection_volume
    if union_volume <= 0.0:
        return 0.0
    return float(intersection_volume / union_volume)


def center_distance(left: Box3D, right: Box3D) -> float:
    """Return Euclidean distance between box centers."""

    return float(np.linalg.norm(left.center - right.center))


def greedy_match_boxes(
    reference_boxes: tuple[Box3D, ...] | list[Box3D],
    estimated_boxes: tuple[Box3D, ...] | list[Box3D],
    *,
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """Greedily match boxes with the same class label above an IoU threshold."""

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be within (0, 1]")

    candidates: list[tuple[float, float, int, int]] = []
    for ref_index, reference_box in enumerate(reference_boxes):
        for est_index, estimated_box in enumerate(estimated_boxes):
            if reference_box.label != estimated_box.label:
                continue
            iou = box_iou_3d(reference_box, estimated_box)
            if iou < iou_threshold:
                continue
            candidates.append((iou, estimated_box.score, ref_index, est_index))

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    used_reference: set[int] = set()
    used_estimated: set[int] = set()
    matches: list[dict[str, Any]] = []
    for iou, _, ref_index, est_index in candidates:
        if ref_index in used_reference or est_index in used_estimated:
            continue
        used_reference.add(ref_index)
        used_estimated.add(est_index)
        reference_box = reference_boxes[ref_index]
        estimated_box = estimated_boxes[est_index]
        matches.append(
            {
                "reference_index": ref_index,
                "estimated_index": est_index,
                "reference_box": reference_box,
                "estimated_box": estimated_box,
                "iou": float(iou),
                "center_distance": center_distance(reference_box, estimated_box),
            }
        )

    unmatched_reference = [
        index for index in range(len(reference_boxes)) if index not in used_reference
    ]
    unmatched_estimated = [
        index for index in range(len(estimated_boxes)) if index not in used_estimated
    ]
    return matches, unmatched_reference, unmatched_estimated
