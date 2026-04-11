"""KITTI 3D object detection label format parser."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


# KITTI label columns: type truncated occluded alpha
#   bbox2d(left top right bottom) dim(h w l) loc(x y z) rotation_y [score]
_MIN_COLUMNS = 15
_MAX_COLUMNS = 16

_SKIP_TYPES = frozenset({"DontCare", "Misc"})


def parse_kitti_label_file(
    path: str,
    *,
    camera_to_lidar: bool = True,
) -> list[dict[str, Any]]:
    """Parse one KITTI label .txt file into CloudAnalyzer box dicts.

    Args:
        path: Path to a KITTI label file.
        camera_to_lidar: Apply standard KITTI camera-to-Velodyne transform.

    Returns:
        List of box dicts with label, center, size, yaw, and optional score.
    """
    text = Path(path).read_text(encoding="utf-8")
    boxes: list[dict[str, Any]] = []
    for line in text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < _MIN_COLUMNS:
            continue
        label_type = parts[0]
        if label_type in _SKIP_TYPES:
            continue

        # Dimensions: height, width, length (KITTI order)
        h = float(parts[8])
        w = float(parts[9])
        l = float(parts[10])  # noqa: E741

        # Location: x, y, z in camera coordinates
        cam_x = float(parts[11])
        cam_y = float(parts[12])
        cam_z = float(parts[13])

        # Rotation around Y axis in camera frame
        rotation_y = float(parts[14])

        # Optional score
        score: float | None = None
        if len(parts) >= _MAX_COLUMNS:
            score = float(parts[15])

        if camera_to_lidar:
            # Standard KITTI camera-to-Velodyne:
            # lidar_x = cam_z, lidar_y = -cam_x, lidar_z = -(cam_y - h/2)
            center = [cam_z, -cam_x, -(cam_y - h / 2.0)]
            size = [l, w, h]
            # Camera rotation_y maps to negative yaw in lidar frame
            yaw = -rotation_y
        else:
            center = [cam_x, cam_y, cam_z]
            size = [l, w, h]
            yaw = rotation_y

        box: dict[str, Any] = {
            "label": label_type,
            "center": [round(v, 6) for v in center],
            "size": [round(v, 6) for v in size],
            "yaw": round(yaw, 6),
        }
        if score is not None:
            box["score"] = round(score, 6)
        boxes.append(box)

    return boxes


def convert_kitti_labels(
    input_dir: str,
    output_path: str,
    *,
    camera_to_lidar: bool = True,
) -> dict[str, Any]:
    """Convert a directory of KITTI label files to CloudAnalyzer JSON.

    Args:
        input_dir: Directory containing KITTI .txt label files.
        output_path: Output JSON file path.
        camera_to_lidar: Apply standard KITTI camera-to-Velodyne transform.

    Returns:
        Summary dict with frame count and total box count.
    """
    label_dir = Path(input_dir)
    if not label_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        raise ValueError(f"No .txt files found in: {input_dir}")

    frames: list[dict[str, Any]] = []
    total_boxes = 0
    for label_file in label_files:
        frame_id = label_file.stem
        boxes = parse_kitti_label_file(str(label_file), camera_to_lidar=camera_to_lidar)
        total_boxes += len(boxes)
        frames.append({
            "frame_id": frame_id,
            "boxes": boxes,
        })

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"frames": frames}, indent=2) + "\n", encoding="utf-8")

    return {
        "input_dir": input_dir,
        "output_path": output_path,
        "frames": len(frames),
        "total_boxes": total_boxes,
        "camera_to_lidar": camera_to_lidar,
    }
