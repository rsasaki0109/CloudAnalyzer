#!/usr/bin/env python3
"""Build public detection/tracking JSON examples from the RELLIS-3D seed frame.

Detection boxes are derived from semantic label clusters on the public frame.
Tracking sequences are deterministic synthetic 3-frame sequences seeded from
those derived boxes so the JSON contract can be demonstrated without bundling a
full public MOT dataset in-repo.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "cloudanalyzer"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from public_benchmark_assets import download_rellis_lidar_example

DEFAULT_FRAME = "000001"
LOCAL_RELLIS_FRAME_ROOT = REPO_ROOT / "demo_assets" / "public" / "rellis3d-frame-000001"

LABEL_SPECS = (
    {
        "label_id": 17,
        "label": "person",
        "cluster_eps": 0.8,
        "min_points": 10,
        "cluster_rank": 0,
    },
    {
        "label_id": 31,
        "label": "puddle",
        "cluster_eps": 0.6,
        "min_points": 20,
        "cluster_rank": 0,
    },
    {
        "label_id": 19,
        "label": "bush",
        "cluster_eps": 1.5,
        "min_points": 30,
        "cluster_rank": 0,
    },
)


def _read_rellis_scan(example_root: Path, frame: str) -> tuple[np.ndarray, np.ndarray]:
    bin_path = example_root / "os1_cloud_node_kitti_bin" / f"{frame}.bin"
    label_path = example_root / "os1_cloud_node_semantickitti_label_id" / f"{frame}.label"
    if not bin_path.exists():
        raise FileNotFoundError(f"missing point cloud frame: {bin_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"missing label frame: {label_path}")
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    labels = np.fromfile(label_path, dtype=np.uint32)
    if points.shape[0] != labels.shape[0]:
        raise RuntimeError("point and label counts do not match")
    return np.asarray(points, dtype=np.float64), labels


def _cluster_boxes(points: np.ndarray, labels: np.ndarray) -> list[dict]:
    boxes: list[dict] = []
    for spec in LABEL_SPECS:
        label_points = points[labels == spec["label_id"]]
        if label_points.size == 0:
            raise RuntimeError(f"label {spec['label_id']} ({spec['label']}) is empty in the seed frame")
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(label_points)
        cluster_ids = np.asarray(
            cloud.cluster_dbscan(
                eps=float(spec["cluster_eps"]),
                min_points=int(spec["min_points"]),
                print_progress=False,
            ),
            dtype=np.int64,
        )
        valid_cluster_ids = [cluster_id for cluster_id in np.unique(cluster_ids) if cluster_id >= 0]
        if not valid_cluster_ids:
            raise RuntimeError(f"no clusters found for label {spec['label']}")
        ranked_clusters = sorted(
            valid_cluster_ids,
            key=lambda cluster_id: int(np.sum(cluster_ids == cluster_id)),
            reverse=True,
        )
        cluster_id = ranked_clusters[int(spec["cluster_rank"])]
        cluster_points = label_points[cluster_ids == cluster_id]
        min_corner = np.min(cluster_points, axis=0)
        max_corner = np.max(cluster_points, axis=0)
        center = ((min_corner + max_corner) / 2.0).tolist()
        size = np.maximum(max_corner - min_corner, np.array([0.15, 0.15, 0.15], dtype=np.float64)).tolist()
        boxes.append(
            {
                "label": spec["label"],
                "center": [round(float(value), 6) for value in center],
                "size": [round(float(value), 6) for value in size],
                "source_label_id": int(spec["label_id"]),
                "source_cluster_rank": int(spec["cluster_rank"]),
                "source_point_count": int(cluster_points.shape[0]),
            }
        )
    return boxes


def _with_box_fields(box: dict, *, center_delta: tuple[float, float, float] = (0.0, 0.0, 0.0), size_scale: float = 1.0, score: float | None = None, track_id: str | None = None) -> dict:
    center = [
        round(float(box["center"][index] + center_delta[index]), 6)
        for index in range(3)
    ]
    size = [round(float(value * size_scale), 6) for value in box["size"]]
    payload = {
        "label": box["label"],
        "center": center,
        "size": size,
    }
    if score is not None:
        payload["score"] = round(float(score), 6)
    if track_id is not None:
        payload["track_id"] = track_id
    return payload


def _detection_payloads(reference_boxes: list[dict]) -> tuple[dict, dict, dict]:
    by_label = {box["label"]: box for box in reference_boxes}
    reference = {
        "frames": [
            {
                "frame_id": DEFAULT_FRAME,
                "boxes": [
                    _with_box_fields(by_label["person"]),
                    _with_box_fields(by_label["puddle"]),
                    _with_box_fields(by_label["bush"]),
                ],
            }
        ]
    }
    estimated_good = {
        "frames": [
            {
                "frame_id": DEFAULT_FRAME,
                "boxes": [
                    _with_box_fields(by_label["person"], center_delta=(0.03, -0.02, 0.0), score=0.97),
                    _with_box_fields(by_label["puddle"], center_delta=(0.08, 0.05, 0.0), size_scale=0.98, score=0.92),
                    _with_box_fields(by_label["bush"], center_delta=(-0.25, 0.15, 0.0), size_scale=1.01, score=0.88),
                ],
            }
        ]
    }
    estimated_regressed = {
        "frames": [
            {
                "frame_id": DEFAULT_FRAME,
                "boxes": [
                    _with_box_fields(by_label["person"], center_delta=(0.55, 0.45, 0.0), size_scale=0.82, score=0.91),
                    _with_box_fields(by_label["bush"], center_delta=(3.2, -1.5, 0.0), size_scale=0.88, score=0.77),
                    {
                        "label": "barrier",
                        "center": [-3.25, 8.5, 0.2],
                        "size": [2.4, 0.8, 1.1],
                        "score": 0.42,
                    },
                ],
            }
        ]
    }
    return reference, estimated_good, estimated_regressed


def _tracking_payloads(reference_boxes: list[dict]) -> tuple[dict, dict, dict]:
    by_label = {box["label"]: box for box in reference_boxes}
    reference_offsets = {
        "person": ((0.0, 0.0, 0.0), (0.18, 0.03, 0.0), (0.35, 0.05, 0.0)),
        "puddle": ((0.0, 0.0, 0.0), (0.02, 0.0, 0.0), (0.04, -0.01, 0.0)),
        "bush": ((0.0, 0.0, 0.0), (0.12, -0.05, 0.0), (0.24, -0.08, 0.0)),
    }
    estimated_offsets = {
        "person": ((0.01, -0.01, 0.0), (0.02, 0.0, 0.0), (0.02, 0.01, 0.0)),
        "puddle": ((0.02, 0.01, 0.0), (0.02, 0.0, 0.0), (0.02, -0.01, 0.0)),
        "bush": ((-0.03, 0.02, 0.0), (0.03, -0.01, 0.0), (0.04, -0.02, 0.0)),
    }

    frame_ids = ["000001", "000002", "000003"]
    reference_frames = []
    estimated_good_frames = []
    estimated_regressed_frames = []
    for frame_index, frame_id in enumerate(frame_ids):
        reference_frame_boxes = []
        estimated_frame_boxes = []
        estimated_regressed_boxes = []
        for label in ("person", "puddle", "bush"):
            base_box = by_label[label]
            reference_frame_boxes.append(
                _with_box_fields(
                    base_box,
                    center_delta=reference_offsets[label][frame_index],
                    track_id=f"gt-{label}",
                )
            )
            estimated_frame_boxes.append(
                _with_box_fields(
                    base_box,
                    center_delta=tuple(
                        reference_offsets[label][frame_index][axis] + estimated_offsets[label][frame_index][axis]
                        for axis in range(3)
                    ),
                    track_id=f"pred-{label}",
                )
            )
            if label == "person" and frame_index == 1:
                continue
            estimated_track_id = (
                "pred-bush-alt"
                if label == "bush" and frame_index == 2
                else f"pred-{label}"
            )
            center_delta = tuple(
                reference_offsets[label][frame_index][axis] + estimated_offsets[label][frame_index][axis]
                for axis in range(3)
            )
            if label == "person" and frame_index == 2:
                center_delta = (
                    center_delta[0] + 0.2,
                    center_delta[1] + 0.05,
                    center_delta[2],
                )
            estimated_regressed_boxes.append(
                _with_box_fields(
                    base_box,
                    center_delta=center_delta,
                    track_id=estimated_track_id,
                )
            )

        reference_frames.append({"frame_id": frame_id, "boxes": reference_frame_boxes})
        estimated_good_frames.append({"frame_id": frame_id, "boxes": estimated_frame_boxes})
        estimated_regressed_frames.append({"frame_id": frame_id, "boxes": estimated_regressed_boxes})

    return (
        {"frames": reference_frames},
        {"frames": estimated_good_frames},
        {"frames": estimated_regressed_frames},
    )


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=LOCAL_RELLIS_FRAME_ROOT / "object_eval",
        help="Directory to write the generated JSON examples",
    )
    parser.add_argument(
        "--frame",
        default=DEFAULT_FRAME,
        help="RELLIS-3D example frame stem to use (default: %(default)s)",
    )
    parser.add_argument(
        "--example-root",
        type=Path,
        default=None,
        help=(
            "Path to a local RELLIS example root containing "
            "os1_cloud_node_kitti_bin/ and os1_cloud_node_semantickitti_label_id/"
        ),
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_json in output_dir.glob("*.json"):
        stale_json.unlink()

    if args.example_root is not None:
        example_root = args.example_root
        points, labels = _read_rellis_scan(example_root, args.frame)
    elif LOCAL_RELLIS_FRAME_ROOT.exists():
        example_root = LOCAL_RELLIS_FRAME_ROOT
        points, labels = _read_rellis_scan(example_root, args.frame)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            example_root = download_rellis_lidar_example(Path(tmp_dir))
            points, labels = _read_rellis_scan(example_root, args.frame)

    reference_boxes = _cluster_boxes(points, labels)
    detection_reference, detection_good, detection_regressed = _detection_payloads(reference_boxes)
    tracking_reference, tracking_good, tracking_regressed = _tracking_payloads(reference_boxes)

    _write_json(output_dir / "detection_reference.json", detection_reference)
    _write_json(output_dir / "detection_estimated_good.json", detection_good)
    _write_json(output_dir / "detection_estimated_regressed.json", detection_regressed)
    _write_json(output_dir / "tracking_reference.json", tracking_reference)
    _write_json(output_dir / "tracking_estimated_good.json", tracking_good)
    _write_json(output_dir / "tracking_estimated_regressed.json", tracking_regressed)


if __name__ == "__main__":
    main()
