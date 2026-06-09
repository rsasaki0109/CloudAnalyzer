"""Extract LiDAR scans from ROS bag / MCAP recordings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from ca.experiments.bag_ingest.common import require_rosbags

POINTCLOUD2_TYPE = "sensor_msgs/msg/PointCloud2"

_DATATYPE_TO_DTYPE = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def _header_timestamp_sec(message: Any) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def pointcloud2_to_xyz(message: Any) -> np.ndarray:
    """Decode ``sensor_msgs/msg/PointCloud2`` into an ``(N, 3)`` float64 array."""
    field_map = {field.name: field for field in message.fields}
    for axis in ("x", "y", "z"):
        if axis not in field_map:
            raise ValueError(f"PointCloud2 message is missing '{axis}' field")

    point_count = int(message.height) * int(message.width)
    if point_count == 0:
        return np.empty((0, 3), dtype=np.float64)

    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if raw.size < point_count * int(message.point_step):
        raise ValueError("PointCloud2 data buffer is smaller than width * height * point_step")

    coords = np.empty((point_count, 3), dtype=np.float64)
    point_step = int(message.point_step)
    for axis_index, axis in enumerate(("x", "y", "z")):
        field = field_map[axis]
        dtype = _DATATYPE_TO_DTYPE.get(int(field.datatype))
        if dtype is None:
            raise ValueError(f"Unsupported PointCloud2 datatype for '{axis}': {field.datatype}")
        column = np.ndarray(
            shape=(point_count,),
            dtype=dtype,
            buffer=raw,
            offset=int(field.offset),
            strides=(point_step,),
        )
        coords[:, axis_index] = column.astype(np.float64, copy=False)

    finite = np.isfinite(coords).all(axis=1)
    return coords[finite]


def _pick_pointcloud_connection(connections: list[Any], topic: str | None) -> Any:
    pointclouds = [conn for conn in connections if conn.msgtype == POINTCLOUD2_TYPE]
    if topic is not None:
        matches = [conn for conn in pointclouds if conn.topic == topic]
        if not matches:
            available = ", ".join(sorted(conn.topic for conn in connections)) or "(none)"
            raise ValueError(
                f"PointCloud2 topic not found: {topic}. Available topics: {available}"
            )
        return matches[0]
    if not pointclouds:
        raise ValueError(
            "No sensor_msgs/msg/PointCloud2 topic found in bag. "
            "Use --pointcloud-topic to select a topic."
        )
    if len(pointclouds) > 1:
        candidates = ", ".join(sorted(conn.topic for conn in pointclouds))
        raise ValueError(
            "Multiple PointCloud2 topics found in bag; use --pointcloud-topic. "
            f"Candidates: {candidates}"
        )
    return pointclouds[0]


def materialize_pointcloud_bag(
    path: str | Path,
    output_dir: str | Path,
    *,
    topic: str | None = None,
    max_frames: int | None = None,
) -> tuple[list[Path], tuple[float, ...]]:
    """Write each PointCloud2 message to ``frame_XXXXXX.pcd`` under *output_dir*."""
    AnyReader = require_rosbags()
    bag_path = Path(path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    timestamps: list[float] = []

    with AnyReader([bag_path]) as reader:
        connection = _pick_pointcloud_connection(reader.connections, topic)
        for _connection, _timestamp, rawdata in reader.messages(connections=[connection]):
            if max_frames is not None and len(frame_paths) >= max_frames:
                break
            message = reader.deserialize(rawdata, connection.msgtype)
            points = pointcloud2_to_xyz(message)
            if points.shape[0] == 0:
                continue
            index = len(frame_paths)
            frame_path = out_dir / f"frame_{index:06d}.pcd"
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points)
            if not o3d.io.write_point_cloud(str(frame_path), cloud):
                raise ValueError(f"Failed to write extracted scan: {frame_path}")
            frame_paths.append(frame_path)
            timestamps.append(_header_timestamp_sec(message))

    if not frame_paths:
        raise ValueError(
            f"No non-empty PointCloud2 frames extracted from {bag_path} on topic "
            f"{connection.topic}"
        )

    return frame_paths, tuple(timestamps)
