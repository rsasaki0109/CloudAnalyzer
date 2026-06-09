"""Extract trajectories from ROS bag / MCAP recordings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ca.experiments.bag_ingest.common import require_rosbags

SUPPORTED_TRAJECTORY_TYPES = frozenset(
    {
        "nav_msgs/msg/Odometry",
        "geometry_msgs/msg/PoseStamped",
    }
)


def _header_timestamp_sec(message: Any) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _position_from_message(message: Any, msgtype: str) -> list[float]:
    if msgtype == "nav_msgs/msg/Odometry":
        position = message.pose.pose.position
    elif msgtype == "geometry_msgs/msg/PoseStamped":
        position = message.pose.position
    else:
        raise ValueError(f"Unsupported trajectory message type: {msgtype}")
    return [float(position.x), float(position.y), float(position.z)]


def _pick_trajectory_connection(connections: list[Any], topic: str | None) -> Any:
    supported = [conn for conn in connections if conn.msgtype in SUPPORTED_TRAJECTORY_TYPES]
    if topic is not None:
        matches = [conn for conn in supported if conn.topic == topic]
        if not matches:
            available = ", ".join(sorted(conn.topic for conn in connections)) or "(none)"
            raise ValueError(f"Trajectory topic not found: {topic}. Available topics: {available}")
        return matches[0]
    if not supported:
        raise ValueError(
            "No supported trajectory topic found in bag. "
            f"Supported types: {', '.join(sorted(SUPPORTED_TRAJECTORY_TYPES))}. "
            "Use --topic to select a topic."
        )
    if len(supported) > 1:
        candidates = ", ".join(sorted(conn.topic for conn in supported))
        raise ValueError(
            "Multiple supported trajectory topics found in bag; use --topic. "
            f"Candidates: {candidates}"
        )
    return supported[0]


def load_trajectory_from_bag(path: str, *, topic: str | None = None) -> dict:
    """Load timestamps and XYZ positions from a ROS bag-like recording."""
    AnyReader = require_rosbags()
    bag_path = Path(path)
    with AnyReader([bag_path]) as reader:
        connection = _pick_trajectory_connection(reader.connections, topic)
        timestamps: list[float] = []
        positions: list[list[float]] = []
        for _bag_timestamp, _connection, raw in reader.messages(connections=[connection]):
            message = reader.deserialize(raw, connection.msgtype)
            timestamps.append(_header_timestamp_sec(message))
            positions.append(_position_from_message(message, connection.msgtype))

    if len(timestamps) < 2:
        raise ValueError("Trajectory bag must contain at least 2 poses on the selected topic")
    timestamp_array = np.asarray(timestamps, dtype=float)
    position_array = np.asarray(positions, dtype=float)
    if np.any(np.diff(timestamp_array) <= 0):
        raise ValueError("Trajectory timestamps must be strictly increasing")

    return {
        "path": path,
        "format": "rosbag",
        "topic": connection.topic,
        "message_type": connection.msgtype,
        "timestamps": timestamp_array,
        "positions": position_array,
        "num_poses": int(timestamp_array.size),
    }
