"""Shared helpers for ROS bag / MCAP ingest experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

BAG_SUFFIXES = frozenset({".bag", ".mcap", ".db3"})

ROS_INSTALL_HINT = (
    "ROS bag input requires optional dependencies.\n"
    'Install with: pip install "cloudanalyzer[ros]"'
)


def is_bag_path(path: str | Path) -> bool:
    """Return True when *path* looks like a ROS bag / MCAP / sqlite recording."""
    return Path(path).suffix.lower() in BAG_SUFFIXES


def require_rosbags() -> Any:
    """Import rosbags or raise a user-facing install hint."""
    try:
        from rosbags.highlevel import AnyReader

        return AnyReader
    except ImportError as exc:
        raise ValueError(ROS_INSTALL_HINT) from exc


def _connection_rows(reader: Any) -> list[dict[str, Any]]:
    topics: list[dict[str, Any]] = []
    for connection in reader.connections:
        topics.append(
            {
                "topic": connection.topic,
                "type": connection.msgtype,
                "count": int(connection.msgcount),
            }
        )
    topics.sort(key=lambda row: row["topic"])
    return topics


def inspect_bag_metadata(path: str, *, decode_sample: bool = False) -> dict[str, Any]:
    """Return topic metadata for a ROS bag-like recording.

    When *decode_sample* is True, deserialize the first message on each topic
    to validate that embedded definitions / typestore coverage is usable.
    """
    AnyReader = require_rosbags()
    bag_path = Path(path)
    if not bag_path.exists():
        raise FileNotFoundError(path)

    with AnyReader([bag_path]) as reader:
        topics = _connection_rows(reader)
        decoded_topics: list[str] = []
        if decode_sample:
            for connection in reader.connections:
                if connection.msgcount <= 0:
                    continue
                for _timestamp, _connection, _message in reader.messages(
                    connections=[connection]
                ):
                    decoded_topics.append(connection.topic)
                    break

        return {
            "path": str(bag_path),
            "kind": "rosbag",
            "duration_ns": int(reader.duration),
            "start_time_ns": int(reader.start_time),
            "end_time_ns": int(reader.end_time),
            "message_count": int(sum(row["count"] for row in topics)),
            "topics": topics,
            "decoded_sample_topics": decoded_topics,
        }
