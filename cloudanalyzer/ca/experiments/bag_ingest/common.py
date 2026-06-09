"""Shared helpers for ROS bag / MCAP ingest experiments."""

from __future__ import annotations

from ca.core.bag_ingest import (
    BAG_SUFFIXES,
    ROS_INSTALL_HINT,
    inspect_bag,
    is_bag_path,
    require_rosbags,
)

inspect_bag_metadata = inspect_bag

__all__ = [
    "BAG_SUFFIXES",
    "ROS_INSTALL_HINT",
    "inspect_bag_metadata",
    "is_bag_path",
    "require_rosbags",
]
