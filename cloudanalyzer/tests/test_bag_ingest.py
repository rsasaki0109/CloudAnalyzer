"""Tests for ROS bag ingest helpers and `ca info` on bag files."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ca.experiments.bag_ingest.common import (
    ROS_INSTALL_HINT,
    inspect_bag_metadata,
    is_bag_path,
)
from ca.info import get_info


def test_is_bag_path() -> None:
    assert is_bag_path("run.mcap")
    assert is_bag_path("/data/recording.db3")
    assert is_bag_path("legacy.bag")
    assert not is_bag_path("map.pcd")


def test_get_info_missing_bag_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        get_info("/tmp/does-not-exist.mcap")


def test_get_info_without_ros_extra_shows_hint() -> None:
    with patch("ca.experiments.bag_ingest.common.require_rosbags", side_effect=ValueError(ROS_INSTALL_HINT)):
        with pytest.raises(ValueError, match="cloudanalyzer\\[ros\\]"):
            get_info("/tmp/run.mcap")


rosbags = pytest.importorskip("rosbags")
from rosbags.highlevel import AnyReader  # noqa: E402
from rosbags.rosbag2 import StoragePlugin, Writer  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402


def _write_string_bag(path: Path) -> None:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with Writer(path, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        connection = writer.add_connection(
            "/chatter",
            "std_msgs/msg/String",
            typestore=typestore,
        )
        msgcls = typestore.get_msgdef("std_msgs/msg/String").cls
        payload = typestore.serialize_cdr(msgcls(data="hello"), "std_msgs/msg/String")
        writer.write(connection, 1_000_000_000, payload)
        writer.write(connection, 2_000_000_000, payload)


def test_inspect_bag_metadata_on_mcap(tmp_path: Path) -> None:
    bag_path = tmp_path / "sample.mcap"
    _write_string_bag(bag_path)

    info = inspect_bag_metadata(str(bag_path))
    assert info["kind"] == "rosbag"
    assert info["message_count"] == 2
    assert info["topics"] == [
        {"topic": "/chatter", "type": "std_msgs/msg/String", "count": 2},
    ]


def test_get_info_on_mcap(tmp_path: Path) -> None:
    bag_path = tmp_path / "sample.mcap"
    _write_string_bag(bag_path)

    info = get_info(str(bag_path))
    assert info["kind"] == "rosbag"
    assert info["topics"][0]["topic"] == "/chatter"


def test_extract_all_decodes_sample_message(tmp_path: Path) -> None:
    bag_path = tmp_path / "sample.mcap"
    _write_string_bag(bag_path)

    info = inspect_bag_metadata(str(bag_path), decode_sample=True)
    assert info["decoded_sample_topics"] == ["/chatter"]
