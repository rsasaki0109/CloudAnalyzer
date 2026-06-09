"""Tests for ROS bag ingest helpers and `ca info` on bag files."""

from __future__ import annotations

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
from rosbags.rosbag2 import StoragePlugin, Writer  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402
from typer.testing import CliRunner

from ca.experiments.bag_ingest.pointcloud import materialize_pointcloud_bag
from ca.experiments.bag_ingest.trajectory import load_trajectory_from_bag
from ca.core.slam_run import SlamRunRequest
from ca.experiments.slam_run.identity_passthrough import IdentityPassthroughSlamDriver
from cloudanalyzer_cli.main import app


runner = CliRunner()


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


def _write_odometry_bag(path: Path, topic: str = "/odom") -> None:
    import numpy as np

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    Header = typestore.get_msgdef("std_msgs/msg/Header").cls
    Time = typestore.get_msgdef("builtin_interfaces/msg/Time").cls
    Point = typestore.get_msgdef("geometry_msgs/msg/Point").cls
    Vector3 = typestore.get_msgdef("geometry_msgs/msg/Vector3").cls
    Quat = typestore.get_msgdef("geometry_msgs/msg/Quaternion").cls
    Pose = typestore.get_msgdef("geometry_msgs/msg/Pose").cls
    PoseCov = typestore.get_msgdef("geometry_msgs/msg/PoseWithCovariance").cls
    Twist = typestore.get_msgdef("geometry_msgs/msg/Twist").cls
    TwistCov = typestore.get_msgdef("geometry_msgs/msg/TwistWithCovariance").cls
    Odom = typestore.get_msgdef("nav_msgs/msg/Odometry").cls
    zero = Vector3(x=0.0, y=0.0, z=0.0)
    covariance = np.zeros(36, dtype=np.float64)

    def make_odometry(t_sec: float, x: float):
        stamp = Time(sec=int(t_sec), nanosec=int(round((t_sec - int(t_sec)) * 1e9)))
        header = Header(stamp=stamp, frame_id="odom")
        pose = Pose(
            position=Point(x=float(x), y=0.0, z=0.0),
            orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        return Odom(
            header=header,
            child_frame_id="base",
            pose=PoseCov(pose=pose, covariance=covariance),
            twist=TwistCov(twist=Twist(linear=zero, angular=zero), covariance=covariance),
        )

    with Writer(path, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        connection = writer.add_connection(
            topic,
            "nav_msgs/msg/Odometry",
            typestore=typestore,
        )
        for t_sec, x in ((0.0, 0.1), (1.0, 1.1), (2.0, 2.1)):
            payload = typestore.serialize_cdr(
                make_odometry(t_sec, x),
                "nav_msgs/msg/Odometry",
            )
            writer.write(connection, int(t_sec * 1e9), payload)


def _write_tum(path: Path, rows: list[tuple[float, float, float, float]]) -> None:
    path.write_text(
        "\n".join(f"{t:.1f} {x:.1f} {y:.1f} {z:.1f}" for t, x, y, z in rows) + "\n",
        encoding="utf-8",
    )


def _write_pointcloud_bag(path: Path, topic: str = "/points", frames: int = 3) -> None:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    Header = typestore.get_msgdef("std_msgs/msg/Header").cls
    Time = typestore.get_msgdef("builtin_interfaces/msg/Time").cls
    PointField = typestore.get_msgdef("sensor_msgs/msg/PointField").cls
    PC2 = typestore.get_msgdef("sensor_msgs/msg/PointCloud2").cls
    fields = [
        PointField(name="x", offset=0, datatype=7, count=1),
        PointField(name="y", offset=4, datatype=7, count=1),
        PointField(name="z", offset=8, datatype=7, count=1),
    ]

    def make_pointcloud(t_sec: float, point_count: int):
        import numpy as np

        stamp = Time(sec=int(t_sec), nanosec=int(round((t_sec - int(t_sec)) * 1e9)))
        header = Header(stamp=stamp, frame_id="lidar")
        pts = np.array([[float(i), 0.0, 0.0] for i in range(point_count)], dtype=np.float32)
        data = np.frombuffer(pts.tobytes(), dtype=np.uint8)
        return PC2(
            header=header,
            height=1,
            width=point_count,
            fields=fields,
            is_bigendian=False,
            point_step=12,
            row_step=12 * point_count,
            data=data,
            is_dense=True,
        )

    with Writer(path, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        connection = writer.add_connection(
            topic,
            "sensor_msgs/msg/PointCloud2",
            typestore=typestore,
        )
        for index in range(frames):
            t_sec = float(index) * 0.1
            payload = typestore.serialize_cdr(
                make_pointcloud(t_sec, 3),
                "sensor_msgs/msg/PointCloud2",
            )
            writer.write(connection, int(t_sec * 1e9), payload)


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


def test_load_trajectory_from_odometry_bag(tmp_path: Path) -> None:
    bag_path = tmp_path / "odom.mcap"
    _write_odometry_bag(bag_path)

    traj = load_trajectory_from_bag(str(bag_path), topic="/odom")
    assert traj["format"] == "rosbag"
    assert traj["num_poses"] == 3
    assert traj["positions"][1][0] == pytest.approx(1.1)


def test_traj_evaluate_mcap_against_tum(tmp_path: Path) -> None:
    bag_path = tmp_path / "odom.mcap"
    _write_odometry_bag(bag_path)
    reference = tmp_path / "reference.tum"
    _write_tum(
        reference,
        [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
    )

    result = runner.invoke(
        app,
        ["traj-evaluate", str(bag_path), str(reference), "--topic", "/odom", "--format-json"],
    )
    assert result.exit_code == 0, result.output
    import json

    data = json.loads(result.output)
    assert data["matching"]["matched_poses"] == 3
    assert data["ate"]["rmse"] == pytest.approx(0.1)


def test_materialize_pointcloud_bag(tmp_path: Path) -> None:
    bag_path = tmp_path / "points.mcap"
    _write_pointcloud_bag(bag_path)
    frame_dir = tmp_path / "frames"

    frame_paths, timestamps = materialize_pointcloud_bag(
        bag_path,
        frame_dir,
        topic="/points",
    )
    assert len(frame_paths) == 3
    assert timestamps == pytest.approx((0.0, 0.1, 0.2))
    assert frame_paths[0].name == "frame_000000.pcd"


def test_slam_run_from_pointcloud_bag(tmp_path: Path) -> None:
    bag_path = tmp_path / "points.mcap"
    _write_pointcloud_bag(bag_path, frames=2)
    frame_dir = tmp_path / "frames"
    frame_paths, timestamps = materialize_pointcloud_bag(
        bag_path,
        frame_dir,
        topic="/points",
    )

    result = IdentityPassthroughSlamDriver().run(
        SlamRunRequest(
            frame_paths=tuple(frame_paths),
            timestamps_s=timestamps,
        )
    )
    assert result.frames_processed == 2
    assert result.map_points.shape[0] == 6
