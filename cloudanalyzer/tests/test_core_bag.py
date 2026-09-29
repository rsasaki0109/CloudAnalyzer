"""The Rust core reads ROS 1 bags and MCAP files as rosbags does, without a ROS install."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cloudanalyzer_core")
rosbags = pytest.importorskip("rosbags")

from rosbags.rosbag1 import Writer as Writer1  # noqa: E402
from rosbags.rosbag2 import StoragePlugin, Writer as Writer2  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402

import ca.core.bag_ingest as bag_ingest  # noqa: E402
from ca.core.bag_ingest import imu_ups, inspect_bag, materialize_pointcloud_bag  # noqa: E402

POINTS = np.array(
    [[1.0, 2.0, 3.0, 0.5, 0.0], [np.nan, 0.0, 0.0, 0.1, 0.05], [4.0, 5.0, 6.0, 0.25, 0.1]],
    dtype=np.float32,
)


def _messages(ts, ros1: bool):
    """Three PointCloud2 scans (x y z intensity t, one point not finite) and an Imu each."""
    Header = ts.get_msgdef("std_msgs/msg/Header").cls
    Time = ts.get_msgdef("builtin_interfaces/msg/Time").cls
    PointField = ts.get_msgdef("sensor_msgs/msg/PointField").cls
    PC2 = ts.get_msgdef("sensor_msgs/msg/PointCloud2").cls
    Imu = ts.get_msgdef("sensor_msgs/msg/Imu").cls
    Quaternion = ts.get_msgdef("geometry_msgs/msg/Quaternion").cls
    Vector3 = ts.get_msgdef("geometry_msgs/msg/Vector3").cls
    fields = [PointField(name=n, offset=4 * k, datatype=7, count=1) for k, n in enumerate(["x", "y", "z", "intensity", "t"])]
    out = []
    for k in range(3):
        t = 100.0 + 0.1 * k
        stamp = Time(sec=int(t), nanosec=int(round((t % 1) * 1e9)))
        header = Header(stamp=stamp, frame_id="lidar", **({"seq": k} if ros1 else {}))
        points = POINTS + np.array([0, 0, 0, 0, 0], dtype=np.float32)
        points[:, 1] += k
        cloud = PC2(
            header=header, height=1, width=3, fields=fields, is_bigendian=False, point_step=20, row_step=60,
            data=np.frombuffer(points.tobytes(), dtype=np.uint8), is_dense=False,
        )
        yaw = 0.2 * k
        imu = Imu(
            header=header,
            orientation=Quaternion(x=0.0, y=0.0, z=np.sin(yaw / 2), w=np.cos(yaw / 2)),
            orientation_covariance=np.zeros(9),
            angular_velocity=Vector3(x=0.0, y=0.0, z=0.0),
            angular_velocity_covariance=np.zeros(9),
            linear_acceleration=Vector3(x=0.0, y=0.0, z=9.8),
            linear_acceleration_covariance=np.zeros(9),
        )
        out.append((t, cloud, imu))
    return out


def _ros1_bag(path: Path, compression) -> None:
    ts = get_typestore(Stores.ROS1_NOETIC)
    w = Writer1(path)
    if compression:
        w.set_compression(compression)
    with w:
        points = w.add_connection("/points", "sensor_msgs/msg/PointCloud2", typestore=ts)
        imu = w.add_connection("/imu", "sensor_msgs/msg/Imu", typestore=ts)
        for t, cloud, m in _messages(ts, ros1=True):
            w.write(imu, int((t - 0.01) * 1e9), ts.serialize_ros1(m, "sensor_msgs/msg/Imu"))
            w.write(points, int(t * 1e9), ts.serialize_ros1(cloud, "sensor_msgs/msg/PointCloud2"))


def _mcap_file(folder: Path) -> Path:
    """A rosbag2 folder holding one MCAP file (rosbags writes zstd chunks): that file."""
    ts = get_typestore(Stores.ROS2_HUMBLE)
    with Writer2(folder, version=9, storage_plugin=StoragePlugin.MCAP) as w:
        points = w.add_connection("/points", "sensor_msgs/msg/PointCloud2", typestore=ts)
        imu = w.add_connection("/imu", "sensor_msgs/msg/Imu", typestore=ts)
        for t, cloud, m in _messages(ts, ros1=False):
            w.write(imu, int((t - 0.01) * 1e9), ts.serialize_cdr(m, "sensor_msgs/msg/Imu"))
            w.write(points, int(t * 1e9), ts.serialize_cdr(cloud, "sensor_msgs/msg/PointCloud2"))
    return next(folder.glob("*.mcap"))


def _check(path: Path) -> None:
    import cloudanalyzer_core

    reader = cloudanalyzer_core.BagReader(str(path))
    assert reader.topics() == [("/imu", "sensor_msgs/Imu", 3), ("/points", "sensor_msgs/PointCloud2", 3)]
    start, end = reader.time_range()
    assert start <= 99.99 + 1e-6 and end >= 100.2 - 1e-6
    scans = list(reader.scans())
    assert [s.stamp for s in scans] == pytest.approx([100.0, 100.1, 100.2])
    assert np.allclose(scans[1].positions(), [[1, 3, 3], [4, 6, 6]])
    assert np.allclose(scans[1].intensity(), [0.5, 0.25])
    assert np.allclose(scans[1].time(), [0.0, 1.0])
    stamps, ups, accel = reader.imu()
    assert stamps == pytest.approx([100.0, 100.1, 100.2])
    assert np.allclose(ups, [[0, 0, 1]] * 3) and np.allclose(accel, [[0, 0, 9.8]] * 3)
    # Through the same functions as rosbags: the same files and up directions.
    info = inspect_bag(str(path))
    assert info["message_count"] == 6 and [t["topic"] for t in info["topics"]] == ["/imu", "/points"]
    results = []
    for use_core in (True, False):
        out = path.parent / f"frames_{use_core}"
        with pytest.MonkeyPatch.context() as mp:
            if not use_core:
                mp.setattr(bag_ingest, "core_reader", lambda p: None)
            frames, stamps = materialize_pointcloud_bag(path, out, kitti_bin=True)
            results.append(([np.fromfile(f, dtype=np.float32) for f in frames], stamps, imu_ups(path, stamps)))
    (fa, sa, ua), (fb, sb, ub) = results
    assert sa == sb and all(np.array_equal(a, b) for a, b in zip(fa, fb))
    assert ua.keys() == ub.keys() and all(np.allclose(ua[k], ub[k]) for k in ua)


@pytest.mark.parametrize("compression", [None, Writer1.CompressionFormat.LZ4, Writer1.CompressionFormat.BZ2])
def test_a_ros1_bag_reads_as_rosbags_reads_it(tmp_path, compression):
    path = tmp_path / "drive.bag"
    _ros1_bag(path, compression)
    _check(path)


def test_an_mcap_reads_as_rosbags_reads_it(tmp_path):
    _check(_mcap_file(tmp_path / "drive"))
