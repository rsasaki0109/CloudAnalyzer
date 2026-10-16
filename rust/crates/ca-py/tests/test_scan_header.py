"""Exact PointCloud2 sensor headers through native bag readers; not SLAM accuracy."""

import numpy as np
import pytest

core = pytest.importorskip("cloudanalyzer_core")
pytest.importorskip("rosbags")
from rosbags.highlevel import AnyReader  # noqa: E402
from rosbags.rosbag1 import Writer as Writer1  # noqa: E402
from rosbags.rosbag2 import StoragePlugin, Writer as Writer2  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402

EPOCH = 1_700_000_000_000_000_123
KIND = "sensor_msgs/msg/PointCloud2"


def _write(tmp_path, times, frames, format):
    ros1 = format == "ros1"
    ts = get_typestore(Stores.ROS1_NOETIC if ros1 else Stores.ROS2_HUMBLE)
    cls = lambda name: ts.get_msgdef(name).cls  # noqa: E731
    fields = [cls("sensor_msgs/msg/PointField")(name=name, offset=4 * i, datatype=7, count=1)
              for i, name in enumerate(["x", "y", "z", "intensity", "t"])]
    messages = []
    for i, (stamp_ns, frame) in enumerate(zip(times, frames)):
        sec, nsec = divmod(stamp_ns, 10**9)
        # rosbags normalizes Time to ROS 2's signed type, even for ROS 1.
        # Serialize an in-range placeholder, then restore unsigned ROS 1 wire bits.
        stamp = cls("builtin_interfaces/msg/Time")(sec=min(sec, 2**31 - 1) if ros1 else sec, nanosec=nsec)
        header = cls("std_msgs/msg/Header")(stamp=stamp, frame_id=frame, **({"seq": i} if ros1 else {}))
        points = np.array([[1., 2., 3., .5, 0.], [np.nan, 0., 0., .1, .05],
                           [4., 5., 6., .25, .1]], dtype="<f4")
        if i == 1:
            points[:, :3] = np.nan  # Even an empty decoded cloud retains its header.
        messages.append(cls(KIND)(header=header, height=1, width=3, fields=fields,
                                  is_bigendian=False, point_step=20, row_step=60,
                                  data=np.frombuffer(points.tobytes(), dtype=np.uint8), is_dense=False))
    if ros1:
        path = tmp_path / "scans.bag"
        with Writer1(path) as writer:
            connection = writer.add_connection("/lidar", KIND, typestore=ts)
            for i, message in enumerate(messages):
                payload = bytearray(ts.serialize_ros1(message, KIND))
                # PointCloud2 starts with Header: uint32 sequence, uint32 seconds.
                payload[4:8] = (times[i] // 10**9).to_bytes(4, "little", signed=False)
                writer.write(connection, EPOCH + 10**9 + i * 10**6, payload)
    else:
        path = tmp_path / "scans"
        storage = StoragePlugin.SQLITE3 if format == "sqlite" else StoragePlugin.MCAP
        with Writer2(path, version=9, storage_plugin=storage) as writer:
            connection = writer.add_connection("/lidar", KIND, typestore=ts)
            for i, message in enumerate(messages):
                writer.write(connection, EPOCH + 10**9 + i * 10**6,
                             ts.serialize_cdr(message, KIND, little_endian=format != "cdr-be"))
    return path


@pytest.mark.parametrize("format", ["ros1", "cdr", "cdr-be", "sqlite"])
def test_scan_headers_match_rosbags_without_float_roundtrip_or_recording_time(format, tmp_path):
    times = [EPOCH + 2, EPOCH, EPOCH + 1]
    frames = ["lidar/body", "", "lidar/基準"]
    path = _write(tmp_path, times, frames, format)
    reader = core.BagReader(str(path))
    scans = list(reader.scans())
    assert [s.stamp_ns for s in scans] == times  # Preserve recording order, not header-time sort.
    assert [s.frame_id for s in scans] == frames
    assert all(type(s.stamp_ns) is int for s in scans)
    # All three legacy floats collide here, while exact nanoseconds remain distinct.
    assert len(set(s.stamp for s in scans)) == 1
    assert len(set(s.stamp_ns for s in scans)) == 3
    with AnyReader([path]) as reference:
        for scan, (connection, recording_ns, raw) in zip(scans, reference.messages(), strict=True):
            message = reference.deserialize(raw, connection.msgtype)
            expected = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
            assert scan.stamp_ns == expected and scan.stamp_ns != recording_ns
            assert scan.frame_id == message.header.frame_id
            assert scan.stamp == message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
    assert len(scans[1]) == 0 and scans[1].time() is None
    np.testing.assert_array_equal(scans[0].positions(), [[1., 2., 3.], [4., 5., 6.]])
    np.testing.assert_array_equal(scans[0].intensity(), [.5, .25])
    np.testing.assert_array_equal(scans[0].time(), [0., 1.])
    # Generic message iteration and direct bag-file readers retain the same fields.
    for alternative in [path, *(path.glob("*.mcap") if path.is_dir() else []),
                         *(path.glob("*.db3") if path.is_dir() else [])]:
        decoded = list(core.BagReader(str(alternative)).messages(["/lidar"]))
        assert [(s.stamp_ns, s.frame_id) for s in decoded] == list(zip(times, frames))
    for name, value in [("stamp_ns", 0), ("frame_id", "other")]:
        with pytest.raises(AttributeError):
            setattr(scans[0], name, value)


@pytest.mark.parametrize("format", ["ros1", "cdr", "cdr-be", "sqlite"])
def test_scan_header_time_boundaries(format, tmp_path):
    if format == "ros1":
        times = [0, 1, 999_999_999, (2**32 - 1) * 10**9 + 999_999_999]
    else:
        times = [-(2**31) * 10**9, -1, 0, 1, (2**31 - 1) * 10**9 + 999_999_999]
    path = _write(tmp_path, times, ["sensor"] * len(times), format)
    scans = list(core.BagReader(str(path)).scans())
    assert [scan.stamp_ns for scan in scans] == times
    for scan, t in zip(scans, times):
        sec, nsec = divmod(t, 10**9)
        assert scan.stamp == sec + nsec * 1e-9
