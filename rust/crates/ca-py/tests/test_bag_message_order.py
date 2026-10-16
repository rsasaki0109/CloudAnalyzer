"""Recording-time ties retain file order; sensor-header corruption stays visible."""
import struct

import numpy as np
import pytest

core = pytest.importorskip("cloudanalyzer_core")
pytest.importorskip("rosbags")
from rosbags.rosbag1 import Writer as Writer1  # noqa: E402
from rosbags.rosbag2 import StoragePlugin, Writer as Writer2  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402

EPOCH = 1_700_000_000_000_000_123
KIND = "sensor_msgs/msg/Imu"


def write_messages(tmp_path, format, records):
    ros1 = format.startswith("ros1")
    types = get_typestore(Stores.ROS1_NOETIC if ros1 else Stores.ROS2_HUMBLE)
    cls = lambda name: types.get_msgdef(name).cls  # noqa: E731
    path = tmp_path / ("messages.bag" if ros1 else "messages")
    writer = Writer1(path) if ros1 else Writer2(
        path, version=9, storage_plugin=StoragePlugin.SQLITE3 if format == "sqlite" else StoragePlugin.MCAP)
    if ros1 and format != "ros1-none":
        writer.set_compression(Writer1.CompressionFormat.LZ4 if format == "ros1-lz4" else Writer1.CompressionFormat.BZ2)
    with writer:
        connections = {topic: writer.add_connection(topic, KIND, typestore=types) for topic in ["/imu", "/other"]}
        for index, (topic, recording_ns, header_ms) in enumerate(records):
            sec, ns = divmod(EPOCH + header_ms * 1_000_000, 10**9)
            header = cls("std_msgs/msg/Header")(
                stamp=cls("builtin_interfaces/msg/Time")(sec=sec, nanosec=ns), frame_id="imu/body",
                **({"seq": index} if ros1 else {}))
            message = cls(KIND)(
                header=header, orientation=cls("geometry_msgs/msg/Quaternion")(x=0., y=0., z=0., w=1.),
                orientation_covariance=np.zeros(9),
                angular_velocity=cls("geometry_msgs/msg/Vector3")(x=float(header_ms), y=.2, z=.3),
                angular_velocity_covariance=np.zeros(9),
                linear_acceleration=cls("geometry_msgs/msg/Vector3")(x=1., y=2., z=9.81),
                linear_acceleration_covariance=np.zeros(9))
            payload = types.serialize_ros1(message, KIND) if ros1 else types.serialize_cdr(
                message, KIND, little_endian=format != "mcap-be")
            writer.write(connections[topic], recording_ns, payload)
    if format.startswith("mcap"):
        # Ensure this exercises Cursor's indexed chunk queue, not linear MCAP.
        data = next(path.glob("*.mcap")).read_bytes()
        assert data[-37] == 2
        position = struct.unpack_from("<Q", data, len(data) - 28)[0]
        assert position > 0
        opcodes = []
        while position < len(data) - 37:
            opcode, size = struct.unpack_from("<BQ", data, position)
            opcodes.append(opcode)
            position += 9 + size
        assert 8 in opcodes
    return core.BagReader(str(path)), types, ros1


@pytest.mark.parametrize("format", ["ros1-none", "ros1-lz4", "ros1-bz2", "mcap-le", "mcap-be", "sqlite"])
def test_recording_time_ties_keep_physical_order_across_topics_and_snapshots(tmp_path, format):
    records = [("/imu", EPOCH + 2 * 10**9, 4), ("/other", EPOCH + 10**9, 90),
               ("/imu", EPOCH + 10**9, 0), ("/other", EPOCH + 10**9, 91),
               ("/imu", EPOCH + 10**9, 1), ("/imu", EPOCH + 10**9, 2), ("/imu", EPOCH + 3 * 10**9, 5)]
    if format == "sqlite":
        # SQLite streams row IDs; use chronological inserts to isolate tie stability.
        records = [records[i] for i in [1, 2, 3, 4, 5, 0, 6]]
    reader, types, ros1 = write_messages(tmp_path, format, records)
    for topics, expected in [(["/imu", "/other"], [90, 0, 91, 1, 2, 4, 5]),
                             (["/imu"], [0, 1, 2, 4, 5]), (["/other"], [90, 91])]:
        messages = list(reader.messages(topics))
        stamps = [stamp for topic, stamp, encoding, raw in messages]
        assert stamps == sorted(stamps)
        decoded = [(types.deserialize_ros1(raw, KIND) if ros1 else types.deserialize_cdr(raw, KIND))
                   for topic, stamp, encoding, raw in messages]
        assert [m.angular_velocity.x for m in decoded] == expected
    expected = EPOCH + np.array([0, 1, 2, 4, 5], dtype=np.int64) * 1_000_000
    np.testing.assert_array_equal(reader.imu_raw("/imu")["timestamps_ns"], expected)
    np.testing.assert_array_equal(reader.imu_series("/imu").timestamps_ns(), expected)


@pytest.mark.parametrize("format", ["ros1-none", "mcap-le"])
@pytest.mark.parametrize("headers", [[0, 2, 1, 3], [0, 1, 1, 3]])
def test_equal_recording_times_do_not_hide_bad_sensor_header_order(tmp_path, format, headers):
    reader, types, ros1 = write_messages(tmp_path, format, [("/imu", EPOCH, value) for value in headers])
    np.testing.assert_array_equal(reader.imu_raw("/imu")["timestamps_ns"],
                                  EPOCH + np.array(headers, dtype=np.int64) * 1_000_000)
    with pytest.raises(ValueError, match="increase"):
        reader.imu_series("/imu")
