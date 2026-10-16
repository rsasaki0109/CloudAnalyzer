"""Original scalar point fields and finite-point alignment, not physical clock calibration."""
import struct

import numpy as np
import pytest

core = pytest.importorskip("cloudanalyzer_core")
pytest.importorskip("rosbags")
from rosbags.rosbag1 import Writer as Writer1  # noqa: E402
from rosbags.rosbag2 import StoragePlugin, Writer as Writer2  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402

KIND = "sensor_msgs/msg/PointCloud2"
EPOCH = 1_700_000_000_000_000_123
TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def write_cloud(tmp_path, format, datatype, values, *, field="sensor_clock", empty=False, problem=None, stamp_ns=EPOCH):
    ros1 = format == "ros1"
    types = get_typestore(Stores.ROS1_NOETIC if ros1 else Stores.ROS2_HUMBLE)
    cls = lambda name: types.get_msgdef(name).cls  # noqa: E731
    offset = 16 if field == "t" else 24
    big = datatype % 2 == 0
    order = ">" if big else "<"
    data = bytearray(152)  # Two rows, two 32-byte points per row, 12-byte row padding.
    for i, value in enumerate(values):
        at = (i // 2) * 76 + (i % 2) * 32
        xyz = [np.nan if empty or i == 1 else float(i + 1), 2., 3.]
        struct.pack_into(order + "ffffI", data, at, *xyz, .25 * (i + 1), 111 * i)
        raw = np.asarray(value, dtype=order + TYPES[datatype]).tobytes()
        data[at + offset:at + offset + len(raw)] = raw
    fields = [cls("sensor_msgs/msg/PointField")(name=name, offset=4 * i, datatype=7, count=1)
              for i, name in enumerate(["x", "y", "z", "intensity"])]
    fields.append(cls("sensor_msgs/msg/PointField")(name="t", offset=16,
                                                   datatype=datatype if field == "t" else 6, count=1))
    if field != "t":
        fields.append(cls("sensor_msgs/msg/PointField")(name=field, offset=24, datatype=datatype, count=1))
    selected = fields[-1]
    if problem in ["datatype-zero", "datatype-unsupported"]:
        selected.datatype = 0 if problem == "datatype-zero" else 9
    elif problem in ["count-zero", "count-array"]:
        selected.count = 0 if problem == "count-zero" else 2
    elif problem in ["offset", "offset-max"]:
        selected.offset = 31 if problem == "offset" else 2**32 - 1
    elif problem == "duplicate":
        fields.append(cls("sensor_msgs/msg/PointField")(
            name=selected.name, offset=selected.offset, datatype=selected.datatype, count=selected.count))
    elif problem == "truncated":
        data = data[:-13]  # Remove final row padding plus one byte of its last point.
    sec, ns = divmod(stamp_ns, 10**9)
    header = cls("std_msgs/msg/Header")(
        stamp=cls("builtin_interfaces/msg/Time")(sec=sec, nanosec=ns),
        frame_id="sensor/frame", **({"seq": 0} if ros1 else {}))
    message = cls(KIND)(header=header, height=2, width=2, fields=fields, is_bigendian=big,
                        point_step=32, row_step=63 if problem == "rows" else 76,
                        data=np.frombuffer(data, dtype=np.uint8), is_dense=False)
    path = tmp_path / ("scans.bag" if ros1 else "scans")
    writer = Writer1(path) if ros1 else Writer2(
        path, version=9, storage_plugin=StoragePlugin.SQLITE3 if format == "sqlite" else StoragePlugin.MCAP)
    with writer:
        connection = writer.add_connection("/lidar", KIND, typestore=types)
        raw = types.serialize_ros1(message, KIND) if ros1 else types.serialize_cdr(
            message, KIND, little_endian=format != "cdr-be")
        writer.write(connection, EPOCH + 10**9, raw)
    return core.BagReader(str(path))


@pytest.mark.parametrize("format", ["ros1", "cdr", "cdr-be", "sqlite"])
@pytest.mark.parametrize("datatype", range(1, 9))
def test_all_scalar_types_keep_source_bits_and_row_alignment(tmp_path, format, datatype):
    dtype = np.dtype(TYPES[datatype])
    if datatype <= 6:
        info = np.iinfo(dtype)
        values = [info.min, 17, info.max, -1 if datatype % 2 else info.max - 1]
    elif datatype == 7:
        values = [-0., 17., np.nextafter(np.float32(1.), np.float32(2.)), -3.75]
    else:
        values = [1_700_000_000.0000002, 17., -0., 1e-300]
    reader = write_cloud(tmp_path, format, datatype, values)
    ordinary = next(reader.scans())
    assert ordinary.raw_time() is ordinary.raw_time_field is ordinary.raw_time_datatype is None
    expected = np.asarray(values, dtype=dtype).astype(np.float64)[[0, 2, 3]]
    for iterator in [reader.scans(raw_time_field="sensor_clock"),
                     reader.messages(["/lidar"], raw_time_field="sensor_clock")]:
        scan = next(iterator)
        assert scan.raw_time_field == "sensor_clock" and scan.raw_time_datatype == datatype
        assert scan.raw_time().dtype == np.float64
        np.testing.assert_array_equal(scan.raw_time().view(np.uint64), expected.view(np.uint64))
        assert scan.stamp_ns == ordinary.stamp_ns == EPOCH and scan.frame_id == ordinary.frame_id
        for method in ["positions", "intensity", "time"]:
            np.testing.assert_array_equal(getattr(scan, method)(), getattr(ordinary, method)())
        scan.raw_time()[:] = -123
        np.testing.assert_array_equal(scan.raw_time().view(np.uint64), expected.view(np.uint64))
        for name, value in [("raw_time_field", "other"), ("raw_time_datatype", 0)]:
            with pytest.raises(AttributeError):
                setattr(scan, name, value)
    np.testing.assert_array_equal(ordinary.time(), np.array([0., 2 / 3, 1.], dtype=np.float32))


@pytest.mark.parametrize("format", ["ros1", "cdr", "cdr-be", "sqlite"])
@pytest.mark.parametrize("empty", [False, True])
def test_constant_shared_time_keeps_values_and_empty_scan_metadata(tmp_path, format, empty):
    reader = write_cloud(tmp_path, format, 8, [7.5] * 4, field="t", empty=empty)
    scan = next(reader.scans(raw_time_field="t"))
    assert scan.time() is None
    assert scan.raw_time_field == "t" and scan.raw_time_datatype == 8
    np.testing.assert_array_equal(scan.raw_time(), [7.5] * (0 if empty else 3))
    assert len(scan) == (0 if empty else 3)


@pytest.mark.parametrize("format", ["ros1", "cdr", "cdr-be", "sqlite"])
def test_nonfinite_raw_values_are_preserved_without_changing_normalized_time(tmp_path, format):
    values = [np.nan, 17., np.inf, -np.inf]
    reader = write_cloud(tmp_path, format, 8, values)
    scan = next(reader.scans(raw_time_field="sensor_clock"))
    np.testing.assert_array_equal(scan.raw_time().view(np.uint64), np.asarray(values)[[0, 2, 3]].view(np.uint64))
    np.testing.assert_array_equal(scan.time(), np.array([0., 2 / 3, 1.], dtype=np.float32))


@pytest.mark.parametrize("format", ["ros1", "cdr-be"])
@pytest.mark.parametrize("problem", ["missing", "datatype-zero", "datatype-unsupported", "count-zero", "count-array",
                                    "offset", "offset-max", "duplicate", "truncated", "rows"])
def test_raw_time_request_rejects_malformed_fields_instead_of_filling_zero(tmp_path, format, problem):
    reader = write_cloud(tmp_path, format, 6, [0, 1, 2, 3], problem=problem)
    field = "absent" if problem == "missing" else "sensor_clock"
    with pytest.raises(ValueError, match="raw time field"):
        next(reader.scans(raw_time_field=field))
