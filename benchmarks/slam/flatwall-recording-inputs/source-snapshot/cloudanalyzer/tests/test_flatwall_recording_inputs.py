"""Boundary tests for exact recording metadata, never synthetic SLAM scores."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("flatwall_inputs", ROOT / "scripts/audit_flatwall_recording_inputs.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_ros_stamp_never_uses_float_epoch_seconds():
    assert audit.stamp_ns(NS(sec=1_656_505_431, nanosec=199_620_008)) == 1_656_505_431_199_620_008
    for sec, ns in [(-1, 0), (1, -1), (1, 10 ** 9), (10 ** 10, 0)]:
        with pytest.raises(ValueError, match="timestamp"):
            audit.stamp_ns(NS(sec=sec, nanosec=ns))


@pytest.mark.parametrize("values", [np.array([], dtype=np.int64), np.array([1., 2.]),
    np.array([1, 1], dtype=np.int64), np.array([2, 1], dtype=np.int64), np.array([-1, 0], dtype=np.int64)])
def test_invalid_imu_headers_are_rejected(values):
    with pytest.raises(ValueError, match="int64"):
        audit.times(values)


def test_coverage_checks_internal_gaps_including_partial_intervals():
    t = np.array([100, 150, 50_000_150, 100_000_151], dtype=np.int64)
    assert audit.coverage(t, 100, 50_000_150)["status"] == "covered"
    assert audit.coverage(t, 99, 100)["status"] == "outside_imu_range"
    assert audit.coverage(t, 100_000_151, 100_000_152)["status"] == "outside_imu_range"
    assert audit.coverage(t, 50_000_150, 50_000_151)["status"] == "imu_gap"
    assert audit.coverage(t, 50_000_151, 50_000_151)["status"] == "imu_gap"
    assert audit.coverage(t, 100_000_151, 100_000_151)["maximum_gap_ns"] == 0
    assert audit.coverage(t, 50_000_150, 50_000_150)["status"] == "covered"
    with pytest.raises(ValueError, match="interval"):
        audit.coverage(t, 101, 100)


def cloud(big=False):
    order = ">" if big else "<"
    dtype = np.dtype([(n, order + kind) for n, (_, kind) in audit.FIELDS.items()])
    data = np.zeros((2, 2), dtype=dtype)
    data["x"] = [[1, 2], [3, 4]]
    data["y"] = -1
    data["t"] = [[0, 20], [10, 30]]
    data["intensity"] = 42
    data["tag"], data["line"] = 16, 2
    # Eight padding bytes after each row must not appear as points.
    payload = b"".join(row.tobytes() + b"padding!" for row in data)
    fields = [NS(name=n, datatype=dt, count=1, offset=dtype.fields[n][1]) for n, (dt, _) in audit.FIELDS.items()]
    return NS(fields=fields, height=2, width=2, point_step=dtype.itemsize,
              row_step=dtype.itemsize * 2 + 8, data=payload, is_bigendian=big)


@pytest.mark.parametrize("big", [False, True])
def test_row_padding_byte_order_and_original_point_order_are_retained(big):
    columns = audit.point_columns(cloud(big))
    arrays = audit.cloud_arrays(columns)
    np.testing.assert_array_equal(arrays["positions"][:, 0], [1, 2, 3, 4])
    np.testing.assert_array_equal(arrays["offsets_ns"], [0, 20, 10, 30])
    assert arrays["positions"].dtype == np.dtype("float64")
    assert audit.cloud_digest(arrays) == audit.cloud_digest(audit.cloud_arrays(audit.point_columns(cloud())))


@pytest.mark.parametrize("change", ["duplicate", "datatype", "count", "overlap", "negative_offset", "past_end", "row", "length"])
def test_malformed_point_layout_is_rejected(change):
    m = cloud()
    if change == "duplicate":
        m.fields.append(m.fields[0])
    elif change == "datatype":
        m.fields[3].datatype = 7
    elif change == "count":
        m.fields[3].count = 2
    elif change == "overlap":
        m.fields[3].offset = 0
    elif change == "negative_offset":
        m.fields[3].offset = -1
    elif change == "past_end":
        m.fields[3].offset = m.point_step
    elif change == "row":
        m.row_step = 1
    else:
        m.data = m.data[:-1]
    with pytest.raises(ValueError, match="point"):
        audit.point_columns(m)


def test_custom_parity_rejects_a_single_nanosecond_or_point_reorder():
    arrays = audit.cloud_arrays(audit.point_columns(cloud()))
    points = [NS(x=float(p[0]), y=float(p[1]), z=float(p[2]), offset_time=int(t), reflectivity=42, tag=16, line=2)
              for p, t in zip(arrays["positions"], arrays["offsets_ns"])]
    message = NS(point_num=4, points=points)
    audit.compare_custom(message, arrays)
    points[2].offset_time += 1
    with pytest.raises(ValueError, match="offsets_ns"):
        audit.compare_custom(message, arrays)
    points[2].offset_time -= 1
    points[0], points[1] = points[1], points[0]
    with pytest.raises(ValueError, match="positions"):
        audit.compare_custom(message, arrays)
    message.point_num = 3
    with pytest.raises(ValueError, match="count"):
        audit.compare_custom(message, arrays)
