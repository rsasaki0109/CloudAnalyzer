"""Exact-rational timestamp oracle; no physical unit/origin inference."""
from fractions import Fraction

import numpy as np
import pytest

pytest.importorskip("cloudanalyzer_core")
pytest.importorskip("rosbags")
from test_raw_point_time import EPOCH, TYPES, write_cloud  # noqa: E402

SCALES = {"s": 10**9, "ms": 10**6, "us": 1000, "ns": 1}
FORMATS = ["ros1", "cdr", "cdr-be", "sqlite"]


def expected_times(scan, unit, origin):
    header = scan.stamp_ns if origin == "header" else 0
    # Fraction.from_float uses the exact represented value. round(Fraction)
    # independently implements nearest integer with ties to even.
    return np.asarray([round(Fraction.from_float(float(value)) * SCALES[unit]) + header
                       for value in scan.raw_time()], dtype=np.int64)


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("datatype", range(1, 9))
def test_units_origins_and_source_types_match_exact_rational_oracle(tmp_path, format, datatype):
    if datatype <= 6:
        info = np.iinfo(np.dtype(TYPES[datatype]))
        values = [info.max, 17, info.min, 1]
    elif datatype == 7:
        values = [1.5, 17, -2.5, np.nextafter(np.float32(.5), np.float32(0.))]
    else:
        values = [1_700_000_000.0000002, 17, 5e-10, -5e-10]
    reader = write_cloud(tmp_path, format, datatype, values)
    scan = next(reader.scans(raw_time_field="sensor_clock"))
    raw_bits = scan.raw_time().view(np.uint64).copy()
    xyz = scan.positions().copy()
    normalized = scan.time().copy()
    for unit in SCALES:
        for origin in ["header", "absolute"]:
            expected = expected_times(scan, unit, origin)
            actual = scan.point_times_ns(unit=unit, origin=origin)
            assert actual.dtype == np.int64 and actual.shape == (3,)
            np.testing.assert_array_equal(actual, expected)
            actual[:] = -123
            np.testing.assert_array_equal(scan.point_times_ns(unit=unit, origin=origin), expected)
    np.testing.assert_array_equal(scan.raw_time().view(np.uint64), raw_bits)
    np.testing.assert_array_equal(scan.positions(), xyz)
    np.testing.assert_array_equal(scan.time(), normalized)
    assert scan.stamp_ns == EPOCH


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("values", [
    [.5, 17., 1.5, -2.5],
    [np.nextafter(.5, 0.), 17., np.nextafter(.5, 1.), np.nextafter(-.5, -1.)],
    [np.nextafter(0., 1.), 17., -np.nextafter(0., 1.), -0.],
])
def test_halfway_neighbors_and_subnormals_round_before_odd_header_addition(tmp_path, format, values):
    scan = next(write_cloud(tmp_path, format, 8, values).scans(raw_time_field="sensor_clock"))
    for unit in SCALES:
        for origin in ["header", "absolute"]:
            np.testing.assert_array_equal(scan.point_times_ns(unit=unit, origin=origin),
                                          expected_times(scan, unit, origin))


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, np.finfo(np.float64).max,
                                 -np.finfo(np.float64).max, float(2**63)])
def test_nonfinite_and_final_overflow_fail_without_mutating_source(tmp_path, format, bad):
    scan = next(write_cloud(tmp_path, format, 8, [0., 17., bad, 1.]).scans(raw_time_field="sensor_clock"))
    before = scan.raw_time().view(np.uint64).copy()
    match = r"index 1.*not finite" if not np.isfinite(bad) else r"index 1.*outside signed int64"
    with pytest.raises(ValueError, match=match):
        scan.point_times_ns(unit="ns", origin="absolute")
    np.testing.assert_array_equal(scan.raw_time().view(np.uint64), before)


@pytest.mark.parametrize("format", ["cdr", "cdr-be", "sqlite"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_range_is_checked_after_origin_addition_including_negative_headers(tmp_path, format, sign):
    header = -sign * EPOCH
    value = sign * float(2**63)
    scan = next(write_cloud(tmp_path, format, 8, [value, 17., value, 0.], stamp_ns=header).scans(
        raw_time_field="sensor_clock"))
    np.testing.assert_array_equal(scan.point_times_ns(unit="ns", origin="header"),
                                  expected_times(scan, "ns", "header"))
    assert scan.stamp_ns == header


@pytest.mark.parametrize("format", FORMATS)
def test_empty_retained_source_returns_empty_int64_array(tmp_path, format):
    scan = next(write_cloud(tmp_path, format, 8, [np.nan] * 4, empty=True).scans(raw_time_field="sensor_clock"))
    result = scan.point_times_ns(unit="ns", origin="header")
    assert result.dtype == np.int64 and result.shape == (0,)


def test_explicit_policy_and_retained_field_are_required(tmp_path):
    reader = write_cloud(tmp_path, "ros1", 6, [0, 1, 2, 3])
    scan = next(reader.scans(raw_time_field="sensor_clock"))
    for kwargs in [{}, {"unit": "ns"}, {"origin": "header"}]:
        with pytest.raises(TypeError):
            scan.point_times_ns(**kwargs)
    with pytest.raises(TypeError):
        scan.point_times_ns("ns", "header")
    for unit in ["", "seconds", "NS"]:
        with pytest.raises(ValueError, match="point time unit"):
            scan.point_times_ns(unit=unit, origin="header")
    for origin in ["", "auto", "timebase"]:
        with pytest.raises(ValueError, match="point time origin"):
            scan.point_times_ns(unit="ns", origin=origin)
    with pytest.raises(ValueError, match="explicitly retained raw_time_field"):
        next(reader.scans()).point_times_ns(unit="ns", origin="header")
