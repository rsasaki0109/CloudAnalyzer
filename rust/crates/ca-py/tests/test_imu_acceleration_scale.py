"""Explicit input units, SI bias/noise and raw-data preservation; no trajectory score."""
import numpy as np
import pytest

core = pytest.importorskip("cloudanalyzer_core")
from test_imu_preintegration import EPOCH, _write  # noqa: E402


def assert_measurement_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.mark.parametrize("format", ["ros1", "cdr", "cdr-be", "sqlite"])
@pytest.mark.parametrize("known_covariance", [False, True])
def test_scale_matches_si_recording_and_keeps_raw_fields(tmp_path, format, known_covariance):
    times = EPOCH + np.arange(31, dtype=np.int64) * 10_000_000
    k = np.arange(len(times))
    gyro = np.column_stack([.1 * np.sin(k), .2 * np.cos(k), .3 + .01 * k])
    accel = np.column_stack([.01 * k, .02 * np.cos(k), 1. + .01 * np.sin(k)])
    cov = np.zeros((len(times), 3, 9))
    if known_covariance:
        cov[:] = np.tile(np.diag([.001, .002, .003]).ravel(), (3, 1))
    scale = 9.80665
    reader = _write(tmp_path, times, gyro, accel, format=format, covariances=cov)
    si_dir = tmp_path / "si"
    si_dir.mkdir()
    si_cov = cov.copy()
    si_cov[:, 2] *= scale**2
    si = _write(si_dir, times, gyro, accel * scale, format=format, covariances=si_cov)
    raw = reader.imu_raw()
    series = reader.imu_series("/imu", acceleration_scale=scale)
    assert series.acceleration_scale == scale
    with pytest.raises(AttributeError):
        series.acceleration_scale = 1.
    np.testing.assert_array_equal(series.timestamps_ns(), times)
    assert series.frame_ids() == ["imu/body"] * len(times)
    kwargs = dict(gyro_noise_density=.02, accel_noise_density=.15, bias=(.01, -.02, .03, .1, -.05, .02))
    for start, end in [(int(times[0]), int(times[-1])), (int(times[3]) + 17, int(times[23]) - 59)]:
        expected = si.preintegrate_imu(start, end, **kwargs)
        for actual in [series.preintegrate(start, end, **kwargs), series.factor(start, end, **kwargs).measurement(),
                       reader.preintegrate_imu(start, end, acceleration_scale=scale, **kwargs),
                       reader.imu_factor(start, end, acceleration_scale=scale, **kwargs).measurement()]:
            assert_measurement_equal(actual, expected)
        # The explicit identity scale must preserve every legacy output field.
        legacy = reader.preintegrate_imu(start, end, **kwargs)
        for actual in [reader.preintegrate_imu(start, end, acceleration_scale=1., **kwargs),
                       reader.imu_series(acceleration_scale=1.).preintegrate(start, end, **kwargs)]:
            assert_measurement_equal(actual, legacy)
    after = reader.imu_raw()
    for key in raw:
        np.testing.assert_array_equal(after[key], raw[key], err_msg=key)


@pytest.mark.parametrize("scale", [0., -1., np.nan, np.inf, -np.inf])
def test_invalid_scale_is_rejected_by_all_loading_routes(tmp_path, scale):
    times = EPOCH + np.arange(3, dtype=np.int64) * 10_000_000
    reader = _write(tmp_path, times, np.zeros((3, 3)), np.ones((3, 3)))
    with pytest.raises(ValueError, match="acceleration_scale.*finite and positive"):
        reader.imu_series(acceleration_scale=scale)
    for method in [reader.preintegrate_imu, reader.imu_factor]:
        with pytest.raises(ValueError, match="acceleration_scale.*finite and positive"):
            method(int(times[0]), int(times[-1]), gyro_noise_density=.02,
                   accel_noise_density=.15, acceleration_scale=scale)


@pytest.mark.parametrize("sentinel", [-1., -2.])
def test_scaling_retains_missing_invalid_and_unused_endpoint_semantics(tmp_path, sentinel):
    times = EPOCH + np.arange(4, dtype=np.int64) * 10_000_000
    cov = np.zeros((4, 3, 9))
    cov[0, 2, 0] = sentinel
    cov[-1, 1:, 0] = -1.
    reader = _write(tmp_path, times, np.zeros((4, 3)), np.ones((4, 3)), covariances=cov)
    series = reader.imu_series(acceleration_scale=.001)
    kwargs = dict(gyro_noise_density=.02, accel_noise_density=.15)
    for method in [series.preintegrate, series.factor]:
        with pytest.raises(ValueError, match="missing/invalid"):
            method(int(times[0]), int(times[1]), **kwargs)
    expected = reader.preintegrate_imu(int(times[1]), int(times[-1]), acceleration_scale=.001, **kwargs)
    assert_measurement_equal(series.preintegrate(int(times[1]), int(times[-1]), **kwargs), expected)
    assert_measurement_equal(series.factor(int(times[1]), int(times[-1]), **kwargs).measurement(), expected)
    assert reader.imu_raw()["linear_acceleration_covariance"][0, 0, 0] == sentinel


def test_finite_scale_with_overflowing_acceleration_fails_integration(tmp_path):
    times = EPOCH + np.arange(3, dtype=np.int64) * 10_000_000
    reader = _write(tmp_path, times, np.zeros((3, 3)), np.full((3, 3), 9.8))
    series = reader.imu_series(acceleration_scale=1e308)
    with pytest.raises(ValueError, match="nonfinite"):
        series.preintegrate(int(times[0]), int(times[-1]), gyro_noise_density=.02, accel_noise_density=.15)
