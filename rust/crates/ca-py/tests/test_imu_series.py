"""Snapshot I/O, interval parity and validation; not trajectory accuracy."""
import shutil

import numpy as np
import pytest

core = pytest.importorskip("cloudanalyzer_core")
from test_imu_preintegration import EPOCH, _write  # noqa: E402


@pytest.mark.parametrize("format", ["ros1", "cdr", "cdr-be", "sqlite"])
def test_series_is_independent_of_recording_and_returns_identical_factors(tmp_path, format):
    times = EPOCH + np.arange(41, dtype=np.int64) * 10_000_000
    gyro = np.tile([.2, -.1, .3], (len(times), 1))
    accel = np.tile([.5, -1., 9.81], (len(times), 1))
    reader = _write(tmp_path, times, gyro, accel, format=format)
    series = reader.imu_series("/imu")
    assert isinstance(series, core.ImuSeries) and len(series) == len(times)
    np.testing.assert_array_equal(series.timestamps_ns(), times)
    assert series.frame_ids() == ["imu/body"] * len(times)
    kwargs = dict(gyro_noise_density=.02, accel_noise_density=.15, bias=(.01, -.02, .03, -.1, .05, .02))
    intervals = [(int(times[0]), int(times[5])), (int(times[4]) + 17, int(times[30]) - 59),
                 (int(times[-3]), int(times[-1]))]
    expected = [reader.imu_factor(a, b, **kwargs).measurement() for a, b in intervals]
    # The snapshot must neither reopen the recording nor depend on its lifetime.
    for path in [tmp_path / "imu.bag", tmp_path / "imu"]:
        assert path.resolve().is_relative_to(tmp_path.resolve())
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    series.timestamps_ns()[:] = -1
    series.frame_ids()[0] = "changed"
    for (a, b), reference in zip(intervals, expected):
        for actual in [series.preintegrate(a, b, **kwargs), series.factor(a, b, **kwargs).measurement()]:
            assert actual.keys() == reference.keys()
            for key in reference:
                np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)
    np.testing.assert_array_equal(series.timestamps_ns(), times)
    assert series.frame_ids()[0] == "imu/body"
    for a, b in [(int(times[0]) - 1, int(times[2])), (int(times[-2]), int(times[-1]) + 1), (1, 1), (2, 1)]:
        with pytest.raises(ValueError):
            series.factor(a, b, **kwargs)
    with pytest.raises(ValueError, match="finite"):
        series.preintegrate(*intervals[0], **(kwargs | {"gyro_noise_density": np.nan}))


def test_series_checks_global_order_and_does_not_clip_away_missing_samples(tmp_path):
    times = EPOCH + np.array([0, 100, 110, 120], dtype=np.int64) * 1_000_000
    reader = _write(tmp_path, times, np.zeros((4, 3)), np.zeros((4, 3)))
    series = reader.imu_series()
    kwargs = dict(gyro_noise_density=.02, accel_noise_density=.15)
    with pytest.raises(ValueError, match="gap"):
        series.factor(int(times[1]) - 1, int(times[1]), **kwargs)
    series.preintegrate(int(times[1]), int(times[-1]), **kwargs)
    other = tmp_path / "unordered"
    other.mkdir()
    reader = _write(other, times[[0, 2, 1, 3]], np.zeros((4, 3)), np.zeros((4, 3)))
    with pytest.raises(ValueError, match="increase"):
        reader.imu_series()
