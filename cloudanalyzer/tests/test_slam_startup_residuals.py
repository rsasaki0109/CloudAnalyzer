"""Closed-form diagnostic contracts; these fixtures are not sensor benchmarks."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def controller():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/audit_flatwall_startup_residuals.py"
    )
    spec = importlib.util.spec_from_file_location("startup_residual_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_constant_acceleration_with_uneven_intervals_recovers_gravity():
    module = controller()
    times = np.array([0.0, 0.1, 0.3, 0.7, 1.0])
    dt = np.diff(times)
    force, gravity = np.array([0.5, -0.2, 9.7]), np.array([0.1, -0.3, -9.8])
    positions = (
        [3.0, -2.0, 5.0]
        + times[:, None] * [1.0, -0.5, 0.2]
        + 0.5 * times[:, None] ** 2 * (force + gravity)
    )
    result = module.decompose(
        positions, dt, 0.5 * dt[:, None] ** 2 * force, dt[:, None] * force
    )
    np.testing.assert_allclose(result["gravity"], gravity, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["residual"], 0, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result["pose_residual"], 0, rtol=0, atol=1e-13)
    np.testing.assert_allclose(result["imu_residual"], 0, rtol=0, atol=1e-13)


def test_signed_cross_energy_keeps_cancellation_instead_of_positive_sensor_shares():
    module = controller()
    dt = np.ones(4)
    dv = np.zeros((4, 3))
    dv[:, 0] = [1.0, -2.0, 4.0, 0.0]
    rate = np.vstack([np.zeros(3), np.cumsum(2 * dv[:-1], axis=0)])
    positions = np.vstack([np.zeros(3), np.cumsum(rate, axis=0)])
    result = module.decompose(positions, dt, np.zeros((4, 3)), dv)
    np.testing.assert_allclose(
        result["residual"][:, 0], [0.0, -3.0, 3.0], rtol=0, atol=1e-14
    )
    assert result["energy"] == dict(total=18.0, pose=72.0, imu=18.0, cross=-72.0)
    assert result["max_sum_error"] == 0.0


def test_checked_clock_offsets_and_coarse_references_preserve_exact_endpoints():
    module = controller()
    stamps = 1_700_000_000_000_000_123 + np.arange(64, dtype=np.int64) * 100_000_000
    shifted = module.shifted_references(stamps[::-1][::-1], 5_000_000)
    np.testing.assert_array_equal(shifted - stamps, 5_000_000)
    np.testing.assert_array_equal(stamps % 100_000_000, 123)
    for stride, count in [(1, 64), (3, 22), (7, 10)]:
        indices = module.reference_indices(64, stride)
        assert len(indices) == count and indices[0] == 0 and indices[-1] == 63
    for invalid in [0, 5, 63]:
        with pytest.raises(ValueError, match="endpoints"):
            module.reference_indices(64, invalid)
    limit = np.iinfo(np.int64)
    for bad_stamps, offset in [
        (np.array([limit.min, limit.min + 1, limit.min + 2], dtype=np.int64), -1),
        (np.array([limit.min, 0, limit.max], dtype=np.int64), 0),
        (stamps[::-1], 0),
        (stamps.astype(np.float64), 0),
    ]:
        with pytest.raises(ValueError):
            module.shifted_references(bad_stamps, offset)


def test_residual_distribution_reports_all_energy_and_stable_ties():
    module = controller()
    result = module.distribution(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    )
    assert result["rms"] == pytest.approx(np.sqrt(10 / 3))
    assert result["maximum_row"] == 2 and result["top_rows"] == [2, 1, 0]
    assert result["top1_energy_share"] == 0.9 and result["top5_energy_share"] == 1.0
    zero = module.distribution(np.zeros((3, 3)))
    assert zero["top_rows"] == [0, 1, 2]
    assert zero["top1_energy_share"] == zero["top5_energy_share"] == 0.0


def test_decomposition_rejects_nonfinite_data_and_invalid_intervals():
    module = controller()
    positions, dt, dp, dv = (
        np.zeros((4, 3)),
        np.ones(3),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
    )
    for values in [
        (positions[:2], dt, dp, dv),
        (positions, dt[:2], dp, dv),
        (positions, [1.0, 0.0, 1.0], dp, dv),
        (positions, [1.0, np.nan, 1.0], dp, dv),
        (np.full_like(positions, np.inf), dt, dp, dv),
    ]:
        with pytest.raises(ValueError):
            module.decompose(*values)
