"""Boundary controls for the frozen-cohort comparison; no native module required."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("flatwall_controls", ROOT / "scripts/audit_flatwall_frontend_controls.py")
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


def test_equal_length_but_different_scan_times_are_rejected(tmp_path):
    times = np.arange(185, dtype=np.int64) + 1_656_505_431_000_000_000
    poses = np.tile(np.eye(4), (185, 1, 1))
    path = tmp_path / "trajectory.npz"
    np.savez(path, timestamps_ns=times, poses=poses)
    controls.load_cohort(path, times)
    different = times.copy()
    different[70] += 1
    with pytest.raises(ValueError, match="cohort"):
        controls.load_cohort(path, different)


@pytest.mark.parametrize("count", [64, 184, 186])
def test_dropped_or_extended_cohort_is_rejected(tmp_path, count):
    path = tmp_path / "trajectory.npz"
    np.savez(path, timestamps_ns=np.arange(count, dtype=np.int64), poses=np.tile(np.eye(4), (count, 1, 1)))
    with pytest.raises(ValueError, match="185"):
        controls.load_cohort(path)


def test_reference_policy_change_is_rejected(tmp_path, monkeypatch):
    inherited = dict(association=dict(maximum_bracket_span_ns=100_000_001),
                     validation=dict(quaternion_norm_atol=1e-5), alignment=dict(rank_relative_threshold=1e-10))
    path = tmp_path / "policy.json"
    controls.save(path, inherited)
    monkeypatch.setattr(controls, "ROOT", tmp_path)
    policy = dict(schema="cloudanalyzer.flatwall_frontend_controls.v1", arms=controls.ARMS, lidar=controls.LIDAR,
                  original_scan_indices=list(range(1, 186)), gyro_bias_rad_s=[0., 0., 0.], max_gap_s=.05,
                  imu_from_lidar=np.eye(4).tolist(), threads=dict(RAYON_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1"),
                  physical_provenance_certified=False, scoring_protocol="policy.json", scoring_protocol_sha256=controls.sha(path))
    with pytest.raises(ValueError, match="reference policy"):
        controls.validate_protocol(policy)


def test_changed_frozen_file_is_rejected(tmp_path):
    path = tmp_path / "estimate.json"
    controls.save(path, dict(scans=185))
    frozen = {str(path): controls.sha(path)}
    controls.verify(frozen)
    path.write_text('{"scans":184}', encoding="utf-8")
    with pytest.raises(ValueError, match="frozen file changed"):
        controls.verify(frozen)
