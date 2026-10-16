"""Frozen-input audit I/O; the fixture is not a real accuracy benchmark."""
import csv
import importlib
import json

import pytest

pytest.importorskip("cloudanalyzer_core")
from test_nclt_imu_frontend import recording  # noqa: E402, F401
from test_imu_preintegration import EPOCH  # noqa: E402
from test_scan_header import _write as write_scans  # noqa: E402


def test_audit_preserves_frozen_estimates_and_checks_every_scan(recording, tmp_path, monkeypatch):
    replay, series, times, config = recording
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")
    lidar_dir = tmp_path / "lidar"
    lidar_dir.mkdir()
    lidar = write_scans(lidar_dir, [int(t) for t in times], [replay.frames.FRAME_ID] * len(times), "cdr")
    euler = tmp_path / "euler.csv"
    euler.write_text(f"{EPOCH // 1000},0,0,0\n")
    frozen = tmp_path / "frontend"
    lidar_file = next(lidar.glob("*.mcap"))
    replay.replay(lidar_file, tmp_path / "imu", euler, frozen, imu_topic="/imu")
    before = {p.name: replay.pilot.artifact(p) for p in frozen.iterdir()}
    audit = importlib.import_module("audit_nclt_registration")
    out = tmp_path / "audit"
    result = audit.audit(lidar_file, frozen, out)
    assert {p.name: replay.pilot.artifact(p) for p in frozen.iterdir()} == before
    assert json.loads((out / "protocol.json").read_text())["diagnostics"] is True
    with (out / "registrations.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 * len(times)
    for arm in ["baseline", "assisted"]:
        block = result["arms"][arm]
        assert block["all"]["scans"] == block["pose_bits_unchanged"] == block["iteration_counts_unchanged"] == len(times)
        assert sum(block["all"]["terminations"].values()) == len(times)
        assert sum(x["scans"] for x in block["by_frontend_status"].values()) == len(times)
        selected = [r for r in rows if r["arm"] == arm]
        assert [int(r["timestamp_ns"]) for r in selected] == times.tolist()
        assert selected[0]["termination"] == "empty_map"
    with pytest.raises(FileExistsError):
        audit.audit(lidar_file, frozen, out)
    monkeypatch.setenv("RAYON_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="RAYON_NUM_THREADS"):
        audit.audit(lidar_file, frozen, tmp_path / "wrong-threads")
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")
    (frozen / "trajectory.npz").write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.audit(lidar_file, frozen, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()
