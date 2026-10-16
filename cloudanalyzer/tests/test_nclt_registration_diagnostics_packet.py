"""Integrity and independent aggregation of the real registration reports."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "benchmarks/slam/nclt-registration-diagnostics"
spec = importlib.util.spec_from_file_location("registration_packet", PACKET / "verify.py")
assert spec is not None and spec.loader is not None
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def copied(tmp_path):
    packet = tmp_path / PACKET.name
    shutil.copytree(PACKET, packet)
    shutil.copytree(PACKET.parent / "nclt-imu-frontend", tmp_path / "nclt-imu-frontend")
    return packet


def test_real_registration_reports_and_all_fallback_groups():
    assert verifier.verify(PACKET) == 948


def test_lf_and_crlf_checkout_variants(tmp_path):
    packet = copied(tmp_path)
    for path in list(packet.glob("*.csv")) + list((tmp_path / "nclt-imu-frontend").glob("*.csv")):
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert verifier.verify(packet) == 948


def test_altered_rows_and_summary_are_rejected(tmp_path):
    packet = copied(tmp_path)
    summary = packet / "results.json"
    original = summary.read_text()
    result = json.loads(original)
    result["sessions"][0]["arms"]["baseline"]["all"]["iterations"] += 1
    summary.write_text(json.dumps(result))
    with pytest.raises(AssertionError):
        verifier.verify(packet)
    summary.write_text(original)
    csv_path = packet / "april-registrations.csv"
    csv_path.write_text(csv_path.read_text().replace("converged", "singular_system", 1))
    with pytest.raises(AssertionError):
        verifier.verify(packet)
