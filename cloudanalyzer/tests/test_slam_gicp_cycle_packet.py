"""Public cycle witness packet integrity and changed-decision rejection."""
import importlib.util
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT/"benchmarks/slam/flatwall-gicp-cycle"
spec = importlib.util.spec_from_file_location("gicp_cycle_packet", PACKET/"verify.py")
packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)


def test_published_witnesses_and_status_accounting():
    report = packet.verify()
    assert report["cycles"] == 4 and report["histories"] == 72
    assert report["statuses"] == dict(converged_step=54, cycle_detected=4, max_iterations=6, insufficient_correspondences=8)


def test_changed_decision_is_rejected(tmp_path):
    copy = tmp_path/"packet"
    shutil.copytree(PACKET, copy)
    path = copy/"audit/frames.json"
    path.write_bytes(path.read_bytes().replace(b'"cycle_detected"', b'"converged_step"', 1))
    with pytest.raises(AssertionError): packet.verify(copy)
