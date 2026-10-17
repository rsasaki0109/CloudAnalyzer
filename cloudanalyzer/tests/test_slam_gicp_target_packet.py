"""Check packet identities and detect modified public evidence."""
import importlib.util
from pathlib import Path
import shutil

import pytest

PACKET = Path(__file__).resolve().parents[2]/"benchmarks/slam/flatwall-gicp-target"
spec = importlib.util.spec_from_file_location("gicp_target_packet", PACKET/"verify.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_real_target_packet_identities_and_summaries():
    result = verifier.verify()
    assert result["registrations"] == 72
    assert result["families"] == 2


def test_changed_evidence_fails_verification(tmp_path):
    destination = tmp_path/"packet"
    shutil.copytree(PACKET, destination)
    path = destination/"audit/summary.json"
    path.write_bytes(path.read_bytes()+b"\n")
    with pytest.raises(AssertionError):
        verifier.verify(destination)
