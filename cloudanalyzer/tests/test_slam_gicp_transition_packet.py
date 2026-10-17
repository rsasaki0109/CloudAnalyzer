"""Detect changed transition evidence and check published accounting summaries."""
import importlib.util
from pathlib import Path
import shutil

import pytest

PACKET = Path(__file__).resolve().parents[2]/"benchmarks/slam/flatwall-gicp-transition"
spec = importlib.util.spec_from_file_location("gicp_transition_packet", PACKET/"verify.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_real_transition_packet_bytes_identities_and_accounting():
    result = verifier.verify()
    assert result["native_cube_cost_checks"] == 5488
    assert result["accepted_transitions"] == 686


def test_changed_cost_cube_fails_verification(tmp_path):
    destination = tmp_path/"packet"
    shutil.copytree(PACKET, destination)
    path = destination/"audit/limited-tails.json"
    path.write_bytes(path.read_bytes()+b"\n")
    with pytest.raises(AssertionError):
        verifier.verify(destination)
