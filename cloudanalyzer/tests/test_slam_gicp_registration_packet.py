"""Published registration receipt consistency and tamper detection."""
import importlib.util
from pathlib import Path
import shutil

import pytest

PACKET = Path(__file__).resolve().parents[2]/"benchmarks/slam/flatwall-gicp-registration"
SPEC = importlib.util.spec_from_file_location("gicp_registration_packet", PACKET/"verify.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def test_published_registration_packet():
    result = verifier.verify()
    assert result["registrations"] == 36 and result["cases"] == 4


def test_published_packet_detects_modified_bytes(tmp_path):
    destination = tmp_path/"packet"
    shutil.copytree(PACKET, destination)
    with (destination/"audit/frames.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(AssertionError, match="changed frames.json"):
        verifier.verify(destination)
