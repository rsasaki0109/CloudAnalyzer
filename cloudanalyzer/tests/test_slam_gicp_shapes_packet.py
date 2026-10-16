"""Published native-shape packet consistency and tamper detection."""
import importlib.util
from pathlib import Path
import shutil

import pytest

PACKET = Path(__file__).resolve().parents[2]/"benchmarks/slam/flatwall-gicp-shapes"
SPEC = importlib.util.spec_from_file_location("gicp_shapes_packet", PACKET/"verify.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def test_published_native_shape_packet():
    result = verifier.verify()
    assert result["shape_points"] == 47889 and result["point_sets"] == 72


def test_packet_detects_modified_summary_bytes(tmp_path):
    target = tmp_path/"packet"
    shutil.copytree(PACKET, target)
    with (target/"audit/summary.json").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(AssertionError, match="changed summary.json"):
        verifier.verify(target)
