"""Published GICP evidence checks without native code or private point arrays."""
import importlib.util
from pathlib import Path
import shutil

import pytest


def verifier():
    path = Path(__file__).resolve().parents[2] / "benchmarks/slam/flatwall-gicp-linearization/verify.py"
    spec = importlib.util.spec_from_file_location("gicp_packet_verify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packet_preserves_all_cases_pairs_and_reported_geometry():
    result = verifier().verify()
    assert result["public_json_files_verified"] == 9
    assert result["frames"] == 256 and result["retained_pairs"] == 118040


def test_changed_evidence_bytes_are_rejected(tmp_path):
    module = verifier()
    for path in module.PACKET.rglob("*.json"):
        target = tmp_path/path.relative_to(module.PACKET)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    module.verify(tmp_path)
    with (tmp_path/"audit/frames.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(AssertionError, match="changed frames.json"):
        module.verify(tmp_path)
