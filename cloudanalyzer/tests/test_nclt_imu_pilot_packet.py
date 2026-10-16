"""Verify public real-data scores without rerunning the frozen estimator."""
import importlib.util
import json
from pathlib import Path

import pytest


def _verifier():
    path = Path(__file__).resolve().parents[2] / "benchmarks/slam/nclt-imu-pilot/verify.py"
    spec = importlib.util.spec_from_file_location("nclt_imu_pilot_packet_verify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_real_pilot_metrics_match_independent_public_pose_verifier():
    module = _verifier()
    result = module.verify()
    assert result["verified"] == [
        {"session": "2012-04-29", "windows": 20, "nodes": 186, "edges": 166},
        {"session": "2012-06-15", "windows": 12, "nodes": 59, "edges": 47},
    ]


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_packet_verification_survives_checkout_newlines_but_rejects_changed_content(tmp_path, newline):
    module = _verifier()
    packet_path = module.PACKET / "results.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    for relative in [*packet["files_sha256"], str(packet_path.relative_to(module.ROOT))]:
        original = module.ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(original.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", newline))
    module.ROOT = tmp_path
    module.PACKET = tmp_path / "benchmarks/slam/nclt-imu-pilot"
    assert len(module.verify()["verified"]) == 2
    with (module.PACKET / "april-poses.csv").open("ab") as stream:
        stream.write(b"changed content")
    with pytest.raises(AssertionError, match="changed input"):
        module.verify()
