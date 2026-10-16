"""Verify every public frontend pose and fallback with independent pose math."""
import importlib.util
import json
from pathlib import Path

import pytest


def verifier():
    path = Path(__file__).resolve().parents[2] / "benchmarks/slam/nclt-imu-frontend/verify.py"
    spec = importlib.util.spec_from_file_location("frontend_packet_verify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frontend_packet_scores_every_pose_including_unassisted_transitions():
    assert verifier().verify()["verified"] == [
        {"session": "2012-04-29", "poses": 236, "edges": 235, "predictions": 158, "gaps": 42},
        {"session": "2012-06-15", "poses": 238, "edges": 237, "predictions": 45, "gaps": 135},
    ]


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_frontend_packet_accepts_checkout_newlines_and_rejects_tampering(tmp_path, newline):
    module = verifier()
    results = module.PACKET / "results.json"
    packet = json.loads(results.read_text(encoding="utf-8"))
    for name in [*packet["files_sha256"], str(results.relative_to(module.ROOT))]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((module.ROOT / name).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", newline))
    module.ROOT = tmp_path
    module.PACKET = tmp_path / "benchmarks/slam/nclt-imu-frontend"
    module.MATH = tmp_path / "benchmarks/slam/nclt-imu-pilot/verify.py"
    assert len(module.verify()["verified"]) == 2
    with (module.PACKET / "june-poses.csv").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(AssertionError, match="changed input"):
        module.verify()
