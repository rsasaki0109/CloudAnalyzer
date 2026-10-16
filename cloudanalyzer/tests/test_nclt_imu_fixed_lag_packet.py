"""Recheck full real-data replay outputs without rerunning the estimator."""
import importlib.util
import json
from pathlib import Path

import pytest


def verifier():
    path = Path(__file__).resolve().parents[2] / "benchmarks/slam/nclt-imu-fixed-lag/verify.py"
    spec = importlib.util.spec_from_file_location("fixed_lag_packet_verify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_replay_scores_and_gap_transitions_match_independent_pose_math():
    module = verifier()
    assert module.verify()["verified"] == [
        {"session": "2012-04-29", "poses": 236, "edges": 235, "fused_updates": 193, "gap_fallbacks": 42, "marginalizations": 112},
        {"session": "2012-06-15", "poses": 238, "edges": 237, "fused_updates": 102, "gap_fallbacks": 135, "marginalizations": 11},
    ]


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_full_replay_packet_handles_checkout_newlines_and_rejects_tampering(tmp_path, newline):
    module = verifier()
    result_path = module.PACKET / "results.json"
    packet = json.loads(result_path.read_text(encoding="utf-8"))
    for name in [*packet["files_sha256"], str(result_path.relative_to(module.ROOT))]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((module.ROOT / name).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", newline))
    module.ROOT = tmp_path
    module.PACKET = tmp_path / "benchmarks/slam/nclt-imu-fixed-lag"
    module.MATH = tmp_path / "benchmarks/slam/nclt-imu-pilot/verify.py"
    assert len(module.verify()["verified"]) == 2
    with (module.PACKET / "june-poses.csv").open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(AssertionError, match="changed input"):
        module.verify()
