"""Check the frozen diagnostic evidence without rereading the recording."""

import importlib.util
from pathlib import Path
import shutil

import pytest


def verifier():
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/slam/flatwall-startup-residuals/verify.py"
    )
    spec = importlib.util.spec_from_file_location(
        "startup_residual_packet_verify", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_residual_packet_checks_schedule_algebra_and_rejected_states():
    result = verifier().verify()
    assert result["public_json_files_verified"] == 6
    assert result["cases"] == 52 and result["accepted_cases"] == 0


def test_packet_verifier_rejects_changed_evidence_bytes(tmp_path):
    module = verifier()
    for path in module.PACKET.rglob("*.json"):
        target = tmp_path / path.relative_to(module.PACKET)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    module.verify(tmp_path)
    with (tmp_path / "audit/sensor.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(AssertionError, match="changed sensor.json"):
        module.verify(tmp_path)
