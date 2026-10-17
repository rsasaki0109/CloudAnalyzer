"""Reject changed score, coverage or physical-accuracy claims in the packet."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "benchmarks/slam/flatwall-trajectory"
spec = importlib.util.spec_from_file_location("flatwall_trajectory_packet", PACKET / "verify.py")
packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)


def test_conditional_coverage_and_metrics():
    result = packet.verify()
    assert result["paired_estimates"] == 183 and result["excluded_after_reference"] == 2
    assert 34.9 < result["conditional_rigid_position_rmse_m"] < 35.0
    assert result["physical_provenance_certified"] is False


@pytest.mark.parametrize("change", ["metric", "coverage", "physical_claim", "bridge", "parser_policy"])
def test_changed_accounting_is_rejected_even_with_updated_file_hash(tmp_path, change):
    target = tmp_path / "packet"
    shutil.copytree(PACKET, target)
    name = "strict-time-comparison.json" if change == "parser_policy" else ("edges.json" if change == "bridge" else "summary.json")
    data = json.loads((target / name).read_text(encoding="utf-8"))
    if change == "metric": data["metrics"]["rigid_position_ape_m"]["rmse"] = 1.
    if change == "coverage": data["full_cohort_covered"] = True
    if change == "physical_claim": data["status"] = "validated_physical_accuracy"
    if change == "bridge": data[-1]["right_estimate_index"] = 184
    if change == "parser_policy": data["policy_and_numeric_thresholds_unchanged"] = False
    encoded = (json.dumps(data, indent=2) + "\n").encode("utf-8")
    (target / name).write_bytes(encoded)
    manifest_name = "strict-time-files.json" if change == "parser_policy" else "packet-files.json"
    manifest = json.loads((target / manifest_name).read_text(encoding="utf-8"))
    manifest["files"][name] = dict(bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest())
    (target / manifest_name).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(AssertionError): packet.verify(target)
