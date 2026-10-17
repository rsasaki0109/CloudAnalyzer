"""Reject changed control comparisons even when a packet hash is updated."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "benchmarks/slam/flatwall-frontend-controls"
spec = importlib.util.spec_from_file_location("flatwall_controls_packet", PACKET/"verify.py")
packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)


def test_complete_exploratory_packet():
    result = packet.verify()
    assert result["arms"] == 5 and result["paired_per_arm"] == 183
    assert result["physical_provenance_certified"] is False


@pytest.mark.parametrize("change",["metric","omit_arm","coverage","physical_claim","hide_failed_scan","bridge","freeze_claim"])
def test_semantic_tamper_is_rejected(tmp_path, change):
    target = tmp_path/"packet"
    shutil.copytree(PACKET,target)
    name = {"hide_failed_scan":"raw-legacy/per-scan.json","bridge":"raw-legacy/edges.json"}.get(change,"summary.json")
    data = json.loads((target/name).read_text(encoding="utf-8"))
    if change == "metric": data["controls"]["raw-0.1875"]["metrics"]["rigid_position_ape_m"]["rmse"] = .01
    if change == "omit_arm": del data["controls"]["gyro-0.1875"]
    if change == "coverage": data["full_cohort_covered"] = True
    if change == "physical_claim": data["physical_provenance_certified"] = True
    if change == "freeze_claim": data["all_estimates_frozen_before_this_scoring"] = False
    if change == "hide_failed_scan": data.pop()
    if change == "bridge": data[-1]["right_estimate_index"] = 184
    encoded = (json.dumps(data,indent=2)+"\n").encode("utf-8")
    (target/name).write_bytes(encoded)
    manifest = json.loads((target/"packet-files.json").read_text(encoding="utf-8"))
    manifest["files"][name] = dict(bytes=len(encoded),sha256=hashlib.sha256(encoded).hexdigest())
    (target/"packet-files.json").write_text(json.dumps(manifest),encoding="utf-8")
    with pytest.raises(AssertionError): packet.verify(target)
