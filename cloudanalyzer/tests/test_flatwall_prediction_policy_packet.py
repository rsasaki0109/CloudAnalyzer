"""Reject changed score semantics even after the published output hashes change."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT=Path(__file__).resolve().parents[2]
PACKET=ROOT/"benchmarks/slam/flatwall-prediction-policy"
spec=importlib.util.spec_from_file_location("flatwall_prediction_packet",PACKET/"verify.py")
packet=importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)


def test_all_eight_comparisons_are_recomputed():
    result=packet.verify()
    assert result["cases"]==8 and result["paired_per_case"]==183
    assert result["new_snapshots"]==740 and result["physical_provenance_certified"] is False


@pytest.mark.parametrize("change",["metric","missing_policy","coverage","physical_claim","sample_error","bridge","uncovered_error"])
def test_changed_arithmetic_is_rejected_with_resigned_output_hashes(tmp_path,change):
    target=tmp_path/"packet"
    shutil.copytree(PACKET,target)
    name={"sample_error":"last-pose/raw-legacy/samples.json","uncovered_error":"last-pose/raw-legacy/samples.json",
          "bridge":"last-pose/raw-legacy/edges.json"}.get(change,"summary.json")
    data=json.loads((target/name).read_text(encoding="utf-8"))
    if change=="metric": data["controls"]["last-pose"]["raw-legacy"]["metrics"]["rigid_position_ape_m"]["rmse"]=.01
    if change=="missing_policy": del data["controls"]["last-pose"]
    if change=="coverage": data["full_cohort_covered"]=True
    if change=="physical_claim": data["physical_provenance_certified"]=True
    if change=="sample_error": data[50]["errors"]["rigid_position_ape_m"]+=.25
    if change=="bridge": data[-1]["right_estimate_index"]=184
    if change=="uncovered_error": data[-1]["errors"]={"rigid_position_ape_m":0.}
    encoded=(json.dumps(data,indent=2)+"\n").encode("utf-8")
    (target/name).write_bytes(encoded)
    manifest=json.loads((target/"packet-files.json").read_text(encoding="utf-8"))
    manifest["files"][name]=dict(bytes=len(encoded),sha256=hashlib.sha256(encoded).hexdigest())
    receipt=json.loads((target/"audit-receipt.json").read_text(encoding="utf-8"))
    receipt["output_files_sha256"][name]=hashlib.sha256(encoded).hexdigest()
    encoded_receipt=(json.dumps(receipt,indent=2)+"\n").encode("utf-8")
    (target/"audit-receipt.json").write_bytes(encoded_receipt)
    manifest["files"]["audit-receipt.json"]=dict(bytes=len(encoded_receipt),sha256=hashlib.sha256(encoded_receipt).hexdigest())
    (target/"packet-files.json").write_text(json.dumps(manifest),encoding="utf-8")
    with pytest.raises(AssertionError): packet.verify(target)
