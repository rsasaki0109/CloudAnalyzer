"""Reject altered diagnostics after their published hash chain is updated."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT=Path(__file__).resolve().parents[2]
PACKET=ROOT/"benchmarks/slam/flatwall-frontend-constraints"
spec=importlib.util.spec_from_file_location("flatwall_constraints_packet",PACKET/"verify.py")
packet=importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)


def test_complete_diagnostic_packet():
    result=packet.verify()
    assert result["verified_snapshots"]==740 and result["scans_per_control"]==185
    assert result["physical_provenance_certified"] is False


@pytest.mark.parametrize("change",["weak_count","physical_claim","drop_scan","spectrum","prediction","pose","hide_prediction_only"])
def test_semantic_change_is_rejected_with_resigned_hashes(tmp_path,change):
    target=tmp_path/"packet"
    shutil.copytree(PACKET,target)
    row_change=change in {"drop_scan","spectrum","prediction","pose","hide_prediction_only"}
    name="raw-legacy/frames.jsonl" if row_change else "summary.json"
    if row_change:
        rows=[json.loads(line) for line in (target/name).read_text(encoding="utf-8").splitlines()]
        if change=="drop_scan": rows.pop()
        if change=="spectrum": rows[1]["geometry"]["eigenvalues"][0]*=.5
        if change=="prediction": rows[2]["prediction_pose"][0][3]+=.1
        if change=="pose": rows[2]["returned_pose"][0][3]+=.1
        if change=="hide_prediction_only": rows[0]["prediction_only"]=False
        encoded=("\n".join(json.dumps(row,separators=(",",":")) for row in rows)+"\n").encode("utf-8")
    else:
        data=json.loads((target/name).read_text(encoding="utf-8"))
        if change=="weak_count": data["controls"]["raw-legacy"]["weak_spectra"]+=1
        if change=="physical_claim": data["physical_provenance_certified"]=True
        encoded=(json.dumps(data,indent=2)+"\n").encode("utf-8")
    (target/name).write_bytes(encoded)
    manifest=json.loads((target/"packet-files.json").read_text(encoding="utf-8"))
    manifest["files"][name]=dict(bytes=len(encoded),sha256=hashlib.sha256(encoded).hexdigest())
    receipt=json.loads((target/"receipt.json").read_text(encoding="utf-8"))
    receipt["output_files_sha256"][name]=hashlib.sha256(encoded).hexdigest()
    receipt_bytes=(json.dumps(receipt,indent=2)+"\n").encode("utf-8")
    (target/"receipt.json").write_bytes(receipt_bytes)
    manifest["files"]["receipt.json"]=dict(bytes=len(receipt_bytes),sha256=hashlib.sha256(receipt_bytes).hexdigest())
    (target/"packet-files.json").write_text(json.dumps(manifest),encoding="utf-8")
    with pytest.raises(AssertionError): packet.verify(target)
