"""Published real eight-case arithmetic and resigned semantic tamper controls."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT/"benchmarks/slam/flatwall02-prediction"
spec = importlib.util.spec_from_file_location("flatwall02_packet", PACKET/"verify.py")
packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)

def save(path, value): path.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
def copy_packet(tmp_path):
    target = tmp_path/"checkout/benchmarks/slam/flatwall02-prediction"
    shutil.copytree(PACKET, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = packet.read(PACKET/"manifest.json")
    for name in manifest["dependencies"] | manifest["checkout_sources_sha256"]:
        destination = target.parents[2]/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
    return target
def resign(target, changed):
    closure = packet.read(target/"estimates-receipt.json")
    if changed in closure["output_files_sha256"]:
        closure["output_files_sha256"][changed] = packet.sha(target/changed)
    save(target/"estimates-receipt.json", closure)
    manifest = packet.read(target/"manifest.json")
    for name in manifest["files_sha256"]: manifest["files_sha256"][name] = packet.sha(target/name)
    save(target/"manifest.json", manifest)

def test_original_reference_pairing_and_all_eight_real_cases():
    result = packet.verify(PACKET)
    assert (result["cases"], result["estimates_per_case"], result["paired_per_case"], result["consecutive_edges_per_case"], result["snapshots"]) == (8, 135, 114, 111, 1080)
    assert result["physical_provenance_certified"] is False

@pytest.mark.parametrize("change", ["nanosecond", "original_id", "coverage", "edge", "metric", "pose", "spectrum", "boundary", "reference", "cross_recording"])
def test_semantic_tampering_is_rejected_after_resigning(tmp_path, change):
    target = copy_packet(tmp_path)
    base = "last-pose/raw-0.1875/"
    if change in {"nanosecond", "pose"}:
        name = base+"trajectory.npz"
        with np.load(target/name, allow_pickle=False) as saved: arrays = {k: saved[k].copy() for k in saved.files}
        if change == "nanosecond": arrays["timestamps_ns"][60] += 1
        else: arrays["poses"][60, 0, 3] += .5
        np.savez_compressed(target/name, **arrays)
    elif change == "original_id":
        name = base+"per-scan.json"
        rows = packet.read(target/name)
        rows[60]["scan_index"] += 1
        save(target/name, rows)
    elif change == "coverage":
        name = base+"samples.json"
        rows = packet.read(target/name)
        next(r for r in rows if r["status"] == "reference_gap")["status"] = "interpolated"
        save(target/name, rows)
    elif change == "edge":
        name = base+"edges.json"
        rows = packet.read(target/name)
        rows[30]["right_scan_index"] += 1
        save(target/name, rows)
    elif change == "metric":
        name = "summary.json"
        data = packet.read(target/name)
        data["controls"]["last-pose"]["raw-0.1875"]["metrics"]["rigid_position_ape_m"]["rmse"] *= .75
        save(target/name, data)
    elif change == "spectrum":
        name = base+"frames.jsonl"
        rows = [json.loads(line) for line in (target/name).read_text(encoding="utf-8").splitlines()]
        next(r for r in rows if r["geometry"] is not None)["geometry"]["eigenvalues"][0] *= .5
        (target/name).write_text("\n".join(json.dumps(r) for r in rows)+"\n", encoding="utf-8")
    elif change == "boundary":
        name = "scoring-boundary.json"
        data = packet.read(target/name)
        data["full_selected_reference_parsed_before_this_boundary"] = True
        save(target/name, data)
    elif change == "reference":
        name = "reference-source.txt"
        rows = (target/name).read_text(encoding="utf-8").splitlines()
        first = next(i for i, line in enumerate(rows) if line.strip() and not line.lstrip().startswith("#"))
        tokens = rows[first].split()
        tokens[0] += "1"  # A nonzero sub-nanosecond digit must not be rounded away.
        rows[first] = " ".join(tokens)
        (target/name).write_text("\n".join(rows)+"\n", encoding="utf-8")
    else:
        name = "cross-recording.json"
        data = packet.read(target/name)
        data["cases"][-1]["rigid_position_ape_m"] = 0.
        save(target/name, data)
    resign(target, name)
    with pytest.raises((ValueError, AssertionError)):
        packet.verify(target)

def test_modified_original_bytes_are_detected(tmp_path):
    target = copy_packet(tmp_path)
    with (target/"paired-reference.npz").open("ab") as stream: stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        packet.verify(target)
