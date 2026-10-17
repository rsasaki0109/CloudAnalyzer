"""Tamper tests exercise the public input verifier beyond resigned file hashes."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "benchmarks/slam/flatwall-recording-inputs"
spec = importlib.util.spec_from_file_location("flatwall_inputs_packet", PACKET / "verify.py")
packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def copy_packet(tmp_path):
    target = tmp_path / "checkout/benchmarks/slam/flatwall-recording-inputs"
    shutil.copytree(PACKET, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    checkout = target.parents[2]
    for name in packet.read(PACKET / "manifest.json")["checkout_sources_sha256"]:
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    return target


def resign(target, changed):
    receipt = packet.read(target / "inspection-receipt.json")
    for name in changed:
        if name in receipt["output_files_sha256"]:
            receipt["output_files_sha256"][name] = packet.sha(target / name)
    save(target / "inspection-receipt.json", receipt)
    manifest = packet.read(target / "manifest.json")
    for name in manifest["files_sha256"]:
        manifest["files_sha256"][name] = packet.sha(target / name)
    save(target / "manifest.json", manifest)


def test_checked_in_real_input_accounting():
    result = packet.verify(PACKET)
    assert result["recordings"]["flatwall_01"]["scans"] == 186
    assert set(result["recordings"]) == {"flatwall_01", "flatwall_02"}
    assert result["physical_calibration_certified"] is False


@pytest.mark.parametrize("change", ["one_ns", "missing_exclusion", "false_coverage", "summary", "physical_claim", "tf_claim", "imu_scale", "imu_gap"])
def test_semantic_tampering_is_rejected_after_resigning(tmp_path, change):
    target = copy_packet(tmp_path)
    name = "flatwall_02/scans.json"
    if change in {"one_ns", "missing_exclusion", "false_coverage"}:
        rows = packet.read(target / name)
        if change == "one_ns":
            rows[10]["custom_timebase_ns"] += 1
        elif change == "missing_exclusion":
            rows.pop(0)
        else:
            row = rows[len(rows) // 2]
            row["header_coverage"]["status"] = "outside_imu_range" if row["header_coverage"]["status"] == "covered" else "covered"
        save(target / name, rows)
    elif change == "summary":
        name = "flatwall_02/summary.json"
        summary = packet.read(target / name)
        summary["raw_acceleration_norm"]["median"] *= 9.80665
        save(target / name, summary)
    elif change == "physical_claim":
        name = "provenance.json"
        evidence = packet.read(target / name)
        evidence["physical_calibration_certified"] = True
        save(target / name, evidence)
    elif change == "tf_claim":
        name = "flatwall_02/metadata.json"
        metadata = packet.read(target / name)
        metadata["tf_or_calibration_topics_present"] = True
        save(target / name, metadata)
    else:
        name = "flatwall_02/imu.npz"
        with np.load(target / name, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        if change == "imu_scale":
            arrays["raw_linear_acceleration"] *= 9.80665
        else:
            middle = len(arrays["timestamps_ns"]) // 2
            arrays["timestamps_ns"][middle:] += 50_000_001
        np.savez_compressed(target / name, **arrays)
    resign(target, [name])
    with pytest.raises(ValueError):
        packet.verify(target)


def test_one_changed_byte_is_rejected_without_resigning(tmp_path):
    target = copy_packet(tmp_path)
    with (target / "flatwall_02/imu.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        packet.verify(target)
