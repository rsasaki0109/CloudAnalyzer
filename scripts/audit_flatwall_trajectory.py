#!/usr/bin/env python3
"""Evaluate a frozen Flatwall trajectory under explicit, unresolved reference assumptions."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

import flatwall_trajectory_reference as scoring

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def checked_path(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or Path(name).is_absolute():
        raise ValueError("input paths must be relative to this checkout")
    return path


def audit(protocol_path, freeze_path, out):
    if out.exists():
        raise FileExistsError(out)
    protocol, freeze = read(protocol_path), read(freeze_path)
    freeze_digest = sha(freeze_path)
    if protocol["schema"] != "cloudanalyzer.flatwall_trajectory_exploratory.v1":
        raise ValueError("unsupported evaluation protocol")
    if (protocol["association"]["maximum_bracket_span_ns"] != scoring.MAX_BRACKET_NS
            or protocol["validation"]["quaternion_norm_atol"] != scoring.QUATERNION_ATOL
            or protocol["alignment"]["rank_relative_threshold"] != scoring.RANK_RTOL):
        raise ValueError("protocol and implemented numeric policy differ")
    inputs = {key: checked_path(value["path"]) for key, value in protocol["input_artifacts"].items()}
    for path in inputs.values():
        if out.resolve().is_relative_to(path.parent):
            raise ValueError("output must stay outside frozen input directories")
    expected = {value["path"]: value["sha256"] for value in protocol["input_artifacts"].values()}
    expected.update(freeze["source_files_sha256"])
    if any(name not in freeze["source_files_sha256"] for name in [Path(__file__).relative_to(ROOT).as_posix(), Path(scoring.__file__).relative_to(ROOT).as_posix()]):
        raise ValueError("evaluator sources must be frozen before reference parsing")
    if sha(protocol_path) != freeze["protocol_sha256"]:
        raise ValueError("scoring protocol changed after controller freeze")
    receipt = read(inputs["estimate_receipt"])
    if receipt["schema"] != "cloudanalyzer.flatwall_imu_odometry.v1":
        raise ValueError("expected original frozen Flatwall frontend")
    for name, value in receipt["outputs"].items():
        if Path(name).name != name:
            raise ValueError("invalid estimate output name")
        expected[(inputs["estimate_receipt"].parent / name).relative_to(ROOT).as_posix()] = value["sha256"]

    def verify():
        for name, digest in expected.items():
            if sha(checked_path(name)) != digest:
                raise ValueError(f"frozen input changed: {name}")
        if sha(protocol_path) != freeze["protocol_sha256"]:
            raise ValueError("evaluation protocol changed")
        if sha(freeze_path) != freeze_digest:
            raise ValueError("controller freeze changed")

    verify()
    out.mkdir(parents=True, exist_ok=False)
    with (out / "protocol.json").open("xb") as f:
        f.write(protocol_path.read_bytes())
    save(out / "inputs.json", expected)
    try:
        with np.load(inputs["estimate"], allow_pickle=False) as arrays:
            times, poses = arrays["timestamps_ns"].copy(), arrays["poses"].copy()
        scoring.timestamps(times)
        scoring.validate_poses(poses)
        scans = read(inputs["estimate_receipt"].parent / "per-scan.json")
        if len(times) != 185 or len(scans) != 185 or len(poses) != 185:
            raise ValueError("expected all 185 original estimates")
        if [s["scan_index"] for s in scans] != list(range(1, 186)) or [s["reference_ns"] for s in scans] != times.tolist():
            raise ValueError("original scan identity/time mismatch")
        ref_times, ref_poses, reference = scoring.read_selected_reference(inputs["reference_archive"])
        rows, indices, paired = scoring.associate(times, ref_times, ref_poses)
        if len(indices):
            result, errors = scoring.score(poses[indices], paired, indices, times[indices])
        else:
            result, errors = {"metrics": {}, "position_fit": {"status": "no_coverage"}, "samples": [], "edges": []}, {}
        selected = {row["estimate_index"]: row for row in result.pop("samples")}
        for row, scan in zip(rows, scans):
            row["scan_index"] = scan["scan_index"]
            row["errors"] = {k: v for k, v in selected.get(row["estimate_index"], {}).items() if k != "estimate_index"} or None
        save(out / "samples.json", rows)
        save(out / "edges.json", result.pop("edges"))
        np.savez_compressed(out / "paired.npz", original_indices=indices, timestamps_ns=times[indices], estimated=poses[indices], reference=paired)
        np.savez_compressed(out / "errors.npz", **errors)
        save(out / "reference-member.json", reference)
        summary = dict(schema="cloudanalyzer.flatwall_trajectory_exploratory_result.v1", status="conditional_numeric_comparison",
                       reference_inspection="exploratory; reference prefixes inspected before scoring policy", estimate_commit=protocol["estimate_commit"],
                       original_estimates=len(times), paired_estimates=len(indices), full_cohort_covered=len(indices) == len(times),
                       coverage_counts=dict(Counter(r["status"] for r in rows)), original_recording_excluded_scan_indices=[0],
                       interpretation=protocol["conditional_interpretation"], unresolved_provenance=protocol["unresolved_provenance"],
                       declared_initial_state=receipt["declared_state"], **result,
                       scope="Arithmetic under declared reference body/clock/layout assumptions on an uncalibrated frozen frontend. No validated physical accuracy, improvement, GLIM comparison, runtime or adoption claim.")
        save(out / "summary.json", summary)
        verify()
        outputs = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
        save(out / "receipt.json", dict(protocol_sha256=sha(protocol_path), controller_freeze_sha256=freeze_digest, input_files_sha256=expected,
                                        output_files_sha256=outputs, input_hashes_unchanged=True, reference_member=reference))
    except Exception as exc:
        save(out / "failure.json", dict(error_type=type(exc).__name__, error=str(exc)))
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--controller-freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.protocol, args.controller_freeze, args.out), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
