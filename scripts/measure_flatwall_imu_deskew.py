#!/usr/bin/env python3
"""Counterbalanced serial/four-thread API timings on eight real flat-wall scans."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import cloudanalyzer_core as core
import cloudanalyzer_core._core as native
import audit_flatwall_raw_point_time as fields

SCAN_INDICES = [1, 2, 9, 17, 49, 93, 139, 185]
ROOT = Path(__file__).resolve().parents[1]


def sources():
    paths = set()
    for crate in ["ca-core", "ca-py", "ca-wasm"]:
        paths.update((ROOT / "rust/crates" / crate / "src").rglob("*.rs"))
        paths.add(ROOT / "rust/crates" / crate / "Cargo.toml")
    paths.update(ROOT / p for p in ["rust/Cargo.toml", "rust/Cargo.lock", "rust/crates/ca-py/Cargo.lock"])
    paths.update(p for p in (ROOT / "rust/crates/ca-py/python/cloudanalyzer_core").iterdir()
                 if p.is_file() and (p.suffix in [".py", ".pyi"] or p.name == "py.typed"))
    return {str(p.relative_to(ROOT)): fields.artifact(p)["sha256"] for p in sorted(paths)}


def measure(bag, audit, output):
    output = output.resolve()
    if output.exists() or audit.resolve() in output.parents:
        raise ValueError("use a fresh output outside the frozen functional audit")
    if os.environ.get("RAYON_NUM_THREADS") != "4":
        raise ValueError("run a fresh process with RAYON_NUM_THREADS=4")
    receipt = json.loads((audit / "receipt.json").read_text(encoding="utf-8"))
    if receipt["schema"] != "cloudanalyzer.flatwall_imu_deskew.v1" or receipt["baseline"] or not receipt["native_results"]["serial_parallel_bits_match"]:
        raise ValueError("a successful native deskew audit is required")
    if receipt["protocol"] != fields.artifact(audit / "protocol.json"):
        raise ValueError("functional audit protocol changed")
    frozen = {"recording": fields.artifact(bag), "native": fields.artifact(Path(native.__file__)),
              "controller": fields.artifact(Path(__file__)), "helper": fields.artifact(Path(fields.__file__)),
              "functional_protocol": fields.artifact(audit / "protocol.json"),
              "functional_receipt": fields.artifact(audit / "receipt.json"), "source_sha256": sources()}
    if frozen["recording"]["sha256"] != fields.BAG_SHA256 or frozen["native"] != receipt["inputs"]["native"]:
        raise ValueError("recording/native differs from the successful functional audit")
    output.mkdir(parents=True, exist_ok=False)
    fields.save(output / "protocol.json", {"schema": "cloudanalyzer.flatwall_imu_deskew_timing.v1", "inputs": frozen,
                "scan_indices": SCAN_INDICES, "rayon_threads_requested": 4, "warmup_pairs": 3, "measured_pairs": 7,
                "order": "alternate serial/parallel and parallel/serial", "metric": "elapsed nanoseconds per eight-scan Python deskew API batch, including input/output copies",
                "exclusions": "bag I/O and trajectory construction; no builds, tests, source scans or hashing inside timing windows",
                "state": "same declared identity pose, zero velocity, world gravity [0,0,-9.80665], zero bias, acceleration scale 9.80665 and identity extrinsic as the functional controller",
                "scope": "Prewarmed API on this host and fixed real inputs; no end-to-end SLAM, GPU or trajectory-accuracy result."})
    reader = core.BagReader(str(bag))
    series = reader.imu_series("/livox/imu", acceleration_scale=9.80665)
    imu_times = series.timestamps_ns()
    trajectory = series.trajectory(int(imu_times[0]), int(imu_times[-1]), np.eye(4), np.zeros(3), gravity=(0., 0., -9.80665), bias=(0.,) * 6)
    prepared = []
    for index, scan in enumerate(reader.scans("/livox/points", raw_time_field="t")):
        if index in SCAN_INDICES:
            stamps = scan.point_times_ns(unit="ns", origin="header")
            prepared.append((index, scan.positions(), stamps, int(stamps.max())))
    if [p[0] for p in prepared] != SCAN_INDICES or any(len(p[1]) != 24_000 for p in prepared):
        raise ValueError("unexpected sample coverage")
    extrinsic = np.eye(4)

    def input_digests():
        return [hashlib.sha256(points.astype("<f8").tobytes() + stamps.astype("<i8").tobytes()).hexdigest()
                for index, points, stamps, reference in prepared]

    def outputs(parallel):
        return [hashlib.sha256(trajectory.deskew(points, stamps, reference, imu_from_lidar=extrinsic, parallel=parallel).astype("<f8").tobytes()).hexdigest()
                for index, points, stamps, reference in prepared]

    def batch(parallel):
        result = None
        for index, points, stamps, reference in prepared:
            result = trajectory.deskew(points, stamps, reference, imu_from_lidar=extrinsic, parallel=parallel)
        return result

    input_hashes = input_digests()
    expected = outputs(False)
    if outputs(True) != expected:
        raise ValueError("serial/parallel outputs differ before timing")
    for _ in range(3):
        batch(False)
        batch(True)
    samples = []
    for pair in range(7):
        row = {"pair": pair, "order": ["serial", "parallel"] if pair % 2 == 0 else ["parallel", "serial"]}
        for mode in row["order"]:
            start = time.perf_counter_ns()
            result = batch(mode == "parallel")
            elapsed = time.perf_counter_ns() - start
            row[mode + "_ns"] = elapsed
            # Result checks happen after the timed call, never inside its window.
            if hashlib.sha256(result.astype("<f8").tobytes()).hexdigest() != expected[-1]:
                raise ValueError("timed output differs")
        samples.append(row)
    if outputs(False) != expected or outputs(True) != expected or input_digests() != input_hashes:
        raise ValueError("results or caller inputs changed")
    if sources() != frozen["source_sha256"] or fields.artifact(bag) != frozen["recording"] or fields.artifact(Path(native.__file__)) != frozen["native"] or fields.artifact(Path(__file__)) != frozen["controller"] or fields.artifact(Path(fields.__file__)) != frozen["helper"] or fields.artifact(audit / "protocol.json") != frozen["functional_protocol"] or fields.artifact(audit / "receipt.json") != frozen["functional_receipt"]:
        raise ValueError("frozen inputs changed")
    serial = [r["serial_ns"] for r in samples]
    parallel = [r["parallel_ns"] for r in samples]
    summary = {"scans_per_batch": len(prepared), "points_per_batch": sum(len(p[1]) for p in prepared),
               "serial_median_ns": statistics.median(serial), "parallel_median_ns": statistics.median(parallel),
               "serial_range_ns": [min(serial), max(serial)], "parallel_range_ns": [min(parallel), max(parallel)],
               "median_ratio_serial_over_parallel": statistics.median(serial) / statistics.median(parallel),
               "serial_parallel_outputs_match": True, "caller_inputs_unchanged": True, "scope": "Eight fixed real scans, prewarmed Python API and four requested CPU workers; this host only."}
    fields.save(output / "samples.json", {"input_sha256": input_hashes, "output_sha256": expected, "pairs": samples})
    fields.save(output / "receipt.json", {"protocol": fields.artifact(output / "protocol.json"), "samples": fields.artifact(output / "samples.json"), "results": summary})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(measure(args.bag, args.audit, args.out), indent=2))


if __name__ == "__main__":
    main()
