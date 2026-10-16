#!/usr/bin/env python3
"""Compare every flat-wall IMU payload with an independent ROS 1 index reader.

This is an input-order audit. It reads no reference trajectory, runs no SLAM,
does not scale measurements and makes no runtime or accuracy claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import struct

import numpy as np
from rosbags.rosbag1 import Reader

import cloudanalyzer_core as core
import cloudanalyzer_core._core as native

BAG_SHA256 = "4721e667166e11b948aea83b221f99ca0fec21fabdce5b292662a4b442efbab6"


def artifact(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def scan_digest(path):
    digest = hashlib.sha256()
    counts, times = [], []
    for scan in core.BagReader(str(path)).scans():
        counts.append(len(scan))
        times.append(scan.stamp_ns)
        frame = scan.frame_id.encode("utf-8")
        digest.update(struct.pack("<qI", scan.stamp_ns, len(frame)))
        digest.update(frame)
        for array, dtype in [(scan.positions(), "<f8"), (scan.intensity(), "<f4"), (scan.time(), "<f4")]:
            digest.update(struct.pack("<q", -1 if array is None else array.size))
            if array is not None:
                digest.update(array.astype(dtype, copy=False).tobytes())
    return {"scans": len(counts), "finite_points": sum(counts), "first_header_ns": times[0],
            "last_header_ns": times[-1], "decoded_sensor_fields_sha256": digest.hexdigest()}


def time_summary(times):
    delta = np.diff(times)
    return {"messages": len(times), "first_ns": int(times[0]), "last_ns": int(times[-1]),
            "non_increasing": int(np.sum(delta <= 0)), "median_delta_ns": float(np.median(delta)),
            "max_delta_ns": int(delta.max()), "gaps_over_50ms": int(np.sum(delta > 50_000_000))}


def audit(bag, output, *, expected_order, comparison=None, scan_controls=()):
    if output.exists():
        raise FileExistsError(f"audit output already exists: {output}")
    if comparison is not None and output.resolve().is_relative_to(comparison.resolve()):
        raise ValueError("write outside the frozen comparison audit")
    recordings = [bag, *scan_controls]
    if len({p.name for p in recordings}) != len(recordings):
        raise ValueError("recordings need distinct names")

    def inputs():
        result = {"recordings": [artifact(p) for p in recordings], "controller": artifact(Path(__file__)),
                  "native": artifact(Path(native.__file__)), "rosbags_version": version("rosbags"),
                  "independent_reader": artifact(Path(inspect.getfile(Reader)))}
        if comparison is not None:
            result["comparison"] = artifact(comparison / "receipt.json")
        return result

    frozen = inputs()
    if frozen["recordings"][0]["sha256"] != BAG_SHA256:
        raise ValueError("expected the unmodified flatwall_01.bag from Zenodo record 7641866")
    protocol = {"schema": "cloudanalyzer.flatwall_imu_order.v1", "inputs": frozen,
                "expected_order": expected_order, "topic": "/livox/imu",
                "reference_policy": "Single-connection independent ROS 1 index order, checked against exact header times; GT archive not read",
                "scope": "All IMU payload bytes and sensor-header order, plus decoded scan-field parity. No SLAM, scaling, deskew, score or timing."}
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    with Reader(bag) as independent:
        connections = [c for c in independent.connections if c.topic == "/livox/imu"]
        if len(connections) != 1:
            raise ValueError("expected one physical IMU connection")
        reference = [(timestamp, bytes(payload)) for connection, timestamp, payload in independent.messages(connections)]
    reader = core.BagReader(str(bag))
    messages = list(reader.messages(["/livox/imu"]))
    reference_payloads = [payload for timestamp, payload in reference]
    native_payloads = [payload for topic, timestamp, encoding, payload in messages]
    if len(reference) != 3755 or len(messages) != len(reference):
        raise ValueError("IMU coverage changed")
    if Counter(native_payloads) != Counter(reference_payloads):
        raise ValueError("IMU payload bytes changed or messages were lost/duplicated")
    hashes = lambda payloads: [hashlib.sha256(payload).digest() for payload in payloads]
    reference_hashes, native_hashes = hashes(reference_payloads), hashes(native_payloads)

    def header_times(payloads):
        return np.array([int(sec) * 10 ** 9 + int(ns) for sequence, sec, ns in
                         (struct.unpack_from("<III", payload) for payload in payloads)], dtype=np.int64)

    expected_times = header_times(reference_payloads)
    actual_times = header_times(native_payloads)
    if np.any(np.diff(expected_times) <= 0):
        raise ValueError("independent source sensor headers are not strictly increasing")
    matches = native_payloads == reference_payloads
    if matches != (expected_order == "matched"):
        raise ValueError("unexpected native/reference order outcome")
    np.testing.assert_array_equal(reader.imu_raw("/livox/imu")["timestamps_ns"], actual_times)
    try:
        series = reader.imu_series("/livox/imu")
    except ValueError as exc:
        if matches:
            raise
        series_status = {"accepted": False, "error": str(exc)}
    else:
        np.testing.assert_array_equal(series.timestamps_ns(), expected_times)
        series_status = {"accepted": True, "messages": len(series)}
    scans = {p.name: scan_digest(p) for p in recordings}
    result = {"native_matches_reference": matches, "all_payload_bytes_preserved": True,
              "positions_differing_from_reference": sum(a != b for a, b in zip(native_payloads, reference_payloads)),
              "reference": time_summary(expected_times), "native": time_summary(actual_times), "imu_series": series_status,
              "reference_payload_order_sha256": hashlib.sha256(b"".join(reference_hashes)).hexdigest(),
              "native_payload_order_sha256": hashlib.sha256(b"".join(native_hashes)).hexdigest(),
              "payload_multiset_sha256": hashlib.sha256(b"".join(sorted(reference_hashes))).hexdigest(),
              "scans": scans, "ground_truth_archive_read": False}
    if comparison is not None:
        before = json.loads((comparison / "receipt.json").read_text(encoding="utf-8"))["results"]
        if before["scans"] != scans or before["payload_multiset_sha256"] != result["payload_multiset_sha256"]:
            raise ValueError("sensor bytes changed relative to the comparison audit")
    if inputs() != frozen:
        raise ValueError("audit input changed")
    receipt = {**protocol, "results": result, "protocol": artifact(output / "protocol.json")}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-order", choices=["matched", "mismatched"], required=True)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--scan-control", type=Path, action="append", default=[])
    args = parser.parse_args()
    result = audit(args.bag, args.out, expected_order=args.expected_order,
                   comparison=args.comparison, scan_controls=args.scan_control)
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
