#!/usr/bin/env python3
"""Verify published input accounting; raw-bag comparisons need the original bags."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def increasing(values):
    if (values.ndim != 1 or values.dtype.kind != "i" or values.dtype.itemsize != 8 or not len(values)
            or int(values[0]) < 0 or any(int(b) <= int(a) for a, b in zip(values, values[1:]))):
        raise ValueError("original timestamps must be increasing int64")


def coverage(times, start, end):
    if start > end:
        raise ValueError("reversed point interval")
    if start < times[0] or end > times[-1]:
        return dict(status="outside_imu_range", left_index=None, right_index=None, maximum_gap_ns=None)
    first, last = bisect_right(times, start) - 1, bisect_left(times, end)
    gap = max((times[i+1] - times[i] for i in range(first, last)), default=0)
    return dict(status="imu_gap" if gap > 50_000_000 else "covered", left_index=first, right_index=last, maximum_gap_ns=gap)


def timing(values):
    values = np.asarray(values, dtype=np.int64)
    increasing(values)
    delta = [int(b) - int(a) for a, b in zip(values, values[1:])]
    return dict(count=len(values), first_ns=int(values[0]), last_ns=int(values[-1]),
                duration_ns=int(values[-1]) - int(values[0]), minimum_delta_ns=min(delta) if delta else None,
                median_delta_ns=float(np.median(delta)) if delta else None, maximum_delta_ns=max(delta) if delta else None,
                gaps_over_50ms=sum(d > 50_000_000 for d in delta))


def close(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError("summary structure mismatch")
        for key in expected:
            close(actual[key], expected[key])
    elif isinstance(expected, list):
        if actual != expected:
            raise ValueError("original scan identity mismatch")
    elif isinstance(expected, float):
        if not np.isfinite(actual) or not np.isclose(actual, expected, rtol=2e-13, atol=1e-14):
            raise ValueError("numeric input summary mismatch")
    elif actual != expected:
        raise ValueError("input accounting mismatch")


def verify(folder):
    folder = folder.resolve()
    manifest = read(folder / "manifest.json")
    for name, digest in manifest["files_sha256"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or sha(path) != digest:
            raise ValueError("published file hash mismatch")
    root = folder.parents[2]
    for name, digest in manifest["checkout_sources_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha(path) != digest:
            raise ValueError("checkout source footprint changed")
    protocol, receipt = read(folder / "protocol.json"), read(folder / "inspection-receipt.json")
    acquired = read(folder / "recording-acquisition.json")["recording"]
    frozen = read(folder / "controller-freeze.json")
    if (sha(folder / "protocol.json") != receipt["protocol_sha256"]
            or sha(folder / "controller-freeze.json") != receipt["controller_freeze_sha256"]
            or receipt["reference_archive_read_in_this_stage"] is not False
            or protocol["physical_calibration_certified"] is not False
            or set(protocol["recordings"]) != {"flatwall_01", "flatwall_02"}):
        raise ValueError("inspection boundary changed")
    if (acquired["bytes"] != 135975445 or acquired["md5"] != "234a60eef9d90ec069a2cbe6042f6cf9"
            or frozen["files_sha256"][protocol["recordings"]["flatwall_02"]] != acquired["sha256"]
            or frozen["files_sha256"][protocol["recordings"]["flatwall_01"]] != "4721e667166e11b948aea83b221f99ca0fec21fabdce5b292662a4b442efbab6"):
        raise ValueError("original recording identity changed")
    results = {}
    for name in protocol["recordings"]:
        scans = read(folder / name / "scans.json")
        meta = read(folder / name / "metadata.json")
        if not scans or [r["scan_index"] for r in scans] != list(range(len(scans))):
            raise ValueError("incomplete original scan identities")
        with np.load(folder / name / "imu.npz", allow_pickle=False) as data:
            imu = {k: data[k].copy() for k in data.files}
        increasing(imu["timestamps_ns"])
        ns = [int(t) for t in imu["timestamps_ns"]]
        if (imu["recorded_ns"].dtype.kind != "i" or imu["recorded_ns"].shape != (len(ns),)
                or imu["sequence"].shape != (len(ns),)
                or any(imu[k].shape != (len(ns), 3) or not np.isfinite(imu[k]).all()
                       for k in ["raw_linear_acceleration", "raw_angular_velocity"])):
            raise ValueError("invalid original IMU metadata")
        origins, digest = {}, hashlib.sha256()
        for row in scans:
            if (not 0 < row["finite_points"] <= row["raw_points"]
                    or row["header_minus_timebase_ns"] != row["header_ns"] - row["custom_timebase_ns"]
                    or not 0 <= row["offset_min_ns"] <= row["offset_max_ns"] < 2 ** 32):
                raise ValueError("point count/timebase identity mismatch")
            digest.update(row["scan_index"].to_bytes(4, "little") + bytes.fromhex(row["points_and_offsets_sha256"]))
            for origin, key in [("header", "header_ns"), ("timebase", "custom_timebase_ns")]:
                if row[origin + "_coverage"] != coverage(ns, row[key]+row["offset_min_ns"], row[key]+row["offset_max_ns"]):
                    raise ValueError("IMU bracket/gap accounting mismatch")
        for origin in ["header", "timebase"]:
            origins[origin] = [r["scan_index"] for r in scans if r[origin+"_coverage"]["status"] == "covered"]
        acceleration = np.linalg.norm(imu["raw_linear_acceleration"], axis=1)
        gyro = np.linalg.norm(imu["raw_angular_velocity"], axis=1)
        delta = [r["header_minus_timebase_ns"] for r in scans]
        result = dict(scans=len(scans), raw_points=sum(r["raw_points"] for r in scans),
            finite_points=sum(r["finite_points"] for r in scans), imu=timing(imu["timestamps_ns"]),
            scan_headers=timing([r["header_ns"] for r in scans]),
            point_time_decreases_retained=sum(r["point_time_decreases"] for r in scans),
            header_minus_timebase_ns=dict(min=min(delta), max=max(delta), nonzero=sum(d != 0 for d in delta)),
            coverage_counts={o: dict(Counter(r[o+"_coverage"]["status"] for r in scans)) for o in origins},
            covered_scan_indices=origins,
            raw_acceleration_norm=dict(min=float(acceleration.min()), median=float(np.median(acceleration)), max=float(acceleration.max())),
            raw_angular_velocity_norm=dict(min=float(gyro.min()), median=float(np.median(gyro)), max=float(gyro.max())))
        close(read(folder / name / "summary.json"), result)
        close(receipt["results"][name], result)
        close(manifest["recordings"][name], {k: result[k] for k in ["scans", "finite_points", "imu", "header_minus_timebase_ns", "coverage_counts"]})
        if (meta["complete_points_and_offsets_sha256"] != digest.hexdigest()
                or not meta["all_custom_points_and_offsets_match_pointcloud2"]
                or not meta["native_raw_imu_and_points_match_independent"]
                or meta["recording_producer_build_identified"] is not False
                or meta["tf_or_calibration_topics_present"] is not False
                or meta["reference_archive_read_in_this_stage"] is not False
                or Counter((c["topic"], c["messages"]) for c in meta["connections"]) != Counter([
                    ("/livox/points", len(scans)), ("/livox/lidar", len(scans)), ("/livox/imu", len(ns))])):
            raise ValueError("raw inspection receipt/connection accounting mismatch")
        for file in ["scans.json", "imu.npz", "metadata.json", "summary.json"]:
            if receipt["output_files_sha256"][name+"/"+file] != sha(folder / name / file):
                raise ValueError("original inspection receipt file hash mismatch")
        results[name] = {k: result[k] for k in ["scans", "finite_points", "header_minus_timebase_ns", "coverage_counts"]}
    evidence = read(folder / "provenance.json")
    if (evidence["physical_calibration_certified"] is not False
            or evidence["reference_archive_read_in_this_stage"] is not False
            or evidence["author_setup"]["acc_scale"] != 9.80665
            or evidence["author_setup"]["imu_time_offset"] != 0.
            or evidence["author_setup"]["T_lidar_imu"] != [0, 0, 0, 0, 0, 0, 1]):
        raise ValueError("physical provenance boundary changed")
    return dict(recordings=results, physical_calibration_certified=False,
                scope="Published exact input metadata, raw IMU arrays, time brackets and receipt consistency. Raw point/cloud parity and producer/calibration claims are not independently certified by this packet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(verify(parser.parse_args().packet), indent=2))
