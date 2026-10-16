#!/usr/bin/env python3
"""Audit declared ns-offset/header-origin conversion without clock calibration."""
from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import re

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

import cloudanalyzer_core as core
import cloudanalyzer_core._core as native
import audit_flatwall_raw_point_time as raw_audit


def audit(bag, output, *, baseline=False, comparison=None, controls=()):
    output = output.resolve()
    comparison = comparison.resolve() if comparison is not None else None
    if output.exists() or (comparison is not None and comparison in output.parents):
        raise ValueError("use a fresh output outside the frozen baseline")
    recordings = [("flatwall", bag.resolve(), "/livox/points")] + [
        (label, path.resolve(), topic) for label, path, topic in controls]
    labels = [label for label, path, topic in recordings]
    if len(set(labels)) != len(labels) or any(not re.fullmatch(r"[a-z0-9-]+", label) for label in labels):
        raise ValueError("unique lowercase recording labels required")

    def inputs():
        result = {
            "recordings": {label: {**raw_audit.artifact(path), "topic": topic} for label, path, topic in recordings},
            "controller": raw_audit.artifact(Path(__file__)),
            "field_audit_helpers": raw_audit.artifact(Path(raw_audit.__file__)),
            "native": raw_audit.artifact(Path(native.__file__)),
            "numpy_version": version("numpy"), "rosbags_version": version("rosbags"),
            "independent_reader": raw_audit.artifact(Path(inspect.getfile(AnyReader))),
            "standard_type_sources": [raw_audit.artifact(Path(import_module("rosbags.typesys.stores." + name).__file__))
                                      for name in ["ros1_noetic", "ros2_humble"]],
        }
        if comparison is not None:
            result["comparison_protocol"] = raw_audit.artifact(comparison / "protocol.json")
            result["comparison_receipt"] = raw_audit.artifact(comparison / "receipt.json")
        return result

    frozen = inputs()
    if frozen["recordings"]["flatwall"]["sha256"] != raw_audit.BAG_SHA256:
        raise ValueError("expected the original flatwall_01.bag from Zenodo record 7641866")
    prior = None
    if comparison is not None:
        prior = json.loads((comparison / "receipt.json").read_text(encoding="utf-8"))
        if not prior["baseline"] or prior["protocol"] != frozen["comparison_protocol"]:
            raise ValueError("invalid or changed baseline protocol")
        for key in ["recordings", "controller", "field_audit_helpers", "numpy_version", "rosbags_version",
                    "independent_reader", "standard_type_sources"]:
            if prior["inputs"][key] != frozen[key]:
                raise ValueError(f"baseline input changed: {key}")
    protocol = {
        "schema": "cloudanalyzer.flatwall_point_times_ns.v1", "inputs": frozen, "baseline": baseline,
        "declared_policy": {"field": "t", "datatype": 6, "unit": "ns", "origin": "PointCloud2 integer Header",
                            "rounding": "nearest integer ns, ties to even, before integer origin addition"},
        "baseline_method": "old raw_time() checked UINT32 to int64, then integer Header addition",
        "new_method": "point_times_ns(unit=ns, origin=header); no manual fallback",
        "independent_method": "standard ROS1_NOETIC UINT32 t plus Python integer Header; standard ROS2_HUMBLE control fields",
        "controls": "Separately count float64-before-header-rounding errors; NCLT has no time field",
        "scope": "Declared conversion and IMU coverage only. No unit/origin inference, clock calibration, reference archive access, deskew, trajectory or runtime claim.",
    }
    output.mkdir(parents=True, exist_ok=False)
    raw_audit.save(output / "protocol.json", protocol)
    imu = core.BagReader(str(bag)).imu_series("/livox/imu").timestamps_ns()
    if len(imu) != 3755 or not np.all(np.diff(imu) > 0):
        raise ValueError("unexpected IMU coverage")
    results = {}
    for label, path, topic in recordings:
        timing = label == "flatwall"
        reader = core.BagReader(str(path))
        ordinary = iter(reader.scans(topic))
        retained = iter(reader.scans(topic, raw_time_field="t")) if timing else None
        legacy_digest, raw_digest, ns_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
        scans = points = decreasing = before = after = fully_bracketed = naive_mismatches = 0
        naive_min = naive_max = 0
        with AnyReader([path]) as independent:
            connections = [c for c in independent.connections if c.topic == topic]
            if len(connections) != 1 or connections[0].msgtype != "sensor_msgs/msg/PointCloud2":
                raise ValueError("expected one standard PointCloud2 connection")
            standard = get_typestore(Stores.ROS2_HUMBLE if independent.is2 else Stores.ROS1_NOETIC)
            for connection, recorded, payload in independent.messages(connections=connections):
                message = (standard.deserialize_cdr(payload, connection.msgtype) if independent.is2 else
                           standard.deserialize_ros1(payload, connection.msgtype))
                expected, raw_values, layout = raw_audit.reference_fields(message, timing=timing)
                scan = next(ordinary)
                digest = raw_audit.fields_digest(raw_audit.scan_fields(scan))
                if digest != raw_audit.fields_digest(expected) or scan.raw_time() is not None:
                    raise ValueError("default legacy fields changed")
                legacy_digest.update(bytes.fromhex(digest))
                scans += 1
                points += len(scan)
                if timing:
                    extra = next(retained)
                    if raw_audit.fields_digest(raw_audit.scan_fields(extra)) != digest:
                        raise ValueError("retained legacy fields changed")
                    if extra.raw_time_field != "t" or extra.raw_time_datatype != 6:
                        raise ValueError("raw source metadata changed")
                    np.testing.assert_array_equal(extra.raw_time().view(np.uint64), raw_values.view(np.uint64))
                    field = next(f for f in message.fields if f.name == "t")
                    if field.datatype != 6 or field.count != 1 or len(scan) != message.width * message.height:
                        raise ValueError("expected all original finite UINT32-t flat-wall points")
                    source = np.ndarray((message.height, message.width), dtype=">u4" if message.is_bigendian else "<u4",
                                        buffer=message.data, offset=field.offset,
                                        strides=(message.row_step, message.point_step)).reshape(-1)
                    header = expected["stamp_ns"]
                    expected_ns = np.fromiter((header + int(value) for value in source), dtype=np.int64, count=len(source))
                    if baseline:
                        values = extra.raw_time()
                        if not np.isfinite(values).all() or not np.all((values >= 0) & (values <= 2**32 - 1)) or not np.all(values == np.trunc(values)):
                            raise ValueError("invalid UINT32 values in baseline")
                        if not np.iinfo(np.int64).min <= header <= np.iinfo(np.int64).max - int(values.max()):
                            raise ValueError("baseline integer addition would overflow")
                        actual_ns = values.astype(np.int64) + np.int64(header)
                    else:
                        actual_ns = extra.point_times_ns(unit="ns", origin="header")
                    np.testing.assert_array_equal(actual_ns, expected_ns)
                    if actual_ns.dtype != np.int64:
                        raise ValueError("expected int64 timestamps")
                    np.testing.assert_array_equal(extra.raw_time().view(np.uint64), raw_values.view(np.uint64))
                    if raw_audit.fields_digest(raw_audit.scan_fields(extra)) != digest:
                        raise ValueError("timestamp conversion changed legacy fields")
                    raw_digest.update(raw_values.astype("<f8").tobytes())
                    ns_digest.update(actual_ns.astype("<i8").tobytes())
                    decreasing += int(np.sum(np.diff(actual_ns) < 0))
                    before += int(np.sum(actual_ns < imu[0]))
                    after += int(np.sum(actual_ns > imu[-1]))
                    fully_bracketed += int(actual_ns.min() >= imu[0] and actual_ns.max() <= imu[-1])
                    errors = np.rint(raw_values + float(header)).astype(np.int64) - expected_ns
                    naive_mismatches += int(np.sum(errors != 0))
                    naive_min, naive_max = min(naive_min, int(errors.min())), max(naive_max, int(errors.max()))
                elif not baseline:
                    if any(f.name == "t" for f in message.fields):
                        raise ValueError("control unexpectedly contains t")
                    try:
                        scan.point_times_ns(unit="ns", origin="header")
                    except ValueError as exc:
                        if "explicitly retained raw_time_field" not in str(exc):
                            raise
                    else:
                        raise ValueError("control fabricated point timestamps")
        if next(ordinary, None) is not None or (retained is not None and next(retained, None) is not None):
            raise ValueError("extra native scans")
        result = {"scans": scans, "finite_points": points, "legacy_fields_sha256": legacy_digest.hexdigest()}
        if timing:
            result.update(raw_time_values_sha256=raw_digest.hexdigest(), point_times_ns_sha256=ns_digest.hexdigest(),
                          decreasing_point_time_steps=decreasing, points_before_imu=before, points_after_imu=after,
                          fully_bracketed_scans=fully_bracketed, float_addition_mismatches=naive_mismatches,
                          float_min_error_ns=naive_min, float_max_error_ns=naive_max)
        if prior is not None and result != prior["results"][label]:
            raise ValueError("old/new raw, legacy or point timestamps differ")
        results[label] = result
    flat = results["flatwall"]
    if flat["scans"] != 186 or flat["finite_points"] != 4_464_000 or inputs() != frozen:
        raise ValueError("input coverage/provenance changed")
    receipt = {**protocol, "protocol": raw_audit.artifact(output / "protocol.json"), "results": results,
               "integer_timestamps_match_independent": True, "legacy_fields_match_independent": True,
               "legacy_fields_match_old_native": comparison is not None, "ground_truth_archive_read": False}
    raw_audit.save(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--comparison", type=Path)
    parser.add_argument("--control", nargs=3, action="append", default=[], metavar=("LABEL", "BAG", "TOPIC"))
    args = parser.parse_args()
    result = audit(args.bag, args.out, baseline=args.baseline, comparison=args.comparison,
                   controls=[(label, Path(path), topic) for label, path, topic in args.control])
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
