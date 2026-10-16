#!/usr/bin/env python3
"""Compare explicit raw point times with independent standard ROS decoding.

Input/precision validation only: no units/origin inference, deskew or trajectory.
"""
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

BAG_SHA256 = "4721e667166e11b948aea83b221f99ca0fec21fabdce5b292662a4b442efbab6"
TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def artifact(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def fields_digest(fields):
    digest = hashlib.sha256()
    for key, value in sorted(fields.items()):
        if isinstance(value, np.ndarray):
            array = value.astype(value.dtype.newbyteorder("<"), copy=False)
            metadata, payload = [key, array.dtype.str, array.shape], array.tobytes()
        else:
            metadata, payload = [key, "json"], json.dumps(value, sort_keys=True).encode()
        digest.update(json.dumps(metadata).encode() + b"\0")
        digest.update(len(payload).to_bytes(8, "little") + payload)
    return digest.hexdigest()


def scan_fields(scan):
    return {"stamp_ns": scan.stamp_ns, "stamp": scan.stamp, "frame_id": scan.frame_id,
            "positions": scan.positions(), "intensity": scan.intensity(), "time": scan.time()}


def reference_fields(message, *, timing):
    fields = {f.name: f for f in message.fields}
    order = ">" if message.is_bigendian else "<"

    def column(name):
        field = fields[name]
        if field.count != 1:
            raise ValueError("reference requires scalar fields")
        return np.ndarray((int(message.height), int(message.width)), dtype=order + TYPES[field.datatype],
                          buffer=message.data, offset=field.offset,
                          strides=(int(message.row_step), int(message.point_step))).reshape(-1)

    xyz = np.column_stack([column(name) for name in "xyz"]).astype("<f8")
    finite = np.isfinite(xyz).all(axis=1)
    intensity = column("intensity")[finite].astype("<f4")
    stamp = message.header.stamp
    raw_time = column("t")[finite].astype("<f8") if timing else None
    fractions = None
    if raw_time is not None and len(raw_time) and raw_time.max() > raw_time.min():
        fractions = ((raw_time - raw_time.min()) / (raw_time.max() - raw_time.min())).astype("<f4")
    return {"stamp_ns": int(stamp.sec) * 10**9 + int(stamp.nanosec),
            "stamp": int(stamp.sec) + int(stamp.nanosec) * 1e-9,
            "frame_id": message.header.frame_id, "positions": xyz[finite], "intensity": intensity,
            "time": fractions}, raw_time, [[f.name, int(f.datatype), int(f.offset), int(f.count)] for f in message.fields]


def audit(bag, output, *, baseline, comparison, controls):
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if comparison is not None and output.resolve().is_relative_to(comparison.resolve()):
        raise ValueError("write outside the frozen baseline directory")
    recordings = [("flatwall", bag, "/livox/points"), *controls]
    labels = [label for label, path, topic in recordings]
    if len(set(labels)) != len(labels) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", label) for label in labels):
        raise ValueError("recordings need unique simple labels")

    def inputs():
        result = {"recordings": {label: {**artifact(path), "topic": topic} for label, path, topic in recordings},
                  "controller": artifact(Path(__file__)), "native": artifact(Path(native.__file__)),
                  "numpy_version": version("numpy"), "rosbags_version": version("rosbags"),
                  "independent_reader": artifact(Path(inspect.getfile(AnyReader))),
                  "standard_type_sources": [artifact(Path(import_module("rosbags.typesys.stores." + name).__file__))
                                            for name in ["ros1_noetic", "ros2_humble"]]}
        if comparison is not None:
            result["comparison_receipt"] = artifact(comparison / "receipt.json")
            result["comparison_protocol"] = artifact(comparison / "protocol.json")
        return result

    frozen = inputs()
    if frozen["recordings"]["flatwall"]["sha256"] != BAG_SHA256:
        raise ValueError("expected the original flatwall_01.bag from Zenodo record 7641866")
    prior = None
    if comparison is not None:
        prior = json.loads((comparison / "receipt.json").read_text(encoding="utf-8"))
        if prior["protocol"] != frozen["comparison_protocol"] or not prior["baseline"]:
            raise ValueError("baseline protocol changed or is not a baseline")
        for key in ["recordings", "controller", "independent_reader", "standard_type_sources"]:
            if prior["inputs"][key] != frozen[key]:
                raise ValueError(f"baseline input changed: {key}")
    protocol = {"schema": "cloudanalyzer.flatwall_raw_point_time.v1", "inputs": frozen, "baseline": baseline,
                "requested_field": "t for flatwall; controls have no t field and retain None timing",
                "independent_decoder": "Standard ROS1_NOETIC/ROS2_HUMBLE PointCloud2, explicitly supplied even when the NCLT MCAP schema text is empty",
                "scope": "Every finite point, raw timing and legacy field digest. No unit/origin inference, reference archive access, clock shift, deskew, trajectory or runtime claim."}
    output.mkdir(parents=True, exist_ok=False)
    save(output / "protocol.json", protocol)
    results = {}
    for label, path, topic in recordings:
        timing = label == "flatwall"
        reader = core.BagReader(str(path))
        ordinary = iter(reader.scans(topic))
        retained = None if baseline or not timing else iter(reader.scans(topic, raw_time_field="t"))
        combined, raw_combined = hashlib.sha256(), hashlib.sha256()
        count, points, decreasing, duplicates = 0, 0, 0, 0
        layouts = []
        with AnyReader([path]) as independent:
            connections = [c for c in independent.connections if c.topic == topic]
            if len(connections) != 1 or connections[0].msgtype != "sensor_msgs/msg/PointCloud2":
                raise ValueError("expected one standard PointCloud2 connection")
            standard = get_typestore(Stores.ROS2_HUMBLE if independent.is2 else Stores.ROS1_NOETIC)
            for connection, recorded, payload in independent.messages(connections=connections):
                message = (standard.deserialize_cdr(payload, connection.msgtype) if independent.is2 else
                           standard.deserialize_ros1(payload, connection.msgtype))
                expected, values, layout = reference_fields(message, timing=timing)
                if not timing and any(f.name == "t" for f in message.fields):
                    raise ValueError("expected a control without a t field")
                if layout not in layouts:
                    layouts.append(layout)
                scan = next(ordinary)
                digest = fields_digest(scan_fields(scan))
                if digest != fields_digest(expected):
                    raise ValueError("native legacy fields differ from independent decoding")
                combined.update(bytes.fromhex(digest))
                count += 1
                points += len(scan)
                if not baseline and (scan.raw_time() is not None or scan.raw_time_field is not None or scan.raw_time_datatype is not None):
                    raise ValueError("default decoding retained raw times")
                if timing:
                    raw_combined.update(scan.stamp_ns.to_bytes(8, "little", signed=True))
                    raw_combined.update(values.tobytes())
                    decreasing += int(np.sum(np.diff(values) < 0))
                    duplicates += int(np.sum(np.diff(values) == 0))
                if retained is not None:
                    extra = next(retained)
                    if fields_digest(scan_fields(extra)) != digest:
                        raise ValueError("opt-in decoding changed legacy fields")
                    if extra.raw_time_field != "t" or extra.raw_time_datatype != 6:
                        raise ValueError("raw field metadata changed")
                    np.testing.assert_array_equal(extra.raw_time().view(np.uint64), values.view(np.uint64))
        if next(ordinary, None) is not None or (retained is not None and next(retained, None) is not None):
            raise ValueError("extra native scans")
        if not baseline and not timing:
            try:
                next(reader.scans(topic, raw_time_field="t"))
            except ValueError as exc:
                if "raw time field not found" not in str(exc):
                    raise
            else:
                raise ValueError("missing control timing field was silently fabricated")
        result = {"scans": count, "finite_points": points, "layouts": layouts,
                  "legacy_fields_sha256": combined.hexdigest(),
                  "raw_time_values_with_headers_sha256": raw_combined.hexdigest() if timing else None,
                  "decreasing_point_time_steps": decreasing, "duplicate_point_time_steps": duplicates}
        if prior is not None and result != prior["results"][label]:
            raise ValueError("raw source or legacy output changed from the old native build")
        results[label] = result
    if results["flatwall"]["scans"] != 186 or results["flatwall"]["finite_points"] != 4_464_000 or inputs() != frozen:
        raise ValueError("input coverage/provenance changed")
    receipt = {**protocol, "protocol": artifact(output / "protocol.json"), "results": results,
               "legacy_fields_match_independent": True, "raw_time_matches_independent": not baseline,
               "legacy_fields_match_old_native": comparison is not None, "ground_truth_archive_read": False}
    save(output / "receipt.json", receipt)
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
