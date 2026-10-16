#!/usr/bin/env python3
"""Audit explicit IMU units against a prior native build and SciPy dynamics.

Input/preintegration checks only: no trajectory, reference poses, deskew or timing.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation
import scipy.spatial.transform._rotation as rotation_binary

import cloudanalyzer_core as core
import cloudanalyzer_core._core as native

BAG_SHA256 = "4721e667166e11b948aea83b221f99ca0fec21fabdce5b292662a4b442efbab6"
PARAMS = {"gyro_noise_density": .02, "accel_noise_density": .15, "max_gap_s": .05}
SCALE = 9.80665


def artifact(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def fields_digest(fields):
    digest = hashlib.sha256()
    for key, value in sorted(fields.items()):
        if isinstance(value, np.ndarray):
            array = value.astype(value.dtype.newbyteorder("<"), copy=False)
            metadata = [key, array.dtype.str, array.shape]
            payload = array.tobytes()
        else:
            metadata = [key, "json"]
            payload = json.dumps(value, ensure_ascii=True, sort_keys=True).encode()
        digest.update(json.dumps(metadata).encode() + b"\0")
        digest.update(len(payload).to_bytes(8, "little") + payload)
    return digest.hexdigest()


def intervals(times):
    first, last = int(times[0]), int(times[-1])
    if last - first < 300_000_000:
        raise ValueError("recording must cover the predeclared 100 ms windows")
    middle = first + (last - first) // 2 + 17
    return [(first, last), (first + 17, first + 100_000_017),
            (middle, middle + 100_000_000), (last - 100_000_059, last - 59)]


def nominal_oracle(raw, start, end):
    rotation, velocity, position = np.eye(3), np.zeros(3), np.zeros(3)
    times = raw["timestamps_ns"]
    gyro, accel = raw["angular_velocity_rad_s"], raw["linear_acceleration_m_s2"] * SCALE
    for i in range(len(times) - 1):
        left, right = max(start, int(times[i])), min(end, int(times[i + 1]))
        if right <= left:
            continue
        dt = (right - left) * 1e-9
        force = rotation @ accel[i]
        position += velocity * dt + .5 * force * dt**2
        velocity += force * dt
        rotation = rotation @ Rotation.from_rotvec(gyro[i] * dt).as_matrix()
    return {"delta_rotation": rotation, "delta_velocity": velocity, "delta_position": position}


def audit(bag, output, *, baseline, comparison, controls):
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if comparison is not None and output.resolve().is_relative_to(comparison.resolve()):
        raise ValueError("write outside the frozen baseline directory")
    recordings = [("flatwall", bag, "/livox/imu"), *controls]
    labels = [label for label, path, topic in recordings]
    if len(set(labels)) != len(labels) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", label) for label in labels):
        raise ValueError("recordings require unique simple labels")

    def inputs():
        result = {"recordings": {label: {**artifact(path), "topic": topic} for label, path, topic in recordings},
                  "controller": artifact(Path(__file__)), "native": artifact(Path(native.__file__)),
                  "numpy_version": version("numpy"), "scipy_version": version("scipy"),
                  "scipy_rotation_binary": artifact(Path(rotation_binary.__file__))}
        if comparison is not None:
            result["comparison_receipt"] = artifact(comparison / "receipt.json")
            result["comparison_protocol"] = artifact(comparison / "protocol.json")
        return result

    frozen = inputs()
    if frozen["recordings"]["flatwall"]["sha256"] != BAG_SHA256:
        raise ValueError("expected the unmodified flatwall_01.bag from Zenodo record 7641866")
    prior = None
    if comparison is not None:
        prior = json.loads((comparison / "receipt.json").read_text(encoding="utf-8"))
        if prior["protocol"] != frozen["comparison_protocol"] or prior["inputs"]["recordings"] != frozen["recordings"]:
            raise ValueError("baseline protocol/recordings changed")
        if not prior["baseline"] or prior["inputs"]["controller"] != frozen["controller"]:
            raise ValueError("expected a baseline from this same controller")
    protocol = {"schema": "cloudanalyzer.flatwall_imu_scale.v1", "inputs": frozen, "baseline": baseline,
                "acceleration_scale": SCALE, "params": PARAMS, "bias": [0.] * 6,
                "intervals": "Whole topic, first/middle/last 100 ms windows; integer-ns clipping; no fitting",
                "oracle_tolerance": {"rotation_atol": 1e-11, "velocity_rtol": 1e-10, "velocity_atol": 1e-11,
                                     "position_rtol": 1e-10, "position_atol": 1e-10},
                "scope": "Raw-field/default-result parity and scaled nominal preintegration only. No reference archive access, trajectory, deskew, calibrated noise model or speed claim."}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    results = {}
    for label, path, topic in recordings:
        reader = core.BagReader(str(path))
        raw = reader.imu_raw(topic)
        series = reader.imu_series(topic)
        times = raw["timestamps_ns"]
        if np.any(np.diff(times) <= 0):
            raise ValueError("source IMU headers must strictly increase")
        np.testing.assert_array_equal(series.timestamps_ns(), times)
        record = {"samples": len(times), "raw_fields_sha256": fields_digest(raw), "intervals": []}
        scaled = None if baseline or label != "flatwall" else reader.imu_series(topic, acceleration_scale=SCALE)
        if scaled is not None and scaled.acceleration_scale != SCALE:
            raise ValueError("configured scale changed")
        for start, end in intervals(times):
            row = {"start_ns": start, "end_ns": end}
            try:
                legacy = series.preintegrate(start, end, **PARAMS)
            except ValueError as exc:
                row["legacy"] = {"accepted": False, "error": str(exc)}
                if label == "flatwall":
                    raise
            else:
                row["legacy"] = {"accepted": True, "fields_sha256": fields_digest(legacy)}
                if not baseline:
                    identity = reader.imu_series(topic, acceleration_scale=1.).preintegrate(start, end, **PARAMS)
                    if fields_digest(identity) != row["legacy"]["fields_sha256"]:
                        raise ValueError("identity scale changed the legacy result")
            if scaled is not None:
                actual = scaled.preintegrate(start, end, **PARAMS)
                oracle = nominal_oracle(raw, start, end)
                for key, expected in oracle.items():
                    rtol, atol = (0., 1e-11) if key == "delta_rotation" else (1e-10, 1e-11 if key == "delta_velocity" else 1e-10)
                    np.testing.assert_allclose(actual[key], expected, rtol=rtol, atol=atol)
                digest = fields_digest(actual)
                for other in [scaled.factor(start, end, **PARAMS).measurement(),
                              reader.preintegrate_imu(start, end, topic=topic, acceleration_scale=SCALE, **PARAMS),
                              reader.imu_factor(start, end, topic=topic, acceleration_scale=SCALE, **PARAMS).measurement()]:
                    if fields_digest(other) != digest:
                        raise ValueError("scaled series/direct/factor results differ")
                row["scaled"] = {"fields_sha256": digest,
                                 "max_absolute_oracle_errors": {key: float(np.max(np.abs(actual[key] - expected)))
                                                                for key, expected in oracle.items()},
                                 "series_direct_factor_exact": True}
            record["intervals"].append(row)
        if fields_digest(reader.imu_raw(topic)) != record["raw_fields_sha256"]:
            raise ValueError("raw data changed after scaled integration")
        if prior is not None:
            previous = prior["results"][label]
            if record["samples"] != previous["samples"] or record["raw_fields_sha256"] != previous["raw_fields_sha256"]:
                raise ValueError("native raw fields changed from the baseline")
            for row, old in zip(record["intervals"], previous["intervals"], strict=True):
                if {key: value for key, value in row.items() if key != "scaled"} != old:
                    raise ValueError("default results changed from the baseline")
        results[label] = record
    if results["flatwall"]["samples"] != 3755 or inputs() != frozen:
        raise ValueError("input coverage or provenance changed")
    receipt = {**protocol, "protocol": artifact(output / "protocol.json"), "results": results,
               "raw_fields_preserved": True, "default_results_match_baseline": comparison is not None,
               "ground_truth_archive_read": False}
    write_json(output / "receipt.json", receipt)
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
    receipt = audit(args.bag, args.out, baseline=args.baseline, comparison=args.comparison,
                    controls=[(label, Path(path), topic) for label, path, topic in args.control])
    print(json.dumps(receipt["results"], indent=2))


if __name__ == "__main__":
    main()
