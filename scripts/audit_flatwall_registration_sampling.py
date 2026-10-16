#!/usr/bin/env python3
"""Fixed-map startup source-sampling sensitivity; no timing or reference poses."""
from __future__ import annotations

import argparse
from collections import Counter
import inspect
import json
import os
from pathlib import Path

import numpy as np

import cloudanalyzer_core as core
import cloudanalyzer_core._core as native
import audit_flatwall_imu_initialization as startup
import audit_flatwall_raw_point_time as fields

SCANS = list(range(1, 65))
LIDAR = dict(min_range=1.5, max_range=80.0, map_voxel=1.0, map_points=20, deskew=False, diagnostics=True)
CASES = [("omitted", None), ("none", None), ("explicit-1.5", 1.5),
         ("explicit-0.75", 0.75), ("explicit-0.375", 0.375), ("explicit-0.1875", 0.1875)]


def sequence(bag, destination, name, voxel, reference_times):
    keyword = {} if name == "omitted" else {"registration_voxel": voxel}
    odometry = core.LidarOdometry(**LIDAR, **keyword)
    if name != "omitted":
        assert odometry.registration_voxel == voxel
    reader = core.BagReader(str(bag))
    stamps, poses, reports = [], [], []
    for index, scan in enumerate(reader.scans("/livox/points", raw_time_field="t")):
        if index not in SCANS:
            if index > SCANS[-1]:
                break
            continue
        stamps.append(int(scan.point_times_ns(unit="ns", origin="header").max()))
        poses.append(odometry.register(scan.positions()))
        reports.append(odometry.last_report())
    stamps, poses = np.asarray(stamps, dtype=np.int64), np.asarray(poses)
    np.testing.assert_array_equal(stamps, reference_times)
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / "startup-poses.npz", timestamps_ns=stamps, lidar_poses=poses)
    fields.save(destination / "lidar-diagnostics.json", reports)
    independent = startup.reference(bag, stamps, poses)
    fields.save(destination / "independent.json", independent)
    points = [report["registration_points"] for report in reports]
    return stamps, poses, reports, independent, {
        "case": name, "registration_voxel": voxel,
        "registration_points": {"min": min(points), "median": float(np.median(points)), "max": max(points)},
        "termination_counts": dict(Counter(report["termination"] for report in reports)),
        "icp_iterations": sum(report["iterations"] for report in reports),
        "poses_sha256": fields.fields_digest({"poses": poses}),
        "independent": {key: value for key, value in independent.items() if key != "velocities"},
    }


def audit(bag, output, previous, *, baseline, comparison):
    if os.environ.get("RAYON_NUM_THREADS") != "4" or os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        raise ValueError("use fresh RAYON_NUM_THREADS=4 and OPENBLAS_NUM_THREADS=1 processes")
    if output.exists():
        raise FileExistsError("audit output must be fresh")
    bag, output, previous = bag.resolve(), output.resolve(), previous.resolve()
    previous_receipt = json.loads((previous / "receipt.json").read_text(encoding="utf-8"))
    with np.load(previous / "startup-poses.npz", allow_pickle=False) as arrays:
        reference_times, previous_poses = arrays["timestamps_ns"].copy(), arrays["lidar_poses"].copy()
    previous_reports = json.loads((previous / "lidar-diagnostics.json").read_text(encoding="utf-8"))

    def inputs():
        return {"recording": fields.artifact(bag), "native": fields.artifact(Path(native.__file__)),
                "controller": fields.artifact(Path(__file__)),
                "startup_controller": fields.artifact(Path(startup.__file__)),
                "helpers": fields.artifact(Path(fields.__file__)),
                "previous_receipt": fields.artifact(previous / "receipt.json"),
                "previous_poses": fields.artifact(previous / "startup-poses.npz"),
                "previous_diagnostics": fields.artifact(previous / "lidar-diagnostics.json"),
                "oracle_inputs": {key: value for key, value in previous_receipt["inputs"].items()
                                  if key in {"numpy_version", "scipy_version", "rosbags_version", "rotation_backend", "optimizer_source", "independent_reader", "standard_types"}}}

    frozen = inputs()
    # Verify the same oracle libraries/backends as the previous immutable audit.
    from importlib import import_module
    from importlib.metadata import version
    from rosbags.highlevel import AnyReader
    from scipy.optimize import least_squares
    current_oracle = {"numpy_version": version("numpy"), "scipy_version": version("scipy"), "rosbags_version": version("rosbags"),
                      "rotation_backend": fields.artifact(Path(import_module("scipy.spatial.transform._rotation").__file__)),
                      "optimizer_source": fields.artifact(Path(inspect.getfile(least_squares))),
                      "independent_reader": fields.artifact(Path(inspect.getfile(AnyReader))),
                      "standard_types": fields.artifact(Path(import_module("rosbags.typesys.stores.ros1_noetic").__file__))}
    assert frozen["oracle_inputs"] == current_oracle
    assert frozen["recording"] == previous_receipt["inputs"]["recording"]
    prior = None
    if not baseline:
        if comparison is None:
            raise ValueError("comparison baseline required")
        comparison = comparison.resolve()
        prior = json.loads((comparison / "receipt.json").read_text(encoding="utf-8"))
        assert prior["baseline"]
        for key in frozen.keys() - {"native"}:
            assert prior["inputs"][key] == frozen[key], key
    protocol = {"schema": "cloudanalyzer.flatwall_registration_sampling.v1", "inputs": frozen,
                "baseline": baseline, "scan_indices": SCANS, "lidar": LIDAR, "registration_cases": CASES,
                "initializer_parameters": startup.PARAMS, "acceleration_scale": 9.80665,
                "imu_from_lidar": np.eye(4).tolist(), "reference_time": "exact previous maximum integer point timestamps",
                "scope": "Startup consistency and sampling-setting sensitivity only. Map algorithms/settings remain fixed; future map coordinates depend on fitted poses. No reference poses, physical accuracy/calibration, timing, GPU execution or adoption into the frontend."}
    output.mkdir(parents=True, exist_ok=False)
    fields.save(output / "protocol.json", protocol)
    results = []
    series = None if baseline else core.BagReader(str(bag)).imu_series("/livox/imu", acceleration_scale=9.80665)
    for name, voxel in CASES[:1] if baseline else CASES:
        destination = output / name
        stamps, poses, reports, independent, result = sequence(bag, destination, name, voxel, reference_times)
        if name in {"omitted", "none"}:
            np.testing.assert_array_equal(poses.view(np.uint64), previous_poses.view(np.uint64))
            assert reports == previous_reports
            result["all_previous_pose_bits_and_reports_match"] = True
        if baseline:
            result["native"] = None
        else:
            original = fields.fields_digest({"stamps": stamps, "poses": poses})
            outputs, errors = [], []
            for parallel in [False, True]:
                try:
                    value = series.initialize_lidar(stamps, poses, imu_from_lidar=np.eye(4), parallel=parallel, **startup.PARAMS)
                except ValueError as error:
                    errors.append(str(error))
                else:
                    outputs.append(value)
            assert fields.fields_digest({"stamps": stamps, "poses": poses}) == original
            if independent["accepted"]:
                assert len(outputs) == 2 and not errors, errors
                for key in ["timestamps_ns", "imu_poses", "velocities", "gravity", "bias"]:
                    np.testing.assert_array_equal(outputs[0][key].view(np.uint64), outputs[1][key].view(np.uint64))
                for key in ["gravity", "velocities", "bias"]:
                    np.testing.assert_allclose(outputs[0][key], independent[key], rtol=0, atol=2e-5)
                native_result = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in outputs[0].items()}
                native_result.update(accepted=True, serial_parallel_bits_match=True)
            else:
                assert not outputs and len(errors) == 2 and errors[0] == errors[1], errors
                assert "IMU initialization rejected:" in errors[0]
                for label in independent["failed_limits"]:
                    assert label in errors[0]
                native_result = {"accepted": False, "error": errors[0], "serial_parallel_rejection_matches": True}
            fields.save(destination / "native.json", native_result)
            result["native"] = {key: value for key, value in native_result.items()
                                if key in {"accepted", "error", "serial_parallel_bits_match", "serial_parallel_rejection_matches", "gravity_error", "rotation_rms", "velocity_rms"}}
        result["outputs"] = {p.name: fields.artifact(p) for p in sorted(destination.iterdir()) if p.is_file()}
        fields.save(destination / "receipt.json", result)
        results.append(result)
    if prior:
        assert results[0]["poses_sha256"] == prior["results"][0]["poses_sha256"]
    assert inputs() == frozen
    receipt = {**protocol, "protocol": fields.artifact(output / "protocol.json"), "results": results,
               "all_caller_pose_arrays_unchanged": True, "no_reference_poses_used": True, "no_state_adopted": True}
    fields.save(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["bag", "out", "previous"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--comparison", type=Path)
    args = parser.parse_args()
    result = audit(args.bag, args.out, args.previous, baseline=args.baseline, comparison=args.comparison)
    print(json.dumps([{key: value for key, value in row.items() if key != "outputs"} for row in result["results"]], indent=2))


if __name__ == "__main__":
    main()
