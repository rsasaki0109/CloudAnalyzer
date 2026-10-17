#!/usr/bin/env python3
"""Count-aware eight-case experiment, closed before selected reference parsing."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

import audit_flatwall_frontend_controls as controls
import audit_flatwall_frontend_constraints as constraints
import audit_flatwall_recording_inputs as inputs
import flatwall_selected_reference as selected_reference
import flatwall_trajectory_reference as scoring

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ["repeat-motion", "last-pose"]
CASES = [(p, a) for p in POLICIES for a in controls.ARMS]


def cohort(rows, imu_times):
    if not rows or [r["scan_index"] for r in rows] != list(range(len(rows))):
        raise ValueError("retain every original input scan identity")
    ids, times, excluded = [], [], []
    for row in rows:
        first = row["header_ns"] + row["offset_min_ns"]
        last = row["header_ns"] + row["offset_max_ns"]
        status = inputs.coverage(imu_times, first, last)
        if status != row["header_coverage"]:
            raise ValueError("input IMU coverage differs from frozen metadata")
        if status["status"] == "covered":
            ids.append(row["scan_index"])
            times.append(last)
        else:
            excluded.append(dict(scan_index=row["scan_index"], coverage=status))
    if not ids:
        raise ValueError("no IMU-covered original scans")
    return np.array(ids, dtype=np.int64), scoring.timestamps(np.array(times, dtype=np.int64)), excluded


def step(odometry, points, history, policy):
    if policy not in POLICIES:
        raise ValueError("unsupported prediction policy")
    predicted = constraints.prediction(history) if policy == "repeat-motion" else (history[-1].copy() if history else np.eye(4))
    kwargs = {"relative_prediction": np.eye(4)} if policy == "last-pose" and history else {}
    return odometry.register(points, **kwargs), predicted


def load_estimates(folder, ids, times):
    with np.load(folder / "trajectory.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["original_scan_indices"], ids)
        np.testing.assert_array_equal(arrays["timestamps_ns"], times)
        poses = scoring.validate_poses(arrays["poses"].copy())
    reports = controls.read(folder / "per-scan.json")
    if (len(poses) != len(ids) or len(reports) != len(ids)
            or [r["scan_index"] for r in reports] != ids.tolist()
            or [r["reference_ns"] for r in reports] != times.tolist()):
        raise ValueError("all original cohort identities and timestamps must match")
    return poses, reports


def setup(protocol_path, freeze_path, out):
    protocol, freeze = controls.read(protocol_path), controls.read(freeze_path)
    if (protocol["schema"] != "cloudanalyzer.flatwall_recording_prediction.v1"
            or protocol["policies"] != POLICIES or protocol["arms"] != controls.ARMS
            or protocol["lidar"] != controls.LIDAR or protocol["tolerances"] != constraints.TOLERANCES
            or protocol["reference_member"] != "gt/flatwall_02.txt"
            or protocol["physical_provenance_certified"] is not False
            or protocol["threads"] != {"RAYON_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"}):
        raise ValueError("unsupported experiment policy")
    expected = dict(freeze["files_sha256"])
    expected[protocol_path.resolve().relative_to(ROOT).as_posix()] = freeze["protocol_sha256"]
    expected[freeze_path.resolve().relative_to(ROOT).as_posix()] = controls.sha(freeze_path)
    for source in [__file__, controls.__file__, constraints.__file__, constraints.independent.__file__,
                   inputs.__file__, selected_reference.__file__, scoring.__file__]:
        if Path(source).resolve().relative_to(ROOT).as_posix() not in expected:
            raise ValueError("all controller/arithmetic sources must be frozen")
    if out.exists():
        raise FileExistsError(out)
    for name in expected:
        path = Path(name) if Path(name).is_absolute() else controls.checked_path(name)
        if out.resolve() == path or out.resolve() in path.parents:
            raise ValueError("output would contain a frozen input")
    controls.verify(expected)
    with np.load(controls.checked_path(protocol["input_imu"]), allow_pickle=False) as imu:
        ids, times, excluded = cohort(controls.read(controls.checked_path(protocol["input_scans"])), imu["timestamps_ns"])
    if ids.tolist() != protocol["scan_indices"] or times.tolist() != protocol["reference_timestamps_ns"] or excluded != protocol["excluded_scans"]:
        raise ValueError("declared cohort differs from original input coverage")
    out.mkdir(parents=True, exist_ok=False)
    return protocol, expected, ids, times


def estimate(protocol_path, freeze_path, out):
    protocol, expected, ids, times = setup(protocol_path, freeze_path, out)
    histories, reports, frames = ({case: [] for case in CASES} for _ in range(3))
    catalog, skipped = [], []
    try:
        if any(os.environ.get(k) != v for k, v in protocol["threads"].items()):
            raise ValueError("set frozen thread environment before starting Python")
        sys.path.insert(0, str(controls.checked_path(protocol["native_package"]).parent))
        import cloudanalyzer_core as core
        import cloudanalyzer_core._core as native
        if Path(native.__file__).resolve().parent != controls.checked_path(protocol["native_package"]):
            raise ValueError("unexpected native package")
        reader = core.BagReader(str(controls.checked_path(protocol["recording"])))
        series = reader.imu_series("/livox/imu", acceleration_scale=9.80665)
        imu_times = series.timestamps_ns()
        with np.load(controls.checked_path(protocol["input_imu"]), allow_pickle=False) as imu:
            np.testing.assert_array_equal(imu_times, imu["timestamps_ns"])
        rotation = series.rotation_trajectory(int(imu_times[0]), int(imu_times[-1]), gyro_bias=(0., 0., 0.), max_gap_s=.05)
        odometries = {case: core.LidarOdometry(**controls.LIDAR,
            registration_voxel=controls.ARMS[case[1]]["registration_voxel"], capture_linearization=True) for case in CASES}
        for policy, arm in CASES:
            (out / policy / arm).mkdir(parents=True)
        source_rows = controls.read(controls.checked_path(protocol["input_scans"]))
        digest, count, point_count = hashlib.sha256(), 0, 0
        for index, scan in enumerate(reader.scans("/livox/points", raw_time_field="t")):
            row = source_rows[index]
            xyz, stamps = scan.positions(), scan.point_times_ns(unit="ns", origin="header")
            reference = int(stamps.max())
            if (scan.stamp_ns != row["header_ns"] or len(xyz) != row["finite_points"]
                    or int(stamps.min()) != row["header_ns"] + row["offset_min_ns"]
                    or reference != row["header_ns"] + row["offset_max_ns"]):
                raise ValueError("recording differs from frozen input metadata")
            count += 1
            if row["header_coverage"]["status"] != "covered":
                skipped.append(index)
                continue
            position = len(histories[CASES[0]])
            if index != ids[position] or reference != times[position]:
                raise ValueError("original cohort order changed")
            compensated = rotation.deskew(xyz, stamps, reference, imu_from_lidar=np.eye(4), parallel=True)
            digest.update(index.to_bytes(4, "little") + compensated.astype("<f8").tobytes())
            prepared = {}
            for arm, settings in controls.ARMS.items():
                points = xyz if settings["compensation"] == "raw" else compensated
                near = constraints.independent.filtered(points)
                insertion = constraints.independent.selected(near, .5)
                source = constraints.independent.selected(insertion, 1.5) if settings["registration_voxel"] is None else constraints.independent.selected(near, settings["registration_voxel"])
                prepared[arm] = points, insertion, source
            for case in CASES:
                policy, arm = case
                points, insertion, source = prepared[arm]
                pose, predicted = step(odometries[case], points, histories[case], policy)
                scoring.validate_poses(np.array([pose]))
                histories[case].append(pose.copy())
                report = odometries[case].last_report()
                reports[case].append(dict(scan_index=index, reference_ns=reference, input_points=len(points), registration=report))
                snapshot = odometries[case].last_linearization()
                prefix = out / policy / arm / f"scan-{index:03d}"
                arrays = {k: v for k, v in snapshot.items() if isinstance(v, np.ndarray)}
                with prefix.with_suffix(".npz").open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                controls.save(prefix.with_suffix(".json"), {k: v for k, v in snapshot.items() if k not in arrays})
                catalog.append(dict(policy=policy, arm=arm, scan_index=index, files={p.relative_to(out).as_posix():
                    dict(bytes=p.stat().st_size, sha256=controls.sha(p)) for p in [prefix.with_suffix(".npz"), prefix.with_suffix(".json")]}))
                np.testing.assert_array_equal(snapshot["pose"].view(np.uint64), pose.view(np.uint64))
                np.testing.assert_array_equal(snapshot["source_sensor"].view(np.uint64), source.view(np.uint64))
                world_error = float(np.max(np.abs(constraints.independent.transformed(source, pose)-snapshot["source_world"]), initial=0.))
                insertion_error = float(np.max(np.abs(constraints.independent.transformed(insertion, pose)-snapshot["inserted_world"]), initial=0.))
                if max(world_error, insertion_error) > 2e-8:
                    raise ValueError("source/insertion transform mismatch")
                frames[case].append(dict(policy=policy, arm=arm, scan_index=index, reference_ns=reference,
                    **constraints.describe(snapshot, report, predicted), source_world_max_error_m=world_error, insertion_max_error_m=insertion_error))
                if policy == "last-pose":
                    old_case = ("repeat-motion", arm)
                    if report["registration_points"] != reports[old_case][-1]["registration"]["registration_points"]:
                        raise ValueError("prediction changed source point count")
                    if position < 2:
                        np.testing.assert_array_equal(pose.view(np.uint64), histories[old_case][-1].view(np.uint64))
                        if report != reports[old_case][-1]["registration"]:
                            raise ValueError("first two policy-equivalent reports differ")
            point_count += len(xyz)
            if position % 15 == 0:
                print(json.dumps(dict(completed_estimates_per_case=position+1, snapshots=len(catalog))), flush=True)
        if (count != len(source_rows) or skipped != [r["scan_index"] for r in protocol["excluded_scans"]]
                or any(len(h) != len(ids) for h in histories.values())):
            raise ValueError("incomplete original recording cohort")
        for case in CASES:
            policy, arm = case
            folder = out / policy / arm
            np.testing.assert_array_equal(np.array(histories[case]).view(np.uint64), odometries[case].poses().view(np.uint64))
            with (folder / "trajectory.npz").open("xb") as stream:
                np.savez_compressed(stream, original_scan_indices=ids, timestamps_ns=times, poses=np.array(histories[case]))
            controls.save(folder / "per-scan.json", reports[case])
            with (folder / "frames.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
                for row in frames[case]:
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        controls.save(out / "diagnostics-summary.json", {p: {a: constraints.summarize(frames[p, a]) for a in controls.ARMS} for p in POLICIES})
        controls.save(out / "snapshot-catalog.json", catalog)
        controls.verify(expected)
        receipt = dict(schema="cloudanalyzer.flatwall_recording_estimates.v1", frozen_at_utc=controls.now(),
            protocol_sha256=controls.sha(protocol_path), controller_freeze_sha256=controls.sha(freeze_path),
            cases=[list(c) for c in CASES], scan_indices=ids.tolist(), excluded_scans=protocol["excluded_scans"],
            estimates_per_case=len(ids), snapshots=len(catalog), points_per_case=point_count,
            gyro_compensated_points_sha256=digest.hexdigest(), first_two_pose_bits_and_reports_match=True,
            independent_source_selection_insertion_transforms_and_fixed_match_equations_checked=True,
            map_matching_independently_replayed=False, reference_poses_used_in_estimation=False,
            output_files_sha256={p.relative_to(out).as_posix(): controls.sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
        controls.save(out / "receipt.json", receipt)
    except Exception as exc:
        controls.save(out / "failure.json", dict(at_utc=controls.now(), error_type=type(exc).__name__, error=str(exc), snapshots_saved=len(catalog)))
        for case in CASES:
            folder = out / case[0] / case[1]
            if folder.is_dir():
                controls.save(folder / "partial-scans.json", reports[case])
                with (folder / "partial-poses.npz").open("xb") as stream:
                    np.savez_compressed(stream, poses=np.array(histories[case]))
        raise
    return receipt


def evaluate(protocol_path, freeze_path, estimates, out):
    protocol, expected, ids, times = setup(protocol_path, freeze_path, out)
    closure = controls.read(estimates / "receipt.json")
    if (closure["schema"] != "cloudanalyzer.flatwall_recording_estimates.v1"
            or closure["protocol_sha256"] != controls.sha(protocol_path)
            or closure["controller_freeze_sha256"] != controls.sha(freeze_path)
            or closure["cases"] != [list(c) for c in CASES] or closure["scan_indices"] != ids.tolist()
            or closure["snapshots"] != len(CASES)*len(ids)):
        raise ValueError("all eight complete estimates must be closed before scoring")
    files = closure["output_files_sha256"]
    required = {f"{p}/{a}/{f}" for p, a in CASES for f in ["trajectory.npz", "per-scan.json", "frames.jsonl"]}
    required |= {f"{p}/{a}/scan-{int(i):03d}.{ext}" for p, a in CASES for i in ids for ext in ["npz", "json"]}
    if set(files) != required | {"snapshot-catalog.json", "diagnostics-summary.json"}:
        raise ValueError("incomplete promised estimate and diagnostic closure")
    frozen = {(estimates / n).resolve().relative_to(ROOT).as_posix(): digest for n, digest in files.items()}
    frozen[(estimates / "receipt.json").resolve().relative_to(ROOT).as_posix()] = controls.sha(estimates / "receipt.json")
    controls.verify(frozen)
    cases = {c: load_estimates(estimates / c[0] / c[1], ids, times) for c in CASES}
    controls.save(out / "scoring-boundary.json", dict(at_utc=controls.now(), all_eight_estimates_and_snapshots_closed=True,
        estimates_receipt_sha256=controls.sha(estimates / "receipt.json"), files_sha256=frozen,
        full_selected_reference_parsed_before_this_boundary=False, historical_prefixes_already_exposed=True))
    ref_times, ref_poses, reference = selected_reference.read_selected(controls.checked_path(protocol["reference_archive"]),
        protocol["reference_member"], copy_to=out / "reference-source.txt")
    coverage, paired_indices, paired = scoring.associate(times, ref_times, ref_poses)
    if not len(paired_indices):
        raise ValueError("no conditional reference coverage")
    with (out / "paired-reference.npz").open("xb") as stream:
        np.savez_compressed(stream, cohort_indices=paired_indices, scan_indices=ids[paired_indices], timestamps_ns=times[paired_indices], reference=paired)
    results = {p: {} for p in POLICIES}
    for (policy, arm), (poses, _) in cases.items():
        folder = out / policy / arm
        folder.mkdir(parents=True)
        result, errors = scoring.score(poses[paired_indices], paired, ids[paired_indices], times[paired_indices])
        samples = {r.pop("estimate_index"): r for r in result.pop("samples")}
        controls.save(folder / "samples.json", [dict(cohort_index=r["estimate_index"], scan_index=int(ids[r["estimate_index"]]),
            **{k: v for k, v in r.items() if k != "estimate_index"}, errors=samples.get(int(ids[r["estimate_index"]]))) for r in coverage])
        controls.save(folder / "edges.json", [dict(left_scan_index=r["left_estimate_index"], right_scan_index=r["right_estimate_index"],
            **{k: v for k, v in r.items() if k not in {"left_estimate_index", "right_estimate_index"}}) for r in result.pop("edges")])
        with (folder / "errors.npz").open("xb") as stream:
            np.savez_compressed(stream, **errors)
        controls.save(folder / "summary.json", result)
        results[policy][arm] = result
    controls.save(out / "reference-member.json", reference)
    summary = dict(schema="cloudanalyzer.flatwall_recording_prediction_result.v1", status="exploratory_conditional_comparison",
        estimates_per_case=len(ids), paired_estimates_per_case=len(paired_indices), original_scan_indices=ids.tolist(),
        excluded_input_scans=protocol["excluded_scans"], coverage_counts=dict(Counter(r["status"] for r in coverage)),
        physical_provenance_certified=False, defaults_changed=False, all_estimates_frozen_before_scoring=True,
        controls=results, diagnostics=controls.read(estimates / "diagnostics-summary.json"))
    controls.save(out / "summary.json", summary)
    controls.verify(expected)
    controls.verify(frozen)
    controls.save(out / "receipt.json", dict(at_utc=controls.now(), input_hashes_unchanged=True,
        output_files_sha256={p.relative_to(out).as_posix(): controls.sha(p) for p in sorted(out.rglob("*")) if p.is_file()}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["estimate", "score"])
    for name in ["protocol", "freeze", "out"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--estimates", type=Path)
    args = parser.parse_args()
    if args.stage == "score" and args.estimates is None:
        parser.error("score requires --estimates")
    result = estimate(args.protocol, args.freeze, args.out) if args.stage == "estimate" else evaluate(args.protocol, args.freeze, args.estimates, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))
