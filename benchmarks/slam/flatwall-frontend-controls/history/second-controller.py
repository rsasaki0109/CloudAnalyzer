#!/usr/bin/env python3
"""Freeze all four real Flatwall controls before conditional reference scoring."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

import flatwall_trajectory_reference as scoring

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "raw-legacy": dict(compensation="raw", registration_voxel=None),
    "gyro-legacy": dict(compensation="gyro", registration_voxel=None),
    "raw-0.1875": dict(compensation="raw", registration_voxel=0.1875),
    "gyro-0.1875": dict(compensation="gyro", registration_voxel=0.1875),
}
LIDAR = dict(min_range=1.5, max_range=80., map_voxel=1., map_points=20,
             deskew=False, diagnostics=True)


def now():
    return datetime.now(timezone.utc).isoformat()


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
    if Path(name).is_absolute() or not path.is_relative_to(ROOT):
        raise ValueError("paths must stay inside this checkout")
    return path


def verify(manifest):
    for name, digest in manifest.items():
        path = Path(name) if Path(name).is_absolute() else checked_path(name)
        if sha(path) != digest:
            raise ValueError(f"frozen file changed: {name}")


def validate_protocol(protocol):
    if (protocol["schema"] != "cloudanalyzer.flatwall_frontend_controls.v1"
            or protocol["arms"] != ARMS or protocol["lidar"] != LIDAR
            or protocol["original_scan_indices"] != list(range(1, 186))
            or protocol["gyro_bias_rad_s"] != [0., 0., 0.]
            or protocol["max_gap_s"] != 0.05
            or protocol["imu_from_lidar"] != np.eye(4).tolist()
            or protocol["threads"] != {"RAYON_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"}
            or protocol["physical_provenance_certified"] is not False):
        raise ValueError("unsupported control policy")
    inherited = read(checked_path(protocol["scoring_protocol"]))
    if (sha(checked_path(protocol["scoring_protocol"])) != protocol["scoring_protocol_sha256"]
            or inherited["association"]["maximum_bracket_span_ns"] != scoring.MAX_BRACKET_NS
            or inherited["validation"]["quaternion_norm_atol"] != scoring.QUATERNION_ATOL
            or inherited["alignment"]["rank_relative_threshold"] != scoring.RANK_RTOL):
        raise ValueError("inherited reference policy changed")


def load_cohort(path, expected_times=None):
    with np.load(path, allow_pickle=False) as arrays:
        times, poses = arrays["timestamps_ns"].copy(), arrays["poses"].copy()
    scoring.timestamps(times)
    scoring.validate_poses(poses)
    if len(times) != 185 or len(poses) != 185:
        raise ValueError("all 185 original estimates are required")
    if expected_times is not None and not np.array_equal(times, expected_times):
        raise ValueError("control timestamps differ from the frozen original cohort")
    return times, poses


def setup(protocol_path, freeze_path, out):
    protocol, freeze = read(protocol_path), read(freeze_path)
    validate_protocol(protocol)
    if out.exists():
        raise FileExistsError(out)
    manifest = dict(freeze["files_sha256"])
    manifest[protocol_path.resolve().relative_to(ROOT).as_posix()] = freeze["protocol_sha256"]
    manifest[freeze_path.resolve().relative_to(ROOT).as_posix()] = sha(freeze_path)
    for source in [__file__, scoring.__file__]:
        if Path(source).resolve().relative_to(ROOT).as_posix() not in manifest:
            raise ValueError("controller and scoring sources must be frozen")
    for path in manifest:
        source = Path(path) if Path(path).is_absolute() else checked_path(path)
        if out.resolve() == source or out.resolve() in source.parents:
            raise ValueError("output would contain a frozen input")
    verify(manifest)
    out.mkdir(parents=True, exist_ok=False)
    return protocol, manifest


def estimate(protocol_path, freeze_path, out):
    protocol, manifest = setup(protocol_path, freeze_path, out)
    for key, value in protocol["threads"].items():
        if os.environ.get(key) != value:
            raise ValueError("declare thread environment before starting Python")
    sys.path.insert(0, str(checked_path(protocol["native_package"]).parent))
    import cloudanalyzer_core as core
    import cloudanalyzer_core._core as native
    if Path(native.__file__).resolve().parent != checked_path(protocol["native_package"]):
        raise ValueError("unexpected native package loaded")
    times_expected, _ = load_cohort(checked_path(protocol["nominal_trajectory"]))
    reader = core.BagReader(str(checked_path(protocol["recording"])))
    series = reader.imu_series("/livox/imu", acceleration_scale=9.80665)
    imu_times = series.timestamps_ns()
    rotation = series.rotation_trajectory(int(imu_times[0]), int(imu_times[-1]),
                                           gyro_bias=(0., 0., 0.), max_gap_s=0.05)
    odometries = {name: core.LidarOdometry(**LIDAR, **({"registration_voxel": arm["registration_voxel"]}
                   if arm["registration_voxel"] is not None else {})) for name, arm in ARMS.items()}
    poses, rows, prefixes = {}, {}, {}
    for name in ARMS:
        (out / name).mkdir()
        poses[name], rows[name] = [], []
        folder = checked_path(protocol["prefix_directories"][name])
        with np.load(folder / "startup-poses.npz", allow_pickle=False) as data:
            prefixes[name] = (data["timestamps_ns"].copy(), data["lidar_poses"].copy(),
                              read(folder / "lidar-diagnostics.json"))
    digest, skipped, point_count, decreasing = hashlib.sha256(), [], 0, 0
    started = now()
    try:
        for index, scan in enumerate(reader.scans("/livox/points", raw_time_field="t")):
            xyz, stamps = scan.positions.astype(np.float64), scan.point_times_ns(unit="ns", origin="header")
            reference = int(stamps.max())
            if stamps.min() < imu_times[0] or reference > imu_times[-1]:
                if index != 0:
                    raise ValueError("unexpected IMU-uncovered scan")
                skipped.append(index)
                continue
            if index not in range(1, 186) or len(xyz) != 24000 or reference != times_expected[index-1]:
                raise ValueError("recording cohort differs from the original")
            compensated = rotation.deskew(xyz, stamps, reference,
                                           imu_from_lidar=np.eye(4), parallel=True)
            digest.update(index.to_bytes(4, "little") + compensated.astype("<f8").tobytes())
            for name, arm in ARMS.items():
                points = xyz if arm["compensation"] == "raw" else compensated
                pose = odometries[name].register(points)
                scoring.validate_poses(np.asarray([pose]))
                report = odometries[name].last_report()
                poses[name].append(pose.copy())
                rows[name].append(dict(scan_index=index, reference_ns=reference,
                                       input_points=len(points), registration=report))
                if index <= 64:
                    old_times, old_poses, old_reports = prefixes[name]
                    if reference != old_times[index-1] or report != old_reports[index-1]:
                        raise ValueError(f"prefix report/time mismatch: {name}, {index}")
                    np.testing.assert_array_equal(pose.view(np.uint64), old_poses[index-1].view(np.uint64))
            point_count += len(xyz)
            decreasing += int(np.count_nonzero(np.diff(stamps) < 0))
            if index % 20 == 0:
                print(json.dumps(dict(stage="estimate", completed_scans=index, arms=len(ARMS))), flush=True)
        if skipped != [0] or point_count != 4_440_000 or any(len(p) != 185 for p in poses.values()):
            raise ValueError("incomplete original cohort")
        old = read(checked_path(protocol["gyro_component_receipt"]))
        if digest.hexdigest() != old["component"]["digests"]["identity"]["native"]:
            raise ValueError("full-cohort compensated point digest changed")
        for name in ARMS:
            np.testing.assert_array_equal(np.asarray(poses[name]).view(np.uint64), odometries[name].poses().view(np.uint64))
            np.savez_compressed(out / name / "trajectory.npz", timestamps_ns=times_expected, poses=np.asarray(poses[name]))
            save(out / name / "per-scan.json", rows[name])
        verify(manifest)
        receipt = dict(schema="cloudanalyzer.flatwall_frontend_controls_estimates.v1", started_at_utc=started,
                       frozen_at_utc=now(), protocol_sha256=sha(protocol_path), controller_freeze_sha256=sha(freeze_path),
                       arms=list(ARMS), original_scan_indices=list(range(1, 186)), skipped_scan_indices=skipped,
                       points_per_arm=point_count, decreasing_point_steps_retained=decreasing,
                       gyro_compensated_points_sha256=digest.hexdigest(), all_64_prefix_pose_bits_and_reports_match=True,
                       prior_full_gyro_component_digest_matches=True, no_reference_poses_used_in_this_stage=True,
                       reference_already_inspected=True, no_initializer_state_adopted=True,
                       termination_counts={n: dict(Counter(r["registration"]["termination"] for r in rows[n])) for n in ARMS},
                       output_files_sha256={p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
        save(out / "receipt.json", receipt)
    except Exception as exc:
        for name in ARMS:
            save(out / name / "partial-scans.json", rows[name])
            np.savez_compressed(out / name / "partial-poses.npz", poses=np.asarray(poses[name]))
        save(out / "failure.json", dict(at_utc=now(), error_type=type(exc).__name__, error=str(exc)))
        raise
    return receipt


def evaluate(protocol_path, freeze_path, estimates, out):
    protocol, manifest = setup(protocol_path, freeze_path, out)
    closure = read(estimates / "receipt.json")
    if (closure["schema"] != "cloudanalyzer.flatwall_frontend_controls_estimates.v1"
            or closure["protocol_sha256"] != sha(protocol_path)
            or closure["controller_freeze_sha256"] != sha(freeze_path)
            or closure["arms"] != list(ARMS)
            or closure["original_scan_indices"] != list(range(1, 186))):
        raise ValueError("all control estimates must be frozen before scoring")
    estimate_files = {estimates.resolve().relative_to(ROOT).as_posix()+"/"+k: v for k, v in closure["output_files_sha256"].items()}
    required = {f"{name}/{file}" for name in ARMS for file in ["trajectory.npz", "per-scan.json"]}
    if set(closure["output_files_sha256"]) != required:
        raise ValueError("estimate closure must include all trajectories and diagnostics")
    estimate_files[(estimates / "receipt.json").resolve().relative_to(ROOT).as_posix()] = sha(estimates / "receipt.json")
    verify(estimate_files)
    times, nominal = load_cohort(checked_path(protocol["nominal_trajectory"]))
    cohorts = {name: load_cohort(estimates / name / "trajectory.npz", times)[1] for name in ARMS}
    cohorts["nominal-imu-legacy"] = nominal
    for name in ARMS:
        rows = read(estimates / name / "per-scan.json")
        if ([r["scan_index"] for r in rows] != list(range(1, 186))
                or [r["reference_ns"] for r in rows] != times.tolist()):
            raise ValueError("control diagnostics must retain original scan identity")
    ref_times, ref_poses, reference = scoring.read_selected_reference(checked_path(protocol["reference_archive"]))
    coverage, indices, paired = scoring.associate(times, ref_times, ref_poses)
    if not len(indices):
        raise ValueError("no reference coverage")
    summaries = {}
    for name, poses in cohorts.items():
        folder = out / name
        folder.mkdir()
        result, errors = scoring.score(poses[indices], paired, indices, times[indices])
        samples = {r["estimate_index"]: r for r in result.pop("samples")}
        save(folder / "samples.json", [dict(row, scan_index=row["estimate_index"]+1,
             errors={k: v for k, v in samples.get(row["estimate_index"], {}).items() if k != "estimate_index"} or None) for row in coverage])
        save(folder / "edges.json", result.pop("edges"))
        np.savez_compressed(folder / "errors.npz", **errors)
        np.savez_compressed(folder / "paired.npz", original_indices=indices, timestamps_ns=times[indices], estimated=poses[indices], reference=paired)
        save(folder / "summary.json", result)
        summaries[name] = result
    original = read(checked_path(protocol["nominal_evaluation_summary"]))
    if summaries["nominal-imu-legacy"]["metrics"] != original["metrics"]:
        raise ValueError("original nominal evaluation did not reproduce exactly")
    save(out / "reference-member.json", reference)
    summary = dict(schema="cloudanalyzer.flatwall_frontend_controls_result.v1", status="exploratory_conditional_comparison",
                   original_estimates_per_arm=185, paired_estimates_per_arm=len(indices),
                   coverage_counts=dict(Counter(r["status"] for r in coverage)), full_cohort_covered=len(indices)==185,
                   physical_provenance_certified=False, all_estimates_frozen_before_this_scoring=True,
                   reference_previously_inspected=True, no_initializer_state_adopted=True,
                   original_nominal_metrics_reproduced_exactly=True, controls=summaries)
    save(out / "summary.json", summary)
    verify(manifest)
    verify(estimate_files)
    save(out / "receipt.json", dict(at_utc=now(), protocol_sha256=sha(protocol_path), controller_freeze_sha256=sha(freeze_path),
         estimates_receipt_sha256=sha(estimates / "receipt.json"), input_hashes_unchanged=True,
         output_files_sha256={p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["estimate", "score"])
    for name in ["protocol", "controller-freeze", "out"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--estimates", type=Path)
    args = parser.parse_args()
    if args.stage == "estimate":
        result = estimate(args.protocol, args.controller_freeze, args.out)
    else:
        if args.estimates is None:
            parser.error("score requires --estimates")
        result = evaluate(args.protocol, args.controller_freeze, args.estimates, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
