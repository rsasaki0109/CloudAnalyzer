#!/usr/bin/env python3
"""Replay all four frozen controls with independently checked ICP diagnostics."""
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
import audit_flatwall_odometry_linearization as independent
import flatwall_trajectory_reference as rigid

ROOT = Path(__file__).resolve().parents[1]
TOLERANCES = dict(coordinate_atol=2e-8, matrix_atol=2e-8, matrix_rtol=2e-12,
                  weight_atol=2e-12, weight_rtol=2e-12, no_alignment_rotation_atol_deg=1e-5)


def prediction(previous):
    """Repeat the last body-frame displacement; use only preceding estimates."""
    if not len(previous):
        return np.eye(4)
    rigid.validate_poses(previous)
    motion = np.eye(4) if len(previous) == 1 else rigid.inverse(previous[-2]) @ previous[-1]
    return previous[-1] @ motion


def describe(snapshot, report, predicted):
    """Independently reconstruct the fixed-match surrogate, without solving it."""
    if not snapshot["finite"]:
        raise ValueError("nonfinite native diagnostic")
    weights, h, b, support, radius = independent.reconstruct(snapshot)
    np.testing.assert_allclose(snapshot["weights"], weights, rtol=TOLERANCES["weight_rtol"], atol=TOLERANCES["weight_atol"])
    np.testing.assert_allclose(snapshot["normal_matrix"], h, rtol=TOLERANCES["matrix_rtol"], atol=TOLERANCES["matrix_atol"])
    np.testing.assert_allclose(snapshot["right_hand_side"], b, rtol=TOLERANCES["matrix_rtol"], atol=TOLERANCES["matrix_atol"])
    np.testing.assert_allclose(snapshot["weight_sum"], support, rtol=TOLERANCES["weight_rtol"], atol=TOLERANCES["weight_atol"])
    if radius is None:
        if snapshot["weighted_rms_radius_m"] is not None:
            raise ValueError("empty support must retain an absent radius")
    else:
        np.testing.assert_allclose(snapshot["weighted_rms_radius_m"], radius, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(snapshot["center_world"], predicted[:3, 3], rtol=0., atol=TOLERANCES["coordinate_atol"])
    np.testing.assert_allclose(snapshot["threshold_sq"], report["threshold_m"]**2, rtol=2e-15, atol=0.)
    np.testing.assert_allclose(snapshot["kernel"], report["threshold_m"]/9., rtol=2e-15, atol=0.)
    eligible = snapshot["alignment_eligible"]
    matched = len(snapshot["matched_source_indices"])
    if eligible:
        if report["correspondences"]["matched_points"] != matched:
            raise ValueError("diagnostic and stopping-report support differ")
    elif matched or report["termination"] not in {"empty_map", "insufficient_points"}:
        raise ValueError("unsupported registration must retain its failure state")
    correction = rigid.inverse(predicted) @ snapshot["pose"]
    delta_t = float(np.linalg.norm(correction[:3, 3]))
    delta_r = float(rigid.angles_deg(correction[:3, :3]))
    if not eligible and (delta_t > TOLERANCES["coordinate_atol"] or delta_r > TOLERANCES["no_alignment_rotation_atol_deg"]):
        raise ValueError("unregistered pose differs from the existing motion prediction")
    squared = snapshot["squared_distances"]
    return dict(source_points=len(snapshot["source_sensor"]), matches=matched,
                alignment_eligible=eligible, prediction_only=not eligible, termination=report["termination"],
                iterations=report["iterations"], threshold_m=report["threshold_m"], kernel=snapshot["kernel"],
                weight_sum=support, radius_m=radius, normal_matrix=h.tolist(), right_hand_side=b.tolist(),
                geometry=independent.spectrum(h, support, radius),
                matched_rmse_m=float(np.sqrt(np.mean(squared))) if matched else None,
                weighted_rmse_m=float(np.sqrt(np.sum(weights*squared)/support)) if support > 0 else None,
                prediction_pose=predicted.tolist(), returned_pose=snapshot["pose"].tolist(),
                prediction_correction_translation_m=delta_t, prediction_correction_rotation_deg=delta_r,
                normal_matrix_max_error=float(np.max(np.abs(snapshot["normal_matrix"]-h))),
                rhs_max_error=float(np.max(np.abs(snapshot["right_hand_side"]-b))))


def summarize(rows):
    supported = [row for row in rows if row["geometry"] is not None]
    return dict(scans=len(rows), termination_counts=dict(Counter(r["termination"] for r in rows)),
                prediction_only_scans=[r["scan_index"] for r in rows if r["prediction_only"]],
                supported_spectra=len(supported),
                weak_spectra=sum(r["geometry"]["weak_directions_below_1e-3"] > 0 for r in supported),
                source_points=rigid.summarize(np.asarray([r["source_points"] for r in rows], dtype=float)),
                spectrum_ratio=rigid.summarize(np.asarray([r["geometry"]["min_max_ratio"] for r in supported])),
                prediction_correction_translation_m=rigid.summarize(np.asarray([r["prediction_correction_translation_m"] for r in rows])),
                prediction_correction_rotation_deg=rigid.summarize(np.asarray([r["prediction_correction_rotation_deg"] for r in rows])),
                max_normal_matrix_error=max(r["normal_matrix_max_error"] for r in rows),
                max_rhs_error=max(r["rhs_max_error"] for r in rows),
                max_source_world_error_m=max(r["source_world_max_error_m"] for r in rows),
                max_insertion_error_m=max(r["insertion_max_error_m"] for r in rows))


def audit(protocol_path, freeze_path, out):
    protocol, freeze = controls.read(protocol_path), controls.read(freeze_path)
    if (protocol["schema"] != "cloudanalyzer.flatwall_frontend_constraints.v1"
            or protocol["arms"] != controls.ARMS or protocol["lidar"] != controls.LIDAR
            or protocol["tolerances"] != TOLERANCES or protocol["weak_ratio"] != 1e-3
            or protocol["scan_indices"] != list(range(1, 186))):
        raise ValueError("unsupported constraint-audit policy")
    expected = dict(freeze["files_sha256"])
    expected[protocol_path.resolve().relative_to(ROOT).as_posix()] = freeze["protocol_sha256"]
    expected[freeze_path.resolve().relative_to(ROOT).as_posix()] = controls.sha(freeze_path)
    for source in [__file__, controls.__file__, independent.__file__, rigid.__file__]:
        if Path(source).resolve().relative_to(ROOT).as_posix() not in expected:
            raise ValueError("all audit sources must be frozen before measurements")
    if out.exists():
        raise FileExistsError(out)
    for name in expected:
        path = Path(name) if Path(name).is_absolute() else controls.checked_path(name)
        if out.resolve() == path or out.resolve() in path.parents:
            raise ValueError("output would contain a frozen input")
    for name, value in protocol["threads"].items():
        if os.environ.get(name) != value:
            raise ValueError("set thread environment before starting Python")
    controls.verify(expected)
    out.mkdir(parents=True, exist_ok=False)
    rows = {name: [] for name in controls.ARMS}
    catalog = []
    try:
        sys.path.insert(0, str(controls.checked_path(protocol["native_package"]).parent))
        import cloudanalyzer_core as core
        import cloudanalyzer_core._core as native
        if Path(native.__file__).resolve().parent != controls.checked_path(protocol["native_package"]):
            raise ValueError("unexpected native package")
        baseline = controls.checked_path(protocol["baseline_estimates"])
        old = {name: controls.load_cohort(baseline/name/"trajectory.npz") for name in controls.ARMS}
        reports = {name: controls.read(baseline/name/"per-scan.json") for name in controls.ARMS}
        for name in controls.ARMS:
            if ([r["scan_index"] for r in reports[name]] != list(range(1, 186))
                    or [r["reference_ns"] for r in reports[name]] != old[name][0].tolist()):
                raise ValueError("baseline scan identity mismatch")
            np.testing.assert_array_equal(old[name][0], next(iter(old.values()))[0])
            (out/name).mkdir()
        reader = core.BagReader(str(controls.checked_path(protocol["recording"])))
        series = reader.imu_series("/livox/imu", acceleration_scale=9.80665)
        imu_times = series.timestamps_ns()
        rotation = series.rotation_trajectory(int(imu_times[0]), int(imu_times[-1]), gyro_bias=(0.,0.,0.), max_gap_s=.05)
        odometries = {name: core.LidarOdometry(**controls.LIDAR, registration_voxel=arm["registration_voxel"],
                       capture_linearization=True) for name, arm in controls.ARMS.items()}
        maps = {name: independent.MapReplay() for name in controls.ARMS}
        digest, skipped, points_count = hashlib.sha256(), [], 0
        started = controls.now()
        for index, scan in enumerate(reader.scans("/livox/points", raw_time_field="t")):
            xyz, stamps = scan.positions(), scan.point_times_ns(unit="ns", origin="header")
            reference = int(stamps.max())
            if stamps.min() < imu_times[0] or reference > imu_times[-1]:
                if index != 0:
                    raise ValueError("unexpected IMU-uncovered scan")
                skipped.append(index)
                continue
            if index not in range(1,186) or len(xyz) != 24000:
                raise ValueError("unexpected recording cohort")
            compensated = rotation.deskew(xyz, stamps, reference, imu_from_lidar=np.eye(4), parallel=True)
            digest.update(index.to_bytes(4,"little")+compensated.astype("<f8").tobytes())
            for name, arm in controls.ARMS.items():
                times, poses = old[name]
                if reference != times[index-1]:
                    raise ValueError("baseline timestamp mismatch")
                points = xyz if arm["compensation"] == "raw" else compensated
                odometry, replay = odometries[name], maps[name]
                pose = odometry.register(points)
                np.testing.assert_array_equal(pose.view(np.uint64), poses[index-1].view(np.uint64))
                report = odometry.last_report()
                if report != reports[name][index-1]["registration"]:
                    raise ValueError("capturing changed a registration report")
                snapshot = odometry.last_linearization()
                np.testing.assert_array_equal(snapshot["pose"].view(np.uint64), pose.view(np.uint64))
                near = independent.filtered(points)
                insertion = independent.selected(near,.5)
                source = independent.selected(insertion,1.5) if arm["registration_voxel"] is None else independent.selected(near,arm["registration_voxel"])
                np.testing.assert_array_equal(snapshot["source_sensor"].view(np.uint64), source.view(np.uint64))
                world_error = float(np.max(np.abs(independent.transformed(source,pose)-snapshot["source_world"]),initial=0.))
                insertion_error = float(np.max(np.abs(independent.transformed(insertion,pose)-snapshot["inserted_world"]),initial=0.))
                if max(world_error,insertion_error) > TOLERANCES["coordinate_atol"]:
                    raise ValueError("source/insertion transform mismatch")
                if snapshot["alignment_eligible"] != bool(replay.cells and len(source)>=10):
                    raise ValueError("alignment eligibility mismatch")
                indices, targets, distances = replay.matches(snapshot["source_world"],snapshot["threshold_sq"]) if snapshot["alignment_eligible"] else (np.empty(0,dtype=np.int64),np.empty((0,3)),np.empty(0))
                np.testing.assert_array_equal(snapshot["matched_source_indices"],indices)
                np.testing.assert_array_equal(snapshot["target_world"].view(np.uint64),targets.view(np.uint64))
                np.testing.assert_array_equal(snapshot["squared_distances"].view(np.uint64),distances.view(np.uint64))
                item = dict(arm=name,scan_index=index,reference_ns=reference,
                            **describe(snapshot,report,prediction(poses[:index-1])),
                            source_world_max_error_m=world_error,insertion_max_error_m=insertion_error)
                arrays = {key:value for key,value in snapshot.items() if isinstance(value,np.ndarray)}
                prefix = out/name/f"scan-{index:03d}"
                np.savez_compressed(prefix.with_suffix(".npz"),**arrays)
                metadata = {key:value for key,value in snapshot.items() if key not in arrays}
                controls.save(prefix.with_suffix(".json"),metadata)
                catalog.append(dict(arm=name,scan_index=index,files={p.name:dict(bytes=p.stat().st_size,sha256=controls.sha(p)) for p in [prefix.with_suffix(".npz"),prefix.with_suffix(".json")]}))
                rows[name].append(item)
                replay.insert(snapshot["inserted_world"],pose[:3,3])
            points_count += len(xyz)
            if index % 10 == 0:
                print(json.dumps(dict(completed_original_scans=index,checked_snapshots=sum(len(r) for r in rows.values()))),flush=True)
        baseline_receipt = controls.read(baseline/"receipt.json")
        if skipped != [0] or points_count != 4_440_000 or any(len(r)!=185 for r in rows.values()):
            raise ValueError("incomplete cohort")
        if digest.hexdigest() != baseline_receipt["gyro_compensated_points_sha256"]:
            raise ValueError("gyro compensation digest differs from baseline")
        for name in controls.ARMS:
            np.testing.assert_array_equal(odometries[name].poses().view(np.uint64),old[name][1].view(np.uint64))
            np.savez_compressed(out/name/"trajectory.npz",timestamps_ns=old[name][0],poses=odometries[name].poses())
            with (out/name/"frames.jsonl").open("x",encoding="utf-8",newline="\n") as f:
                for row in rows[name]:
                    f.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
        summary = dict(schema="cloudanalyzer.flatwall_frontend_constraints_result.v1",status="exploratory_conditional_diagnostic",
                       reference_previously_inspected=True,reference_poses_used_in_this_run=False,
                       all_baseline_pose_bits_reports_and_gyro_digest_match=True,physical_provenance_certified=False,
                       controls={name:summarize(value) for name,value in rows.items()})
        controls.save(out/"summary.json",summary)
        controls.save(out/"snapshot-catalog.json",catalog)
        controls.verify(expected)
        receipt = dict(schema="cloudanalyzer.flatwall_frontend_constraints_receipt.v1",started_at_utc=started,completed_at_utc=controls.now(),
                       protocol_sha256=controls.sha(protocol_path),controller_freeze_sha256=controls.sha(freeze_path),
                       verified_snapshots=len(catalog),original_scan_indices=list(range(1,186)),skipped_scan_indices=skipped,
                       points_per_control=points_count,all_baseline_pose_bits_reports_match=True,
                       gyro_compensated_points_sha256=digest.hexdigest(),input_hashes_unchanged=True,
                       independent_map_selection_matching_and_normal_equations_checked=True,
                       output_files_sha256={p.relative_to(out).as_posix():controls.sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
        controls.save(out/"receipt.json",receipt)
    except Exception as exc:
        controls.save(out/"failure.json",dict(at_utc=controls.now(),error_type=type(exc).__name__,error=str(exc),
                      completed_rows_by_arm={name:len(value) for name,value in rows.items()}))
        for name,value in rows.items():
            controls.save(out/name/"partial-frames.json",value) if (out/name).is_dir() else None
        raise
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["protocol","controller-freeze","out"]:
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.protocol,args.controller_freeze,args.out),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
