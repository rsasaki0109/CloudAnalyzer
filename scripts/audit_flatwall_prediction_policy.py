#!/usr/bin/env python3
"""Freeze a last-pose warm-start control before the eight-case comparison."""
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
import flatwall_trajectory_reference as scoring

ROOT=Path(__file__).resolve().parents[1]
POLICIES=["repeat-motion","last-pose"]


def last_pose_step(odometry,points,previous):
    predicted=np.eye(4) if previous is None else previous.copy()
    pose=odometry.register(points) if previous is None else odometry.register(points,relative_prediction=np.eye(4))
    return pose,predicted


def cohort(directory,expected_times=None):
    times,poses=controls.load_cohort(directory/"trajectory.npz",expected_times)
    reports=controls.read(directory/"per-scan.json")
    if (len(reports)!=185 or [r["scan_index"] for r in reports]!=list(range(1,186))
            or [r["reference_ns"] for r in reports]!=times.tolist()):
        raise ValueError("all original scan identities and times must match")
    return times,poses,reports


def setup(protocol_path,freeze_path,out):
    protocol,freeze=controls.read(protocol_path),controls.read(freeze_path)
    if (protocol["schema"]!="cloudanalyzer.flatwall_prediction_policy.v1"
            or protocol["policies"]!=POLICIES or protocol["arms"]!=controls.ARMS
            or protocol["lidar"]!=controls.LIDAR or protocol["scan_indices"]!=list(range(1,186))
            or protocol["tolerances"]!=constraints.TOLERANCES
            or protocol["candidate_relative_prediction"]!=np.eye(4).tolist()):
        raise ValueError("unsupported prediction-policy protocol")
    inherited=controls.read(controls.checked_path(protocol["baseline_control_protocol"]))
    controls.validate_protocol(inherited)
    if (controls.sha(controls.checked_path(protocol["scoring_protocol"]))!=protocol["scoring_protocol_sha256"]
            or inherited["scoring_protocol_sha256"]!=protocol["scoring_protocol_sha256"]):
        raise ValueError("inherited scoring policy changed")
    expected=dict(freeze["files_sha256"])
    expected[protocol_path.resolve().relative_to(ROOT).as_posix()]=freeze["protocol_sha256"]
    expected[freeze_path.resolve().relative_to(ROOT).as_posix()]=controls.sha(freeze_path)
    for source in [__file__,controls.__file__,constraints.__file__,constraints.independent.__file__,scoring.__file__]:
        if Path(source).resolve().relative_to(ROOT).as_posix() not in expected:
            raise ValueError("all controller and arithmetic sources must be frozen")
    if out.exists():
        raise FileExistsError(out)
    for name in expected:
        path=Path(name) if Path(name).is_absolute() else controls.checked_path(name)
        if out.resolve()==path or out.resolve() in path.parents:
            raise ValueError("output would contain a frozen input")
    controls.verify(expected)
    out.mkdir(parents=True,exist_ok=False)
    return protocol,expected


def estimate(protocol_path,freeze_path,out):
    protocol,expected=setup(protocol_path,freeze_path,out)
    rows={name:[] for name in controls.ARMS}
    poses={name:[] for name in controls.ARMS}
    reports={name:[] for name in controls.ARMS}
    catalog=[]
    try:
        for key,value in protocol["threads"].items():
            if os.environ.get(key)!=value:
                raise ValueError("set thread environment before starting Python")
        sys.path.insert(0,str(controls.checked_path(protocol["native_package"]).parent))
        import cloudanalyzer_core as core
        import cloudanalyzer_core._core as native
        if Path(native.__file__).resolve().parent!=controls.checked_path(protocol["native_package"]):
            raise ValueError("unexpected native package")
        baseline=controls.checked_path(protocol["baseline_estimates"])
        old={name:cohort(baseline/name) for name in controls.ARMS}
        times=next(iter(old.values()))[0]
        for name in controls.ARMS:
            np.testing.assert_array_equal(old[name][0],times)
            (out/name).mkdir()
        reader=core.BagReader(str(controls.checked_path(protocol["recording"])))
        series=reader.imu_series("/livox/imu",acceleration_scale=9.80665)
        imu_times=series.timestamps_ns()
        trajectory=series.rotation_trajectory(int(imu_times[0]),int(imu_times[-1]),gyro_bias=(0.,0.,0.),max_gap_s=.05)
        odometries={name:core.LidarOdometry(**controls.LIDAR,registration_voxel=arm["registration_voxel"],capture_linearization=True)
                    for name,arm in controls.ARMS.items()}
        digest,skipped,point_count=hashlib.sha256(),[],0
        started=controls.now()
        for index,scan in enumerate(reader.scans("/livox/points",raw_time_field="t")):
            xyz,stamps=scan.positions(),scan.point_times_ns(unit="ns",origin="header")
            reference=int(stamps.max())
            if stamps.min()<imu_times[0] or reference>imu_times[-1]:
                if index!=0:
                    raise ValueError("unexpected IMU-uncovered scan")
                skipped.append(index)
                continue
            if index not in range(1,186) or reference!=times[index-1] or len(xyz)!=24000:
                raise ValueError("recording differs from the original cohort")
            compensated=trajectory.deskew(xyz,stamps,reference,imu_from_lidar=np.eye(4),parallel=True)
            digest.update(index.to_bytes(4,"little")+compensated.astype("<f8").tobytes())
            for name,arm in controls.ARMS.items():
                points=xyz if arm["compensation"]=="raw" else compensated
                pose,predicted=last_pose_step(odometries[name],points,poses[name][-1] if poses[name] else None)
                scoring.validate_poses(np.asarray([pose]))
                poses[name].append(pose.copy())
                report=odometries[name].last_report()
                reports[name].append(dict(scan_index=index,reference_ns=reference,input_points=len(points),registration=report))
                old_report=old[name][2][index-1]["registration"]
                if report["registration_points"]!=old_report["registration_points"]:
                    raise ValueError("prediction policy changed selected source count")
                if index<=2:
                    np.testing.assert_array_equal(pose.view(np.uint64),old[name][1][index-1].view(np.uint64))
                    if report!=old_report:
                        raise ValueError("first two policy-equivalent reports differ")
                snapshot=odometries[name].last_linearization()
                np.testing.assert_array_equal(snapshot["pose"].view(np.uint64),pose.view(np.uint64))
                original=controls.checked_path(protocol["baseline_constraints"])/name/f"scan-{index:03d}.npz"
                with np.load(original,allow_pickle=False) as saved:
                    np.testing.assert_array_equal(snapshot["source_sensor"].view(np.uint64),saved["source_sensor"].view(np.uint64))
                    # A transform-consistency check on prior canonical insertion bytes,
                    # not a fresh independent voxel selection or map search.
                    insertion_sensor=(saved["inserted_world"]-saved["pose"][:3,3])@saved["pose"][:3,:3]
                world_error=float(np.max(np.abs(constraints.independent.transformed(snapshot["source_sensor"],pose)-snapshot["source_world"]),initial=0.))
                insertion_error=float(np.max(np.abs(constraints.independent.transformed(insertion_sensor,pose)-snapshot["inserted_world"]),initial=0.))
                if max(world_error,insertion_error)>constraints.TOLERANCES["coordinate_atol"]:
                    raise ValueError("source/insertion transform mismatch")
                item=dict(arm=name,scan_index=index,reference_ns=reference,prediction_policy="last-pose",
                          **constraints.describe(snapshot,report,predicted),
                          source_world_max_error_m=world_error,insertion_max_error_m=insertion_error)
                prefix=out/name/f"scan-{index:03d}"
                arrays={key:value for key,value in snapshot.items() if isinstance(value,np.ndarray)}
                np.savez_compressed(prefix.with_suffix(".npz"),**arrays)
                controls.save(prefix.with_suffix(".json"),{key:value for key,value in snapshot.items() if key not in arrays})
                catalog.append(dict(arm=name,scan_index=index,files={p.name:dict(bytes=p.stat().st_size,sha256=controls.sha(p))
                    for p in [prefix.with_suffix(".npz"),prefix.with_suffix(".json")]}))
                rows[name].append(item)
            point_count+=len(xyz)
            if index%20==0:
                print(json.dumps(dict(completed_original_scans=index,candidate_snapshots=len(catalog))),flush=True)
        old_receipt=controls.read(baseline/"receipt.json")
        if skipped!=[0] or point_count!=4_440_000 or any(len(value)!=185 for value in rows.values()):
            raise ValueError("incomplete candidate cohort")
        if digest.hexdigest()!=old_receipt["gyro_compensated_points_sha256"]:
            raise ValueError("gyro compensation digest changed")
        for name in controls.ARMS:
            np.testing.assert_array_equal(np.asarray(poses[name]).view(np.uint64),odometries[name].poses().view(np.uint64))
            np.savez_compressed(out/name/"trajectory.npz",timestamps_ns=times,poses=np.asarray(poses[name]))
            controls.save(out/name/"per-scan.json",reports[name])
            with (out/name/"frames.jsonl").open("x",encoding="utf-8",newline="\n") as f:
                for row in rows[name]:
                    f.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
        controls.save(out/"diagnostics-summary.json",{name:constraints.summarize(value) for name,value in rows.items()})
        controls.save(out/"snapshot-catalog.json",catalog)
        controls.verify(expected)
        receipt=dict(schema="cloudanalyzer.flatwall_prediction_estimates.v1",started_at_utc=started,frozen_at_utc=controls.now(),
            protocol_sha256=controls.sha(protocol_path),controller_freeze_sha256=controls.sha(freeze_path),
            arms=list(controls.ARMS),scan_indices=list(range(1,186)),skipped_scan_indices=skipped,points_per_arm=point_count,
            candidate_policy="last-pose",candidate_snapshots=len(catalog),first_two_pose_bits_and_reports_match=True,
            all_source_sensor_bits_match=True,gyro_compensated_points_sha256=digest.hexdigest(),
            fixed_match_equations_checked=True,candidate_map_matching_independently_replayed=False,
            reference_previously_inspected=True,no_reference_poses_used_in_this_stage=True,no_initializer_state_adopted=True,
            input_hashes_unchanged=True,output_files_sha256={p.relative_to(out).as_posix():controls.sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
        controls.save(out/"receipt.json",receipt)
    except Exception as exc:
        controls.save(out/"failure.json",dict(at_utc=controls.now(),error_type=type(exc).__name__,error=str(exc),completed_snapshots=len(catalog)))
        for name in controls.ARMS:
            if (out/name).is_dir():
                controls.save(out/name/"partial-scans.json",reports[name])
                np.savez_compressed(out/name/"partial-poses.npz",poses=np.asarray(poses[name]))
        raise
    return receipt


def evaluate(protocol_path,freeze_path,estimates,out):
    protocol,expected=setup(protocol_path,freeze_path,out)
    closure=controls.read(estimates/"receipt.json")
    if (closure["schema"]!="cloudanalyzer.flatwall_prediction_estimates.v1" or closure["arms"]!=list(controls.ARMS)
            or closure["scan_indices"]!=list(range(1,186)) or closure["candidate_snapshots"]!=740
            or closure["protocol_sha256"]!=controls.sha(protocol_path) or closure["controller_freeze_sha256"]!=controls.sha(freeze_path)
            or closure["candidate_policy"]!="last-pose"):
        raise ValueError("all four candidate estimates and diagnostics must be closed before scoring")
    candidate_files={(estimates/name).resolve().relative_to(ROOT).as_posix():digest for name,digest in closure["output_files_sha256"].items()}
    candidate_files[(estimates/"receipt.json").resolve().relative_to(ROOT).as_posix()]=controls.sha(estimates/"receipt.json")
    required={f"{name}/{filename}" for name in controls.ARMS for filename in ["trajectory.npz","per-scan.json","frames.jsonl"]}
    if not required.issubset(closure["output_files_sha256"]):
        raise ValueError("incomplete candidate closure")
    controls.verify(candidate_files)
    baseline=controls.checked_path(protocol["baseline_estimates"])
    baseline_data={name:cohort(baseline/name) for name in controls.ARMS}
    times=next(iter(baseline_data.values()))[0]
    cases={"repeat-motion":baseline_data,"last-pose":{name:cohort(estimates/name,times) for name in controls.ARMS}}
    for group in cases.values():
        for value in group.values():
            np.testing.assert_array_equal(value[0],times)
    started=controls.now()
    ref_times,ref_poses,reference=scoring.read_selected_reference(controls.checked_path(protocol["reference_archive"]))
    coverage,indices,paired=scoring.associate(times,ref_times,ref_poses)
    previous=controls.checked_path(protocol["baseline_evaluation"])
    with np.load(previous/"raw-legacy/paired.npz",allow_pickle=False) as old:
        np.testing.assert_array_equal(old["reference"].view(np.uint64),paired.view(np.uint64))
        np.testing.assert_array_equal(old["original_indices"],indices)
        np.testing.assert_array_equal(old["timestamps_ns"],times[indices])
    results={policy:{} for policy in POLICIES}
    for policy,group in cases.items():
        for name,(_,poses,_) in group.items():
            destination=out/policy/name
            destination.mkdir(parents=True)
            result,errors=scoring.score(poses[indices],paired,indices,times[indices])
            samples={r["estimate_index"]:r for r in result.pop("samples")}
            controls.save(destination/"samples.json",[dict(row,scan_index=row["estimate_index"]+1,
                errors={k:v for k,v in samples.get(row["estimate_index"],{}).items() if k!="estimate_index"} or None) for row in coverage])
            controls.save(destination/"edges.json",result.pop("edges"))
            np.savez_compressed(destination/"paired.npz",original_indices=indices,timestamps_ns=times[indices],estimated=poses[indices],reference=paired)
            np.savez_compressed(destination/"errors.npz",**errors)
            controls.save(destination/"summary.json",result)
            results[policy][name]=result
    old_summary=controls.read(previous/"summary.json")
    if any(results["repeat-motion"][name]["metrics"]!=old_summary["controls"][name]["metrics"] for name in controls.ARMS):
        raise ValueError("frozen default metrics did not reproduce exactly")
    controls.save(out/"reference-member.json",reference)
    summary=dict(schema="cloudanalyzer.flatwall_prediction_policy_result.v1",status="exploratory_conditional_comparison",
        policies=POLICIES,arms=list(controls.ARMS),original_estimates_per_case=185,paired_estimates_per_case=len(indices),
        coverage_counts=dict(Counter(r["status"] for r in coverage)),full_cohort_covered=len(indices)==185,
        reference_previously_inspected=True,physical_provenance_certified=False,all_candidate_estimates_frozen_before_scoring=True,
        default_metrics_reproduced_exactly=True,previous_reference_pairing_reproduced_bitwise=True,
        no_initializer_state_adopted=True,controls=results,
        diagnostics={"repeat-motion":controls.read(controls.checked_path(protocol["baseline_constraints"])/"summary.json")["controls"],
                     "last-pose":controls.read(estimates/"diagnostics-summary.json")})
    controls.save(out/"summary.json",summary)
    controls.verify(expected)
    controls.verify(candidate_files)
    controls.save(out/"receipt.json",dict(started_at_utc=started,completed_at_utc=controls.now(),protocol_sha256=controls.sha(protocol_path),
        controller_freeze_sha256=controls.sha(freeze_path),estimates_receipt_sha256=controls.sha(estimates/"receipt.json"),
        input_hashes_unchanged=True,output_files_sha256={p.relative_to(out).as_posix():controls.sha(p) for p in sorted(out.rglob("*")) if p.is_file()}))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["estimate","score"])
    for name in ["protocol","controller-freeze","out"]:
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--estimates",type=Path)
    args=parser.parse_args()
    if args.stage=="estimate":
        result=estimate(args.protocol,args.controller_freeze,args.out)
    else:
        if args.estimates is None:
            parser.error("score requires --estimates")
        result=evaluate(args.protocol,args.controller_freeze,args.estimates,args.out)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
