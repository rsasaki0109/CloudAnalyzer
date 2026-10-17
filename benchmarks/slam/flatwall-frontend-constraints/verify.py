#!/usr/bin/env python3
"""Verify published diagnostics, spectra and bindings to the frozen baselines."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS=["raw-legacy","gyro-legacy","raw-0.1875","gyro-0.1875"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    a=np.asarray(values,dtype=float)
    assert a.size and np.isfinite(a).all()
    return dict(count=len(a),rmse=float(np.sqrt(np.mean(a*a))),mean=float(a.mean()),
                median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()))


def close(actual,expected,atol=2e-12):
    assert np.allclose(actual,expected,rtol=2e-12,atol=atol),"numeric accounting differs"


def inverse(pose):
    result=np.eye(4)
    result[:3,:3]=pose[:3,:3].T
    result[:3,3]=-pose[:3,:3].T@pose[:3,3]
    return result


def verify(root=None):
    source_root=Path(__file__).resolve().parents[3] if root is None else None
    root=Path(root) if root is not None else Path(__file__).resolve().parent
    files=read(root/"packet-files.json")["files"]
    for name,item in files.items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        data=(root/name).read_bytes()
        assert len(data)==item["bytes"] and hashlib.sha256(data).hexdigest()==item["sha256"],name
    protocol,freeze,receipt,summary,baseline=[read(root/name) for name in
        ["protocol.json","controller-freeze.json","receipt.json","summary.json","baseline-receipt.json"]]
    assert protocol["schema"]=="cloudanalyzer.flatwall_frontend_constraints.v1"
    assert protocol["capture_linearization"] is True and protocol["weak_ratio"]==1e-3
    assert protocol["scan_indices"]==receipt["original_scan_indices"]==list(range(1,186))
    assert protocol["excluded_scan_indices"]==receipt["skipped_scan_indices"]==[0]
    assert list(protocol["arms"])==list(summary["controls"])==ARMS
    assert receipt["verified_snapshots"]==740 and receipt["points_per_control"]==4_440_000
    assert summary["status"]=="exploratory_conditional_diagnostic"
    assert summary["reference_previously_inspected"] is True and summary["reference_poses_used_in_this_run"] is False
    assert summary["physical_provenance_certified"] is protocol["physical_provenance_certified"] is False
    assert summary["all_baseline_pose_bits_reports_and_gyro_digest_match"] is receipt["all_baseline_pose_bits_reports_match"] is True
    assert receipt["independent_map_selection_matching_and_normal_equations_checked"] is receipt["input_hashes_unchanged"] is True
    assert receipt["gyro_compensated_points_sha256"]==baseline["gyro_compensated_points_sha256"]
    assert protocol["baseline_estimates_receipt_sha256"]==files["baseline-receipt.json"]["sha256"]
    assert receipt["protocol_sha256"]==freeze["protocol_sha256"]==files["protocol.json"]["sha256"]
    assert receipt["controller_freeze_sha256"]==files["controller-freeze.json"]["sha256"]
    assert freeze["native_build_inputs"]==protocol["archived_native_build_inputs_verified"]==113
    assert freeze["native_package_files"]==7 and freeze["analytic_controls_passed"]==7
    assert datetime.fromisoformat(protocol["frozen_at_utc"])<=datetime.fromisoformat(freeze["at_utc"])<=datetime.fromisoformat(receipt["started_at_utc"])<=datetime.fromisoformat(receipt["completed_at_utc"])
    catalog=read(root/"snapshot-catalog.json")
    assert len(catalog)==740 and Counter(row["arm"] for row in catalog)==Counter({name:185 for name in ARMS})
    for name in ARMS:
        assert [r["scan_index"] for r in catalog if r["arm"]==name]==list(range(1,186))
        for row in (r for r in catalog if r["arm"]==name):
            assert set(row["files"])=={f"scan-{row['scan_index']:03d}.npz",f"scan-{row['scan_index']:03d}.json"}
            for filename,item in row["files"].items():
                assert item["bytes"]>0 and len(item["sha256"])==64
                assert receipt["output_files_sha256"][name+"/"+filename]==item["sha256"]
    for name in ARMS:
        frame_name=name+"/frames.jsonl"
        assert receipt["output_files_sha256"][frame_name]==files[frame_name]["sha256"]
        rows=[json.loads(line) for line in (root/frame_name).read_text(encoding="utf-8").splitlines()]
        reports=read(root/name/"baseline-per-scan.json")
        with np.load(root/name/"baseline-trajectory.npz",allow_pickle=False) as data:
            times,poses=data["timestamps_ns"].copy(),data["poses"].copy()
        assert files[name+"/baseline-trajectory.npz"]["sha256"]==baseline["output_files_sha256"][name+"/trajectory.npz"]
        assert files[name+"/baseline-per-scan.json"]["sha256"]==baseline["output_files_sha256"][name+"/per-scan.json"]
        assert len(rows)==len(reports)==len(times)==len(poses)==185
        assert times.dtype==np.int64 and np.all(np.diff(times)>0)
        assert [r["scan_index"] for r in rows]==[r["scan_index"] for r in reports]==list(range(1,186))
        assert all(r["arm"]==name for r in rows)
        supported=[]
        for index,(row,old) in enumerate(zip(rows,reports)):
            report=old["registration"]
            assert row["reference_ns"]==old["reference_ns"]==int(times[index])
            assert row["termination"]==report["termination"] and row["iterations"]==report["iterations"]
            assert row["source_points"]==report["registration_points"]
            close(row["threshold_m"],report["threshold_m"])
            close(row["kernel"],report["threshold_m"]/9.)
            np.testing.assert_array_equal(np.asarray(row["returned_pose"]).view(np.uint64),poses[index].view(np.uint64))
            expected=np.eye(4) if index==0 else poses[index-1]@(np.eye(4) if index==1 else inverse(poses[index-2])@poses[index-1])
            predicted=np.asarray(row["prediction_pose"])
            close(predicted,expected,2e-8)
            correction=inverse(predicted)@poses[index]
            rotation=correction[:3,:3]
            skew=[rotation[2,1]-rotation[1,2],rotation[0,2]-rotation[2,0],rotation[1,0]-rotation[0,1]]
            angle=float(np.degrees(np.arctan2(np.linalg.norm(skew)/2.,(np.trace(rotation)-1.)/2.)))
            close(row["prediction_correction_translation_m"],np.linalg.norm(correction[:3,3]),2e-8)
            close(row["prediction_correction_rotation_deg"],angle,1e-5)
            assert row["prediction_only"] is (not row["alignment_eligible"])
            h,b=np.asarray(row["normal_matrix"]),np.asarray(row["right_hand_side"])
            assert h.shape==(6,6) and b.shape==(6,) and np.isfinite(h).all() and np.isfinite(b).all()
            close(h,h.T)
            if row["prediction_only"]:
                assert row["termination"] in {"empty_map","insufficient_points"} and report["correspondences"] is None
                assert row["matches"]==0 and row["weight_sum"]==0 and row["radius_m"] is row["geometry"] is row["weighted_rmse_m"] is None
                assert not np.any(h) and not np.any(b)
                assert row["prediction_correction_translation_m"]<=2e-8 and row["prediction_correction_rotation_deg"]<=1e-5
            else:
                assert row["source_points"]>=10 and row["matches"]==report["correspondences"]["matched_points"]
                close(row["matched_rmse_m"],report["correspondences"]["rmse_m"])
                if row["geometry"] is not None:
                    assert row["weight_sum"]>0 and row["radius_m"]>0
                    diagonal=np.array([1.,1.,1.,*[1./row["radius_m"]]*3])
                    scaled=h*diagonal[:,None]*diagonal[None,:]/row["weight_sum"]
                    eigenvalues=np.linalg.eigvalsh((scaled+scaled.T)/2.)
                    close(row["geometry"]["eigenvalues"],eigenvalues)
                    close(row["geometry"]["min_max_ratio"],eigenvalues[0]/eigenvalues[-1])
                    assert row["geometry"]["weak_directions_below_1e-3"]==int(np.count_nonzero(eigenvalues/eigenvalues[-1]<1e-3))
                    supported.append(row)
            assert 0<=row["source_world_max_error_m"]<=2e-8 and 0<=row["insertion_max_error_m"]<=2e-8
            assert 0<=row["normal_matrix_max_error"] and 0<=row["rhs_max_error"]
        result=summary["controls"][name]
        assert result["scans"]==185 and result["supported_spectra"]==len(supported)
        assert result["termination_counts"]==dict(Counter(r["termination"] for r in rows))==baseline["termination_counts"][name]
        assert result["prediction_only_scans"]==[r["scan_index"] for r in rows if r["prediction_only"]]
        assert result["weak_spectra"]==sum(r["geometry"]["weak_directions_below_1e-3"]>0 for r in supported)
        values={"source_points":[r["source_points"] for r in rows],"spectrum_ratio":[r["geometry"]["min_max_ratio"] for r in supported],
                **{key:[r[key] for r in rows] for key in ["prediction_correction_translation_m","prediction_correction_rotation_deg"]}}
        for key,items in values.items():
            for field,value in stats(items).items():
                close(result[key][field],value)
        for key,row_key in [("max_normal_matrix_error","normal_matrix_max_error"),("max_rhs_error","rhs_max_error"),
                             ("max_source_world_error_m","source_world_max_error_m"),("max_insertion_error_m","insertion_max_error_m")]:
            close(result[key],max(row[row_key] for row in rows))
    if source_root is not None:
        for name,digest in freeze["source_files_sha256"].items():
            if name.startswith(("scripts/","cloudanalyzer/tests/")):
                assert files["source-snapshot/"+name]["sha256"]==digest
                assert hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest,name
    return dict(controls=4,scans_per_control=185,verified_snapshots=740,
                prediction_only_scans={name:len(value["prediction_only_scans"]) for name,value in summary["controls"].items()},
                supported_spectra={name:value["supported_spectra"] for name,value in summary["controls"].items()},
                weak_spectra={name:value["weak_spectra"] for name,value in summary["controls"].items()},physical_provenance_certified=False,
                scope="Published bytes, baseline pose/report bindings, prediction arithmetic, matrix spectra and summaries; no rerun of private raw snapshots or map matching, no physical observability or accuracy certificate.")


if __name__=="__main__":
    print(json.dumps(verify(),indent=2))
