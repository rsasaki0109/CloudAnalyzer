#!/usr/bin/env python3
"""Recompute eight policy comparisons from published poses and paired reference."""
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ARMS=["raw-legacy","gyro-legacy","raw-0.1875","gyro-0.1875"]
POLICIES=["repeat-motion","last-pose"]
BASE="benchmarks/slam/flatwall-frontend-controls"
DIAG="benchmarks/slam/flatwall-frontend-constraints"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def angles(rotations):
    skew=np.stack([rotations[...,2,1]-rotations[...,1,2],rotations[...,0,2]-rotations[...,2,0],rotations[...,1,0]-rotations[...,0,1]],axis=-1)
    return np.degrees(np.arctan2(np.linalg.norm(skew,axis=-1)/2.,(np.trace(rotations,axis1=-2,axis2=-1)-1.)/2.))


def verify(root=None):
    repo=Path(__file__).resolve().parents[3]
    default=root is None
    root=Path(root) if root is not None else Path(__file__).resolve().parent
    manifest=read(root/"packet-files.json")
    for mapping,directory in [(manifest["files"],root),(manifest["dependencies"],repo)]:
        for name,item in mapping.items():
            assert not Path(name).is_absolute() and ".." not in Path(name).parts
            data=(directory/name).read_bytes()
            assert len(data)==item["bytes"] and hashlib.sha256(data).hexdigest()==item["sha256"],name
    required={f"{BASE}/{arm}/{name}" for arm in ARMS for name in ["trajectory.npz","per-scan.json","samples.json","edges.json"]}
    required|={f"{DIAG}/{arm}/frames.jsonl" for arm in ARMS}
    required|={f"{BASE}/estimates-receipt.json",f"{BASE}/summary.json",f"{DIAG}/summary.json",f"{DIAG}/verify.py",
               "benchmarks/slam/flatwall-trajectory/protocol.json","benchmarks/slam/flatwall-trajectory/reference-policy-deviation.json"}
    assert set(manifest["dependencies"])==required
    spec=importlib.util.spec_from_file_location("flatwall_policy_accounting",repo/DIAG/"verify.py")
    accounting=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(accounting)
    close,inverse,stats=accounting.close,accounting.inverse,accounting.stats
    files=manifest["files"]
    p,f,c,b,s,a,o=[read(root/name) for name in ["protocol.json","controller-freeze.json","estimates-receipt.json",
        "scoring-boundary.json","summary.json","audit-receipt.json","oracle-receipt.json"]]
    assert p["schema"]=="cloudanalyzer.flatwall_prediction_policy.v1" and s["status"]=="exploratory_conditional_comparison"
    assert p["policies"]==s["policies"]==list(s["controls"])==POLICIES
    assert list(p["arms"])==s["arms"]==c["arms"]==ARMS
    assert all(list(s["controls"][policy])==ARMS for policy in POLICIES)
    assert p["candidate_relative_prediction"]==np.eye(4).tolist() and c["candidate_policy"]=="last-pose"
    assert p["scan_indices"]==c["scan_indices"]==list(range(1,186)) and c["skipped_scan_indices"]==[0]
    assert c["candidate_snapshots"]==740 and c["points_per_arm"]==4_440_000
    assert c["first_two_pose_bits_and_reports_match"] is c["all_source_sensor_bits_match"] is c["fixed_match_equations_checked"] is True
    assert c["candidate_map_matching_independently_replayed"] is False
    assert c["input_hashes_unchanged"] is a["input_hashes_unchanged"] is o["input_hashes_unchanged"] is True
    assert c["no_reference_poses_used_in_this_stage"] is c["reference_previously_inspected"] is s["reference_previously_inspected"] is True
    assert c["no_initializer_state_adopted"] is s["no_initializer_state_adopted"] is True
    assert p["physical_provenance_certified"] is s["physical_provenance_certified"] is o["physical_provenance_certified"] is False
    assert s["all_candidate_estimates_frozen_before_scoring"] is b["all_four_candidate_estimates_and_740_snapshots_frozen"] is True
    assert s["default_metrics_reproduced_exactly"] is s["previous_reference_pairing_reproduced_bitwise"] is True
    assert c["protocol_sha256"]==a["protocol_sha256"]==f["protocol_sha256"]==files["protocol.json"]["sha256"]
    assert c["controller_freeze_sha256"]==a["controller_freeze_sha256"]==o["controller_freeze_sha256"]==files["controller-freeze.json"]["sha256"]
    assert a["estimates_receipt_sha256"]==files["estimates-receipt.json"]["sha256"]
    assert o["scoring_boundary_sha256"]==files["scoring-boundary.json"]["sha256"]
    assert a["output_files_sha256"]["summary.json"]==files["summary.json"]["sha256"]
    assert datetime.fromisoformat(p["frozen_at_utc"])<=datetime.fromisoformat(f["at_utc"])<=datetime.fromisoformat(c["started_at_utc"])<=datetime.fromisoformat(c["frozen_at_utc"])<=datetime.fromisoformat(b["at_utc"])<=datetime.fromisoformat(a["started_at_utc"])
    assert p["scoring_protocol_sha256"]==manifest["dependencies"]["benchmarks/slam/flatwall-trajectory/protocol.json"]["sha256"]
    assert f["analytic_controls_passed"]==6 and f["native_build_inputs"]==113 and f["native_package_files"]==7
    baseline=read(repo/BASE/"estimates-receipt.json")
    assert c["gyro_compensated_points_sha256"]==baseline["gyro_compensated_points_sha256"]
    assert s["original_estimates_per_case"]==185 and s["paired_estimates_per_case"]==183
    assert s["coverage_counts"]=={"interpolated":183,"after_reference":2} and s["full_cohort_covered"] is False
    derived=read(root/"derived-reference.json")
    assert derived["physical_provenance_certified"] is False
    assert derived["output_sha256"]==files["paired-reference.npz"]["sha256"]
    assert derived["source_pairing_sha256"]==f["files_sha256"][p["baseline_evaluation"]+"/raw-legacy/paired.npz"]
    assert derived["source_reference_member"]==read(root/"reference-member.json")
    with np.load(root/"paired-reference.npz",allow_pickle=False) as paired:
        indices,times_ref,reference=[paired[name].copy() for name in ["original_indices","timestamps_ns","reference"]]
    np.testing.assert_array_equal(indices,np.arange(183,dtype=np.int64))
    assert reference.shape==(183,4,4) and times_ref.dtype==np.int64 and np.all(np.diff(times_ref)>0)
    catalog=read(root/"snapshot-catalog.json")
    assert len(catalog)==740
    for arm in ARMS:
        assert [row["scan_index"] for row in catalog if row["arm"]==arm]==list(range(1,186))
        for row in (row for row in catalog if row["arm"]==arm):
            assert set(row["files"])=={f"scan-{row['scan_index']:03d}.npz",f"scan-{row['scan_index']:03d}.json"}
            for name,item in row["files"].items():
                assert item["bytes"]>0 and item["sha256"]==c["output_files_sha256"][arm+"/"+name]
    old_summary=read(repo/BASE/"summary.json")
    assert s["diagnostics"]["repeat-motion"]==read(repo/DIAG/"summary.json")["controls"]
    first_times=None
    for arm in ARMS:
        with np.load(repo/BASE/arm/"trajectory.npz",allow_pickle=False) as data:
            old_poses=data["poses"].copy()
        old_reports=read(repo/BASE/arm/"per-scan.json")
        for policy in POLICIES:
            folder=root/"last-pose"/arm if policy=="last-pose" else repo/BASE/arm
            with np.load(folder/"trajectory.npz",allow_pickle=False) as data:
                times,poses=data["timestamps_ns"].copy(),data["poses"].copy()
            scans=read(folder/"per-scan.json")
            samples,edges=read(folder/"samples.json"),read(folder/"edges.json")
            result=s["controls"][policy][arm]
            assert len(poses)==len(times)==len(scans)==len(samples)==185 and poses.shape==(185,4,4)
            assert times.dtype==np.int64 and np.all(np.diff(times)>0)
            if first_times is None:
                first_times=times
            np.testing.assert_array_equal(times,first_times)
            np.testing.assert_array_equal(times[indices],times_ref)
            assert [r["scan_index"] for r in scans]==[r["scan_index"] for r in samples]==list(range(1,186))
            assert [r["estimate_index"] for r in samples]==list(range(185))
            assert [r["reference_ns"] for r in scans]==[r["timestamp_ns"] for r in samples]==times.tolist()
            assert dict(Counter(row["status"] for row in samples))==s["coverage_counts"]
            assert np.isfinite(poses).all()
            close(poses[:,3,:],np.tile([0.,0.,0.,1.],(185,1)))
            close(poses[:,:3,:3].transpose(0,2,1)@poses[:,:3,:3],np.tile(np.eye(3),(185,1,1)),1e-6)
            close(np.linalg.det(poses[:,:3,:3]),np.ones(185),1e-6)
            np.testing.assert_array_equal(poses[:2].view(np.uint64),old_poses[:2].view(np.uint64))
            assert scans[:2]==old_reports[:2]
            assert [r["registration"]["registration_points"] for r in scans]==[r["registration"]["registration_points"] for r in old_reports]
            if policy=="last-pose":
                for name in ["trajectory.npz","per-scan.json","frames.jsonl"]:
                    assert c["output_files_sha256"][arm+"/"+name]==files["last-pose/"+arm+"/"+name]["sha256"]
                for name in ["samples.json","edges.json"]:
                    assert a["output_files_sha256"]["last-pose/"+arm+"/"+name]==files["last-pose/"+arm+"/"+name]["sha256"]
            else:
                assert result["metrics"]==old_summary["controls"][arm]["metrics"]
            estimate=poses[indices]
            origin=reference[0]@inverse(estimate[0])
            anchored=origin@estimate
            values=dict(origin_position_ape_m=np.linalg.norm(anchored[:,:3,3]-reference[:,:3,3],axis=1),
                        origin_rotation_ape_deg=angles(reference[:,:3,:3].transpose(0,2,1)@anchored[:,:3,:3]))
            x,y=estimate[:,:3,3]-estimate[:,:3,3].mean(axis=0),reference[:,:3,3]-reference[:,:3,3].mean(axis=0)
            u,sigma,vt=np.linalg.svd(x.T@y)
            rotation=vt.T@np.diag([1.,1.,1. if np.linalg.det(vt.T@u.T)>=0 else -1.])@u.T
            translation=reference[:,:3,3].mean(axis=0)-rotation@estimate[:,:3,3].mean(axis=0)
            assert result["position_fit"]["status"]=="identified" and result["position_fit"]["scale"]==1.
            close(np.asarray(result["position_fit"]["world_alignment"])[:3,:3],rotation,2e-8)
            close(np.asarray(result["position_fit"]["world_alignment"])[:3,3],translation,2e-8)
            values["rigid_position_ape_m"]=np.linalg.norm(estimate[:,:3,3]@rotation.T+translation-reference[:,:3,3],axis=1)
            delta=np.array([inverse(inverse(reference[i])@reference[i+1])@(inverse(estimate[i])@estimate[i+1]) for i in range(182)])
            values["consecutive_rpe_translation_m"]=np.linalg.norm(delta[:,:3,3],axis=1)
            values["consecutive_rpe_rotation_deg"]=angles(delta[:,:3,:3])
            assert [(r["left_estimate_index"],r["right_estimate_index"]) for r in edges]==[(i,i+1) for i in range(182)]
            for index,row in enumerate(samples):
                if index<183:
                    assert row["status"]=="interpolated" and 0<row["bracket_span_ns"]<=100_000_000 and 0<row["fraction"]<1
                    assert set(row["errors"])=={key for key in values if "ape_" in key}
                    for key,value in row["errors"].items():
                        close(value,values[key][index],1e-5 if key.endswith("deg") else 2e-8)
                else:
                    assert row["status"]=="after_reference" and row["errors"] is None
            for index,row in enumerate(edges):
                assert row["elapsed_ns"]==int(times[index+1])-int(times[index])
                close(row["translation_m"],values["consecutive_rpe_translation_m"][index],2e-8)
                close(row["rotation_deg"],values["consecutive_rpe_rotation_deg"][index],1e-5)
            for key,value in values.items():
                for field,expected in stats(value).items():
                    close(result["metrics"][key][field],expected,1e-5 if key.endswith("deg") else 2e-8)
            close(result["consecutive_path_lengths"]["estimate_m"],np.linalg.norm(np.diff(estimate[:,:3,3],axis=0),axis=1).sum(),2e-8)
            close(result["consecutive_path_lengths"]["reference_m"],np.linalg.norm(np.diff(reference[:,:3,3],axis=0),axis=1).sum(),2e-8)
            frame_path=folder/"frames.jsonl" if policy=="last-pose" else repo/DIAG/arm/"frames.jsonl"
            frames=[json.loads(line) for line in frame_path.read_text(encoding="utf-8").splitlines()]
            assert len(frames)==185 and [r["scan_index"] for r in frames]==list(range(1,186))
            supported=[]
            for index,(row,scan) in enumerate(zip(frames,scans)):
                predicted=np.eye(4) if index==0 else (poses[index-1] if policy=="last-pose" or index==1 else poses[index-1]@inverse(poses[index-2])@poses[index-1])
                close(row["prediction_pose"],predicted,2e-8)
                np.testing.assert_array_equal(np.asarray(row["returned_pose"]).view(np.uint64),poses[index].view(np.uint64))
                correction=inverse(predicted)@poses[index]
                close(row["prediction_correction_translation_m"],np.linalg.norm(correction[:3,3]),2e-8)
                close(row["prediction_correction_rotation_deg"],float(angles(correction[:3,:3])),1e-5)
                assert row["reference_ns"]==int(times[index]) and row["termination"]==scan["registration"]["termination"]
                assert row["source_points"]==scan["registration"]["registration_points"]
                assert row["prediction_only"] is (not row["alignment_eligible"])
                h=np.asarray(row["normal_matrix"])
                if row["prediction_only"]:
                    assert row["matches"]==0 and row["geometry"] is row["radius_m"] is None and not np.any(h)
                    assert row["prediction_correction_translation_m"]<=2e-8 and row["prediction_correction_rotation_deg"]<=1e-5
                elif row["geometry"] is not None:
                    d=np.array([1.,1.,1.,*[1./row["radius_m"]]*3])
                    normalized=h*d[:,None]*d[None,:]/row["weight_sum"]
                    eigenvalues=np.linalg.eigvalsh((normalized+normalized.T)/2.)
                    close(row["geometry"]["eigenvalues"],eigenvalues)
                    close(row["geometry"]["min_max_ratio"],eigenvalues[0]/eigenvalues[-1])
                    assert row["geometry"]["weak_directions_below_1e-3"]==int(np.count_nonzero(eigenvalues/eigenvalues[-1]<1e-3))
                    supported.append(row)
            diagnostic=s["diagnostics"][policy][arm]
            assert diagnostic["termination_counts"]==dict(Counter(row["termination"] for row in frames))
            assert diagnostic["prediction_only_scans"]==[r["scan_index"] for r in frames if r["prediction_only"]]
            assert diagnostic["supported_spectra"]==len(supported) and diagnostic["weak_spectra"]==sum(r["geometry"]["weak_directions_below_1e-3"]>0 for r in supported)
            for key,items in {"source_points":[r["source_points"] for r in frames],"spectrum_ratio":[r["geometry"]["min_max_ratio"] for r in supported],
                              **{key:[r[key] for r in frames] for key in ["prediction_correction_translation_m","prediction_correction_rotation_deg"]}}.items():
                for field,expected in stats(items).items():
                    close(diagnostic[key][field],expected)
            oracle=o["controls"][policy][arm]
            assert oracle["paired_estimates"]==183 and oracle["consecutive_edges"]==182 and oracle["prior_exact_pairing_reproduced"] is True
            for key,value in oracle["maximum_abs_differences"].items():
                assert 0<=value<=(1e-5 if key.endswith("deg") else 2e-8)
    if default:
        for name,digest in f["source_files_sha256"].items():
            if name.startswith(("scripts/","cloudanalyzer/tests/")):
                assert files["source-snapshot/"+name]["sha256"]==digest
                assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
    return dict(cases=8,original_estimates_per_case=185,paired_per_case=183,edges_per_case=182,new_snapshots=740,
        conditional_rigid_position_rmse_m={policy:{arm:s["controls"][policy][arm]["metrics"]["rigid_position_ape_m"]["rmse"] for arm in ARMS} for policy in POLICIES},
        physical_provenance_certified=False,scope="Published bytes, prior baseline dependencies, exact poses, derived paired reference and numeric error/diagnostic accounting. No raw-reference association, private snapshot/map-search rerun or physical-accuracy certificate.")


if __name__=="__main__":
    print(json.dumps(verify(),indent=2))
