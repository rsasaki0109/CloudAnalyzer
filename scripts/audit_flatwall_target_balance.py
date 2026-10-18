#!/usr/bin/env python3
"""Closed-loop target-balanced trial, with default replay before reference scoring."""
from collections import Counter
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

import audit_flatwall_frontend_controls as common
import audit_flatwall_odometry_linearization as independent
import audit_flatwall_recording_prediction as legacy
import flatwall_selected_reference as selected
import flatwall_target_balance as arithmetic
import flatwall_trajectory_reference as scoring

ROOT=Path(__file__).resolve().parents[1]


def setup(protocol_path,freeze_path,out):
    protocol,freeze=common.read(protocol_path),common.read(freeze_path)
    if (protocol['schema']!='cloudanalyzer.flatwall_target_balance.v1' or protocol['rule']!='robust_weight / exact_target_multiplicity'
            or len(protocol['cases'])!=16 or protocol['total_candidate_frames']!=2560 or protocol['defaults_changed'] is not False):
        raise ValueError('unsupported target-balance protocol')
    expected=dict(freeze['files_sha256'])
    expected[protocol_path.resolve().relative_to(ROOT).as_posix()]=freeze['protocol_sha256']
    expected[freeze_path.resolve().relative_to(ROOT).as_posix()]=common.sha(freeze_path)
    for module in [__file__,common.__file__,independent.__file__,legacy.__file__,arithmetic.__file__,scoring.__file__,selected.__file__]:
        if Path(module).resolve().relative_to(ROOT).as_posix() not in expected: raise ValueError('freeze all trial sources')
    common.verify(expected)
    out.mkdir(parents=True,exist_ok=False)
    return protocol,expected


def estimate(protocol_path,freeze_path,out):
    protocol,expected=setup(protocol_path,freeze_path,out)
    if any(os.environ.get(k)!=v for k,v in protocol['threads'].items()): raise ValueError('set frozen threads before startup')
    sys.path.insert(0,str(common.checked_path(protocol['native_package']).parent))
    import cloudanalyzer_core as core
    import cloudanalyzer_core._core as native
    if Path(native.__file__).resolve().parent!=common.checked_path(protocol['native_package']): raise ValueError('unexpected native')
    catalog, summaries, total, controls_checked=[],{},0,0
    try:
        for recording in protocol['recordings']:
            cases=[c for c in protocol['cases'] if c['name'].startswith(recording['name']+'/')]
            frames={c['name']:[json.loads(s) for s in common.checked_path(c['frames']).read_text(encoding='utf-8').splitlines()] for c in cases}
            ids=cases[0]['scan_indices']
            if any([r['scan_index'] for r in frames[c['name']]]!=ids for c in cases): raise ValueError('cohort changed')
            histories={(c['name'],balance):[] for c in cases for balance in [False,True]}
            odometries={(c['name'],balance):core.LidarOdometry(**common.LIDAR,
                registration_voxel=common.ARMS[c['name'].split('/')[-1]]['registration_voxel'],capture_linearization=True,target_balance=balance)
                for c in cases for balance in [False,True]}
            maps={c['name']:independent.MapReplay() for c in cases}
            rows={c['name']:[] for c in cases}
            for c in cases:(out/c['name']).mkdir(parents=True)
            reader=core.BagReader(str(common.checked_path(recording['bag'])))
            imu=reader.imu_series('/livox/imu',acceleration_scale=9.80665)
            imu_times=imu.timestamps_ns()
            rotation=imu.rotation_trajectory(int(imu_times[0]),int(imu_times[-1]),gyro_bias=(0.,0.,0.),max_gap_s=.05)
            skipped,seen=[],0
            for index,scan in enumerate(reader.scans('/livox/points',raw_time_field='t')):
                seen+=1
                xyz,stamps=scan.positions(),scan.point_times_ns(unit='ns',origin='header')
                reference=int(stamps.max())
                if stamps.min()<imu_times[0] or reference>imu_times[-1]:
                    skipped.append(index);continue
                position=len(histories[cases[0]['name'],True])
                if position>=len(ids) or index!=ids[position] or len(xyz)!=24000: raise ValueError('original scan coverage changed')
                compensated=rotation.deskew(xyz,stamps,reference,imu_from_lidar=np.eye(4),parallel=True)
                prepared={}
                for arm,settings in common.ARMS.items():
                    points=xyz if settings['compensation']=='raw' else compensated
                    near=independent.filtered(points)
                    insertion=independent.selected(near,.5)
                    source=independent.selected(insertion,1.5) if settings['registration_voxel'] is None else independent.selected(near,settings['registration_voxel'])
                    prepared[arm]=points,insertion,source
                for case in cases:
                    name=case['name'];_,policy,arm=name.split('/')
                    original_frame=frames[name][position]
                    if reference!=original_frame['reference_ns']: raise ValueError('original time changed')
                    points,insertion,source=prepared[arm]
                    for balance in [False,True]:
                        odometry=odometries[name,balance]
                        pose,predicted=legacy.step(odometry,points,histories[name,balance],policy)
                        scoring.validate_poses(np.array([pose]))
                        histories[name,balance].append(pose.copy())
                        snapshot=odometry.last_linearization()
                        report=odometry.last_report()
                        if not snapshot['finite'] or snapshot['target_balance'] is not balance: raise ValueError('invalid native snapshot')
                        np.testing.assert_array_equal(snapshot['source_sensor'].view(np.uint64),source.view(np.uint64))
                        np.testing.assert_allclose(snapshot['source_world'],independent.transformed(source,pose),rtol=0,atol=2e-8)
                        np.testing.assert_allclose(snapshot['inserted_world'],independent.transformed(insertion,pose),rtol=0,atol=2e-8)
                        np.testing.assert_allclose(snapshot['center_world'],predicted[:3,3],rtol=0,atol=2e-8)
                        description=arithmetic.describe(snapshot)
                        if not balance:
                            np.testing.assert_array_equal(pose.view(np.uint64),np.array(original_frame['returned_pose']).view(np.uint64))
                            original=common.checked_path(case['directory'])/f'scan-{index:03d}.npz'
                            with np.load(original,allow_pickle=False) as old:
                                for key in old.files:
                                    a,b=snapshot[key],old[key]
                                    np.testing.assert_array_equal(a.view(np.uint64) if a.dtype.kind=='f' else a,
                                                                  b.view(np.uint64) if b.dtype.kind=='f' else b)
                            if (report['termination']!=original_frame['termination'] or report['iterations']!=original_frame['iterations']
                                    or report['registration_points']!=original_frame['source_points'] or report['threshold_m']!=original_frame['threshold_m']):
                                raise ValueError('default report differs')
                            controls_checked+=1
                            continue
                        replay=maps[name]
                        cloud=np.array([p for cell in replay.cells.values() for p in cell]).reshape(-1,3)
                        membership={p.tobytes() for p in cloud}
                        if any(p.tobytes() not in membership for p in snapshot['target_world']): raise ValueError('target absent from replayed candidate map')
                        arrays={k:v for k,v in snapshot.items() if isinstance(v,np.ndarray)}
                        if position in {1,len(ids)//2,len(ids)-1}:
                            arrays['map_points_world']=cloud[np.lexsort((cloud[:,2],cloud[:,1],cloud[:,0]))]
                        prefix=out/name/f'scan-{index:03d}'
                        with prefix.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**arrays)
                        common.save(prefix.with_suffix('.json'),{k:v for k,v in snapshot.items() if not isinstance(v,np.ndarray)})
                        catalog.append(dict(case=name,scan_index=index,public_sample=position in {1,len(ids)//2,len(ids)-1},
                            files={p.relative_to(out).as_posix():dict(bytes=p.stat().st_size,sha256=common.sha(p)) for p in [prefix.with_suffix('.npz'),prefix.with_suffix('.json')]}))
                        row=dict(case=name,scan_index=index,timestamp_ns=reference,original_snapshot_sha256=common.sha(common.checked_path(case['directory'])/f'scan-{index:03d}.npz'),
                            **description,center_world=snapshot['center_world'].tolist(),prediction_pose=predicted.tolist(),returned_pose=pose.tolist(),
                            registration=report,map_points=len(cloud),alignment_eligible=snapshot['alignment_eligible'])
                        rows[name].append(row)
                        with (out/name/'frames.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                        replay.insert(snapshot['inserted_world'],pose[:3,3])
                        total+=1
                if position%15==0:print(json.dumps(dict(recording=recording['name'],scans=position+1,candidate_snapshots=total,default_replays=controls_checked)),flush=True)
            if seen!=recording['input_scan_count'] or skipped!=recording['excluded_scan_indices'] or any(len(h)!=len(ids) for h in histories.values()):
                raise ValueError('incomplete original recording')
            for case in cases:
                name=case['name'];folder=out/name
                times=np.array([r['reference_ns'] for r in frames[name]],dtype=np.int64)
                for balance,filename in [(True,'trajectory.npz'),(False,'default-replay-trajectory.npz')]:
                    np.testing.assert_array_equal(np.array(histories[name,balance]).view(np.uint64),odometries[name,balance].poses().view(np.uint64))
                    with (folder/filename).open('xb') as f:np.savez_compressed(f,original_scan_indices=np.array(ids,dtype=np.int64),timestamps_ns=times,poses=np.array(histories[name,balance]))
                summaries[name]=dict(scans=len(rows[name]),termination_counts=dict(Counter(r['registration']['termination'] for r in rows[name])),
                    matches=sum(r['matches'] for r in rows[name]),unique_targets=sum(r['unique_targets'] for r in rows[name]),
                    supported_systems=sum(r['geometry'] is not None for r in rows[name]),
                    weak_systems=sum(r['geometry'] is not None and r['geometry']['weak_directions_below_1e_3']>0 for r in rows[name]))
        if total!=2560 or controls_checked!=2560: raise ValueError('incomplete sixteen-case trial')
        common.save(out/'diagnostics-summary.json',summaries)
        common.save(out/'snapshot-catalog.json',catalog)
        common.verify(expected)
        receipt=dict(schema='cloudanalyzer.flatwall_target_balance_estimates.v1',at_utc=common.now(),protocol_sha256=common.sha(protocol_path),
            controller_freeze_sha256=common.sha(freeze_path),candidate_frames=total,default_replays=controls_checked,
            default_pose_and_existing_snapshot_array_bits_match=True,default_report_fields_match=True,
            source_and_insertion_selection_and_transforms_checked=True,candidate_map_insertion_and_target_membership_replayed=True,
            candidate_forward_search_independently_replayed=False,all_final_pose_effective_weights_and_H_b_checked=True,reference_poses_used=False,
            output_files_sha256={p.relative_to(out).as_posix():common.sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
        common.save(out/'receipt.json',receipt)
        return receipt
    except Exception as exc:
        common.save(out/'failure.json',dict(at_utc=common.now(),error_type=type(exc).__name__,error=str(exc),candidate_snapshots=total,default_replays=controls_checked))
        raise


def evaluate(protocol_path,freeze_path,estimates,out):
    protocol,expected=setup(protocol_path,freeze_path,out)
    receipt=common.read(estimates/'receipt.json')
    if (receipt['schema']!='cloudanalyzer.flatwall_target_balance_estimates.v1' or receipt['candidate_frames']!=2560 or receipt['default_replays']!=2560
            or receipt['protocol_sha256']!=common.sha(protocol_path) or receipt['controller_freeze_sha256']!=common.sha(freeze_path)):
        raise ValueError('all sixteen candidate trajectories and controls must close before scoring')
    required={'diagnostics-summary.json','snapshot-catalog.json'}
    for case in protocol['cases']:
        required|={f"{case['name']}/{name}" for name in ['trajectory.npz','default-replay-trajectory.npz','frames.jsonl']}
        required|={f"{case['name']}/scan-{index:03d}.{ext}" for index in case['scan_indices'] for ext in ['npz','json']}
    if set(receipt['output_files_sha256'])!=required: raise ValueError('incomplete promised output inventory')
    frozen={(estimates/name).relative_to(ROOT).as_posix():digest for name,digest in receipt['output_files_sha256'].items()}
    frozen[(estimates/'receipt.json').relative_to(ROOT).as_posix()]=common.sha(estimates/'receipt.json')
    common.verify(frozen)
    common.save(out/'scoring-boundary.json',dict(at_utc=common.now(),all_candidate_frames_closed=2560,all_default_replays_closed=2560,
        estimates_receipt_sha256=common.sha(estimates/'receipt.json'),files_sha256=frozen,reference_poses_read_in_this_trial_before_boundary=False,
        both_full_recordings_and_prior_scores_already_exposed=True))
    results={}
    for recording in protocol['recordings']:
        folder=out/recording['name'];folder.mkdir()
        rt,rp,reference=selected.read_selected(common.checked_path(protocol['reference_archive']),recording['reference_member'],copy_to=folder/'reference-source.txt')
        common.save(folder/'reference-member.json',reference)
        for case in [c for c in protocol['cases'] if c['name'].startswith(recording['name']+'/')]:
            name=case['name'];destination=out/name;destination.mkdir(parents=True)
            with np.load(estimates/name/'trajectory.npz',allow_pickle=False) as data:ids,times,poses=[data[k].copy() for k in ['original_scan_indices','timestamps_ns','poses']]
            coverage,paired_ids,paired=scoring.associate(times,rt,rp)
            result,errors=scoring.score(poses[paired_ids],paired,ids[paired_ids],times[paired_ids])
            with (destination/'paired-reference.npz').open('xb') as f:np.savez_compressed(f,cohort_indices=paired_ids,scan_indices=ids[paired_ids],timestamps_ns=times[paired_ids],reference=paired)
            common.save(destination/'coverage.json',coverage)
            common.save(destination/'metrics.json',result)
            with (destination/'errors.npz').open('xb') as f:np.savez_compressed(f,**errors)
            results[name]=result
    summary=dict(schema='cloudanalyzer.flatwall_target_balance_result.v1',status='exploratory_conditional_comparison',cases=results,
        defaults_changed=False,physical_provenance_certified=False,all_estimates_closed_before_scoring=True,speed_claim=False,
        diagnostics=common.read(estimates/'diagnostics-summary.json'))
    common.save(out/'summary.json',summary)
    common.verify(expected);common.verify(frozen)
    common.save(out/'receipt.json',dict(at_utc=common.now(),input_hashes_unchanged=True,estimates_receipt_sha256=common.sha(estimates/'receipt.json'),
        output_files_sha256={p.relative_to(out).as_posix():common.sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['estimate','score'])
    for name in ['protocol','freeze','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--estimates',type=Path)
    args=parser.parse_args()
    if args.stage=='score' and args.estimates is None:parser.error('score requires estimates')
    result=estimate(args.protocol,args.freeze,args.out) if args.stage=='estimate' else evaluate(args.protocol,args.freeze,args.estimates,args.out)
    print(json.dumps(result,indent=2))
