#!/usr/bin/env python3
"""Independent exact-time trajectory and weighted-system packet verification."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ARMS=['raw-legacy','gyro-legacy','raw-0.1875','gyro-0.1875']
POLICIES=['repeat-motion','last-pose']

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check_sample(path,metadata,row,oracle):
    with np.load(path,allow_pickle=False) as saved:d={k:saved[k].copy() for k in saved.files}
    if metadata['target_balance'] is not True or metadata['finite'] is not True:raise ValueError('invalid sample policy')
    q=d['source_world'][d['matched_source_indices']]
    p,c=d['target_world'],d['center_world']
    if len(q)!=row['matches'] or len(d['source_sensor'])!=row['source_points']:raise ValueError('sample dimensions differ')
    oracle.close(d['pose'],row['returned_pose'])
    oracle.close(c,row['center_world'])
    oracle.close(d['source_world'],d['source_sensor']@d['pose'][:3,:3].T+d['pose'][:3,3])
    cloud=d['map_points_world']
    if cloud.shape!=(row['map_points'],3):raise ValueError('map dimensions differ')
    oracle.close(cloud[np.lexsort((cloud[:,2],cloud[:,1],cloud[:,0]))],cloud)
    membership={point.tobytes() for point in cloud}
    if any(point.tobytes() not in membership for point in p):raise ValueError('target absent from public candidate map')
    counts=Counter(map(tuple,p.tolist()))
    multiplicity=np.array([counts[tuple(point)] for point in p],dtype=np.int64)
    np.testing.assert_array_equal(d['target_multiplicity'],multiplicity)
    residual,arm=q-p,q-c
    distance=np.sum(residual*residual,axis=1)
    oracle.close(d['squared_distances'],distance)
    oracle.close(metadata['threshold_sq'],row['registration']['threshold_m']**2)
    oracle.close(metadata['kernel'],row['registration']['threshold_m']/9)
    robust=metadata['kernel']**2/(metadata['kernel']+distance)**2
    weight=robust/multiplicity
    oracle.close(d['robust_weights'],robust,2e-12)
    oracle.close(d['weights'],weight,2e-12)
    j=np.zeros((len(q),3,6))
    for i,(x,y,z) in enumerate(arm):j[i]=np.column_stack([np.eye(3),[[0.,z,-y],[-z,0.,x],[y,-x,0.]]])
    h=np.einsum('nri,n,nrj->ij',j,weight,j)
    b=-np.einsum('nri,n,nr->i',j,weight,residual)
    for key,expected in [('normal_matrix',h),('right_hand_side',b)]:
        oracle.close(d[key],expected);oracle.close(row[key],expected)
    oracle.close(row['weight_sum'],weight.sum())
    oracle.close(row['robust_weight_sum'],robust.sum())
    oracle.close(row['weighted_cost'],np.sum(weight*distance))
    if len(counts)!=row['unique_targets']:raise ValueError('unique-target count differs')
    radius=float(np.sqrt(np.sum(weight*np.sum(arm*arm,axis=1))/weight.sum())) if weight.sum()>0 else None
    if radius is None:
        if row['radius_m'] is not None or metadata['weighted_rms_radius_m'] is not None:raise ValueError('empty radius differs')
    else:
        oracle.close(row['radius_m'],radius);oracle.close(metadata['weighted_rms_radius_m'],radius)
    return len(q)


def check_metrics(result,estimate,reference,ids,times,oracle,full=True):
    origin=reference[0]@oracle.inverse(estimate[0])
    anchored=origin@estimate
    errors=dict(origin_position_ape_m=np.linalg.norm(anchored[:,:3,3]-reference[:,:3,3],axis=1),
        origin_rotation_ape_deg=oracle.angles(reference[:,:3,:3].transpose(0,2,1)@anchored[:,:3,:3]))
    oracle.close(result['origin_world_alignment'],origin)
    x,y=estimate[:,:3,3]-estimate[:,:3,3].mean(0),reference[:,:3,3]-reference[:,:3,3].mean(0)
    u,s,vt=np.linalg.svd(x.T@y)
    spectra=dict(estimate=np.linalg.svd(x,compute_uv=False),reference=np.linalg.svd(y,compute_uv=False),cross_covariance=s)
    ranks={k:int(sum(v>v[0]*1e-10)) if v[0]>0 else 0 for k,v in spectra.items()}
    if result['position_fit']['ranks']!=ranks or result['position_fit']['scale']!=1.:raise ValueError('fit rank/scale differs')
    for key,value in spectra.items():oracle.close(result['position_fit']['singular_values'][key],value)
    if min(ranks.values())>=2:
        r=vt.T@np.diag([1.,1.,1. if np.linalg.det(vt.T@u.T)>=0 else -1.])@u.T
        t=reference[:,:3,3].mean(0)-r@estimate[:,:3,3].mean(0)
        if result['position_fit']['status']!='identified':raise ValueError('fit status differs')
        oracle.close(np.array(result['position_fit']['world_alignment'])[:3,:3],r)
        oracle.close(np.array(result['position_fit']['world_alignment'])[:3,3],t)
        errors['rigid_position_ape_m']=np.linalg.norm(estimate[:,:3,3]@r.T+t-reference[:,:3,3],axis=1)
    elif result['position_fit']['status']!='degenerate_rank':raise ValueError('missing rank guard')
    edges=np.flatnonzero(np.diff(ids)==1)
    delta=np.array([oracle.inverse(oracle.inverse(reference[i])@reference[i+1])@(oracle.inverse(estimate[i])@estimate[i+1]) for i in edges]).reshape(-1,4,4)
    errors['consecutive_rpe_translation_m']=np.linalg.norm(delta[:,:3,3],axis=1)
    errors['consecutive_rpe_rotation_deg']=oracle.angles(delta[:,:3,:3])
    if set(result['metrics'])!=set(errors):raise ValueError('metric inventory differs')
    for key,value in errors.items():oracle.check_stats(result['metrics'][key],value,1e-5 if key.endswith('deg') else 2e-8)
    if full:
        if [r['estimate_index'] for r in result['samples']]!=ids.tolist():raise ValueError('sample identity differs')
        for i,row in enumerate(result['samples']):
            for key,value in errors.items():
                if 'ape_' in key:oracle.close(row[key],value[i],1e-5 if key.endswith('deg') else 2e-8)
        if [(e['left_estimate_index'],e['right_estimate_index']) for e in result['edges']]!=[(int(ids[i]),int(ids[i+1])) for i in edges]:raise ValueError('RPE bridges an original gap')
        for j,i in enumerate(edges):
            row=result['edges'][j]
            if row['elapsed_ns']!=int(times[i+1])-int(times[i]):raise ValueError('edge time differs')
            oracle.close(row['translation_m'],errors['consecutive_rpe_translation_m'][j])
            oracle.close(row['rotation_deg'],errors['consecutive_rpe_rotation_deg'][j],1e-5)
    for key,value in [('estimate_m',estimate),('reference_m',reference)]:oracle.close(result['consecutive_path_lengths'][key],np.linalg.norm(np.diff(value[:,:3,3],axis=0)[edges],axis=1).sum())
    return errors


def verify(folder=None):
    folder=Path(folder) if folder else Path(__file__).resolve().parent
    root=folder.parents[2]
    manifest=read(folder/'manifest.json')
    for base,mapping in [(folder,manifest['files_sha256']),(root,manifest['dependencies']),(root,manifest['checkout_sources_sha256'])]:
        for name,digest in mapping.items():
            if sha(base/name)!=digest:raise ValueError('file hash differs: '+name)
    for name,digest in manifest['normalized_native_sources_sha256'].items():
        if hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:raise ValueError('normalized native source differs')
    source=root/'benchmarks/slam/flatwall02-prediction/verify.py'
    spec=importlib.util.spec_from_file_location('flatwall_target_reference_oracle',source)
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    protocol,freeze,closure,summary=[read(folder/name) for name in ['protocol.json','controller-freeze.json','estimates-receipt.json','summary.json']]
    if (protocol['schema']!='cloudanalyzer.flatwall_target_balance.v1' or protocol['rule']!='robust_weight / exact_target_multiplicity'
            or protocol['defaults_changed'] is not False or protocol['parameter_search'] is not False or protocol['total_candidate_frames']!=2560
            or closure['candidate_frames']!=2560 or closure['default_replays']!=2560 or summary['defaults_changed'] is not False
            or summary['physical_provenance_certified'] is not False or summary['all_estimates_closed_before_scoring'] is not True or summary['speed_claim'] is not False):raise ValueError('trial scope differs')
    if (freeze['protocol_sha256']!=sha(folder/'protocol.json') or closure['protocol_sha256']!=freeze['protocol_sha256']
            or closure['controller_freeze_sha256']!=sha(folder/'controller-freeze.json')):raise ValueError('freeze binding differs')
    invocation=read(folder/'scoring-invocation-freeze.json')
    if invocation['controller_freeze_sha256']!=sha(folder/'controller-freeze.json') or invocation['estimates_receipt_sha256']!=sha(folder/'estimates-receipt.json') or invocation['failure_before_reference_read'] is not True:raise ValueError('scoring invocation freeze differs')
    for name,digest in manifest['checkout_sources_sha256'].items():
        if freeze['files_sha256'].get(name,invocation['source_files_sha256'].get(name))!=digest:raise ValueError('source freeze differs')
    build=read(folder/'native-build-inputs.json');artifact=read(folder/'native-artifact.json')
    if build['normalized_sources_sha256']!=manifest['normalized_native_sources_sha256'] or artifact['binary_sha256']!=protocol['native_binary_sha256'] or artifact['installed_extension_replaced'] is not False:raise ValueError('native identity differs')
    boundary=read(folder/'scoring-boundary.json')
    if (boundary['all_candidate_frames_closed']!=2560 or boundary['all_default_replays_closed']!=2560
            or boundary['reference_poses_read_in_this_trial_before_boundary'] is not False or boundary['estimates_receipt_sha256']!=sha(folder/'estimates-receipt.json')):raise ValueError('scoring boundary differs')
    receipt_keys=[k for k in boundary['files_sha256'] if k.endswith('/receipt.json')]
    if len(receipt_keys)!=1:raise ValueError('boundary closure identity differs')
    prefix=receipt_keys[0][:-len('receipt.json')]
    expected_boundary={prefix+name:digest for name,digest in closure['output_files_sha256'].items()}
    expected_boundary[prefix+'receipt.json']=sha(folder/'estimates-receipt.json')
    if boundary['files_sha256']!=expected_boundary:raise ValueError('complete estimate boundary inventory differs')
    score_receipt=read(folder/'scoring-receipt.json')
    if score_receipt['estimates_receipt_sha256']!=sha(folder/'estimates-receipt.json') or score_receipt['input_hashes_unchanged'] is not True:raise ValueError('scoring closure input differs')
    for name,digest in score_receipt['output_files_sha256'].items():
        if sha(folder/name)!=digest:raise ValueError('scoring output closure differs')
    for name in ['default_pose_and_existing_snapshot_array_bits_match','default_report_fields_match','source_and_insertion_selection_and_transforms_checked',
                 'candidate_map_insertion_and_target_membership_replayed','all_final_pose_effective_weights_and_H_b_checked']:
        if closure[name] is not True:raise ValueError('control validation scope differs')
    if closure['candidate_forward_search_independently_replayed'] is not False or closure['reference_poses_used'] is not False:raise ValueError('unsupported control claim')
    for case in protocol['cases']:
        for name in ['trajectory.npz','default-replay-trajectory.npz','frames.jsonl']:
            key=case['name']+'/'+name
            if closure['output_files_sha256'][key]!=sha(folder/key):raise ValueError('estimate closure differs')
    catalog=read(folder/'snapshot-catalog.json')
    lookup={(r['case'],r['scan_index']):r for r in catalog}
    cases=[f'{r}/{p}/{a}' for r in ['flatwall_01','flatwall_02'] for p in POLICIES for a in ARMS]
    if [c['name'] for c in protocol['cases']]!=cases or set(summary['cases'])!=set(cases) or len(lookup)!=len(catalog) or len(catalog)!=2560:raise ValueError('case/catalog accounting differs')
    comparison=read(folder/'comparison.json');comparisons={r['case']:r for r in comparison['cases']}
    if len(comparisons)!=16 or set(comparisons)!=set(cases):raise ValueError('comparison case inventory differs')
    total,samples,sample_pairs=0,0,0
    for case,definition in zip(cases,protocol['cases']):
        recording,policy,arm=case.split('/')
        count=185 if recording=='flatwall_01' else 135
        ids=np.array(definition['scan_indices'],dtype=np.int64)
        if ids.tolist()!=list(range(1,count+1)):raise ValueError('original cohort differs')
        if recording=='flatwall_01' and policy=='repeat-motion':
            oldbase=root/'benchmarks/slam/flatwall-frontend-constraints';old=oldbase/arm/'frames.jsonl'
        else:
            oldbase=root/('benchmarks/slam/flatwall-prediction-policy' if recording=='flatwall_01' else 'benchmarks/slam/flatwall02-prediction');old=oldbase/policy/arm/'frames.jsonl'
        old_catalog={r['scan_index']:r for r in read(oldbase/'snapshot-catalog.json') if r['arm']==arm and (recording=='flatwall_01' or r['policy']==policy)}
        original=rows(old);times=np.array([r['reference_ns'] for r in original],dtype=np.int64)
        with np.load(folder/case/'trajectory.npz',allow_pickle=False) as d:
            np.testing.assert_array_equal(d['original_scan_indices'],ids);np.testing.assert_array_equal(d['timestamps_ns'],times);entire=d['poses'].copy()
        with np.load(folder/case/'default-replay-trajectory.npz',allow_pickle=False) as d:
            np.testing.assert_array_equal(d['original_scan_indices'],ids);np.testing.assert_array_equal(d['timestamps_ns'],times);control=d['poses'].copy()
        np.testing.assert_array_equal(control.view(np.uint64),np.array([r['returned_pose'] for r in original]).view(np.uint64))
        oracle.poses(entire)
        paired,reference,coverage,ref_stamps=oracle.reference_pairs(folder/recording/'reference-source.txt',times)
        member=read(folder/recording/'reference-member.json')
        if (member['member']!='gt/'+recording+'.txt' or member['sha256']!=sha(folder/recording/'reference-source.txt') or member['rows']!=len(ref_stamps)
                or member['first_timestamp_ns']!=ref_stamps[0] or member['last_timestamp_ns']!=ref_stamps[-1]):raise ValueError('reference identity differs')
        expected_paired=183 if recording=='flatwall_01' else 114
        if len(paired)!=expected_paired:raise ValueError('inherited reference coverage differs')
        with np.load(folder/case/'paired-reference.npz',allow_pickle=False) as d:
            np.testing.assert_array_equal(d['cohort_indices'],paired);np.testing.assert_array_equal(d['scan_indices'],ids[paired]);np.testing.assert_array_equal(d['timestamps_ns'],times[paired]);oracle.close(d['reference'],reference)
        actual_coverage=read(folder/case/'coverage.json')
        if len(actual_coverage)!=count:raise ValueError('coverage row count differs')
        for i,(actual,expected) in enumerate(zip(actual_coverage,coverage)):
            if actual['estimate_index']!=i:raise ValueError('coverage index differs')
            for key,value in expected.items():
                if isinstance(value,float):oracle.close(actual[key],value)
                elif actual[key]!=value:raise ValueError('exact-time association differs')
        result=read(folder/case/'metrics.json')
        if result!=summary['cases'][case]:raise ValueError('metric summary binding differs')
        errors=check_metrics(result,entire[paired],reference,ids[paired],times[paired],oracle)
        with np.load(folder/case/'errors.npz',allow_pickle=False) as d:
            if set(d.files)!=set(errors):raise ValueError('error inventory differs')
            for key,value in errors.items():oracle.close(d[key],value,1e-5 if key.endswith('deg') else 2e-8)
        prior_summary=read(root/('benchmarks/slam/flatwall-prediction-policy/summary.json' if recording=='flatwall_01' else 'benchmarks/slam/flatwall02-prediction/summary.json'))
        default=prior_summary['controls'][policy][arm]
        check_metrics(default,control[paired],reference,ids[paired],times[paired],oracle,full=False)
        for key in ['rigid_position_ape_m','origin_position_ape_m','consecutive_rpe_translation_m','consecutive_rpe_rotation_deg']:
            before=default['metrics'][key]['rmse'] if default['metrics'].get(key) else None
            after=result['metrics'][key]['rmse'] if result['metrics'].get(key) else None
            entry=comparisons[case]['metrics'][key]
            if before is None or after is None:
                if entry['default']!=before or entry['balanced']!=after or entry['delta'] is not None:raise ValueError('degenerate comparison differs')
            else:
                oracle.close(entry['default'],before);oracle.close(entry['balanced'],after);oracle.close(entry['delta'],after-before)
                if before>0:oracle.close(entry['percent_change'],100*(after-before)/before)
                elif entry['percent_change'] is not None:raise ValueError('zero-baseline percentage differs')
        frames=rows(folder/case/'frames.jsonl')
        if [r['scan_index'] for r in frames]!=ids.tolist():raise ValueError('frame identities differ')
        for i,(row,oldrow) in enumerate(zip(frames,original)):
            predicted=np.eye(4) if i==0 else (entire[i-1] if policy=='last-pose' or i==1 else entire[i-1]@(oracle.inverse(entire[i-2])@entire[i-1]))
            original_digest=next(v['sha256'] for k,v in old_catalog[int(ids[i])]['files'].items() if k.endswith('.npz'))
            if row['original_snapshot_sha256']!=original_digest:raise ValueError('original snapshot identity differs')
            if row['case']!=case or row['timestamp_ns']!=int(times[i]) or row['target_balance'] is not True or row['source_points']!=oldrow['source_points']:raise ValueError('frame policy/source identity differs')
            oracle.close(row['returned_pose'],entire[i]);oracle.close(row['prediction_pose'],predicted);oracle.close(row['center_world'],predicted[:3,3])
            if row['registration']['registration_points']!=row['source_points'] or not 0<=row['unique_targets']<=row['matches']<=row['source_points'] or not 0<=row['weight_sum']<=row['robust_weight_sum']+2e-8:raise ValueError('support accounting differs')
            h=np.array(row['normal_matrix']);b=np.array(row['right_hand_side'])
            if h.shape!=(6,6) or b.shape!=(6,):raise ValueError('invalid system dimensions')
            oracle.close(h,h.T);oracle.close(b,b)
            if np.linalg.eigvalsh(h)[0]<-2e-8:raise ValueError('nonpositive system')
            if row['geometry'] is not None:
                d=np.diag([1.,1.,1.,1/row['radius_m'],1/row['radius_m'],1/row['radius_m']]);v=np.linalg.eigvalsh(d@((h+h.T)/2)@d/row['weight_sum'])
                oracle.close(row['geometry']['eigenvalues'],v);oracle.close(row['geometry']['min_max_ratio'],v[0]/v[-1])
                if row['geometry']['weak_directions_below_1e_3']!=int(sum(v/v[-1]<1e-3)):raise ValueError('weak spectrum count differs')
            elif row['weight_sum']>0 and row['radius_m'] is not None and row['radius_m']>0:raise ValueError('missing spectrum')
            if not row['alignment_eligible']:
                if row['matches'] or np.any(h) or np.any(b) or row['geometry'] is not None:raise ValueError('unsupported alignment has support')
                oracle.close(row['returned_pose'],predicted)
            if not 0<=row['matrix_oracle_error']<=2e-8 or not 0<=row['rhs_oracle_error']<=2e-8:raise ValueError('all-frame arithmetic oracle exceeded tolerance')
            item=lookup[(case,int(ids[i]))];chosen=i in {1,count//2,count-1}
            if item['public_sample'] is not chosen:raise ValueError('sample selection differs')
            expected_names={f'{case}/scan-{int(ids[i]):03d}.{ext}' for ext in ['npz','json']}
            if set(item['files'])!=expected_names:raise ValueError('raw inventory differs')
            for name,info in item['files'].items():
                if closure['output_files_sha256'].get(name)!=info['sha256']:raise ValueError('raw receipt binding differs')
                if chosen and (sha(folder/name)!=info['sha256'] or (folder/name).stat().st_size!=info['bytes']):raise ValueError('sample bytes differ')
            if chosen:
                prefix=folder/case/f'scan-{int(ids[i]):03d}'
                sample_pairs+=check_sample(prefix.with_suffix('.npz'),read(prefix.with_suffix('.json')),row,oracle);samples+=1
        diagnostic=dict(scans=len(frames),termination_counts=dict(Counter(r['registration']['termination'] for r in frames)),matches=sum(r['matches'] for r in frames),unique_targets=sum(r['unique_targets'] for r in frames),
            supported_systems=sum(r['geometry'] is not None for r in frames),weak_systems=sum(r['geometry'] is not None and r['geometry']['weak_directions_below_1e_3']>0 for r in frames))
        if diagnostic!=summary['diagnostics'][case]:raise ValueError('diagnostic summary differs')
        total+=len(frames)
    if total!=2560 or samples!=48:raise ValueError('total accounting differs')
    position=[r['metrics']['rigid_position_ape_m']['delta'] for r in comparisons.values()]
    rpe=[r['metrics']['consecutive_rpe_translation_m']['delta'] for r in comparisons.values()]
    fine_gyro=[r for r in comparison['cases'] if r['case'].endswith('/gyro-0.1875')]
    expected_counts=dict(cases=16,position_improved=sum(v<0 for v in position),position_regressed=sum(v>0 for v in position),position_equal=sum(v==0 for v in position),
        rpe_translation_improved=sum(v<0 for v in rpe),rpe_translation_regressed=sum(v>0 for v in rpe),fine_gyro_cases=len(fine_gyro),
        fine_gyro_position_improved=sum(r['metrics']['rigid_position_ape_m']['delta']<0 for r in fine_gyro))
    if comparison['summary']!=expected_counts:raise ValueError('cross-case comparison counts differ')
    return dict(cases=16,candidate_frames=total,default_replay_pose_bits_checked=2560,public_raw_samples=samples,public_sample_pairs=sample_pairs,
        exact_reference_associations_checked=True,all_trajectory_metrics_and_frame_spectra_checked=True,defaults_changed=False,physical_provenance_certified=False)


if __name__=='__main__':print(json.dumps(verify(),indent=2))
