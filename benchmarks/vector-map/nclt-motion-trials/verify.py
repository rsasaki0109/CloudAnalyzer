#!/usr/bin/env python3
"""Verify alternative-motion evidence with standard-library arithmetic only."""
import bisect
import hashlib
import json
import math
import runpy
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-9, (a, b)


def rmse(values):
    return math.sqrt(sum(v*v for v in values) / len(values))


def rpe(poses, truth):
    return [math.sqrt(sum(((b[j]-a[j])-(d[j]-c[j]))**2 for j in range(3)))
            for a, b, c, d in zip(poses, poses[1:], truth, truth[1:])]


def read(path):
    return json.loads(path.read_text())


def verify():
    root = Path(__file__).resolve().parent
    references = root.parent / 'nclt-reference-accuracy'
    subprocess.run([sys.executable, str(root.parent / 'nclt-trajectory-error-regions/verify.py')], check=True)
    helpers = runpy.run_path(str(references / 'verify.py'))
    text_rows, transform, quaternion_matrix = (helpers[key] for key in ('text_rows', 'transformed', 'quaternion_matrix'))
    expected_files = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        sha, filename = line.split('  ', 1)
        assert filename not in expected_files and (root / filename).resolve().is_relative_to(root)
        assert digest(root / filename) == sha, filename
        expected_files[filename] = sha
    actual_files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'SHA256SUMS'}
    assert set(expected_files) == actual_files
    receipt, local_receipt = read(root / 'receipt.json'), read(root / 'comparison-receipt.json')
    old_receipt = read(references / 'receipt.json')
    assert receipt['schema'] == 'cloudanalyzer.nclt_motion_trial_receipt.v1'
    assert local_receipt['schema'] == 'cloudanalyzer.nclt_motion_comparison_receipt.v1'
    for session, entry in receipt['sessions'].items():
        assert entry['baseline'] == old_receipt['sessions'][session]['results']['prefix-30']
        assert entry['unchanged_baseline_files'] == old_receipt['sessions'][session]['unchanged_job_files']
        assert entry['unchanged_baseline_manifest_sha256'] == old_receipt['sessions'][session]['unchanged_manifest_sha256']
        base = read(references / session / 'comparison-prefix-30.json')
        original = text_rows(references / session / 'original.tum')
        reference = text_rows(references / session / 'reference.tum')
        reference_times = [r[0] for r in reference]
        ids = base['coverage']['matched_original_frame_ids']
        fit_ids, evaluated_ids = base['coverage']['alignment_original_frame_ids'], base['coverage']['evaluated_original_frame_ids']
        truths = {}
        for frame in ids:
            time = original[frame][0]
            right = bisect.bisect_left(reference_times, time)
            lo, hi = reference[right-1], reference[right]
            assert time-lo[0] <= .05 and hi[0]-time <= .05
            fraction = (time-lo[0])/(hi[0]-lo[0])
            truths[frame] = [(1-fraction)*a+fraction*b for a, b in zip(lo[1:4], hi[1:4])]
        assert set(entry['trials']) == {'loops-only', 'gravity-only'}
        for policy_name, saved in entry['trials'].items():
            folder = root / session / policy_name
            job, trial, processing = read(folder/'job.json'), read(folder/'trial.json'), read(folder/'pointcloud-report.json')
            report, response = read(folder/'comparison-prefix-30.json'), read(folder/'mcp-trial.json')
            evaluation, scans = read(folder/'mcp-evaluation.json'), read(folder/'scans.json')
            for artifact, filename in ((saved['job'],'job.json'), (saved['trial_report'],'trial.json'), (saved['comparison'],'comparison-prefix-30.json')):
                assert digest(folder/filename) == artifact['sha256'] and (folder/filename).stat().st_size == artifact['bytes']
            assert response['job'] == saved['job'] and response['report'] == saved['trial_report']
            assert evaluation['report'] == saved['comparison']
            assert trial['status'] == response['status'] == 'ready_unverified'
            assert job['status'] == 'pointcloud_ready' and job['attempts'] == [] and job['selected'] is None
            assert job['max_attempts'] == 4 and job['pointcloud']['quality_status'] == 'generated_unverified'
            policy = {'find_loops': policy_name == 'loops-only', 'use_gravity': policy_name == 'gravity-only'}
            assert trial['policy'] == saved['policy'] == policy
            assert trial['protocol']['reference_used_for_generation'] is False
            assert trial['protocol']['old_hd_geometry_or_audits_inherited'] is False
            assert trial['protocol']['expanded_fusion_frames_inherited'] is False
            assert trial['protocol']['baseline_adopted'] is False
            assert trial['retained_original_frame_ids'] == ids
            assert trial['excluded_original_frame_ids'] == [i for i in range(len(original)) if i not in ids]
            assert trial['inputs']['baseline_job'] == base['inputs']['job']
            assert trial['inputs']['source'] == job['source'] == base['inputs']['source'] == scans['source']
            assert job['pointcloud']['source_motion']['trajectory'] == base['inputs']['source_motion_trajectory']
            assert ('loops' in processing) == policy['find_loops']
            assert ('gravity' in processing) == policy['use_gravity']
            assert processing['map_points'] > 0 and processing['nodes'] == processing['scans'] == len(ids)
            assert len(scans['frames']) == len(original) and len(scans['timestamps_s']) == len(original)
            assert sum(a['bytes'] for a in scans['frames'])//16 <= 5_000_000
            for i, artifact in enumerate(scans['frames']):
                assert Path(artifact['path']).name == f'frame_{i:06d}.bin'
                assert trial['generated_inputs'][f'scan_{i}'] == artifact
                close(scans['timestamps_s'][i], original[i][0])
            for key, filename in (('initial_graph','initial.g2o'), ('scans_manifest','scans.json')):
                assert digest(folder/filename) == trial['generated_inputs'][key]['sha256']
            if policy['use_gravity']:
                assert digest(folder/'gravity.txt') == trial['generated_inputs']['gravity']['sha256']
                assert set(ids) <= {int(line.split()[0]) for line in (folder/'gravity.txt').read_text().splitlines()}
                assert processing['gravity']['tied'] == len(ids)
            initial_rows = [line.split() for line in (folder/'initial.g2o').read_text().splitlines()]
            initial = [row for row in initial_rows if row and row[0] == 'VERTEX_SE3:QUAT']
            assert [int(row[1]) for row in initial] == ids
            edges = [row for row in initial_rows if row and row[0] == 'EDGE_SE3:QUAT']
            assert [(int(row[1]),int(row[2])) for row in edges] == list(zip(ids,ids[1:]))
            for frame, row in zip(ids,initial):
                for a,b in zip(map(float,row[2:5]),original[frame][1:4]): close(a,b)
                # TUM quaternions are normalized on read; compare rotations, not
                # rounded quaternion components or their arbitrary sign.
                actual_rotation=quaternion_matrix([float(v) for v in row[5:9]])
                original_rotation=quaternion_matrix(original[frame][4:8])
                for j in range(3):
                    for k in range(3):close(actual_rotation[j][k],original_rotation[j][k])
            corrected = text_rows(folder/'corrected-kitti.txt')
            vertices = [line.split() for line in (folder/'corrected.g2o').read_text().splitlines() if line.startswith('VERTEX_SE3:QUAT')]
            assert [int(row[1]) for row in vertices] == ids and len(corrected) == len(ids)
            for vertex, row in zip(vertices,corrected):
                for a,b in zip(map(float,vertex[2:5]),[row[3],row[7],row[11]]):close(a,b)
                rotation = quaternion_matrix([float(q) for q in vertex[5:9]])
                for j in range(3):
                    for k in range(3):close(rotation[j][k],row[4*j+k])
            for key, filename in (('job','job.json'),('pointcloud_graph','corrected.g2o'),('pointcloud_trajectory','corrected-kitti.txt')):
                assert digest(folder/filename) == report['inputs'][key]['sha256']
            assert report['inputs']['reference'] == base['inputs']['reference']
            assert report['reference_provenance'] == base['reference_provenance']
            assert report['coverage'] == base['coverage']
            assert fit_ids+evaluated_ids == ids and not set(fit_ids)&set(evaluated_ids)
            assert report['protocol']['held_out_alignment'] is True and report['protocol']['alignment_prefix_fraction'] == .3
            for estimate in ('original','corrected'):
                result = report['results'][estimate]
                matched = result['matched_trajectory']
                alignment = result['alignment']
                positions=[]
                for index,frame in enumerate(evaluated_ids):
                    row=original[frame] if estimate=='original' else corrected[ids.index(frame)]
                    raw=row[1:4] if estimate=='original' else [row[3],row[7],row[11]]
                    position=transform(raw,alignment['rotation_matrix'],alignment['translation'])
                    positions.append(position)
                    assert matched['timestamps'][index] == original[frame][0]
                    for a,b in zip(position,matched['estimated_positions'][index]):close(a,b)
                    for a,b in zip(truths[frame],matched['reference_positions'][index]):close(a,b)
                    close(math.dist(position,truths[frame]),matched['ate_errors'][index])
                close(rmse([math.dist(p,truths[f]) for p,f in zip(positions,evaluated_ids)]),result['ate']['rmse'])
                close(rmse(rpe(positions,[truths[f] for f in evaluated_ids])),result['rpe_translation']['rmse'])
            close(report['results']['corrected']['ate']['rmse'],saved['corrected_ate_rmse_m'])
            close(report['results']['corrected']['rpe_translation']['rmse'],saved['corrected_rpe_translation_rmse_m'])
            close(report['results']['original']['ate']['rmse'],entry['baseline']['original_ate_rmse_m'])
            close(saved['baseline_corrected_ate_rmse_m'],base['results']['corrected']['ate']['rmse'])
            close(saved['ate_rmse_m_trial_minus_baseline'],report['results']['corrected']['ate']['rmse']-base['results']['corrected']['ate']['rmse'])
            original_windows=[]
            for filename in saved['regression_pages']:
                page=read(folder/filename)
                assert page['report']==saved['comparison']
                original_windows+=page['windows']
            assert sorted(i for w in original_windows for i in w['original_frame_ids'])==evaluated_ids
            for window in original_windows:
                start=window['window_id']*12; stop=min(start+12,len(evaluated_ids))
                assert window['original_frame_ids']==evaluated_ids[start:stop]
                for estimate in ('original','corrected'):
                    matched=report['results'][estimate]['matched_trajectory']
                    positions=matched['estimated_positions'][start:stop]; truth=matched['reference_positions'][start:stop]
                    close(window['results'][estimate]['ate_rmse_m'],rmse([math.dist(p,q) for p,q in zip(positions,truth)]))
                    close(window['results'][estimate]['rpe_translation_rmse_m'],rmse(rpe(positions,truth)))
                close(window['ate_rmse_m_corrected_minus_original'],window['results']['corrected']['ate_rmse_m']-window['results']['original']['ate_rmse_m'])
            local = local_receipt['sessions'][session][policy_name]
            windows=[]
            for page_index,filename in enumerate(local['pages']):
                page=read(folder/filename)
                assert page['schema']=='cloudanalyzer.mapping_motion_trial_comparison.v1'
                assert page['baseline_report']==entry['baseline']['report'] and page['candidate_report']==saved['comparison']
                assert page['offset']==page_index*8 and page['total_windows']==12
                assert page['next_offset']==(8 if page_index==0 else None)
                assert page['coverage']=={k:base['coverage'][k] for k in page['coverage']}
                close(page['ate_rmse_m_candidate_minus_baseline'],saved['ate_rmse_m_trial_minus_baseline'])
                for name,data in (('baseline',base),('candidate',report)):
                    close(page['global_results'][name]['ate_rmse_m'],data['results']['corrected']['ate']['rmse'])
                    close(page['global_results'][name]['rpe_translation_rmse_m'],data['results']['corrected']['rpe_translation']['rmse'])
                windows+=page['windows']
            assert sorted(i for w in windows for i in w['original_frame_ids'])==evaluated_ids
            assert windows==sorted(windows,key=lambda w:(-w['ate_rmse_m_candidate_minus_baseline'],w['window_id']))
            assert local['worst_window']==windows[0]
            assert local['worsened_windows']==sum(w['ate_rmse_m_candidate_minus_baseline']>0 for w in windows)
            assert local['improved_windows']==sum(w['ate_rmse_m_candidate_minus_baseline']<0 for w in windows)
            base_rows=text_rows(references/session/'corrected-kitti.txt')
            for window in windows:
                start=window['window_id']*12; stop=min(start+12,len(evaluated_ids))
                assert window['original_frame_ids']==evaluated_ids[start:stop]
                assert window['evaluated_poses']==stop-start
                assert window['timestamp_range_s']==[original[evaluated_ids[start]][0],original[evaluated_ids[stop-1]][0]]
                for name,data,rows in (('baseline',base,base_rows),('candidate',report,corrected)):
                    matched=data['results']['corrected']['matched_trajectory']
                    positions=matched['estimated_positions'][start:stop]; truth=matched['reference_positions'][start:stop]
                    close(window[name]['metrics']['ate_rmse_m'],rmse([math.dist(p,q) for p,q in zip(positions,truth)]))
                    close(window[name]['metrics']['rpe_translation_rmse_m'],rmse(rpe(positions,truth)))
                    raw=[[rows[ids.index(f)][j] for j in (3,7,11)] for f in window['original_frame_ids']]
                    bounds=[min(p[0] for p in raw),min(p[1] for p in raw),max(p[0] for p in raw),max(p[1] for p in raw)]
                    for a,b in zip(window[name]['sensor_origin_bounds_xy'],bounds):close(a,b)
                close(window['ate_rmse_m_candidate_minus_baseline'],window['candidate']['metrics']['ate_rmse_m']-window['baseline']['metrics']['ate_rmse_m'])
            for name,data in (('baseline',base),('candidate',report)):
                close(math.sqrt(sum(w['evaluated_poses']*w[name]['metrics']['ate_rmse_m']**2 for w in windows)/len(evaluated_ids)),data['results']['corrected']['ate']['rmse'])
            print(session,policy_name,'generation links, pose/error arithmetic and matching local comparisons verified')
    print('NCLT alternative motion trial evidence verified; no map accuracy/adoption claim')


if __name__=='__main__':
    verify()
