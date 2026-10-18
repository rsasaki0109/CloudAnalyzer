#!/usr/bin/env python3
"""Prepare original covered real scans for a separate native replay audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(previous, out):
    previous, out = previous.resolve(), out.resolve()
    if out.exists():
        raise ValueError('use a fresh preparation directory')
    protocol = read(previous/'protocol.json')
    completion = read(previous/'completion.json')
    if completion['state'] != 'MERGED' or not completion['merged_tree_matches_tested_tree']:
        raise ValueError('require the completed original trial')
    if os.environ.get('RAYON_NUM_THREADS') != '4' or os.environ.get('OPENBLAS_NUM_THREADS') != '1':
        raise ValueError('set frozen threads before startup')
    original = read(previous/'estimates-01/receipt.json')
    for name, digest in original['output_files_sha256'].items():
        if sha(previous/'estimates-01'/name) != digest:
            raise ValueError('original estimates changed: '+name)
    package = ROOT/protocol['native_package']
    sys.path.insert(0, str(package.parent))
    import cloudanalyzer_core as core
    import cloudanalyzer_core._core as native
    if Path(native.__file__).resolve().parent != package or sha(Path(native.__file__)) != protocol['native_binary_sha256']:
        raise ValueError('unexpected preparation extension')
    out.mkdir(parents=True)
    inputs = {str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__))}
    for name in ['protocol.json', 'completion.json', 'estimates-01/receipt.json']:
        path = previous/name
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    data = dict(schema='cloudanalyzer.flatwall_balance_dynamics_input.v1', recordings=[], cases=[],
        points_per_scan=24000, warmup_pairs=1, measured_pairs=3, tail_states=24,
        candidate_changed=False, reference_poses_read=False,
        trace_selection='Every previously observed iteration-limit frame, without replacement or retuning.',
        timing_order='One warmup pair, then three measured pairs per case; alternate order by round plus case index.',
        timing_scope='Native registration on preloaded real scans, four requested Rayon threads; diagnostic capture/residual reports disabled. No full application or GPU measurement.',
        replay_tolerance='2e-8 maximum absolute pose/threshold error, exact iteration and termination equality; report exact pose bits separately.')
    trace_count = 0
    for case in protocol['cases']:
        name = case['name']
        folder = previous/'estimates-01'/name
        frames_path = folder/'frames.jsonl'
        frames = [json.loads(line) for line in frames_path.read_text(encoding='utf-8').splitlines()]
        baseline = [json.loads(line) for line in (ROOT/case['frames']).read_text(encoding='utf-8').splitlines()]
        with np.load(folder/'default-replay-trajectory.npz', allow_pickle=False) as arrays:
            baseline_poses = arrays['poses'].copy()
        recording, policy, arm = name.split('/')
        row = dict(name=name, recording=recording, policy=policy,
            compensation=protocol['arms'][arm]['compensation'],
            registration_voxel=protocol['arms'][arm]['registration_voxel'], frames=[], trace_indices=[])
        for i, (actual, old) in enumerate(zip(frames, baseline)):
            index = actual['scan_index']
            if index != case['scan_indices'][i] or index != old['scan_index']:
                raise ValueError('original scan order changed')
            report = actual['registration']
            row['frames'].append(dict(scan_index=index,
                baseline=dict(pose=baseline_poses[i].reshape(-1).tolist(), iterations=old['iterations'],
                    termination=old['termination'], threshold_m=old['threshold_m']),
                balanced=dict(pose=np.asarray(actual['returned_pose']).reshape(-1).tolist(),
                    iterations=report['iterations'], termination=report['termination'], threshold_m=report['threshold_m'])))
            if report['termination'] == 'iteration_limit':
                row['trace_indices'].append(index)
                trace_count += 1
        for path in [frames_path, ROOT/case['frames'], folder/'default-replay-trajectory.npz']:
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
        data['cases'].append(row)
    if len(data['cases']) != 16 or trace_count != 8:
        raise ValueError('original sixteen cases and eight capped frames required')
    for recording in protocol['recordings']:
        bag = ROOT/recording['bag']
        inputs[bag.relative_to(ROOT).as_posix()] = sha(bag)
        reader = core.BagReader(str(bag))
        imu = reader.imu_series('/livox/imu', acceleration_scale=9.80665)
        times = imu.timestamps_ns()
        rotation = imu.rotation_trajectory(int(times[0]), int(times[-1]), gyro_bias=(0.,0.,0.), max_gap_s=.05)
        raw_path, gyro_path = [out/(recording['name']+'-'+arm+'.f64') for arm in ['raw', 'gyro']]
        ids, skipped, seen = [], [], 0
        with raw_path.open('xb') as raw, gyro_path.open('xb') as gyro:
            for index, scan in enumerate(reader.scans('/livox/points', raw_time_field='t')):
                seen += 1
                xyz, stamps = scan.positions(), scan.point_times_ns(unit='ns', origin='header')
                reference = int(stamps.max())
                if stamps.min() < times[0] or reference > times[-1]:
                    skipped.append(index)
                    continue
                if xyz.shape != (24000, 3):
                    raise ValueError('original point count changed')
                compensated = rotation.deskew(xyz, stamps, reference, imu_from_lidar=np.eye(4), parallel=True)
                raw.write(xyz.astype('<f8').tobytes())
                gyro.write(compensated.astype('<f8').tobytes())
                ids.append(index)
        wanted = next(c['scan_indices'] for c in protocol['cases'] if c['name'].startswith(recording['name']+'/'))
        if ids != wanted or seen != recording['input_scan_count'] or skipped != recording['excluded_scan_indices']:
            raise ValueError('original coverage changed')
        data['recordings'].append(dict(name=recording['name'], raw=str(raw_path), gyro=str(gyro_path), scan_indices=ids))
        print(json.dumps(dict(prepared=recording['name'], scans=len(ids))), flush=True)
    save(out/'input.json', data)
    for path in out.iterdir():
        if path.is_file():
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    inputs[Path(native.__file__).resolve().relative_to(ROOT).as_posix()] = sha(Path(native.__file__))
    save(out/'receipt.json', dict(inputs_sha256=inputs, original_closed_candidate_frames=2560,
        trace_frames=trace_count, timing_cases=16, reference_poses_read=False))
    return dict(prepared_scans=sum(len(r['scan_indices']) for r in data['recordings']), trace_frames=trace_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.previous, args.out)))
