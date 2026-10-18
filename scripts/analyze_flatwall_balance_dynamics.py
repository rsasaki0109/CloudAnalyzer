#!/usr/bin/env python3
"""Close independent native-step checks and every counterbalanced timing sample."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import flatwall_balance_dynamics as independent

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False,
            default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist())
        stream.write('\n')


def compact_trace(raw):
    result = {k:np.asarray(raw[k], dtype=np.float64) for k in ['source_at_prediction', 'map_points_world',
        'center_world', 'threshold_sq', 'kernel', 'final_source_world', 'final_target_world', 'final_weights']}
    result['returned_pose'] = np.asarray(raw['returned_pose']).reshape(4, 4)
    result['final_source_indices'] = np.asarray(raw['final_source_indices'], dtype=np.int64)
    states = raw['states']
    for key in ['before', 'after', 'normal_matrix', 'right_hand_side', 'weighted_cost']:
        result[key] = np.asarray([s[key] for s in states], dtype=np.float64)
    for key in ['before', 'after']:
        result[key] = result[key].reshape(-1, 4, 4)
    for key in ['matches', 'unique_targets']:
        result[key] = np.asarray([s[key] for s in states], dtype=np.int64)
    tail = states[-24:]
    result['tail_offsets'] = np.r_[0, np.cumsum([len(s['source_indices']) for s in tail])].astype(np.int64)
    for key in ['source_indices', 'target_indices', 'multiplicity', 'weights']:
        result['tail_'+key] = np.concatenate([np.asarray(s[key], dtype=np.float64 if key == 'weights' else np.int64) for s in tail])
    return result


def compact_runs(raw, cases):
    runs = raw['runs']
    names = [c['name'] for c in cases]
    result = dict(case_index=np.array([names.index(r['case']) for r in runs], dtype=np.int64),
        target_balance=np.array([r['target_balance'] for r in runs]),
        round=np.array([r['round'] for r in runs], dtype=np.int64),
        warmup=np.array([r['warmup'] for r in runs]),
        total_registration_ns=np.array([r['total_registration_ns'] for r in runs], dtype=np.int64),
        offsets=np.r_[0,np.cumsum([len(r['frames']) for r in runs])].astype(np.int64))
    rows = [f for r in runs for f in r['frames']]
    result['poses'] = np.asarray([f['pose'] for f in rows]).reshape(-1, 4, 4)
    for key in ['scan_index', 'iterations']:
        result[key] = np.asarray([f[key] for f in rows], dtype=np.int64)
    for key in ['pose_max_abs_error', 'threshold_m']:
        result[key] = np.asarray([f[key] for f in rows], dtype=np.float64)
    result['pose_bits_match'] = np.asarray([f['pose_bits_match'] for f in rows], dtype=bool)
    result['termination'] = np.asarray([f['termination'] for f in rows])
    if raw['mode'] == 'time':
        result['registration_ns'] = np.asarray([t for r in runs for t in r['registration_ns']], dtype=np.int64)
        if len(result['registration_ns']) != len(rows):
            raise ValueError('missing frame timings')
    return result


def timing_summary(arrays, cases):
    rows = []
    for i, case in enumerate(cases):
        samples = []
        for round_index in [1, 2, 3]:
            pair = []
            for balance in [False, True]:
                hit = np.flatnonzero((arrays['case_index'] == i) & (arrays['round'] == round_index) & (arrays['target_balance'] == balance))
                if len(hit) != 1:
                    raise ValueError('incomplete paired timing cohort')
                row = int(hit[0])
                if arrays['warmup'][row]:
                    raise ValueError('warmup included in measured pairs')
                start, end = arrays['offsets'][row:row+2]
                if arrays['total_registration_ns'][row] != arrays['registration_ns'][start:end].sum():
                    raise ValueError('case timing sum differs')
                pair.append(int(arrays['total_registration_ns'][row]))
            samples.append(dict(round=round_index, baseline_ns=pair[0], balanced_ns=pair[1], balanced_over_baseline=pair[1]/pair[0]))
        baseline = [s['baseline_ns'] for s in samples]
        balanced = [s['balanced_ns'] for s in samples]
        rows.append(dict(case=case['name'], scans=len(case['frames']), measured_pairs=samples,
            baseline_median_ns=float(np.median(baseline)), balanced_median_ns=float(np.median(balanced)),
            baseline_range_ns=[min(baseline),max(baseline)], balanced_range_ns=[min(balanced),max(balanced)],
            balanced_over_baseline_median_ratio=float(np.median(balanced)/np.median(baseline)),
            paired_ratio_median=float(np.median([s['balanced_over_baseline'] for s in samples]))))
    return dict(cases=rows, measured_pairs_per_case=3,
        sum_case_medians_balanced_over_baseline=sum(r['balanced_median_ns'] for r in rows)/sum(r['baseline_median_ns'] for r in rows),
        case_median_ratio_range=[min(r['balanced_over_baseline_median_ratio'] for r in rows),max(r['balanced_over_baseline_median_ratio'] for r in rows)],
        faster_cases=sum(r['balanced_median_ns'] < r['baseline_median_ns'] for r in rows),
        scope='Native registration policy cost on this host and the complete original real cohort. No diagnostic observer or capture inside measured calls; input preparation, bag I/O and gyro deskew excluded.')


def analyze(study, out):
    study, out = study.resolve(), out.resolve()
    if out.exists():
        raise ValueError('use a fresh analysis directory')
    inputs = {}
    for name in ['trace-01', 'time-01']:
        folder = study/name
        receipt = read(folder/'receipt.json')
        if not receipt['input_hashes_unchanged']:
            raise ValueError('native stage did not close')
        inputs[(folder/'receipt.json').relative_to(ROOT).as_posix()] = sha(folder/'receipt.json')
        for relative, digest in receipt['output_files_sha256'].items():
            path = folder/relative
            if sha(path) != digest:
                raise ValueError('closed native output changed')
            inputs[path.relative_to(ROOT).as_posix()] = digest
    for path in [Path(__file__).resolve(), Path(independent.__file__).resolve(), study/'protocol.json', study/'prepared-01/input.json']:
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    out.mkdir(parents=True)
    save(out/'invocation.json', dict(inputs_sha256=inputs, state_feedback=False, reference_poses_read=False))
    cases = read(study/'prepared-01/input.json')['cases']
    traces, total = [], 0
    for case in cases:
        for index in case['trace_indices']:
            path = study/'trace-01'/case['name']/f'scan-{index:03d}.json'
            raw = read(path)
            if raw['termination'] != 'iteration_limit':
                raise ValueError('selected frame no longer capped')
            arrays = compact_trace(raw)
            result = independent.verify_trace(arrays)
            destination = out/case['name']/f'scan-{index:03d}.npz'
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            traces.append(dict(case=case['name'], scan_index=index, source_json_sha256=sha(path),
                file=destination.relative_to(out).as_posix(), sha256=sha(destination), **result))
            total += result['native_steps_checked']
            print(json.dumps(dict(verified_trace=case['name'], scan_index=index, native_steps=total)), flush=True)
    if len(traces) != 8 or total != 4000:
        raise ValueError('incomplete capped-frame audit')
    for mode in ['trace', 'time']:
        arrays = compact_runs(read(study/(mode+'-01')/'runs.json'), cases)
        with (out/(mode+'-runs.npz')).open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        if mode == 'time':
            timing = timing_summary(arrays, cases)
    summary = dict(schema='cloudanalyzer.flatwall_balance_dynamics_result.v1', capped_frames=8, native_steps_checked=total,
        traces=traces, timing=timing, candidate_changed=False, defaults_changed=False, state_feedback=False,
        reference_poses_read=False, runtime_improvement_adopted=False)
    save(out/'summary.json', summary)
    for name, digest in inputs.items():
        if sha(ROOT/name) != digest:
            raise ValueError('analysis input changed')
    save(out/'receipt.json', dict(inputs_unchanged=True, invocation_sha256=sha(out/'invocation.json'),
        output_files_sha256={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return dict(capped_frames=8, native_steps_checked=total, timing_ratio=timing['sum_case_medians_balanced_over_baseline'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.study, args.out)))
