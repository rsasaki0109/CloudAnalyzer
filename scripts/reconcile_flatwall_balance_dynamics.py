#!/usr/bin/env python3
"""Separate native JSON-input validation bits from original binary pose parity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

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


def load(path):
    with np.load(path,allow_pickle=False) as saved:
        return {k:saved[k].copy() for k in saved.files}


def reconcile(study, out):
    study, out = study.resolve(), out.resolve()
    if out.exists():
        raise ValueError('use a fresh reconciliation directory')
    previous = study/'analysis-01'
    closure = read(previous/'receipt.json')
    inputs = {}
    for name,digest in closure['output_files_sha256'].items():
        path = previous/name
        if sha(path) != digest:
            raise ValueError('original closed analysis changed')
        inputs[path.relative_to(ROOT).as_posix()] = digest
    for path in [previous/'receipt.json', study/'expected-input-bits-01.json', study/'prepared-01/input.json',
                 Path(__file__).resolve(), ROOT/'rust/crates/ca-core/examples/audit_odometry_pose_inputs.rs']:
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    original = read(study/'prepared-01/input.json')['cases']
    exported = read(study/'expected-input-bits-01.json')
    if exported['estimator_uses_expected_poses'] is not False:
        raise ValueError('expected poses must never initialize the solver')
    parsed = dict(offsets=np.r_[0,np.cumsum([len(c['frames']) for c in original])].astype(np.int64))
    for balance in ['baseline','balanced']:
        parsed[balance+'_bits'] = np.asarray([f[balance+'_bits'] for c in exported['cases'] for f in c['frames']],dtype=np.uint64).reshape(-1,4,4)
    for actual, expected in zip(exported['cases'],original):
        if actual['case'] != expected['name'] or [r['scan_index'] for r in actual['frames']] != [r['scan_index'] for r in expected['frames']]:
            raise ValueError('parser probe cohort differs')
    out.mkdir(parents=True)
    save(out/'invocation.json',dict(inputs_sha256=inputs, changes_estimation=False, changes_timings=False))
    with (out/'expected-input-bits.npz').open('xb') as stream:
        np.savez_compressed(stream,**parsed)
    counts = {}
    for mode in ['trace','time']:
        data = load(previous/(mode+'-runs.npz'))
        data['native_input_pose_max_abs_error'] = data.pop('pose_max_abs_error')
        data['native_input_pose_bits_match'] = data.pop('pose_bits_match')
        corrected_error, corrected_bits = [], []
        for row,case_index in enumerate(data['case_index']):
            start,end = data['offsets'][row:row+2]
            balance = 'balanced' if data['target_balance'][row] else 'baseline'
            expected = np.asarray([f[balance]['pose'] for f in original[case_index]['frames']]).reshape(-1,4,4)
            pstart,pend = parsed['offsets'][case_index:case_index+2]
            probe_bits = parsed[balance+'_bits'][pstart:pend]
            probe = probe_bits.view(np.float64)
            actual = data['poses'][start:end]
            np.testing.assert_allclose(probe,expected,rtol=0,atol=2e-15)
            np.testing.assert_array_equal(data['native_input_pose_max_abs_error'][start:end],np.max(np.abs(actual-probe),axis=(1,2)))
            np.testing.assert_array_equal(data['native_input_pose_bits_match'][start:end],np.all(actual.view(np.uint64)==probe_bits,axis=(1,2)))
            corrected_error.extend(np.max(np.abs(actual-expected),axis=(1,2)))
            corrected_bits.extend(np.all(actual.view(np.uint64)==expected.view(np.uint64),axis=(1,2)))
        data['pose_max_abs_error'] = np.asarray(corrected_error)
        data['pose_bits_match'] = np.asarray(corrected_bits,dtype=bool)
        with (out/(mode+'-runs.npz')).open('xb') as stream:
            np.savez_compressed(stream,**data)
        counts[mode] = dict(frames=len(corrected_bits), original_pose_bit_matches=int(np.sum(corrected_bits)),
            native_parsed_input_bit_matches=int(data['native_input_pose_bits_match'].sum()),
            maximum_original_pose_abs_error=float(np.max(corrected_error)))
    summary = read(previous/'summary.json')
    for trace in summary['traces']:
        destination = out/trace['file']
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(previous/trace['file'],destination)
    summary['pose_diagnostics'] = dict(reconciled=True, counts=counts,
        distinction='The executed Rust check compared against its deserialized decimal JSON expected poses. The independent check compares actual output poses against original binary poses. Both records are retained; expected poses never drive registration.',
        estimation_changed=False, timings_changed=False)
    save(out/'summary.json',summary)
    for name,digest in inputs.items():
        if sha(ROOT/name) != digest:
            raise ValueError('reconciliation input changed')
    save(out/'receipt.json',dict(inputs_unchanged=True,invocation_sha256=sha(out/'invocation.json'),
        output_files_sha256={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.study,args.out)))
