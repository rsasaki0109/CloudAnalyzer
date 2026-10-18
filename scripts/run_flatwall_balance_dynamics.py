#!/usr/bin/env python3
"""Run a fixed native trace or timing cohort and close its exact output inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def execute(study, stage, out):
    study, out = study.resolve(), out.resolve()
    if out.exists() or out.parent != study:
        raise ValueError('use a fresh direct child of the study directory')
    frozen = read(study/'controller-freeze.json')
    if sha(study/'protocol.json') != frozen['protocol_sha256']:
        raise ValueError('protocol changed')
    expected = dict(frozen['files_sha256'])
    artifact = read(study/'native-artifact.json')
    binary = study/'native/audit_odometry_steps.exe'
    if sha(binary) != artifact['sha256']:
        raise ValueError('unexpected observer binary')
    expected[binary.relative_to(ROOT).as_posix()] = sha(binary)
    for name in ['scripts/run_flatwall_balance_dynamics.py', 'scripts/flatwall_balance_dynamics.py']:
        expected[name] = sha(ROOT/name)
    for name, digest in expected.items():
        if sha(ROOT/name) != digest:
            raise ValueError('frozen input changed: '+name)
    save(study/(out.name+'-invocation.json'), dict(stage=stage, inputs_sha256=expected,
        protocol_sha256=frozen['protocol_sha256'], native_artifact_sha256=sha(study/'native-artifact.json')))
    log = study/(out.name+'.log')
    with log.open('xb') as stream:
        result = subprocess.run([str(binary), stage, str(study/'prepared-01/input.json'), str(out)],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        save(study/(out.name+'-failure.json'), dict(stage=stage, returncode=result.returncode, log_sha256=sha(log)))
        raise RuntimeError('native replay failed; complete failure log retained at '+str(log))
    for name, digest in expected.items():
        if sha(ROOT/name) != digest:
            raise ValueError('frozen input changed: '+name)
    save(out/'receipt.json', dict(stage=stage, invocation_sha256=sha(study/(out.name+'-invocation.json')),
        input_hashes_unchanged=True, log_sha256=sha(log),
        output_files_sha256={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    return dict(stage=stage, status='closed', files=len(read(out/'receipt.json')['output_files_sha256']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['trace', 'time'])
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(execute(args.study, args.stage, args.out)))
    except Exception:
        sys.exit(__import__('traceback').print_exc() or 1)
