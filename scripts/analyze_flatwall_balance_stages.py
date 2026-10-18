#!/usr/bin/env python3
"""Verify and close all fixed-state native samples without selecting timings."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import platform

import flatwall_balance_stages as independent


def save(path, value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def analyze(root, study, runs_path, output):
    output.mkdir(exist_ok=False)
    protocol = independent.read(study/'timing-protocol.json')
    receipt = independent.read(runs_path/'receipt.json')
    if receipt['invocation_sha256'] != independent.sha(runs_path/'invocation.json') or not receipt['input_hashes_unchanged']:
        raise ValueError('timing stage did not close')
    files = {path.relative_to(root).as_posix():independent.sha(path) for path in [
        study/'timing-protocol.json',study/'timing-freeze.json',study/'native-artifact.json',
        study/'prepared-02/origins.json',study/'prepared-02/input.json',study/'prepared-02/receipt.json',
        runs_path/'invocation.json',runs_path/'receipt.json',Path(__file__),Path(independent.__file__),root/'scripts/flatwall_balance_dynamics.py']}
    for name,digest in receipt['output_files_sha256'].items():
        if independent.sha(runs_path/name) != digest:
            raise ValueError('timing output hash differs')
        files[(runs_path/name).relative_to(root).as_posix()] = digest
    files.update(independent.read(study/'prepared-02/receipt.json')['inputs_sha256'])
    save(output/'invocation.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),inputs_sha256=files))
    origins = independent.read(study/'prepared-02/origins.json')
    expected = independent.expected(root,origins)
    runs, checks = [],[]
    for i,threads in enumerate(protocol['process_thread_order']):
        path = runs_path/f'run-{i:02d}-threads-{threads}.json'
        run = independent.read(path)
        checks.append(independent.check_run(run,expected,threads))
        runs.append(run)
        with (output/(path.name+'.gz')).open('xb') as stream:
            stream.write(gzip.compress(path.read_bytes(),compresslevel=9,mtime=0))
    summary = independent.summarize(runs,origins)
    save(output/'summary.json',summary)
    save(output/'verification.json',dict(states=64,eligible_states=sum(o['alignment_eligible'] for o in origins),
        per_process=checks,original_pair_and_H_b_checks=True,all_fixed_searches_exhaustive=True,
        timing_samples_not_filtered=True,batch_outputs_scope='Independently check 64 untimed validation outputs per process. Timed batch last-result errors are native self-checks; individual inner-loop outputs are not retained.'))
    save(output/'host-environment.json',dict(platform=platform.platform(),python=platform.python_version(),logical_cpus=os.cpu_count(),
        requested_and_observed_local_pool_sizes=protocol['process_thread_order'],
        placement_or_frequency_pinned=False,scope='One host, warm fixed states; not application, GPU or whole-registration throughput.'))
    for name,digest in files.items():
        if independent.sha(root/name) != digest:
            raise ValueError('analysis input changed: '+name)
    save(output/'receipt.json',dict(closed_at_utc=datetime.now(timezone.utc).isoformat(),invocation_sha256=independent.sha(output/'invocation.json'),
        inputs_unchanged=True,output_files_sha256={p.name:independent.sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
    print(json.dumps(dict(states=64,measured_blocks=summary['measured_blocks'],groups=summary['groups'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',type=Path)
    parser.add_argument('runs',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    analyze(args.root.resolve(),args.study.resolve(),args.runs.resolve(),args.output.resolve())


if __name__ == '__main__':
    main()
