#!/usr/bin/env python3
"""Run the frozen fixed-state benchmark in counterbalanced native processes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run(study, output, root):
    protocol = read(study/'protocol.json')
    freeze = read(study/'controller-freeze.json')
    assert sha(study/'protocol.json') == freeze['protocol_sha256']
    inputs = dict(freeze['files_sha256'])
    binary = study/'native/odometry-stage-profile.exe'
    assert sha(binary) == read(study/'native-artifact.json')['binary_sha256']
    for path in [binary,study/'protocol.json',study/'native-artifact.json',study/'controller-freeze.json',Path(__file__)]:
        inputs[path.relative_to(root).as_posix()] = sha(path)
    for name,digest in inputs.items():
        assert sha(root/name) == digest,name
    output.mkdir(exist_ok=False)
    save(output/'invocation.json', dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(study/'protocol.json'),native_artifact_sha256=sha(study/'native-artifact.json'),inputs_sha256=inputs))
    for index, threads in enumerate(protocol['process_thread_order']):
        destination = output/f'run-{index:02d}-threads-{threads}.json'
        env = dict(os.environ,CA_ODOMETRY_PROFILE_INPUT=str(study/'prepared-01/input.json'),
            CA_ODOMETRY_PROFILE_OUTPUT=str(destination),CA_ODOMETRY_PROFILE_THREADS=str(threads))
        with (output/f'run-{index:02d}.log').open('xb') as log:
            result = subprocess.run([str(binary),'odometry::profile::record_fixed_states','--exact','--ignored','--nocapture'],
                cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode == 0,(output/f'run-{index:02d}.log').read_text(encoding='utf-8',errors='replace')[-6000:]
        assert read(destination)['threads'] == threads
        print(json.dumps(dict(process=index,threads=threads,completed=True)),flush=True)
    for name,digest in inputs.items():
        assert sha(root/name) == digest,name
    save(output/'receipt.json', dict(closed_at_utc=datetime.now(timezone.utc).isoformat(),
        invocation_sha256=sha(output/'invocation.json'),input_hashes_unchanged=True,
        output_files_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run(args.study.resolve(),args.output.resolve(),args.root.resolve())


if __name__ == '__main__':
    main()
