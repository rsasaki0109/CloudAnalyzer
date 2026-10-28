#!/usr/bin/env python3
"""Run the prospective fixed-gain experiment, then score every completed run."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PACKET = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def experiment(binary, out):
    out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((PACKET/'protocol.json').read_text())
    save(out/'protocol.json', protocol)
    save(out/'identities.json', dict(binary=sha(binary), runner=sha(Path(__file__)),
        example=sha(ROOT/'rust/crates/ca-core/examples/bench_bag_relaxation.rs'),
        core=sha(ROOT/'rust/crates/ca-core/src/odometry.rs'),
        inputs={date:sha(ROOT/f'web/public/samples/nclt-{date}.mcap') for date in protocol['sessions']}))
    rows = []
    for date in protocol['sessions']:
        for repeat in range(protocol['repeats']):
            gains = protocol['gains'] if repeat % 2 == 0 else protocol['gains'][::-1]
            for gain in gains:
                path = out/f'{date}-{repeat}-{gain}.json'
                tick = time.perf_counter()
                subprocess.run([str(binary), str(ROOT/f'web/public/samples/nclt-{date}.mcap'), str(path), str(gain)],
                    env={**os.environ, 'RAYON_NUM_THREADS':'1'}, check=True, timeout=600)
                wall = time.perf_counter()-tick
                raw = path.read_bytes()
                data = json.loads(raw)['frames']
                compressed = path.with_suffix('.json.gz')
                compressed.write_bytes(gzip.compress(raw, mtime=0))
                path.unlink()
                row = dict(session=date, repeat=repeat, gain=gain, file=compressed.name,
                    sha256=sha(compressed), frames=len(data), wall_seconds=wall,
                    registration_seconds=sum(f['seconds'] for f in data),
                    iterations=sum(f['iterations'] for f in data),
                    terminations=dict(Counter(f['termination'] for f in data)))
                rows.append(row)
                save(out/'runs.json', rows)
                print(json.dumps(row), flush=True)
    # Only now open reference trajectories: no score affects any estimator run.
    sys.path.insert(0, str(ROOT/'scripts'))
    from evaluate_nclt_imu_pilot import reference_at, pose_errors, summarize
    for row in rows:
        data=json.loads(gzip.decompress((out/row['file']).read_bytes()))['frames']
        poses=np.array([f['pose'] for f in data]).reshape(-1,4,4)
        times=np.array([f['stamp_ns'] for f in data],dtype=np.int64)
        session={'2012-04-29':'april','2012-06-15':'june'}[row['session']]
        reference=ROOT/f'benchmarks/vector-map/nclt-reference-accuracy/{session}/reference-raw-rows.csv'
        row['reference_sha256']=sha(reference)
        row['accuracy']=summarize(pose_errors(poses,reference_at(reference,times)))
    save(out/'results.json', rows)
    print('All estimates finished and all trajectories scored', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args=parser.parse_args()
    experiment(args.binary.resolve(), args.out.resolve())
