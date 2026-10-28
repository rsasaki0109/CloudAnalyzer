#!/usr/bin/env python3
"""Independently recompute all saved scores and check gain-1 baseline agreement."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PACKET = Path(__file__).resolve().parent


def verify(folder):
    spec=importlib.util.spec_from_file_location('independent_se3', ROOT/'benchmarks/slam/nclt-imu-pilot/verify.py')
    math=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math)
    rows=json.loads((folder/'results.json').read_text())
    expected={(date,repeat,gain) for date in ['2012-04-29','2012-06-15'] for repeat in range(3) for gain in [1.,1.5]}
    assert len(rows)==len(expected)==12
    assert {(r['session'],r['repeat'],r['gain']) for r in rows}==expected
    first={}
    for row in rows:
        path=folder/row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        raw=json.loads(gzip.decompress(path.read_bytes()))
        assert raw['gain']==row['gain']
        data=raw['frames']
        poses=np.array([f['pose'] for f in data]).reshape(-1,4,4)
        times=np.array([f['stamp_ns'] for f in data],dtype=np.int64)
        assert np.all(np.diff(times)>0) and np.isfinite(poses).all()
        assert len(data)==row['frames']
        assert sum(f['iterations'] for f in data)==row['iterations']
        np.testing.assert_allclose(sum(f['seconds'] for f in data),row['registration_seconds'],rtol=1e-12)
        name={'2012-04-29':'april','2012-06-15':'june'}[row['session']]
        reference=ROOT/f'benchmarks/vector-map/nclt-reference-accuracy/{name}/reference-raw-rows.csv'
        assert hashlib.sha256(reference.read_bytes()).hexdigest()==row['reference_sha256']
        truth=math.truth_at(reference,times)
        for key, errors in math.errors(poses,truth).items():
            math.check_stats(errors,row['accuracy'][key])
        key=row['session'],row['gain']
        if key in first:
            np.testing.assert_array_equal(poses,first[key])
        first[key]=poses
        if row['gain']==1.:
            old=ROOT/f"benchmarks/slam/nclt-sparse-equations/results/{row['session']}-0-candidate.json"
            prior=json.loads(old.read_text())['frames']
            np.testing.assert_allclose(poses,np.array([f['pose'] for f in prior]).reshape(-1,4,4),atol=1e-11,rtol=0.)
            assert [f['iterations'] for f in data]==[f['iterations'] for f in prior]
    print('Verified all 12 trajectory scores, repeatability and gain-1 production baseline agreement')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path,nargs='?',default=PACKET/'results')
    verify(parser.parse_args().folder)
