"""Recompute saved trajectory scores; does not rerun any estimator or validate speed."""
from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[2]
spec = importlib.util.spec_from_file_location('independent_nclt_math', ROOT/'benchmarks/slam/nclt-imu-pilot/verify.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def check_artifact(path, expected):
    raw = path.read_bytes()
    assert len(raw) == expected['bytes'], path
    assert hashlib.sha256(raw).hexdigest() == expected['sha256'], path


def verify():
    for name, identity in json.loads((PACKET/'source-identities.json').read_text()).items():
        check_artifact(ROOT/name, identity)
    result = json.loads((PACKET/'results.json').read_text())
    rows = result['runs']
    assert len(rows) == 18
    seen = set()
    for row in rows:
        key = row['session'], row['driver'], row['repeat']
        assert key not in seen
        seen.add(key)
        folder = PACKET/row['directory']
        check_artifact(folder/'poses.npz', row['poses'])
        check_artifact(ROOT/'scripts/compare_nclt_odometry.py', row['script'])
        check_artifact(ROOT/'web/public/samples'/row['input']['name'], row['input'])
        reference_path = ROOT/'benchmarks/vector-map/nclt-reference-accuracy'/row['session']/'reference-raw-rows.csv'
        check_artifact(reference_path, row['reference'])
        saved = json.loads((folder/'run.json').read_text())
        assert all(row[key] == value for key, value in saved.items())
        data = np.load(folder/'poses.npz', allow_pickle=False)
        times, poses, durations = data['timestamps_ns'], data['poses'], data['frame_seconds']
        assert len(times) == len(poses) == len(durations) == row['frames']
        assert np.all(np.diff(times) > 0) and np.all(durations >= 0)
        assert np.isfinite(durations).all()
        assert np.isclose(sum(durations), row['registration_seconds'], rtol=1e-12)
        assert np.isclose(np.percentile(durations, 95)*1000, row['frame_p95_ms'], rtol=1e-12)
        truth = helper.truth_at(reference_path, times)
        for metric, values in helper.errors(poses, truth).items():
            helper.check_stats(values, row['accuracy'][metric])
    expected = {(session, driver, repeat) for session in ('april', 'june')
                for driver in ('cloudanalyzer', 'kiss-icp', 'small-gicp') for repeat in range(3)}
    assert seen == expected
    print('Verified 18 saved runs, input/pose identities, per-frame timing aggregates and all trajectory scores.')


if __name__ == '__main__':
    verify()
