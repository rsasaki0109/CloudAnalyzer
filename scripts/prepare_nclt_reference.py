#!/usr/bin/env python3
"""Prepare bounded, sensor-frame NCLT ground truth for a frozen mapping job.

The original NCLT data and derived reference remain under ODbL / DBCL.
No recording, mapping job, map or native binary is modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Same synchronized Velodyne calibration/axis convention as prepare_nclt.py and
# make_nclt_bag.py. velodyne_sync already applies the devkit's -90.7 degree yaw.
BODY_VEL = [0.002, -0.004, -0.957, 0.807, 0.166, 0.0]
FLIP = np.diag([1., -1., -1., 1.])


def artifact(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def sensor_pose(row: np.ndarray) -> np.ndarray:
    """World/body NCLT xyz+rpy -> z-up world/synchronized scan-sensor origin."""
    body = np.eye(4)
    body[:3, :3] = Rotation.from_euler('xyz', row[4:7]).as_matrix()
    body[:3, 3] = row[1:4]
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = Rotation.from_euler('xyz', np.radians(BODY_VEL[3:])).as_matrix()
    extrinsic[:3, 3] = BODY_VEL[:3]
    return FLIP @ body @ extrinsic @ FLIP


def prepare(job_dir: Path, ground_truth: Path, out_dir: Path, dataset_url: str, tolerance: float = .05) -> dict:
    from ca import mapping_job as jobs
    from ca.trajectory import load_trajectory

    root, target = job_dir.resolve(), out_dir.resolve()
    if target.exists():
        raise FileExistsError(f'reference output already exists: {target}')
    if any((p / 'job.json').is_file() for p in (target.parent, *target.parent.parents)):
        raise ValueError('prepare references outside mapping jobs')
    if not np.isfinite(tolerance) or not 0 < tolerance <= 1:
        raise ValueError('tolerance must be in (0, 1] seconds')
    job_artifact = artifact(root / 'job.json')
    job = jobs._load(root)
    motion = job['pointcloud'].get('source_motion', {}).get('trajectory')
    if not motion:
        raise ValueError('job needs hashed original source_motion')
    jobs._verify(motion)
    data = load_trajectory(motion['path'])
    stamps = data['timestamps']
    if len(stamps) > 4096 or not np.isfinite(stamps).all():
        raise ValueError('original motion needs at most 4096 finite timestamps')
    raw_artifact = artifact(ground_truth)
    low, high = (stamps[0] - tolerance) * 1e6, (stamps[-1] + tolerance) * 1e6
    rows, invalid = [], 0
    previous = -float('inf')
    with ground_truth.open(newline='') as stream:
        for line in csv.reader(stream):
            if len(line) != 7:
                raise ValueError('NCLT CSV needs timestamp_us, x, y, z, roll, pitch, yaw')
            row = np.array([float(v) for v in line])
            if not np.isfinite(row[0]) or row[0] <= previous:
                raise ValueError('NCLT timestamps must be finite and strictly increasing')
            previous = row[0]
            if row[0] < low:
                continue
            if row[0] > high:
                break
            if not np.isfinite(row).all():
                invalid += 1
                continue
            rows.append(row)
            if len(rows) > 200000:
                raise ValueError('reference window exceeds 200000 raw rows')
    if len(rows) < 3:
        raise ValueError('too few finite reference rows in the recording window')
    raw = np.array(rows)
    indices: set[int] = set()
    for stamp in stamps:
        right = int(np.searchsorted(raw[:, 0], stamp * 1e6))
        if 0 < right < len(raw):
            indices.update((right - 1, right))
        elif right == 0:
            indices.add(0)
        else:
            indices.add(len(raw) - 1)
    selected = raw[sorted(indices)]
    if len(selected) > 4096:
        raise ValueError('selected reference brackets exceed 4096 poses')
    poses = np.array([sensor_pose(row) for row in selected])
    tum = np.column_stack((selected[:, 0] / 1e6, poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat()))
    # Verify the frozen inputs once more before creating new output files.
    jobs._verify(motion)
    if artifact(root / 'job.json') != job_artifact or artifact(ground_truth) != raw_artifact:
        raise ValueError('reference preparation input changed')
    provenance = {'source': dataset_url + '; original NCLT body ground truth transformed to synchronized Velodyne sensor origin; see preparation.json',
                  'license': 'NCLT Open Database License (ODbL 1.0), contents DBCL 1.0',
                  'frame': 'NCLT z-up world, synchronized Velodyne sensor origin, xyz metres; FLIP @ body_pose @ BODY_VEL @ FLIP',
                  'time_basis': 'NCLT original microsecond timestamps / 1e6, same clock as bundled scan headers',
                  'used_for_generation': False}
    preparation = {'schema': 'cloudanalyzer.nclt_reference_preparation.v1',
                   'inputs': {'job': job_artifact, 'source_motion': motion, 'ground_truth': raw_artifact,
                              'preparation_script': artifact(Path(__file__))},
                   'source_url': dataset_url, 'body_vel_xyz_m_rpy_deg': BODY_VEL,
                   'flip_matrix': FLIP.tolist(), 'ground_truth_rpy_units': 'radians',
                   'timestamp_units': 'microseconds_to_seconds', 'window_tolerance_s': tolerance,
                   'original_scan_poses': len(stamps), 'finite_raw_window_rows': len(raw),
                   'invalid_raw_window_rows': invalid, 'selected_bracket_rows': len(selected),
                   'scope': 'External supplied ground truth was not used as generation input. This is not a statistically independent sensor measurement or HD-map surface/traffic certification.'}
    target.mkdir(parents=True, exist_ok=False)
    np.savetxt(target / 'reference.tum', tum, fmt='%.17g')
    np.savetxt(target / 'reference-raw-rows.csv', selected, fmt='%.17g', delimiter=',')
    (target / 'reference-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    preparation['outputs'] = {name: artifact(target / name) for name in ('reference.tum', 'reference-raw-rows.csv', 'reference-provenance.json')}
    (target / 'preparation.json').write_text(json.dumps(preparation, indent=2) + '\n')
    return preparation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--ground-truth', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-url', required=True)
    parser.add_argument('--tolerance', type=float, default=.05)
    args = parser.parse_args()
    print(json.dumps(prepare(args.job, args.ground_truth, args.out, args.source_url, args.tolerance), indent=2))
