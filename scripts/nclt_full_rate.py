#!/usr/bin/env python3
"""Stream original NCLT synchronized scans without temporal/spatial thinning.

Estimation deliberately has no reference-trajectory argument. Use a fresh output
folder per run; failed attempts remain recorded and must not be discarded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from nclt_body_frames import FRAME_ID

THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS')
POINT = np.dtype([('x', '<u2'), ('y', '<u2'), ('z', '<u2'), ('i', 'u1'), ('laser', 'u1')])


def scan_index(folder: Path, duration_s: int = 600) -> list[tuple[int, Path]]:
    """First fixed-duration window; require evidence the source covers its end."""
    if duration_s <= 0:
        raise ValueError('duration must be positive')
    rows = []
    for path in folder.glob('*.bin'):
        if not path.stem.isascii() or not path.stem.isdecimal() or not path.is_file():
            raise ValueError(f'invalid scan filename: {path.name}')
        stamp = int(path.stem) * 1000
        if not 0 < stamp <= np.iinfo(np.int64).max:
            raise ValueError('timestamp outside int64 nanoseconds')
        rows.append((stamp, path))
    rows.sort(key=lambda row: row[0])
    if len(rows) < 2 or any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError('need distinct chronological scan timestamps')
    end = rows[0][0] + duration_s * 1_000_000_000
    if rows[-1][0] < end:
        raise ValueError('source does not cover the full requested window')
    selected = [row for row in rows if row[0] < end]
    if len(selected) < 2:
        raise ValueError('window needs at least two scans')
    return selected


def decode_scan(raw: bytes) -> np.ndarray:
    if not raw or len(raw) % POINT.itemsize:
        raise ValueError('scan must contain complete 8-byte NCLT point records')
    records = np.frombuffer(raw, dtype=POINT)
    points = np.column_stack([records[key] for key in ('x', 'y', 'z')]).astype(np.float64)
    points = (points * .005 - 100.) * [1., -1., -1.]
    distance = np.linalg.norm(points, axis=1)
    return points[(distance > 1.5) & (distance < 80.)]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def estimate(folder: Path, driver: str, output: Path) -> None:
    if driver not in ('cloudanalyzer', 'kiss-icp'):
        raise ValueError('unsupported driver')
    if any(os.environ.get(key) != '1' for key in THREADS):
        raise ValueError('set all four documented thread variables to 1 before startup')
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'running', 'driver': driver, 'frame_id': FRAME_ID,
               'window_seconds': 600, 'min_range_m': 1.5, 'max_range_m': 80.,
               'map_voxel_m': 1., 'map_points_per_voxel': 20, 'deskew': False,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(output/'run.json', receipt)
    stamps, poses, timings, inputs = [], [], [], []
    started = time.perf_counter()
    try:
        rows = scan_index(folder)
        receipt['expected_frames'] = len(rows)
        if driver == 'cloudanalyzer':
            import cloudanalyzer_core as core
            import cloudanalyzer_core._core as native
            receipt['native_sha256'] = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
            estimator = core.LidarOdometry(min_range=1.5, max_range=80., map_voxel=1., map_points=20, deskew=False)
            register = estimator.register
        else:
            from importlib.metadata import version
            from kiss_icp.config import load_config
            from kiss_icp.kiss_icp import KissICP
            receipt['package_version'] = version('kiss-icp')
            if receipt['package_version'] != '1.2.3':
                raise ValueError('protocol requires kiss-icp 1.2.3')
            config = load_config(None, 80.)
            config.data.min_range = 1.5
            config.data.deskew = False
            config.mapping.voxel_size = 1.
            config.mapping.max_points_per_voxel = 20
            config.registration.max_num_threads = 1
            receipt['upstream_config'] = config.model_dump()
            estimator = KissICP(config)
            def register(points):
                estimator.register_frame(points, np.zeros(len(points)))
                return estimator.last_pose.copy()
        for stamp, path in rows:
            raw = path.read_bytes()
            points = decode_scan(raw)
            inputs.append({'name': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                           'cropped_points': len(points)})
            if len(points) < 20:
                raise ValueError(f'insufficient points: {path.name}; no scan skipping')
            tick = time.perf_counter()
            pose = np.asarray(register(points), dtype=np.float64).copy()
            elapsed = time.perf_counter() - tick
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError(f'invalid estimated pose: {path.name}')
            stamps.append(stamp)
            poses.append(pose)
            timings.append(elapsed)
        receipt['status'] = 'completed'
    except Exception as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        receipt.update(pipeline_seconds=time.perf_counter()-started, completed_frames=len(stamps),
                       registration_seconds=sum(timings),
                       peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        np.savez_compressed(output/'poses.npz', timestamps_ns=np.asarray(stamps, dtype=np.int64),
                            poses=np.asarray(poses).reshape(-1, 4, 4), frame_seconds=timings)
        write_json(output/'inputs.json', inputs)
        for name in ('poses.npz', 'inputs.json'):
            receipt[name+'_sha256'] = hashlib.sha256((output/name).read_bytes()).hexdigest()
        write_json(output/'run.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scans', type=Path, required=True, help='original velodyne_sync directory')
    parser.add_argument('--driver', choices=['cloudanalyzer', 'kiss-icp'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    estimate(args.scans, args.driver, args.out)
