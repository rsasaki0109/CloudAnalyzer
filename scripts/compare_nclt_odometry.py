#!/usr/bin/env python3
"""Frozen NCLT LiDAR-only baseline experiment; estimate before reading references."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = {'april': '2012-04-29', 'june': '2012-06-15'}
DRIVERS = ('cloudanalyzer', 'kiss-icp', 'small-gicp')
THREAD_ENV = {key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS')}


def artifact(path: Path) -> dict:
    raw = path.read_bytes()
    return {'name': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def estimate(session: str, driver: str, output: Path) -> None:
    """No reference trajectory is opened by this function or its child process."""
    import cloudanalyzer_core as core
    import cloudanalyzer_core._core as native
    if any(os.environ.get(k) != v for k, v in THREAD_ENV.items()):
        raise ValueError('Set all documented thread environment variables to 1 before startup')
    output.mkdir(parents=True, exist_ok=False)
    recording = ROOT/'web/public/samples'/f'nclt-{SESSIONS[session]}.mcap'
    frozen = artifact(recording)
    started = time.perf_counter()
    scans = list(core.BagReader(str(recording)).scans())
    stamps = np.array([s.stamp_ns for s in scans], dtype=np.int64)
    if len(scans) < 2 or np.any(np.diff(stamps) <= 0):
        raise ValueError('Need chronological scans')
    points = [np.asarray(s.positions(), dtype=np.float64) for s in scans]
    # Shared explicit crop. No frame is skipped, including any invalid/empty one.
    points = [p[(np.linalg.norm(p, axis=1) > 1.5) & (np.linalg.norm(p, axis=1) < 80.)] for p in points]
    if any(len(p) < 20 or not np.isfinite(p).all() for p in points):
        raise ValueError('Invalid or insufficient input points')
    settings = {'min_range_m': 1.5, 'max_range_m': 80., 'deskew': False, 'threads': 1}
    if driver == 'cloudanalyzer':
        odometry = core.LidarOdometry(min_range=1.5, max_range=80., map_voxel=1., map_points=20, deskew=False)
        register = lambda p: np.asarray(odometry.register(p), dtype=np.float64)
        settings.update(map_voxel_m=1., map_points_per_voxel=20)
        implementation = artifact(Path(native.__file__))
    elif driver == 'kiss-icp':
        from kiss_icp.config import load_config
        from kiss_icp.kiss_icp import KissICP
        config = load_config(None, 80.)
        config.data.min_range = 1.5
        config.data.deskew = False
        config.mapping.voxel_size = 1.
        config.mapping.max_points_per_voxel = 20
        config.registration.max_num_threads = 1
        odometry = KissICP(config)
        def register(p):
            odometry.register_frame(p, np.zeros(len(p)))
            return odometry.last_pose.copy()
        settings['upstream_config'] = config.model_dump()
        implementation = {'package': 'kiss-icp', 'version': importlib.metadata.version('kiss-icp')}
    else:
        import small_gicp
        voxelmap = small_gicp.GaussianVoxelMap(1.)
        previous, delta = np.eye(4), np.eye(4)
        count = 0
        def register(p):
            nonlocal previous, delta, count
            source, _ = small_gicp.preprocess_points(p, downsampling_resolution=.5, num_threads=1)
            if count:
                result = small_gicp.align(voxelmap, source, init_T_target_source=previous @ delta,
                    max_correspondence_distance=5., num_threads=1, max_iterations=30)
                pose = np.asarray(result.T_target_source)
            else:
                pose = np.eye(4)
            voxelmap.insert(source, T=pose)
            delta, previous = np.linalg.inv(previous) @ pose, pose.copy()
            count += 1
            return pose
        settings.update(map_voxel_m=1., downsampling_m=.5, correspondence_distance_m=5., max_iterations=30,
                        map_retention='all scans; unlike bounded local maps in the other arms')
        implementation = {'package': 'small-gicp', 'version': importlib.metadata.version('small-gicp')}
    durations, poses = [], []
    for p in points:
        tick = time.perf_counter()
        pose = register(p)
        durations.append(time.perf_counter() - tick)
        poses.append(pose)
    elapsed = time.perf_counter() - started
    poses = np.asarray(poses)
    if poses.shape != (len(scans), 4, 4) or not np.isfinite(poses).all():
        raise ValueError('Invalid estimated poses')
    if artifact(recording) != frozen:
        raise ValueError('Input recording changed')
    np.savez_compressed(output/'poses.npz', timestamps_ns=stamps, poses=poses, frame_seconds=durations)
    write_json(output/'run.json', {'driver': driver, 'session': session, 'input': frozen,
        'settings': settings, 'implementation': implementation, 'script': artifact(Path(__file__)),
        'frames': len(scans), 'points': sum(map(len, points)), 'frame_ids': sorted({s.frame_id for s in scans}),
        'registration_seconds': sum(durations), 'pipeline_seconds': elapsed,
        'frame_p50_ms': float(np.percentile(durations, 50) * 1000),
        'frame_p95_ms': float(np.percentile(durations, 95) * 1000),
        'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'memory_scope': 'Linux process high-water RSS including imports, decoded scans and maps; not incremental algorithm memory',
        'poses': artifact(output/'poses.npz')})


def experiment(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    protocol = {'sessions': SESSIONS, 'drivers': DRIVERS, 'repeats': 3, 'thread_environment': THREAD_ENV,
        'script': artifact(Path(__file__)), 'python': sys.version, 'platform': platform.platform(),
        'numpy': np.__version__, 'scope': 'Existing sparse, motion-compensated NCLT demo cohort; no held-out or SOTA claim',
        'order': 'cyclic driver order per repetition; April then June; separate process per run',
        'accuracy': 'Body-z-up reference interpolated from original archived NCLT rows; global SE3 position-fit ATE, consecutive SE3 RPE, no scale',
        'timing': 'registration loop separately from decode/setup/registration pipeline; no map export; all startup frames included',
        'parameters': 'Fixed before scoring; shared range crop; 1m map voxels; upstream/previous adapter convergence settings, not identical optimizers',
        'reference_use': 'Only after every estimator subprocess finishes; no truth supplied to estimation'}
    write_json(output/'protocol.json', protocol)
    rows = []
    for session in SESSIONS:
        for repeat in range(3):
            for driver in DRIVERS[repeat:] + DRIVERS[:repeat]:
                folder = output/f'{session}-{driver}-{repeat}'
                print(f'Running {folder.name}', flush=True)
                tick = time.perf_counter()
                subprocess.run([sys.executable, str(Path(__file__)), 'estimate', '--session', session,
                    '--driver', driver, '--out', str(folder)], env={**os.environ, **THREAD_ENV}, check=True, timeout=600)
                row = json.loads((folder/'run.json').read_text())
                row.update(repeat=repeat, directory=folder.name, subprocess_seconds=time.perf_counter()-tick)
                rows.append(row)
    write_json(output/'runs.json', rows)
    # Import the established scorer only after all trajectories have been saved.
    from evaluate_nclt_imu_pilot import reference_at, pose_errors, summarize
    for row in rows:
        folder = output/row['directory']
        if artifact(folder/'poses.npz') != row['poses']:
            raise ValueError('Estimated trajectory changed before scoring')
        data = np.load(folder/'poses.npz', allow_pickle=False)
        reference_path = ROOT/'benchmarks/vector-map/nclt-reference-accuracy'/row['session']/'reference-raw-rows.csv'
        reference_id = artifact(reference_path)
        reference = reference_at(reference_path, data['timestamps_ns'])
        row['accuracy'] = summarize(pose_errors(data['poses'], reference))
        row['reference'] = reference_id
        row['distance_m'] = float(np.linalg.norm(np.diff(reference[:, :3, 3], axis=0), axis=1).sum())
        if artifact(reference_path) != reference_id:
            raise ValueError('Reference changed during scoring')
    write_json(output/'results.json', {'protocol': protocol, 'runs': rows})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['estimate', 'experiment'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--session', choices=SESSIONS)
    parser.add_argument('--driver', choices=DRIVERS)
    args = parser.parse_args()
    if args.mode == 'estimate':
        if args.session is None or args.driver is None:
            parser.error('estimate requires --session and --driver')
        estimate(args.session, args.driver, args.out)
    else:
        experiment(args.out)
