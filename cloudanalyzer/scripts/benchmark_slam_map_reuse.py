"""Compare three repeated real-source diagnostic attempts with scoped map reuse.

Real PLY loading and complete SLAM diagnostic assembly are timed. The saved
actual ICP solve is supplied identically to both paths; solving and artifacts
are excluded. The dispatch CSV is a fixture, not recorded SLAM telemetry.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def info(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ['source-root', 'reference', 'compared', 'input-manifest', 'scan-baseline',
                   'slam-baseline', 'registration-record', 'work-dir', 'output']:
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--scan-baseline-sha256', required=True)
    parser.add_argument('--slam-baseline-sha256', required=True)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--attempts', type=int, default=3)
    options = [value.split('=')[0] for value in sys.argv[1:] if value.startswith('--')]
    require(len(options) == len(set(options)), 'duplicate options')
    args = parser.parse_args()
    require(1 <= args.threads <= 64 and 1 <= args.repeats <= 100, 'invalid threads/repeats')
    require(2 <= args.attempts <= 20, 'invalid attempts')
    require(not args.output.exists() and not args.work_dir.exists(), 'output/work directory must be new')
    require(sha(args.scan_baseline) == args.scan_baseline_sha256, 'scan baseline differs')
    require(sha(args.slam_baseline) == args.slam_baseline_sha256, 'slam baseline differs')
    os.environ['RAYON_NUM_THREADS'] = os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['CA_DISABLE_RUST_CORE'] = '0'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str((args.source_root / 'cloudanalyzer').resolve()))
    import numpy as np
    import open3d as o3d
    import cloudanalyzer_core as cc
    import cloudanalyzer_core._core as core
    import ca.scan_match_debug as current_scan
    import ca.slam_debug as current_slam
    from ca.io import load_point_cloud
    from ca.registration import RegistrationResult

    require(hasattr(cc, 'PreparedReference'), 'extension lacks prepared references')
    manifest = json.loads(args.input_manifest.read_text(encoding='utf-8'))
    prior = json.loads(args.registration_record.read_text(encoding='utf-8'))
    require(prior['comparison'] == 'scan-match-distance-reuse', 'unexpected registration record')
    require(manifest == prior['inputManifest'], 'registration manifest differs')
    for key in ['reference', 'compared']:
        require(sha(getattr(args, key)) == manifest['inputs'][key]['sha256'] == prior['inputs'][key]['sha256'], key + ' differs')
    original = (args.source_root / 'web' / manifest['source']['path']).resolve()
    require(sha(original) == manifest['source']['sha256'], 'original source differs')
    snapshot = prior['registration']['snapshot']
    require(sha(snapshot['path']) == snapshot['sha256'], 'registration snapshot differs')
    data = np.load(snapshot['path'])
    for key, path in [('initial', args.compared), ('reference', args.reference)]:
        cloud = load_point_cloud(str(path))
        require(np.asarray(cloud.points).tobytes() == data[key].tobytes(), 'recorded coordinates differ: ' + key)
        require(len(cloud.points) == manifest['inputs']['compared' if key == 'initial' else 'reference']['points'], 'point count differs')
    del cloud
    matrix = np.asarray(prior['registration']['transformation'])
    require(matrix.tobytes() == data['transformation'].tobytes(), 'recorded matrix differs')
    transformed = o3d.geometry.PointCloud()
    transformed.points = o3d.utility.Vector3dVector(data['initial'])
    transformed.normals = o3d.utility.Vector3dVector(data['initial_normals'])
    transformed.transform(matrix)
    require(np.asarray(transformed.points).tobytes() == data['aligned'].tobytes(), 'aligned coordinates differ')
    require(np.asarray(transformed.normals).tobytes() == data['aligned_normals'].tobytes(), 'aligned normals differ')
    registration = RegistrationResult(transformed, prior['registration']['fitness'], prior['registration']['rmse'], matrix)

    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
        return value
    baseline_scan = module('slam_map_fresh_scan', args.scan_baseline)
    sys.modules['ca.scan_match_debug'] = baseline_scan
    try:
        baseline_slam = module('slam_map_fresh_slam', args.slam_baseline)
    finally:
        sys.modules['ca.scan_match_debug'] = current_scan
    modules = dict(fresh=(baseline_scan, baseline_slam), reuse=(current_scan, current_slam))
    files = ['cloudanalyzer/ca/scan_match_debug.py', 'cloudanalyzer/ca/slam_debug.py',
             'cloudanalyzer/ca/metrics.py', 'cloudanalyzer/ca/registration.py', 'cloudanalyzer/ca/io.py',
             'cloudanalyzer/ca/visualization.py', 'cloudanalyzer/ca/_rust.py', 'cloudanalyzer/pyproject.toml',
             'cloudanalyzer/tests/test_scan_match_debug.py', 'cloudanalyzer/tests/test_slam_debug.py',
             'cloudanalyzer/tests/test_slam_map_reuse.py', 'cloudanalyzer/tests/test_metrics.py',
             'cloudanalyzer/tests/test_registration.py', 'cloudanalyzer/scripts/benchmark_slam_map_reuse.py']
    hashes = {path: sha(args.source_root / path) for path in files}
    inputs = {key: info(getattr(args, key)) for key in ['reference', 'compared', 'input_manifest',
                                                      'scan_baseline', 'slam_baseline', 'registration_record']}
    inputs.update(originalSource=info(original), registrationSnapshot=info(snapshot['path']))
    extension = info(core.__file__)
    args.work_dir.mkdir(parents=True, exist_ok=False)
    metrics, scans = args.work_dir / 'metrics.csv', args.work_dir / 'scans.csv'
    with metrics.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('scan_id,timestamp_sec,scan_match_rmse_m\n')
        stream.writelines(f'attempt_{index},{index},{args.attempts-index}\n' for index in range(args.attempts))
    # CSV quoting permits spaces/commas in input paths.
    import csv
    with scans.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scan_id', 'points_csv'])
        writer.writerows((f'attempt_{index}', str(args.compared.resolve())) for index in range(args.attempts))
    workload = dict(metrics=info(metrics), scans=info(scans), attempts=args.attempts,
                    policy='repeated_same_full_source_scan', recordedSlamTelemetry=False)

    def call(policy, capture=False):
        scan, slam = modules[policy]
        load, prepare, solve = scan.load_point_cloud, scan.prepare_nn_reference, scan.register_detailed
        distances, capacities = [], []
        loads = dict(scan=0, map=0)
        def fixed_registration(source, target, **kwargs):
            if capture:
                require(np.asarray(source.points).tobytes() == data['initial'].tobytes(), 'solver source differs')
                require(np.asarray(target.points).tobytes() == data['reference'].tobytes(), 'solver target differs')
                require(kwargs == dict(method='icp', max_correspondence_distance=1.), 'solver arguments differ')
            source.normals = o3d.utility.Vector3dVector(data['initial_normals'])
            target.normals = o3d.utility.Vector3dVector(data['reference_normals'])
            return registration
        scan.register_detailed = fixed_registration
        if capture:
            def traced_load(path):
                loads['scan' if path == str(args.compared.resolve()) else 'map'] += 1
                return load(path)
            def traced_prepare(target):
                native = prepare(target)
                require(native is not None, 'reference preparation missing')
                capacities.append(native.retained_bytes)
                class Reference:
                    def distances(self, points):
                        values = native.distances(points)
                        distances.append(values)
                        return values
                return Reference()
            scan.load_point_cloud, scan.prepare_nn_reference = traced_load, traced_prepare
        start = time.perf_counter_ns()
        try:
            result = slam.analyze_slam_run(str(metrics), scans_manifest_csv=str(scans), map_path=str(args.reference.resolve()),
                                          top_k=args.attempts, run_scan_match_debug_frames=True,
                                          scan_match_method='icp', scan_match_threshold=.5)
        finally:
            scan.load_point_cloud, scan.prepare_nn_reference, scan.register_detailed = load, prepare, solve
        elapsed = (time.perf_counter_ns() - start) / 1e6
        require(len(result['selected_frames']) == args.attempts and
                all(frame['scan_match_debug_error'] is None for frame in result['selected_frames']), 'frame diagnosis failed')
        captured = []
        for index, values in enumerate(distances):
            path = args.work_dir / f'{policy}-distance-{index}.bin'
            with path.open('xb') as stream:
                stream.write(values.astype('<f8', copy=False).tobytes())
            captured.append(info(path))
        return dict(millis=elapsed, report=result, loads=loads, referenceCapacities=capacities, capturedDistances=captured)

    parity = {policy: call(policy, True) for policy in ['fresh', 'reuse']}
    require(parity['fresh']['report'] == parity['reuse']['report'], 'complete SLAM reports differ')
    expected_hashes = [item['sha256'] for item in prior['firstUse']['fresh']['capturedDistances'][:2]] * args.attempts
    for policy in ['fresh', 'reuse']:
        require([item['sha256'] for item in parity[policy]['capturedDistances']] == expected_hashes, 'distance bytes differ')
        require(parity[policy]['loads'] == dict(scan=args.attempts, map=args.attempts if policy == 'fresh' else 1), 'load count differs')
        require(len(parity[policy]['referenceCapacities']) == (args.attempts if policy == 'fresh' else 1), 'preparation count differs')
    require(parity['fresh']['referenceCapacities'] == parity['reuse']['referenceCapacities'] * args.attempts, 'payload differs')
    print(json.dumps(dict(stage='parity_complete')), flush=True)
    first = {policy: call(policy) for policy in ['fresh', 'reuse']}
    expected = parity['fresh']['report']
    require(all(item['report'] == expected for item in first.values()), 'first reports differ')
    def pair(index):
        order = ['fresh', 'reuse'] if index % 2 == 0 else ['reuse', 'fresh']
        observed = {policy: call(policy) for policy in order}
        require(all(item['report'] == expected for item in observed.values()), 'timed report differs')
        return dict(order=order, millis={policy: item['millis'] for policy, item in observed.items()})
    warmups = [pair(index) for index in range(2)]
    print(json.dumps(dict(stage='warmup_complete')), flush=True)
    trials = []
    for index in range(args.repeats):
        trial = pair(index+2)
        trials.append(trial)
        print(json.dumps(dict(stage='trial_complete', trial=index+1, millis=trial['millis'])), flush=True)
    a, b = [item['millis']['fresh'] for item in trials], [item['millis']['reuse'] for item in trials]
    summary = dict(freshMedianMillis=statistics.median(a), reuseMedianMillis=statistics.median(b),
                   pairedShorteningPercent=statistics.median(100*(1-y/x) for x, y in zip(a, b)),
                   pairedSavedMedianMillis=statistics.median(x-y for x, y in zip(a, b)), fasterTrials=sum(x > y for x, y in zip(a, b)))
    for path, expected_hash in hashes.items():
        require(sha(args.source_root / path) == expected_hash, 'source changed: ' + path)
    for item in [*inputs.values(), extension, workload['metrics'], workload['scans']]:
        require(sha(item['path']) == item['sha256'], 'input/build changed')
    report = dict(schemaVersion=1, comparison='slam-map-reuse', createdAt=datetime.now(timezone.utc).isoformat(),
                  sourceHashes=hashes, inputs=inputs, inputManifest=manifest, extension=extension, workload=workload,
                  environment=dict(python=platform.python_version(), numpy=np.__version__, open3d=o3d.__version__,
                                   platform=platform.system(), logicalCpus=os.cpu_count(), rayonNumThreads=args.threads,
                                   ompNumThreads=args.threads, openblasNumThreads=1),
                  repeats=args.repeats, warmupRounds=2, attempts=args.attempts,
                  parity=parity, firstUse=first, warmups=warmups, trials=trials, summary=summary,
                  accuracy=dict(fullF64ArraysChecked=4*args.attempts, reportsChecked=2*(4+args.repeats)),
                  note='Three default repeated attempts use the same full original-coordinate scan/map and saved actual ICP result, '
                       'not independent drive frames or recorded SLAM telemetry. The CSV fixture only dispatches/ranks attempts. '
                       'Real file loading, cloud/normal copies, ranking/report assembly, NN preparation/search/statistics/thresholds '
                       'and scoped cleanup are timed. Registration solving, parity tracing, snapshot preparation/checks/hashing, '
                       'artifact/color/PLY work and CLI writes are excluded. Pages are warmed; cold ingestion and RSS are not measured. '
                       'A traced parity pair, excluded first pair and two warm-up pairs precede alternating timed pairs. '
                       'All observations remain. The control already prepares once per attempt; the current path shares map '
                       'loading/preprocessing and an uncropped reference across attempts, retaining private registration targets. '
                       'No entire ICP/SLAM-run, cropped/downsampled-workload or GPU speed claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), summary=summary)), flush=True)


if __name__ == '__main__':
    main()
