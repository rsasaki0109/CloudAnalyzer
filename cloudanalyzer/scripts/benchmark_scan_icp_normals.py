"""Compare no-artifact ICP diagnostics with and without unused normal estimation.

The native solver boundary receives a saved actual solve after real preparation.
Loading, preparation, NN diagnostics and report assembly are timed; ICP iterations
are excluded. Each run dispatches the same real-source scan three times by default.
"""

import argparse
import csv
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
from types import SimpleNamespace


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


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ['source-root', 'reference', 'compared', 'input-manifest', 'scan-baseline',
                   'registration-baseline', 'registration-record', 'work-dir', 'output']:
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--scan-baseline-sha256', required=True)
    parser.add_argument('--registration-baseline-sha256', required=True)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--attempts', type=int, default=3)
    options = [value.split('=')[0] for value in sys.argv[1:] if value.startswith('--')]
    require(len(options) == len(set(options)), 'duplicate options')
    args = parser.parse_args()
    require(1 <= args.threads <= 64 and 1 <= args.repeats <= 100, 'invalid threads/repeats')
    require(2 <= args.attempts <= 20, 'invalid attempts')
    require(not args.output.exists() and not args.work_dir.exists(), 'output/work directory must be new')
    require(sha(args.scan_baseline) == args.scan_baseline_sha256, 'scan baseline differs')
    require(sha(args.registration_baseline) == args.registration_baseline_sha256, 'registration baseline differs')
    os.environ['RAYON_NUM_THREADS'] = os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['CA_DISABLE_RUST_CORE'] = '0'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str((args.source_root / 'cloudanalyzer').resolve()))
    import numpy as np
    import open3d as o3d
    import cloudanalyzer_core as cc
    import cloudanalyzer_core._core as core
    import ca.registration as current_registration
    import ca.scan_match_debug as current_scan
    import ca.slam_debug as current_slam
    from ca.io import load_point_cloud

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
        require(not cloud.has_normals() and not cloud.has_covariances() and not cloud.has_colors(), 'unexpected attributes')
    del cloud
    matrix = np.asarray(prior['registration']['transformation'])
    require(matrix.tobytes() == data['transformation'].tobytes(), 'recorded matrix differs')
    fixed = SimpleNamespace(transformation=matrix, fitness=prior['registration']['fitness'], inlier_rmse=prior['registration']['rmse'])

    baseline_registration = module('diagnostic_icp_registration_baseline', args.registration_baseline)
    sys.modules['ca.registration'] = baseline_registration
    try:
        baseline_scan = module('diagnostic_icp_scan_baseline', args.scan_baseline)
    finally:
        sys.modules['ca.registration'] = current_registration
    sys.modules['ca.scan_match_debug'] = baseline_scan
    try:
        baseline_slam = module('diagnostic_icp_slam_baseline', args.source_root / 'cloudanalyzer/ca/slam_debug.py')
    finally:
        sys.modules['ca.scan_match_debug'] = current_scan
    modules = dict(normals=(baseline_registration, baseline_scan, baseline_slam),
                   skip=(current_registration, current_scan, current_slam))
    files = ['cloudanalyzer/ca/scan_match_debug.py', 'cloudanalyzer/ca/slam_debug.py',
             'cloudanalyzer/ca/metrics.py', 'cloudanalyzer/ca/registration.py', 'cloudanalyzer/ca/io.py',
             'cloudanalyzer/ca/visualization.py', 'cloudanalyzer/ca/_rust.py', 'cloudanalyzer/pyproject.toml',
             'cloudanalyzer/tests/test_scan_icp_normals.py', 'cloudanalyzer/tests/test_registration_target_normals.py',
             'cloudanalyzer/tests/test_scan_match_debug.py', 'cloudanalyzer/tests/test_slam_debug.py',
             'cloudanalyzer/tests/test_slam_map_reuse.py', 'cloudanalyzer/tests/test_metrics.py',
             'cloudanalyzer/tests/test_registration.py', 'cloudanalyzer/scripts/benchmark_scan_icp_normals.py']
    hashes = {path: sha(args.source_root / path) for path in files}
    inputs = {key: info(getattr(args, key)) for key in ['reference', 'compared', 'input_manifest',
                                                      'scan_baseline', 'registration_baseline', 'registration_record']}
    inputs.update(originalSource=info(original), registrationSnapshot=info(snapshot['path']))
    extension, normal_library = info(core.__file__), info(o3d.cpu.pybind.__file__)
    args.work_dir.mkdir(parents=True, exist_ok=False)
    metrics, scans = args.work_dir / 'metrics.csv', args.work_dir / 'scans.csv'
    with metrics.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('scan_id,timestamp_sec,scan_match_rmse_m\n')
        stream.writelines(f'attempt_{index},{index},{args.attempts-index}\n' for index in range(args.attempts))
    with scans.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scan_id', 'points_csv'])
        writer.writerows((f'attempt_{index}', str(args.compared.resolve())) for index in range(args.attempts))
    workload = dict(metrics=info(metrics), scans=info(scans), attempts=args.attempts,
                    policy='repeated_same_full_source_scan', recordedSlamTelemetry=False)

    def call(policy, capture=False):
        registration, scan, slam = modules[policy]
        load, prepare = scan.load_point_cloud, scan.prepare_nn_reference
        estimate = registration._estimate_registration_normals
        native_solver = o3d.pipelines.registration.registration_icp
        distances, capacities, solver_inputs, normal_estimates = [], [], [], []
        loads = dict(scan=0, map=0)

        def solver(source, target, distance, initial, estimator):
            if capture:
                require(distance == 1. and initial.tobytes() == np.eye(4).tobytes(), 'solver options differ')
                require(isinstance(estimator, o3d.pipelines.registration.TransformationEstimationPointToPoint)
                        and not estimator.with_scaling, 'estimator differs')
                fields = {}
                for label, cloud in [('initial', source), ('reference', target)]:
                    require(np.asarray(cloud.points).tobytes() == data[label].tobytes(), 'solver coordinates differ')
                    require(not cloud.has_colors() and not cloud.has_covariances(), 'solver attributes differ')
                    require(cloud.has_normals() == (policy == 'normals'), 'normal preparation policy differs')
                    if policy == 'normals':
                        require(np.asarray(cloud.normals).tobytes() == data[label + '_normals'].tobytes(), 'estimated normals differ')
                    fields[label] = dict(pointsSha256=hashlib.sha256(np.asarray(cloud.points).tobytes()).hexdigest(),
                                         normalsSha256=hashlib.sha256(np.asarray(cloud.normals).tobytes()).hexdigest(),
                                         normalCount=len(cloud.normals))
                solver_inputs.append(dict(**fields, distance=distance, initTransform=initial.tolist(),
                                          estimator=type(estimator).__name__, withScaling=estimator.with_scaling))
            return fixed

        o3d.pipelines.registration.registration_icp = solver
        if capture:
            def traced_estimate(cloud):
                estimate(cloud)
                normal_estimates.append(dict(pointsSha256=hashlib.sha256(np.asarray(cloud.points).tobytes()).hexdigest(),
                                             normalsSha256=hashlib.sha256(np.asarray(cloud.normals).tobytes()).hexdigest(),
                                             points=len(cloud.points)))
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
            registration._estimate_registration_normals = traced_estimate
            scan.load_point_cloud, scan.prepare_nn_reference = traced_load, traced_prepare
        start = time.perf_counter_ns()
        try:
            result = slam.analyze_slam_run(str(metrics), scans_manifest_csv=str(scans), map_path=str(args.reference.resolve()),
                                          top_k=args.attempts, run_scan_match_debug_frames=True,
                                          scan_match_method='icp', scan_match_threshold=.5)
        finally:
            scan.load_point_cloud, scan.prepare_nn_reference = load, prepare
            registration._estimate_registration_normals = estimate
            o3d.pipelines.registration.registration_icp = native_solver
        elapsed = (time.perf_counter_ns() - start) / 1e6
        require(len(result['selected_frames']) == args.attempts and
                all(frame['scan_match_debug_error'] is None for frame in result['selected_frames']), 'frame diagnosis failed')
        for frame in result['selected_frames']:
            observed = frame['scan_match_debug_result']['registration']
            require(observed['fitness'] == fixed.fitness and observed['inlier_rmse'] == fixed.inlier_rmse
                    and np.asarray(observed['final_transform']).tobytes() == matrix.tobytes(), 'registration output differs')
        captured = []
        for index, values in enumerate(distances):
            path = args.work_dir / f'{policy}-distance-{index}.bin'
            with path.open('xb') as stream:
                stream.write(values.astype('<f8', copy=False).tobytes())
            captured.append(info(path))
        return dict(millis=elapsed, report=result, loads=loads, referenceCapacities=capacities,
                    capturedDistances=captured, solverInputs=solver_inputs, normalEstimates=normal_estimates)

    policies = ['normals', 'skip']
    parity = {policy: call(policy, True) for policy in policies}
    expected = parity['normals']['report']
    require(parity['skip']['report'] == expected, 'complete SLAM reports differ')
    expected_hashes = [item['sha256'] for item in prior['firstUse']['fresh']['capturedDistances'][:2]] * args.attempts
    for policy in policies:
        observed = parity[policy]
        require([item['sha256'] for item in observed['capturedDistances']] == expected_hashes, 'distance bytes differ')
        require(observed['loads'] == dict(scan=args.attempts, map=1) and len(observed['referenceCapacities']) == 1,
                'map reuse policy differs')
        require(len(observed['solverInputs']) == args.attempts, 'solver count differs')
        require(len(observed['normalEstimates']) == (args.attempts + 1 if policy == 'normals' else 0), 'normal estimate count differs')
    print(json.dumps(dict(stage='parity_complete')), flush=True)
    first = {policy: call(policy) for policy in policies}
    require(all(item['report'] == expected for item in first.values()), 'first reports differ')
    def pair(index):
        order = policies if index % 2 == 0 else list(reversed(policies))
        observed = {policy: call(policy) for policy in order}
        require(all(item['report'] == expected for item in observed.values()), 'timed report differs')
        return dict(order=order, millis={policy: item['millis'] for policy, item in observed.items()})
    warmups = [pair(0)]
    print(json.dumps(dict(stage='warmup_complete')), flush=True)
    trials = []
    for index in range(args.repeats):
        trial = pair(index+1)
        trials.append(trial)
        print(json.dumps(dict(stage='trial_complete', trial=index+1, millis=trial['millis'])), flush=True)
    a, b = [item['millis']['normals'] for item in trials], [item['millis']['skip'] for item in trials]
    summary = dict(normalsMedianMillis=statistics.median(a), skipMedianMillis=statistics.median(b),
                   pairedShorteningPercent=statistics.median(100*(1-y/x) for x, y in zip(a, b)),
                   pairedSavedMedianMillis=statistics.median(x-y for x, y in zip(a, b)), fasterTrials=sum(x > y for x, y in zip(a, b)))
    for path, expected_hash in hashes.items():
        require(sha(args.source_root / path) == expected_hash, 'source changed: ' + path)
    for item in [*inputs.values(), extension, normal_library, workload['metrics'], workload['scans']]:
        require(sha(item['path']) == item['sha256'], 'input/build changed')
    report = dict(schemaVersion=1, comparison='scan-icp-unused-normals', createdAt=datetime.now(timezone.utc).isoformat(),
                  sourceHashes=hashes, inputs=inputs, inputManifest=manifest, extension=extension, normalLibrary=normal_library,
                  workload=workload, environment=dict(python=platform.python_version(), numpy=np.__version__, open3d=o3d.__version__,
                      platform=platform.system(), logicalCpus=os.cpu_count(), rayonNumThreads=args.threads,
                      ompNumThreads=args.threads, openblasNumThreads=1),
                  repeats=args.repeats, warmupRounds=1, attempts=args.attempts,
                  parity=parity, firstUse=first, warmups=warmups, trials=trials, summary=summary,
                  note='Actual normal preparation and complete no-artifact diagnostic assembly on three default repetitions of the '
                       'same full real-source scan/map. The control already reuses map loading, NN reference and target normals. '
                       'Both paths run actual loading, copies, native NN preparation/search, statistics, report assembly and cleanup; '
                       'a saved actual ICP result is supplied at the native point-to-point solver boundary after preparation. '
                       'Live solver iterations, artifact/PLY work, parity tracing/hashing/writes and CLI writes are excluded. '
                       'No generated/perturbed/newly thinned coordinates, independent drive-frame or recorded SLAM telemetry claim. '
                       'A traced parity pair, excluded first pair and one warm-up pair precede alternating timed pairs. All observations '
                       'remain. Warmed pages; no cold-ingestion, RSS, full ICP/SLAM runtime or GPU speed claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), summary=summary)), flush=True)


if __name__ == '__main__':
    main()
