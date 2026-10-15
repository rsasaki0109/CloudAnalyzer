"""Compare strict binary XYZ PLY loading against the preceding Open3D loader.

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
    for option in ['source-root', 'reference', 'compared', 'input-manifest', 'io-baseline',
                   'build-index', 'registration-record', 'work-dir', 'output']:
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--io-baseline-sha256', required=True)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--attempts', type=int, default=3)
    options = [value.split('=')[0] for value in sys.argv[1:] if value.startswith('--')]
    require(len(options) == len(set(options)), 'duplicate options')
    args = parser.parse_args()
    require(1 <= args.threads <= 64 and 1 <= args.repeats <= 100, 'invalid threads/repeats')
    require(2 <= args.attempts <= 20, 'invalid attempts')
    require(not args.output.exists() and not args.work_dir.exists(), 'output/work directory must be new')
    require(sha(args.io_baseline) == args.io_baseline_sha256, 'I/O baseline differs')
    os.environ['RAYON_NUM_THREADS'] = os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['CA_DISABLE_RUST_CORE'] = '0'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str((args.source_root / 'cloudanalyzer').resolve()))
    import numpy as np
    import open3d as o3d
    import cloudanalyzer_core as cc
    import cloudanalyzer_core._core as core
    import ca.io as current_io
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

    baseline_io = module('ply_xyz_open3d_baseline', args.io_baseline)
    loaders = dict(open3d=baseline_io.load_point_cloud, bulk=current_io.load_point_cloud)
    files = ['cloudanalyzer/ca/scan_match_debug.py', 'cloudanalyzer/ca/slam_debug.py',
             'cloudanalyzer/ca/metrics.py', 'cloudanalyzer/ca/registration.py', 'cloudanalyzer/ca/io.py',
             'cloudanalyzer/ca/visualization.py', 'cloudanalyzer/ca/_rust.py', 'cloudanalyzer/pyproject.toml',
             'cloudanalyzer/tests/test_io_binary_xyz_ply.py', 'cloudanalyzer/tests/test_io.py', 'cloudanalyzer/tests/test_scan_icp_normals.py', 'cloudanalyzer/tests/test_registration_target_normals.py',
             'cloudanalyzer/tests/test_scan_match_debug.py', 'cloudanalyzer/tests/test_slam_debug.py',
             'cloudanalyzer/tests/test_slam_map_reuse.py', 'cloudanalyzer/tests/test_metrics.py',
             'cloudanalyzer/tests/test_registration.py', 'cloudanalyzer/scripts/benchmark_ply_xyz_loading.py']
    hashes = {path: sha(args.source_root / path) for path in files}
    inputs = {key: info(getattr(args, key)) for key in ['reference', 'compared', 'input_manifest',
                                                      'io_baseline', 'build_index', 'registration_record']}
    inputs.update(originalSource=info(original), registrationSnapshot=info(snapshot['path']))
    extension, normal_library = info(core.__file__), info(o3d.cpu.pybind.__file__)
    build = json.loads(args.build_index.read_text(encoding='utf-8'))
    require(extension['sha256'] == build['extension']['sha256'], 'native extension differs')
    for path, expected in build['sourceHashes'].items():
        require(sha(args.source_root / path) == expected, 'native source differs: ' + path)
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
        registration, scan, slam = current_registration, current_scan, current_slam
        load, prepare = scan.load_point_cloud, scan.prepare_nn_reference
        loader, bulk_loader = loaders[policy], current_io._load_binary_xyz_ply
        estimate = registration._estimate_registration_normals
        native_solver = o3d.pipelines.registration.registration_icp
        distances, capacities, solver_inputs, normal_estimates = [], [], [], []
        loaded_clouds, bulk_selections = [], []
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
                    require(not cloud.has_normals(), 'unexpected normal preparation')
                    fields[label] = dict(pointsSha256=hashlib.sha256(np.asarray(cloud.points).tobytes()).hexdigest(),
                                         normalsSha256=hashlib.sha256(np.asarray(cloud.normals).tobytes()).hexdigest(),
                                         normalCount=len(cloud.normals))
                solver_inputs.append(dict(**fields, distance=distance, initTransform=initial.tolist(),
                                          estimator=type(estimator).__name__, withScaling=estimator.with_scaling))
            return fixed

        o3d.pipelines.registration.registration_icp = solver
        scan.load_point_cloud = loader
        if capture:
            def traced_estimate(cloud):
                estimate(cloud)
                normal_estimates.append(dict(pointsSha256=hashlib.sha256(np.asarray(cloud.points).tobytes()).hexdigest(),
                                             normalsSha256=hashlib.sha256(np.asarray(cloud.normals).tobytes()).hexdigest(),
                                             points=len(cloud.points)))
            def traced_load(path):
                loads['scan' if path == str(args.compared.resolve()) else 'map'] += 1
                cloud = loader(path)
                label = 'initial' if path == str(args.compared.resolve()) else 'reference'
                require(np.asarray(cloud.points).tobytes() == data[label].tobytes(), 'loaded coordinates differ')
                require(not cloud.has_normals() and not cloud.has_colors() and not cloud.has_covariances(), 'loaded attributes differ')
                loaded_clouds.append(dict(path=path, points=len(cloud.points),
                    xyzSha256=hashlib.sha256(np.asarray(cloud.points).tobytes()).hexdigest()))
                return cloud
            def traced_bulk(path):
                cloud = bulk_loader(path)
                bulk_selections.append(dict(path=str(path), selected=cloud is not None))
                return cloud
            current_io._load_binary_xyz_ply = traced_bulk
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
            current_io._load_binary_xyz_ply = bulk_loader
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
                    capturedDistances=captured, solverInputs=solver_inputs, normalEstimates=normal_estimates,
                    loadedClouds=loaded_clouds, bulkSelections=bulk_selections)

    policies = ['open3d', 'bulk']
    parity = {policy: call(policy, True) for policy in policies}
    expected = parity['open3d']['report']
    require(parity['bulk']['report'] == expected, 'complete SLAM reports differ')
    expected_hashes = [item['sha256'] for item in prior['firstUse']['fresh']['capturedDistances'][:2]] * args.attempts
    for policy in policies:
        observed = parity[policy]
        require([item['sha256'] for item in observed['capturedDistances']] == expected_hashes, 'distance bytes differ')
        require(observed['loads'] == dict(scan=args.attempts, map=1) and len(observed['referenceCapacities']) == 1,
                'map reuse policy differs')
        require(len(observed['solverInputs']) == args.attempts, 'solver count differs')
        require(len(observed['normalEstimates']) == 0, 'normal estimate count differs')
        require(len(observed['loadedClouds']) == args.attempts + 1, 'load count differs')
        require(len(observed['bulkSelections']) == (args.attempts + 1 if policy == 'bulk' else 0), 'bulk selection count differs')
        require(all(item['selected'] for item in observed['bulkSelections']), 'bulk layout was not selected')
    require(parity['open3d']['solverInputs'] == parity['bulk']['solverInputs'], 'prepared solver inputs differ')
    require(parity['open3d']['loadedClouds'] == parity['bulk']['loadedClouds'], 'loaded fields differ')
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
    a, b = [item['millis']['open3d'] for item in trials], [item['millis']['bulk'] for item in trials]
    summary = dict(open3dMedianMillis=statistics.median(a), bulkMedianMillis=statistics.median(b),
                   pairedShorteningPercent=statistics.median(100*(1-y/x) for x, y in zip(a, b)),
                   pairedSavedMedianMillis=statistics.median(x-y for x, y in zip(a, b)), fasterTrials=sum(x > y for x, y in zip(a, b)))
    def load_only(policy, capture=False):
        counts, fields = [], []
        start = time.perf_counter_ns()
        for path, label in [(args.compared, 'initial'), (args.reference, 'reference')] + [(args.compared, 'initial')] * (args.attempts - 1):
            cloud = loaders[policy](str(path))
            counts.append(len(cloud.points))
            if capture:
                require(np.asarray(cloud.points).tobytes() == data[label].tobytes(), 'load-only coordinates differ')
                require(not cloud.has_normals() and not cloud.has_colors() and not cloud.has_covariances(), 'load-only attributes differ')
                fields.append(dict(points=len(cloud.points), xyzSha256=hashlib.sha256(np.asarray(cloud.points).tobytes()).hexdigest()))
            del cloud
        elapsed = (time.perf_counter_ns() - start) / 1e6
        require(counts == [manifest['inputs']['compared']['points'], manifest['inputs']['reference']['points']]
                + [manifest['inputs']['compared']['points']] * (args.attempts - 1), 'loaded count differs')
        return dict(millis=elapsed, fields=fields)
    load_parity = {policy: load_only(policy, True) for policy in policies}
    require(load_parity['open3d']['fields'] == load_parity['bulk']['fields'], 'load-only field parity differs')
    load_first = {policy: load_only(policy) for policy in policies}
    def load_pair(index):
        order = policies if index % 2 == 0 else list(reversed(policies))
        observed = {policy: load_only(policy) for policy in order}
        return dict(order=order, millis={policy: item['millis'] for policy, item in observed.items()})
    load_warmups = [load_pair(0)]
    load_trials = [load_pair(index + 1) for index in range(args.repeats)]
    a, b = [item['millis']['open3d'] for item in load_trials], [item['millis']['bulk'] for item in load_trials]
    load_summary = dict(open3dMedianMillis=statistics.median(a), bulkMedianMillis=statistics.median(b),
        pairedShorteningPercent=statistics.median(100*(1-y/x) for x, y in zip(a, b)),
        pairedSavedMedianMillis=statistics.median(x-y for x, y in zip(a, b)), fasterTrials=sum(x > y for x, y in zip(a, b)))
    loading = dict(parity=load_parity, firstUse=load_first, warmupRounds=1, warmups=load_warmups, trials=load_trials, summary=load_summary)
    print(json.dumps(dict(stage='load_only_complete', summary=load_summary)), flush=True)
    for path, expected_hash in {**hashes, **build['sourceHashes']}.items():
        require(sha(args.source_root / path) == expected_hash, 'source changed: ' + path)
    for item in [*inputs.values(), extension, normal_library, workload['metrics'], workload['scans']]:
        require(sha(item['path']) == item['sha256'], 'input/build changed')
    report = dict(schemaVersion=1, comparison='ply-xyz-loading', createdAt=datetime.now(timezone.utc).isoformat(),
                  sourceHashes=hashes, nativeSourceHashes=build['sourceHashes'], inputs=inputs, inputManifest=manifest, extension=extension, normalLibrary=normal_library,
                  workload=workload, environment=dict(python=platform.python_version(), numpy=np.__version__, open3d=o3d.__version__,
                      platform=platform.system(), logicalCpus=os.cpu_count(), rayonNumThreads=args.threads,
                      ompNumThreads=args.threads, openblasNumThreads=1),
                  repeats=args.repeats, warmupRounds=1, attempts=args.attempts,
                  parity=parity, firstUse=first, warmups=warmups, trials=trials, summary=summary, loading=loading,
                  note='Complete no-artifact diagnostics on three default repetitions of the same full real-source scan/map. '
                       'Both policies already skip unused ICP normals and share map loading and its NN reference. Only the I/O '
                       'loader changes: eligible little-endian binary PLY with only double XYZ fields uses a bounded bulk read. '
                       'Both execute actual loading, cloud copies, native NN preparation/search, statistics, report assembly, '
                       'transformed result construction and cleanup. A saved actual full-source ICP result is supplied at the '
                       'native solver boundary after preparation. Live solver iterations, artifact/PLY output, tracing/hashing/ '
                       'writes and CLI writes are excluded. The separately timed loading interval sequentially loads and releases '
                       'the same four default inputs without the rest of the diagnostic. Both intervals have excluded parity '
                       'and first-use pairs, one warm-up pair and five default alternating timed pairs; no observations removed. '
                       'Measured coordinates are unchanged, with no generated geometry, perturbation or additional thinning. '
                       'Repeated attempts are not independent drive frames or recorded SLAM telemetry. Warmed file pages; no '
                       'cold-ingestion, RSS, full iterative ICP/SLAM or GPU performance claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), summary=summary)), flush=True)


if __name__ == '__main__':
    main()
