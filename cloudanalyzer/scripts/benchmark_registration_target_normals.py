"""Measure complete normal preparation for repeated registration attempts.

Both source and target normals are estimated normally. A recorded actual ICP
solver result isolates preparation, owned copies and transformed-result work;
this does not measure solver iterations or the entire SLAM workflow.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ['source-root', 'reference', 'compared', 'input-manifest', 'baseline',
                   'registration-record', 'work-dir', 'output']:
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--baseline-sha256', required=True)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    options = [value.split('=')[0] for value in sys.argv[1:] if value.startswith('--')]
    require(len(options) == len(set(options)), 'duplicate options')
    args = parser.parse_args()
    require(1 <= args.threads <= 64 and 1 <= args.repeats <= 100, 'invalid threads/repeats')
    require(2 <= args.attempts <= 20, 'invalid attempts')
    require(not args.work_dir.exists() and not args.output.exists(), 'output/work directory must be new')
    require(sha(args.baseline) == args.baseline_sha256, 'baseline differs')
    os.environ['OMP_NUM_THREADS'] = os.environ['RAYON_NUM_THREADS'] = str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str((args.source_root / 'cloudanalyzer').resolve()))
    import numpy as np
    import open3d as o3d
    import ca.registration as current
    from ca.io import load_point_cloud

    manifest = json.loads(args.input_manifest.read_text(encoding='utf-8'))
    prior = json.loads(args.registration_record.read_text(encoding='utf-8'))
    require(prior['comparison'] == 'scan-match-distance-reuse', 'unexpected registration record')
    require(prior['inputManifest'] == manifest, 'registration manifest differs')
    for key in ['compared', 'reference']:
        require(sha(getattr(args, key)) == manifest['inputs'][key]['sha256'] == prior['inputs'][key]['sha256'], key + ' differs')
    original = (args.source_root / 'web' / manifest['source']['path']).resolve()
    require(sha(original) == manifest['source']['sha256'], 'original source differs')
    snapshot = prior['registration']['snapshot']
    require(sha(snapshot['path']) == snapshot['sha256'], 'registration snapshot differs')
    data = np.load(snapshot['path'])
    clouds = [load_point_cloud(str(path)) for path in [args.compared, args.reference]]
    for key, cloud in zip(['initial', 'reference'], clouds, strict=True):
        require(np.asarray(cloud.points).tobytes() == data[key].tobytes(), 'recorded coordinates differ')
        require(not cloud.has_normals() and not cloud.has_covariances(), 'unexpected provided attributes')
    matrix = np.asarray(prior['registration']['transformation'])
    require(matrix.tobytes() == data['transformation'].tobytes(), 'recorded matrix differs')
    spec = importlib.util.spec_from_file_location('registration_fresh_target_normals', args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    files = ['cloudanalyzer/ca/registration.py', 'cloudanalyzer/ca/scan_match_debug.py', 'cloudanalyzer/ca/slam_debug.py',
             'cloudanalyzer/ca/io.py', 'cloudanalyzer/ca/metrics.py', 'cloudanalyzer/ca/_rust.py',
             'cloudanalyzer/pyproject.toml', 'cloudanalyzer/tests/test_registration_target_normals.py',
             'cloudanalyzer/tests/test_slam_map_reuse.py', 'cloudanalyzer/tests/test_slam_debug.py',
             'cloudanalyzer/tests/test_registration.py', 'cloudanalyzer/tests/test_scan_match_debug.py',
             'cloudanalyzer/tests/test_metrics.py', 'cloudanalyzer/scripts/benchmark_registration_target_normals.py']
    hashes = {path: sha(args.source_root / path) for path in files}
    inputs = {key: info(getattr(args, key)) for key in ['compared', 'reference', 'input_manifest', 'baseline', 'registration_record']}
    inputs.update(originalSource=info(original), registrationSnapshot=info(snapshot['path']))
    normal_library = info(o3d.cpu.pybind.__file__)
    args.work_dir.mkdir(parents=True, exist_ok=False)
    native_solver = o3d.pipelines.registration.registration_icp

    def call(policy, capture=False):
        module = baseline if policy == 'fresh' else current
        snapshots = []
        def solver(source, target, distance, initial, estimator):
            if capture:
                require(distance == 1. and initial.tobytes() == np.eye(4).tobytes(), 'solver options differ')
                for cloud, key in [(source, 'initial'), (target, 'reference')]:
                    require(np.asarray(cloud.points).tobytes() == data[key].tobytes(), 'solver coordinates differ')
                    require(np.asarray(cloud.normals).tobytes() == data[key + '_normals'].tobytes(), 'solver normals differ')
                snapshots.append(dict(sourceNormalsSha256=hashlib.sha256(np.asarray(source.normals).tobytes()).hexdigest(),
                                      targetNormalsSha256=hashlib.sha256(np.asarray(target.normals).tobytes()).hexdigest()))
                if len(snapshots) == 1:
                    for label, cloud in [('source', source), ('target', target)]:
                        path = args.work_dir / f'{policy}-{label}-normals.bin'
                        with path.open('xb') as stream:
                            stream.write(np.asarray(cloud.normals).astype('<f8', copy=False).tobytes())
            return SimpleNamespace(transformation=matrix, fitness=prior['registration']['fitness'], inlier_rmse=prior['registration']['rmse'])
        o3d.pipelines.registration.registration_icp = solver
        output_fields = []
        scalar_results = []
        start = time.perf_counter_ns()
        try:
            cache = current._TargetNormalCache() if policy == 'reuse' else None
            for _ in range(args.attempts):
                source, target = [o3d.geometry.PointCloud(cloud) for cloud in clouds]
                kwargs = dict(method='icp', max_correspondence_distance=1.)
                if cache is not None:
                    kwargs['_target_normals'] = cache
                result = module.register_detailed(source, target, **kwargs)
                scalar_results.append((result.fitness, result.rmse, result.transformation))
                if capture:
                    require(np.asarray(result.transformed.points).tobytes() == data['aligned'].tobytes(), 'aligned points differ')
                    require(np.asarray(result.transformed.normals).tobytes() == data['aligned_normals'].tobytes(), 'aligned normals differ')
                    output_fields.append(dict(fitness=result.fitness, rmse=result.rmse,
                                              transformation=result.transformation.tolist(),
                                              alignedPointsSha256=hashlib.sha256(np.asarray(result.transformed.points).tobytes()).hexdigest(),
                                              alignedNormalsSha256=hashlib.sha256(np.asarray(result.transformed.normals).tobytes()).hexdigest()))
                del source, target, result, kwargs
            normal_bytes = cache._normals.nbytes if cache is not None else 0
            if cache is not None:
                cache.clear()
            del cache
        finally:
            o3d.pipelines.registration.registration_icp = native_solver
        elapsed = (time.perf_counter_ns()-start)/1e6
        require(all(fitness == prior['registration']['fitness'] and rmse == prior['registration']['rmse']
                    and transform.tobytes() == matrix.tobytes() for fitness, rmse, transform in scalar_results), 'registration result fields differ')
        files = {label: info(args.work_dir / f'{policy}-{label}-normals.bin') for label in ['source', 'target']} if capture else {}
        return dict(millis=elapsed, solverInputs=snapshots, outputFields=output_fields, normalArrays=files, normalSnapshotBytes=normal_bytes)

    parity = {policy: call(policy, True) for policy in ['fresh', 'reuse']}
    for field in ['solverInputs', 'outputFields']:
        require(parity['fresh'][field] == parity['reuse'][field] and len(parity['fresh'][field]) == args.attempts, 'full prepared/output fields differ')
    require(parity['reuse']['normalSnapshotBytes'] == 24*len(clouds[1].points), 'snapshot payload differs')
    print(json.dumps(dict(stage='parity_complete')), flush=True)
    first = {policy: call(policy) for policy in ['fresh', 'reuse']}
    def pair(index):
        order = ['fresh', 'reuse'] if index % 2 == 0 else ['reuse', 'fresh']
        observed = {policy: call(policy) for policy in order}
        return dict(order=order, millis={policy: value['millis'] for policy, value in observed.items()})
    warmups = [pair(0)]
    print(json.dumps(dict(stage='warmup_complete')), flush=True)
    trials = []
    for index in range(args.repeats):
        item = pair(index+1)
        trials.append(item)
        print(json.dumps(dict(stage='trial_complete', trial=index+1, millis=item['millis'])), flush=True)
    a, b = [item['millis']['fresh'] for item in trials], [item['millis']['reuse'] for item in trials]
    summary = dict(freshMedianMillis=statistics.median(a), reuseMedianMillis=statistics.median(b),
                   pairedShorteningPercent=statistics.median(100*(1-y/x) for x, y in zip(a,b)),
                   fasterTrials=sum(x > y for x, y in zip(a,b)))
    for path, expected in hashes.items():
        require(sha(args.source_root / path) == expected, 'source changed')
    for item in [*inputs.values(), normal_library]:
        require(sha(item['path']) == item['sha256'], 'input changed')
    report = dict(schemaVersion=1, comparison='registration-target-normal-reuse', createdAt=datetime.now(timezone.utc).isoformat(),
                  sourceHashes=hashes, inputs=inputs, inputManifest=manifest, normalLibrary=normal_library,
                  environment=dict(python=platform.python_version(), numpy=np.__version__, open3d=o3d.__version__, platform=platform.system(),
                                   logicalCpus=os.cpu_count(), ompNumThreads=args.threads, openblasNumThreads=1),
                  attempts=args.attempts, repeats=args.repeats, warmupRounds=1, parity=parity, firstUse=first,
                  warmups=warmups, trials=trials, summary=summary,
                  policy=dict(sourceNormalPreparations=args.attempts, freshTargetNormalPreparations=args.attempts,
                              reusedTargetNormalPreparations=1, radius=.5, maxNeighbors=30),
                  note='Complete preparation for three default repeated calls on the same full real-source scan/map: cloud copies, '
                       'source/target normal estimation, cached validation/copies, fixed actual recorded ICP result, transformed output and cleanup. '
                       'No point coordinates are generated, perturbed or additionally thinned. The same recorded ICP solve is supplied at '
                       'the Open3D solver boundary, after actual normal preparation. Actual solve iterations, loading, ranking/NN diagnostics, '
                       'artifact/PLY work, capture hashing and writes are excluded from timing summaries. Source normals are recomputed '
                       'each attempt. A traced parity pair, excluded first pair and one warm-up pair precede alternating timed pairs; all '
                       'observations remain. This isolates registration preparation, not full ICP/GICP or SLAM performance. Native solver '
                       'correctness is checked separately; no bit-identical parallel live-solver output or GPU speed claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), summary=summary)), flush=True)


if __name__ == '__main__':
    main()
