"""Compare complete streaming AWD/SCS with expanded groups and bounded gather blocks."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import struct
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def info(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def report_sha(report):
    return hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source-root', 'reference', 'compared', 'input-manifest', 'baseline', 'io-baseline', 'work-dir', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--baseline-sha256', required=True)
    parser.add_argument('--io-baseline-sha256', required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--chunk-size', type=int, default=100_000)
    parser.add_argument('--structure-voxel-size', type=float, default=3.)
    options = [value.split('=')[0] for value in sys.argv[1:] if value.startswith('--')]
    require(len(options) == len(set(options)), 'duplicate options')
    args = parser.parse_args()
    require(1 <= args.repeats <= 100, 'invalid repeats')
    require(args.chunk_size > 0, 'invalid chunk size')
    require(math.isfinite(args.structure_voxel_size) and args.structure_voxel_size > 0, 'invalid structure voxel size')
    require(not args.work_dir.exists() and not args.output.exists(), 'output/work directory must be new')
    require(sha(args.baseline) == args.baseline_sha256, 'baseline differs')
    require(sha(args.io_baseline) == args.io_baseline_sha256, 'IO baseline differs')
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() or 1)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    sys.path.insert(0, str((args.source_root / 'cloudanalyzer').resolve()))
    import numpy as np
    import scipy
    import scipy.spatial._ckdtree as tree_binary
    import open3d as o3d
    import ca.core.map_evaluate as current
    import ca.io as current_io
    from ca.io import PointAccumulator, load_point_cloud

    spec = importlib.util.spec_from_file_location('map_stream_small_moments_baseline', args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    io_spec = importlib.util.spec_from_file_location('stream_small_io_baseline', args.io_baseline)
    baseline_io = importlib.util.module_from_spec(io_spec)
    sys.modules[io_spec.name] = baseline_io
    io_spec.loader.exec_module(baseline_io)
    manifest = json.loads(args.input_manifest.read_text(encoding='utf-8'))
    for name in ['reference', 'compared']:
        require(sha(getattr(args, name)) == manifest['inputs'][name]['sha256'], name + ' differs')
    original = (args.source_root / 'web' / manifest['source']['path']).resolve()
    require(sha(original) == manifest['source']['sha256'], 'original source differs')
    inputs = {name: info(getattr(args, name)) for name in ['reference', 'compared', 'input_manifest', 'baseline', 'io_baseline']}
    inputs['originalSource'] = info(original)
    compiled_tree = info(tree_binary.__file__)
    files = ['cloudanalyzer/ca/core/map_evaluate.py', 'cloudanalyzer/ca/io.py', 'cloudanalyzer/ca/_rust.py', 'cloudanalyzer/pyproject.toml', 'cloudanalyzer/tests/test_map_nn_parallel.py', 'cloudanalyzer/tests/test_map_scs_index.py', 'cloudanalyzer/tests/test_map_covariance.py', 'cloudanalyzer/tests/test_map_grouping.py', 'cloudanalyzer/tests/test_map_stream_groups.py', 'cloudanalyzer/tests/test_map_downsample_centroids.py', 'cloudanalyzer/tests/test_map_downsample_small_groups.py', 'cloudanalyzer/tests/test_map_mean_sums.py', 'cloudanalyzer/tests/test_map_wasserstein_batch.py', 'cloudanalyzer/tests/test_map_scs_batch.py', 'cloudanalyzer/tests/test_map_nn_midpoint.py', 'cloudanalyzer/tests/test_map_scs_query_batch.py', 'cloudanalyzer/tests/test_map_stream_accumulators.py', 'cloudanalyzer/tests/test_map_stream_moments.py', 'cloudanalyzer/tests/test_io.py', 'cloudanalyzer/tests/test_map_stream_expanded.py', 'cloudanalyzer/scripts/benchmark_map_stream_expanded.py']
    hashes = {path: sha(args.source_root / path) for path in files}
    args.work_dir.mkdir(parents=True, exist_ok=False)
    snapshots, mapped = {}, {}
    for name in ['compared', 'reference']:
        cloud = load_point_cloud(str(getattr(args, name)))
        points = np.asarray(cloud.points)
        require(len(points) == manifest['inputs'][name]['points'], name + ' point count differs')
        path = args.work_dir / (name + '-xyz-f64.bin')
        with path.open('xb') as stream:
            stream.write(np.asarray(points, dtype='<f8').tobytes())
        snapshots[name] = {**info(path), 'points': len(points)}
        mapped[name] = np.memmap(path, mode='r', dtype='<f8', shape=(len(points), 3))
        del points, cloud

    def chunks(name):
        points = mapped[name]
        for start in range(0, len(points), args.chunk_size):
            yield np.asarray(points[start:start + args.chunk_size])

    def call(policy, capture=False):
        module = baseline if policy == 'small' else current
        accumulator_class = baseline_io.PointAccumulator if policy == 'small' else PointAccumulator
        builder = module.build_voxel_gaussians_from_chunks
        current_io.PointAccumulator = accumulator_class
        gaussians = []
        if capture:
            def traced_builder(values, size):
                digest = hashlib.sha256()
                group_calls = group_points = constructors = batched_groups = batched_points = 0
                bounds = []
                update = accumulator_class.update
                constructor = accumulator_class.__init__
                finalizer = accumulator_class.finalize
                handler = getattr(module, '_update_stream_voxel', None)
                def traced_constructor(self, *args, **kwargs):
                    nonlocal constructors
                    constructors += 1
                    constructor(self, *args, **kwargs)
                def trace(points, batched):
                    nonlocal group_calls, group_points, batched_groups, batched_points
                    digest.update(struct.pack('<Q', len(points)))
                    digest.update(np.asarray(points, dtype='<f8').tobytes())
                    group_calls += 1
                    group_points += len(points)
                    batched_groups += int(batched)
                    batched_points += len(points) * int(batched)
                def traced_handler(accumulator, points, moments, slot):
                    trace(points, moments is not None and moments.selected[slot])
                    return handler(accumulator, points, moments, slot)
                def traced_finalize(self):
                    summary = finalizer(self)
                    bounds.append(struct.pack('<6d', *summary['minimum'], *summary['maximum']))
                    return summary
                accumulator_class.__init__ = traced_constructor
                accumulator_class.finalize = traced_finalize
                module._update_stream_voxel = traced_handler
                try:
                    voxels = builder(values, size)
                finally:
                    accumulator_class.__init__ = constructor
                    accumulator_class.finalize = finalizer
                    accumulator_class.update = update
                    module._update_stream_voxel = handler
                path = args.work_dir / (policy + '-gaussians-' + str(len(gaussians)) + '.bin')
                packer = struct.Struct('<4q12d')
                with path.open('xb') as stream:
                    for key, voxel in voxels.items():
                        stream.write(packer.pack(*key, voxel.num_points, *voxel.mu, *voxel.sigma.ravel()))
                bounds_path = args.work_dir / (policy + '-bounds-' + str(len(gaussians)) + '.bin')
                with bounds_path.open('xb') as stream:
                    for key, payload in zip(voxels, bounds, strict=True):
                        stream.write(struct.pack('<3q', *key) + payload)
                gaussians.append({**info(path), 'voxels': len(voxels), 'recordBytes': packer.size,
                    'bounds': {**info(bounds_path), 'recordBytes': 72}, 'constructorCalls': constructors,
                    'groupCalls': group_calls, 'groupPoints': group_points, 'groupsSha256': digest.hexdigest(),
                    'scalarUpdates': group_calls - batched_groups, 'batchedGroups': batched_groups, 'batchedPoints': batched_points})
                return voxels
            module.build_voxel_gaussians_from_chunks = traced_builder
        start = time.perf_counter_ns()
        try:
            result = module.evaluate_map_streaming(chunks('compared'), chunks('reference'),
                         protocol=module.MapEvalProtocol(voxel_size_m=args.structure_voxel_size))
        finally:
            module.build_voxel_gaussians_from_chunks = builder
            current_io.PointAccumulator = PointAccumulator
        millis = (time.perf_counter_ns() - start) / 1e6
        report = asdict(result)
        return dict(millis=millis, report=report, reportSha256=report_sha(report), gaussians=gaussians)

    parity = {policy: call(policy, True) for policy in ['small', 'expanded']}
    require(parity['small']['report'] == parity['expanded']['report'], 'complete streaming reports differ')
    require(len(parity['small']['gaussians']) == len(parity['expanded']['gaussians']) == 2, 'Gaussian count differs')
    for name, a, b in zip(['compared', 'reference'], parity['small']['gaussians'], parity['expanded']['gaussians'], strict=True):
        for key in ['sha256', 'bytes', 'voxels', 'recordBytes', 'groupCalls', 'groupPoints', 'groupsSha256', 'constructorCalls']:
            require(a[key] == b[key], 'Gaussian/group stream differs: ' + key)
        for key in ['sha256', 'bytes', 'recordBytes']:
            require(a['bounds'][key] == b['bounds'][key], 'bounds differ: ' + key)
        require(a['groupPoints'] == snapshots[name]['points'], 'group point count differs')
        require(a['constructorCalls'] == b['constructorCalls'] == b['voxels'], 'allocation count differs')
        require(a['scalarUpdates'] + a['batchedGroups'] == a['groupCalls'], 'control dispatch count differs')
        require(b['scalarUpdates'] + b['batchedGroups'] == b['groupCalls'], 'dispatch count differs')
    print(json.dumps(dict(stage='parity_complete', source=manifest['source']['path'])), flush=True)
    first = {policy: call(policy) for policy in ['small', 'expanded']}
    expected = parity['small']['report']
    require(all(item['report'] == expected for item in first.values()), 'first reports differ')

    def pair(index):
        order = ['small', 'expanded'] if index % 2 == 0 else ['expanded', 'small']
        observed = {policy: call(policy) for policy in order}
        require(all(item['report'] == expected for item in observed.values()), 'timed report differs')
        return dict(order=order, millis={key: value['millis'] for key, value in observed.items()},
                    reportHashes={key: value['reportSha256'] for key, value in observed.items()})

    warmups = [pair(index) for index in range(2)]
    print(json.dumps(dict(stage='warmup_complete')), flush=True)
    trials = []
    for index in range(args.repeats):
        observed = pair(index + 2)
        trials.append(observed)
        print(json.dumps(dict(stage='trial_complete', trial=index + 1, millis=observed['millis'])), flush=True)
    a, b = [item['millis']['small'] for item in trials], [item['millis']['expanded'] for item in trials]
    summary = dict(smallMedianMillis=statistics.median(a), expandedMedianMillis=statistics.median(b),
                   pairedShorteningPercent=statistics.median(100 * (1 - y / x) for x, y in zip(a, b)),
                   pairedSavedMedianMillis=statistics.median(x - y for x, y in zip(a, b)), fasterTrials=sum(x > y for x, y in zip(a, b)))
    for path, expected_hash in hashes.items():
        require(sha(args.source_root / path) == expected_hash, 'source changed: ' + path)
    for item in [*inputs.values(), compiled_tree, *snapshots.values()]:
        require(sha(item['path']) == item['sha256'], 'input/library changed')
    report = dict(schemaVersion=1, comparison='map-stream-expanded', createdAt=datetime.now(timezone.utc).isoformat(),
        sourceHashes=hashes, inputs=inputs, inputManifest=manifest, coordinateSnapshots=snapshots, compiledTree=compiled_tree,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, open3d=o3d.__version__,
                         platform=platform.system(), logicalCpus=os.cpu_count(), ompNumThreads=os.environ['OMP_NUM_THREADS'],
                         openblasNumThreads=os.environ['OPENBLAS_NUM_THREADS']),
        policy=dict(groupSort='stable', accumulatorAllocation='first_occurrence', packedVoxelMinPoints=current._PACKED_VOXEL_MIN_POINTS,
                    packedCapacityMax=int(np.iinfo(np.int64).max), scsIndexMinVoxels=current._SCS_INDEX_MIN_VOXELS,
                    scsExactSpan=current._SCS_INDEX_EXACT_SPAN,
                    momentBatchMinPoints=current._STREAM_MOMENT_BATCH_MIN_POINTS,
                    momentBatchMaxGroups=current._STREAM_MOMENT_BATCH_MAX_GROUPS,
                    momentBatchMaxGroupPoints=current._STREAM_MOMENT_BATCH_MAX_GROUP_POINTS,
                    momentBatchMinSameSize=current._STREAM_MOMENT_BATCH_MIN_SAME_SIZE,
                    momentBatchMinSelected=current._STREAM_MOMENT_BATCH_MIN_SELECTED,
                    momentBatchSafeMagnitude=current._STREAM_MOMENT_BATCH_SAFE_MAGNITUDE,
                    momentBatchMaxPoints=current._STREAM_MOMENT_BATCH_MAX_POINTS,
                    priorMomentBatchMaxGroupPoints=baseline._STREAM_MOMENT_BATCH_MAX_GROUP_POINTS,
                    priorMomentBatchMinSameSize=baseline._STREAM_MOMENT_BATCH_MIN_SAME_SIZE),
        repeats=args.repeats, warmupRounds=2, chunkSize=args.chunk_size, structureVoxelSize=args.structure_voxel_size,
        parity=parity, firstUse=first, warmups=warmups, trials=trials, summary=summary,
        accuracy=dict(gaussianMapsChecked=4, orderedGroupStreamsChecked=4, boundsMapsChecked=4, reportsChecked=2 * (4 + args.repeats)),
        note='Complete streaming AWD/SCS reduction includes fixed XYZ f64 memory-map chunk iteration, integer '
             'classification, stable grouping, allocation, moment reductions/merges/finalization, Wasserstein and '
             'SCS setup/queries/statistics. Control already contains preceding 1-32-point moment batches requiring '
             '32 groups per size, allocation reuse and all preceding MapEval optimizations. Both policies require '
             '128 selected groups per 1024-group window and retain preceding chunk/numerical guards. Current allows '
             '1-128-point groups with eight groups per size, splitting each size cohort into at most 32768-point '
             'gather blocks and releasing gathered/centered/mean scratch before each next block. Merging order and '
             'arithmetic stay fixed. Source coordinates, sampling and 100000-point chunk boundaries are unchanged; '
             'no generated data. Original-file loading, snapshot preparation/mapping/hashing, Gaussian/bounds/group/ '
             'allocation tracing and CLI writes are outside clocks. A parity pair, excluded first pair and two warm-ups '
             'precede five default alternating timed pairs. All observations remain. Pages are warmed; cold ingestion '
             'and peak RSS are not measured. No NN/PLY, resident-evaluation, GPU, worker-pool or persistent-cache speed claim.')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), summary=summary)), flush=True)


if __name__ == '__main__':
    main()
