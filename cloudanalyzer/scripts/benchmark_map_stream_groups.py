"""Compare complete streaming AWD/SCS with stable per-chunk voxel groups."""

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
    for name in ['source-root', 'reference', 'compared', 'input-manifest', 'baseline', 'work-dir', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--baseline-sha256', required=True)
    parser.add_argument('--repeats', type=int, default=3)
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
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() or 1)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    sys.path.insert(0, str((args.source_root / 'cloudanalyzer').resolve()))
    import numpy as np
    import scipy
    import scipy.spatial._ckdtree as tree_binary
    import open3d as o3d
    import ca.core.map_evaluate as current
    from ca.io import PointAccumulator, load_point_cloud

    spec = importlib.util.spec_from_file_location('map_stream_mask_baseline', args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    manifest = json.loads(args.input_manifest.read_text(encoding='utf-8'))
    for name in ['reference', 'compared']:
        require(sha(getattr(args, name)) == manifest['inputs'][name]['sha256'], name + ' differs')
    original = (args.source_root / 'web' / manifest['source']['path']).resolve()
    require(sha(original) == manifest['source']['sha256'], 'original source differs')
    inputs = {name: info(getattr(args, name)) for name in ['reference', 'compared', 'input_manifest', 'baseline']}
    inputs['originalSource'] = info(original)
    compiled_tree = info(tree_binary.__file__)
    files = ['cloudanalyzer/ca/core/map_evaluate.py', 'cloudanalyzer/ca/io.py', 'cloudanalyzer/ca/_rust.py',
             'cloudanalyzer/pyproject.toml', 'cloudanalyzer/tests/test_map_nn_parallel.py',
             'cloudanalyzer/tests/test_map_scs_index.py', 'cloudanalyzer/tests/test_map_covariance.py',
             'cloudanalyzer/tests/test_map_grouping.py', 'cloudanalyzer/tests/test_map_stream_groups.py',
             'cloudanalyzer/scripts/benchmark_map_stream_groups.py']
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
        module = baseline if policy == 'mask' else current
        builder = module.build_voxel_gaussians_from_chunks
        gaussians = []
        if capture:
            def traced_builder(values, size):
                digest = hashlib.sha256()
                batch_calls = batch_points = 0
                update = PointAccumulator.update
                def traced_update(self, points):
                    nonlocal batch_calls, batch_points
                    digest.update(struct.pack('<Q', len(points)))
                    digest.update(np.asarray(points, dtype='<f8').tobytes())
                    batch_calls += 1
                    batch_points += len(points)
                    return update(self, points)
                PointAccumulator.update = traced_update
                try:
                    voxels = builder(values, size)
                finally:
                    PointAccumulator.update = update
                path = args.work_dir / (policy + '-gaussians-' + str(len(gaussians)) + '.bin')
                packer = struct.Struct('<4q12d')
                with path.open('xb') as stream:
                    for key, voxel in voxels.items():
                        stream.write(packer.pack(*key, voxel.num_points, *voxel.mu, *voxel.sigma.ravel()))
                gaussians.append({**info(path), 'voxels': len(voxels), 'recordBytes': packer.size,
                                  'updateCalls': batch_calls, 'updatePoints': batch_points, 'updatesSha256': digest.hexdigest()})
                return voxels
            module.build_voxel_gaussians_from_chunks = traced_builder
        start = time.perf_counter_ns()
        try:
            result = module.evaluate_map_streaming(chunks('compared'), chunks('reference'),
                         protocol=module.MapEvalProtocol(voxel_size_m=args.structure_voxel_size))
        finally:
            module.build_voxel_gaussians_from_chunks = builder
        millis = (time.perf_counter_ns() - start) / 1e6
        report = asdict(result)
        return dict(millis=millis, report=report, reportSha256=report_sha(report), gaussians=gaussians)

    parity = {policy: call(policy, True) for policy in ['mask', 'stable']}
    require(parity['mask']['report'] == parity['stable']['report'], 'complete streaming reports differ')
    require(len(parity['mask']['gaussians']) == len(parity['stable']['gaussians']) == 2, 'Gaussian count differs')
    for name, a, b in zip(['compared', 'reference'], parity['mask']['gaussians'], parity['stable']['gaussians'], strict=True):
        for key in ['sha256', 'bytes', 'voxels', 'recordBytes', 'updateCalls', 'updatePoints', 'updatesSha256']:
            require(a[key] == b[key], 'Gaussian/update stream differs: ' + key)
        require(a['updatePoints'] == snapshots[name]['points'], 'update point count differs')
    print(json.dumps(dict(stage='parity_complete', source=manifest['source']['path'])), flush=True)
    first = {policy: call(policy) for policy in ['mask', 'stable']}
    expected = parity['mask']['report']
    require(all(item['report'] == expected for item in first.values()), 'first reports differ')

    def pair(index):
        order = ['mask', 'stable'] if index % 2 == 0 else ['stable', 'mask']
        observed = {policy: call(policy) for policy in order}
        require(all(item['report'] == expected for item in observed.values()), 'timed report differs')
        return dict(order=order, millis={key: value['millis'] for key, value in observed.items()},
                    reportHashes={key: value['reportSha256'] for key, value in observed.items()})

    warmups = [pair(0)]
    print(json.dumps(dict(stage='warmup_complete')), flush=True)
    trials = []
    for index in range(args.repeats):
        observed = pair(index + 1)
        trials.append(observed)
        print(json.dumps(dict(stage='trial_complete', trial=index + 1, millis=observed['millis'])), flush=True)
    a, b = [item['millis']['mask'] for item in trials], [item['millis']['stable'] for item in trials]
    summary = dict(maskMedianMillis=statistics.median(a), stableMedianMillis=statistics.median(b),
                   pairedShorteningPercent=statistics.median(100 * (1 - y / x) for x, y in zip(a, b)),
                   pairedSavedMedianMillis=statistics.median(x - y for x, y in zip(a, b)), fasterTrials=sum(x > y for x, y in zip(a, b)))
    for path, expected_hash in hashes.items():
        require(sha(args.source_root / path) == expected_hash, 'source changed: ' + path)
    for item in [*inputs.values(), compiled_tree, *snapshots.values()]:
        require(sha(item['path']) == item['sha256'], 'input/library changed')
    report = dict(schemaVersion=1, comparison='map-stream-groups', createdAt=datetime.now(timezone.utc).isoformat(),
        sourceHashes=hashes, inputs=inputs, inputManifest=manifest, coordinateSnapshots=snapshots, compiledTree=compiled_tree,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, open3d=o3d.__version__,
                         platform=platform.system(), logicalCpus=os.cpu_count(), ompNumThreads=os.environ['OMP_NUM_THREADS'],
                         openblasNumThreads=os.environ['OPENBLAS_NUM_THREADS']),
        policy=dict(groupSort='stable', packedVoxelMinPoints=current._PACKED_VOXEL_MIN_POINTS,
                    packedCapacityMax=int(np.iinfo(np.int64).max), scsIndexMinVoxels=current._SCS_INDEX_MIN_VOXELS,
                    scsExactSpan=current._SCS_INDEX_EXACT_SPAN),
        repeats=args.repeats, warmupRounds=1, chunkSize=args.chunk_size, structureVoxelSize=args.structure_voxel_size,
        parity=parity, firstUse=first, warmups=warmups, trials=trials, summary=summary,
        accuracy=dict(gaussianMapsChecked=4, orderedBatchStreamsChecked=4, reportsChecked=2 * (3 + args.repeats)),
        note='Complete streaming AWD/SCS reduction includes iterating fixed XYZ f64 memory-map chunks, integer classification, '
             'stable grouping, per-voxel PointAccumulator updates, finalization, Wasserstein and SCS index setup/destruction. '
             'The control uses preceding axis classification and one full-chunk boolean scan per voxel. Source coordinates, '
             'point order and chunk boundaries are unchanged; no generated data. Original-file loading, coordinate snapshot '
             'preparation/mapping/hashing, Gaussian/ordered-update tracing and CLI writes are outside clocks. The parity pair, '
             'an excluded first pair and one warm-up pair precede alternating timed pairs. No observations removed. Pages are '
             'warmed; this is not cold storage/CSV/LAS/PCD/PLY I/O or peak-RSS measurement. No NN/PLY, resident-evaluation, GPU '
             'or persistent-cache speed claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), summary=summary)), flush=True)


if __name__ == '__main__':
    main()
