"""Reproduce rejected, bounded streaming moment-merge candidates in isolation."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import struct
import sys
import time
import types


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


_GENERATOR = 'candidate = original\ndef replace_once(old, new):\n    global candidate\n    assert candidate.count(old) == 1, old\n    candidate = candidate.replace(old, new)\nreplace_once(\'    maximum: np.ndarray\\n\\n\\ndef _stream_moments_batch_allowed\',\n             \'    maximum: np.ndarray\\n    merged: np.ndarray | None = None\\n    totals: np.ndarray | None = None\\n\\n\\n_STREAM_MERGE_BATCH_MIN_GROUPS = 128\\n_STREAM_MERGE_EXACT_COUNT = 2**53\\n\\n\\ndef _stream_moments_batch_allowed\')\nhelper = \'\'\'def _merge_stream_group_moments(\n    moments: _StreamGroupMoments, counts: np.ndarray, accumulators: list[Any]\n) -> None:\n    """Prepare independent existing-cell merges, then commit in stream order."""\n    if np.geterr() != {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}:\n        return\n    slots = [slot for slot in np.flatnonzero(moments.selected)\n             if accumulators[slot] is not None and accumulators[slot].count > 0]\n    if len(slots) < _STREAM_MERGE_BATCH_MIN_GROUPS:\n        return\n    states = [accumulators[slot] for slot in slots]\n    additions = counts[slots]\n    old_counts = [state.count for state in states]\n    # Float vector division matches Python integer division only while each\n    # integer operand is exactly representable. Larger states stay scalar.\n    if any(type(count) is not int or count + int(addition) > _STREAM_MERGE_EXACT_COUNT\n           or count * int(addition) > _STREAM_MERGE_EXACT_COUNT\n           for count, addition in zip(old_counts, additions, strict=True)):\n        return\n    if any(type(array) is not np.ndarray or array.dtype != np.float64 or array.shape != shape\n           or not array.flags.writeable\n           for state in states\n           for array, shape in [(state.mean, (3,)), (state.m2, (3, 3)),\n                                (state.minimum, (3,)), (state.maximum, (3,))]):\n        return\n    means = np.stack([state.mean for state in states])\n    m2 = np.stack([state.m2 for state in states])\n    minima = np.stack([state.minimum for state in states])\n    maxima = np.stack([state.maximum for state in states])\n    # Headroom covers differences, outer products, count factors and addition.\n    for array, limit in [(means, 1e140), (minima, 1e140), (maxima, 1e140), (m2, 1e300),\n                         (moments.mean[slots], 1e140), (moments.m2[slots], 1e300)]:\n        if not (np.max(array) <= limit and np.min(array) >= -limit):\n            return\n    old = np.asarray(old_counts, dtype=np.int64)\n    totals = old + additions\n    delta = moments.mean[slots] - means\n    # Preserve scalar parentheses and individual ufunc operations.\n    moments.m2[slots] = m2 + (moments.m2[slots] + delta[:, :, None] * delta[:, None, :]\n                             * ((old * additions) / totals)[:, None, None])\n    moments.mean[slots] = means + delta * (additions / totals)[:, None]\n    moments.minimum[slots] = np.minimum(minima, moments.minimum[slots])\n    moments.maximum[slots] = np.maximum(maxima, moments.maximum[slots])\n    moments.merged = np.zeros(len(counts), dtype=bool)\n    moments.merged[slots] = True\n    moments.totals = np.zeros(len(counts), dtype=np.int64)\n    moments.totals[slots] = totals\n\n\n\'\'\'\nreplace_once(\'def _update_stream_voxel(\\n\', helper + \'def _update_stream_voxel(\\n\')\nreplace_once(\'    mean, m2 = moments.mean[slot], moments.m2[slot]\', \'\'\'    if moments.merged is not None and moments.merged[slot]:\n        accumulator.m2[...] = moments.m2[slot]\n        accumulator.mean[...] = moments.mean[slot]\n        accumulator.count = int(moments.totals[slot])\n        accumulator.minimum = moments.minimum[slot].copy()\n        accumulator.maximum = moments.maximum[slot].copy()\n        return\n    mean, m2 = moments.mean[slot], moments.m2[slot]\'\'\')\nreplace_once(\'        moments = None\\n        group_end = 0\', \'        moments = None\\n        window_keys = window_accumulators = None\\n        group_end = 0\')\nreplace_once(\'\'\'                moments = _stream_group_moments(ordered_points, starts[group_index:stop], counts[group_index:stop])\'\'\', \'\'\'                moments = _stream_group_moments(ordered_points, starts[group_index:stop], counts[group_index:stop])\n                window_keys = window_accumulators = None\n                if moments is not None and accumulators:\n                    window_keys = [tuple(int(value) for value in row) for row in unique[group_index:stop]]\n                    window_accumulators = [accumulators.get(key) for key in window_keys]\n                    _merge_stream_group_moments(moments, counts[group_index:stop], window_accumulators)\'\'\')\nreplace_once(\'\'\'            key_values = tuple(int(value) for value in raw_key)\n            if len(key_values) != 3:\'\'\', \'\'\'            key_values = window_keys[slot] if window_keys is not None else tuple(int(value) for value in raw_key)\n            if len(key_values) != 3:\'\'\')\nreplace_once(\'            accumulator = accumulators.get(key)\\n\', \'            accumulator = window_accumulators[slot] if window_accumulators is not None else accumulators.get(key)\\n\')\nreplace_once(\'del ordered_points, order, indices, unique, inverse, counts, starts, moments\',\n             \'del ordered_points, order, indices, unique, inverse, counts, starts, moments, window_keys, window_accumulators\')\n\nif variant != \'checked\':\n    start = candidate.index(\'    if any(type(array) is not np.ndarray\', candidate.index(\'def _merge_stream_group_moments\'))\n    end = candidate.index(\'    means = np.stack\', start)\n    candidate = candidate[:start] + \'    # States are private accumulators created by this builder: their owned\\n    # f64 shapes and initialized bounds follow from the accumulator methods.\\n\' + candidate[end:]\n    start = candidate.index(\'    for array, limit in [(means, 1e140)\', candidate.index(\'def _merge_stream_group_moments\'))\n    end = candidate.index(\'    old = np.asarray\', start)\n    candidate = candidate[:start] + \'    for array, limit in [(means, 1e150), (m2, 1e304)]:\\n        if not (np.max(array) <= limit and np.min(array) >= -limit):\\n            return\\n\' + candidate[end:]\nif variant == \'compact\':\n    for field in [\'mean\', \'m2\', \'minimum\', \'maximum\']:\n        replace_once(f\'np.stack([state.{field} for state in states])\',\n                     f\'np.asarray([state.{field} for state in states], dtype=np.float64)\')\n'


def candidate_source(original, variant):
    namespace = dict(original=original, variant=variant)
    exec(compile(_GENERATOR, '<merge-candidate-generator>', 'exec'), namespace)
    return namespace['candidate']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--coordinate-report', type=Path, required=True)
    parser.add_argument('--variant', choices=['checked', 'trusted', 'compact'], required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    options = [value.split('=')[0] for value in sys.argv[1:] if value.startswith('--')]
    require(len(options) == len(set(options)), 'duplicate options')
    args = parser.parse_args()
    require(1 <= args.repeats <= 100, 'invalid repeats')
    require(not args.work_dir.exists() and not args.output.exists(), 'output/work directory must be new')
    root = args.source_root.resolve()
    data = args.baseline.read_bytes()
    io_path = root / 'cloudanalyzer/ca/io.py'
    require(hashlib.sha256(data).hexdigest() == '6216e8fb8ea8faa441a6a1fd126bff0740e3dbe887c8232d22199cd90ee2f603', 'baseline differs')
    require(sha(io_path) == '10c1c542a4ad4a70052d947b48c499f2dbf5b9d6d2a49e900dfe40494f61ebff', 'IO source differs')
    original = data.decode('utf-8').replace('\r\n', '\n')
    candidate = candidate_source(original, args.variant)
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() or 1)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / 'cloudanalyzer'))
    import numpy as np
    import scipy
    import scipy.spatial._ckdtree as tree_binary
    import open3d as o3d
    import ca.io as live_io

    previous = json.loads(args.coordinate_report.read_text(encoding='utf-8'))
    require(previous['comparison'] == 'map-stream-finite' and previous['chunkSize'] == 100_000
            and previous['structureVoxelSize'] == 3., 'coordinate report protocol differs')
    for key, entry in previous['inputs'].items():
        require(sha(entry['path']) == entry['sha256'], 'input differs: ' + key)
    arrays = []
    for key in ['compared', 'reference']:
        entry = previous['coordinateSnapshots'][key]
        require(type(entry['points']) is int and entry['points'] > 0, 'invalid snapshot count')
        require(Path(entry['path']).stat().st_size == 24 * entry['points']
                and sha(entry['path']) == entry['sha256'], 'snapshot differs: ' + key)
        arrays.append(np.memmap(entry['path'], mode='r', dtype='<f8', shape=(entry['points'], 3)))
    baseline_path = args.baseline
    candidate_path = args.work_dir / 'candidate.py'
    args.work_dir.mkdir(parents=True, exist_ok=False)
    with candidate_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(candidate)
    def module(name, source, path):
        result = types.ModuleType(name)
        sys.modules[name] = result
        exec(compile(source, str(path), 'exec'), result.__dict__)
        return result

    modules = dict(scalar=module('probe_stream_scalar_merge', original, baseline_path),
                   batch=module('probe_stream_batch_merge', candidate, candidate_path))

    def call(policy, arrays, capture=False):
        current = modules[policy]
        cls = live_io.PointAccumulator
        builder = current.build_voxel_gaussians_from_chunks
        captured = []
        def traced_builder(chunks, size):
            handler, finalizer = current._update_stream_voxel, cls.finalize
            digest = hashlib.sha256()
            summaries = []
            counts = dict(groups=0, points=0, mergedGroups=0, mergedPoints=0)
            def traced_handler(accumulator, points, moments, slot, **kwargs):
                merged = bool(moments is not None and getattr(moments, 'merged', None) is not None and moments.merged[slot])
                counts['groups'] += 1
                counts['points'] += len(points)
                counts['mergedGroups'] += int(merged)
                counts['mergedPoints'] += len(points) * int(merged)
                digest.update(struct.pack('<Q', len(points)))
                digest.update(np.asarray(points, dtype='<f8').tobytes())
                return handler(accumulator, points, moments, slot, **kwargs)
            def traced_finalize(self):
                summary = finalizer(self)
                summaries.append(struct.pack('<q18d', summary['count'], *summary['mean'], *summary['covariance'].ravel(),
                                             *summary['minimum'], *summary['maximum']))
                return summary
            current._update_stream_voxel, cls.finalize = traced_handler, traced_finalize
            try:
                voxels = builder(chunks, size)
            finally:
                current._update_stream_voxel, cls.finalize = handler, finalizer
            packed = b''.join(struct.pack('<3q', *key) + value for key, value in zip(voxels, summaries, strict=True))
            captured.append(dict(**counts, stateBytes=packed, groupsSha256=digest.hexdigest()))
            return voxels
        if capture:
            current.build_voxel_gaussians_from_chunks = traced_builder
        start = time.perf_counter_ns()
        try:
            result = current.evaluate_map_streaming(
                (arrays[0][first:first + 100_000] for first in range(0, len(arrays[0]), 100_000)),
                (arrays[1][first:first + 100_000] for first in range(0, len(arrays[1]), 100_000)))
        finally:
            current.build_voxel_gaussians_from_chunks = builder
        return asdict(result), (time.perf_counter_ns() - start) / 1e6, captured

    expected, _, before = call('scalar', arrays, True)
    observed, _, after = call('batch', arrays, True)
    require(expected == observed == previous['parity']['finite']['report'], 'complete reports differ')
    for a, b in zip(before, after, strict=True):
        require(a['stateBytes'] == b['stateBytes'] and a['groupsSha256'] == b['groupsSha256'], 'complete state/group bytes differ')
    print(json.dumps(dict(stage='parity_complete')), flush=True)
    call('scalar', arrays)
    call('batch', arrays)
    trials = []
    for index in range(args.repeats):
        order = ['scalar', 'batch'] if index % 2 == 0 else ['batch', 'scalar']
        timings = {}
        for policy in order:
            report, timings[policy], _ = call(policy, arrays)
            require(report == expected, 'timed report differs')
        trials.append(dict(order=order, millis=timings))
        print(json.dumps(dict(trial=index + 1, millis=timings)), flush=True)
    a = [trial['millis']['scalar'] for trial in trials]
    b = [trial['millis']['batch'] for trial in trials]
    for capture in [before, after]:
        for item in capture:
            item['stateSha256'] = hashlib.sha256(item.pop('stateBytes')).hexdigest()
    summary = dict(pairedShorteningPercent=statistics.median(100 * (1 - y / x) for x, y in zip(a, b)),
                   fasterTrials=sum(x > y for x, y in zip(a, b)))
    require(args.baseline.read_bytes() == data and sha(io_path) == '10c1c542a4ad4a70052d947b48c499f2dbf5b9d6d2a49e900dfe40494f61ebff', 'sources changed')
    for entry in [*previous['inputs'].values(), *previous['coordinateSnapshots'].values()]:
        require(sha(entry['path']) == entry['sha256'], 'input changed')
    result = dict(schemaVersion=1, comparison='map-stream-merge-candidate', adopted=False,
                  createdAt=datetime.now(timezone.utc).isoformat(), variant=args.variant,
                  baselineSha256=hashlib.sha256(data).hexdigest(), candidateSha256=sha(candidate_path),
                  ioSha256=sha(io_path), compiledTreeSha256=sha(tree_binary.__file__),
                  coordinateReport=dict(path=str(args.coordinate_report), sha256=sha(args.coordinate_report)),
                  inputs=previous['inputs'], inputManifest=previous['inputManifest'],
                  coordinateSnapshots=previous['coordinateSnapshots'],
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                   open3d=o3d.__version__, platform=platform.system(), logicalCpus=os.cpu_count(),
                                   ompNumThreads=os.environ['OMP_NUM_THREADS'], openblasNumThreads=os.environ['OPENBLAS_NUM_THREADS']),
                  chunkSize=100_000, structureVoxelSize=3., repeats=args.repeats, warmupRounds=1,
                  captures=dict(scalar=before, batch=after), report=expected, trials=trials, summary=summary,
                  note='Complete streaming AWD/SCS; unchanged fixed real-source XYZ snapshots and chunk boundaries. '
                       'Parity tracing, snapshot mapping/hashing, loading and report writes are excluded. '
                       'One excluded warm-up per policy precedes alternating trials; all trials are retained. '
                       'These isolated candidates have not been adopted into the runtime.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), summary=summary)), flush=True)


if __name__ == '__main__':
    main()
