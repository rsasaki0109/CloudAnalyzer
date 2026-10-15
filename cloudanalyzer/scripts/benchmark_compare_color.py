"""Compare complete numeric-only comparisons with and without unused coloring.

Each call loads full real-source clouds, runs actual native nearest distances,
computes statistics/thresholds and returns a report. Registration and snapshot
output are disabled. No solver or distance results are replaced for timing.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
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
    for option in ['source-root', 'reference', 'compared', 'input-manifest', 'baseline',
                   'build-index', 'distance-record', 'work-dir', 'output']:
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--baseline-sha256', required=True)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=10)
    options = [value.split('=')[0] for value in sys.argv[1:] if value.startswith('--')]
    require(len(options) == len(set(options)), 'duplicate options')
    args = parser.parse_args()
    require(1 <= args.threads <= 64 and 1 <= args.repeats <= 100, 'invalid threads/repeats')
    require(not args.output.exists() and not args.work_dir.exists(), 'output/work directory must be new')
    require(sha(args.baseline) == args.baseline_sha256, 'compare baseline differs')
    os.environ['RAYON_NUM_THREADS'] = os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['CA_DISABLE_RUST_CORE'] = '0'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str((args.source_root / 'cloudanalyzer').resolve()))
    import numpy as np
    import open3d as o3d
    import cloudanalyzer_core._core as core
    import ca.compare as current
    logging.getLogger('ca').setLevel(logging.ERROR)
    spec = importlib.util.spec_from_file_location('compare_always_colors_baseline', args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    modules = dict(always=baseline, requested=current)
    manifest = json.loads(args.input_manifest.read_text(encoding='utf-8'))
    prior = json.loads(args.distance_record.read_text(encoding='utf-8'))
    require(prior['comparison'] == 'scan-match-distance-reuse', 'unexpected distance record')
    require(prior['inputManifest'] == manifest, 'distance manifest differs')
    for key in ['reference', 'compared']:
        require(sha(getattr(args, key)) == manifest['inputs'][key]['sha256'] == prior['inputs'][key]['sha256'], key + ' differs')
    original = (args.source_root / 'web' / manifest['source']['path']).resolve()
    require(sha(original) == manifest['source']['sha256'], 'original source differs')
    snapshot = prior['registration']['snapshot']
    require(sha(snapshot['path']) == snapshot['sha256'], 'coordinate snapshot differs')
    data = np.load(snapshot['path'])
    for key, path in [('initial', args.compared), ('reference', args.reference)]:
        cloud = current.load_point_cloud(str(path))
        require(np.asarray(cloud.points).tobytes() == data[key].tobytes(), 'original coordinates differ')
        require(not cloud.has_colors() and not cloud.has_normals() and not cloud.has_covariances(), 'unexpected input attributes')
    del cloud
    build = json.loads(args.build_index.read_text(encoding='utf-8'))
    extension = info(core.__file__)
    require(extension['sha256'] == build['extension']['sha256'], 'native extension differs')
    for path, expected in build['sourceHashes'].items():
        require(sha(args.source_root / path) == expected, 'native source differs: ' + path)
    files = ['cloudanalyzer/ca/compare.py', 'cloudanalyzer/ca/io.py', 'cloudanalyzer/ca/metrics.py',
             'cloudanalyzer/ca/registration.py', 'cloudanalyzer/ca/visualization.py', 'cloudanalyzer/ca/report.py',
             'cloudanalyzer/ca/log.py', 'cloudanalyzer/ca/_rust.py', 'cloudanalyzer/pyproject.toml',
             'cloudanalyzer/tests/test_compare_color_consumers.py', 'cloudanalyzer/tests/test_compare.py',
             'cloudanalyzer/tests/test_diff.py', 'cloudanalyzer/tests/test_metrics.py',
             'cloudanalyzer/tests/test_visualization.py', 'cloudanalyzer/tests/test_cli.py',
             'cloudanalyzer/scripts/benchmark_compare_color.py']
    hashes = {path: sha(args.source_root / path) for path in files}
    inputs = {key: info(getattr(args, key)) for key in ['reference', 'compared', 'input_manifest',
                                                      'baseline', 'build_index', 'distance_record']}
    inputs.update(originalSource=info(original), coordinateSnapshot=info(snapshot['path']))
    normal_library = info(o3d.cpu.pybind.__file__)
    args.work_dir.mkdir(parents=True, exist_ok=False)

    def call(policy, capture=False, files=False):
        module = modules[policy]
        query, color = module.compute_nn_distance, module.colorize
        distances, colors = [], []
        def traced_query(source, target):
            require(np.asarray(source.points).tobytes() == data['initial'].tobytes(), 'query source differs')
            require(np.asarray(target.points).tobytes() == data['reference'].tobytes(), 'query target differs')
            values = query(source, target)
            distances.append(values.copy())
            return values
        def traced_color(cloud, values):
            result = color(cloud, values)
            colors.append(np.asarray(cloud.colors).copy())
            return result
        if capture:
            module.compute_nn_distance, module.colorize = traced_query, traced_color
        kwargs = dict(method=None, threshold=.5)
        if files:
            destination = args.work_dir / ('files-' + policy)
            require(not destination.exists(), 'report directory must be new')
            kwargs.update(json_path=str(destination / 'report.json'), report_path=str(destination / 'report.md'))
        start = time.perf_counter_ns()
        try:
            report = module.run_compare(str(args.compared), str(args.reference), **kwargs)
        finally:
            module.compute_nn_distance, module.colorize = query, color
        elapsed = (time.perf_counter_ns() - start) / 1e6
        saved = {}
        if capture:
            require(len(distances) == 1 and len(colors) == int(policy == 'always'), 'query/color call count differs')
            for name, arrays in [('distance', distances), ('colors', colors)]:
                for index, values in enumerate(arrays):
                    path = args.work_dir / f'{policy}-{name}-{index}.bin'
                    with path.open('xb') as stream:
                        stream.write(values.astype('<f8', copy=False).tobytes())
                    saved[name] = info(path)
        output_files = {name: info(destination / ('report.' + name)) for name in ['json', 'md']} if files else {}
        return dict(millis=elapsed, report=report, queryCalls=len(distances), colorCalls=len(colors),
                    captured=saved, outputFiles=output_files)

    policies = ['always', 'requested']
    parity = {policy: call(policy, True) for policy in policies}
    expected = parity['always']['report']
    require(parity['requested']['report'] == expected, 'complete reports differ')
    oracle = prior['firstUse']['fresh']['capturedDistances'][0]
    require(all(value['captured']['distance']['sha256'] == oracle['sha256'] for value in parity.values()), 'full distance bytes differ')
    require(parity['always']['captured']['colors']['bytes'] == 24 * manifest['inputs']['compared']['points'], 'color payload differs')
    output_parity = {policy: call(policy, files=True) for policy in policies}
    require(all(item['report'] == expected for item in output_parity.values()), 'written reports differ')
    for name in ['json', 'md']:
        require(output_parity['always']['outputFiles'][name]['sha256'] == output_parity['requested']['outputFiles'][name]['sha256'], 'report file bytes differ')
    print(json.dumps(dict(stage='parity_complete')), flush=True)
    first = {policy: call(policy) for policy in policies}
    require(all(item['report'] == expected for item in first.values()), 'first reports differ')
    def pair(index):
        order = policies if index % 2 == 0 else list(reversed(policies))
        observed = {policy: call(policy) for policy in order}
        require(all(item['report'] == expected for item in observed.values()), 'timed report differs')
        return dict(order=order, millis={policy: item['millis'] for policy, item in observed.items()})
    warmups = [pair(index) for index in range(2)]
    trials = []
    for index in range(args.repeats):
        trial = pair(index + 2)
        trials.append(trial)
        print(json.dumps(dict(stage='trial_complete', trial=index+1, millis=trial['millis'])), flush=True)
    a, b = [item['millis']['always'] for item in trials], [item['millis']['requested'] for item in trials]
    summary = dict(alwaysMedianMillis=statistics.median(a), requestedMedianMillis=statistics.median(b),
        pairedShorteningPercent=statistics.median(100*(1-y/x) for x, y in zip(a,b)),
        pairedSavedMedianMillis=statistics.median(x-y for x, y in zip(a,b)), fasterTrials=sum(x > y for x, y in zip(a,b)))
    for path, expected_hash in {**hashes, **build['sourceHashes']}.items():
        require(sha(args.source_root / path) == expected_hash, 'source changed: ' + path)
    for item in [*inputs.values(), extension, normal_library]:
        require(sha(item['path']) == item['sha256'], 'input/build changed')
    report = dict(schemaVersion=1, comparison='compare-requested-colors', createdAt=datetime.now(timezone.utc).isoformat(),
        sourceHashes=hashes, nativeSourceHashes=build['sourceHashes'], inputs=inputs, inputManifest=manifest,
        extension=extension, normalLibrary=normal_library,
        environment=dict(python=platform.python_version(), numpy=np.__version__, open3d=o3d.__version__,
            platform=platform.system(), logicalCpus=os.cpu_count(), rayonNumThreads=args.threads,
            ompNumThreads=args.threads, openblasNumThreads=1),
        repeats=args.repeats, warmupRounds=2, parity=parity, outputParity=output_parity,
        firstUse=first, warmups=warmups, trials=trials, summary=summary,
        note='Complete run_compare function with method=None, threshold=.5, no snapshot or report-file output during timing. '
             'Actual full unchanged-coordinate real-source inputs are loaded through the existing bulk-capable I/O path; '
             'native NN construction/search, statistics, report creation and cleanup execute normally. No registration, solver '
             'replay, replaced NN arrays, generated points, perturbations or further thinning. The old path also constructs '
             'unused colors; the candidate skips them. Separate parity calls verify complete JSON/Markdown file bytes. '
             'Tracing/capture writes, file-output parity, imports/hashes and CLI serialization are excluded from summaries. '
             'A traced parity pair and excluded first-use pair precede two warm-up pairs and ten default alternating timed '
             'pairs. Logging disabled for both; all observations retained. Warmed pages; no cold-ingestion, peak RSS, '
             'registered/snapshot-producing comparison, whole CLI or GPU performance claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), summary=summary)), flush=True)


if __name__ == '__main__':
    main()
