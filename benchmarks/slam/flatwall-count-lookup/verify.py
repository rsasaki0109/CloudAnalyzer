#!/usr/bin/env python3
"""Verify every retained real-data count comparison and recompute its summaries."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(actual, expected):
    if type(actual) is not type(expected):
        raise ValueError('derived summary type differs')
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError('derived summary fields differ')
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('derived summary inventory differs')
        for a, b in zip(actual, expected):
            equal(a, b)
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-12):
            raise ValueError('derived summary differs')
    elif actual != expected:
        raise ValueError('derived summary differs')


def verify(folder=None):
    folder = Path(folder) if folder else Path(__file__).resolve().parent
    root = folder.parents[2]
    manifest = read(folder/'manifest.json')
    for name, digest in manifest['files_sha256'].items():
        if sha(folder/name) != digest:
            raise ValueError('packet hash differs: '+name)
    for name, digest in manifest['dependencies'].items():
        if sha(root/name) != digest:
            raise ValueError('dependency hash differs: '+name)
    for name, digest in manifest['normalized_checkout_sources_sha256'].items():
        if hashlib.sha256((root/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError('analysis source differs: '+name)
    footprint = read(folder/'source-footprints.json')
    analysis = read(folder/'analysis-invocation.json')
    if analysis['analysis_source_sha256'] != sha(folder/'analysis-source.py'):
        raise ValueError('executed analysis source differs')
    if analysis['protocol_sha256'] != sha(folder/'protocol.json'):
        raise ValueError('executed analysis protocol differs')
    preflight = read(folder/'preflight-receipt.json')
    if preflight['files_sha256']['scripts/flatwall_count_comparison.py'] != sha(folder/'preflight-source.py'):
        raise ValueError('original preflight source differs')
    failed = read(folder/'prior-validation/invocation.json')
    if (failed['analysis_source_sha256'] != sha(folder/'prior-validation/source.py')
            or failed['controller_sha256'] != sha(folder/'prior-validation/controller.py')):
        raise ValueError('original validation source differs')
    builds = {}
    for implementation in ['baseline', 'candidate']:
        builds[implementation] = read(folder/f'build-{implementation}-inputs.json')
        receipt = read(folder/f'build-{implementation}-receipt.json')
        if receipt['inputs_sha256'] != sha(folder/f'build-{implementation}-inputs.json'):
            raise ValueError('native build receipt differs')
        if builds[implementation]['protocol_sha256'] != sha(folder/'protocol.json'):
            raise ValueError('native build protocol differs')
        sources = builds[implementation]['sources']
        if len(sources) != 84 or set(sources) != set(footprint[implementation]):
            raise ValueError('native source inventory differs')
        for name, item in sources.items():
            path = root/footprint[implementation][name]
            if hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != item['normalized_sha256']:
                raise ValueError('executed native source differs: '+name)
    if builds['baseline']['commands'] != builds['candidate']['commands'] or builds['baseline']['versions'] != builds['candidate']['versions']:
        raise ValueError('native build conditions differ')
    changed = sorted(name for name, item in builds['baseline']['sources'].items()
        if item['normalized_sha256'] != builds['candidate']['sources'][name]['normalized_sha256'])
    if changed != ['rust/crates/ca-core/src/odometry_profile.rs', 'rust/crates/ca-core/src/odometry_target_balance.rs']:
        raise ValueError('native implementation change inventory differs')
    protocol = read(folder/'protocol.json')
    if (protocol['schema'] != 'cloudanalyzer.single_lookup_counts.v1' or protocol['defaults_changed'] is not False
            or protocol['replay_process_order'] != ['baseline', 'candidate', 'candidate', 'baseline']
            or protocol['replay_threads'] != 4 or protocol['replay_rounds'] != dict(warmup=1, measured=3)):
        raise ValueError('comparison protocol differs')
    spec = importlib.util.spec_from_file_location('real_count_comparison', root/'scripts/flatwall_count_comparison.py')
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    catalog = read(folder/'catalog.json')
    records = {}
    for mode in ['fixed', 'trace', 'time']:
        records[mode] = []
        receipt = read(folder/mode/'receipt.json')
        if receipt['invocation_sha256'] != sha(folder/mode/'invocation.json'):
            raise ValueError('native execution receipt differs')
        for name, digest in receipt['output_files_sha256'].items():
            if name.endswith('.log') or name.endswith('-invocation.json'):
                if sha(folder/mode/name) != digest:
                    raise ValueError('original native execution log differs')
        for item in catalog[mode]:
            invocation = read(folder/item['invocation'])
            if (invocation['index'], invocation['implementation'], invocation['threads']) != (
                    item['index'], item['implementation'], item['threads']):
                raise ValueError('native invocation identity differs')
            binary = 'odometry-stage-profile.exe' if mode == 'fixed' else 'audit_odometry_steps.exe'
            build = read(folder/f"build-{item['implementation']}-receipt.json")
            if invocation['binary_sha256'] != build['artifacts'][binary]['sha256']:
                raise ValueError('native invoked binary differs')
            if invocation['environment']['RAYON_NUM_THREADS'] != str(item['threads']):
                raise ValueError('native requested thread configuration differs')
            stem = Path(item['file']).name.removesuffix('.json.gz')
            original = stem+'.json' if mode == 'fixed' else stem+'/runs.json'
            raw = gzip.decompress((folder/item['file']).read_bytes())
            digest = hashlib.sha256(raw).hexdigest()
            if digest != item['original_output_sha256'] or digest != receipt['output_files_sha256'][original]:
                raise ValueError('native original output bytes differ')
            for trace in item.get('traces', []):
                raw_trace = gzip.decompress((folder/trace['file']).read_bytes())
                digest = hashlib.sha256(raw_trace).hexdigest()
                original_trace = f"{stem}/{trace['case']}/scan-{trace['scan_index']:03}.json"
                if digest != trace['original_output_sha256'] or digest != receipt['output_files_sha256'][original_trace]:
                    raise ValueError('native original trace bytes differ')
            records[mode].append(dict(item, data=json.loads(raw)))
    if len(records['trace']) != 2 or [r['implementation'] for r in records['trace']] != ['baseline', 'candidate']:
        raise ValueError('trace build ordering differs')
    originals = comparison.original_cases(root)
    validation = dict(fixed=comparison.check_fixed(records['fixed'], root),
        trace_replay=[comparison.check_replay(r['data'], originals, 'trace') for r in records['trace']],
        defaults_changed=False, installed_extension_replaced=False)
    comparison.compare_replays(records['trace'][1]['data'], records['trace'][0]['data'])
    traces = []
    for item in records['trace']:
        traces.append({(r['case'], r['scan_index']): comparison.read(folder/r['file']) for r in item['traces']})
    validation['trace_systems'] = comparison.check_traces(*traces, root)
    equal(validation, read(folder/'verification.json'))
    time_reports = [comparison.check_replay(r['data'], originals, 'time') for r in records['time']]
    for row in records['time'][1:]:
        comparison.compare_replays(row['data'], records['time'][0]['data'])
    summary = dict(fixed=comparison.summarize_fixed(records['fixed'], comparison.read(root/'benchmarks/slam/flatwall-balance-stages/origins.json')),
        registration=comparison.summarize_time(records['time'], [r['name'] for r in originals]), time_verification=time_reports)
    equal(summary, read(folder/'summary.json'))
    return dict(real_cases=16, real_fixed_states=64, eligible_fixed_states=60, empty_source_controls=4,
        replay_frames_checked=sum(r['frames'] for r in time_reports)+sum(r['frames'] for r in validation['trace_replay']),
        capped_step_systems_checked=validation['trace_systems']['exact_capped_step_systems_and_updates'],
        native_source_files_per_build=84, implementation_changes=changed,
        original_pose_and_matrix_bits_match=True, defaults_changed=False, installed_extension_replaced=False,
        whole_application_speedup_established=False)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
