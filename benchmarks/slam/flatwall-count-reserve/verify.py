#!/usr/bin/env python3
"""Verify real replay bits, fixed states, build bindings and elapsed-time cohorts."""
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


def production_change(baseline, candidate):
    old = "    let mut index: HashMap<[u64; 3], usize, BuildHasherDefault<VoxelHasher>> = HashMap::default();\n    let mut frequencies = Vec::<usize>::new();"
    new = "    // There can be at most one unique coordinate per item. Reserve that upper\n    // bound once so counting does not repeatedly grow and rehash its table.\n    let mut index: HashMap<[u64; 3], usize, BuildHasherDefault<VoxelHasher>> =\n        HashMap::with_capacity_and_hasher(items.len(), BuildHasherDefault::default());\n    let mut frequencies = Vec::<usize>::with_capacity(items.len());"
    if baseline.count(old) != 1 or candidate.count(new) != 1 or candidate.replace(new, old) != baseline:
        raise ValueError('production change differs from capacity reservations')


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
    protocol = read(folder/'protocol.json')
    order = [dict(implementation=i, threads=t) for t in [1, 4]
        for i in ['baseline', 'candidate', 'candidate', 'baseline']]
    if (protocol['schema'] != 'cloudanalyzer.target_count_reserve.v1'
            or protocol['base'] != '7a95da1c16d6770fb5feee47bee3bc85c304ce15'
            or protocol['replay_process_order'] != ['baseline', 'candidate', 'candidate', 'baseline']
            or protocol['fixed_process_order'] != order or protocol['replay_threads'] != 4
            or protocol['replay_rounds'] != dict(warmup=1, measured=3)
            or protocol['defaults_changed'] is not False or protocol['installed_extension_replaced'] is not False):
        raise ValueError('count-capacity comparison protocol differs')
    footprints = read(folder/'source-footprints.json')
    builds, receipts = {}, {}
    for implementation in ['baseline', 'candidate']:
        build = builds[implementation] = read(folder/f'build-{implementation}-inputs.json')
        receipt = receipts[implementation] = read(folder/f'build-{implementation}-receipt.json')
        if (receipt['inputs_sha256'] != sha(folder/f'build-{implementation}-inputs.json')
                or build['protocol_sha256'] != sha(folder/'protocol.json')
                or build['controller_sha256'] != sha(folder/'controller.py.raw')
                or receipt['sources_checked'] != 84 or receipt['installed_extension_replaced'] is not False):
            raise ValueError('native build binding differs')
        inventory = build['sources']
        if len(inventory) != 84 or set(inventory) != set(footprints[implementation]):
            raise ValueError('native source inventory differs')
        for name, item in inventory.items():
            data = (root/footprints[implementation][name]).read_bytes().replace(b'\r\n', b'\n')
            if hashlib.sha256(data).hexdigest() != item['normalized_sha256']:
                raise ValueError('executed native source differs: '+name)
    if (builds['baseline']['commands'] != builds['candidate']['commands']
            or builds['baseline']['versions'] != builds['candidate']['versions']):
        raise ValueError('native build conditions differ')
    changes = sorted(name for name, item in builds['baseline']['sources'].items()
        if item['normalized_sha256'] != builds['candidate']['sources'][name]['normalized_sha256'])
    if changes != ['rust/crates/ca-core/src/odometry_target_balance.rs']:
        raise ValueError('native implementation change inventory differs')
    original = read(folder/'baseline-original-inputs.json')
    original_receipt = read(folder/'baseline-original-receipt.json')
    reused = receipts['baseline']
    if (reused['reused_preserved_executables'] is not True
            or reused['original_inputs_sha256'] != sha(folder/'baseline-original-inputs.json')
            or reused['original_receipt_sha256'] != sha(folder/'baseline-original-receipt.json')
            or original_receipt['inputs_sha256'] != sha(folder/'baseline-original-inputs.json')
            or original_receipt['artifacts'] != reused['artifacts']
            or original['commands'] != builds['baseline']['commands']
            or original['versions'] != builds['baseline']['versions']
            or set(original['sources']) != set(builds['baseline']['sources'])):
        raise ValueError('preserved baseline binding differs')
    for name, item in original['sources'].items():
        if item['normalized_sha256'] != builds['baseline']['sources'][name]['normalized_sha256']:
            raise ValueError('preserved baseline source differs')
    sources = [(root/footprints[i][changes[0]]).read_text(encoding='utf-8') for i in ['baseline', 'candidate']]
    production_change(*sources)
    validation = read(folder/'native-validation/receipt.json')
    call = read(folder/'native-validation/invocation.json')
    if (validation['invocation_sha256'] != sha(folder/'native-validation/invocation.json')
            or call['controller_sha256'] != sha(folder/'native-validation/controller.py.raw')
            or call['source_sha256'] != builds['candidate']['sources'][changes[0]]['sha256']):
        raise ValueError('native validation binding differs')
    if set(validation['checks']) != {'baseline-odometry', 'candidate-odometry', 'clippy-serial', 'clippy-parallel', 'rustfmt'}:
        raise ValueError('native validation inventory differs')
    for name, digest in validation['files_sha256'].items():
        if sha(folder/'native-validation'/name) != digest:
            raise ValueError('native validation bytes differ')
    for name, check in validation['checks'].items():
        if check['returncode'] != 0 or check['log_sha256'] != sha(folder/'native-validation'/(name+'.log')):
            raise ValueError('native validation result differs')
    for implementation, count in [('baseline', 29), ('candidate', 29)]:
        if f'{count} passed; 0 failed; 1 ignored'.encode() not in (folder/'native-validation'/(implementation+'-odometry.log')).read_bytes():
            raise ValueError('native test inventory differs')
    spec = importlib.util.spec_from_file_location('real_count_capacity_comparison', root/'scripts/flatwall_count_comparison.py')
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    catalog = read(folder/'catalog.json')
    records = {}
    for mode in ['fixed', 'time']:
        receipt = read(folder/mode/'receipt.json')
        invocation = read(folder/mode/'invocation.json')
        if (receipt['invocation_sha256'] != sha(folder/mode/'invocation.json')
                or invocation['controller_sha256'] != sha(folder/'controller.py.raw')
                or invocation['protocol_sha256'] != sha(folder/'protocol.json')
                or receipt['input_hashes_unchanged'] is not True):
            raise ValueError('native execution receipt differs')
        wanted = order if mode == 'fixed' else [dict(implementation=i, threads=4) for i in protocol['replay_process_order']]
        if [(r['index'], r['implementation'], r['threads']) for r in catalog[mode]] != [
                (index, r['implementation'], r['threads']) for index, r in enumerate(wanted)]:
            raise ValueError('native process order differs')
        records[mode] = []
        for item in catalog[mode]:
            call = read(folder/item['invocation'])
            index, implementation, threads = item['index'], item['implementation'], item['threads']
            binary = 'odometry-stage-profile.exe' if mode == 'fixed' else 'audit_odometry_steps.exe'
            if ((call['index'], call['implementation'], call['threads']) != (index, implementation, threads)
                    or call['binary_sha256'] != receipts[implementation]['artifacts'][binary]['sha256']
                    or call['environment']['RAYON_NUM_THREADS'] != str(threads)):
                raise ValueError('native invocation identity differs')
            stem = f'run-{index:02}-{implementation}-threads-{threads}'
            for name in [stem+'.log', stem+'-invocation.json']:
                if sha(folder/mode/name) != receipt['output_files_sha256'][name]:
                    raise ValueError('original native execution log differs')
            raw = gzip.decompress((folder/item['file']).read_bytes())
            original = stem+'.json' if mode == 'fixed' else stem+'/runs.json'
            digest = hashlib.sha256(raw).hexdigest()
            if digest != item['original_output_sha256'] or digest != receipt['output_files_sha256'][original]:
                raise ValueError('original native output bytes differ')
            records[mode].append(dict(index=index, implementation=implementation, threads=threads, data=json.loads(raw)))
    fixed = dict(schema='cloudanalyzer.target_count_reserve_summary.v1',
        fixed=comparison.summarize_fixed(records['fixed'], read(root/'benchmarks/slam/flatwall-balance-stages/origins.json')),
        fixed_verification=comparison.check_fixed(records['fixed'], root), defaults_changed=False,
        installed_extension_replaced=False, whole_application_speedup_established=False)
    equal(fixed, read(folder/'fixed-summary.json'))
    originals = comparison.original_cases(root)
    reports = [comparison.check_replay(r['data'], originals, 'time') for r in records['time']]
    for row in records['time'][1:]:
        comparison.compare_replays(row['data'], records['time'][0]['data'])
    timing = dict(schema='cloudanalyzer.target_count_reserve_summary.v1',
        registration=comparison.summarize_time(records['time'], [r['name'] for r in originals]),
        time_verification=reports, replay_frames_checked=sum(r['frames'] for r in reports),
        original_pose_and_threshold_bits_match=True, defaults_changed=False,
        installed_extension_replaced=False, whole_application_speedup_established=False)
    equal(timing, read(folder/'time-summary.json'))
    for mode, controller in [('time', 'controller.py.raw'), ('fixed', 'controller.py.raw')]:
        receipt = read(folder/(mode+'-analysis-receipt.json'))
        if (receipt['summary_sha256'] != sha(folder/(mode+'-summary.json'))
                or receipt['run_receipt_sha256'] != sha(folder/mode/'receipt.json')
                or receipt['controller_sha256'] != sha(folder/controller)
                or receipt['analysis_source_sha256'] != sha(folder/'analysis-source.py.raw')):
            raise ValueError('analysis binding differs')
    return dict(real_cases=16, real_fixed_states=64, replay_frames_checked=timing['replay_frames_checked'],
        fixed_validation_states=fixed['fixed_verification']['validation_states'],
        native_source_files_per_build=84, implementation_changes=changes,
        original_pose_and_threshold_bits_match=True, original_fixed_pairs_and_matrices_checked=True,
        defaults_changed=False, installed_extension_replaced=False, whole_application_speedup_established=False)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
