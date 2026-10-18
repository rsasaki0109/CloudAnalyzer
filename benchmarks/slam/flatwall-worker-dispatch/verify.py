#!/usr/bin/env python3
"""Verify original real-data replay bits and recompute worker-dispatch timings."""
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
    protocol = read(folder/'protocol.json')
    if (protocol['schema'] != 'cloudanalyzer.gated_workers.v1'
            or protocol['gate'] != 'needs_alignment && (registration.target_balance || registration.source.len() > 256) && iterations > 0 && termination.is_none() && current_thread_index().is_none()'
            or protocol['replay_process_order'] != ['baseline', 'candidate', 'candidate', 'baseline']
            or protocol['replay_threads'] != 4 or protocol['replay_rounds'] != dict(warmup=1, measured=3)
            or protocol['defaults_changed'] is not False or protocol['installed_extension_replaced'] is not False):
        raise ValueError('worker comparison protocol differs')
    footprints = read(folder/'source-footprints.json')
    builds, receipts = {}, {}
    for implementation in ['baseline', 'candidate']:
        builds[implementation] = read(folder/f'build-{implementation}-inputs.json')
        receipts[implementation] = read(folder/f'build-{implementation}-receipt.json')
        if receipts[implementation]['inputs_sha256'] != sha(folder/f'build-{implementation}-inputs.json'):
            raise ValueError('native build receipt differs')
        if builds[implementation]['protocol_sha256'] != sha(folder/'protocol.json'):
            raise ValueError('native build protocol differs')
        inventory = builds[implementation]['sources']
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
    if changes != ['rust/crates/ca-core/src/odometry.rs']:
        raise ValueError('native implementation change inventory differs')
    promotion = read(folder/'source-promotion.json')
    old_comment = ('    // Balanced search also splits within a chunk. Enter the pool once for an\n'
                   '    // remaining ICP iterations once repeated work is needed. Keep an existing\n')
    new_comment = ('    // Balanced search also splits within a chunk. Enter the pool once for the\n'
                   '    // remaining iterations when repeated work is needed. Keep an existing\n')
    executed = (root/promotion['executed_source']).read_bytes()
    promoted = (root/promotion['promoted_source']).read_bytes()
    if (promotion['path'] != changes[0] or promotion['occurrences'] != 1
            or promotion['old_comment'] != old_comment or promotion['new_comment'] != new_comment
            or footprints['candidate'][changes[0]] != promotion['executed_source']
            or sha(root/promotion['executed_source']) != promotion['executed_sha256']
            or sha(root/promotion['promoted_source']) != promotion['promoted_sha256']
            or executed.count(old_comment.encode()) != 1
            or executed.replace(old_comment.encode(), new_comment.encode()) != promoted):
        raise ValueError('source comment promotion differs')
    reused_inputs = read(folder/'baseline-original-inputs.json')
    reused_receipt = read(folder/'baseline-original-receipt.json')
    if (receipts['baseline']['reused_preserved_executables'] is not True
            or receipts['baseline']['original_inputs_sha256'] != sha(folder/'baseline-original-inputs.json')
            or receipts['baseline']['original_receipt_sha256'] != sha(folder/'baseline-original-receipt.json')
            or reused_receipt['inputs_sha256'] != sha(folder/'baseline-original-inputs.json')
            or reused_receipt['artifacts'] != receipts['baseline']['artifacts']):
        raise ValueError('preserved baseline reuse differs')
    if (reused_inputs['commands'] != builds['baseline']['commands']
            or reused_inputs['versions'] != builds['baseline']['versions']
            or set(reused_inputs['sources']) != set(builds['baseline']['sources'])):
        raise ValueError('preserved baseline build conditions differ')
    for name, item in reused_inputs['sources'].items():
        if item['normalized_sha256'] != builds['baseline']['sources'][name]['normalized_sha256']:
            raise ValueError('preserved baseline source differs')
    native_validation = read(folder/'native-validation/receipt.json')
    validation_call = read(folder/'native-validation/invocation.json')
    if (native_validation['invocation_sha256'] != sha(folder/'native-validation/invocation.json')
            or validation_call['controller_sha256'] != sha(folder/'native-validation/controller.py')
            or validation_call['source_sha256'] != builds['candidate']['sources']['rust/crates/ca-core/src/odometry.rs']['sha256']):
        raise ValueError('native validation source binding differs')
    for name, digest in native_validation['files_sha256'].items():
        if sha(folder/'native-validation'/name) != digest:
            raise ValueError('native validation bytes differ')
    if set(native_validation['checks']) != {'baseline-odometry', 'candidate-odometry', 'clippy-serial', 'clippy-parallel', 'rustfmt'}:
        raise ValueError('native validation inventory differs')
    for name, check in native_validation['checks'].items():
        if check['returncode'] != 0 or check['log_sha256'] != sha(folder/'native-validation'/(name+'.log')):
            raise ValueError('native validation result differs')
    for implementation, count in [('baseline', 24), ('candidate', 27)]:
        if f'{count} passed; 0 failed; 1 ignored'.encode() not in (folder/'native-validation'/(implementation+'-odometry.log')).read_bytes():
            raise ValueError('native odometry test inventory differs')
    analysis = read(folder/'analysis-receipt.json')
    if (analysis['controller_sha256'] != sha(folder/'controller.py')
            or analysis['comparison_sha256'] != sha(folder/'analysis-source.py')
            or analysis['time_receipt_sha256'] != sha(folder/'time/receipt.json')
            or analysis['summary_sha256'] != sha(folder/'summary.json')):
        raise ValueError('executed analysis binding differs')
    if ((folder/'analysis-source.py').read_bytes().replace(b'\r\n', b'\n')
            != (root/'scripts/flatwall_count_comparison.py').read_bytes().replace(b'\r\n', b'\n')):
        raise ValueError('executed and verifying analysis sources differ')
    spec = importlib.util.spec_from_file_location('worker_dispatch_comparison', root/'scripts/flatwall_count_comparison.py')
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    invocation = read(folder/'time/invocation.json')
    receipt = read(folder/'time/receipt.json')
    if (receipt['invocation_sha256'] != sha(folder/'time/invocation.json')
            or invocation['protocol_sha256'] != sha(folder/'protocol.json')
            or invocation['controller_sha256'] != sha(folder/'controller.py')
            or receipt['input_hashes_unchanged'] is not True):
        raise ValueError('native execution receipt differs')
    catalog = read(folder/'catalog.json')
    if [(r['index'], r['implementation']) for r in catalog] != list(enumerate(protocol['replay_process_order'])):
        raise ValueError('native process order differs')
    records, reports = [], []
    originals = comparison.original_cases(root)
    for item in catalog:
        call = read(folder/item['invocation'])
        implementation = item['implementation']
        binary = receipts[implementation]['artifacts']['audit_odometry_steps.exe']['sha256']
        if ((call['index'], call['implementation'], call['threads']) != (item['index'], implementation, 4)
                or call['binary_sha256'] != binary or call['environment'] != dict(RAYON_NUM_THREADS='4')):
            raise ValueError('native invocation identity differs')
        stem = f"run-{item['index']:02}-{implementation}-threads-4"
        for name in [stem+'.log', stem+'-invocation.json']:
            if sha(folder/'time'/name) != receipt['output_files_sha256'][name]:
                raise ValueError('original native execution log differs')
        raw = gzip.decompress((folder/item['file']).read_bytes())
        digest = hashlib.sha256(raw).hexdigest()
        if digest != item['original_output_sha256'] or digest != receipt['output_files_sha256'][stem+'/runs.json']:
            raise ValueError('original native output bytes differ')
        run = json.loads(raw)
        reports.append(comparison.check_replay(run, originals, 'time'))
        records.append(dict(index=item['index'], implementation=implementation, data=run))
    for record in records[1:]:
        comparison.compare_replays(record['data'], records[0]['data'])
    summary = dict(schema='cloudanalyzer.gated_workers_summary.v1',
        registration=comparison.summarize_time(records, [r['name'] for r in originals]),
        time_verification=reports, original_pose_and_threshold_bits_match=True,
        replay_frames_checked=sum(r['frames'] for r in reports), defaults_changed=False,
        installed_extension_replaced=False, whole_application_speedup_established=False)
    equal(summary, read(folder/'summary.json'))
    return dict(real_cases=16, replay_frames_checked=summary['replay_frames_checked'],
        native_source_files_per_build=84, implementation_changes=changes,
        original_pose_and_threshold_bits_match=True, defaults_changed=False,
        installed_extension_replaced=False, whole_application_speedup_established=False)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
