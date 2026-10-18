#!/usr/bin/env python3
"""Verify fixed real states, native stage samples and immutable source archives."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_independent(root):
    spec = importlib.util.spec_from_file_location('independent_stage_accounting',root/'scripts/flatwall_balance_stages.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_origins(root, origins, definitions):
    trial = root/'benchmarks/slam/flatwall-target-balance'
    dynamics = root/'benchmarks/slam/flatwall-balance-dynamics'
    promised = []
    for row in read(trial/'snapshot-catalog.json'):
        if row['public_sample']:
            path = trial/next(n for n in row['files'] if n.endswith('.npz'))
            config = read(path.with_suffix('.json'))
            promised.append((row['case'],row['scan_index'],'final',None,path.relative_to(root).as_posix(),config['alignment_eligible']))
    for row in read(dynamics/'summary.json')['traces']:
        for step in [0,499]:
            promised.append((row['case'],row['scan_index'],'step',step,(dynamics/row['file']).relative_to(root).as_posix(),True))
    if len(origins) != len(definitions) or len(origins) != 64 or len(promised) != 64:
        raise ValueError('all 64 predeclared states required')
    for i,(origin,definition,wanted) in enumerate(zip(origins,definitions,promised)):
        actual = tuple(origin.get(k) for k in ['case','scan_index','kind','step','original_npz','alignment_eligible'])
        if actual != wanted:
            raise ValueError('state selection or eligibility differs')
        case,scan,kind,step,_,_ = wanted
        name = case+f'/scan-{scan:03d}/'+('final' if kind=='final' else f'step-{step:04d}')
        if origin['name'] != name or definition['name'] != name or definition['file'] != f'state-{i:03d}.f64' or origin['binary_file'] != definition['file']:
            raise ValueError('state binary identity differs')


def verify(folder=None):
    folder = Path(folder) if folder else Path(__file__).resolve().parent
    root = folder.parents[2]
    manifest = read(folder/'manifest.json')
    for base,mapping in [(folder,manifest['files_sha256']),(root,manifest['dependencies'])]:
        for name,digest in mapping.items():
            if sha(base/name) != digest:
                raise ValueError('packet/dependency hash differs: '+name)
    footprint = read(folder/'source-footprints.json')
    build,supplement = read(folder/'native-build-inputs.json'),read(folder/'native-source-supplement.json')
    normalized = build['normalized_sources_sha256']|supplement['normalized_sources_sha256']
    if normalized != footprint['normalized_native_sources_sha256'] or set(normalized) != set(footprint['native_source_aliases']):
        raise ValueError('complete native source footprint differs')
    prefix = folder.relative_to(root).as_posix()+'/executed-source/'
    for name,digest in normalized.items():
        target = footprint['native_source_aliases'][name]
        if target != prefix+name or hashlib.sha256((root/target).read_bytes().replace(b'\r\n',b'\n')).hexdigest() != digest:
            raise ValueError('normalized executed source differs')
    for name,digest in footprint['normalized_checkout_sources_sha256'].items():
        if hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() != digest:
            raise ValueError('normalized diagnostic source differs')
    preservation = read(folder/'source-preservation.json')
    for original,item in preservation['public_history'].items():
        if sha(root/item['preserved_as']) != item['sha256']:
            raise ValueError('original public source was replaced: '+original)
    old = (root/preservation['public_history']['rust/crates/ca-core/src/odometry.rs']['preserved_as']).read_bytes()
    hook = b'\n// Real fixed-state timing is opt-in and excluded from library builds.\n#[cfg(all(test, feature = "parallel"))]\n#[path = "odometry_profile.rs"]\nmod profile;\n'
    promoted = old.replace(b'pub(crate) mod target_balance;\n',b'pub(crate) mod target_balance;\n'+hook)
    if (folder/'executed-source/rust/crates/ca-core/src/odometry.rs').read_bytes() != promoted:
        raise ValueError('library arithmetic changed beyond test-only hook')
    previous = read(root/'benchmarks/slam/flatwall-balance-dynamics/native-build-inputs.json')['normalized_sources_sha256']
    for name,digest in normalized.items():
        if name not in ['rust/crates/ca-core/src/odometry.rs','rust/crates/ca-core/src/odometry_profile.rs'] and previous.get(name) != digest:
            raise ValueError('other native algorithm source changed')
    protocol = read(folder/'protocol.json')
    if (protocol['states'],protocol['eligible_states'],protocol['insufficient_point_controls'],protocol['process_thread_order'],
            protocol['warmup_rounds'],protocol['measured_rounds'],protocol['evaluations_per_sample']) != (64,60,4,[1,4,4,1],1,5,32):
        raise ValueError('frozen timing scope differs')
    if any(protocol[k] for k in ['production_arithmetic_changed','defaults_changed','installed_extension_replaced','reference_poses_read','throughput_claim','native_executable_rebuilt_after_correction']):
        raise ValueError('unsupported algorithm or throughput claim')
    if protocol['earlier_protocol_sha256'] != sha(folder/'build-protocol.json') or read(folder/'build-controller-freeze.json')['protocol_sha256'] != sha(folder/'build-protocol.json'):
        raise ValueError('original pre-build protocol binding differs')
    artifact = read(folder/'native-artifact.json')
    if artifact['source_inputs_sha256'] != sha(folder/'native-build-inputs.json') or artifact['installed_extension_replaced']:
        raise ValueError('native artifact identity differs')
    original_freeze = read(folder/'build-controller-freeze.json')['files_sha256']
    for name,digest in build['sources_sha256'].items():
        if original_freeze[name] != digest:
            raise ValueError('original build source freeze differs')
    for name,archive in [('scripts/prepare_flatwall_balance_stages.py','original-preparation-source.py'),('scripts/run_flatwall_balance_stages.py','original-runner-source.py')]:
        if original_freeze[name] != sha(folder/archive):
            raise ValueError('initial preparation/runner source was replaced')
    correction = read(folder/'preparation-correction.json')
    if (correction['original_prepared_receipt_sha256'] != sha(folder/'original-prepared-receipt.json')
            or correction['corrected_prepared_receipt_sha256'] != sha(folder/'prepared-receipt.json')
            or correction['failed_preflight_sha256'] != sha(folder/'initial-preflight.log')
            or not correction['correction_before_first_timing'] or correction['native_executable_changed'] or correction['original_artifacts_rewritten']):
        raise ValueError('preparation correction provenance differs')
    prepared,initial = read(folder/'prepared-receipt.json'),read(folder/'original-prepared-receipt.json')
    for receipt,stem in [(prepared,''),(initial,'original-')]:
        if receipt['outputs_sha256']['input.json'] != sha(folder/(stem+'prepared-input.json')) or receipt['outputs_sha256']['origins.json'] != sha(folder/(stem+'origins.json')):
            raise ValueError('prepared state closure differs')
    if read(folder/'build-protocol.json')['prepared_receipt_sha256'] != sha(folder/'original-prepared-receipt.json') or protocol['prepared_receipt_sha256'] != sha(folder/'prepared-receipt.json'):
        raise ValueError('prepared protocol binding differs')
    origins,input_data = read(folder/'origins.json'),read(folder/'prepared-input.json')
    check_origins(root,origins,input_data['states'])
    independent = load_independent(root)
    expected = independent.expected(root,origins)
    for origin,definition,wanted in zip(origins,input_data['states'],expected):
        if (definition['source_points'],definition['map_points']) != (wanted['source_points'],wanted['map_points']) or prepared['outputs_sha256'][definition['file']] != origin['binary_sha256']:
            raise ValueError('native binary dimensions or hash differs')
    timing_freeze = read(folder/'timing-freeze.json')
    if timing_freeze['protocol_sha256'] != sha(folder/'protocol.json'):
        raise ValueError('timing protocol freeze differs')
    for name in ['input.json','origins.json','receipt.json']:
        mapped = {'input.json':'prepared-input.json','origins.json':'origins.json','receipt.json':'prepared-receipt.json'}[name]
        hits = [digest for path,digest in timing_freeze['files_sha256'].items() if path.endswith('/prepared-02/'+name)]
        if hits != [sha(folder/mapped)]:
            raise ValueError('executed prepared input differs')
    for name in ['scripts/prepare_flatwall_balance_stages.py','scripts/run_flatwall_balance_stages.py','scripts/flatwall_balance_stages.py']:
        if timing_freeze['files_sha256'][name] != hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest():
            raise ValueError('timed diagnostic controller changed')
    receipt,invocation = read(folder/'timing-receipt.json'),read(folder/'timing-invocation.json')
    binaries = [digest for path,digest in invocation['inputs_sha256'].items() if path.endswith('/native/odometry-stage-profile.exe')]
    if binaries != [artifact['binary_sha256']]:
        raise ValueError('executed native binary differs')
    if (not receipt['input_hashes_unchanged'] or receipt['invocation_sha256'] != sha(folder/'timing-invocation.json')
            or invocation['protocol_sha256'] != sha(folder/'protocol.json') or invocation['native_artifact_sha256'] != sha(folder/'native-artifact.json')
            or invocation['inputs_sha256'][(next(p for p in invocation['inputs_sha256'] if p.endswith('/timing-freeze.json')))] != sha(folder/'timing-freeze.json')):
        raise ValueError('native timing invocation did not close')
    runs,checks = [],[]
    for i,threads in enumerate([1,4,4,1]):
        name = f'run-{i:02d}-threads-{threads}.json'
        raw = gzip.decompress((folder/(name+'.gz')).read_bytes())
        if hashlib.sha256(raw).hexdigest() != receipt['output_files_sha256'][name] or sha(folder/f'run-{i:02d}.log') != receipt['output_files_sha256'][f'run-{i:02d}.log']:
            raise ValueError('native uncompressed output or log differs')
        run = json.loads(raw)
        checks.append(independent.check_run(run,expected,threads))
        runs.append(run)
    summary = independent.summarize(runs,origins)
    if summary != read(folder/'summary.json') or checks != read(folder/'verification.json')['per_process']:
        raise ValueError('complete timing summary differs')
    analysis_receipt = read(folder/'analysis-receipt.json')
    if not analysis_receipt['inputs_unchanged'] or analysis_receipt['invocation_sha256'] != sha(folder/'analysis-invocation.json'):
        raise ValueError('independent analysis did not close')
    for name,digest in analysis_receipt['output_files_sha256'].items():
        mapped = 'analysis-invocation.json' if name=='invocation.json' else name
        if sha(folder/mapped) != digest:
            raise ValueError('independent analysis output differs')
    return dict(states=64,eligible_states=60,original_empty_source_controls=4,native_validation_states=256,
        ordered_pair_rows_checked=sum(c['ordered_pair_rows_checked'] for c in checks),measured_blocks=1280,warmup_blocks=256,
        native_source_files_checked=len(normalized),production_arithmetic_changed=False,defaults_changed=False,
        whole_registration_speedup_established=False)


if __name__ == '__main__':
    print(json.dumps(verify()))
