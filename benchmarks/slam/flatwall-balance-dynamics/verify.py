#!/usr/bin/env python3
"""Verify all native trace systems, replay cohorts and published timing samples."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k:saved[k].copy() for k in saved.files}


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def equal(actual, expected):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError('summary keys differ')
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('summary length differs')
        for a, b in zip(actual, expected):
            equal(a, b)
    elif isinstance(expected, (float, np.floating)):
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    elif actual != expected:
        raise ValueError('summary value differs')


def check_runs(data, cohort, root, mode):
    parsed = load(root/'benchmarks/slam/flatwall-balance-dynamics/expected-input-bits.npz')
    rounds = 1 if mode == 'trace' else 4
    run_count = rounds*16*2
    if len(data['case_index']) != run_count or len(data['offsets']) != run_count+1 or data['offsets'][0] != 0:
        raise ValueError('incomplete native runs')
    total, exact, maximum = 0, 0, 0.
    expected_cases = []
    for case in cohort:
        baseline, balanced = rows(root/case['baseline_frames']), rows(root/case['balanced_frames'])
        with np.load(root/case['baseline_trajectory'], allow_pickle=False) as saved:
            baseline_poses = saved['poses'].copy()
            ids = saved['original_scan_indices'].copy()
        if ids.tolist() != [f['scan_index'] for f in baseline] or ids.tolist() != [f['scan_index'] for f in balanced]:
            raise ValueError('original cohort differs')
        expected_cases.append((baseline, balanced, baseline_poses, ids))
    cursor = 0
    for round_index in range(rounds):
        for case_index, case in enumerate(cohort):
            baseline, balanced, baseline_poses, ids = expected_cases[case_index]
            order = [False, True] if (round_index+case_index)%2 == 0 else [True, False]
            for balance in order:
                i = cursor
                cursor += 1
                if (int(data['case_index'][i]), int(data['round'][i]), bool(data['target_balance'][i]), bool(data['warmup'][i])) != (
                        case_index, round_index, balance, mode == 'time' and round_index == 0):
                    raise ValueError('counterbalanced ordering or warmup differs')
                start, end = data['offsets'][i:i+2]
                if end-start != len(ids):
                    raise ValueError('a run omits original scans')
                np.testing.assert_array_equal(data['scan_index'][start:end], ids)
                original = balanced if balance else baseline
                expected_pose = np.asarray([f['returned_pose'] for f in balanced]) if balance else baseline_poses
                actual = data['poses'][start:end]
                pstart, pend = parsed['offsets'][case_index:case_index+2]
                parsed_bits = parsed[('balanced_bits' if balance else 'baseline_bits')][pstart:pend]
                parsed_pose = parsed_bits.view(np.float64)
                np.testing.assert_allclose(parsed_pose, expected_pose, rtol=0, atol=2e-15)
                np.testing.assert_array_equal(data['native_input_pose_max_abs_error'][start:end], np.max(np.abs(actual-parsed_pose),axis=(1,2)))
                np.testing.assert_array_equal(data['native_input_pose_bits_match'][start:end], np.all(actual.view(np.uint64)==parsed_bits,axis=(1,2)))
                np.testing.assert_allclose(actual, expected_pose, rtol=0, atol=2e-8)
                differences = np.max(np.abs(actual-expected_pose), axis=(1,2))
                np.testing.assert_array_equal(data['pose_max_abs_error'][start:end], differences)
                bits = np.all(actual.view(np.uint64) == expected_pose.view(np.uint64), axis=(1,2))
                np.testing.assert_array_equal(data['pose_bits_match'][start:end], bits)
                reports = [f['registration'] for f in original] if balance else original
                for key in ['iterations', 'termination']:
                    np.testing.assert_array_equal(data[key][start:end], [r[key] for r in reports])
                np.testing.assert_allclose(data['threshold_m'][start:end], [r['threshold_m'] for r in reports], rtol=0, atol=2e-8)
                if mode == 'time':
                    times = data['registration_ns'][start:end]
                    if len(times) != len(ids) or np.any(times <= 0) or data['total_registration_ns'][i] != times.sum():
                        raise ValueError('native timing sample is missing or sum differs')
                total += len(ids)
                exact += int(bits.sum())
                maximum = max(maximum, float(differences.max()))
    if data['offsets'][-1] != total or len(data['poses']) != total:
        raise ValueError('extra native frames')
    return dict(frames_checked=total, exact_pose_bit_frames=exact, maximum_pose_abs_error=maximum)


def check_timing(data, saved, cohort):
    cases = []
    for i, case in enumerate(cohort):
        observed = saved['cases'][i]
        baseline, balanced, ratios, samples = [], [], [], []
        for round_index in [1,2,3]:
            values = []
            for balance in [False, True]:
                hits = np.flatnonzero((data['case_index'] == i) & (data['round'] == round_index) & (data['target_balance'] == balance))
                if len(hits) != 1 or data['warmup'][hits[0]]:
                    raise ValueError('invalid measured pair')
                values.append(int(data['total_registration_ns'][hits[0]]))
            a, b = values
            baseline.append(a)
            balanced.append(b)
            ratios.append(b/a)
            samples.append(dict(round=round_index, baseline_ns=a, balanced_ns=b, balanced_over_baseline=b/a))
        scans = int(data['offsets'][hits[0]+1]-data['offsets'][hits[0]])
        expected = dict(case=case['name'], scans=scans,
            measured_pairs=samples, baseline_median_ns=float(statistics.median(baseline)),
            balanced_median_ns=float(statistics.median(balanced)), baseline_range_ns=[min(baseline),max(baseline)],
            balanced_range_ns=[min(balanced),max(balanced)],
            balanced_over_baseline_median_ratio=statistics.median(balanced)/statistics.median(baseline),
            paired_ratio_median=float(statistics.median(ratios)))
        equal(observed, expected)
        cases.append(expected)
    equal(saved['measured_pairs_per_case'], 3)
    equal(saved['sum_case_medians_balanced_over_baseline'], sum(c['balanced_median_ns'] for c in cases)/sum(c['baseline_median_ns'] for c in cases))
    values = [c['balanced_over_baseline_median_ratio'] for c in cases]
    equal(saved['case_median_ratio_range'], [min(values),max(values)])
    equal(saved['faster_cases'], sum(v < 1 for v in values))


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
    footprint = read(folder/'source-footprints.json')
    for group in ['normalized_native_sources_sha256','normalized_checkout_sources_sha256']:
        for name, digest in footprint[group].items():
            if hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() != digest:
                raise ValueError('normalized source differs: '+name)
    build = read(folder/'native-build-inputs.json')
    for name, digest in build['normalized_sources_sha256'].items():
        mapped = footprint['native_source_aliases'].get(name, name)
        if footprint['normalized_native_sources_sha256'][mapped] != digest:
            raise ValueError('executed source alias differs')
    promotion = read(folder/'source-promotion.json')
    archived = root/promotion['preserved_as']
    if sha(archived) != promotion['executed_sha256'] or build['sources_sha256'][promotion['original_path']] != promotion['executed_sha256']:
        raise ValueError('executed example was replaced')
    original = archived.read_bytes().replace(b'\r\n',b'\n')
    replacement = original.replace(b'.chunks_exact(24)', b'.as_chunks::<24>()\n                .0\n                .iter()')
    if replacement != (root/promotion['original_path']).read_bytes().replace(b'\r\n',b'\n'):
        raise ValueError('reader-only source promotion differs')
    artifact = read(folder/'native-artifact.json')
    if artifact['source_inputs_sha256'] != sha(folder/'native-build-inputs.json') or artifact['installed_extension_replaced']:
        raise ValueError('native provenance differs')
    protocol = read(folder/'protocol.json')
    if read(folder/'controller-freeze.json')['protocol_sha256'] != sha(folder/'protocol.json'):
        raise ValueError('native protocol differs')
    for stage in ['trace-01', 'time-01']:
        receipt = read(folder/(stage+'-receipt.json'))
        if receipt['invocation_sha256'] != sha(folder/(stage+'-invocation.json')) or not receipt['input_hashes_unchanged']:
            raise ValueError('native stage did not close')
        invocation = read(folder/(stage+'-invocation.json'))
        if invocation['protocol_sha256'] != sha(folder/'protocol.json') or invocation['native_artifact_sha256'] != sha(folder/'native-artifact.json'):
            raise ValueError('executed invocation differs')
    cohort, summary = read(folder/'cohort.json'), read(folder/'summary.json')
    arms = ['raw-legacy','gyro-legacy','raw-0.1875','gyro-0.1875']
    names = [f'{recording}/{policy}/{arm}' for recording in ['flatwall_01','flatwall_02'] for policy in ['repeat-motion','last-pose'] for arm in arms]
    if [c['name'] for c in cohort] != names or protocol['cases'] != names:
        raise ValueError('sixteen original conditions required')
    selected = {(c['name'],r['scan_index']) for c in cohort for r in rows(root/c['balanced_frames']) if r['registration']['termination'] == 'iteration_limit'}
    promised = {(c['name'],i) for c in cohort for i in c['trace_indices']}
    if selected != promised or len(promised) != 8 or promised != {(t['case'],t['scan_index']) for t in summary['traces']}:
        raise ValueError('capped-frame selection differs')
    spec = importlib.util.spec_from_file_location('independent_balance_steps',root/'scripts/flatwall_balance_dynamics.py')
    independent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(independent)
    original_receipt = read(root/'benchmarks/slam/flatwall-target-balance/estimates-receipt.json')
    trace_receipt = read(folder/'trace-01-receipt.json')
    original_analysis = read(folder/'analysis-01-receipt.json')
    for name, digest in original_analysis['output_files_sha256'].items():
        mapped = ('native-reported-'+name if name in ['summary.json','trace-runs.npz','time-runs.npz']
                  else 'analysis-01-invocation.json' if name == 'invocation.json' else name)
        if sha(folder/mapped) != digest:
            raise ValueError('original analysis changed')
    analysis_receipt = read(folder/'analysis-02-receipt.json')
    if not analysis_receipt['inputs_unchanged'] or analysis_receipt['invocation_sha256'] != sha(folder/'analysis-02-invocation.json'):
        raise ValueError('pose-diagnostic reconciliation did not close')
    for name, digest in analysis_receipt['output_files_sha256'].items():
        mapped = 'analysis-02-invocation.json' if name == 'invocation.json' else name
        if sha(folder/mapped) != digest:
            raise ValueError('reconciled analysis changed')
    probe = read(folder/'pose-input-probe.json')
    if probe['compact_input_bits_sha256'] != sha(folder/'expected-input-bits.npz'):
        raise ValueError('native parsed-input bits changed')
    if hashlib.sha256((root/probe['source']).read_bytes().replace(b'\r\n',b'\n')).hexdigest() != probe['source_sha256']:
        raise ValueError('native parser probe source changed')
    if any(probe[k] for k in ['estimator_inputs_changed','estimation_repeated','timings_repeated']):
        raise ValueError('reconciliation changed the executed trial')
    for mode in ['trace','time']:
        old = load(folder/('native-reported-'+mode+'-runs.npz'))
        current = load(folder/(mode+'-runs.npz'))
        for key,value in old.items():
            mapped = 'native_input_'+key if key in ['pose_max_abs_error','pose_bits_match'] else key
            np.testing.assert_array_equal(current[mapped],value)
    total = 0
    periods = []
    for trace in summary['traces']:
        name = f"{trace['case']}/scan-{trace['scan_index']:03d}"
        if trace['source_json_sha256'] != trace_receipt['output_files_sha256'][name+'.json']:
            raise ValueError('trace does not bind the executed record')
        if trace['file'] != name+'.npz' or trace['sha256'] != sha(folder/trace['file']) or analysis_receipt['output_files_sha256'][trace['file']] != trace['sha256']:
            raise ValueError('closed trace conversion differs')
        for suffix in ['npz','json']:
            if sha(folder/'original'/(name+'.'+suffix)) != original_receipt['output_files_sha256'][name+'.'+suffix]:
                raise ValueError('original snapshot changed')
        data = load(folder/trace['file'])
        original = load(folder/'original'/(name+'.npz'))
        case = next(c for c in cohort if c['name'] == trace['case'])
        frame = next(r for r in rows(root/case['balanced_frames']) if r['scan_index'] == trace['scan_index'])
        np.testing.assert_allclose(data['source_at_prediction'], independent.transformed(original['source_sensor'],np.asarray(frame['prediction_pose'])), rtol=0,atol=2e-12)
        np.testing.assert_allclose(data['returned_pose'],frame['returned_pose'],rtol=0,atol=2e-8)
        np.testing.assert_allclose(data['center_world'], original['center_world'],rtol=0,atol=2e-12)
        metadata = read(folder/'original'/(name+'.json'))
        equal(float(data['kernel']),metadata['kernel'])
        equal(float(data['threshold_sq']),metadata['threshold_sq'])
        np.testing.assert_allclose(data['final_source_world'],original['source_world'],rtol=0,atol=2e-8)
        np.testing.assert_array_equal(data['final_source_indices'],original['matched_source_indices'])
        np.testing.assert_array_equal(data['final_target_world'],original['target_world'])
        np.testing.assert_allclose(data['final_weights'],original['weights'],rtol=2e-12,atol=2e-12)
        checked = independent.verify_trace(data)
        for key, value in checked.items():
            equal(trace[key], value)
        total += checked['native_steps_checked']
        # Numeric tail recurrence is descriptive, never a changed convergence rule.
        matching = [r['lag'] for r in checked['tail']['lags'] if r['identical_ordered_pair_sets'] == r['comparisons']
                    and r['identical_multiplicities'] == r['comparisons'] and r['maximum_translation_m'] <= 1e-9 and r['maximum_rotation_rad'] <= 1e-10]
        periods.append(min(matching) if matching else None)
    trace_replay = check_runs(load(folder/'trace-runs.npz'),cohort,root,'trace')
    timing_data = load(folder/'time-runs.npz')
    time_replay = check_runs(timing_data,cohort,root,'time')
    check_timing(timing_data,summary['timing'],cohort)
    diagnostic_counts = summary['pose_diagnostics']['counts']
    for mode, checked in [('trace',trace_replay),('time',time_replay)]:
        data = load(folder/(mode+'-runs.npz'))
        equal(diagnostic_counts[mode], dict(frames=checked['frames_checked'],original_pose_bit_matches=checked['exact_pose_bit_frames'],
            native_parsed_input_bit_matches=int(data['native_input_pose_bits_match'].sum()),maximum_original_pose_abs_error=checked['maximum_pose_abs_error']))
    if summary['pose_diagnostics']['estimation_changed'] or summary['pose_diagnostics']['timings_changed']:
        raise ValueError('pose diagnostics changed estimation or timing')
    for key in ['candidate_changed','defaults_changed','state_feedback','reference_poses_read','runtime_improvement_adopted']:
        if summary[key] is not False:
            raise ValueError('audit fed state back into the estimator')
    if total != summary['native_steps_checked'] or total != 4000 or summary['capped_frames'] != 8:
        raise ValueError('incomplete step audit')
    return dict(capped_frames=8,native_steps_checked=total,tail_recurrence_periods=periods,
        trace_replay=trace_replay,time_replay=time_replay,measured_case_pairs=48,
        native_sources_checked=len(build['normalized_sources_sha256']),
        sum_case_medians_balanced_over_baseline=summary['timing']['sum_case_medians_balanced_over_baseline'],
        candidate_changed=False,defaults_changed=False,reference_poses_read=False)


if __name__ == '__main__':
    print(json.dumps(verify(),indent=2))
