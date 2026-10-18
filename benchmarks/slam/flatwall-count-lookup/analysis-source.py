"""Check count-optimization replay identity and complete real-data timing cohorts."""
from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import statistics
import struct

import numpy as np


def read(path):
    data = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(data)


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact(actual, expected):
    """Native outputs must retain numeric types and every floating-point bit."""
    if type(actual) is not type(expected):
        raise ValueError('native value type differs')
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError('native field inventory differs')
        for key in expected:
            exact(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError('native row inventory differs')
        for a, b in zip(actual, expected):
            exact(a, b)
    elif isinstance(expected, float):
        if not np.isfinite(actual) or struct.pack('<d', actual) != struct.pack('<d', expected):
            raise ValueError('native floating-point bits differ')
    elif actual != expected:
        raise ValueError('native value differs')


def bits(actual, expected):
    a, b = np.asarray(actual, dtype='<f8'), np.asarray(expected, dtype='<f8')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite native values')
    np.testing.assert_array_equal(a.view('<u8'), b.view('<u8'))


def original_cases(root):
    cohort = read(root/'benchmarks/slam/flatwall-balance-dynamics/cohort.json')
    result = []
    for case in cohort:
        baseline, balanced = rows(root/case['baseline_frames']), rows(root/case['balanced_frames'])
        with np.load(root/case['baseline_trajectory'], allow_pickle=False) as saved:
            baseline_poses = saved['poses'].copy()
            indices = saved['original_scan_indices'].tolist()
        if indices != [r['scan_index'] for r in baseline] or indices != [r['scan_index'] for r in balanced]:
            raise ValueError('original real cohort differs')
        result.append(dict(name=case['name'], indices=indices,
            poses={False: baseline_poses.reshape(-1, 16), True: np.asarray([r['returned_pose'] for r in balanced]).reshape(-1, 16)},
            reports={False: baseline, True: [r['registration'] for r in balanced]}))
    if len(result) != 16 or sum(len(r['indices']) for r in result) != 2560:
        raise ValueError('incomplete original real cohort')
    return result


def check_replay(run, originals, mode):
    if mode not in ['time', 'trace'] or run['mode'] != mode:
        raise ValueError('native replay mode differs')
    if run['diagnostic_capture_enabled'] != (mode == 'trace') or run['requested_rayon_threads'] != 4:
        raise ValueError('capture policy or requested threads differs')
    rounds = 4 if mode == 'time' else 1
    if len(run['runs']) != 16 * 2 * rounds:
        raise ValueError('incomplete native replay runs')
    cursor, frame_count = 0, 0
    for round_index in range(rounds):
        for case_index, case in enumerate(originals):
            order = [False, True] if (round_index + case_index) % 2 == 0 else [True, False]
            for balance in order:
                row = run['runs'][cursor]
                cursor += 1
                if type(row['target_balance']) is not bool or type(row['warmup']) is not bool or type(row['round']) is not int:
                    raise ValueError('invalid native replay policy types')
                if (row['case'], row['target_balance'], row['round'], row['warmup']) != (
                        case['name'], balance, round_index, mode == 'time' and round_index == 0):
                    raise ValueError('native replay order or warmup differs')
                if [f['scan_index'] for f in row['frames']] != case['indices']:
                    raise ValueError('native replay frame inventory differs')
                bits([f['pose'] for f in row['frames']], case['poses'][balance])
                reports = case['reports'][balance]
                for key in ['iterations', 'termination']:
                    if [f[key] for f in row['frames']] != [f[key] for f in reports]:
                        raise ValueError('native replay convergence policy differs')
                bits([f['threshold_m'] for f in row['frames']], [f['threshold_m'] for f in reports])
                if mode == 'time':
                    times = row['registration_ns']
                    if len(times) != len(case['indices']) or any(type(v) is not int or v <= 0 for v in times):
                        raise ValueError('missing native registration timing')
                    if type(row['total_registration_ns']) is not int or row['total_registration_ns'] != sum(times):
                        raise ValueError('native registration time sum differs')
                frame_count += len(case['indices'])
    return dict(runs=cursor, frames=frame_count, exact_original_pose_frames=frame_count)


def compare_replays(actual, expected):
    """Compare complete numerical results, excluding elapsed times only."""
    if len(actual['runs']) != len(expected['runs']):
        raise ValueError('paired replay run inventory differs')
    for a, b in zip(actual['runs'], expected['runs']):
        for key in ['case', 'target_balance', 'round', 'warmup', 'frames']:
            exact(a[key], b[key])


def summarize_time(runs, names):
    if [(r['implementation'], r['index']) for r in runs] != [
            ('baseline', 0), ('candidate', 1), ('candidate', 2), ('baseline', 3)]:
        raise ValueError('counterbalanced implementation order differs')
    cases = []
    for name in names:
        item = dict(case=name, modes={})
        for balance in [False, True]:
            samples = {}
            for implementation in ['baseline', 'candidate']:
                samples[implementation] = [row['total_registration_ns'] for r in runs if r['implementation'] == implementation
                    for row in r['data']['runs'] if row['case'] == name and row['target_balance'] == balance and not row['warmup']]
                if len(samples[implementation]) != 6:
                    raise ValueError('require all six measured recording totals per implementation')
            medians = {k: statistics.median(v) for k, v in samples.items()}
            item['modes']['balanced' if balance else 'default_control'] = dict(samples_ns=samples,
                median_ns=medians, candidate_over_baseline=medians['candidate']/medians['baseline'])
        cases.append(item)
    groups = {}
    for group, selected in [('all', cases), ('legacy', [c for c in cases if c['case'].endswith('legacy')]),
                            ('fine', [c for c in cases if c['case'].endswith('0.1875')])]:
        groups[group] = dict(cases=len(selected), modes={})
        for mode in ['default_control', 'balanced']:
            sums = {implementation: sum(c['modes'][mode]['median_ns'][implementation] for c in selected)
                    for implementation in ['baseline', 'candidate']}
            ratios = [c['modes'][mode]['candidate_over_baseline'] for c in selected]
            groups[group]['modes'][mode] = dict(sum_case_medians_ns=sums,
                candidate_over_baseline=sums['candidate']/sums['baseline'],
                case_ratio_range=[min(ratios), max(ratios)], faster_cases=sum(v < 1 for v in ratios))
    return dict(cases=cases, groups=groups, measured_recordings_per_case_and_implementation=6)


def summarize_fixed(runs, origins):
    order = [(i, t) for t in [1, 4] for i in ['baseline', 'candidate', 'candidate', 'baseline']]
    if [(r['implementation'], r['threads']) for r in runs] != order:
        raise ValueError('fixed-state implementation/thread order differs')
    per_state = []
    groups = dict(eligible=[], fine_final=[], legacy_final=[], capped_steps=[], ineligible=[])
    for i, origin in enumerate(origins):
        group = ('ineligible' if not origin['alignment_eligible'] else 'capped_steps' if origin['kind'] == 'step'
                 else 'fine_final' if origin['case'].endswith('0.1875') else 'legacy_final')
        groups[group].append(i)
        if origin['alignment_eligible']:
            groups['eligible'].append(i)
        item = dict(name=origin['name'], group=group, threads={})
        for threads in [1, 4]:
            implementations = {}
            for implementation in ['baseline', 'candidate']:
                samples = [s for r in runs if r['implementation'] == implementation and r['threads'] == threads
                    for s in r['data']['samples'] if s['state_index'] == i and not s['warmup']]
                if len(samples) != 10:
                    raise ValueError('incomplete fixed-state timing blocks')
                implementations[implementation] = {key: statistics.median(s[key]/32 for s in samples)
                    for key in ['balanced_ns', 'baseline_ns', 'count_ns', 'staged_ns']}
            item['threads'][str(threads)] = implementations
        per_state.append(item)
    result = {}
    for group, indices in groups.items():
        item = dict(states=len(indices), threads={})
        for threads in [1, 4]:
            values = {}
            for key in ['balanced_ns', 'baseline_ns', 'count_ns', 'staged_ns']:
                sums = {implementation: sum(per_state[i]['threads'][str(threads)][implementation][key] for i in indices)
                    for implementation in ['baseline', 'candidate']}
                values[key] = dict(sum_state_medians_ns=sums, candidate_over_baseline=sums['candidate']/sums['baseline'])
            item['threads'][str(threads)] = values
        result[group] = item
    return dict(groups=result, per_state=per_state, measured_blocks_per_state_thread_and_implementation=10)


def check_traces(baseline, candidate, root):
    catalog = read(root/'benchmarks/slam/flatwall-balance-dynamics/summary.json')['traces']
    expected_keys = [(r['case'], r['scan_index']) for r in catalog]
    if list(baseline) != expected_keys or list(candidate) != expected_keys:
        raise ValueError('capped trace cohort differs')
    steps, pairs = 0, 0
    for row in catalog:
        key = (row['case'], row['scan_index'])
        a, b = baseline[key], candidate[key]
        exact(a, b)
        if len(a['states']) != 500 or a['termination'] != 'iteration_limit':
            raise ValueError('capped native step inventory differs')
        with np.load(root/'benchmarks/slam/flatwall-balance-dynamics'/row['file'], allow_pickle=False) as z:
            for field in ['source_at_prediction', 'map_points_world', 'center_world', 'threshold_sq', 'kernel',
                          'final_source_world', 'final_target_world', 'final_weights', 'returned_pose']:
                bits(np.asarray(a[field], dtype='<f8').reshape(z[field].shape), z[field])
            np.testing.assert_array_equal(a['final_source_indices'], z['final_source_indices'])
            for field in ['before', 'after', 'normal_matrix', 'right_hand_side', 'weighted_cost']:
                values = np.asarray([s[field] for s in a['states']], dtype='<f8').reshape(z[field].shape)
                bits(values, z[field])
            for field in ['matches', 'unique_targets']:
                np.testing.assert_array_equal([s[field] for s in a['states']], z[field])
            for i in range(500):
                state = a['states'][i]
                if i < 476:
                    if any(field in state for field in ['source_indices', 'target_indices', 'multiplicity', 'weights']):
                        raise ValueError('unpromised full correspondence rows outside trace tail')
                else:
                    start, end = z['tail_offsets'][i-476:i-474]
                    for field, original in [('source_indices', 'tail_source_indices'), ('target_indices', 'tail_target_indices'),
                                            ('multiplicity', 'tail_multiplicity')]:
                        np.testing.assert_array_equal(state[field], z[original][start:end])
                    bits(state['weights'], z['tail_weights'][start:end])
            pairs += len(z['final_source_indices'])
        steps += 500
    return dict(traces=len(catalog), exact_capped_step_systems_and_updates=steps,
        exact_final_pairs=pairs, original_published_trace_bits_match=True)


def check_fixed(runs, root):
    original = load_module(root/'scripts/flatwall_balance_stages.py', 'fixed_count_comparison_oracle')
    origins = read(root/'benchmarks/slam/flatwall-balance-stages/origins.json')
    expected = original.expected(root, origins)
    reports = []
    references = {}
    for run in runs:
        reports.append(check_fixed_run(run['data'], expected, run['threads']))
        key = run['threads']
        observed = run['data']['validation']
        if key in references:
            exact(observed, references[key])
        else:
            references[key] = observed
    return dict(real_states=64, eligible_states=60, empty_source_controls=4,
        validation_states=sum(r['states_checked'] for r in reports),
        ordered_pair_rows_checked=sum(r['ordered_pair_rows_checked'] for r in reports),
        measured_blocks=len(runs)*64*5, warmup_blocks=len(runs)*64,
        native_count_and_matrix_bits_identical=True, independent_original_pair_and_matrix_checks=True)


def check_fixed_run(run, expected, threads):
    if (run['schema'], run['threads'], run['repetitions']) != ('cloudanalyzer.odometry_stage_profile.v1', threads, 32):
        raise ValueError('fixed native run identity differs')
    if len(run['validation']) != 64 or len(run['samples']) != 64*6:
        raise ValueError('incomplete fixed native state/sample inventory')
    maximum = 0.0
    for observed, wanted in zip(run['validation'], expected):
        for name in ['name', 'source_points', 'map_points']:
            if observed[name] != wanted[name]:
                raise ValueError('fixed state identity differs')
        for name in ['source_indices', 'targets', 'multiplicity']:
            value = np.asarray(observed[name]).reshape(-1, 3) if name == 'targets' else observed[name]
            np.testing.assert_array_equal(value, wanted[name])
        np.testing.assert_allclose(observed['distance_sq'], wanted['distance_sq'], rtol=5e-12, atol=2e-12)
        for name in ['baseline_h', 'baseline_b', 'balanced_h', 'balanced_b']:
            np.testing.assert_allclose(observed[name], wanted[name], rtol=5e-12, atol=2e-8)
        errors = []
        for suffix in ['h', 'b']:
            native, decomposed = np.asarray(observed['balanced_'+suffix]), np.asarray(observed['decomposed_'+suffix])
            np.testing.assert_allclose(decomposed, native, rtol=5e-12, atol=5e-9)
            errors.append(float(np.max(np.abs(native-decomposed))))
        if observed['maximum_decomposition_abs_error'] != max(errors):
            raise ValueError('fixed native decomposition error differs')
        maximum = max(maximum, max(errors))
    components = ['search_ns', 'count_ns', 'accumulate_ns', 'destroy_ns']
    times = ['baseline_ns', 'balanced_ns', *components, 'staged_ns']
    for index, sample in enumerate(run['samples']):
        round_index, state_index = divmod(index, 64)
        order = [(round_index+state_index+i) % 3 for i in range(3)]
        if (sample['round'], sample['state_index'], sample['warmup'], sample['repetitions'], sample['order']) != (
                round_index, state_index, round_index == 0, 32, order):
            raise ValueError('fixed sample order, warmup or repetition count differs')
        for key in times:
            # Empty controls can finish a component within one native clock tick.
            # Retain that recorded zero; eligible work and whole calls stay positive.
            lower = 0 if expected[state_index]['source_points'] == 0 and key in components else 1
            if type(sample[key]) is not int or sample[key] < lower:
                raise ValueError('invalid fixed native elapsed time')
        if sum(sample[key] for key in components) > sample['staged_ns']:
            raise ValueError('fixed stages exceed their outer clock window')
        for name in ['maximum_baseline_abs_error', 'maximum_balanced_abs_error', 'maximum_staged_abs_error']:
            if not np.isfinite(sample[name]) or not 0 <= sample[name] <= 1e-6:
                raise ValueError('fixed native batch self-check differs')
    return dict(states_checked=64, ordered_pair_rows_checked=sum(len(s['source_indices']) for s in expected),
        samples_checked=64*6, maximum_decomposition_abs_error=maximum)
