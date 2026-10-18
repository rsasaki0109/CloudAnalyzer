"""Real replay identity, timing accounting and executed-source regression checks."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT/'benchmarks/slam/flatwall-count-lookup'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


packet = module(PACKET/'verify.py', 'count_lookup_packet')
comparison = module(ROOT/'scripts/flatwall_count_comparison.py', 'count_lookup_comparison')


@pytest.fixture(scope='module')
def fixed_reference():
    original = module(ROOT/'scripts/flatwall_balance_stages.py', 'count_lookup_original')
    origins = packet.read(ROOT/'benchmarks/slam/flatwall-balance-stages/origins.json')
    expected = original.expected(ROOT, origins)
    catalog = packet.read(PACKET/'catalog.json')
    # This recorded candidate run contains a genuine zero-duration empty control.
    item = catalog['fixed'][2]
    return comparison.read(PACKET/item['file']), expected


def test_all_original_real_replays_systems_and_recording_timings():
    result = packet.verify(PACKET)
    assert result['replay_frames_checked'] == 92160
    assert result['capped_step_systems_checked'] == 4000
    assert result['real_fixed_states'] == 64 and result['eligible_fixed_states'] == 60
    assert result['original_pose_and_matrix_bits_match'] is True
    assert result['defaults_changed'] is result['whole_application_speedup_established'] is False


def test_empty_clock_resolution_does_not_permit_zero_elapsed_real_work(fixed_reference):
    run, expected = fixed_reference
    zero = [s for s in run['samples'] if s['search_ns'] == 0 or s['count_ns'] == 0]
    assert zero and all(expected[s['state_index']]['source_points'] == 0 for s in zero)
    assert comparison.check_fixed_run(run, expected, 1)['states_checked'] == 64
    changed = copy.deepcopy(run)
    changed['samples'][0]['count_ns'] = 0
    assert expected[0]['source_points'] > 0
    with pytest.raises(ValueError, match='elapsed'):
        comparison.check_fixed_run(changed, expected, 1)
    changed = copy.deepcopy(run)
    next(s for s in changed['samples'] if s['state_index'] == zero[0]['state_index'])['count_ns'] = -1
    with pytest.raises(ValueError, match='elapsed'):
        comparison.check_fixed_run(changed, expected, 1)


@pytest.mark.parametrize('field', ['multiplicity', 'balanced_h'])
def test_changed_real_counts_or_systems_are_rejected(fixed_reference, field):
    run, expected = fixed_reference
    changed = copy.deepcopy(run)
    value = changed['validation'][0][field]
    if field == 'multiplicity':
        value[0] += 1
    else:
        value[0][0] += 1.0
    with pytest.raises(AssertionError):
        comparison.check_fixed_run(changed, expected, 1)


def test_warmup_cannot_be_reclassified_as_measured_recording():
    item = packet.read(PACKET/'catalog.json')['time'][0]
    run = comparison.read(PACKET/item['file'])
    assert run['runs'][0]['warmup'] is True
    run['runs'][0]['warmup'] = False
    with pytest.raises(ValueError, match='order or warmup'):
        comparison.check_replay(run, comparison.original_cases(ROOT), 'time')


def test_summary_roundoff_is_allowed_but_speedup_changes_are_rejected():
    expected = packet.read(PACKET/'summary.json')
    changed = copy.deepcopy(expected)
    row = changed['registration']['groups']['all']['modes']['balanced']
    row['candidate_over_baseline'] = float(np.nextafter(row['candidate_over_baseline'], np.inf))
    packet.equal(changed, expected)
    row['candidate_over_baseline'] *= 0.95
    with pytest.raises(ValueError, match='summary'):
        packet.equal(changed, expected)


def copy_packet(tmp_path):
    target = tmp_path/'checkout/benchmarks/slam/flatwall-count-lookup'
    shutil.copytree(PACKET, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    manifest = packet.read(PACKET/'manifest.json')
    for name in manifest['dependencies'] | manifest['normalized_checkout_sources_sha256']:
        destination = target.parents[2]/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
    return target


def test_source_archive_still_matches_native_build_after_resigning_packet(tmp_path):
    target = copy_packet(tmp_path)
    footprint = packet.read(target/'source-footprints.json')
    name = footprint['candidate']['rust/crates/ca-core/src/odometry_target_balance.rs']
    with (target.parents[2]/name).open('ab') as stream:
        stream.write(b'\n// changed executed source\n')
    manifest = packet.read(target/'manifest.json')
    for name in manifest['files_sha256']:
        manifest['files_sha256'][name] = packet.sha(target/name)
    for name in manifest['dependencies']:
        manifest['dependencies'][name] = packet.sha(target.parents[2]/name)
    (target/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    with pytest.raises(ValueError, match='executed native source'):
        packet.verify(target)


def test_development_checkout_newlines_leave_archived_sources_unchanged(tmp_path):
    target = copy_packet(tmp_path)
    manifest = packet.read(target/'manifest.json')
    for name in manifest['normalized_checkout_sources_sha256']:
        path = target.parents[2]/name
        data = path.read_bytes().replace(b'\r\n', b'\n')
        path.write_bytes(data.replace(b'\n', b'\r\n'))
    assert packet.verify(target)['original_pose_and_matrix_bits_match'] is True
