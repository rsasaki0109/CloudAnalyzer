"""Real registration bits, complete timing cohorts and executed-source bindings."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT/'benchmarks/slam/flatwall-worker-dispatch'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


packet = module(PACKET/'verify.py', 'worker_dispatch_packet')
comparison = module(ROOT/'scripts/flatwall_count_comparison.py', 'worker_dispatch_originals')


@pytest.fixture(scope='module')
def recorded():
    item = packet.read(PACKET/'catalog.json')[0]
    return comparison.read(PACKET/item['file']), comparison.original_cases(ROOT)


def test_complete_original_real_recordings_and_native_builds():
    result = packet.verify(PACKET)
    assert result['replay_frames_checked'] == 81920 and result['real_cases'] == 16
    assert result['implementation_changes'] == ['rust/crates/ca-core/src/odometry.rs']
    assert result['original_pose_and_threshold_bits_match'] is True
    assert result['defaults_changed'] is result['whole_application_speedup_established'] is False


@pytest.mark.parametrize('field', ['pose', 'threshold_m', 'iterations', 'termination'])
def test_a_changed_real_frame_is_rejected(recorded, field):
    run, originals = recorded
    changed = copy.deepcopy(run)
    frame = changed['runs'][0]['frames'][1]
    if field == 'pose':
        frame[field][0] = float(np.nextafter(frame[field][0], np.inf))
    elif field == 'threshold_m':
        frame[field] = float(np.nextafter(frame[field], np.inf))
    elif field == 'iterations':
        frame[field] += 1
    else:
        frame[field] = 'unrecognized'
    with pytest.raises((AssertionError, ValueError)):
        comparison.check_replay(changed, originals, 'time')


@pytest.mark.parametrize('change', ['warmup', 'missing_frame', 'zero_clock', 'total'])
def test_incomplete_or_reclassified_timings_are_rejected(recorded, change):
    run, originals = recorded
    changed = copy.deepcopy(run)
    row = changed['runs'][0]
    if change == 'warmup':
        row['warmup'] = False
    elif change == 'missing_frame':
        row['frames'].pop()
    elif change == 'zero_clock':
        row['registration_ns'][0] = 0
    else:
        row['total_registration_ns'] += 1
    with pytest.raises(ValueError):
        comparison.check_replay(changed, originals, 'time')


def test_summary_roundoff_does_not_allow_a_changed_speedup():
    expected = packet.read(PACKET/'summary.json')
    changed = copy.deepcopy(expected)
    row = changed['registration']['groups']['all']['modes']['balanced']
    row['candidate_over_baseline'] = float(np.nextafter(row['candidate_over_baseline'], np.inf))
    packet.equal(changed, expected)
    row['candidate_over_baseline'] *= 0.95
    with pytest.raises(ValueError, match='summary'):
        packet.equal(changed, expected)


def copy_packet(tmp_path):
    target = tmp_path/'checkout/benchmarks/slam/flatwall-worker-dispatch'
    shutil.copytree(PACKET, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    manifest = packet.read(PACKET/'manifest.json')
    for name in manifest['dependencies'] | manifest['normalized_checkout_sources_sha256']:
        destination = target.parents[2]/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
    return target


def test_resigning_packet_cannot_change_the_executed_native_source(tmp_path):
    target = copy_packet(tmp_path)
    footprint = packet.read(target/'source-footprints.json')
    name = footprint['candidate']['rust/crates/ca-core/src/odometry.rs']
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


def test_working_checkout_newlines_preserve_the_original_proof(tmp_path):
    target = copy_packet(tmp_path)
    manifest = packet.read(PACKET/'manifest.json')
    for name in manifest['normalized_checkout_sources_sha256']:
        path = target.parents[2]/name
        data = path.read_bytes().replace(b'\r\n', b'\n')
        path.write_bytes(data.replace(b'\n', b'\r\n'))
    assert packet.verify(target)['original_pose_and_threshold_bits_match'] is True


def test_comment_promotion_cannot_hide_a_statement_change(tmp_path):
    target = copy_packet(tmp_path)
    promotion = packet.read(target/'source-promotion.json')
    source = target.parents[2]/promotion['promoted_source']
    text = source.read_text(encoding='utf-8')
    assert text.count('&& self.iterations > 0') == 1
    source.write_text(text.replace('&& self.iterations > 0', '&& self.iterations > 1'), encoding='utf-8')
    promotion['promoted_sha256'] = packet.sha(source)
    (target/'source-promotion.json').write_text(json.dumps(promotion, indent=2)+'\n', encoding='utf-8')
    manifest = packet.read(target/'manifest.json')
    for name in manifest['files_sha256']:
        manifest['files_sha256'][name] = packet.sha(target/name)
    (target/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    with pytest.raises(ValueError, match='source comment promotion'):
        packet.verify(target)
