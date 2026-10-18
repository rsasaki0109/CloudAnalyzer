"""Real native-step checks and resigned semantic tamper controls."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT/'benchmarks/slam/flatwall-balance-dynamics'
spec = importlib.util.spec_from_file_location('balance_dynamics_packet',PACKET/'verify.py')
packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def copy_packet(tmp_path):
    target = tmp_path/'checkout/benchmarks/slam/flatwall-balance-dynamics'
    shutil.copytree(PACKET,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    manifest = packet.read(PACKET/'manifest.json')
    for name in manifest['dependencies'] | manifest['normalized_native_sources_sha256'] | manifest['normalized_checkout_sources_sha256']:
        destination = target.parents[2]/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    return target


def resign(target):
    summary = packet.read(target/'summary.json')
    for trace in summary['traces']:
        trace['sha256'] = packet.sha(target/trace['file'])
    save(target/'summary.json',summary)
    for stage in ['analysis-01','analysis-02']:
        receipt = packet.read(target/(stage+'-receipt.json'))
        for name in receipt['output_files_sha256']:
            mapped = stage+'-invocation.json' if name == 'invocation.json' else name
            if stage == 'analysis-01' and name in ['summary.json','trace-runs.npz','time-runs.npz']:
                mapped = 'native-reported-'+name
            receipt['output_files_sha256'][name] = packet.sha(target/mapped)
        save(target/(stage+'-receipt.json'),receipt)
    manifest = packet.read(target/'manifest.json')
    for name in manifest['files_sha256']:
        manifest['files_sha256'][name] = packet.sha(target/name)
    save(target/'manifest.json',manifest)


def test_all_real_native_steps_cohorts_and_counterbalanced_measurements():
    result = packet.verify(PACKET)
    assert result['native_steps_checked'] == 4000
    assert result['tail_recurrence_periods'] == [2,2,4,3,2,2,4,6]
    assert result['trace_replay']['frames_checked'] == 5120
    assert result['time_replay']['frames_checked'] == 20480
    assert result['trace_replay']['exact_pose_bit_frames'] == 5120
    assert result['time_replay']['exact_pose_bit_frames'] == 20480
    assert result['trace_replay']['maximum_pose_abs_error'] == 0
    assert result['measured_case_pairs'] == 48
    assert result['candidate_changed'] is result['defaults_changed'] is False


@pytest.mark.parametrize('change',['matrix','missing_step','final_target','omitted_frame'])
def test_resigned_trace_semantic_changes_are_rejected(tmp_path,change):
    target = copy_packet(tmp_path)
    summary = packet.read(target/'summary.json')
    if change == 'omitted_frame':
        summary['traces'].pop(0)
        save(target/'summary.json',summary)
    else:
        path = target/summary['traces'][0]['file']
        arrays = packet.load(path)
        if change == 'matrix':
            arrays['normal_matrix'][0,0,0] *= 1.1
        elif change == 'missing_step':
            arrays['before'] = arrays['before'][:-1]
        else:
            arrays['final_target_world'][0,0] += .25
        np.savez_compressed(path,**arrays)
    resign(target)
    with pytest.raises((AssertionError,ValueError)):
        packet.verify(target)


def test_timing_summary_cannot_replace_a_measured_sample_with_a_faster_number():
    data = packet.load(PACKET/'time-runs.npz')
    summary = packet.read(PACKET/'summary.json')['timing']
    summary['cases'][0]['measured_pairs'][0]['balanced_ns'] //= 2
    with pytest.raises((AssertionError,ValueError)):
        packet.check_timing(data,summary,packet.read(PACKET/'cohort.json'))


def test_warmup_or_counterbalanced_order_cannot_be_relabeled():
    data = packet.load(PACKET/'time-runs.npz')
    data['warmup'][0] = False
    with pytest.raises(ValueError,match='order|warmup'):
        packet.check_runs(data,packet.read(PACKET/'cohort.json'),ROOT,'time')


def test_native_timing_sum_cannot_omit_one_scan():
    data = packet.load(PACKET/'time-runs.npz')
    data['registration_ns'][0] = 0
    with pytest.raises(ValueError,match='timing'):
        packet.check_runs(data,packet.read(PACKET/'cohort.json'),ROOT,'time')


def test_changed_executed_source_is_detected_even_after_resigning_packet_files(tmp_path):
    target = copy_packet(tmp_path)
    path = target/'executed-source/audit_odometry_steps.rs'
    with path.open('ab') as stream:
        stream.write(b'\n// changed executed source\n')
    resign(target)
    with pytest.raises(ValueError,match='normalized source'):
        packet.verify(target)
