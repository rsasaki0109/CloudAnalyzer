import json
import sys
from pathlib import Path

import pytest

from ca.mapping_measure import measure, output_size
from ca.mapping_scorecard import scorecard, compare


def command(root, exit_code=0):
    return [sys.executable, '-c', '''import sys,json,time
from pathlib import Path
p=Path(sys.argv[1]);p.mkdir()
(p/'run.json').write_text(json.dumps({'schema':'cloudanalyzer.mapping_run.v1','layout_file':{'sha256':'b'*64},'history':[]}))
(p/'job.json').write_text(json.dumps({'source':{'sha256':'a'*64}}))
(p/'output.bin').write_bytes(b'x'*1024)
data=bytearray(8*1024*1024)
time.sleep(.1)
sys.exit(int(sys.argv[2]))
''', str(root), str(exit_code)]


def test_measured_process_metrics_bind_to_final_run_and_job(tmp_path):
    root = tmp_path/'run'
    result = measure(command(root), str(root), .01)
    assert result['status'] == 'process_succeeded'
    assert result['wall_time_seconds'] > 0
    assert result['rss_samples'] > 0
    assert result['sampled_peak_tree_rss_bytes'] > 0
    assert result['logical_output_bytes'] >= 1024
    assert json.loads((root/'measurement.json').read_text()) == result
    card = scorecard(str(root))
    assert card['wall_time_seconds'] == result['wall_time_seconds']
    assert compare(card, card)['delta']['sampled_peak_tree_rss_bytes'] == 0
    assert 'wall_time_seconds' not in compare(card, {**card, 'status': 'different_generation_stage'})['delta']
    other = {**card, 'sample_interval_seconds': .1}
    assert 'wall_time_seconds' not in compare(card, other)['delta']
    (root/'run.json').write_text((root/'run.json').read_text()+'\n')
    stale = scorecard(str(root))
    assert stale['measurement_status'] == 'stale_or_unsupported'
    assert stale['wall_time_seconds'] is None


def test_failed_command_keeps_evidence_without_claiming_success_metrics(tmp_path):
    root=tmp_path/'failed'
    result=measure(command(root, 3),str(root),.01)
    assert result['status']=='process_failed'
    assert result['exit_code']==3
    assert scorecard(str(root))['wall_time_seconds'] is None
    before=(root/'measurement.json').read_bytes()
    with pytest.raises(FileExistsError):
        measure(command(root),str(root))
    assert (root/'measurement.json').read_bytes()==before


def test_size_ignores_external_symlinks_and_launch_failure_is_recorded(tmp_path):
    root=tmp_path/'size';root.mkdir()
    outside=tmp_path/'external';outside.write_bytes(b'x'*4096)
    (root/'link').symlink_to(outside)
    assert output_size(root)==(0,None)
    result=measure([str(tmp_path/'missing-executable')],str(tmp_path/'failed'))
    assert result['status']=='launch_failed'
    assert result['sampled_peak_tree_rss_bytes'] is None
    assert result['run_sha256'] is None


def test_invalid_sampling_protocol_keeps_performance_unavailable(tmp_path):
    root=tmp_path/'run'
    result=measure(command(root),str(root),.01)
    result['sample_interval_seconds']=-1
    (root/'measurement.json').write_text(json.dumps(result))
    assert scorecard(str(root))['wall_time_seconds'] is None


@pytest.mark.parametrize('kind', ['missing', 'file', 'symlink'])
def test_unavailable_output_root_has_no_size(tmp_path, kind):
    root = tmp_path/'output'
    if kind == 'file':
        root.write_bytes(b'contents')
    elif kind == 'symlink':
        target = tmp_path/'actual'; target.mkdir(); root.symlink_to(target)
    size, error = output_size(root)
    assert size is None
    assert error


def test_unreadable_subdirectory_does_not_report_partial_size(tmp_path, monkeypatch):
    from ca import mapping_measure
    root = tmp_path/'output'; root.mkdir(); (root/'visible').write_bytes(b'visible')
    def walk(path, *, followlinks, onerror):
        yield str(root), [], ['visible']
        onerror(PermissionError('unreadable child'))
    monkeypatch.setattr(mapping_measure.os, 'walk', walk)
    size, error = output_size(root)
    assert size is None
    assert 'unreadable child' in error


def test_empty_directories_count_toward_inventory_limit(tmp_path, monkeypatch):
    from ca import mapping_measure
    monkeypatch.setattr(mapping_measure, 'MAX_OUTPUT_ENTRIES', 2)
    for name in ('a', 'b', 'c'):
        (tmp_path/name).mkdir()
    size, error = output_size(tmp_path)
    assert size is None
    assert 'entry count' in error


def test_special_output_file_is_unavailable(tmp_path):
    import os
    os.mkfifo(tmp_path/'pipe')
    size, error = output_size(tmp_path)
    assert size is None
    assert 'nonregular' in error


def test_size_error_retains_other_metrics_but_not_comparison_size(tmp_path, monkeypatch):
    from ca import mapping_measure
    root = tmp_path/'run'
    monkeypatch.setattr(mapping_measure, 'output_size', lambda path: (None, 'unreadable child'))
    result = measure(command(root), str(root), .01)
    assert result['status'] == 'process_succeeded'
    assert result['logical_output_bytes'] is None
    assert result['output_size_error'] == 'unreadable child'
    card = scorecard(str(root))
    assert card['logical_output_bytes'] is None
    assert card['wall_time_seconds'] > 0
    assert compare(card, card)['delta']['logical_output_bytes'] is None


@pytest.mark.parametrize('interrupt', [False, True])
def test_monitor_failure_stops_real_child_and_saves_receipt(tmp_path, monkeypatch, interrupt):
    import psutil
    from ca import mapping_measure
    started = []
    popen = mapping_measure.subprocess.Popen
    def launch(*args, **kwargs):
        process = popen(*args, **kwargs)
        started.append(process)
        return process
    def unavailable(pid):
        if interrupt:
            raise KeyboardInterrupt()
        raise psutil.AccessDenied(pid)
    monkeypatch.setattr(mapping_measure.subprocess, 'Popen', launch)
    monkeypatch.setattr(psutil, 'Process', unavailable)
    root = tmp_path/'monitor-failed'
    try:
        result = measure([sys.executable, '-c', 'import time; time.sleep(30)'], str(root))
        assert result['status'] == ('interrupted' if interrupt else 'measurement_failed')
        assert started[0].poll() is not None
        assert result['cleanup_error'] is None
        assert result['exit_code'] is not None
        assert json.loads((root/'measurement.json').read_text()) == result
    finally:
        for process in started:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


@pytest.mark.parametrize('mode', ['forced', 'still_running', 'denied', 'interrupted'])
def test_cleanup_waits_are_bounded_and_failures_explicit(monkeypatch, mode):
    from ca import mapping_measure
    from types import SimpleNamespace
    import subprocess
    waits, signals = [], []
    def wait(*, timeout):
        waits.append(timeout)
        if mode == 'interrupted':
            raise KeyboardInterrupt()
        if len(waits) == 1 or mode == 'still_running':
            raise subprocess.TimeoutExpired('fixture', timeout)
        return -9
    def send(*args):
        signals.append(args)
        if mode == 'denied':
            raise PermissionError('fixture signal denied')
    process = SimpleNamespace(pid=123, returncode=None, wait=wait, terminate=send, kill=send)
    monkeypatch.setattr(mapping_measure.os, 'killpg', send, raising=False)
    code, error = mapping_measure._stop_process(process)
    assert all(timeout == 5 for timeout in waits)
    assert len(waits) <= 2
    if mode == 'forced':
        assert code == -9 and error is None
        assert len(signals) == 2
    else:
        assert code is None and error
