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
