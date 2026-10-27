"""Container orchestration contract tests; no estimator is simulated as real accuracy."""
import json
import subprocess
from pathlib import Path

import pytest

from ca import glim_run as runner

IMAGE_ID = "sha256:" + "a" * 64


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    bag = tmp_path/'input.mcap'; bag.write_bytes(b'fixture')
    config = tmp_path/'settings'; config.mkdir()
    (config/'config.json').write_text('{}')
    (config/'config_ros.json').write_text(json.dumps({'glim_ros':{'points_topic':'/points','imu_topic':'/raw'}}))
    out = tmp_path/'run'
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: IMAGE_ID + '\n')
    monkeypatch.setattr(runner, '_space', lambda *a: None)
    return bag, config, out


@pytest.mark.parametrize('mode,expected', [('success','process_succeeded'), ('missing','output_missing'),
    ('failed','process_failed'), ('timeout','timed_out'), ('changed','inputs_changed')])
def test_run_records_failure_and_success(inputs, monkeypatch, mode, expected):
    bag, config, out = inputs
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        if 'rm' in command:
            return subprocess.CompletedProcess(command, 0)
        assert '--network=none' in command
        assert 'auto_quit:=true' in command
        assert command[command.index('ros2') - 1] == IMAGE_ID
        kwargs['stdout'].write(b'process log\n')
        if mode == 'timeout':
            raise subprocess.TimeoutExpired(command, 1)
        if mode != 'missing':
            (out/'dump'/'traj_lidar.txt').write_text('1 0 0 0 0 0 0 1\n2 3 4 0 0 0 0 1\n')
        if mode == 'changed':
            bag.write_bytes(b'changed')
        return subprocess.CompletedProcess(command, 2 if mode == 'failed' else 0)
    monkeypatch.setattr(runner.subprocess, 'run', execute)
    result = runner.run_glim(str(bag), str(config), str(out), 'test:local')
    assert result['status'] == expected
    assert result['accuracy_verified'] is False
    if mode == 'success':
        assert result['trajectory_check']['pose_rows'] == 2
        assert result['trajectory_check']['path_length_m'] == 5
        assert result['trajectory_check']['sha256'] == result['trajectory']['sha256']
    assert json.loads((out/'glim-run.json').read_text()) == result
    assert (out/'glim.log').read_text() == 'process log\n'
    assert calls[-1][2:4] == ['rm', '-f']
    with pytest.raises(FileExistsError):
        runner.run_glim(str(bag), str(config), str(out), 'test:local')


def test_rejects_symlink_and_nested_outputs(inputs):
    bag, config, out = inputs
    (config/'escape.json').symlink_to(bag)
    with pytest.raises(ValueError, match='symlink'):
        runner.run_glim(str(bag), str(config), str(out), 'test')
    (config/'escape.json').unlink()
    with pytest.raises(ValueError, match='outside'):
        runner.run_glim(str(bag), str(config), str(config/'run'), 'test')
    assert not out.exists()


@pytest.mark.parametrize('status,code', [('process_succeeded', 0), ('output_missing', 1), ('output_invalid', 1)])
def test_cli_exit_status(inputs, monkeypatch, status, code):
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    bag, config, out = inputs
    monkeypatch.setattr(runner, 'run_glim', lambda *a, **kw: {'status': status})
    result = CliRunner().invoke(app, ['mapping-run-glim', str(bag), '--config', str(config),
                                     '--out', str(out), '--image', 'test:local'])
    assert result.exit_code == code
    assert json.loads(result.stdout)['status'] == status


@pytest.mark.parametrize('contents', [
    'invalid trajectory',
    '1 0 0 0 0 0 0 1\n',
    '1 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n',
    '2 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n',
    '1 0 0 0 0 0 0 1\n2 nan 0 0 0 0 0 1\n',
    '1 0 0 0 0 0 0 1\n2 1 0 0 0 0 0 0\n',
    '1 0 0 0 0 0 0 1\n2 1 0 0 0 0 0 2\n',
    '1 -1e308 0 0 0 0 0 1\n2 1e308 0 0 0 0 0 1\n',
])
def test_zero_exit_with_invalid_output_is_not_success(inputs, monkeypatch, contents):
    bag, config, out = inputs
    def execute(command, **kwargs):
        if 'run' in command:
            (out/'dump'/'traj_lidar.txt').write_text(contents)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(runner.subprocess, 'run', execute)
    result = runner.run_glim(str(bag), str(config), str(out), 'test')
    assert result['exit_code'] == 0
    assert result['status'] == 'output_invalid'
    assert result['trajectory_error']
    assert result['trajectory_check'] is None
    assert json.loads((out/'glim-run.json').read_text()) == result


def test_oversized_trajectory_is_rejected_before_parsing(inputs, monkeypatch):
    from ca.mapping_import import MAX_TRAJECTORY_BYTES
    bag, config, out = inputs
    def execute(command, **kwargs):
        if 'run' in command:
            with (out/'dump'/'traj_lidar.txt').open('wb') as f:
                f.truncate(MAX_TRAJECTORY_BYTES + 1)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(runner.subprocess, 'run', execute)
    result = runner.run_glim(str(bag), str(config), str(out), 'test')
    assert result['status'] == 'output_invalid'
    assert 'size limit' in result['trajectory_error']


def test_output_mutation_between_validation_and_hashing_is_rejected(inputs, monkeypatch):
    bag, config, out = inputs
    original = runner.inspect_tum_trajectory
    def validate(path):
        result = original(path)
        path.write_text('changed')
        return result
    def execute(command, **kwargs):
        if 'run' in command:
            (out/'dump'/'traj_lidar.txt').write_text('1 0 0 0 0 0 0 1\n2 3 4 0 0 0 0 1\n')
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(runner, 'inspect_tum_trajectory', validate)
    monkeypatch.setattr(runner.subprocess, 'run', execute)
    result = runner.run_glim(str(bag), str(config), str(out), 'test')
    assert result['status'] == 'output_invalid'
    assert 'changed during validation' in result['trajectory_error']


def test_invalid_partial_output_does_not_mask_process_failure(inputs, monkeypatch):
    bag, config, out = inputs
    def execute(command, **kwargs):
        if 'run' in command:
            (out/'dump'/'traj_lidar.txt').write_text('partial')
        return subprocess.CompletedProcess(command, 2)
    monkeypatch.setattr(runner.subprocess, 'run', execute)
    result = runner.run_glim(str(bag), str(config), str(out), 'test')
    assert result['status'] == 'process_failed'
    assert result['exit_code'] == 2
    assert result['trajectory_error']


def test_real_demo_preflight_blocks_before_any_docker_call(inputs,monkeypatch):
    pytest.importorskip('rosbags')
    _,config,out=inputs
    (config/'config_ros.json').write_text(json.dumps({'glim_ros':{'points_topic':'/velodyne_points','imu_topic':'/imu'}}))
    bag=Path(__file__).resolve().parents[2]/'web/public/samples/nclt-2012-04-29.mcap'
    def forbidden(*args,**kwargs): pytest.fail('Docker must not be contacted for blocked input')
    monkeypatch.setattr(runner.subprocess,'check_output',forbidden)
    monkeypatch.setattr(runner.subprocess,'run',forbidden)
    result=runner.run_glim(str(bag),str(config),str(out),'missing:image',points_topic='/velodyne_points',imu_topic='/imu')
    assert result['status']=='input_blocked'
    assert result['input_preflight']['imu_unavailable']['angular_velocity']>0
    assert result['command']==[] and result['exit_code'] is None
    assert result['image_id'] is None
    assert not (out/'dump').exists()
    assert json.loads((out/'glim-run.json').read_text())==result


def test_passing_preflight_is_retained_with_exact_input_and_policy(inputs,monkeypatch):
    bag,config,out=inputs
    def inspect(path,points,imu,*,max_imu_gap_s):
        assert (points,imu,max_imu_gap_s)==('/points','/raw',.02)
        return {'status':'basic_checks_passed','input':runner._artifact(Path(path)), 'policy':max_imu_gap_s}
    def execute(command,**kwargs):
        if 'run' in command:
            (out/'dump'/'traj_lidar.txt').write_text('1 0 0 0 0 0 0 1\n2 1 0 0 0 0 0 1\n')
        return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(runner,'inspect_glim_bag',inspect)
    monkeypatch.setattr(runner.subprocess,'run',execute)
    result=runner.run_glim(str(bag),str(config),str(out),'test',points_topic='/points',imu_topic='/raw',max_imu_gap_s=.02)
    assert result['status']=='process_succeeded'
    assert result['input_preflight_status']=='basic_checks_passed'
    assert result['input_preflight']['input']['sha256']==result['inputs']['bag'][0]['sha256']


def test_changed_input_during_preflight_never_starts_docker(inputs,monkeypatch):
    bag,config,out=inputs
    def inspect(*args,**kwargs):
        identity=runner._artifact(bag)
        bag.write_bytes(b'changed')
        return {'status':'basic_checks_passed','input':identity}
    monkeypatch.setattr(runner,'inspect_glim_bag',inspect)
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**kw: pytest.fail('Docker contacted'))
    result=runner.run_glim(str(bag),str(config),str(out),'test',points_topic='/points',imu_topic='/imu')
    assert result['status']=='inputs_changed'
    assert result['command']==[]


def test_unavailable_runtime_keeps_failure_receipt(inputs,monkeypatch):
    bag,config,out=inputs
    def missing(*args,**kwargs): raise FileNotFoundError('docker not installed')
    monkeypatch.setattr(runner.subprocess,'check_output',missing)
    result=runner.run_glim(str(bag),str(config),str(out),'test')
    assert result['status']=='runtime_unavailable'
    assert result['input_preflight_status']=='not_requested'
    assert json.loads((out/'glim-run.json').read_text())==result


def test_partial_topic_selection_rejected_before_output(inputs):
    bag,config,out=inputs
    with pytest.raises(ValueError,match='both'):
        runner.run_glim(str(bag),str(config),str(out),'test',points_topic='/points')
    assert not out.exists()


def test_cli_propagates_preflight_options(inputs,monkeypatch):
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    bag,config,out=inputs
    def run(*args,**kwargs):
        assert kwargs=={'points_topic':'/points','imu_topic':'/raw','max_imu_gap_s':.02}
        return {'status':'input_blocked'}
    monkeypatch.setattr(runner,'run_glim',run)
    result=CliRunner().invoke(app,['mapping-run-glim',str(bag),'--config',str(config),'--out',str(out),
        '--image','test','--points-topic','/points','--imu-topic','/raw','--max-imu-gap','.02'])
    assert result.exit_code==1
    assert json.loads(result.stdout)['status']=='input_blocked'


def test_config_topic_mismatch_blocks_before_docker(inputs,monkeypatch):
    bag,config,out=inputs
    monkeypatch.setattr(runner,'inspect_glim_bag',lambda *a,**kw:{'status':'basic_checks_passed','input':runner._artifact(bag)})
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *a,**kw:pytest.fail('Docker contacted'))
    result=runner.run_glim(str(bag),str(config),str(out),'test',points_topic='/different',imu_topic='/raw')
    assert result['status']=='config_blocked'
    assert 'differ' in result['config_topics']['error']
    assert json.loads((out/'glim-run.json').read_text())==result


def test_config_change_during_image_lookup_stops_before_container(inputs,monkeypatch):
    bag,config,out=inputs
    def lookup(*args,**kwargs):
        (config/'config_ros.json').write_text('{}')
        return IMAGE_ID
    monkeypatch.setattr(runner.subprocess,'check_output',lookup)
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**kw:pytest.fail('container started'))
    result=runner.run_glim(str(bag),str(config),str(out),'test')
    assert result['status']=='inputs_changed'
    assert not (out/'dump').exists()


@pytest.mark.parametrize('interrupted',[False,True])
def test_configuration_copy_failure_is_terminal_and_does_not_launch(inputs,monkeypatch,interrupted):
    bag,config,out=inputs
    def copy(*args,**kwargs):
        (out/'config').mkdir()
        (out/'config'/'partial').write_bytes(b'partial copy')
        if interrupted: raise KeyboardInterrupt()
        raise OSError('copy failed')
    monkeypatch.setattr(runner.shutil,'copytree',copy)
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**kw:pytest.fail('container launched'))
    result=runner.run_glim(str(bag),str(config),str(out),'test')
    assert result['status']==('interrupted' if interrupted else 'preparation_failed')
    assert result['failure_stage']=='configuration_copy'
    assert result['command']==[]
    assert result['image_id']==IMAGE_ID
    assert json.loads((out/'glim-run.json').read_text())==result


@pytest.mark.parametrize('interrupted',[False,True])
def test_image_lookup_timeout_or_interrupt_keeps_terminal_report(inputs,monkeypatch,interrupted):
    bag,config,out=inputs
    def lookup(command,**kwargs):
        assert kwargs['timeout']==30
        if interrupted: raise KeyboardInterrupt()
        raise subprocess.TimeoutExpired(command,kwargs['timeout'])
    monkeypatch.setattr(runner.subprocess,'check_output',lookup)
    result=runner.run_glim(str(bag),str(config),str(out),'test')
    assert result['status']==('interrupted' if interrupted else 'runtime_unavailable')
    assert result['failure_stage']=='image_lookup'
    assert not (out/'config').exists()
    assert json.loads((out/'glim-run.json').read_text())==result


@pytest.mark.parametrize('stage',['container_cleanup','output_verification'])
def test_late_interrupt_is_saved_instead_of_leaving_running_status(inputs,monkeypatch,stage):
    bag,config,out=inputs
    def execute(command,**kwargs):
        if 'run' in command:
            (out/'dump'/'traj_lidar.txt').write_text('1 0 0 0 0 0 0 1\n2 3 4 0 0 0 0 1\n')
        elif stage=='container_cleanup': raise KeyboardInterrupt()
        return subprocess.CompletedProcess(command,0)
    def interrupted(*args,**kwargs): raise KeyboardInterrupt()
    monkeypatch.setattr(runner.subprocess,'run',execute)
    if stage=='output_verification': monkeypatch.setattr(runner,'inspect_tum_trajectory',interrupted)
    result=runner.run_glim(str(bag),str(config),str(out),'test')
    assert result['status']=='interrupted'
    assert result['failure_stage']==stage
    assert json.loads((out/'glim-run.json').read_text())==result


def test_receipt_replacement_failure_keeps_previous_json(inputs,monkeypatch):
    bag,config,out=inputs
    from ca import mapping_job as jobs
    out.mkdir();receipt=out/'glim-run.json'
    receipt.write_text('{"status":"running"}')
    def fail(*args):raise OSError('replacement failed')
    monkeypatch.setattr(jobs.os,'replace',fail)
    with pytest.raises(OSError,match='replacement failed'):
        runner._save(receipt,{'status':'process_failed'})
    assert json.loads(receipt.read_text())=={'status':'running'}


@pytest.mark.parametrize('identity', [
    '', 'test:latest', 'sha256:', 'sha256:fixture',
    'sha256:' + 'a' * 63, 'sha256:' + 'a' * 65,
    'sha256:' + 'g' * 64, 'sha256:' + 'A' * 64,
    IMAGE_ID + '\n' + IMAGE_ID, IMAGE_ID + ' extra',
])
def test_invalid_image_identity_stops_before_copy_or_launch(inputs, monkeypatch, identity):
    bag, config, out = inputs
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: identity)
    monkeypatch.setattr(runner.shutil, 'copytree', lambda *a, **k: pytest.fail('configuration copied'))
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: pytest.fail('container launched'))
    result = runner.run_glim(str(bag), str(config), str(out), 'test:local')
    assert result['status'] == 'runtime_unavailable'
    assert result['failure_stage'] == 'image_lookup'
    assert result['image_id'] is None
    assert result['command'] == []
    assert 'immutable image ID' in result['error']
    assert not (out/'config').exists()
    assert not (out/'dump').exists()
    assert json.loads((out/'glim-run.json').read_text()) == result


def test_overflowing_config_number_blocks_before_docker(inputs, monkeypatch):
    bag, config, out = inputs
    (config/'config_ros.json').write_text(
        '{"glim_ros":{"points_topic":"/points","imu_topic":"/raw","offset":1e999}}')
    monkeypatch.setattr(runner, 'inspect_glim_bag', lambda *a, **k: {
        'status':'basic_checks_passed', 'input':runner._artifact(bag)})
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: pytest.fail('Docker contacted'))
    result = runner.run_glim(str(bag), str(config), str(out), 'test',
                             points_topic='/points', imu_topic='/raw')
    assert result['status'] == 'config_blocked'
    assert 'nonfinite JSON number' in result['config_topics']['error']
    assert result['command'] == []
    assert not (out/'config').exists()
    assert json.loads((out/'glim-run.json').read_text()) == result
