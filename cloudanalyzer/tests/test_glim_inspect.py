import json
from pathlib import Path
import shutil

import pytest
from typer.testing import CliRunner
from ca.glim_inspect import inspect_glim_run
from ca.glim_run import inventory
from ca.mapping_job import _artifact
from cloudanalyzer_cli.main import app


@pytest.fixture
def saved(tmp_path,monkeypatch):
    import subprocess
    def forbidden(*args,**kwargs): pytest.fail('inspection must not start processes')
    monkeypatch.setattr(subprocess,'run',forbidden)
    monkeypatch.setattr(subprocess,'check_output',forbidden)
    bag=tmp_path/'recording.mcap';bag.write_bytes(b'identity fixture')
    config=tmp_path/'settings';config.mkdir();(config/'config.json').write_text('{}')
    root=tmp_path/'run';root.mkdir();shutil.copytree(config,root/'config');(root/'dump').mkdir()
    trajectory=root/'dump'/'traj_lidar.txt'
    trajectory.write_text('1 0 0 0 0 0 0 1\n2 3 4 0 0 0 0 1\n')
    report={'schema':'cloudanalyzer.glim_run.v1','status':'process_succeeded','exit_code':0,
            'image_id':'sha256:'+'a'*64,'command':['not-executed'],
            'inputs':{'bag':inventory(bag),'configuration':inventory(config)},'trajectory':_artifact(trajectory)}
    (root/'glim-run.json').write_text(json.dumps(report))
    return root,bag,config


def test_unchanged_files_and_read_only_report(saved):
    root,_,_=saved
    before=(root/'glim-run.json').read_bytes()
    result=inspect_glim_run(str(root))
    assert result['status']=='recorded_files_match'
    assert result['recorded_execution_status']=='process_succeeded'
    assert result['verified_file_count']==4
    assert result['trajectory_check']['path_length_m']==5
    assert (root/'glim-run.json').read_bytes()==before


@pytest.mark.parametrize('change',['bag','original_config','copy','extra_copy','trajectory','missing','symlink'])
def test_changed_or_missing_dependencies_are_reported(saved,change):
    root,bag,config=saved
    if change=='bag':bag.write_bytes(b'changed')
    if change=='original_config':(config/'config.json').write_text('{"changed":true}')
    if change=='copy':(root/'config'/'config.json').write_text('changed')
    if change=='extra_copy':(root/'config'/'extra.json').write_text('{}')
    if change=='trajectory':(root/'dump'/'traj_lidar.txt').write_text('partial')
    if change=='missing':bag.unlink()
    if change=='symlink':
        bag.unlink();bag.symlink_to(config/'config.json')
    result=inspect_glim_run(str(root))
    assert result['status']=='changed_or_missing'
    assert result['issues']
    assert result['accuracy_verified'] is False


def test_blocked_execution_is_not_relabelled_as_success(saved):
    root,_,_=saved
    path=root/'glim-run.json';report=json.loads(path.read_text())
    report.update(status='preparation_failed',exit_code=None,command=[],trajectory=None,
                  failure_stage='configuration_copy',error='copy failed')
    path.write_text(json.dumps(report))
    shutil.rmtree(root/'config');shutil.rmtree(root/'dump')
    result=inspect_glim_run(str(root))
    assert result['status']=='recorded_files_match'
    assert result['recorded_execution_status']=='preparation_failed'
    assert result['recorded_failure_stage']=='configuration_copy'
    assert result['recorded_error']=='copy failed'
    assert result['trajectory_check'] is None


def test_unsafe_copy_reference_rejected(saved):
    root,_,_=saved;path=root/'glim-run.json';report=json.loads(path.read_text())
    report['inputs']['configuration'][0]['relative_path']='../outside'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='relative path'):inspect_glim_run(str(root))


def test_cli_returns_nonzero_for_changed_files(saved):
    root,bag,_=saved
    good=CliRunner().invoke(app,['mapping-inspect-glim',str(root)])
    assert good.exit_code==0
    bag.write_bytes(b'changed')
    bad=CliRunner().invoke(app,['mapping-inspect-glim',str(root)])
    assert bad.exit_code==1
    assert json.loads(bad.stdout)['status']=='changed_or_missing'
