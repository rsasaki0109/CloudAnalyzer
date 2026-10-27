"""Synthetic fixtures follow GLIM's documented global LiDAR TUM export contract."""
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ca import glim_import as adapter, mapping_import, mapping_run as runs
from cloudanalyzer_cli.main import app
from tests.test_mapping_import import external_pair
from tests.test_mapping_job import job_backend, _attach_lane_native, _run_layout


@pytest.fixture
def glim_pair(external_pair, tmp_path):
    cloud, trajectory, provenance = external_pair
    dump = tmp_path / 'dump'; dump.mkdir()
    (dump / 'traj_lidar.txt').write_bytes(trajectory.read_bytes())
    provenance['backend'] = 'GLIM'
    provenance['version'] = 'synthetic-export-contract-fixture'
    return dump, cloud, provenance


def prepare(pair, path):
    dump, cloud, declaration = pair
    return adapter.prepare_glim_export(str(dump), str(cloud), str(path), declaration)


def test_preparation_preserves_bytes_and_keeps_explicit_frame_and_provenance(glim_pair, tmp_path):
    dump, cloud, original = glim_pair
    cloud_bytes = cloud.read_bytes()
    raw = (dump/'traj_lidar.txt').read_bytes().replace(b'\n', b'\r\n')
    (dump/'traj_lidar.txt').write_bytes(raw)
    out = tmp_path/'prepared'; result=prepare(glim_pair,out)
    assert (out/'trajectory.tum').read_bytes()==raw
    assert cloud.read_bytes()==cloud_bytes
    assert result['map']==str(cloud)
    assert result['preflight']['map_points']==4
    assert result['preflight']['frame_alignment_verified'] is False
    provenance=json.loads((out/'provenance.json').read_text())
    assert provenance['configuration'][adapter.KEY]['inputs']['trajectory']['sha256']==result['inputs']['trajectory']['sha256']
    assert adapter.KEY not in original['configuration']
    with pytest.raises(FileExistsError): prepare(glim_pair,out)


@pytest.mark.parametrize('alternative',['traj_imu.txt','odom_lidar.txt','odom_imu.txt'])
def test_no_silent_sensor_or_odometry_fallback(glim_pair,tmp_path,alternative):
    dump,_,_=glim_pair
    (dump/'traj_lidar.txt').rename(dump/alternative)
    with pytest.raises(ValueError,match='missing'):
        prepare(glim_pair,tmp_path/'prepared')
    assert not (tmp_path/'prepared').exists()


def test_invalid_timestamps_and_wrong_backend_leave_no_prepared_output(glim_pair,tmp_path):
    dump,_,p=glim_pair
    p['backend']='another estimator'
    with pytest.raises(ValueError,match='GLIM'): prepare(glim_pair,tmp_path/'a')
    p['backend']='GLIM'
    (dump/'traj_lidar.txt').write_text('100 0 0 1 0 0 0 1\n100 10 0 1 0 0 0 1\n')
    with pytest.raises(ValueError,match='increase strictly'): prepare(glim_pair,tmp_path/'b')
    assert not (tmp_path/'a').exists() and not (tmp_path/'b').exists()


def test_changed_map_or_prepared_trajectory_cannot_enter_import(glim_pair,tmp_path):
    result=prepare(glim_pair,tmp_path/'prepared')
    declaration=json.loads(Path(result['provenance']).read_text())
    Path(result['trajectory']).write_text('100 0 0 1 0 0 0 1\n101 11 0 1 0 0 0 1\n')
    with pytest.raises(ValueError,match='changed after preparation'):
        mapping_import.import_mapping_job(result['map'],result['trajectory'],str(tmp_path/'run'),declaration)
    assert not (tmp_path/'run').exists()
    Path(result['trajectory']).write_bytes((glim_pair[0]/'traj_lidar.txt').read_bytes())
    glim_pair[1].write_bytes(glim_pair[1].read_bytes()+b'\n')
    with pytest.raises(ValueError,match='changed after preparation'):
        mapping_import.import_mapping_job(result['map'],result['trajectory'],str(tmp_path/'run'),declaration)


def test_prepared_glim_export_enters_drafting_and_finish_without_estimator_execution(glim_pair,execution_report,tmp_path):
    _attach_lane_native()
    import struct
    glim_pair[1].write_bytes(b'ply\nformat binary_little_endian 1.0\nelement vertex 4\nproperty double x\nproperty double y\nproperty double z\nend_header\n' +
                             struct.pack('<12d',0,-1,1,0,1,1,10,-1,1,10,1,1))
    result=adapter.prepare_glim_export(str(glim_pair[0]),str(glim_pair[1]),str(tmp_path/'prepared'),glim_pair[2],run_report=str(execution_report[0]))
    root=tmp_path/'run'
    run=runs.start_mapping_run_from_slam(result['map'],result['trajectory'],str(root),json.loads(Path(result['provenance']).read_text()),_run_layout())
    run=runs.advance_mapping_run(str(root),{'type':'inspect','candidate_ids':[1]},'Inspect fixture source',run['revision'])
    run=runs.advance_mapping_run(str(root),{'type':'draft','decisions':[{'candidate_id':1,'action':'include','reason':'Keep inspected fixture source'}]},'Draft explicit layout',run['revision'])
    assert run['draft_result']['status']=='audited_draft'
    run=runs.advance_mapping_run(str(root),{'type':'finish','candidate_id':2},'Retain unresolved holds',run['revision'])
    assert run['output']['status']=='draft_needs_review'
    assert run['output']['deployment_ready'] is False

    # The delivered review must retain exact producer evidence without its workspace.
    import zipfile
    from ca.mapping_bundle import export_mapping_run, export_mapping_preview, inspect_mapping_bundle
    expected = {'source_trajectory': Path(result['trajectory']).read_bytes(),
                'slam_import': (root/'slam-import.json').read_bytes(),
                'glim_execution_report': (tmp_path/'prepared'/'glim-run.json').read_bytes()}
    archives = [tmp_path/'review.zip', tmp_path/'preview.zip']
    export_mapping_run(str(root),str(archives[0]),'Synthetic GLIM contract fixture')
    export_mapping_preview(str(root),str(archives[1]),'Synthetic GLIM contract fixture',max_preview_points=2)
    for archive_path in archives:
        with zipfile.ZipFile(archive_path) as archive:
            manifest=json.loads(archive.read('manifest.json'))
            for role,data in expected.items():
                assert archive.read(manifest['roles'][role]) == data
    # No source paths are dereferenced by the portable inspector.
    (tmp_path/'prepared'/'glim-run.json').unlink()
    (tmp_path/'prepared'/'trajectory.tum').unlink()
    (root/'slam-import.json').unlink()
    for archive_path in archives:
        assert inspect_mapping_bundle(str(archive_path))['integrity_verified']
    with pytest.raises((ValueError,OSError)):
        export_mapping_run(str(root),str(tmp_path/'missing-evidence.zip'),'Synthetic fixture')
    assert not (tmp_path/'missing-evidence.zip').exists()


def test_cli_prepares_documented_glim_export(glim_pair,tmp_path):
    dump,cloud,p=glim_pair
    provenance=tmp_path/'producer.json';provenance.write_text(json.dumps(p))
    result=CliRunner().invoke(app,['mapping-prepare-glim',str(dump),'--map',str(cloud),'--provenance',str(provenance),'--out',str(tmp_path/'prepared')])
    assert result.exit_code==0,result.output
    assert json.loads(result.output)['trajectory_kind']=='global_lidar'


@pytest.fixture
def execution_report(glim_pair, tmp_path):
    from ca.mapping_job import _artifact
    dump, _, _ = glim_pair
    identity = _artifact(dump/'traj_lidar.txt')
    value = {'schema': 'cloudanalyzer.glim_run.v1', 'status': 'process_succeeded',
             'exit_code': 0, 'image_id': 'sha256:' + 'a'*64,
             'inputs_unchanged': True, 'configuration_copy_matches': True,
             'trajectory': identity, 'trajectory_check': {**identity, 'pose_rows': 2}}
    path = tmp_path/'glim-run-original.json'
    path.write_text(json.dumps(value))
    return path, value


def test_execution_report_is_preserved_and_bound_to_export(glim_pair, execution_report, tmp_path):
    dump, cloud, declaration = glim_pair
    report, _ = execution_report
    original = report.read_bytes()
    out = tmp_path/'linked'
    result = adapter.prepare_glim_export(str(dump),str(cloud),str(out),declaration,run_report=str(report))
    assert (out/'glim-run.json').read_bytes() == original
    assert result['execution']['image_id'] == 'sha256:' + 'a'*64
    assert result['execution']['trajectory_sha256'] == result['inputs']['trajectory']['sha256']
    provenance=json.loads(Path(result['provenance']).read_text())
    assert provenance['configuration'][adapter.KEY]['execution'] == result['execution']
    # The copied snapshot is independent of subsequent changes to the original report.
    report.write_text('changed original')
    assert (out/'glim-run.json').read_bytes() == original
    (out/'glim-run.json').write_text('changed snapshot')
    with pytest.raises(ValueError):
        mapping_import.import_mapping_job(result['map'],result['trajectory'],str(tmp_path/'run'),provenance)
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('mutation', ['failed','exit','bool_exit','changed_inputs','changed_config','image','trajectory','unchecked','schema'])
def test_inconsistent_execution_report_rejected_before_output(glim_pair, execution_report, tmp_path, mutation):
    dump, cloud, declaration = glim_pair
    path, report = execution_report
    if mutation == 'failed': report['status'] = 'output_invalid'
    if mutation == 'exit': report['exit_code'] = 2
    if mutation == 'bool_exit': report['exit_code'] = False
    if mutation == 'changed_inputs': report['inputs_unchanged'] = False
    if mutation == 'changed_config': report['configuration_copy_matches'] = False
    if mutation == 'image': report['image_id'] = 'mutable:latest'
    if mutation == 'trajectory': report['trajectory']['sha256'] = '0'*64
    if mutation == 'unchecked': report.pop('trajectory_check')
    if mutation == 'schema': report['schema'] = 'unknown'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='GLIM'):
        adapter.prepare_glim_export(str(dump),str(cloud),str(tmp_path/'linked'),declaration,run_report=str(path))
    assert not (tmp_path/'linked').exists()


def test_cli_binds_execution_report(glim_pair,execution_report,tmp_path):
    dump,cloud,declaration=glim_pair
    producer=tmp_path/'producer.json';producer.write_text(json.dumps(declaration))
    result=CliRunner().invoke(app,['mapping-prepare-glim',str(dump),'--map',str(cloud),
        '--provenance',str(producer),'--run-report',str(execution_report[0]),'--out',str(tmp_path/'linked')])
    assert result.exit_code==0,result.output
    assert json.loads(result.stdout)['execution']['report']['sha256']


def test_execution_link_enters_job_immutable_dependencies(glim_pair,execution_report,tmp_path):
    dump,cloud,declaration=glim_pair
    _attach_lane_native()
    result=adapter.prepare_glim_export(str(dump),str(cloud),str(tmp_path/'linked'),declaration,run_report=str(execution_report[0]))
    provenance=json.loads(Path(result['provenance']).read_text())
    job=mapping_import.import_mapping_job(result['map'],result['trajectory'],str(tmp_path/'job'),provenance)
    assert job['status']=='pointcloud_ready'
    recorded=job['pointcloud']['files']['glim_execution_report']
    assert recorded==result['execution']['report']
    assert json.loads((tmp_path/'job'/'slam-import.json').read_text())['inputs']['glim_execution_report']==recorded
    Path(recorded['path']).write_text('edited report')
    from ca.mapping_job import _verify
    with pytest.raises(ValueError,match='changed'):
        _verify(recorded)
