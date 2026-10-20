"""Independent real trajectory checks and resigned semantic tamper controls."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
PACKET=ROOT/'benchmarks/slam/flatwall-target-balance'
spec=importlib.util.spec_from_file_location('target_balance_packet',PACKET/'verify.py')
packet=importlib.util.module_from_spec(spec);spec.loader.exec_module(packet)


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def copy_packet(tmp_path):
    target=tmp_path/'checkout/benchmarks/slam/flatwall-target-balance'
    shutil.copytree(PACKET,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    manifest=packet.read(PACKET/'manifest.json')
    for name in manifest['dependencies']|manifest['checkout_sources_sha256']|manifest['normalized_native_sources_sha256']|dict.fromkeys(manifest.get('native_source_aliases',{}).values()):
        destination=target.parents[2]/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    return target


def resign(target):
    catalog=packet.read(target/'snapshot-catalog.json')
    for row in catalog:
        for name,item in row['files'].items():
            if (target/name).is_file():item.update(sha256=packet.sha(target/name),bytes=(target/name).stat().st_size)
    save(target/'snapshot-catalog.json',catalog)
    closure=packet.read(target/'estimates-receipt.json')
    for name in closure['output_files_sha256']:
        if (target/name).is_file():closure['output_files_sha256'][name]=packet.sha(target/name)
    save(target/'estimates-receipt.json',closure)
    invocation=packet.read(target/'scoring-invocation-freeze.json')
    invocation['estimates_receipt_sha256']=packet.sha(target/'estimates-receipt.json')
    save(target/'scoring-invocation-freeze.json',invocation)
    boundary=packet.read(target/'scoring-boundary.json')
    prefix=next(k[:-len('receipt.json')] for k in boundary['files_sha256'] if k.endswith('/receipt.json'))
    boundary['estimates_receipt_sha256']=packet.sha(target/'estimates-receipt.json')
    boundary['files_sha256']={prefix+name:digest for name,digest in closure['output_files_sha256'].items()}
    boundary['files_sha256'][prefix+'receipt.json']=boundary['estimates_receipt_sha256']
    save(target/'scoring-boundary.json',boundary)
    score=packet.read(target/'scoring-receipt.json')
    score['estimates_receipt_sha256']=packet.sha(target/'estimates-receipt.json')
    for name in score['output_files_sha256']:score['output_files_sha256'][name]=packet.sha(target/name)
    save(target/'scoring-receipt.json',score)
    manifest=packet.read(target/'manifest.json')
    for name in manifest['files_sha256']:manifest['files_sha256'][name]=packet.sha(target/name)
    save(target/'manifest.json',manifest)


def test_all_sixteen_real_trajectories_controls_and_sample_systems():
    result=packet.verify(PACKET)
    assert (result['cases'],result['candidate_frames'],result['default_replay_pose_bits_checked'],result['public_raw_samples'])==(16,2560,2560,48)
    assert result['exact_reference_associations_checked'] is True
    assert result['defaults_changed'] is result['physical_provenance_certified'] is False
    assert packet.read(PACKET/'comparison.json')['summary']['position_improved']==11


@pytest.mark.parametrize('change',['weight','multiplicity','pose','time','default_control','source_identity','metric','edge','coverage','spectrum','boundary','comparison'])
def test_resigned_semantic_changes_are_rejected(tmp_path,change):
    target=copy_packet(tmp_path)
    case='flatwall_01/repeat-motion/raw-0.1875'
    if change in {'weight','multiplicity'}:
        path=target/case/'scan-093.npz'
        with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
        if change=='weight':arrays['weights'][0]*=2
        else:arrays['target_multiplicity'][0]+=1
        np.savez_compressed(path,**arrays)
    elif change in {'pose','time','default_control'}:
        path=target/case/('default-replay-trajectory.npz' if change=='default_control' else 'trajectory.npz')
        with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
        if change=='time':arrays['timestamps_ns'][60]+=1
        else:arrays['poses'][60,0,3]+=.5
        np.savez_compressed(path,**arrays)
    elif change in {'source_identity','spectrum'}:
        path=target/case/'frames.jsonl';frames=packet.rows(path)
        if change=='source_identity':frames[92]['original_snapshot_sha256']='0'*64
        else:frames[92]['geometry']['eigenvalues'][0]*=.5
        path.write_text('\n'.join(json.dumps(r) for r in frames)+'\n',encoding='utf-8')
    elif change in {'metric','edge'}:
        path=target/case/'metrics.json';result=packet.read(path)
        if change=='metric':result['metrics']['rigid_position_ape_m']['rmse']*=.75
        else:result['edges'][30]['right_estimate_index']+=1
        save(path,result)
        summary=packet.read(target/'summary.json');summary['cases'][case]=result;save(target/'summary.json',summary)
    elif change=='coverage':
        path=target/'flatwall_02/repeat-motion/raw-0.1875/coverage.json';coverage=packet.read(path)
        next(r for r in coverage if r['status']=='reference_gap')['status']='interpolated';save(path,coverage)
    elif change=='boundary':
        value=packet.read(target/'scoring-boundary.json');value['reference_poses_read_in_this_trial_before_boundary']=True;save(target/'scoring-boundary.json',value)
    else:
        value=packet.read(target/'comparison.json');value['cases'][0]['metrics']['rigid_position_ape_m']['balanced']=0.;save(target/'comparison.json',value)
    resign(target)
    with pytest.raises((ValueError,AssertionError)):packet.verify(target)


def test_changed_original_sample_bytes_are_detected(tmp_path):
    target=copy_packet(tmp_path)
    with (target/'flatwall_01/repeat-motion/raw-0.1875/scan-093.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='hash'):packet.verify(target)


def test_native_checkout_newlines_do_not_change_source_identity(tmp_path):
    target=copy_packet(tmp_path)
    manifest=packet.read(target/'manifest.json')
    aliases=manifest.get('native_source_aliases',{})
    for name in manifest['normalized_native_sources_sha256']:
        if name in aliases:continue  # Exact-byte archives are not development checkouts.
        path=target.parents[2]/name
        data=path.read_bytes().replace(b'\r\n',b'\n')
        path.write_bytes(data.replace(b'\n',b'\r\n'))
    assert packet.verify(target)['default_replay_pose_bits_checked']==2560


def test_native_source_content_changes_are_rejected(tmp_path):
    target=copy_packet(tmp_path)
    name='rust/crates/ca-core/src/odometry_target_balance.rs'
    alias=packet.read(target/'manifest.json').get('native_source_aliases',{}).get(name,name)
    path=target.parents[2]/alias
    with path.open('ab') as f:f.write(b'\n// changed source content\n')
    with pytest.raises(ValueError,match='hash|normalized native source'):packet.verify(target)


def test_archived_executed_native_source_is_still_hash_checked(tmp_path):
    target=copy_packet(tmp_path)
    alias=packet.read(target/'manifest.json')['native_source_aliases']['rust/crates/ca-core/src/odometry.rs']
    with (target.parents[2]/alias).open('ab') as stream:stream.write(b'\n// changed historical source\n')
    with pytest.raises(ValueError,match='hash'):packet.verify(target)


@pytest.mark.parametrize('name', [
    'rust/crates/ca-core/src/vector_map/corridors.rs',
    'rust/crates/ca-py/src/vector_map/corridor_lanes.rs',
    'rust/crates/ca-py/src/lib.rs',
    'rust/crates/ca-py/src/vector_map.rs',
])
def test_development_native_source_can_change_but_resigned_executed_source_cannot(tmp_path,name):
    target=copy_packet(tmp_path)
    manifest=packet.read(target/'manifest.json')
    alias=manifest['native_source_aliases'][name]
    (target.parents[2]/name).write_bytes(b'// newer development source\n')
    assert packet.verify(target)['default_replay_pose_bits_checked']==2560
    archived=target.parents[2]/alias
    with archived.open('ab') as stream:stream.write(b'\n// changed executed source\n')
    manifest['dependencies'][alias]=packet.sha(archived)
    save(target/'manifest.json',manifest)
    with pytest.raises(ValueError,match='normalized native source'):
        packet.verify(target)
