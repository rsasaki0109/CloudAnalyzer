"""Real map arithmetic and semantic tamper controls, including resigned hashes."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT/'benchmarks/slam/flatwall-plane-objective'
spec = importlib.util.spec_from_file_location('plane_packet', PACKET/'verify.py')
packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)


def save(path, value): path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def copy_packet(tmp_path):
    target = tmp_path/'checkout/benchmarks/slam/flatwall-plane-objective'
    shutil.copytree(PACKET, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    manifest = packet.read(PACKET/'manifest.json')
    for name in manifest['dependencies'] | manifest['checkout_sources_sha256']:
        destination = target.parents[2]/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, destination)
    return target


def resign(target):
    catalog = packet.read(target/'snapshot-catalog.json')
    for row in catalog:
        path = target/row['file']
        if path.is_file(): row.update(sha256=packet.sha(path), bytes=path.stat().st_size)
    save(target/'snapshot-catalog.json', catalog)
    receipt = packet.read(target/'receipt.json')
    for name in receipt['output_files_sha256']:
        if (target/name).is_file(): receipt['output_files_sha256'][name] = packet.sha(target/name)
    save(target/'receipt.json', receipt)
    manifest = packet.read(target/'manifest.json')
    for name in manifest['files_sha256']: manifest['files_sha256'][name] = packet.sha(target/name)
    save(target/'manifest.json', manifest)


def test_all_real_frames_and_predetermined_full_map_samples():
    result = packet.verify(PACKET)
    assert (result['cases'], result['frames'], result['public_map_samples'], result['exhaustive_sample_queries']) == (16, 2560, 48, 132)
    assert result['all_frame_spectra_checked'] == 2560
    assert result['state_feedback'] is result['physical_observability_certified'] is False


@pytest.mark.parametrize('change', ['neighbor', 'normal', 'covariance', 'matrix', 'spectrum', 'summary', 'selection', 'scope', 'source_binding'])
def test_resigned_semantic_changes_are_detected(tmp_path, change):
    target = copy_packet(tmp_path)
    case = 'flatwall_01/repeat-motion/raw-0.1875'
    if change in {'neighbor', 'normal', 'covariance'}:
        path = target/case/'scan-093.npz'
        with np.load(path, allow_pickle=False) as saved: arrays = {k: saved[k].copy() for k in saved.files}
        if change == 'neighbor': arrays['neighbors'][0, 0] = -1
        elif change == 'normal': arrays['normals'][np.flatnonzero(arrays['accepted'])[0], 0] += .1
        else: arrays['covariance'][0, 0, 0] += .1
        np.savez_compressed(path, **arrays)
    elif change in {'matrix', 'spectrum', 'source_binding'}:
        path = target/case/'frames.jsonl'
        frames = packet.rows(path)
        if change == 'matrix': frames[92]['plane_right_hand_side'][0] += 1.
        elif change == 'spectrum': frames[92]['plane_geometry']['weak_directions_below_1e_3'] += 1
        else: frames[92]['original_snapshot_sha256'] = '0'*64
        path.write_text('\n'.join(json.dumps(row) for row in frames)+'\n', encoding='utf-8')
    else:
        path = target/('snapshot-catalog.json' if change == 'selection' else 'summary.json')
        value = packet.read(path)
        if change == 'summary': value['cases'][case]['accepted_normal_matches'] += 1
        elif change == 'scope': value['state_feedback'] = True
        else: value[0]['public_sample'] = True
        save(path, value)
    resign(target)
    with pytest.raises((ValueError, AssertionError)):
        packet.verify(target)


def test_original_sample_byte_changes_are_detected(tmp_path):
    target = copy_packet(tmp_path)
    with (target/'flatwall_01/repeat-motion/raw-0.1875/scan-093.npz').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='hash'):
        packet.verify(target)
