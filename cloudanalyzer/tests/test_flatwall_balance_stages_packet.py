"""Real fixed-state arithmetic, timing accounting and source-preservation controls."""
import copy
import gzip
import importlib.util
import json
import functools
import math
import operator
from pathlib import Path
import shutil

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT/'benchmarks/slam/flatwall-balance-stages'
spec = importlib.util.spec_from_file_location('balance_stage_packet',PACKET/'verify.py')
packet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packet)
independent = packet.load_independent(ROOT)


def copy_packet(tmp_path):
    target = tmp_path/'checkout/benchmarks/slam/flatwall-balance-stages'
    shutil.copytree(PACKET,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    manifest = packet.read(PACKET/'manifest.json')
    footprint = packet.read(PACKET/'source-footprints.json')
    for name in manifest['dependencies']|footprint['normalized_checkout_sources_sha256']:
        destination = target.parents[2]/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    return target


@pytest.fixture(scope='module')
def fixed_states():
    return independent.expected(ROOT,packet.read(PACKET/'origins.json'))


@pytest.fixture(scope='module')
def run():
    return json.loads(gzip.decompress((PACKET/'run-00-threads-1.json.gz').read_bytes()))


def test_complete_real_fixed_state_packet():
    result = packet.verify()
    assert (result['states'],result['eligible_states'],result['original_empty_source_controls']) == (64,60,4)
    assert (result['native_validation_states'],result['measured_blocks'],result['warmup_blocks']) == (256,1280,256)
    assert result['ordered_pair_rows_checked'] == 128664
    assert result['production_arithmetic_changed'] is result['defaults_changed'] is result['whole_registration_speedup_established'] is False


@pytest.mark.parametrize('change',['target','count','distance','matrix','rhs','decomposition','identity'])
def test_native_pair_or_matrix_changes_are_rejected(fixed_states,run,change):
    candidate = copy.deepcopy(run)
    row = candidate['validation'][0]
    if change == 'target':
        row['targets'][0][0] += .1
    elif change == 'count':
        row['multiplicity'][0] += 1
    elif change == 'distance':
        row['distance_sq'][0] += .1
    elif change == 'matrix':
        row['balanced_h'][0][0] += .1
    elif change == 'rhs':
        row['baseline_b'][0] += .1
    elif change == 'decomposition':
        row['maximum_decomposition_abs_error'] += .1
    else:
        row['name'] += '/changed'
    with pytest.raises((ValueError,AssertionError)):
        independent.check_run(candidate,fixed_states,1)


@pytest.mark.parametrize('change',['missing','warmup','order','time','stage_window','repetitions','threads'])
def test_missing_or_reordered_timing_samples_are_rejected(fixed_states,run,change):
    candidate = copy.deepcopy(run)
    sample = candidate['samples'][64]
    if change == 'missing':
        candidate['samples'].pop()
    elif change == 'warmup':
        sample['warmup'] = True
    elif change == 'order':
        sample['order'] = list(reversed(sample['order']))
    elif change == 'time':
        sample['balanced_ns'] = 0
    elif change == 'stage_window':
        sample['count_ns'] = sample['staged_ns']+1
    elif change == 'repetitions':
        sample['repetitions'] = 1
    else:
        candidate['threads'] = 4
    with pytest.raises(ValueError):
        independent.check_run(candidate,fixed_states,1)


def test_insufficient_point_states_keep_original_empty_registration_dispatch(fixed_states,run):
    origins = packet.read(PACKET/'origins.json')
    ids = [i for i,o in enumerate(origins) if not o['alignment_eligible']]
    assert len(ids) == 4
    for i in ids:
        assert origins[i]['original_source_points'] in [8,9]
        assert fixed_states[i]['source_points'] == 0
        assert not run['validation'][i]['source_indices']
        np.testing.assert_array_equal(run['validation'][i]['balanced_h'],np.zeros((6,6)))
    changed = dict(origins[ids[0]],alignment_eligible=True)
    with pytest.raises(ValueError,match='eligibility'):
        independent.state(ROOT,changed)


def test_fixed_selection_cannot_replace_an_empty_state():
    origins = packet.read(PACKET/'origins.json')
    definitions = packet.read(PACKET/'prepared-input.json')['states']
    empty = next(i for i,o in enumerate(origins) if not o['alignment_eligible'])
    origins[empty] = dict(origins[0])
    with pytest.raises(ValueError,match='selection'):
        packet.check_origins(ROOT,origins,definitions)


def test_resigned_executed_source_change_still_fails(tmp_path):
    target = copy_packet(tmp_path)
    manifest = packet.read(PACKET/'manifest.json')
    changed = target/'executed-source/rust/crates/ca-core/src/odometry_profile.rs'
    with changed.open('ab') as stream:
        stream.write(b'\n// changed executed profiling arithmetic\n')
    name = changed.relative_to(target).as_posix()
    manifest['files_sha256'][name] = packet.sha(changed)
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='normalized executed source'):
        packet.verify(target)


@pytest.mark.parametrize('crlf',[False,True])
def test_search_dependency_checkout_line_endings_preserve_identity(tmp_path,crlf):
    target = copy_packet(tmp_path)
    source = target.parents[2]/'scripts/flatwall_balance_dynamics.py'
    data = source.read_bytes().replace(b'\r\n',b'\n')
    source.write_bytes(data.replace(b'\n',b'\r\n') if crlf else data)
    assert packet.verify(target)['ordered_pair_rows_checked'] == 128664


@pytest.mark.parametrize('archive',[False,True])
def test_search_dependency_content_changes_are_rejected(tmp_path,archive):
    target = copy_packet(tmp_path)
    source = target/'original-source-flatwall-balance-dynamics.py' if archive else target.parents[2]/'scripts/flatwall_balance_dynamics.py'
    with source.open('ab') as stream:
        stream.write(b'\n# changed dependency source\n')
    if archive:
        manifest = packet.read(target/'manifest.json')
        manifest['files_sha256'][source.name] = packet.sha(source)
        (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='original dependency source' if archive else 'normalized diagnostic source'):
        packet.verify(target)


def test_pre_compensated_float_sum_still_verifies_original_aggregates(monkeypatch):
    legacy = packet.load_independent(ROOT)
    monkeypatch.setattr(legacy,'sum',lambda values,start=0:functools.reduce(operator.add,values,start),raising=False)
    monkeypatch.setattr(packet,'load_independent',lambda root:legacy)
    assert packet.verify()['measured_blocks'] == 1280


def test_summary_roundoff_keeps_integer_counts_types_and_material_ratios_strict():
    reference = packet.read(PACKET/'summary.json')
    changed = copy.deepcopy(reference)
    field = changed['groups']['eligible']['threads']['4']
    field['balanced_over_baseline'] = math.nextafter(field['balanced_over_baseline'],math.inf)
    packet.check_summary(changed,reference)
    field['balanced_over_baseline'] *= 1.001
    with pytest.raises(ValueError,match='summary float'):
        packet.check_summary(changed,reference)
    for name,value in [('measured_blocks',1281),('production_arithmetic_changed',0)]:
        changed = copy.deepcopy(reference)
        changed[name] = value
        with pytest.raises(ValueError,match='summary value'):
            packet.check_summary(changed,reference)
