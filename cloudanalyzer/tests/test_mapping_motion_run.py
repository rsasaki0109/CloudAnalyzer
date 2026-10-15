"""Fresh motion/HD pair comparison binds immutable evidence and restores exact bytes."""
import json
from pathlib import Path

import numpy as np
import pytest

from ca import mapping_job as jobs, mapping_run as runs, mapping_motion_run as pairs, mapping_motion_trial as motion
from ca.mapping_trajectory import _write, evaluate_mapping_trajectory
from ca.posegraph_fix import write_ply
from tests.test_mapping_motion_trial import trial_fixture  # noqa: F401
from tests.test_mapping_trajectory import fixture, PROVENANCE  # noqa: F401
from tests.test_mapping_job import _run_layout


def _snapshot(root):
    return {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def _finish(root):
    run = runs.inspect_mapping_run(str(root))
    ids = [c['id'] for c in run['candidate_index']['candidate_index']]
    assert ids, run
    for start in range(0, len(ids), 8):
        run = runs.advance_mapping_run(str(root), {'type': 'inspect', 'candidate_ids': ids[start:start+8]}, 'Inspect source geometry', run['revision'])
    run = runs.advance_mapping_run(str(root), {'type': 'draft', 'decisions': [
        {'candidate_id': i, 'action': 'include', 'reason': 'Explicit plane-surface test hypothesis'} for i in ids]}, 'Generate fresh lane geometry', run['revision'])
    assert run['draft_result']['status'] == 'audited_draft', run
    return runs.advance_mapping_run(str(root), {'type': 'finish', 'candidate_id': run['draft_result']['diagnosis']['candidate_id']}, 'Retain exact audited draft pair', run['revision'])


@pytest.fixture(params=['all_supported_bands', 'trajectory_containing'])
def pair_fixture(trial_fixture, monkeypatch, request):
    baseline, trial = trial_fixture
    original = np.column_stack((np.arange(7.), (np.arange(7) % 3) * .5, np.ones(7)))
    _write(baseline / 'original.tum', np.arange(7.), original, np.tile([0., 0., 0., 1.], (7, 1)))
    poses = np.tile(np.eye(4), (4, 1, 1)); poses[:, :3, 3] = original[[0, 2, 4, 6]]
    poses[:, 0, 3] *= 1.15
    graph = jobs.core().PoseGraph.from_poses(np.ascontiguousarray(poses), ids=[0, 2, 4, 6])
    (baseline / 'corrected.g2o').write_text(graph.to_g2o())
    (baseline / 'corrected.txt').write_text(graph.to_kitti())
    points = np.array([[x / 5, y / 5, 0.] for x in range(-15, 55) for y in range(-20, 26)])
    write_ply(baseline / 'map.ply', points, {})
    job = jobs._load(baseline)
    job['pointcloud']['files'] = {k: jobs._artifact(baseline / filename) for k, filename in [('map', 'map.ply'), ('trajectory', 'corrected.txt'), ('graph', 'corrected.g2o')]}
    job['pointcloud']['source_motion']['trajectory'] = jobs._artifact(baseline / 'original.tum')
    job['pointcloud'].update(map_points=len(points), coordinate_frame='local_slam_metres')
    job.update(status='pointcloud_ready', attempts=[], selected=None)
    jobs._save(baseline / 'job.json', job)
    jobs.propose_mapping_corridors(str(baseline))
    if request.param == 'trajectory_containing':
        assert jobs._refine_mapping_corridors(str(baseline), 'Frozen path association')['status'] == 'ready'
    jobs._save(baseline / 'layout-hypothesis.json', _run_layout())
    jobs._save(baseline / 'run.json', {'schema': runs.SCHEMA, 'revision': 0, 'status': 'needs_agent',
        'layout_file': jobs._artifact(baseline / 'layout-hypothesis.json'), 'reviewed_candidates': [], 'history': [], 'output': None, 'maximum_actions': 128})
    _finish(baseline)
    reference = baseline.parent / 'reference.tum'
    baseline_report = evaluate_mapping_trajectory(str(baseline), str(reference), PROVENANCE, str(baseline.parent / 'baseline-evaluation.json'))['report']
    fix = motion.fix_session
    def ground(*a, **k):
        result = fix(*a, **k)
        write_ply(Path(result['outputs']['map']), points, {})
        result['map_points'] = len(points)
        return result
    monkeypatch.setattr(motion, 'fix_session', ground)
    result = motion.trial_mapping_motion(str(baseline), str(trial), False, False, 'Retained original-motion candidate')
    assert result['status'] == 'ready_unverified', result
    candidate_report = evaluate_mapping_trajectory(str(trial), str(reference), PROVENANCE, str(baseline.parent / 'trial-evaluation.json'))['report']
    return baseline, trial, baseline_report, candidate_report


def _generate(value):
    baseline, trial, *_ = value
    root = baseline.parent / 'fresh-hd'
    result = pairs.start_mapping_motion_run(str(trial), str(baseline), str(root), 4, 'Fresh HD for changed motion')
    assert result['status'] == 'needs_agent', result
    _finish(root)
    return root


def test_fresh_hd_generation_comparison_selection_and_exact_restore_preserve_inputs(pair_fixture):
    baseline, trial, base_report, trial_report = pair_fixture
    before, trial_before = _snapshot(baseline), _snapshot(trial)
    root = _generate(pair_fixture)
    candidate_before = _snapshot(root)
    result = pairs.compare_mapping_motion_maps(str(root), base_report, trial_report, str(baseline.parent / 'decision'), 'Review new point and HD pair')
    comparison = json.loads(Path(result['comparison']['path']).read_text())
    assert comparison['baseline']['raw_extent']['trajectory_length_m'] != comparison['candidate']['raw_extent']['trajectory_length_m']
    assert comparison['baseline']['common_extent']['trajectory_length_m'] == comparison['candidate']['common_extent']['trajectory_length_m']
    assert len(comparison['motion']['windows']) == 1
    assert result['choice'] == 'baseline' and result['revision'] == 0
    original_pair = runs.inspect_mapping_run(str(baseline))['output']
    assert result['output'] == original_pair
    selected = pairs.choose_mapping_motion_pair(result['selection_dir'], 'candidate', 'Review changed draft without certification', 0)
    assert selected['revision'] == 1 and selected['output'] == runs.inspect_mapping_run(str(root))['output']
    assert selected['output']['deployment_ready'] is False
    assert pairs.choose_mapping_motion_pair(result['selection_dir'], 'candidate', 'Repeated choice is a no-op', 1)['revision'] == 1
    with pytest.raises(ValueError, match='stale'):
        pairs.choose_mapping_motion_pair(result['selection_dir'], 'baseline', 'Stale caller', 0)
    restored = pairs.choose_mapping_motion_pair(result['selection_dir'], 'baseline', 'Restore exact original pair', 1)
    assert restored['revision'] == 2 and restored['output'] == original_pair
    assert len(restored['history']) == 3
    assert _snapshot(baseline) == before and _snapshot(trial) == trial_before and _snapshot(root) == candidate_before
    assert jobs._load(root)['selected'] is None
    assert jobs._load(root)['attempts'][0]['corridor_proposal'] != jobs._load(baseline)['attempts'][0]['corridor_proposal']


def test_start_preserves_failed_proposal_and_never_reruns_existing_output(pair_fixture, monkeypatch):
    baseline, trial, *_ = pair_fixture
    def failure(*a, **k):
        raise RuntimeError('retained source extraction failure')
    monkeypatch.setattr(jobs.core(), 'propose_road_corridors', failure)
    root = baseline.parent / 'failed-hd'
    result = pairs.start_mapping_motion_run(str(trial), str(baseline), str(root), 2, 'Retain failed proposal')
    assert result['status'] == 'processing_failed' and result['remaining_attempts'] == 2
    before = _snapshot(root)
    with pytest.raises(FileExistsError):
        pairs.start_mapping_motion_run(str(trial), str(baseline), str(root), 2, 'Do not repeat')
    assert _snapshot(root) == before


def test_changed_trial_or_baseline_is_rejected_before_starting(pair_fixture):
    baseline, trial, *_ = pair_fixture
    (trial / 'job.json').write_text((trial / 'job.json').read_text().replace('pointcloud_ready', 'candidates_ready'))
    target = baseline.parent / 'rejected'
    with pytest.raises(ValueError, match='untouched'):
        pairs.start_mapping_motion_run(str(trial), str(baseline), str(target), 2, 'Changed trial')
    assert not target.exists()


def test_changed_map_prevents_selection_without_changing_history(pair_fixture):
    baseline, trial, first, second = pair_fixture
    root = _generate(pair_fixture)
    result = pairs.compare_mapping_motion_maps(str(root), first, second, str(baseline.parent / 'decision'), 'Compare immutable pairs')
    saved = Path(result['selection_dir']) / 'selection.json'
    before = saved.read_bytes()
    Path(jobs._load(root)['pointcloud']['files']['map']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        pairs.choose_mapping_motion_pair(result['selection_dir'], 'candidate', 'Reject edited map', 0)
    assert saved.read_bytes() == before


def test_reports_from_another_point_map_cannot_authorize_pair_comparison(pair_fixture):
    baseline, _, first, _ = pair_fixture
    root = _generate(pair_fixture)
    target = baseline.parent / 'rejected-comparison'
    with pytest.raises(ValueError, match='exact frozen'):
        pairs.compare_mapping_motion_maps(str(root), first, first, str(target), 'Wrong candidate report')
    assert not target.exists()


def test_interrupted_preparation_keeps_points_and_can_finish_without_repeating(pair_fixture, monkeypatch):
    baseline, trial, *_ = pair_fixture
    propose = jobs.propose_mapping_corridors
    calls = []
    def interrupted(*a, **k):
        result = propose(*a, **k)
        calls.append(result)
        raise KeyboardInterrupt()
    monkeypatch.setattr(jobs, 'propose_mapping_corridors', interrupted)
    root = baseline.parent / 'interrupted-start'
    with pytest.raises(KeyboardInterrupt):
        pairs.start_mapping_motion_run(str(trial), str(baseline), str(root), 2, 'Retain interrupted preparation')
    inspected = runs.inspect_mapping_run(str(root))
    assert inspected['status'] == 'processing_failed' and inspected['remaining_attempts'] == 2
    finished = runs.advance_mapping_run(str(root), {'type': 'finish', 'candidate_id': None}, 'Retain point map; HD preparation interrupted', 0)
    assert finished['output']['status'] == 'hd_unavailable'
    assert len(calls) == 1 and finished['output']['artifacts'] == jobs._load(trial)['pointcloud']['files']


def test_interleaved_point_edit_during_comparison_does_not_create_selection(pair_fixture, monkeypatch):
    baseline, _, first, second = pair_fixture
    root = _generate(pair_fixture)
    intervals = pairs._intervals
    calls = []
    def edit(*a, **k):
        result = intervals(*a, **k)
        calls.append(1)
        if len(calls) == 2:
            Path(jobs._load(root)['pointcloud']['files']['map']['path']).write_bytes(b'changed during comparison')
        return result
    monkeypatch.setattr(pairs, '_intervals', edit)
    target = baseline.parent / 'rejected-comparison'
    with pytest.raises(ValueError, match='changed'):
        pairs.compare_mapping_motion_maps(str(root), first, second, str(target), 'Reject interleaved edit')
    assert not target.exists()


def test_audit_protocol_change_prevents_comparison(pair_fixture, monkeypatch):
    baseline, _, first, second = pair_fixture
    root = _generate(pair_fixture)
    summary = pairs._audit_summary
    calls = []
    def changed(*a, **k):
        result = summary(*a, **k)
        calls.append(1)
        if len(calls) == 2:
            result['editable']['protocol'] = {**result['editable']['protocol'], 'ground_radius_m': 9.}
        return result
    monkeypatch.setattr(pairs, '_audit_summary', changed)
    target = baseline.parent / 'different-protocol'
    with pytest.raises(ValueError, match='source-audit protocols'):
        pairs.compare_mapping_motion_maps(str(root), first, second, str(target), 'Do not weaken audit')
    assert not target.exists()


@pytest.mark.parametrize('budget', [1, 9, True])
def test_invalid_new_budget_does_not_touch_inputs(tmp_path, budget):
    target = tmp_path / 'new'
    with pytest.raises(ValueError, match='budget'):
        pairs.start_mapping_motion_run('missing', 'missing', str(target), budget, 'Test explicit budget')
    assert not target.exists()


def test_common_stations_map_changed_metric_intervals_through_original_frames():
    evidence = {'extraction': {'trajectory_length': 20.}, 'station_disposition': [
        {'from_m': 0., 'to_m': 5., 'status': 'source_deferred'},
        {'from_m': 5., 'to_m': 20., 'status': 'included_lane_hypothesis'}]}
    interval = pairs._intervals(evidence, np.array([0., 10., 20.]), np.array([0., 2., 8.]), np.array([100., 102., 104.]), [0, 2, 4])[0]
    assert (interval['from_m'], interval['to_m']) == (1., 8.)
    assert interval['from_timestamp_s'] == 101. and interval['from_original_frame_coordinate'] == 1.
    evidence['station_disposition'][0]['to_m'] = 4.
    with pytest.raises(ValueError, match='partition'):
        pairs._intervals(evidence, np.array([0., 10., 20.]), np.array([0., 2., 8.]), np.array([100., 102., 104.]), [0, 2, 4])


def test_stationary_retained_motion_has_no_invented_station_correspondence():
    with pytest.raises(ValueError, match='nonzero'):
        pairs._stations(np.tile(np.eye(4), (3, 1, 1)))
