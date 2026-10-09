"""Correction trials preserve the baseline and never silently inherit old HD geometry."""
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from ca import mapping_job as jobs, mapping_motion_trial as motion
from ca.posegraph_fix import read_trajectory
from ca.mapping_trajectory import evaluate_mapping_trajectory
from ca.mapping_trajectory_review import compare_mapping_motion_trials
from cloudanalyzer_cli.main import app
from tests.test_mapping_trajectory import fixture, PROVENANCE, _localized_report  # Shared real native graph/motion fixture.


@pytest.fixture
def trial_fixture(fixture, monkeypatch):
    baseline, _, _ = fixture
    job = jobs._load(baseline)
    job['runtime'] = {'native': jobs._native(), 'python': 'test'}
    job['pointcloud_options'] = {'keyframe_spacing': 1., 'remove_dynamic': False,
                                'scan_voxel_m': .4, 'map_voxel_m': .2, 'pointcloud_topic': '/points', 'imu_topic': '/imu'}
    job['pointcloud']['path_length_m'] = 6.
    job['max_attempts'] = 4
    job['minimum_retained_fraction'] = .9
    jobs._save(baseline / 'job.json', job)
    def decode(source, folder, **options):
        assert source == str(baseline / 'source.mcap')
        assert options == {'topic': '/points', 'kitti_bin': True, 'max_frames': 4097}
        folder.mkdir()
        paths = []
        for i in range(7):
            path = folder / f'frame_{i:06d}.bin'
            np.array([[0., 0., 0., 7.], [1., 0., 0., 8.], [0., 1., 0., 9.]], dtype=np.float32).tofile(path)
            paths.append(path)
        return paths, tuple(float(i) for i in range(7))
    monkeypatch.setattr(motion, 'materialize_pointcloud_bag', decode)
    return baseline, baseline.parent / 'trial'


def _run(value, **options):
    baseline, out = value
    return motion.trial_mapping_motion(str(baseline), str(out), False, False, 'Test alternative correction', **options)


def test_trial_restarts_original_motion_exact_ids_and_keeps_baseline_and_hd_selection(trial_fixture):
    baseline, out = trial_fixture
    before = {p.name: p.read_bytes() for p in baseline.iterdir()}
    result = _run(trial_fixture, max_attempts=6)
    assert result['status'] == 'ready_unverified', result
    child = jobs.inspect_mapping_job(str(out))
    assert child['attempts'] == [] and child['selected'] is None
    assert child['remaining_attempts'] == 6
    assert 'corridor_proposal' not in child and 'run' not in child
    assert child['pointcloud']['source_motion'] == jobs._load(baseline)['pointcloud']['source_motion']
    graph = jobs.core().PoseGraph.from_g2o(Path(child['pointcloud']['files']['graph']['path']).read_text())
    assert list(graph.node_ids) == [0, 2, 4, 6]
    original, _ = read_trajectory(baseline / 'original.tum')
    np.testing.assert_allclose(graph.poses(), original[[0, 2, 4, 6]], atol=1e-12)
    report = json.loads((out / 'trial.json').read_text())
    assert report['protocol']['reference_used_for_generation'] is False
    assert report['protocol']['old_hd_geometry_or_audits_inherited'] is False
    assert report['excluded_original_frame_ids'] == [1, 3, 5]
    assert 'loops' not in json.loads((out / 'pointcloud-report.json').read_text())
    assert 'gravity' not in json.loads((out / 'pointcloud-report.json').read_text())
    jobs._inputs(child)
    assert {p.name: p.read_bytes() for p in baseline.iterdir()} == before


def test_explicit_loop_and_gravity_choices_are_applied_without_truth(trial_fixture, monkeypatch):
    baseline, out = trial_fixture
    called = {}
    fix = motion.fix_session
    def capture(*args, **kwargs):
        called.update(kwargs)
        return fix(*args, **kwargs)
    monkeypatch.setattr(motion, 'fix_session', capture)
    def ups(source, stamps, **options):
        assert source == str(baseline / 'source.mcap') and options == {'topic': '/imu'}
        return {i: np.array([0., 0., 1.]) for i in range(7)}
    monkeypatch.setattr(motion, 'imu_ups', ups)
    result = motion.trial_mapping_motion(str(baseline), str(out), True, True, 'Explicit full correction')
    assert result['status'] == 'ready_unverified', result
    assert called['loops'] is True and called['gravity'] == str(out / 'gravity.txt')
    assert 'truth' not in called
    assert called['voxel'] == .4 and called['map_voxel'] == .2 and called['remove_dynamic'] is False
    correction = json.loads((out / 'pointcloud-report.json').read_text())
    assert 'loops' in correction and correction['gravity']['tied'] == 4


def test_requested_missing_gravity_is_retained_failure_not_disabled(trial_fixture, monkeypatch):
    monkeypatch.setattr(motion, 'imu_ups', lambda *a, **k: {0: np.array([0., 0., 1.])})
    baseline, out = trial_fixture
    result = motion.trial_mapping_motion(str(baseline), str(out), False, True, 'Require measured gravity')
    assert result['status'] == 'failed' and 'every retained node' in result['error']
    assert jobs._load(out)['pointcloud'] is None
    assert json.loads((out / 'trial.json').read_text())['policy']['use_gravity'] is True


def test_changed_baseline_is_rejected_before_any_directory_is_created(trial_fixture):
    baseline, out = trial_fixture
    (baseline / 'map.ply').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        _run(trial_fixture)
    assert not out.exists()


def test_changed_native_is_rejected_before_trial_creation(trial_fixture, monkeypatch):
    native = jobs._native()
    monkeypatch.setattr(jobs, '_native', lambda: {**native, 'version': 'changed'})
    with pytest.raises(ValueError, match='native core changed'):
        _run(trial_fixture)
    assert not trial_fixture[1].exists()


def test_legacy_missing_motion_is_refused_without_inventing_timestamps(trial_fixture):
    baseline, out = trial_fixture
    job = jobs._load(baseline)
    del job['pointcloud']['source_motion']
    jobs._save(baseline / 'job.json', job)
    with pytest.raises(ValueError, match='source_motion'):
        _run(trial_fixture)
    assert not out.exists()


def test_interleaved_baseline_edit_is_retained_as_failure(trial_fixture, monkeypatch):
    decode = motion.materialize_pointcloud_bag
    def changed(*a, **k):
        result = decode(*a, **k)
        (trial_fixture[0] / 'source.mcap').write_bytes(b'changed during decoding')
        return result
    monkeypatch.setattr(motion, 'materialize_pointcloud_bag', changed)
    result = _run(trial_fixture)
    assert result['status'] == 'failed' and 'changed' in result['error']


def test_fresh_recording_clock_must_match_original_motion(trial_fixture, monkeypatch):
    decode = motion.materialize_pointcloud_bag
    def wrong_clock(*a, **k):
        paths, stamps = decode(*a, **k)
        return paths, tuple(t + .001 for t in stamps)
    monkeypatch.setattr(motion, 'materialize_pointcloud_bag', wrong_clock)
    result = _run(trial_fixture)
    assert result['status'] == 'failed' and 'timestamps' in result['error']


def test_invalid_fresh_scan_and_resource_limit_are_retained(trial_fixture, monkeypatch):
    monkeypatch.setattr(motion, 'MAX_RETURNS', 1)
    result = _run(trial_fixture)
    assert result['status'] == 'failed' and 'five million' in result['error']


def test_nonfinite_fresh_scan_is_retained_failure(trial_fixture, monkeypatch):
    decode = motion.materialize_pointcloud_bag
    def invalid(*a, **k):
        paths, stamps = decode(*a, **k)
        np.array([[float('nan'), 0., 0., 1.]], dtype=np.float32).tofile(paths[0])
        return paths, stamps
    monkeypatch.setattr(motion, 'materialize_pointcloud_bag', invalid)
    result = _run(trial_fixture)
    assert result['status'] == 'failed' and 'nonfinite' in result['error']


def test_generated_scan_changed_during_correction_is_not_published_ready(trial_fixture, monkeypatch):
    fix = motion.fix_session
    def changed(*a, **k):
        result = fix(*a, **k)
        path = trial_fixture[1] / 'scans/frame_000000.bin'
        path.write_bytes(b'edited during processing')
        return result
    monkeypatch.setattr(motion, 'fix_session', changed)
    result = _run(trial_fixture)
    assert result['status'] == 'failed' and 'changed' in result['error']
    assert jobs._load(trial_fixture[1])['pointcloud'] is None


def test_inconsistent_corrected_graph_and_trajectory_cannot_become_ready(trial_fixture, monkeypatch):
    fix = motion.fix_session
    def inconsistent(*a, **k):
        result = fix(*a, **k)
        path = Path(result['outputs']['kitti'])
        rows = np.loadtxt(path)
        rows[0, 3] += 1.
        np.savetxt(path, rows)
        return result
    monkeypatch.setattr(motion, 'fix_session', inconsistent)
    result = _run(trial_fixture)
    assert result['status'] == 'failed' and 'consistent nonempty' in result['error']


def test_changed_retained_node_ids_cannot_become_ready(trial_fixture, monkeypatch):
    fix = motion.fix_session
    def changed_ids(*a, **k):
        result = fix(*a, **k)
        path = Path(result['outputs']['g2o'])
        graph = jobs.core().PoseGraph.from_g2o(path.read_text())
        replacement = jobs.core().PoseGraph.from_poses(np.ascontiguousarray(graph.poses()), ids=[0, 1, 4, 6])
        path.write_text(replacement.to_g2o())
        return result
    monkeypatch.setattr(motion, 'fix_session', changed_ids)
    result = _run(trial_fixture)
    assert result['status'] == 'failed' and 'exact retained nodes' in result['error']


def test_failed_or_partial_trial_is_never_overwritten_or_reexecuted(trial_fixture, monkeypatch):
    calls = []
    def failure(*a, **k):
        calls.append(1)
        raise RuntimeError('intentional native failure')
    monkeypatch.setattr(motion, 'fix_session', failure)
    result = _run(trial_fixture)
    assert result['status'] == 'failed'
    before = {str(p): p.read_bytes() for p in trial_fixture[1].rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError, match='retain and inspect'):
        _run(trial_fixture)
    assert len(calls) == 1
    assert {str(p): p.read_bytes() for p in trial_fixture[1].rglob('*') if p.is_file()} == before


def test_trial_cannot_be_written_inside_baseline_job(trial_fixture):
    baseline, _ = trial_fixture
    with pytest.raises(ValueError, match='outside'):
        motion.trial_mapping_motion(str(baseline), str(baseline / 'new-trial'), False, False, 'test')


@pytest.mark.parametrize('options', [dict(find_loops=1), dict(use_gravity=None), dict(reason=''), dict(max_attempts=True), dict(max_attempts=0), dict(max_attempts=9)])
def test_invalid_trial_decisions_are_rejected_without_processing(tmp_path, options):
    args = dict(job_dir=str(tmp_path / 'missing'), out_dir=str(tmp_path / 'out'), find_loops=False, use_gravity=False, reason='test', max_attempts=4)
    args.update(options)
    with pytest.raises(ValueError):
        motion.trial_mapping_motion(**args)
    assert not (tmp_path / 'out').exists()


def test_cli_uses_explicit_policy_and_returns_failure_exit_status(trial_fixture, monkeypatch):
    baseline, out = trial_fixture
    policy = baseline.parent / 'policy.json'
    policy.write_text(json.dumps({'find_loops': False, 'use_gravity': False}))
    def failure(*a, **k):
        raise RuntimeError('intentional failure')
    monkeypatch.setattr(motion, 'fix_session', failure)
    result = CliRunner().invoke(app, ['mapping-motion-trial', str(baseline), '--out', str(out), '--policy', str(policy), '--reason', 'CLI trial'])
    assert result.exit_code == 1 and json.loads(result.output)['status'] == 'failed'


@pytest.fixture
def comparison_pair(trial_fixture):
    baseline, child = trial_fixture
    result = _run(trial_fixture)
    assert result['status'] == 'ready_unverified'
    truth = baseline.parent / 'reference.tum'
    base = evaluate_mapping_trajectory(str(baseline), str(truth), PROVENANCE, str(baseline.parent / 'baseline-report.json'))['report']
    trial = evaluate_mapping_trajectory(str(child), str(truth), PROVENANCE, str(baseline.parent / 'candidate-report.json'))['report']
    return base, trial


def test_trial_comparison_detects_regression_with_same_ids_and_separate_map_bounds(comparison_pair):
    base, trial = comparison_pair
    result = compare_mapping_motion_trials(base, trial, window_poses=2)
    assert result['ate_rmse_m_candidate_minus_baseline'] > .1
    assert result['global_results']['baseline']['ate_rmse_m'] < 1e-10
    assert result['global_results']['candidate']['ate_rmse_m'] > .1
    assert sorted(i for w in result['windows'] for i in w['original_frame_ids']) == [0, 2, 4, 6]
    assert all(w['ate_rmse_m_candidate_minus_baseline'] > 0 for w in result['windows'])
    assert result['point_maps']['baseline'] != result['point_maps']['candidate']
    assert 'quality_gate' not in result
    unchanged = compare_mapping_motion_trials(base, base, window_poses=2)
    assert unchanged['ate_rmse_m_candidate_minus_baseline'] == 0.
    assert [w['window_id'] for w in unchanged['windows']] == [0, 1]


@pytest.mark.parametrize('field', ['reference_provenance', 'max_time_delta_s', 'alignment_prefix_fraction'])
def test_trial_comparison_refuses_changed_protocol_even_if_metrics_match(comparison_pair, field):
    base, trial = comparison_pair
    report = json.loads(Path(trial['path']).read_text())
    if field == 'reference_provenance':
        report[field]['frame'] = 'different frame declaration'
    else:
        report['protocol'][field] = .25
    Path(trial['path']).write_text(json.dumps(report))
    with pytest.raises(ValueError, match='identical reference/protocol'):
        compare_mapping_motion_trials(base, jobs._artifact(trial['path']))


def test_trial_comparison_refuses_changed_source_content(comparison_pair):
    base, trial = comparison_pair
    report = json.loads(Path(trial['path']).read_text())
    source = Path(trial['path']).parent / 'another-source.mcap'
    source.write_bytes(b'other source data')
    report['inputs']['source'] = jobs._artifact(source)
    Path(trial['path']).write_text(json.dumps(report))
    with pytest.raises(ValueError, match='identical source/motion/reference'):
        compare_mapping_motion_trials(base, jobs._artifact(trial['path']))


def test_trial_comparison_refuses_changed_artifact(comparison_pair):
    base, trial = comparison_pair
    Path(trial['path']).write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        compare_mapping_motion_trials(base, trial)


def test_trial_comparison_refuses_undersized_evaluation_instead_of_nan_rpe(comparison_pair):
    base, trial = comparison_pair
    report = json.loads(Path(trial['path']).read_text())
    report['coverage']['evaluated_original_frame_ids'] = [0]
    Path(trial['path']).write_text(json.dumps(report))
    with pytest.raises(ValueError, match='at least three'):
        compare_mapping_motion_trials(base, jobs._artifact(trial['path']))


def test_trial_comparison_paging_covers_suffix_once_with_single_pose_tail(fixture):
    report = _localized_report(fixture, count=23)
    first = compare_mapping_motion_trials(report, report, window_poses=2)
    assert first['next_offset'] == 8 and len(first['windows']) == 8
    second = compare_mapping_motion_trials(report, report, window_poses=2, offset=8)
    assert second['next_offset'] is None and len(second['windows']) == 2
    windows = first['windows'] + second['windows']
    assert [w['window_id'] for w in windows] == list(range(10))
    assert [i for w in windows for i in w['original_frame_ids']] == list(range(4, 23))
    assert windows[-1]['candidate']['metrics']['rpe_translation_rmse_m'] is None


def test_trial_comparison_rechecks_interleaved_candidate_map_edit(comparison_pair, monkeypatch):
    from ca import mapping_trajectory_review as review
    windows = review._windows
    calls = []
    def changed(report, *a):
        result = windows(report, *a)
        calls.append(1)
        if len(calls) == 2:
            Path(report['inputs']['pointcloud_map']['path']).write_bytes(b'changed during comparison')
        return result
    monkeypatch.setattr(review, '_windows', changed)
    with pytest.raises(ValueError, match='changed'):
        compare_mapping_motion_trials(*comparison_pair)


def test_trial_comparison_refuses_different_valid_retained_coverage(comparison_pair):
    base, trial = comparison_pair
    report = json.loads(Path(trial['path']).read_text())
    root = Path(report['inputs']['job']['path']).parent
    job = jobs._load(root)
    graph = jobs.core().PoseGraph.from_g2o(Path(job['pointcloud']['files']['graph']['path']).read_text())
    ids = list(graph.node_ids)[:3]
    graph = jobs.core().PoseGraph.from_poses(np.ascontiguousarray(graph.poses()[:3]), ids=ids)
    graph_file = Path(job['pointcloud']['files']['graph']['path'])
    poses_file = Path(job['pointcloud']['files']['trajectory']['path'])
    graph_file.write_text(graph.to_g2o())
    poses_file.write_text(graph.to_kitti())
    job['pointcloud']['files'].update(graph=jobs._artifact(graph_file), trajectory=jobs._artifact(poses_file))
    jobs._save(root / 'job.json', job)
    reduced = evaluate_mapping_trajectory(str(root), report['inputs']['reference']['path'], PROVENANCE, str(root.parent / 'reduced.json'))['report']
    with pytest.raises(ValueError, match='identical reference/protocol'):
        compare_mapping_motion_trials(base, reduced)
