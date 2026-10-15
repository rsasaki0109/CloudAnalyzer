"""Per-run map snapshots preserve independent frame diagnostics and cleanup."""

import weakref

import numpy as np
import open3d as o3d
import pytest

from ca import _rust
import ca.scan_match_debug as debug
import ca.slam_debug as slam
from ca.registration import RegistrationResult


def files(tmp_path, source_and_target_files, count=3, missing=False):
    scan, target = source_and_target_files
    metrics, manifest = tmp_path / 'metrics.csv', tmp_path / 'scans.csv'
    metrics.write_text('scan_id,timestamp_sec,scan_match_rmse_m,initial_x_m,initial_y_m,initial_z_m\n' +
                       ''.join(f'frame_{i},{i},{count-i},{i*.03},0,0\n' for i in range(count)), encoding='utf-8')
    manifest.write_text('scan_id,points_csv\n' + ('' if missing else ''.join(f'frame_{i},{scan}\n' for i in range(count))), encoding='utf-8')
    return dict(metrics_csv=str(metrics), scans_manifest_csv=str(manifest), map_path=target,
                run_scan_match_debug_frames=True, top_k=count or 3, scan_match_method='icp')


def registration(source, target, **kwargs):
    # Each private working target starts from the input, even after another
    # frame's solver adds normals/colors to its own target.
    assert not target.has_normals() and not target.has_colors()
    source.normals = o3d.utility.Vector3dVector(np.tile([0., 0., 1.], (len(source.points), 1)))
    target.normals = o3d.utility.Vector3dVector(np.tile([0., 0., 1.], (len(target.points), 1)))
    target.colors = o3d.utility.Vector3dVector(np.tile([.3, .4, .5], (len(target.points), 1)))
    delta = np.eye(4)
    delta[0, 3] = .005
    aligned = o3d.geometry.PointCloud(source)
    aligned.transform(delta)
    return RegistrationResult(aligned, .91, .02, delta)


def normalized(value, base):
    if isinstance(value, str):
        return value.replace(str(base), '<artifacts>')
    if isinstance(value, list):
        return [normalized(item, base) for item in value]
    if isinstance(value, dict):
        return {key: normalized(item, base) for key, item in value.items()}
    return value


class FreshContext:
    def __init__(self, *args):
        pass

    def run(self, **kwargs):
        return debug.run_scan_match_debug(**kwargs)

    def invalidate(self):
        pass

    def clear(self):
        pass


@pytest.mark.parametrize('method,crop,voxel,artifacts,disable_rust', [
    ('icp', None, None, False, False), ('icp', None, .05, True, False),
    ('gicp', None, None, True, False), ('gicp', .1, .05, False, False),
    ('icp', .05, None, True, False), ('icp', 0., .05, False, False),
    ('icp', -1., None, True, False), ('gicp', None, .05, False, True),
    ('icp', None, None, True, True), ('gicp', .1, None, True, True),
    ('icp', 0., None, False, True), ('icp', -.1, .05, True, True),
])
def test_full_reports_distances_and_ply_bytes_match_fresh_frames(
    tmp_path, source_and_target_files, monkeypatch, method, crop, voxel, artifacts, disable_rust,
):
    options = files(tmp_path, source_and_target_files)
    options.update(scan_match_method=method, scan_match_crop_margin=crop,
                   scan_match_map_voxel_size=voxel, scan_match_threshold=.05)
    monkeypatch.setenv('CA_DISABLE_RUST_CORE', '1' if disable_rust else '0')
    _rust._load.cache_clear()
    monkeypatch.setattr(debug, 'register_detailed', registration)
    load, prepare = debug.load_point_cloud, debug.prepare_nn_reference
    counters = dict(map=0, scan=0, prepare=0)
    def traced_load(path):
        counters['map' if path == options['map_path'] else 'scan'] += 1
        return load(path)
    def traced_prepare(target):
        counters['prepare'] += 1
        return prepare(target)
    monkeypatch.setattr(debug, 'load_point_cloud', traced_load)
    monkeypatch.setattr(debug, 'prepare_nn_reference', traced_prepare)
    fresh_dir, reuse_dir = tmp_path / 'fresh', tmp_path / 'reuse'
    try:
        with monkeypatch.context() as patch:
            patch.setattr(slam, '_ScanMatchMapContext', FreshContext)
            expected = slam.analyze_slam_run(**options, artifact_dir=str(fresh_dir) if artifacts else None)
        assert counters == dict(map=3, scan=3, prepare=3)
        counters.update(map=0, scan=0, prepare=0)
        observed = slam.analyze_slam_run(**options, artifact_dir=str(reuse_dir) if artifacts else None)
        assert counters == dict(map=1, scan=3, prepare=3 if crop is not None and crop > 0 else 1)
        assert normalized(observed, reuse_dir) == normalized(expected, fresh_dir)
        assert all(frame['scan_match_debug_error'] is None for frame in observed['selected_frames'])
        if artifacts:
            expected_paths = sorted(path.relative_to(fresh_dir) for path in fresh_dir.rglob('*.ply'))
            assert len(expected_paths) == 9
            assert expected_paths == sorted(path.relative_to(reuse_dir) for path in reuse_dir.rglob('*.ply'))
            for path in expected_paths:
                assert (fresh_dir / path).read_bytes() == (reuse_dir / path).read_bytes()
        else:
            assert not fresh_dir.exists() and not reuse_dir.exists()
    finally:
        _rust._load.cache_clear()


@pytest.mark.parametrize('mode', ['disabled', 'empty', 'missing', 'single', 'top_one'])
def test_no_cache_or_eager_map_work_for_ineligible_selection(tmp_path, source_and_target_files, monkeypatch, mode):
    options = files(tmp_path, source_and_target_files, count=0 if mode == 'empty' else 3,
                    missing=mode == 'missing')
    if mode == 'disabled':
        options['run_scan_match_debug_frames'] = False
    elif mode == 'top_one':
        options['top_k'] = 1
    elif mode == 'single':
        manifest = tmp_path / 'scans.csv'
        manifest.write_text(f'scan_id,points_csv\nframe_0,{source_and_target_files[0]}\n', encoding='utf-8')
    def unexpected(*args, **kwargs):
        raise AssertionError('ineligible selection constructed map context')
    monkeypatch.setattr(slam, '_ScanMatchMapContext', unexpected)
    monkeypatch.setattr(debug, 'register_detailed', registration)
    load, calls = debug.load_point_cloud, []
    def traced(path):
        calls.append(path)
        return load(path)
    monkeypatch.setattr(debug, 'load_point_cloud', traced)
    result = slam.analyze_slam_run(**options)
    assert len(calls) == (2 if mode in ['single', 'top_one'] else 0)
    assert result['scan_match_debug_ran'] is (mode != 'disabled')


@pytest.mark.parametrize('failure', ['load', 'downsample', 'prepare', 'query', 'registration'])
def test_frame_failure_retries_and_keeps_later_frames_independent(tmp_path, source_and_target_files, monkeypatch, failure):
    options = files(tmp_path, source_and_target_files)
    options['artifact_dir'] = str(tmp_path / 'artifacts')
    load, downsample = debug.load_point_cloud, debug._downsample_if_needed
    calls = dict(load=0, prepare=0, registration=0, downsample=0)
    references = []
    def traced_load(path):
        if path == options['map_path']:
            calls['load'] += 1
            if failure == 'load' and calls['load'] == 1:
                raise FileNotFoundError('transient map load')
        return load(path)
    def traced_downsample(cloud, voxel):
        if np.asarray(cloud.points)[0, 0] == np.asarray(load(options['map_path']).points)[0, 0]:
            calls['downsample'] += 1
            if failure == 'downsample' and calls['downsample'] == 1:
                raise ValueError('transient map downsample')
        return downsample(cloud, voxel)
    def prepare(target):
        calls['prepare'] += 1
        if failure == 'prepare' and calls['prepare'] == 1:
            raise RuntimeError('transient reference preparation')
        coordinates = np.asarray(target.points).copy()
        first_reference = calls['prepare'] == 1
        class Reference:
            def distances(self, points):
                if failure == 'query' and first_reference:
                    raise RuntimeError('transient reference query')
                return np.linalg.norm(points[:, None, :] - coordinates, axis=2).min(axis=1)
        value = Reference()
        references.append(weakref.ref(value))
        return value
    def register(source, target, **kwargs):
        calls['registration'] += 1
        if failure == 'registration' and calls['registration'] == 1:
            raise RuntimeError('transient registration')
        return registration(source, target, **kwargs)
    monkeypatch.setattr(debug, 'load_point_cloud', traced_load)
    monkeypatch.setattr(debug, '_downsample_if_needed', traced_downsample)
    monkeypatch.setattr(debug, 'prepare_nn_reference', prepare)
    monkeypatch.setattr(debug, 'register_detailed', register)
    result = slam.analyze_slam_run(**options)
    frames = result['selected_frames']
    assert 'transient' in frames[0]['scan_match_debug_error']
    assert frames[0]['scan_match_debug_result'] is None
    assert all(frame['scan_match_debug_error'] is None for frame in frames[1:])
    assert calls['load'] == (2 if failure in ['load', 'downsample'] else 1)
    assert calls['prepare'] == (1 if failure in ['load', 'downsample'] else 2)
    assert not (tmp_path / 'artifacts/01_frame_0').exists()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize('cropped,artifacts,fatal', [(False, False, False), (False, True, False),
                                                    (True, True, False), (False, False, True)])
def test_context_and_reference_are_cleared_before_return_or_uncaught_failure(
    tmp_path, source_and_target_files, monkeypatch, cropped, artifacts, fatal,
):
    options = files(tmp_path, source_and_target_files)
    options.update(scan_match_crop_margin=.1 if cropped else None,
                   artifact_dir=str(tmp_path / 'artifacts') if artifacts else None)
    contexts, references, calls = [], [], []
    context = slam._ScanMatchMapContext
    class TracedContext(context):
        def __init__(self, *args):
            super().__init__(*args)
            contexts.append(self)
    def prepare(target):
        coordinates = np.asarray(target.points).copy()
        class Reference:
            def distances(self, points):
                return np.linalg.norm(points[:, None, :] - coordinates, axis=2).min(axis=1)
        value = Reference()
        references.append(weakref.ref(value))
        return value
    def register(source, target, **kwargs):
        calls.append(1)
        if fatal and len(calls) == 2:
            raise LookupError('uncaught solver failure')
        return registration(source, target, **kwargs)
    monkeypatch.setattr(slam, '_ScanMatchMapContext', TracedContext)
    monkeypatch.setattr(debug, 'prepare_nn_reference', prepare)
    monkeypatch.setattr(debug, 'register_detailed', register)
    try:
        slam.analyze_slam_run(**options)
        assert not fatal
    except LookupError:
        assert fatal
    assert len(contexts) == 1
    assert contexts[0]._cloud is None and contexts[0]._raw is None
    assert contexts[0]._reference is None and not contexts[0]._reference_ready
    assert all(reference() is None for reference in references)


def test_cached_snapshot_ignores_map_edits_until_a_new_run(tmp_path, source_and_target_files, monkeypatch):
    options = files(tmp_path, source_and_target_files)
    target_path = options['map_path']
    original = debug.load_point_cloud(target_path)
    changed = o3d.geometry.PointCloud(original)
    changed.translate([.2, 0., 0.])
    targets, loads = [], []
    load = debug.load_point_cloud
    def traced_load(path):
        if path == target_path:
            loads.append(1)
        return load(path)
    def register(source, target, **kwargs):
        targets.append(np.asarray(target.points).copy())
        if len(targets) == 1:
            assert o3d.io.write_point_cloud(target_path, changed)
        return RegistrationResult(o3d.geometry.PointCloud(source), 1., 0., np.eye(4))
    monkeypatch.setattr(debug, 'load_point_cloud', traced_load)
    monkeypatch.setattr(debug, 'prepare_nn_reference', lambda target: None)
    monkeypatch.setattr(debug, 'register_detailed', register)
    for _ in range(2):
        result = slam.analyze_slam_run(**options)
        assert all(frame['scan_match_debug_error'] is None for frame in result['selected_frames'])
    assert len(loads) == 2
    for observed in targets[:3]:
        assert observed.tobytes() == np.asarray(original.points).tobytes()
    reloaded = np.asarray(load(target_path).points)
    for observed in targets[3:]:
        assert observed.tobytes() == reloaded.tobytes()


def test_standalone_calls_keep_fresh_loading(source_and_target_files, monkeypatch):
    load, calls = debug.load_point_cloud, []
    def traced(path):
        calls.append(path)
        return load(path)
    monkeypatch.setattr(debug, 'load_point_cloud', traced)
    monkeypatch.setattr(debug, 'register_detailed', registration)
    for _ in range(2):
        debug.run_scan_match_debug(*source_and_target_files)
    assert calls == list(source_and_target_files) * 2


@pytest.mark.parametrize('method', ['icp', 'gicp'])
def test_actual_solver_receives_identical_inputs_for_every_frame(tmp_path, source_and_target_files, monkeypatch, method):
    options = files(tmp_path, source_and_target_files)
    options['scan_match_method'] = method
    solve, captured = debug.register_detailed, []
    def traced(source, target, **kwargs):
        captured.append((tuple((np.asarray(getattr(cloud, field)).shape,
                                np.asarray(getattr(cloud, field)).tobytes())
                               for cloud in [source, target]
                               for field in ['points', 'normals', 'colors', 'covariances']),
                         {key: value for key, value in kwargs.items() if key != '_target_normals'}))
        return solve(source, target, **kwargs)
    monkeypatch.setattr(debug, 'register_detailed', traced)
    with monkeypatch.context() as patch:
        patch.setattr(slam, '_ScanMatchMapContext', FreshContext)
        fresh = slam.analyze_slam_run(**options)
    reuse = slam.analyze_slam_run(**options)
    assert captured[:3] == captured[3:]
    for result in [fresh, reuse]:
        for frame in result['selected_frames']:
            assert frame['scan_match_debug_error'] is None
            registered = frame['scan_match_debug_result']['registration']
            assert registered['fitness'] > .5 and np.isfinite(registered['inlier_rmse'])
