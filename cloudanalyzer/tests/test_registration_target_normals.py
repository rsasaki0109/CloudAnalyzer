"""Owned pre-solver target normals preserve registration inputs and outputs."""

from types import SimpleNamespace
import weakref

import numpy as np
import open3d as o3d
import pytest

import ca.registration as registration
import ca.scan_match_debug as debug
import ca.slam_debug as slam
from ca.registration import _TargetNormalCache, register_detailed
from .test_slam_map_reuse import files, normalized


def signature(cloud):
    return tuple((np.asarray(getattr(cloud, field)).shape, np.asarray(getattr(cloud, field)).tobytes())
                 for field in ['points', 'normals', 'colors', 'covariances'])


def identity_result():
    return SimpleNamespace(transformation=np.eye(4), fitness=1., inlier_rmse=0.)


@pytest.mark.parametrize('kind', ['random', 'plane', 'point', 'duplicate', 'covariances'])
def test_cache_owns_exact_readonly_normals_and_working_copies(simple_pcd, kind):
    cloud = o3d.geometry.PointCloud(simple_pcd)
    if kind == 'plane':
        np.asarray(cloud.points)[:, 2] = 0.
    elif kind in ['point', 'duplicate']:
        cloud.points = o3d.utility.Vector3dVector(np.zeros((1 if kind == 'point' else 100, 3)))
    elif kind == 'covariances':
        cloud.covariances = o3d.utility.Matrix3dVector(np.tile(np.diag([1., 2., 3.]), (len(cloud.points), 1, 1)))
    expected = o3d.geometry.PointCloud(cloud)
    registration._estimate_registration_normals(expected)
    cache = _TargetNormalCache()
    for _ in range(3):
        target = o3d.geometry.PointCloud(cloud)
        cache.prepare(target)
        assert signature(target) == signature(expected)
        assert cache._normals.flags.owndata and not cache._normals.flags.writeable
        assert not np.shares_memory(cache._normals, np.asarray(target.normals))
        np.asarray(target.normals)[:] = 9.
        assert cache._normals.tobytes() == np.asarray(expected.normals).tobytes()
    reference = weakref.ref(cache._normals)
    cache.clear()
    assert reference() is None
    assert np.all(np.asarray(target.normals) == 9.)


@pytest.mark.parametrize('method', ['icp', 'gicp'])
@pytest.mark.parametrize('attributes', ['missing', 'normals', 'covariances'])
def test_actual_native_solver_receives_identical_prepared_fields(simple_pcd, shifted_pcd, monkeypatch, method, attributes):
    base_target = o3d.geometry.PointCloud(shifted_pcd)
    if attributes == 'normals':
        base_target.normals = o3d.utility.Vector3dVector(np.tile([0., 0., 1.], (len(base_target.points), 1)))
    elif attributes == 'covariances':
        base_target.covariances = o3d.utility.Matrix3dVector(np.tile(np.diag([1., 2., 3.]), (len(base_target.points), 1, 1)))
    native_name = 'registration_icp' if method == 'icp' else 'registration_generalized_icp'
    native = getattr(o3d.pipelines.registration, native_name)
    estimate = registration._estimate_registration_normals
    captured, estimates = [], []
    def traced(source, target, distance, initial, estimator):
        captured.append((signature(source), signature(target), distance, initial.tobytes(), type(estimator)))
        return native(source, target, distance, initial, estimator)
    def traced_estimate(cloud):
        estimates.append(1)
        estimate(cloud)
    monkeypatch.setattr(o3d.pipelines.registration, native_name, traced)
    monkeypatch.setattr(registration, '_estimate_registration_normals', traced_estimate)
    for cache in [None, _TargetNormalCache()]:
        for index in range(3):
            source = o3d.geometry.PointCloud(simple_pcd)
            source.translate([index*.002, 0., 0.])
            target = o3d.geometry.PointCloud(base_target)
            result = register_detailed(source, target, method=method, _target_normals=cache)
            assert result.fitness > .5 and np.isfinite(result.rmse)
        if cache is None:
            assert len(estimates) == (3 if attributes == 'normals' else 6)
            estimates.clear()
        else:
            assert len(estimates) == (3 if attributes == 'normals' else 4)
            cache.clear()
    assert captured[:3] == captured[3:]


@pytest.mark.parametrize('method', ['icp', 'gicp'])
def test_snapshot_is_taken_before_solver_mutates_target(simple_pcd, shifted_pcd, monkeypatch, method):
    cache = _TargetNormalCache()
    inputs = []
    def solver(source, target, *args):
        inputs.append(signature(target))
        np.asarray(target.normals)[:] = [11., 12., 13.]
        target.colors = o3d.utility.Vector3dVector(np.tile([.2, .3, .4], (len(target.points), 1)))
        return identity_result()
    monkeypatch.setattr(o3d.pipelines.registration, 'registration_icp' if method == 'icp' else 'registration_generalized_icp', solver)
    for _ in range(3):
        register_detailed(o3d.geometry.PointCloud(simple_pcd), o3d.geometry.PointCloud(shifted_pcd),
                          method=method, _target_normals=cache)
    assert inputs[0] == inputs[1] == inputs[2]
    assert not np.all(cache._normals == [11., 12., 13.])


@pytest.mark.parametrize('invalid', ['method', 'matrix', 'source_failure'])
def test_validation_and_source_first_failures_do_not_publish_target_normals(simple_pcd, shifted_pcd, monkeypatch, invalid):
    cache, calls = _TargetNormalCache(), []
    def estimate(cloud):
        calls.append(cloud)
        raise RuntimeError('source normal failure')
    monkeypatch.setattr(registration, '_estimate_registration_normals', estimate)
    with pytest.raises((ValueError, RuntimeError)):
        register_detailed(simple_pcd, shifted_pcd, method='bad' if invalid == 'method' else 'icp',
                          init_transform=np.eye(3) if invalid == 'matrix' else None, _target_normals=cache)
    assert len(calls) == int(invalid == 'source_failure')
    assert cache._normals is None and not shifted_pcd.has_normals()


def test_nonfinite_normals_are_not_retained(simple_pcd, monkeypatch):
    calls = []
    def estimate(cloud):
        calls.append(1)
        cloud.normals = o3d.utility.Vector3dVector(np.full((len(cloud.points), 3), np.nan))
    monkeypatch.setattr(registration, '_estimate_registration_normals', estimate)
    cache = _TargetNormalCache()
    for _ in range(2):
        cache.prepare(o3d.geometry.PointCloud(simple_pcd))
        assert cache._normals is None
    assert len(calls) == 2


class FreshNormals:
    def prepare(self, target):
        registration._estimate_registration_normals(target)

    def clear(self):
        pass


@pytest.mark.parametrize('method,crop,voxel', [('icp', None, None), ('gicp', None, .05), ('icp', .1, None),
                                             ('gicp', 0., None), ('icp', -1., .05)])
def test_full_slam_reports_and_ply_bytes_match_recomputed_normals(
    tmp_path, source_and_target_files, monkeypatch, method, crop, voxel,
):
    options = files(tmp_path, source_and_target_files)
    options.update(scan_match_method=method, scan_match_crop_margin=crop, scan_match_map_voxel_size=voxel,
                   scan_match_threshold=.05)
    prepared, estimate = [], registration._estimate_registration_normals
    def traced(cloud):
        prepared.append(signature(cloud)[0])
        return estimate(cloud)
    def solver(source, target, *args):
        return identity_result()
    monkeypatch.setattr(registration, '_estimate_registration_normals', traced)
    monkeypatch.setattr(o3d.pipelines.registration, 'registration_icp' if method == 'icp' else 'registration_generalized_icp', solver)
    fresh_dir, reuse_dir = tmp_path / 'fresh', tmp_path / 'reuse'
    with monkeypatch.context() as patch:
        patch.setattr(debug, '_TargetNormalCache', FreshNormals)
        expected = slam.analyze_slam_run(**options, artifact_dir=str(fresh_dir))
    assert len(prepared) == 6
    prepared.clear()
    observed = slam.analyze_slam_run(**options, artifact_dir=str(reuse_dir))
    assert len(prepared) == (6 if crop is not None and crop > 0 else 4)
    assert normalized(expected, fresh_dir) == normalized(observed, reuse_dir)
    assert all(frame['scan_match_debug_error'] is None for frame in observed['selected_frames'])
    paths = sorted(path.relative_to(fresh_dir) for path in fresh_dir.rglob('*.ply'))
    assert len(paths) == 9
    for path in paths:
        assert (fresh_dir / path).read_bytes() == (reuse_dir / path).read_bytes()


@pytest.mark.parametrize('failure', ['normal_estimation', 'solver'])
def test_failed_frame_discards_normal_snapshot_and_later_frames_retry(tmp_path, source_and_target_files, monkeypatch, failure):
    options = files(tmp_path, source_and_target_files)
    options['scan_match_method'] = 'gicp'
    target_first = np.asarray(debug.load_point_cloud(options['map_path']).points)[0].tobytes()
    estimate = registration._estimate_registration_normals
    calls = dict(target=0, solver=0)
    references = []
    def traced(cloud):
        if np.asarray(cloud.points)[0].tobytes() == target_first:
            calls['target'] += 1
            if failure == 'normal_estimation' and calls['target'] == 1:
                raise RuntimeError('normal failure')
        return estimate(cloud)
    original_prepare = _TargetNormalCache.prepare
    def cache_prepare(self, target):
        original_prepare(self, target)
        if self._normals is not None:
            references.append(weakref.ref(self._normals))
    def solver(*args):
        calls['solver'] += 1
        if failure == 'solver' and calls['solver'] == 1:
            raise RuntimeError('solver failure')
        return identity_result()
    monkeypatch.setattr(registration, '_estimate_registration_normals', traced)
    monkeypatch.setattr(_TargetNormalCache, 'prepare', cache_prepare)
    monkeypatch.setattr(o3d.pipelines.registration, 'registration_generalized_icp', solver)
    result = slam.analyze_slam_run(**options)
    assert result['selected_frames'][0]['scan_match_debug_error'] is not None
    assert all(frame['scan_match_debug_error'] is None for frame in result['selected_frames'][1:])
    assert calls['target'] == 2
    assert all(reference() is None for reference in references)
