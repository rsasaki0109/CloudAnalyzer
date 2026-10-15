"""Skip unused normals only for point-to-point diagnostics without artifacts."""

from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest

import ca.registration as registration
import ca.scan_match_debug as debug
import ca.slam_debug as slam
from .test_slam_map_reuse import files


def fixed_result():
    matrix = np.eye(4)
    matrix[:3, 3] = [.004, -.002, .001]
    return SimpleNamespace(transformation=matrix, fitness=.9, inlier_rmse=.01)


@pytest.mark.parametrize('method', ['icp', 'IcP', 'gicp'])
@pytest.mark.parametrize('artifacts', [False, True])
@pytest.mark.parametrize('context', [False, True])
def test_normal_preparation_is_scoped_to_consumer(
    source_and_target_files, tmp_path, monkeypatch, method, artifacts, context,
):
    estimates, native_inputs = [], []
    estimate = registration._estimate_registration_normals

    def traced(cloud):
        estimates.append(1)
        return estimate(cloud)

    def solver(source, target, *args):
        native_inputs.append((source.has_normals(), target.has_normals()))
        return fixed_result()

    monkeypatch.setattr(registration, '_estimate_registration_normals', traced)
    monkeypatch.setattr(o3d.pipelines.registration,
                        'registration_icp' if method.lower() == 'icp' else 'registration_generalized_icp', solver)
    source, target = source_and_target_files
    cache = debug._ScanMatchMapContext(target, None) if context else None
    try:
        for index in range(2):
            runner = debug.run_scan_match_debug if cache is None else cache.run
            result = runner(scan_path=source, map_path=target, method=method,
                            artifact_dir=str(tmp_path / str(index)) if artifacts else None)
            assert bool(result['artifacts']) == artifacts
        needs_normals = method.lower() == 'gicp' or artifacts
        assert native_inputs == [(needs_normals, needs_normals)] * 2
        assert len(estimates) == (4 - int(context) if needs_normals else 0)
        if cache is not None and not needs_normals:
            assert cache._target_normals._normals is None
    finally:
        if cache is not None:
            cache.clear()


@pytest.mark.parametrize('crop', [None, 0., -1., .2])
@pytest.mark.parametrize('voxel', [None, .05])
@pytest.mark.parametrize('context', [False, True])
def test_complete_reports_match_normal_preparing_control(
    source_and_target_files, tmp_path, monkeypatch, crop, voxel, context,
):
    monkeypatch.setattr(o3d.pipelines.registration, 'registration_icp', lambda *args: fixed_result())
    source, target = source_and_target_files
    initial = np.eye(4)
    initial[:3, 3] = [.003, .001, -.002]
    options = dict(scan_path=source, map_path=target, method='icp', crop_margin=crop,
                   scan_voxel_size=voxel, map_voxel_size=voxel, threshold=.05,
                   initial_transform=initial)
    solve = debug.register_detailed

    def with_normals(*args, **kwargs):
        kwargs.pop('_estimate_missing_normals', None)
        return solve(*args, **kwargs)

    def run():
        if not context:
            return [debug.run_scan_match_debug(**options)]
        cache = debug._ScanMatchMapContext(target, voxel)
        try:
            return [cache.run(**options) for _ in range(3)]
        finally:
            cache.clear()

    with monkeypatch.context() as patch:
        patch.setattr(debug, 'register_detailed', with_normals)
        expected = run()
    assert run() == expected


@pytest.mark.parametrize('attributes', ['missing', 'normals', 'covariances'])
def test_native_icp_preserves_relevant_inputs_and_existing_attributes(
    simple_pcd, shifted_pcd, monkeypatch, attributes,
):
    source, target = [o3d.geometry.PointCloud(cloud) for cloud in [simple_pcd, shifted_pcd]]
    for cloud in [source, target]:
        cloud.colors = o3d.utility.Vector3dVector(np.tile([.2, .4, .6], (len(cloud.points), 1)))
        if attributes == 'normals':
            cloud.normals = o3d.utility.Vector3dVector(np.tile([0., 0., 1.], (len(cloud.points), 1)))
        elif attributes == 'covariances':
            cloud.covariances = o3d.utility.Matrix3dVector(np.tile(np.diag([1., 2., 3.]), (len(cloud.points), 1, 1)))
    native, captured = o3d.pipelines.registration.registration_icp, []

    def traced(source, target, distance, initial, estimator):
        captured.append((tuple(np.asarray(getattr(cloud, field)).tobytes()
                               for cloud in [source, target] for field in ['points', 'colors', 'covariances']),
                         distance, initial.tobytes(), type(estimator), estimator.with_scaling))
        return native(source, target, distance, initial, estimator)

    monkeypatch.setattr(o3d.pipelines.registration, 'registration_icp', traced)
    initial = np.eye(4)
    initial[:3, 3] = [.003, -.001, .002]
    outputs = []
    for prepare in [True, False]:
        a, b = o3d.geometry.PointCloud(source), o3d.geometry.PointCloud(target)
        result = registration.register_detailed(a, b, method='icp', init_transform=initial,
                                               _estimate_missing_normals=prepare)
        outputs.append(result)
        assert a.has_normals() == b.has_normals() == (prepare or attributes == 'normals')
        assert result.transformed.has_normals() == a.has_normals()
        if attributes == 'normals':
            assert np.asarray(a.normals).tobytes() == np.asarray(source.normals).tobytes()
    assert captured[0] == captured[1]
    assert outputs[0].fitness == outputs[1].fitness
    assert outputs[0].rmse == pytest.approx(outputs[1].rmse, abs=1e-10)
    np.testing.assert_allclose(outputs[0].transformation, outputs[1].transformation, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(outputs[0].transformed.points), np.asarray(outputs[1].transformed.points), rtol=0, atol=1e-10)
    if attributes == 'normals':
        np.testing.assert_allclose(np.asarray(outputs[0].transformed.normals), np.asarray(outputs[1].transformed.normals), rtol=0, atol=1e-10)


@pytest.mark.parametrize('invalid', ['method', 'initial'])
def test_validation_precedes_optional_preparation(simple_pcd, shifted_pcd, monkeypatch, invalid):
    def unexpected(*args):
        raise AssertionError('validation must precede preparation and solver')
    monkeypatch.setattr(registration, '_estimate_registration_normals', unexpected)
    monkeypatch.setattr(o3d.pipelines.registration, 'registration_icp', unexpected)
    with pytest.raises(ValueError):
        registration.register_detailed(simple_pcd, shifted_pcd,
                                       method='invalid' if invalid == 'method' else 'icp',
                                       init_transform=np.eye(3) if invalid == 'initial' else None,
                                       _estimate_missing_normals=False)


@pytest.mark.parametrize('method,private_flag', [('icp', True), ('gicp', False)])
def test_public_default_and_gicp_keep_normal_preparation(simple_pcd, shifted_pcd, monkeypatch, method, private_flag):
    def solver(source, target, *args):
        assert source.has_normals() and target.has_normals()
        return fixed_result()
    monkeypatch.setattr(o3d.pipelines.registration,
                        'registration_icp' if method == 'icp' else 'registration_generalized_icp', solver)
    if private_flag:
        registration.register(simple_pcd, shifted_pcd, method=method)
    else:
        registration.register_detailed(simple_pcd, shifted_pcd, method=method, _estimate_missing_normals=False)
    assert simple_pcd.has_normals() and shifted_pcd.has_normals()


def test_slam_icp_failure_retries_without_allocating_normals(tmp_path, source_and_target_files, monkeypatch):
    options, calls = files(tmp_path, source_and_target_files), []
    def estimate(*args):
        raise AssertionError('diagnostic ICP must not estimate missing normals')
    def solver(source, target, *args):
        calls.append(1)
        assert not source.has_normals() and not target.has_normals()
        if len(calls) == 1:
            raise RuntimeError('transient solver failure')
        return fixed_result()
    monkeypatch.setattr(registration, '_estimate_registration_normals', estimate)
    monkeypatch.setattr(o3d.pipelines.registration, 'registration_icp', solver)
    result = slam.analyze_slam_run(**options)
    assert len(calls) == 3
    assert 'transient solver failure' in result['selected_frames'][0]['scan_match_debug_error']
    assert all(frame['scan_match_debug_error'] is None for frame in result['selected_frames'][1:])
