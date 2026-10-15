"""No-op preprocessing keeps loaders and map snapshots isolated from solvers."""

from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest

import ca.scan_match_debug as debug
import ca.slam_debug as slam
from .test_slam_map_reuse import files, normalized


def signature(cloud):
    return tuple((np.asarray(getattr(cloud, field)).shape, np.asarray(getattr(cloud, field)).tobytes())
                 for field in ['points', 'normals', 'colors', 'covariances'])


def copying_preprocess(cloud, voxel):
    if voxel is None or voxel <= 0:
        return o3d.geometry.PointCloud(cloud)
    return cloud.voxel_down_sample(float(voxel))


@pytest.mark.parametrize('method', ['icp', 'gicp'])
@pytest.mark.parametrize('voxel', [None, 0., -1., .05])
@pytest.mark.parametrize('crop', [None, .2])
def test_working_cloud_mutations_do_not_reach_loaded_clouds_or_map_snapshot(
    source_and_target_files, tmp_path, monkeypatch, method, voxel, crop,
):
    paths = source_and_target_files
    loaded = {path: debug.load_point_cloud(path) for path in paths}
    for cloud in loaded.values():
        count = len(cloud.points)
        cloud.normals = o3d.utility.Vector3dVector(np.tile([0., 0., 1.], (count, 1)))
        cloud.colors = o3d.utility.Vector3dVector(np.tile([.2, .4, .6], (count, 1)))
        cloud.covariances = o3d.utility.Matrix3dVector(np.tile(np.diag([1., 2., 3.]), (count, 1, 1)))
    originals = {path: signature(cloud) for path, cloud in loaded.items()}
    monkeypatch.setattr(debug, 'load_point_cloud', lambda path: loaded[path])
    captured = []

    def solver(source, target, distance, initial, estimator):
        captured.append((signature(source), signature(target), distance, initial.tobytes(), type(estimator)))
        assert not any(np.shares_memory(np.asarray(source.points), np.asarray(cloud.points))
                       or np.shares_memory(np.asarray(target.points), np.asarray(cloud.points))
                       for cloud in loaded.values())
        for cloud in [source, target]:
            for field in ['points', 'normals', 'colors', 'covariances']:
                np.asarray(getattr(cloud, field))[:] += .001
        matrix = np.eye(4)
        matrix[0, 3] = .002
        return SimpleNamespace(transformation=matrix, fitness=.9, inlier_rmse=.01)

    monkeypatch.setattr(o3d.pipelines.registration,
                        'registration_icp' if method == 'icp' else 'registration_generalized_icp', solver)
    angle = .01
    initial = np.array([[np.cos(angle), -np.sin(angle), 0., .003],
                        [np.sin(angle), np.cos(angle), 0., -.002], [0., 0., 1., .001], [0., 0., 0., 1.]])
    outputs = []
    for policy in ['copies', 'borrow']:
        with monkeypatch.context() as patch:
            if policy == 'copies':
                patch.setattr(debug, '_downsample_if_needed', copying_preprocess)
            context = debug._ScanMatchMapContext(paths[1], voxel)
            try:
                outputs.append([context.run(scan_path=paths[0], map_path=paths[1], method=method,
                                            scan_voxel_size=voxel, map_voxel_size=voxel, crop_margin=crop,
                                            initial_transform=initial, threshold=.05) for _ in range(3)])
                expected_map = copying_preprocess(loaded[paths[1]], voxel)
                assert signature(context._cloud) == signature(expected_map)
            finally:
                context.clear()
            assert {path: signature(cloud) for path, cloud in loaded.items()} == originals
    assert captured[:3] == captured[3:]
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('method,voxel,crop', [('icp', None, None), ('gicp', 0., None),
                                             ('icp', -1., .2), ('gicp', .05, .2)])
def test_complete_slam_reports_and_ply_bytes_match_copying_preprocessing(
    source_and_target_files, tmp_path, monkeypatch, method, voxel, crop,
):
    options = files(tmp_path, source_and_target_files)
    options.update(scan_match_method=method, scan_match_map_voxel_size=voxel,
                   scan_match_scan_voxel_size=voxel, scan_match_crop_margin=crop, scan_match_threshold=.05)
    def solver(*args):
        matrix = np.eye(4)
        matrix[:3, 3] = [.003, -.001, .002]
        return SimpleNamespace(transformation=matrix, fitness=.9, inlier_rmse=.01)
    monkeypatch.setattr(o3d.pipelines.registration,
                        'registration_icp' if method == 'icp' else 'registration_generalized_icp', solver)
    expected_dir, observed_dir = tmp_path / 'copies', tmp_path / 'borrow'
    with monkeypatch.context() as patch:
        patch.setattr(debug, '_downsample_if_needed', copying_preprocess)
        expected = slam.analyze_slam_run(**options, artifact_dir=str(expected_dir))
    observed = slam.analyze_slam_run(**options, artifact_dir=str(observed_dir))
    assert all(frame['scan_match_debug_error'] is None for frame in observed['selected_frames'])
    assert normalized(expected, expected_dir) == normalized(observed, observed_dir)
    paths = sorted(path.relative_to(expected_dir) for path in expected_dir.rglob('*.ply'))
    assert len(paths) == 9
    assert paths == sorted(path.relative_to(observed_dir) for path in observed_dir.rglob('*.ply'))
    for path in paths:
        assert (expected_dir / path).read_bytes() == (observed_dir / path).read_bytes()


@pytest.mark.parametrize('method', ['icp', 'gicp'])
def test_actual_native_solver_inputs_match_before_and_after_borrowing(
    source_and_target_files, monkeypatch, method,
):
    name = 'registration_icp' if method == 'icp' else 'registration_generalized_icp'
    native, captured = getattr(o3d.pipelines.registration, name), []
    def solver(source, target, distance, initial, estimator):
        captured.append((signature(source), signature(target), distance, initial.tobytes(), type(estimator)))
        return native(source, target, distance, initial, estimator)
    monkeypatch.setattr(o3d.pipelines.registration, name, solver)
    with monkeypatch.context() as patch:
        patch.setattr(debug, '_downsample_if_needed', copying_preprocess)
        expected = debug.run_scan_match_debug(*source_and_target_files, method=method)
    observed = debug.run_scan_match_debug(*source_and_target_files, method=method)
    assert captured[0] == captured[1]
    assert observed['registration']['fitness'] == expected['registration']['fitness']
    np.testing.assert_allclose(observed['registration']['final_transform'], expected['registration']['final_transform'], rtol=0, atol=1e-10)


def test_failed_solver_and_later_calls_leave_reused_loader_objects_unchanged(
    source_and_target_files, monkeypatch,
):
    loaded = {path: debug.load_point_cloud(path) for path in source_and_target_files}
    originals = {path: signature(cloud) for path, cloud in loaded.items()}
    monkeypatch.setattr(debug, 'load_point_cloud', lambda path: loaded[path])
    def failing(source, target, *args):
        np.asarray(source.points)[:] += 1.
        np.asarray(target.points)[:] += 2.
        raise RuntimeError('failed after mutation')
    monkeypatch.setattr(o3d.pipelines.registration, 'registration_icp', failing)
    context = debug._ScanMatchMapContext(source_and_target_files[1], None)
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match='failed after mutation'):
                context.run(scan_path=source_and_target_files[0], map_path=source_and_target_files[1], method='icp')
            context.invalidate()
            assert {path: signature(cloud) for path, cloud in loaded.items()} == originals
    finally:
        context.clear()
