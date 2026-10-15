"""Numeric reports omit unused colors while snapshots retain their exact pixels."""

import json

import numpy as np
import open3d as o3d
import pytest

import ca.compare as compare
from ca.metrics import compute_nn_distance, summarize, threshold_stats
from ca.visualization import colorize, save_snapshot


def fixed_register(source, target, method):
    result = o3d.geometry.PointCloud(source)
    result.translate([.003, -.002, .001])
    return result, .91, .02


@pytest.mark.parametrize('method', [None, 'icp', 'gicp'])
@pytest.mark.parametrize('write_reports', [False, True])
def test_numeric_outputs_do_not_require_colors(source_and_target_files, tmp_path, monkeypatch, method, write_reports):
    source, target = source_and_target_files
    monkeypatch.setattr(compare, 'register', fixed_register)
    def unexpected(*args, **kwargs):
        raise AssertionError('numeric reports have no color consumer')
    monkeypatch.setattr(compare, 'colorize', unexpected)
    monkeypatch.setattr(compare, 'save_snapshot', unexpected)
    json_path, markdown_path = tmp_path / 'report.json', tmp_path / 'report.md'
    result = compare.run_compare(source, target, method=method, threshold=.05,
        json_path=str(json_path) if write_reports else None,
        report_path=str(markdown_path) if write_reports else None)
    cloud, reference = compare.load_point_cloud(source), compare.load_point_cloud(target)
    if method is not None:
        cloud, _, _ = fixed_register(cloud, reference, method)
    distances = compute_nn_distance(cloud, reference)
    assert result == dict(source_points=len(cloud.points), target_points=len(reference.points),
                          fitness=.91 if method else None, rmse=.02 if method else None,
                          distance_stats=summarize(distances), threshold=threshold_stats(distances, .05))
    if write_reports:
        assert json.loads(json_path.read_text()) == result
        assert '# CloudAnalyzer Report' in markdown_path.read_text()
    else:
        assert not json_path.exists() and not markdown_path.exists()


@pytest.mark.parametrize('method,identical', [(None, False), ('icp', False), ('gicp', False), (None, True)])
def test_snapshot_png_and_numeric_report_bytes_preserve_color_pipeline(
    source_and_target_files, tmp_path, monkeypatch, method, identical,
):
    monkeypatch.setenv('CLOUDANALYZER_RENDER_BACKEND', 'matplotlib')
    monkeypatch.setattr(compare, 'register', fixed_register)
    source, target = source_and_target_files
    if identical:
        target = source
    cloud, reference = compare.load_point_cloud(source), compare.load_point_cloud(target)
    if method is not None:
        cloud, _, _ = fixed_register(cloud, reference, method)
    distances = compute_nn_distance(cloud, reference)
    expected_cloud = colorize(o3d.geometry.PointCloud(cloud), distances)
    expected_path, observed_path = tmp_path / 'expected.png', tmp_path / 'observed.png'
    save_snapshot(expected_cloud, str(expected_path))
    captured, original_save = [], compare.save_snapshot
    def snapshot(cloud, path):
        captured.append(1)
        for field in ['points', 'normals', 'colors', 'covariances']:
            assert np.asarray(getattr(cloud, field)).tobytes() == np.asarray(getattr(expected_cloud, field)).tobytes()
        return original_save(cloud, path)
    monkeypatch.setattr(compare, 'save_snapshot', snapshot)
    with_image = compare.run_compare(source, target, method=method, threshold=.05,
        snapshot_path=str(observed_path), json_path=str(tmp_path / 'image.json'), report_path=str(tmp_path / 'image.md'))
    without_image = compare.run_compare(source, target, method=method, threshold=.05,
        json_path=str(tmp_path / 'numeric.json'), report_path=str(tmp_path / 'numeric.md'))
    assert captured == [1] and with_image == without_image
    assert observed_path.read_bytes() == expected_path.read_bytes()
    assert (tmp_path / 'image.json').read_bytes() == (tmp_path / 'numeric.json').read_bytes()
    assert (tmp_path / 'image.md').read_bytes() == (tmp_path / 'numeric.md').read_bytes()


def test_empty_snapshot_path_keeps_numeric_behavior(source_and_target_files, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('empty snapshot path has no output consumer')
    monkeypatch.setattr(compare, 'colorize', unexpected)
    monkeypatch.setattr(compare, 'save_snapshot', unexpected)
    observed = compare.run_compare(*source_and_target_files, method=None, snapshot_path='')
    assert observed['fitness'] is None and observed['distance_stats']['mean'] > 0


@pytest.mark.parametrize('snapshot', [False, True])
def test_cli_reports_and_snapshot_keep_requested_output(source_and_target_files, tmp_path, monkeypatch, snapshot):
    from typer.testing import CliRunner
    from cloudanalyzer_cli.main import app
    monkeypatch.setenv('CLOUDANALYZER_RENDER_BACKEND', 'matplotlib')
    calls, original = [], compare.colorize
    def traced(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(compare, 'colorize', traced)
    json_path, markdown_path, image_path = tmp_path / 'cli.json', tmp_path / 'cli.md', tmp_path / 'cli.png'
    command = ['compare', *source_and_target_files, '--register', 'none', '--threshold', '.05',
               '--json', str(json_path), '--report', str(markdown_path)]
    if snapshot:
        command += ['--snapshot', str(image_path)]
    result = CliRunner().invoke(app, command)
    assert result.exit_code == 0, result.output
    assert len(calls) == int(snapshot)
    assert json.loads(json_path.read_text())['threshold']['threshold'] == .05
    assert '# CloudAnalyzer Report' in markdown_path.read_text()
    assert image_path.exists() == snapshot
