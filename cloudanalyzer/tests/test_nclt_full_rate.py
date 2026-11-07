"""Full-rate input fidelity and failed-attempt retention, without large datasets."""
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'scripts'))
    module = importlib.import_module('nclt_full_rate')
    for key in module.THREADS:
        monkeypatch.setenv(key, '1')
    return module


def test_decode_preserves_all_points_and_body_frame(runner):
    records = np.zeros(25, dtype=runner.POINT)
    records['x'], records['y'], records['z'] = 20400, 20600, 19200
    points = runner.decode_scan(records.tobytes())
    # Repeated points remain: no voxel thinning. Body y/z are flipped.
    np.testing.assert_allclose(points, np.tile([2., -3., 4.], (25, 1)))
    with pytest.raises(ValueError, match='8-byte'):
        runner.decode_scan(records.tobytes()[:-1])


def test_window_is_numeric_half_open_and_not_subsampled(runner, tmp_path):
    for stamp in (1, 200001, 400001, 600000001):
        (tmp_path/f'{stamp}.bin').write_bytes(b'')
    rows = runner.scan_index(tmp_path)
    assert [stamp for stamp, _ in rows] == [1000, 200001000, 400001000]
    (tmp_path/'600000001.bin').unlink()
    with pytest.raises(ValueError, match='full requested window'):
        runner.scan_index(tmp_path)


def test_duplicate_timestamp_rejected(runner, tmp_path):
    for name in ('1.bin', '01.bin', '600000001.bin'):
        (tmp_path/name).touch()
    with pytest.raises(ValueError, match='distinct'):
        runner.scan_index(tmp_path)


@pytest.mark.parametrize('fail', [False, True])
def test_streams_each_scan_and_retains_failed_attempt(runner, tmp_path, monkeypatch, fail):
    scans = tmp_path/'scans'
    scans.mkdir()
    records = np.zeros(25, dtype=runner.POINT)
    records['x'], records['y'], records['z'] = 20400, 20000, 20000
    for stamp in (1, 200001, 600000001):
        (scans/f'{stamp}.bin').write_bytes(records.tobytes())
    seen = []
    class Estimator:
        def __init__(self, **settings):
            assert settings['deskew'] is False
        def register(self, points):
            seen.append(len(points))
            if fail and len(seen) == 2:
                raise RuntimeError('registration failed')
            return np.eye(4)
    native = types.ModuleType('cloudanalyzer_core._core')
    native.__file__ = __file__
    core = types.ModuleType('cloudanalyzer_core')
    core.LidarOdometry, core._core = Estimator, native
    monkeypatch.setitem(sys.modules, 'cloudanalyzer_core', core)
    monkeypatch.setitem(sys.modules, 'cloudanalyzer_core._core', native)
    output = tmp_path/'run'
    if fail:
        with pytest.raises(RuntimeError, match='registration failed'):
            runner.estimate(scans, 'cloudanalyzer', output)
    else:
        runner.estimate(scans, 'cloudanalyzer', output)
    receipt = json.loads((output/'run.json').read_text())
    assert receipt['status'] == ('failed' if fail else 'completed')
    assert receipt['completed_frames'] == (1 if fail else 2)
    assert seen == [25, 25]
    assert len(json.loads((output/'inputs.json').read_text())) == 2
    with np.load(output/'poses.npz') as saved:
        assert len(saved['poses']) == receipt['completed_frames']
        assert saved['timestamps_ns'][0] == 1000
    with pytest.raises(FileExistsError):
        runner.estimate(scans, 'cloudanalyzer', output)


def test_invalid_source_leaves_failure_receipt(runner, tmp_path):
    output = tmp_path/'run'
    with pytest.raises(ValueError):
        runner.estimate(tmp_path/'missing', 'cloudanalyzer', output)
    assert json.loads((output/'run.json').read_text())['status'] == 'failed'
    with np.load(output/'poses.npz') as saved:
        assert saved['poses'].shape == (0, 4, 4)
