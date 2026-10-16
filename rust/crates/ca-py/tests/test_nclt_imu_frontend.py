"""Causality and reset checks; fixtures are not real trajectory benchmarks."""
import importlib
from pathlib import Path

import numpy as np
import pytest

core = pytest.importorskip("cloudanalyzer_core")
from test_imu_preintegration import EPOCH, _write  # noqa: E402
from test_scan_header import _write as write_scans  # noqa: E402


@pytest.fixture
def recording(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    module = importlib.import_module("replay_nclt_imu_frontend")
    times = EPOCH + np.arange(16, dtype=np.int64) * 20_000_000
    imu_times = EPOCH + np.r_[np.arange(0, 121, 10), np.arange(190, 301, 10)].astype(np.int64) * 1_000_000
    reader = _write(tmp_path, imu_times, np.zeros((len(imu_times), 3)), np.tile([0., 0., -9.81], (len(imu_times), 1)),
                    frames=["nclt_ms25_native"] * len(imu_times))
    config = module.pilot.settings()
    config["gravity"] = [0., 0., -9.81]
    return module, reader.imu_series(), times, config


def test_frontend_resets_prediction_and_window_at_gaps_without_replacing_icp_poses(recording):
    module, series, times, config = recording
    frontend = module.Frontend(core, series, times, config, use_prediction=True)
    control = core.LidarOdometry(**config["lidar"])
    points = np.array([[x, y, z] for x in [-4., 0., 4.] for y in [-3., 0., 3.] for z in [-2., 1., 2.]])
    for index in range(len(times)):
        frontend.register(index, points)
        expected = control.register(points)
        np.testing.assert_allclose(frontend.poses[-1], expected, atol=1e-10, rtol=0.)
    assert frontend.status == ["lidar_start", "lidar_bootstrap"] + ["imu_prediction"] * 5 + ["lidar_gap"] * 4 + ["lidar_bootstrap"] + ["imu_prediction"] * 4
    assert frontend.coverage() | {"icp_iterations": 0} == {
        "scans": 16, "all_edges": 15, "prediction_available": 9, "prediction_used": 9,
        "gap_fallbacks": 4, "bootstrap_fallbacks": 2, "startup_fallbacks": 1,
        "max_active_states": 4, "marginalizations": 5, "nonconverged_updates": 0, "icp_iterations": 0}
    assert [m["dropped_timestamp_ns"] for m in frontend.marginalizations] == [int(times[i]) for i in [0, 1, 2, 10, 11]]
    np.testing.assert_array_equal(frontend.poses, frontend.odometry.poses())
    with pytest.raises(ValueError, match="once"):
        frontend.register(15, points)


def test_current_scan_cannot_change_its_imu_prediction_or_any_preceding_output(recording):
    module, series, times, config = recording
    a = module.Frontend(core, series, times, config, use_prediction=True)
    b = module.Frontend(core, series, times, config, use_prediction=True)
    points = np.array([[x, y, z] for x in [-4., 0., 4.] for y in [-3., 0., 3.] for z in [-2., 1., 2.]])
    for i in range(len(times)):
        a.register(i, points)
        b.register(i, points if i < len(times) - 1 else points + [.05, -.02, .01])
    np.testing.assert_array_equal(a.predictions, b.predictions)
    np.testing.assert_array_equal(a.poses[:-1], b.poses[:-1])
    np.testing.assert_array_equal(a.velocities[:-1], b.velocities[:-1])
    np.testing.assert_array_equal(a.biases[:-1], b.biases[:-1])
    assert np.linalg.norm(a.poses[-1] - b.poses[-1]) > .01


def test_replay_io_and_evaluator_keep_every_scan_and_freeze_estimation(recording, tmp_path):
    module, series, times, config = recording
    lidar_dir = tmp_path / "lidar"
    lidar_dir.mkdir()
    lidar = write_scans(lidar_dir, [int(t) for t in times], [module.frames.FRAME_ID] * len(times), "cdr")
    euler = tmp_path / "euler.csv"
    euler.write_text(f"{EPOCH // 1000},0,0,0\n")
    out = tmp_path / "estimation"
    report = module.replay(next(lidar.glob("*.mcap")), tmp_path / "imu", euler, out, imu_topic="/imu")
    assert report["arms"]["baseline"]["coverage"]["prediction_used"] == 0
    assert report["arms"]["assisted"]["coverage"]["prediction_used"] == 9
    truth = tmp_path / "truth.csv"
    truth.write_text("".join(f"{EPOCH // 1000 + k * 5000},0,0,0,0,0,0\n" for k in range(62)))
    evaluator = importlib.import_module("evaluate_nclt_imu_frontend")
    before = module.pilot.artifact(out / "receipt.json")
    result = evaluator.evaluate(out, truth, tmp_path / "evaluation")
    assert module.pilot.artifact(out / "receipt.json") == before
    for arm in ["baseline", "assisted"]:
        assert result["metrics"][arm]["all"]["ate_m"]["count"] == 16
        assert result["metrics"][arm]["all"]["rpe_translation_m"]["count"] == 15
        assert result["metrics"][arm]["prediction_edges"]["rpe_translation_m"]["count"] == 9
        assert result["metrics"][arm]["fallback_edges"]["rpe_translation_m"]["count"] == 6
    with pytest.raises(FileExistsError):
        module.replay(next(lidar.glob("*.mcap")), tmp_path / "imu", euler, out, imu_topic="/imu")
    with pytest.raises(ValueError, match="outside"):
        evaluator.evaluate(out, truth, out / "evaluation")
    with (out / "trajectory.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        evaluator.evaluate(out, truth, tmp_path / "tampered-evaluation")
