"""Numerical stationary I/O and lifecycle checks, not real-data accuracy."""
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cloudanalyzer_core")
from test_imu_preintegration import EPOCH, _write  # noqa: E402


@pytest.fixture
def recording(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[4]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    replay = importlib.import_module("replay_nclt_imu_window")
    times = EPOCH + np.arange(16, dtype=np.int64) * 20_000_000
    imu_times = EPOCH + np.r_[np.arange(0, 121, 10), np.arange(190, 301, 10)].astype(np.int64) * 1_000_000
    _write(tmp_path, imu_times, np.zeros((len(imu_times), 3)), np.tile([0., 0., -9.81], (len(imu_times), 1)),
           frames=["nclt_ms25_native"] * len(imu_times))
    euler = tmp_path / "euler.csv"
    euler.write_text(f"{EPOCH // 1000},0,0,0\n")
    return replay, times, tmp_path / "imu", euler


def baseline(module, path, times, body):
    path.mkdir()
    np.savez_compressed(path / "lidar.npz", timestamps_ns=times, body_poses=body)
    receipt = {"schema": module.pilot.SCHEMA, "interpreted_scan_frame": module.frames.FRAME_ID,
               "inputs": {"frame_module": module.pilot.artifact(Path(module.frames.__file__))},
               "outputs": {"lidar.npz": module.pilot.artifact(path / "lidar.npz")}}
    (path / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_bounded_replay_marginalizes_resets_at_gaps_and_scores_every_pose(recording, tmp_path):
    module, times, imu, euler = recording
    body = np.tile(np.eye(4), (len(times), 1, 1))
    source = baseline(module, tmp_path / "baseline", times, body)
    output = tmp_path / "replay"
    report = module.replay(source, imu, euler, output, imu_topic="/imu")
    assert report["gap_resets"] == [6, 7, 8, 9]
    assert report["coverage"]["scans"] == 16
    assert report["coverage"]["fused_updates"] == 11
    assert report["coverage"]["gap_fallbacks"] == 4
    assert report["coverage"]["max_active_states"] == 4
    assert report["coverage"]["marginalizations"] == 5
    assert report["coverage"]["nonconverged_updates"] == 0
    # The post-gap prior starts at scan 10; no marginalized pre-gap information survives.
    assert [m["dropped_timestamp_ns"] for m in report["marginalizations"]] == [int(times[i]) for i in [0, 1, 2, 10, 11]]
    with np.load(output / "trajectory.npz") as data:
        assert list(data["status"]) == ["lidar_start"] + ["fused"] * 6 + ["lidar_gap"] * 4 + ["fused"] * 5
        np.testing.assert_array_equal(data["replay_body_poses"][7:11], body[7:11])
        np.testing.assert_allclose(data["replay_body_poses"], body, atol=1e-8)
    truth = tmp_path / "truth.csv"
    truth.write_text("".join(f"{EPOCH // 1000 + k * 5000},0,0,0,0,0,0\n" for k in range(62)))
    evaluator = importlib.import_module("evaluate_nclt_imu_replay")
    before = module.pilot.artifact(output / "receipt.json")
    metrics = evaluator.evaluate(output, truth, tmp_path / "evaluation")["metrics"]["replay"]
    assert metrics["all"]["rpe_translation_m"]["count"] == 15
    assert metrics["valid_imu_edges"]["rpe_translation_m"]["count"] == 11
    assert metrics["gap_edges"]["rpe_translation_m"]["count"] == 4
    assert metrics["all"]["ate_m"]["count"] == 16
    assert metrics["all"]["rpe_translation_m"]["rmse"] < 1e-8
    assert module.pilot.artifact(output / "receipt.json") == before
    with pytest.raises(FileExistsError):
        module.replay(source, imu, euler, output, imu_topic="/imu")
    with pytest.raises(ValueError, match="outside"):
        evaluator.evaluate(output, truth, output / "evaluation")


def test_later_lidar_observation_cannot_change_already_emitted_states(recording, tmp_path):
    module, times, imu, euler = recording
    body = np.tile(np.eye(4), (len(times), 1, 1))
    first = baseline(module, tmp_path / "baseline-first", times, body)
    module.replay(first, imu, euler, tmp_path / "replay-first", imu_topic="/imu")
    body[-1, :3, 3] += [.01, -.02, .005]
    changed = baseline(module, tmp_path / "baseline-changed", times, body)
    module.replay(changed, imu, euler, tmp_path / "replay-changed", imu_topic="/imu")
    with np.load(tmp_path / "replay-first/trajectory.npz") as a, np.load(tmp_path / "replay-changed/trajectory.npz") as b:
        np.testing.assert_array_equal(a["replay_body_poses"][:-1], b["replay_body_poses"][:-1])
        np.testing.assert_array_equal(a["velocities"][:-1], b["velocities"][:-1])
        np.testing.assert_array_equal(a["biases"][:-1], b["biases"][:-1])
        assert np.linalg.norm(a["replay_body_poses"][-1] - b["replay_body_poses"][-1]) > 1e-6
