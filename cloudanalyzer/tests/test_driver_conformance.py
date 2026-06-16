"""Tests for third-party SLAM driver conformance helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from ca.core.slam_run import SlamRunRequest, SlamRunResult, load_frame
from ca.testing.conformance import run_slam_driver_conformance


class _ConformantTinyDriver:
    name = "tiny_conformant"

    def run(self, request: SlamRunRequest) -> SlamRunResult:
        frame_paths = request.frame_paths
        timestamps = (
            np.asarray(request.timestamps_s, dtype=np.float64)
            if request.timestamps_s is not None
            else np.arange(len(frame_paths), dtype=np.float64) * request.frame_period_s
        )
        poses = np.broadcast_to(np.eye(4), (len(frame_paths), 4, 4)).copy()
        chunks = [load_frame(path) for path in frame_paths]
        map_points = np.vstack(chunks)
        return SlamRunResult(
            driver=self.name,
            poses=poses,
            timestamps_s=timestamps,
            map_points=map_points,
            runtime_s=0.0,
            frames_processed=len(frame_paths),
            metadata={"tiny_conformant": {"fixture": True}},
        )


class _WrongShapeDriver(_ConformantTinyDriver):
    name = "wrong_shape"

    def run(self, request: SlamRunRequest) -> SlamRunResult:
        result = super().run(request)
        return SlamRunResult(
            driver=result.driver,
            poses=result.poses[:, :3, :3],
            timestamps_s=result.timestamps_s,
            map_points=result.map_points,
            runtime_s=result.runtime_s,
            frames_processed=result.frames_processed,
            metadata=result.metadata,
        )


class _NoisyDriver(_ConformantTinyDriver):
    name = "noisy"

    def run(self, request: SlamRunRequest) -> SlamRunResult:
        print("unexpected plugin print")
        return super().run(request)


def test_slam_driver_conformance_passes_for_minimal_driver(tmp_path: Path) -> None:
    summary = run_slam_driver_conformance(_ConformantTinyDriver, tmp_path=tmp_path)

    assert summary["driver"] == "tiny_conformant"
    assert summary["frames_processed"] == 3
    assert summary["map_points"] > 0
    assert summary["metadata_keys"] == ["tiny_conformant"]
    assert (tmp_path / "artifacts" / "trajectory.tum").is_file()
    assert (tmp_path / "artifacts" / "map.ply").is_file()


def test_slam_driver_conformance_passes_for_canonical_example_plugin(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    plugin_src = repo_root / "plugins" / "cloudanalyzer-driver-example" / "src"
    sys.path.insert(0, str(plugin_src))
    try:
        from cloudanalyzer_driver_example.driver import Open3DICPSlamDriver

        summary = run_slam_driver_conformance(Open3DICPSlamDriver, tmp_path=tmp_path)
    finally:
        try:
            sys.path.remove(str(plugin_src))
        except ValueError:
            pass

    assert summary["driver"] == "open3d_icp"
    assert summary["frames_processed"] == 3
    assert "open3d_icp" in summary["metadata_keys"]


def test_slam_driver_conformance_rejects_wrong_pose_shape(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="poses must have shape"):
        run_slam_driver_conformance(_WrongShapeDriver, tmp_path=tmp_path)


def test_slam_driver_conformance_rejects_stdout_prints(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="stdout/stderr"):
        run_slam_driver_conformance(_NoisyDriver, tmp_path=tmp_path)
