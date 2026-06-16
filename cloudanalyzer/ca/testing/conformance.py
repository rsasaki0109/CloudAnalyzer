"""Conformance helpers for third-party CloudAnalyzer plugins."""

from __future__ import annotations

import contextlib
import io
import math
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ca.core.slam_run import (
    SlamRunDriver,
    SlamRunRequest,
    SlamRunResult,
    write_map_ply,
    write_tum_trajectory,
)


DriverLike = type[SlamRunDriver] | Callable[[], SlamRunDriver] | SlamRunDriver


def _instantiate_driver(driver_like: DriverLike) -> SlamRunDriver:
    if isinstance(driver_like, type):
        driver = driver_like()
    elif callable(driver_like):
        driver = driver_like()
    elif isinstance(driver_like, SlamRunDriver):
        driver = driver_like
    else:
        raise AssertionError("driver must be a SlamRunDriver instance, class, or factory")
    if not isinstance(driver, SlamRunDriver):
        raise AssertionError(
            "driver must satisfy ca.core.slam_run.SlamRunDriver "
            "(name attribute + run(request) method)"
        )
    return driver


def _write_fixture_frame(path: Path, offset_x: float) -> None:
    import open3d as o3d

    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.6],
            [1.5, 0.3, 0.2],
        ],
        dtype=np.float64,
    )
    points = base + np.array([offset_x, 0.0, 0.0], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), pcd, write_ascii=True):
        raise RuntimeError(f"failed to write fixture frame: {path}")


def _fixture_request(root: Path) -> SlamRunRequest:
    frames_dir = root / "frames"
    frame_paths = []
    for index, offset in enumerate((0.0, 0.1, 0.2)):
        path = frames_dir / f"{index:06d}.pcd"
        _write_fixture_frame(path, offset)
        frame_paths.append(path)
    return SlamRunRequest(
        frame_paths=tuple(frame_paths),
        timestamps_s=(0.0, 0.1, 0.2),
        frame_period_s=0.1,
        max_range_m=10.0,
        voxel_size_m=0.05,
        deskew=False,
    )


def _run_quietly(driver: SlamRunDriver, request: SlamRunRequest) -> SlamRunResult:
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = driver.run(request)
    except ImportError as exc:
        raise AssertionError(
            f"driver {driver.name!r} raised ImportError during run: {exc}. "
            "Optional dependencies should be installed for conformance or reported "
            "with a clear package extra before the driver is selected."
        ) from exc
    if stdout.getvalue().strip() or stderr.getvalue().strip():
        raise AssertionError(
            f"driver {driver.name!r} wrote to stdout/stderr during run. "
            "Use logging or metadata instead of printing from plugins."
        )
    return result


def _assert_result_contract(
    driver: SlamRunDriver,
    result: SlamRunResult,
    request: SlamRunRequest,
) -> None:
    if not isinstance(result, SlamRunResult):
        raise AssertionError("driver.run() must return SlamRunResult")
    if not isinstance(driver.name, str) or not driver.name.strip():
        raise AssertionError("driver.name must be a non-empty string")
    if result.driver != driver.name:
        raise AssertionError(
            f"result.driver must match driver.name ({driver.name!r}); got {result.driver!r}"
        )
    if not isinstance(result.metadata, dict):
        raise AssertionError("result.metadata must be a dict")
    if result.metadata == {}:
        raise AssertionError("result.metadata must include driver/provenance fields")
    if not math.isfinite(float(result.runtime_s)) or float(result.runtime_s) < 0:
        raise AssertionError("result.runtime_s must be finite and non-negative")
    if not isinstance(result.frames_processed, int) or result.frames_processed <= 0:
        raise AssertionError("result.frames_processed must be a positive int")
    if result.frames_processed > len(request.frame_paths):
        raise AssertionError("result.frames_processed cannot exceed request.frame_paths")

    poses = np.asarray(result.poses)
    timestamps = np.asarray(result.timestamps_s)
    map_points = np.asarray(result.map_points)

    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise AssertionError(f"result.poses must have shape (N, 4, 4); got {poses.shape}")
    if poses.shape[0] != result.frames_processed:
        raise AssertionError("result.poses length must equal result.frames_processed")
    if timestamps.shape != (result.frames_processed,):
        raise AssertionError("result.timestamps_s shape must equal (frames_processed,)")
    if map_points.ndim != 2 or map_points.shape[1] != 3:
        raise AssertionError(f"result.map_points must have shape (M, 3); got {map_points.shape}")
    if map_points.shape[0] == 0:
        raise AssertionError("result.map_points must contain at least one point")
    if not np.all(np.isfinite(poses)):
        raise AssertionError("result.poses contains non-finite values")
    if not np.all(np.isfinite(timestamps)):
        raise AssertionError("result.timestamps_s contains non-finite values")
    if not np.all(np.isfinite(map_points)):
        raise AssertionError("result.map_points contains non-finite values")
    if not np.all(np.diff(timestamps) >= 0):
        raise AssertionError("result.timestamps_s must be monotonic")

    bottom = poses[:, 3, :]
    expected_bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), bottom.shape)
    if not np.allclose(bottom, expected_bottom, atol=1e-6):
        raise AssertionError("each pose must be a homogeneous 4x4 transform")
    rotations = poses[:, :3, :3]
    identity = np.eye(3)
    for index, rotation in enumerate(rotations):
        if not np.allclose(rotation.T @ rotation, identity, atol=1e-4):
            raise AssertionError(f"pose {index} rotation is not orthonormal")
        det = np.linalg.det(rotation)
        if not np.isfinite(det) or not np.isclose(det, 1.0, atol=1e-3):
            raise AssertionError(f"pose {index} rotation determinant must be near 1")


def _assert_artifact_writers(result: SlamRunResult, root: Path) -> None:
    trajectory_path = root / "trajectory.tum"
    map_path = root / "map.ply"
    write_tum_trajectory(trajectory_path, result.poses, result.timestamps_s)
    write_map_ply(map_path, result.map_points)
    if not trajectory_path.is_file() or trajectory_path.stat().st_size == 0:
        raise AssertionError("write_tum_trajectory did not produce a non-empty file")
    if not map_path.is_file() or map_path.stat().st_size == 0:
        raise AssertionError("write_map_ply did not produce a non-empty file")


def _assert_deterministic(
    driver_like: DriverLike,
    first: SlamRunResult,
    request: SlamRunRequest,
) -> None:
    second_driver = _instantiate_driver(driver_like)
    second = _run_quietly(second_driver, request)
    _assert_result_contract(second_driver, second, request)
    if first.frames_processed != second.frames_processed:
        raise AssertionError("driver must process a deterministic number of frames")
    if not np.allclose(first.timestamps_s, second.timestamps_s, atol=1e-9):
        raise AssertionError("driver timestamps must be deterministic")
    if not np.allclose(first.poses, second.poses, atol=1e-6):
        raise AssertionError("driver poses must be deterministic on the fixture")
    if first.map_points.shape != second.map_points.shape:
        raise AssertionError("driver map point shape must be deterministic")
    if not np.allclose(first.map_points, second.map_points, atol=1e-6):
        raise AssertionError("driver map points must be deterministic on the fixture")


def run_slam_driver_conformance(
    driver: DriverLike,
    *,
    tmp_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run a compact conformance suite for a ``SlamRunDriver`` implementation.

    Intended use in third-party plugin packages:

    .. code-block:: python

       from ca.testing.conformance import run_slam_driver_conformance
       from my_pkg.driver import MyDriver

       def test_driver_contract(tmp_path):
           run_slam_driver_conformance(MyDriver, tmp_path=tmp_path)

    The helper creates a tiny three-frame point-cloud fixture, runs the driver
    twice, validates the ``SlamRunResult`` contract, verifies artifact writers,
    and rejects stdout/stderr printing from the plugin.
    """

    if tmp_path is None:
        with tempfile.TemporaryDirectory(prefix="cloudanalyzer-conformance-") as tmp:
            return run_slam_driver_conformance(driver, tmp_path=tmp)

    root = Path(tmp_path)
    request = _fixture_request(root)
    instance = _instantiate_driver(driver)
    result = _run_quietly(instance, request)
    _assert_result_contract(instance, result, request)
    _assert_artifact_writers(result, root / "artifacts")
    _assert_deterministic(driver, result, request)

    return {
        "driver": result.driver,
        "frames_processed": result.frames_processed,
        "map_points": int(result.map_points.shape[0]),
        "metadata_keys": sorted(result.metadata),
    }


__all__ = ["run_slam_driver_conformance"]
