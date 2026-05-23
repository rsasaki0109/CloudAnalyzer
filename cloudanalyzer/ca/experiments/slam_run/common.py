"""Shared datasets for slam_run driver experiments.

Each dataset case synthesizes a small set of LiDAR-style frames in a temp
directory, defines the ground-truth sensor poses, and packages them into a
:class:`SlamRunRequest`. The slice evaluator runs every concrete driver
against every case and compares against the GT trajectory.

The synthetic frames are tiny on purpose — the slice's job is to verify
that the harness wires through, not to benchmark large maps. Real-scale
benchmarking happens via ``ca slam-run`` + ``ca run-evaluate`` on the
KITTI-mini / Newer-College-mini benchmark suites.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ca.core.slam_run import SlamRunRequest


@dataclass(slots=True)
class SlamRunDatasetCase:
    """One comparable SLAM run for the slice's driver bake-off."""

    name: str
    description: str
    build_request: Callable[[Path], SlamRunRequest]
    """Closure that materializes synthetic frames under the given temp dir
    and returns the request pointing at them."""

    gt_poses: np.ndarray
    """``(N, 4, 4)`` ground-truth sensor-to-world poses for ATE comparison."""

    expected_kiss_icp_max_ate_m: float
    """Pass condition for KISS-ICP on this case."""

    keep_files: tuple[str, ...] = field(default_factory=tuple)
    """Filenames the case promises to leave under the temp dir (helps tests
    assert on driver outputs without re-running the synthesis)."""


# ---------------------------------------------------------------------------
# Synthetic frame builders
# ---------------------------------------------------------------------------


def _box_scene(seed: int, n_points: int = 1500) -> np.ndarray:
    """Build a deterministic synthetic 'room' scene in the world frame.

    Returns ``(n_points, 3)`` of points lying on the floor and four walls of
    a 20 × 20 m room — enough structure for ICP to actually have something
    to register against.
    """

    rng = np.random.default_rng(seed)
    # Floor (z = 0)
    floor = rng.uniform([-10, -10, -0.02], [10, 10, 0.02], size=(n_points // 2, 3))
    # Four walls
    rem = n_points - floor.shape[0]
    per_wall = rem // 4
    walls = []
    for axis_pair in [(0, 1), (0, -1), (1, 1), (1, -1)]:
        axis, sign = axis_pair
        fixed = sign * 10.0
        sweeping = rng.uniform(-10, 10, size=(per_wall, 1))
        heights = rng.uniform(0.0, 3.0, size=(per_wall, 1))
        if axis == 0:
            pts = np.hstack(
                [np.full((per_wall, 1), fixed), sweeping, heights]
            )
        else:
            pts = np.hstack(
                [sweeping, np.full((per_wall, 1), fixed), heights]
            )
        pts += rng.normal(0, 0.02, size=pts.shape)
        walls.append(pts)
    return np.vstack([floor, *walls]).astype(np.float64)


def _pose(tx: float, ty: float, tz: float, yaw_rad: float = 0.0) -> np.ndarray:
    """Build a 4×4 SE(3) pose with rotation about z (yaw) only."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    pose = np.eye(4, dtype=np.float64)
    pose[0, 0] = c
    pose[0, 1] = -s
    pose[1, 0] = s
    pose[1, 1] = c
    pose[0, 3] = tx
    pose[1, 3] = ty
    pose[2, 3] = tz
    return pose


def _write_pcd(path: Path, points: np.ndarray) -> None:
    """Minimal ASCII PCD writer suitable for Open3D's reader."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]
    header = (
        "# .PCD v.7 - Point Cloud Data file format\n"
        "VERSION .7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA ascii\n"
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def _straight_line_dataset() -> SlamRunDatasetCase:
    """6 sensor poses sliding forward along +x by 0.5 m / frame in a static room."""
    n_frames = 6
    scene = _box_scene(seed=11, n_points=1200)
    gt_poses = np.stack(
        [_pose(i * 0.5, 0.0, 0.0, yaw_rad=0.0) for i in range(n_frames)], axis=0
    )

    def build(tmp: Path) -> SlamRunRequest:
        frame_dir = tmp / "straight"
        frame_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, gt in enumerate(gt_poses):
            sensor_to_world = gt
            world_to_sensor = np.linalg.inv(sensor_to_world)
            ones = np.ones((scene.shape[0], 1))
            world_h = np.hstack([scene, ones])
            sensor_pts = (world_to_sensor @ world_h.T).T[:, :3]
            p = frame_dir / f"frame_{i:04d}.pcd"
            _write_pcd(p, sensor_pts)
            paths.append(p)
        return SlamRunRequest(
            frame_paths=tuple(paths),
            timestamps_s=tuple(float(i) * 0.1 for i in range(n_frames)),
            max_range_m=50.0,
            voxel_size_m=0.5,
            deskew=False,
            max_frames=None,
        )

    return SlamRunDatasetCase(
        name="straight_line_6frames",
        description=(
            "6 sensor poses sliding +0.5 m/frame along +x in a static 20×20 m room. "
            "Identity-passthrough collapses every pose to origin; KISS-ICP should "
            "recover the slide within a few cm."
        ),
        build_request=build,
        gt_poses=gt_poses,
        expected_kiss_icp_max_ate_m=0.20,
        keep_files=tuple(f"frame_{i:04d}.pcd" for i in range(n_frames)),
    )


def _short_turn_dataset() -> SlamRunDatasetCase:
    """4 poses tracing a quarter-turn: translate + yaw by 22.5° each step."""
    n_frames = 4
    scene = _box_scene(seed=23, n_points=1500)
    gt_poses = np.stack(
        [
            _pose(0.3 * i, 0.1 * i, 0.0, yaw_rad=0.392 * i)  # ~22.5° per step
            for i in range(n_frames)
        ],
        axis=0,
    )

    def build(tmp: Path) -> SlamRunRequest:
        frame_dir = tmp / "turn"
        frame_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, gt in enumerate(gt_poses):
            world_to_sensor = np.linalg.inv(gt)
            ones = np.ones((scene.shape[0], 1))
            world_h = np.hstack([scene, ones])
            sensor_pts = (world_to_sensor @ world_h.T).T[:, :3]
            p = frame_dir / f"frame_{i:04d}.pcd"
            _write_pcd(p, sensor_pts)
            paths.append(p)
        return SlamRunRequest(
            frame_paths=tuple(paths),
            timestamps_s=tuple(float(i) * 0.1 for i in range(n_frames)),
            max_range_m=50.0,
            voxel_size_m=0.5,
            deskew=False,
            max_frames=None,
        )

    return SlamRunDatasetCase(
        name="short_turn_4frames",
        description=(
            "4 sensor poses tracing a short curve (~22.5° yaw + small translation "
            "per step). Tests that the harness handles non-trivial rotation."
        ),
        build_request=build,
        gt_poses=gt_poses,
        expected_kiss_icp_max_ate_m=0.30,
        keep_files=tuple(f"frame_{i:04d}.pcd" for i in range(n_frames)),
    )


def build_default_datasets() -> list[SlamRunDatasetCase]:
    """Return the deterministic dataset suite the slice evaluator runs."""

    return [_straight_line_dataset(), _short_turn_dataset()]


def absolute_trajectory_error_m(
    estimated_poses: np.ndarray, gt_poses: np.ndarray
) -> float:
    """RMS Euclidean distance between the translation components of two
    pose sequences."""

    if estimated_poses.shape[0] == 0 or gt_poses.shape[0] == 0:
        return float("inf")
    n = min(estimated_poses.shape[0], gt_poses.shape[0])
    est_t = estimated_poses[:n, :3, 3]
    gt_t = gt_poses[:n, :3, 3]
    diffs = np.linalg.norm(est_t - gt_t, axis=1)
    return float(np.sqrt(np.mean(diffs**2)))


def make_temp_dir() -> tempfile.TemporaryDirectory:
    """Helper used by the evaluator / tests to allocate a scratch dir."""

    return tempfile.TemporaryDirectory(prefix="ca_slam_run_")


__all__ = [
    "SlamRunDatasetCase",
    "build_default_datasets",
    "absolute_trajectory_error_m",
    "make_temp_dir",
]
