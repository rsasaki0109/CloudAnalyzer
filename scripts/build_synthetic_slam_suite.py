#!/usr/bin/env python3
"""Generate the synthetic-figure8 benchmark suite under benchmarks/slam/.

This script is deterministic — running it on a clean checkout reproduces
exactly the files committed under ``benchmarks/slam/synthetic-figure8/``.

Layout produced::

    benchmarks/slam/synthetic-figure8/
    ├── suite.yaml                            # manifest (loaded by `ca benchmark`)
    ├── reference/
    │   ├── map.pcd
    │   └── trajectory.tum
    ├── sample_outputs/
    │   ├── map_pass.pcd                      # reference + small noise; passes the gate
    │   └── trajectory_pass.tum
    └── scans/                                # raw per-frame sensor-frame scans
        └── frame_NNNNN.pcd                   # consumable by `ca slam-run`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "slam" / "synthetic-figure8"


def _planar_map(seed: int = 42) -> np.ndarray:
    """Return a closed planar room: floor plus four boxed walls.

    The four walls (at x=±8 and y=±8) anchor ICP in every horizontal
    direction — earlier revisions only had north/south walls, which left
    east/west translation under-constrained and made it impossible for
    ``ca slam-run`` (KISS-ICP) to recover a figure-8 trajectory.
    """

    rng = np.random.default_rng(seed)
    xs = np.linspace(-10.0, 10.0, 20)
    ys = np.linspace(-10.0, 10.0, 20)
    xx, yy = np.meshgrid(xs, ys)
    floor = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    floor += rng.normal(0, 0.005, size=floor.shape)

    wall_sweep = np.linspace(-8.0, 8.0, 80)
    wall_z = np.linspace(0.0, 1.5, 6)
    sweep_grid, z_grid = np.meshgrid(wall_sweep, wall_z)

    # North / south walls (constant y, sweep x).
    wall_north = np.column_stack(
        [sweep_grid.ravel(), np.full(sweep_grid.size, 8.0), z_grid.ravel()]
    )
    wall_south = np.column_stack(
        [sweep_grid.ravel(), np.full(sweep_grid.size, -8.0), z_grid.ravel()]
    )
    # East / west walls (constant x, sweep y).
    wall_east = np.column_stack(
        [np.full(sweep_grid.size, 8.0), sweep_grid.ravel(), z_grid.ravel()]
    )
    wall_west = np.column_stack(
        [np.full(sweep_grid.size, -8.0), sweep_grid.ravel(), z_grid.ravel()]
    )
    for w in (wall_north, wall_south, wall_east, wall_west):
        w += rng.normal(0, 0.005, size=w.shape)

    return np.vstack([floor, wall_north, wall_south, wall_east, wall_west]).astype(
        np.float64
    )


def _figure8_trajectory(
    n: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ``(timestamps, positions, yaws, raw_yaw0)`` for a planar figure-8.

    The sensor's heading is tangent to the trajectory at each frame
    (vehicle-style), so yaw varies over the loop. The trajectory and
    yaw series are both rotated so that the first pose is the identity
    (position at origin, yaw = 0). That makes the reference comparable
    to what a SLAM driver naturally outputs — every driver places its
    first frame at identity and accumulates from there.

    The fourth return value is the raw (pre-rotation) tangent yaw at
    ``t=0``. Callers pass it to ``_rotate_map_into_slam_frame`` to bring
    the bundled map into the same coordinate system.
    """

    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = 5.0 * np.sin(t)
    y = 5.0 * np.sin(t) * np.cos(t)
    dx = 5.0 * np.cos(t)
    dy = 5.0 * np.cos(2.0 * t)
    raw_yaws = np.arctan2(dy, dx)
    yaw0 = float(raw_yaws[0])
    c, s = float(np.cos(yaw0)), float(np.sin(yaw0))
    x_rot = c * x + s * y
    y_rot = -s * x + c * y
    z = np.zeros_like(t)
    positions = np.column_stack([x_rot, y_rot, z])
    yaws = raw_yaws - yaw0
    yaws = np.arctan2(np.sin(yaws), np.cos(yaws))
    timestamps = np.arange(n) * 0.1
    return timestamps, positions, yaws, yaw0


def _rotate_map_into_slam_frame(map_world: np.ndarray, yaw0: float) -> np.ndarray:
    """Rotate the world map so it lives in the same frame as ``_figure8_trajectory``.

    ``_figure8_trajectory`` rotates the raw figure-8 by ``-yaw0`` so the
    first sensor pose is identity. The bundled map needs the same
    rotation so scans built from ``(map - position) @ R(yaw)`` agree with
    the reference trajectory.
    """

    c, s = float(np.cos(yaw0)), float(np.sin(yaw0))
    R = np.array(
        [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return map_world @ R.T


def _yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """Return ``(qx, qy, qz, qw)`` for a yaw-only rotation."""

    return 0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0))


def _rotation_z(yaw: float) -> np.ndarray:
    """Return the 3×3 rotation matrix for a yaw-only rotation."""

    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _write_pcd(points: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def _write_tum(
    timestamps: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for ts, (x, y, z), yaw in zip(timestamps, positions, yaws):
        qx, qy, qz, qw = _yaw_to_quaternion(float(yaw))
        lines.append(
            f"{ts:.6f} {x:.6f} {y:.6f} {z:.6f} "
            f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_scans(
    map_world: np.ndarray,
    positions: np.ndarray,
    yaws: np.ndarray,
    out_dir: Path,
    *,
    stride: int = 1,
    max_range: float = 25.0,
    noise_sigma: float = 0.01,
    seed: int = 19,
) -> int:
    """Write per-frame raw sensor-frame scans to ``out_dir``.

    For each frame ``i`` the scan is
    ``R(yaw_i)^T @ (map_world - position_i)`` filtered by ``max_range``
    and perturbed by per-scan Gaussian noise. With the yaw rotation
    applied, the drivers actually have to recover heading and not just
    translation.

    Default ``stride=1`` writes one scan per reference pose (~0.16 m
    per-frame motion). Coarser strides (e.g. stride 4 = ~0.62 m/frame)
    starve KISS-ICP's constant-velocity bootstrap and the trajectory
    comes back the wrong direction. Returns the number of scans written.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    indices = list(range(0, positions.shape[0], stride))
    for j, i in enumerate(indices):
        R_t = _rotation_z(float(yaws[i])).T
        sensor_pts = (map_world - positions[i][np.newaxis, :]) @ R_t.T
        dists = np.linalg.norm(sensor_pts, axis=1)
        sensor_pts = sensor_pts[dists <= max_range]
        if noise_sigma > 0 and sensor_pts.size > 0:
            sensor_pts = sensor_pts + rng.normal(0, noise_sigma, size=sensor_pts.shape)
        _write_pcd(sensor_pts.astype(np.float64), out_dir / f"frame_{j:05d}.pcd")
    return len(indices)


SUITE_YAML = """\
version: 1
name: synthetic-figure8
description: Tiny synthetic figure-8 trajectory with a closed planar room map and tangent-aligned sensor heading. Use it to smoke-test `ca benchmark` without external data.
license: MIT (synthetic data generated by scripts/build_synthetic_slam_suite.py)
sequences:
  default:
    description: 200-pose figure-8 over a ~2.3k point closed planar room with four raised walls. Sensor heading is tangent to the trajectory (vehicle-style), so the suite exercises rotation estimation alongside translation.
    reference_map: reference/map.pcd
    reference_trajectory: reference/trajectory.tum
sample_outputs:
  default:
    map: sample_outputs/map_pass.pcd
    trajectory: sample_outputs/trajectory_pass.tum
gate:
  min_auc: 0.95
  max_chamfer: 0.05
  max_ate: 0.30
  max_rpe: 0.20
  max_drift: 0.50
  min_coverage: 0.90
"""


def build(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # _figure8_trajectory rotates the figure-8 so the initial sensor pose
    # is identity. Apply the matching rotation to the planar map so the
    # SLAM driver's first frame and the bundled reference map share a
    # coordinate system.
    timestamps, traj_ref, yaws_ref, raw_yaw0 = _figure8_trajectory()
    map_ref = _rotate_map_into_slam_frame(_planar_map(), raw_yaw0)

    _write_pcd(map_ref, output_dir / "reference" / "map.pcd")
    _write_tum(timestamps, traj_ref, yaws_ref, output_dir / "reference" / "trajectory.tum")

    # Sample passing output: reference + small noise. Determinism via fixed seed.
    rng = np.random.default_rng(7)
    map_pass = map_ref + rng.normal(0, 0.02, size=map_ref.shape)
    traj_pass = traj_ref + rng.normal(0, 0.05, size=traj_ref.shape)
    yaws_pass = yaws_ref + rng.normal(0, 0.02, size=yaws_ref.shape)

    _write_pcd(map_pass, output_dir / "sample_outputs" / "map_pass.pcd")
    _write_tum(
        timestamps, traj_pass, yaws_pass,
        output_dir / "sample_outputs" / "trajectory_pass.tum",
    )

    _write_scans(map_ref, traj_ref, yaws_ref, output_dir / "scans")

    (output_dir / "suite.yaml").write_text(SUITE_YAML, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the synthetic-figure8 suite",
    )
    args = parser.parse_args()
    build(args.output)
    print(f"Wrote synthetic-figure8 suite to {args.output}")


if __name__ == "__main__":
    main()
