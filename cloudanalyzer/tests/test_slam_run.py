"""Phase 21: end-to-end smoke for ``ca slam-run`` and the slam_run slice.

The KISS-ICP-based tests skip when ``kiss-icp`` is not installed (it ships
as an optional ``[slam]`` extra, so CI's default ``[dev]`` install doesn't
pull it in). The sentinel driver and the contract helpers are exercised
unconditionally — they only need NumPy / Open3D, both of which are
required deps.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ca.core.slam_run import (
    SlamRunRequest,
    SlamRunResult,
    discover_frame_paths,
    load_frame,
    write_map_ply,
    write_tum_trajectory,
    _rotation_matrices_to_quaternions,
)
from ca.experiments.slam_run.common import (
    _straight_line_dataset,
    _short_turn_dataset,
    absolute_trajectory_error_m,
)
from ca.experiments.slam_run.identity_passthrough import IdentityPassthroughSlamDriver


def test_discover_frame_paths_directory_picks_first_supported_ext(tmp_path: Path) -> None:
    (tmp_path / "a.pcd").write_text("")
    (tmp_path / "b.pcd").write_text("")
    (tmp_path / "c.txt").write_text("")
    out = discover_frame_paths(tmp_path)
    assert [p.name for p in out] == ["a.pcd", "b.pcd"]


def test_discover_frame_paths_list_resolves_relative(tmp_path: Path) -> None:
    (tmp_path / "x.pcd").write_text("")
    (tmp_path / "y.pcd").write_text("")
    listing = tmp_path / "frames.txt"
    listing.write_text("x.pcd\ny.pcd\n# comment\n\n")
    out = discover_frame_paths(listing)
    assert [p.name for p in out] == ["x.pcd", "y.pcd"]
    assert all(p.is_absolute() for p in out)


def test_discover_frame_paths_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No .bin/.pcd/.ply"):
        discover_frame_paths(tmp_path)


def test_load_frame_kitti_bin_drops_intensity(tmp_path: Path) -> None:
    data = np.array(
        [[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.9]], dtype=np.float32
    )
    p = tmp_path / "sweep.bin"
    data.tofile(p)
    pts = load_frame(p)
    assert pts.shape == (2, 3)
    assert pts.dtype == np.float64
    np.testing.assert_allclose(pts, data[:, :3])


def test_load_frame_rejects_unknown_extension(tmp_path: Path) -> None:
    p = tmp_path / "x.unknown"
    p.write_text("x")
    with pytest.raises(ValueError, match="Unsupported frame format"):
        load_frame(p)


def test_write_tum_trajectory_emits_canonical_format(tmp_path: Path) -> None:
    poses = np.zeros((2, 4, 4), dtype=np.float64)
    poses[0] = np.eye(4)
    poses[1] = np.eye(4)
    poses[1, 0, 3] = 1.5
    poses[1, 1, 3] = -0.25
    timestamps = np.array([0.0, 0.1])
    out = tmp_path / "traj.tum"
    write_tum_trajectory(out, poses, timestamps)
    text = out.read_text()
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    assert len(lines) == 2
    # First pose at origin, identity rotation -> qw=1
    parts = lines[0].split()
    assert float(parts[1]) == 0.0
    assert float(parts[2]) == 0.0
    assert float(parts[7]) == 1.0  # qw
    # Second pose tx=1.5, ty=-0.25
    parts = lines[1].split()
    assert float(parts[1]) == pytest.approx(1.5)
    assert float(parts[2]) == pytest.approx(-0.25)
    assert float(parts[7]) == pytest.approx(1.0)


def test_rotation_to_quaternion_round_trip_identity() -> None:
    rotmats = np.broadcast_to(np.eye(3), (5, 3, 3)).copy()
    quats = _rotation_matrices_to_quaternions(rotmats)
    np.testing.assert_allclose(quats[:, 3], 1.0)
    np.testing.assert_allclose(quats[:, :3], 0.0, atol=1e-12)


def test_rotation_to_quaternion_90deg_yaw() -> None:
    theta = np.pi / 2
    R = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0],
         [np.sin(theta),  np.cos(theta), 0.0],
         [0.0,            0.0,           1.0]]
    )
    quats = _rotation_matrices_to_quaternions(R[np.newaxis, :, :])
    # qz = sin(theta/2) = sqrt(2)/2, qw = cos(theta/2) = sqrt(2)/2
    np.testing.assert_allclose(quats[0, 2], np.sqrt(2) / 2, atol=1e-6)
    np.testing.assert_allclose(quats[0, 3], np.sqrt(2) / 2, atol=1e-6)


def test_identity_passthrough_driver_emits_identity_poses(tmp_path: Path) -> None:
    ds = _straight_line_dataset()
    request = ds.build_request(tmp_path)
    drv = IdentityPassthroughSlamDriver()
    result = drv.run(request)
    assert isinstance(result, SlamRunResult)
    assert result.driver == "identity_passthrough"
    assert result.frames_processed == 6
    assert result.poses.shape == (6, 4, 4)
    np.testing.assert_allclose(result.poses, np.broadcast_to(np.eye(4), (6, 4, 4)))
    # The map is just the concatenation of all sensor-frame points.
    assert result.map_points.shape[1] == 3
    assert result.map_points.shape[0] > 0


def test_identity_passthrough_is_a_proper_floor_on_curved_trajectory(tmp_path: Path) -> None:
    """On any case with non-trivial motion, identity-passthrough's ATE
    should be much worse than the expected KISS-ICP threshold. This is what
    makes it a useful sentinel."""

    ds = _short_turn_dataset()
    request = ds.build_request(tmp_path)
    drv = IdentityPassthroughSlamDriver()
    result = drv.run(request)
    ate = absolute_trajectory_error_m(result.poses, ds.gt_poses)
    assert ate > ds.expected_kiss_icp_max_ate_m


# --- KISS-ICP ----------------------------------------------------------------


_kiss_icp = pytest.importorskip(
    "kiss_icp",
    reason="kiss-icp not installed (optional [slam] extra)",
)


def test_kiss_icp_driver_recovers_straight_line_motion(tmp_path: Path) -> None:
    from ca.experiments.slam_run.kiss_icp_driver import KissICPSlamDriver

    ds = _straight_line_dataset()
    request = ds.build_request(tmp_path)
    drv = KissICPSlamDriver()
    result = drv.run(request)
    assert result.driver == "kiss_icp"
    assert result.frames_processed == 6
    ate = absolute_trajectory_error_m(result.poses, ds.gt_poses)
    assert ate <= ds.expected_kiss_icp_max_ate_m


def test_kiss_icp_driver_writes_consumable_artifacts(tmp_path: Path) -> None:
    from ca.experiments.slam_run.kiss_icp_driver import KissICPSlamDriver

    ds = _straight_line_dataset()
    request = ds.build_request(tmp_path)
    drv = KissICPSlamDriver()
    result = drv.run(request)
    traj_path = tmp_path / "trajectory.tum"
    map_path = tmp_path / "map.ply"
    write_tum_trajectory(traj_path, result.poses, result.timestamps_s)
    write_map_ply(map_path, result.map_points)
    assert traj_path.is_file()
    assert map_path.is_file()
    # ca.traj-evaluate's TUM parser expects '<ts> <tx> <ty> <tz> <qx> <qy> <qz> <qw>'
    lines = [
        line for line in traj_path.read_text().splitlines() if not line.startswith("#")
    ]
    assert len(lines) == 6
    assert len(lines[0].split()) == 8


def test_cli_slam_run_end_to_end(tmp_path: Path) -> None:
    """Drive the full CLI on a synthetic straight-line case."""

    ds = _straight_line_dataset()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    request = ds.build_request(tmp_path)
    # Copy synthesised frames into a stable input dir the CLI can read.
    for src in request.frame_paths:
        shutil.copy(src, input_dir / src.name)

    output_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "-m",
        "cloudanalyzer_cli.main",
        "slam-run",
        str(input_dir),
        str(output_dir),
        "--driver",
        "kiss-icp",
        "--max-range",
        "60",
        "--voxel-size",
        "0.5",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["driver"] == "kiss_icp"
    assert summary["frames_processed"] == 6
    assert summary["map_points"] > 0
    assert (output_dir / "trajectory.tum").is_file()
    assert (output_dir / "map.ply").is_file()
