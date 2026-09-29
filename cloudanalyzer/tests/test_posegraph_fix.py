"""`ca posegraph-fix`: session folders read as the web app reads them, and a drifted lap closed."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from ca.posegraph_fix import frame_number, keyframes, match_scans, poses_file, read_trajectory, read_ups


def test_frame_numbers_and_matching():
    assert frame_number("000042.pcd") == 42
    assert frame_number("scan_7_0012.bin") == 12
    assert frame_number("map.pcd") is None
    scans = [Path(f"{k:06d}.bin") for k in (2, 0, 1)]
    assert match_scans(scans, [0, 1, 2]) == [2, 0, 1]
    # Unnumbered, one per pose: in name order.
    named = [Path(n) for n in ("b.pcd", "a.pcd", "c.pcd")]
    assert match_scans(named, [10, 11, 12]) == [1, 0, 2]
    with pytest.raises(ValueError, match="cannot match"):
        match_scans(named[:2], [10, 11, 12])


def test_trajectories_and_the_poses_file(tmp_path):
    (tmp_path / "calib.txt").write_text("Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    (tmp_path / "times.txt").write_text("0.0\n0.1\n")
    kitti = tmp_path / "poses.txt"
    kitti.write_text("1 0 0 1 0 1 0 2 0 0 1 3\n0 -1 0 4 1 0 0 5 0 0 1 6\n")
    assert poses_file(tmp_path) == kitti
    poses, stamps = read_trajectory(kitti)
    assert stamps is None and poses.shape == (2, 4, 4)
    assert np.allclose(poses[1, :3, 3], [4, 5, 6]) and np.allclose(poses[1, :2, :2], [[0, -1], [1, 0]])
    tum = tmp_path / "odom.tum"
    tum.write_text(f"10.0 1 2 3 0 0 {math.sin(0.25)} {math.cos(0.25)}\n")
    poses, stamps = read_trajectory(tum)
    assert stamps.tolist() == [10.0]
    assert np.allclose(poses[0, :2, :2], [[math.cos(0.5), -math.sin(0.5)], [math.sin(0.5), math.cos(0.5)]])
    # A g2o graph wins over trajectories.
    (tmp_path / "graph.g2o").write_text("")
    assert poses_file(tmp_path).name == "graph.g2o"


def test_up_directions_from_oxts_and_from_text(tmp_path):
    oxts = tmp_path / "oxts"
    (oxts / "data").mkdir(parents=True)
    fields = ["0"] * 30
    fields[3], fields[4] = "0.1", "0.2"  # roll, pitch
    (oxts / "data" / "0000000003.txt").write_text(" ".join(fields))
    (oxts / "calib_imu_to_velo.txt").write_text("R: 0 -1 0 1 0 0 0 0 1\nT: 0 0 0\n")
    up = read_ups(oxts)[3]
    imu = np.array([-math.sin(0.2), math.cos(0.2) * math.sin(0.1), math.cos(0.2) * math.cos(0.1)])
    assert np.allclose(up, [-imu[1], imu[0], imu[2]])
    plain = tmp_path / "gravity.txt"
    plain.write_text("5 0 0 1\n6 0.1 0 0.99\n")
    assert sorted(read_ups(plain)) == [5, 6]


# --- With the Rust core: a drifted lap round a courtyard, closed ---------------------


def _yawed(x: float, y: float, yaw: float) -> np.ndarray:
    m = np.eye(4)
    m[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    m[:2, 3] = (x, y)
    return m


def _courtyard(ground: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """Walls round a 40 m square with a few pillars, 3 m high, and a patch of ``ground`` (x0, x1, y0, y1)
    so that a change standing on it has something to be measured against."""
    points = []
    for k in range(81):
        t = k * 0.5 - 20.0
        for z in np.arange(0, 3, 0.5):
            points += [(t, -20, z), (t, 20, z), (-20, t, z), (20, t, z)]
    if ground:
        for x in np.arange(ground[0], ground[1], 0.5):
            for y in np.arange(ground[2], ground[3], 0.5):
                points.append((x, y, 0.0))
    for px, py in [(-8, -9), (7, -6), (9, 8), (-6, 7), (0, 12)]:
        for a in np.linspace(0, 2 * math.pi, 12, endpoint=False):
            for z in np.arange(0, 3, 0.5):
                points.append((px + 0.6 * math.cos(a), py + 0.6 * math.sin(a), z))
    return np.array(points, dtype=float)


def _session(folder: Path, extra: np.ndarray | None = None, drift: float = 0.005, ground=None) -> Path:
    """A lap round a 24 m square in 3 m steps, odometry turning ``drift`` radians too far each step."""
    truth = []
    for side in range(4):
        for step in range(8):
            d = step * 3.0 - 12.0
            x, y = [(d, -12.0), (12.0, d), (-d, 12.0), (-12.0, -d)][side]
            truth.append(_yawed(x, y, side * math.pi / 2))
    truth.append(truth[0])
    drifted = [truth[0]]
    for k in range(1, len(truth)):
        step = _yawed(0, 0, drift) @ np.linalg.inv(truth[k - 1]) @ truth[k]
        drifted.append(drifted[-1] @ step)
    points = _courtyard(ground) if extra is None else np.vstack([_courtyard(ground), extra])
    world = np.c_[points, np.ones(len(points))]
    folder.mkdir()
    for k, pose in enumerate(truth):
        local = (np.linalg.inv(pose) @ world.T).T[:, :3]
        np.savetxt(folder / f"{k:06d}.xyz", local, fmt="%.4f")
    rows = lambda poses: "".join(" ".join(f"{v:.9f}" for v in p[:3].ravel()) + "\n" for p in poses)  # noqa: E731
    (folder / "poses.txt").write_text(rows(drifted))
    truth_file = folder.parent / "truth.txt"
    truth_file.write_text(rows(truth))
    return truth_file


def test_a_drifted_lap_is_closed_and_written(tmp_path):
    pytest.importorskip("cloudanalyzer_core")
    from ca.posegraph_fix import fix_session

    truth = _session(tmp_path / "lap")
    report = fix_session(str(tmp_path / "lap"), str(tmp_path / "out"), voxel=0.3, truth=str(truth))
    assert report["nodes"] == 33 and report["scans"] == 33
    assert report["loops"]["added"] >= 1
    assert any(a <= 1 and b >= 31 for a, b, _ in report["loops"]["pairs"])
    before, after = report["ate"]["before"], report["ate"]["after"]
    assert before["end_error"] > 1.0 and after["end_error"] < 0.1
    for path in report["outputs"].values():
        assert Path(path).stat().st_size > 0
    assert Path(report["outputs"]["map"]).read_bytes().startswith(b"ply\nformat binary_little_endian")


def test_the_command_prints_a_json_report(tmp_path):
    pytest.importorskip("cloudanalyzer_core")
    from cloudanalyzer_cli.main import app

    _session(tmp_path / "lap")
    result = CliRunner().invoke(app, ["posegraph-fix", str(tmp_path / "lap"), "--format-json", "--voxel", "0.3"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["loops"]["added"] >= 1 and "outputs" not in report


def _box(cx: float, cy: float, size: float = 2.0, step: float = 0.2) -> np.ndarray:
    """The surface of a cube standing on the ground at (cx, cy)."""
    t = np.arange(0, size + 1e-9, step)
    a, b = np.meshgrid(t, t)
    a, b = a.ravel(), b.ravel()
    x0, y0 = cx - size / 2, cy - size / 2
    faces = [
        (x0 + a, y0 + 0 * a, b), (x0 + a, y0 + size + 0 * a, b),
        (x0 + 0 * a, y0 + a, b), (x0 + size + 0 * a, y0 + a, b),
        (x0 + a, y0 + b, size + 0 * a),
    ]
    return np.vstack([np.c_[x, y, z] for x, y, z in faces])


def test_two_drives_are_joined_and_a_new_box_is_the_change(tmp_path):
    pytest.importorskip("cloudanalyzer_core")
    from ca.posegraph_fix import compare_sessions

    patch = (-2.0, 10.0, -19.5, -13.0)
    _session(tmp_path / "before", ground=patch)
    # The same lap later, drifting the other way, with a container standing by the south wall.
    _session(tmp_path / "after", extra=_box(4.0, -16.0), drift=-0.004, ground=patch)
    report = compare_sessions(
        str(tmp_path / "before"), str(tmp_path / "after"), str(tmp_path / "out"), here=0, there=0, voxel=0.2,
    )
    assert report["join"]["overlap"] > 0.8
    assert report["loops"]["added"] >= 1
    assert report["m3c2"]["significant"] > 0
    largest = report["changes"]["largest"][0]
    assert math.dist(largest["centroid"][:2], (4.0, -16.0)) < 2.0, largest
    for path in report["outputs"].values():
        assert Path(path).stat().st_size > 0


def test_keyframes_every_so_many_metres():
    poses = np.tile(np.eye(4), (11, 1, 1))
    poses[:, 0, 3] = np.arange(11) * 0.4  # 0.4 m a frame
    assert keyframes(poses, 0.0) == list(range(11))
    assert keyframes(poses, 1.0) == [0, 3, 6, 9, 10]


def test_poses_from_elsewhere_thinned_to_keyframes_still_close_the_lap(tmp_path):
    pytest.importorskip("cloudanalyzer_core")
    from ca.posegraph_fix import fix_session

    truth = _session(tmp_path / "lap")
    # The odometry lives outside the scans' folder, as ca slam-run writes it.
    (tmp_path / "lap" / "poses.txt").rename(tmp_path / "odometry.txt")
    report = fix_session(
        str(tmp_path / "lap"), poses=str(tmp_path / "odometry.txt"), keyframe_spacing=5.0, voxel=0.3, truth=str(truth)
    )
    assert report["poses_file"] == "odometry.txt"
    assert report["nodes"] < 33 and report["scans"] == report["nodes"]
    assert report["loops"]["added"] >= 1
    # Keyframes 6 m apart register less tightly than every 3 m, still well within the drift.
    assert report["ate"]["before"]["end_error"] > 1.0 and report["ate"]["after"]["end_error"] < 0.3

