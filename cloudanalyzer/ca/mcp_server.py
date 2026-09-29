"""CloudAnalyzer as MCP tools, for AI agents (``ca mcp``; ``pip install "cloudanalyzer[mcp]"``).

The tools fix and compare SLAM maps with the Rust core (as the web app does,
natively and on all cores) and evaluate clouds and trajectories. Each returns
the JSON report its ``ca`` command prints; paths are local to the machine the
server runs on. Register it with an MCP client, e.g. Claude Code:

    claude mcp add cloudanalyzer -- ca mcp
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

INSTRUCTIONS = """\
CloudAnalyzer fixes and measures LiDAR point clouds and SLAM maps on this machine.

A SLAM session folder holds a poses file (g2o, or a KITTI / TUM trajectory) and one scan
per pose (PCD, PLY, LAS/LAZ, XYZ, KITTI .bin) named by frame number. Look at a folder with
session_layout first (it is quick). Raw scans without poses: run slam_odometry first, then
posegraph_fix with its trajectory. Then run posegraph_fix (loops, IMU gravity, dynamic
points, the fixed map) or posegraph_compare (two drives through the same places: what
changed). Those read every scan and take seconds to minutes on large drives. Outputs go to
out_dir; open the written .ply maps in the CloudAnalyzer web app to look at them.
"""


def session_layout(folder: str) -> dict[str, Any]:
    """Look at a SLAM session folder without loading it: the poses file, how many poses and
    scans, whether scans match poses by frame number, a KITTI calib.txt, and folders or files
    that look like IMU gravity (OXTS or 'frame ux uy uz')."""
    from ca.posegraph_fix import SCAN_SUFFIXES, match_scans, poses_file, read_trajectory

    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(folder)
    poses = poses_file(root)
    scans = sorted(p for p in root.iterdir() if p.suffix.lower() in SCAN_SUFFIXES)
    out: dict[str, Any] = {
        "folder": str(root),
        "poses_file": poses.name if poses else None,
        "scans": len(scans),
        "scan_formats": sorted({p.suffix.lower() for p in scans}),
        "kitti_calib": (root / "calib.txt").exists(),
    }
    if poses is not None and poses.suffix.lower() != ".g2o":
        trajectory, stamps = read_trajectory(poses)
        out["poses"] = int(len(trajectory))
        out["timestamps"] = stamps is not None
        steps = trajectory[1:, :3, 3] - trajectory[:-1, :3, 3]
        out["path_length_m"] = round(float((steps**2).sum(1).__pow__(0.5).sum()), 1)
        try:
            matched = match_scans(scans, list(range(len(trajectory))))
            out["scans_matched"] = sum(m is not None for m in matched)
        except ValueError as e:
            out["scans_matched"] = 0
            out["matching_problem"] = str(e)
    candidates = [root, root.parent]
    out["gravity_candidates"] = sorted(
        {
            str(p)
            for base in candidates
            for p in base.iterdir()
            if (p.is_dir() and (p.name.lower() in {"oxts", "gravity", "imu"}))
            or (p.is_file() and p.name.lower() in {"gravity.txt", "ups.txt"})
        }
    )
    return out


def slam_odometry(
    scans: str,
    out_dir: str,
    max_range: float = 80.0,
    voxel_size: float | None = None,
    max_frames: int | None = None,
    deskew: bool = False,
) -> dict[str, Any]:
    """LiDAR odometry for raw scans (a folder of KITTI .bin, PCD or PLY named in time order)
    with KISS-ICP (pip install "cloudanalyzer[slam]"): writes trajectory.tum (one pose per scan) and map.ply to out_dir.
    Then give posegraph_fix the scans folder and poses=<out_dir>/trajectory.tum (with a
    keyframe_spacing of about 1 m for 10 Hz scans) to close its loops."""
    import time

    from ca.core.slam_run import SlamRunRequest, discover_frame_paths, run_slam, write_map_ply, write_tum_trajectory

    frames = discover_frame_paths(Path(scans))
    request = SlamRunRequest(
        frame_paths=tuple(frames),
        max_range_m=max_range,
        voxel_size_m=voxel_size,
        deskew=deskew,
        max_frames=max_frames,
    )
    clock = time.perf_counter()
    result = run_slam(request)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_tum_trajectory(out / "trajectory.tum", result.poses, result.timestamps_s)
    write_map_ply(out / "map.ply", result.map_points)
    steps = result.poses[1:, :3, 3] - result.poses[:-1, :3, 3]
    return {
        "driver": result.driver,
        "frames": int(result.frames_processed),
        "path_length_m": round(float(np.sqrt((steps**2).sum(1)).sum()), 1),
        "runtime_s": round(time.perf_counter() - clock, 1),
        "trajectory": str(out / "trajectory.tum"),
        "map": str(out / "map.ply"),
    }


def posegraph_fix(
    folder: str,
    out_dir: str | None = None,
    poses: str | None = None,
    keyframe_spacing: float = 0.0,
    gravity: str | None = None,
    remove_dynamic: bool = False,
    truth: str | None = None,
    voxel: float = 0.4,
    map_voxel: float = 0.2,
    find_loops: bool = True,
) -> dict[str, Any]:
    """Fix a SLAM map: find loops with ICP, tie keyframes to IMU gravity (a KITTI OXTS folder
    or a 'frame ux uy uz' file), optimise, optionally leave dynamic points (traffic) out, and
    write the fixed g2o, KITTI/TUM poses and the map (PLY) to out_dir. poses names the poses
    file when it is not in the folder (trajectory.tum from slam_odometry); keyframe_spacing
    keeps one pose every so many metres of it. With truth (ground-truth poses, one per frame)
    the report has the ATE before and after."""
    from ca.posegraph_fix import fix_session

    return fix_session(
        folder,
        out_dir,
        poses=poses,
        keyframe_spacing=keyframe_spacing,
        voxel=voxel,
        loops=find_loops,
        gravity=gravity,
        remove_dynamic=remove_dynamic,
        map_voxel=map_voxel,
        truth=truth,
    )


def posegraph_compare(
    first: str,
    second: str,
    here: int,
    there: int,
    out_dir: str | None = None,
    gravity_first: str | None = None,
    gravity_second: str | None = None,
    reach: float = 50.0,
    min_change: float = 0.3,
) -> dict[str, Any]:
    """Join two drives through the same places and list what changed between them. here and
    there are one node of each (vertex id or frame number) standing at the same spot; the
    report gives the join, the loops, M3C2 statistics and the largest changed objects
    (centroid, size, mean change); out_dir receives both maps and the M3C2 result."""
    from ca.posegraph_fix import compare_sessions

    return compare_sessions(
        first,
        second,
        out_dir,
        here=here,
        there=there,
        gravity_first=gravity_first,
        gravity_second=gravity_second,
        reach=reach,
        min_change=min_change,
    )


def cloud_info(path: str) -> dict[str, Any]:
    """A point cloud's size, bounds, centroid and density."""
    from ca.info import get_info

    return get_info(path)


def evaluate_map(candidate: str, reference: str, thresholds: list[float] | None = None) -> dict[str, Any]:
    """A map against a reference map: Chamfer and Hausdorff distances, F1 at thresholds, AUC."""
    from ca.evaluate import evaluate

    return evaluate(candidate, reference, thresholds)


def evaluate_trajectory(estimate: str, reference: str, align_rigid: bool = True) -> dict[str, Any]:
    """A trajectory (TUM or CSV, timestamped) against a reference: ATE, RPE, drift, coverage."""
    from ca.trajectory import evaluate_trajectory as evaluate

    return evaluate(estimate, reference, align_rigid=align_rigid)


TOOLS = [session_layout, slam_odometry, posegraph_fix, posegraph_compare, cloud_info, evaluate_map, evaluate_trajectory]


def build_server():
    """The MCP server with CloudAnalyzer's tools (MCP Python SDK 1.x or 2.x)."""
    try:
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as Server
        except ImportError as e:
            raise RuntimeError('the MCP server needs the MCP SDK: pip install "cloudanalyzer[mcp]"') from e
    server = Server(name="cloudanalyzer", instructions=INSTRUCTIONS)
    for tool in TOOLS:
        server.tool()(tool)
    return server


def main() -> None:
    build_server().run()
