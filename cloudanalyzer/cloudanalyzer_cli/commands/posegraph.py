"""Pose-graph repair, loop-closure and map inspection commands."""

import json
from pathlib import Path
from typing import List, Optional
import typer
from ca.view import view
from ca.mme import compute_mme
from ca.stats import compute_stats
from ca.posegraph import discover_session_paths, validate_posegraph_session
from ca.loop_closure_report import LoopClosureGate, build_loop_closure_report
from cloudanalyzer_cli._app import app, _dump_json, _handle_error, _parse_thresholds


@app.command("posegraph-fix")
def posegraph_fix_cmd(
    folder: str = typer.Argument(
        ...,
        help="Session folder: a poses file (g2o, KITTI or TUM) and one scan per pose, named by frame number; "
        "or raw scans (a folder without poses, or a ROS bag), for which LiDAR odometry is made first",
    ),
    out: Optional[str] = typer.Option(None, "--out", help="Write the fixed graph, poses and map here"),
    poses: Optional[str] = typer.Option(
        None, "--poses", help="The poses file, when not in the folder (e.g. trajectory.tum from ca slam-run)"
    ),
    keyframe_spacing: float = typer.Option(
        0.0, "--keyframe-spacing", help="Keep a keyframe every this many metres of a trajectory (0: every pose)"
    ),
    gravity: Optional[str] = typer.Option(
        None, "--gravity", help="IMU up directions: a KITTI OXTS folder or a file of 'frame ux uy uz' lines"
    ),
    remove_dynamic: bool = typer.Option(
        False, "--remove-dynamic", help="Leave points other scans saw through (passing cars) out of the map"
    ),
    no_imu_calibration: bool = typer.Option(
        False,
        "--no-imu-calibration",
        help="Take the up directions as given: do not estimate the IMU's rotation into the scans' frame",
    ),
    no_loops: bool = typer.Option(False, "--no-loops", help="Do not look for loops"),
    voxel: float = typer.Option(0.4, "--voxel", help="Thin each scan to one point per voxel (m)"),
    map_voxel: float = typer.Option(0.2, "--map-voxel", help="Thin the written map to one point per voxel (m)"),
    radius: float = typer.Option(10.0, "--loop-radius", help="Loop candidates at most this far apart (m), plus drift"),
    drift: float = typer.Option(3.0, "--drift", help="Odometry drift allowed, in % of the path between two nodes"),
    min_overlap: float = typer.Option(50.0, "--min-overlap", help="Keep a loop when this % of the scan overlaps"),
    truth: Optional[str] = typer.Option(
        None, "--truth", help="Ground-truth poses (KITTI or TUM, same frames) to report the ATE before and after"
    ),
    pointcloud_topic: Optional[str] = typer.Option(None, "--pointcloud-topic", help="A bag's PointCloud2 topic"),
    imu_topic: Optional[str] = typer.Option(None, "--imu-topic", help="A bag's Imu topic (for gravity)"),
    imu_to_lidar: Optional[str] = typer.Option(
        None, "--imu-to-lidar", help="The IMU-to-LiDAR rotation, 9 numbers row-major (identity when omitted)"
    ),
    max_range: float = typer.Option(80.0, "--max-range", help="Odometry: drop points farther than this (m)"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Write the report as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print the report as JSON"),
) -> None:
    """Fix a SLAM map: find loops, tie to IMU gravity, optimise, remove dynamic points, write the map."""
    from ca.posegraph_fix import fix_session, needs_odometry, odometry

    say = None if format_json else (lambda m: typer.echo(f"... {m}", err=True))
    odometry_report = None
    try:
        if needs_odometry(folder, poses):
            if not out:
                raise ValueError("raw scans need --out, where the odometry and the scans it reads are written")
            if say:
                say("no poses: making the odometry")
            rotation = [float(v) for v in imu_to_lidar.replace(",", " ").split()] if imu_to_lidar else None
            odometry_report = odometry(
                folder,
                str(Path(out) / "odometry"),
                max_range=max_range,
                pointcloud_topic=pointcloud_topic,
                imu_topic=imu_topic,
                imu_to_lidar=rotation,
            )
            folder, poses = odometry_report["scans"], odometry_report["trajectory"]
            gravity = gravity or odometry_report["gravity"]
        report = fix_session(
            folder,
            out,
            poses=poses,
            keyframe_spacing=keyframe_spacing,
            voxel=voxel,
            loops=not no_loops,
            loop_options={"max_distance": radius, "drift": drift / 100, "min_fitness": min_overlap / 100},
            gravity=gravity,
            calibrate_gravity=not no_imu_calibration,
            remove_dynamic=remove_dynamic,
            map_voxel=map_voxel,
            truth=truth,
            progress=say,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        _handle_error(e)
    if odometry_report:
        report["odometry"] = odometry_report
    if output_json:
        _dump_json(report, output_json)
    if format_json:
        typer.echo(json.dumps(report, indent=2))
        return
    if odometry_report:
        o = odometry_report
        reading = f" (reading the bag {o['reading_s']} s of it)" if o["reading_s"] >= 1 else ""
        typer.echo(f"odometry: {o['frames']} scans, {o['path_length_m']} m in {o['runtime_s']} s{reading}")
    typer.echo(f"{report['poses_file']}: {report['nodes']} poses, {report['scan_points']:,} scan points")
    if "loops" in report:
        loops = report["loops"]
        typer.echo(f"loops: {loops['added']} of {loops['candidates']} candidates added ({loops['implausible']} implausible)")
    if "gravity" in report:
        g = report["gravity"]
        imu = g["imus"][0]
        spread = f"{imu['spread_deg']:.2f} deg"
        if "spread_deg_calibrated" in imu:
            spread += f" -> {imu['spread_deg_calibrated']:.2f} deg with the IMU rotation estimated"
        typer.echo(f"gravity: {g['tied']} keyframes tied (up directions spread {spread}; sigma {g['sigma_deg']:.2f} deg)")
    if "dynamic" in report:
        d = report["dynamic"]
        typer.echo(f"dynamic: {d['points']:,} of {d['of']:,} points ({100 * d['share']:.2f} %)")
    if "ate" in report:
        b, a = report["ate"]["before"], report["ate"]["after"]
        typer.echo(f"ATE: {b['ate_rmse']:.3f} -> {a['ate_rmse']:.3f} m (aligned {b['ate_rmse_aligned']:.3f} -> {a['ate_rmse_aligned']:.3f} m)")
    for kind, path in report.get("outputs", {}).items():
        typer.echo(f"wrote {kind}: {path}")


@app.command("posegraph-compare")
def posegraph_compare_cmd(
    first: str = typer.Argument(..., help="The first drive's session folder (poses and scans)"),
    second: str = typer.Argument(..., help="The second drive's session folder, through some of the same places"),
    here: int = typer.Option(..., "--here", help="A node of the first drive (vertex id or frame number)..."),
    there: int = typer.Option(..., "--there", help="...and the second drive's node at the same place"),
    out: Optional[str] = typer.Option(None, "--out", help="Write the joined graph, both maps and the M3C2 result here"),
    gravity_first: Optional[str] = typer.Option(None, "--gravity-first", help="The first drive's IMU up directions"),
    gravity_second: Optional[str] = typer.Option(None, "--gravity-second", help="The second drive's IMU up directions"),
    no_loops: bool = typer.Option(False, "--no-loops", help="Only the join, no loop search"),
    voxel: float = typer.Option(0.4, "--voxel", help="Thin each scan to one point per voxel (m)"),
    map_voxel: float = typer.Option(0.3, "--map-voxel", help="Thin the compared maps (m)"),
    reach: float = typer.Option(50.0, "--reach", help="Compare keyframes within this distance of the other drive (m)"),
    min_change: float = typer.Option(0.3, "--min-change", help="Smallest change counted in a changed object (m)"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Write the report as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print the report as JSON"),
) -> None:
    """Join two drives through the same place and list what changed between them (M3C2, changed objects)."""
    from ca.posegraph_fix import compare_sessions

    try:
        report = compare_sessions(
            first,
            second,
            out,
            here=here,
            there=there,
            voxel=voxel,
            loops=not no_loops,
            gravity_first=gravity_first,
            gravity_second=gravity_second,
            map_voxel=map_voxel,
            reach=reach,
            min_change=min_change,
            progress=None if format_json else (lambda m: typer.echo(f"... {m}", err=True)),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        _handle_error(e)
    if output_json:
        _dump_json(report, output_json)
    if format_json:
        typer.echo(json.dumps(report, indent=2))
        return
    j, m, c = report["join"], report["m3c2"], report["changes"]
    typer.echo(f"joined at {j['here']} = {j['there']}: overlap {100 * j['overlap']:.0f} %")
    if "loops" in report:
        typer.echo(f"loops across the drives: {report['loops']['added']} of {report['loops']['candidates']} candidates")
    typer.echo(
        f"M3C2 at {m['core_points']:,} core points: {m['measured']:,} measured, "
        f"{m['significant']:,} significant ({100 * m['significant_share']:.1f} %)"
    )
    typer.echo(f"{c['objects']:,} changed objects; the largest:")
    for o in c["largest"][:10]:
        size = " x ".join(f"{v:.1f}" for v in o["size"])
        typer.echo(f"  #{o['rank']}  {size} m  {o['mean_change']:+.2f} m  {o['points']:,} points  at {o['centroid']}")
    for kind, path in report.get("outputs", {}).items():
        typer.echo(f"wrote {kind}: {path}")


@app.command("mcp")
def mcp_cmd() -> None:
    """Serve CloudAnalyzer's tools to AI agents over MCP (stdio): fix and compare SLAM maps, evaluate clouds."""
    from ca.mcp_server import main as serve

    try:
        serve()
    except RuntimeError as e:
        _handle_error(e)


@app.command("posegraph-validate")
def posegraph_validate_cmd(
    g2o_path: str = typer.Argument(..., help="Path to pose graph file (pose_graph.g2o)"),
    tum_path: Optional[str] = typer.Option(
        None,
        "--tum",
        help="Optional path to optimized poses in TUM format (optimized_poses_tum.txt)",
    ),
    key_point_frame_dir: Optional[str] = typer.Option(
        None,
        "--key-point-frame",
        help="Optional directory containing keyframe point clouds (key_point_frame/*.pcd)",
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump full result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Validate a manual-loop-closure style mapping session layout."""
    try:
        result = validate_posegraph_session(
            g2o_path=g2o_path,
            tum_path=tum_path,
            key_point_frame_dir=key_point_frame_dir,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    if format_json:
        typer.echo(json.dumps(result, indent=2, default=str))
    else:
        g = result["g2o"]
        typer.echo(
            f"g2o: vertices={g['vertex_count']} edges={g['edge_count']} "
            f"components={g.get('connected_components', '?')} "
            f"malformed={g['malformed_lines']}"
        )
        if g["missing_vertex_references"] > 0:
            typer.echo(f"g2o: missing vertex refs: {g['missing_vertex_references']}")
        if g.get("self_loops", 0) or g.get("duplicate_undirected_edges", 0):
            typer.echo(
                f"g2o: self_loops={g.get('self_loops', 0)} "
                f"dup_edges={g.get('duplicate_undirected_edges', 0)}"
            )
        if "tum" in result:
            t = result["tum"]
            typer.echo(f"tum: poses={t['num_poses']} duration={t['duration_s']:.3f}s")
        if "key_point_frame" in result:
            k = result["key_point_frame"]
            typer.echo(f"key_point_frame: pcds={k['pcd_count']}")
        status = "PASS" if result["summary"]["ok"] else "FAIL"
        typer.echo(f"Session: {status}")
        if result["summary"].get("errors"):
            typer.echo("Errors:   " + "; ".join(result["summary"]["errors"]))
        if result["summary"].get("warnings"):
            typer.echo("Warnings: " + "; ".join(result["summary"]["warnings"]))

    if output_json:
        _dump_json(result, output_json)


@app.command("loop-closure-report")
def loop_closure_report_cmd(
    before_map: str = typer.Argument(..., help="Map before manual loop closure (pcd/ply/las/laz)"),
    after_map: str = typer.Argument(..., help="Map after manual loop closure (pcd/ply/las/laz)"),
    reference_map: str = typer.Argument(..., help="Reference/GT map (pcd/ply/las/laz)"),
    thresholds: Optional[str] = typer.Option(
        None,
        "--thresholds",
        help="Comma-separated distance thresholds in meters for AUC/F1 curve.",
    ),
    min_auc_gain: Optional[float] = typer.Option(
        None,
        "--min-auc-gain",
        help="Fail if AUC(after)-AUC(before) is below this value.",
    ),
    max_after_chamfer: Optional[float] = typer.Option(
        None,
        "--max-after-chamfer",
        help="Fail if chamfer(after) exceeds this value.",
    ),
    before_traj: Optional[str] = typer.Option(
        None,
        "--before-traj",
        help="Optional trajectory before loop closure (CSV/TUM).",
    ),
    after_traj: Optional[str] = typer.Option(
        None,
        "--after-traj",
        help="Optional trajectory after loop closure (CSV/TUM).",
    ),
    reference_traj: Optional[str] = typer.Option(
        None,
        "--ref-traj",
        help="Optional reference trajectory (CSV/TUM).",
    ),
    traj_max_time_delta: float = typer.Option(
        0.05,
        "--traj-max-time-delta",
        help="Max time delta for trajectory matching/interpolation.",
    ),
    traj_align_origin: bool = typer.Option(
        False,
        "--traj-align-origin",
        help="Align trajectory by matching origins before scoring.",
    ),
    traj_align_rigid: bool = typer.Option(
        False,
        "--traj-align-rigid",
        help="Rigidly align trajectory to reference before scoring.",
    ),
    min_ate_gain: Optional[float] = typer.Option(
        None,
        "--min-ate-gain",
        help="Fail if trajectory ATE RMSE improvement (before-after) is below this value.",
    ),
    max_after_ate: Optional[float] = typer.Option(
        None,
        "--max-after-ate",
        help="Fail if trajectory ATE RMSE(after) exceeds this value.",
    ),
    require_posegraph_ok: bool = typer.Option(
        False,
        "--require-posegraph-ok",
        help="Fail if any validated posegraph session summary is not ok.",
    ),
    before_g2o: Optional[str] = typer.Option(
        None,
        "--before-g2o",
        help="Optional pose graph before manual loop closure (pose_graph.g2o).",
    ),
    after_g2o: Optional[str] = typer.Option(
        None,
        "--after-g2o",
        help="Optional pose graph after manual loop closure (pose_graph.g2o).",
    ),
    before_tum: Optional[str] = typer.Option(
        None,
        "--before-tum",
        help="Optional optimized poses before loop closure (TUM).",
    ),
    after_tum: Optional[str] = typer.Option(
        None,
        "--after-tum",
        help="Optional optimized poses after loop closure (TUM).",
    ),
    before_key_point_frame: Optional[str] = typer.Option(
        None,
        "--before-key-point-frame",
        help="Optional key_point_frame dir before loop closure.",
    ),
    after_key_point_frame: Optional[str] = typer.Option(
        None,
        "--after-key-point-frame",
        help="Optional key_point_frame dir after loop closure.",
    ),
    before_session_root: Optional[str] = typer.Option(
        None,
        "--before-session-root",
        help="Optional session root to auto-discover before paths (pose_graph.g2o, optimized_poses_tum.txt, key_point_frame/, map.pcd).",
    ),
    after_session_root: Optional[str] = typer.Option(
        None,
        "--after-session-root",
        help="Optional session root to auto-discover after paths (pose_graph.g2o, optimized_poses_tum.txt, key_point_frame/, map.pcd).",
    ),
    session_map_name: str = typer.Option(
        "map.pcd",
        "--session-map-name",
        help="Map filename to use when discovering from --before/after-session-root.",
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump full result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Report before/after map quality for manual loop-closure workflows."""
    gate = LoopClosureGate(
        min_auc_gain=min_auc_gain,
        max_after_chamfer=max_after_chamfer,
        min_ate_gain=min_ate_gain,
        max_after_ate=max_after_ate,
        require_posegraph_ok=require_posegraph_ok,
    )
    t_list = _parse_thresholds(thresholds)
    try:
        before_discovery = None
        after_discovery = None
        if before_session_root is not None:
            before_discovery = discover_session_paths(before_session_root, map_name=session_map_name)
            before_map = before_discovery["map_path"] or before_map
            before_g2o = before_discovery["g2o_path"] or before_g2o
            before_tum = before_discovery["tum_path"] or before_tum
            before_key_point_frame = before_discovery["key_point_frame_dir"] or before_key_point_frame
            if before_discovery["map_path"] is None and not Path(before_map).exists():
                raise ValueError(
                    "Before session discovery did not find a map file.\n"
                    f"Looked for: {before_discovery['expected']['map_path']}\n"
                    f"Also received before_map: {before_map}"
                )
        if after_session_root is not None:
            after_discovery = discover_session_paths(after_session_root, map_name=session_map_name)
            after_map = after_discovery["map_path"] or after_map
            after_g2o = after_discovery["g2o_path"] or after_g2o
            after_tum = after_discovery["tum_path"] or after_tum
            after_key_point_frame = after_discovery["key_point_frame_dir"] or after_key_point_frame
            if after_discovery["map_path"] is None and not Path(after_map).exists():
                raise ValueError(
                    "After session discovery did not find a map file.\n"
                    f"Looked for: {after_discovery['expected']['map_path']}\n"
                    f"Also received after_map: {after_map}"
                )

        report = build_loop_closure_report(
            before_map=before_map,
            after_map=after_map,
            reference_map=reference_map,
            thresholds=t_list,
            before_trajectory=before_traj,
            after_trajectory=after_traj,
            reference_trajectory=reference_traj,
            trajectory_max_time_delta=traj_max_time_delta,
            trajectory_align_origin=traj_align_origin,
            trajectory_align_rigid=traj_align_rigid,
            before_g2o=before_g2o,
            after_g2o=after_g2o,
            before_tum=before_tum,
            after_tum=after_tum,
            before_key_point_frame_dir=before_key_point_frame,
            after_key_point_frame_dir=after_key_point_frame,
            gate=gate,
        )
        if before_discovery is not None or after_discovery is not None:
            report["discovery"] = {"before": before_discovery, "after": after_discovery}
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    if format_json:
        typer.echo(json.dumps(report, indent=2, default=str))
    else:
        m = report["map"]
        typer.echo(f"Reference: {m['reference']}")
        typer.echo(f"Before:    chamfer={m['before']['chamfer_distance']:.6f} auc={m['before']['auc']:.6f}")
        typer.echo(f"After:     chamfer={m['after']['chamfer_distance']:.6f} auc={m['after']['auc']:.6f}")
        typer.echo(f"Delta:     chamfer={m['delta']['chamfer_distance']:.6f} auc={m['delta']['auc']:.6f}")
        qg = report.get("quality_gate")
        if isinstance(qg, dict):
            typer.echo("Gate:      " + ("PASS" if qg["passed"] else "FAIL"))
            if qg["reasons"]:
                typer.echo("Reasons:   " + "; ".join(qg["reasons"]))

    if output_json:
        _dump_json(report, output_json)
    qg = report.get("quality_gate")
    if isinstance(qg, dict) and qg.get("passed") is False:
        raise typer.Exit(code=1)


@app.command("stats")
def stats_cmd(
    path: str = typer.Argument(..., help="Point cloud file"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Show detailed statistics (density, spacing distribution)."""
    try:
        result = compute_stats(path)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(f"File:      {result['path']}")
        typer.echo(f"Points:    {result['num_points']}")
        typer.echo(f"Volume:    {result['volume']:.4f}")
        typer.echo(f"Density:   {result['density']:.4g} pts/unit³")
        typer.echo(f"Robust volume p01-p99:  {result['robust_volume']:.4f}")
        typer.echo(f"Robust density p01-p99: {result['robust_density']:.4g} pts/unit³")
        typer.echo(
            "Outside robust bbox: "
            f"{result['outside_robust_bbox_count']} ({result['outside_robust_bbox_ratio']:.2%})"
        )
        s = result["spacing"]
        typer.echo(f"Spacing samples: {result['spacing_sample_points']}")
        typer.echo(f"Spacing mean:   {s['mean']:.4f}")
        typer.echo(f"Spacing median: {s['median']:.4f}")
        typer.echo(f"Spacing min:    {s['min']:.4f}")
        typer.echo(f"Spacing max:    {s['max']:.4f}")
        typer.echo(f"Spacing std:    {s['std']:.4f}")
    if output_json:
        _dump_json(result, output_json)


@app.command("mme")
def mme_cmd(
    path: str = typer.Argument(..., help="Point cloud file"),
    neighbors: int = typer.Option(20, "--neighbors", "-n", help="k nearest neighbors (min 4)"),
    max_points: int = typer.Option(500_000, "--max-points", help="Max points before random sampling"),
    workers: int = typer.Option(-1, "--workers", "-w", help="Parallel workers for k-NN (-1 = all CPUs)"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Compute Mean Map Entropy (MME) — local geometric consistency metric (no ground truth needed)."""
    try:
        result = compute_mme(path, k_neighbors=neighbors, max_points=max_points, workers=workers)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(f"File:      {result['path']}")
        typer.echo(f"Points:    {result['num_points']:,}")
        if result["sampled"]:
            typer.echo(f"Used:      {result['num_points_used']:,}  (sampled)")
        else:
            typer.echo(f"Used:      {result['num_points_used']:,}")
        typer.echo(f"k:         {result['k_neighbors']}")
        typer.echo(f"MME:       {result['mme']:.4f}")
    if output_json:
        _dump_json(result, output_json)


@app.command("plane-consistency")
def plane_consistency_cmd(
    path: str = typer.Argument(..., help="Point cloud file (no ground truth required)"),
    voxel_size: float = typer.Option(1.0, "--voxel-size", help="Local patch voxel size in meters"),
    min_points: int = typer.Option(12, "--min-points", help="Minimum points per local patch"),
    output_json: Optional[str] = typer.Option(None, "--output-json"),
    format_json: bool = typer.Option(False, "--format-json"),
) -> None:
    """Experimental PNE/CPV-inspired plane-consistency proxies."""
    from ca.core.plane_consistency import evaluate_plane_consistency

    try:
        result = evaluate_plane_consistency(
            path, voxel_size=voxel_size, min_points=min_points
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)
    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo("Experimental proxy (not the published PNE/CPV metrics)")
        typer.echo(f"Plane normal dispersion: {result['plane_normal_dispersion']:.6f}")
        typer.echo(f"Coplanar offset RMSE:    {result['coplanar_offset_rmse']:.6f} m")
        typer.echo(f"Plane patches:           {result['num_plane_patches']}")
    if output_json:
        _dump_json(result, output_json)


@app.command("view")
def view_cmd(
    paths: List[str] = typer.Argument(..., help="Point cloud file(s) to view"),
) -> None:
    """Open interactive 3D viewer for point cloud(s)."""
    try:
        view(paths)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)
