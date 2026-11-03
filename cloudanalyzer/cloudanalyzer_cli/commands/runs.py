"""SLAM run execution and run evaluation commands."""

import json
from pathlib import Path
from typing import Any, List, Optional
import typer
from ca.core.slam_run import (
    SlamRunRequest,
    discover_frame_paths,
    get_driver,
    write_map_ply,
    write_tum_trajectory,
)
from ca.report import make_run_batch_summary, save_run_batch_report, save_run_report
from ca.run_evaluate import evaluate_run, evaluate_run_batch
from ca.log import setup_logging
from cloudanalyzer_cli._app import (
    app,
    _dump_json,
    _handle_error,
    _attach_protocol,
    _parse_thresholds,
)


@app.command("slam-run")
def slam_run_cmd(
    input_path: str = typer.Argument(
        ...,
        help=(
            "Directory of LiDAR scans (.bin/.pcd/.ply), a frames-list .txt "
            "(one path per line), or a ROS bag recording (.bag/.mcap/.db3)."
        ),
    ),
    output_dir: str = typer.Argument(
        ...,
        help=(
            "Directory to write trajectory.tum, map.ply, and summary.json into. "
            "Created if missing."
        ),
    ),
    driver: str = typer.Option(
        "kiss-icp",
        "--driver",
        help=(
            "SLAM driver to run. Built-in: 'kiss-icp' (default, adopted), 'cloudanalyzer' (the Rust core's odometry, "
            "nothing to install), "
            "'kiss-slam' (experimental — pose-graph + loop closures), "
            "'small-gicp' (experimental — scan-to-map VGICP). Third-party "
            "packages can register additional drivers under the "
            "'cloudanalyzer.slam_run_drivers' entry-point group; see "
            "`docs/commands/slam-run.md` for the contract."
        ),
    ),
    max_range: Optional[float] = typer.Option(
        None,
        "--max-range",
        help="Drop scan points farther than this from the sensor (meters).",
    ),
    voxel_size: Optional[float] = typer.Option(
        None,
        "--voxel-size",
        help="Driver-side voxel grid size for the local map (meters). Driver default if omitted.",
    ),
    deskew: bool = typer.Option(
        False,
        "--deskew",
        help=(
            "Enable KISS-ICP motion-deskew. Requires meaningful per-point "
            "timestamps in the input frames; default off because .bin / .pcd "
            "dumps don't typically carry them."
        ),
    ),
    max_frames: Optional[int] = typer.Option(
        None, "--max-frames", help="Cap on the number of frames consumed."
    ),
    pointcloud_topic: Optional[str] = typer.Option(
        None,
        "--pointcloud-topic",
        help="ROS topic when --input is a bag recording with PointCloud2 scans",
    ),
    frame_period: float = typer.Option(
        0.1,
        "--frame-period",
        help=(
            "Fallback per-frame time spacing (seconds), used when no explicit "
            "timestamps file is provided."
        ),
    ),
    evaluate_run_flag: bool = typer.Option(
        False,
        "--evaluate",
        help=(
            "After driving the SLAM, evaluate the resulting map + trajectory "
            "against --reference-map and --reference-trajectory using ca run-evaluate."
        ),
    ),
    reference_map: Optional[str] = typer.Option(
        None,
        "--reference-map",
        help="Reference map cloud for --evaluate (pcd/ply/las).",
    ),
    reference_trajectory: Optional[str] = typer.Option(
        None,
        "--reference-trajectory",
        help="Reference trajectory for --evaluate (.csv/.tum/.txt).",
    ),
    format_json: bool = typer.Option(
        False, "--format-json", help="Print the run summary as JSON on stdout."
    ),
) -> None:
    """Drive a LiDAR-odometry pipeline end-to-end on a sequence of scans.

    Produces three artifacts under OUTPUT_DIR:

    - ``trajectory.tum`` — estimated sensor poses, consumable by ``ca traj-evaluate``.
    - ``map.ply`` — accumulated world-frame map, consumable by ``ca evaluate``.
    - ``summary.json`` — driver name, runtime, frame count, plus the optional
      ``--evaluate`` block when a reference map/trajectory is supplied.

    Example::

        ca slam-run scans/ runs/seq01 --driver kiss-icp --max-range 80
        ca slam-run scans/ runs/seq01 --evaluate \\
            --reference-map ref/map.pcd \\
            --reference-trajectory ref/poses.tum
    """

    if evaluate_run_flag and (reference_map is None or reference_trajectory is None):
        typer.echo(
            "--evaluate requires both --reference-map and --reference-trajectory.",
            err=True,
        )
        raise typer.Exit(code=2)

    in_path = Path(input_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    from ca.core.bag_ingest import is_bag_path, materialize_pointcloud_bag

    timestamps_s: tuple[float, ...] | None = None
    try:
        if is_bag_path(in_path):
            frame_paths, timestamps_s = materialize_pointcloud_bag(
                in_path,
                out_path / ".ingested_frames",
                topic=pointcloud_topic,
                max_frames=max_frames,
            )
        else:
            frame_paths = discover_frame_paths(in_path)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)

    request = SlamRunRequest(
        frame_paths=tuple(frame_paths),
        timestamps_s=timestamps_s,
        frame_period_s=frame_period,
        max_range_m=max_range,
        voxel_size_m=voxel_size,
        deskew=deskew,
        max_frames=max_frames,
    )

    setup_logging()
    from ca.log import logger as _logger
    _logger.info(
        "slam-run: %d frames -> %s (driver=%s)",
        len(frame_paths),
        out_path,
        driver,
    )

    try:
        drv = get_driver(driver)
        result = drv.run(request)
    except (ImportError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)

    trajectory_path = out_path / "trajectory.tum"
    map_path = out_path / "map.ply"
    write_tum_trajectory(trajectory_path, result.poses, result.timestamps_s)
    write_map_ply(map_path, result.map_points)

    summary: dict[str, Any] = {
        "driver": result.driver,
        "frames_processed": result.frames_processed,
        "runtime_s": float(result.runtime_s),
        "map_points": int(result.map_points.shape[0]),
        "trajectory_path": str(trajectory_path),
        "map_path": str(map_path),
        "driver_metadata": result.metadata,
    }

    if evaluate_run_flag:
        try:
            assert reference_map is not None and reference_trajectory is not None
            eval_result = evaluate_run(
                str(map_path),
                reference_map,
                str(trajectory_path),
                reference_trajectory,
            )
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Error during --evaluate: {e}", err=True)
            raise typer.Exit(code=2)
        summary["evaluate"] = eval_result

    summary_path = out_path / "summary.json"
    _dump_json(summary, str(summary_path))

    typer.echo(f"slam-run: wrote {summary_path}")
    typer.echo(f"  trajectory: {trajectory_path}")
    typer.echo(f"  map: {map_path}  ({result.map_points.shape[0]} pts)")
    typer.echo(
        f"  driver={result.driver} frames={result.frames_processed} "
        f"runtime={result.runtime_s:.3f}s"
    )

    if format_json:
        typer.echo(json.dumps(summary, indent=2, default=str))


@app.command("run-evaluate")
def run_evaluate_cmd(
    map_path: str = typer.Argument(..., help="Estimated map point cloud (pcd/ply/las)"),
    map_reference: str = typer.Argument(..., help="Reference map point cloud (pcd/ply/las)"),
    trajectory_path: str = typer.Argument(..., help="Estimated trajectory (.csv/.tum/.txt)"),
    trajectory_reference: str = typer.Argument(..., help="Reference trajectory (.csv/.tum/.txt)"),
    thresholds: Optional[str] = typer.Option(
        None, "--thresholds", "-t",
        help="Comma-separated distance thresholds for map F1/AUC evaluation",
    ),
    max_time_delta: float = typer.Option(
        0.05, "--max-time-delta",
        help="Max timestamp gap allowed for trajectory matching/interpolation (seconds)",
    ),
    align_origin: bool = typer.Option(
        False, "--align-origin",
        help="Translate the estimated trajectory so its first matched pose aligns to the reference",
    ),
    align_rigid: bool = typer.Option(
        False, "--align-rigid",
        help="Fit a rigid transform (rotation + translation) from estimated to reference positions",
    ),
    min_auc: Optional[float] = typer.Option(
        None, "--min-auc",
        help="Minimum map AUC required; contributes to overall quality gate",
    ),
    max_chamfer: Optional[float] = typer.Option(
        None, "--max-chamfer",
        help="Maximum map Chamfer distance allowed; contributes to overall quality gate",
    ),
    max_ate: Optional[float] = typer.Option(
        None, "--max-ate",
        help="Maximum trajectory ATE RMSE allowed; contributes to overall quality gate",
    ),
    max_rpe: Optional[float] = typer.Option(
        None, "--max-rpe",
        help="Maximum trajectory translational RPE RMSE allowed; contributes to overall quality gate",
    ),
    max_rpe_rotation_deg: Optional[float] = typer.Option(
        None, "--max-rpe-rotation-deg",
        help="Maximum trajectory rotational RPE RMSE in degrees; requires quaternion orientations",
    ),
    rpe_distance: Optional[List[float]] = typer.Option(
        None,
        "--rpe-distance",
        help="Distance window in meters for KITTI-style RPE (repeatable)",
    ),
    max_drift: Optional[float] = typer.Option(
        None, "--max-drift",
        help="Maximum trajectory endpoint drift allowed; contributes to overall quality gate",
    ),
    min_coverage: Optional[float] = typer.Option(
        None, "--min-coverage",
        help="Minimum trajectory matched-pose coverage ratio required; contributes to overall quality gate",
    ),
    report: Optional[str] = typer.Option(
        None, "--report",
        help="Write combined run report (.md or .html)",
    ),
    protocol_path: Optional[str] = typer.Option(
        None,
        "--protocol",
        help="Versioned cloudanalyzer.protocol.v1 file to attach to the result.",
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Evaluate one map output and one trajectory output together."""
    thresh_list = _parse_thresholds(thresholds)

    try:
        result = evaluate_run(
            map_path,
            map_reference,
            trajectory_path,
            trajectory_reference,
            thresholds=thresh_list,
            max_time_delta=max_time_delta,
            align_origin=align_origin,
            align_rigid=align_rigid,
            min_auc=min_auc,
            max_chamfer=max_chamfer,
            max_ate=max_ate,
            max_rpe=max_rpe,
            max_rpe_rotation_deg=max_rpe_rotation_deg,
            rpe_distances_m=rpe_distance,
            max_drift=max_drift,
            min_coverage=min_coverage,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    _attach_protocol(
        result,
        protocol_path,
        command="ca run-evaluate",
        kind="run",
        inputs=(map_path, map_reference, trajectory_path, trajectory_reference),
        options={
            "thresholds_m": list(thresh_list) if thresh_list is not None else None,
            "max_time_delta_s": max_time_delta,
            "align_origin": align_origin,
            "align_rigid": align_rigid,
            "min_auc": min_auc,
            "max_chamfer": max_chamfer,
            "max_ate": max_ate,
            "max_rpe": max_rpe,
            "max_rpe_rotation_deg": max_rpe_rotation_deg,
            "rpe_distance_m": list(rpe_distance) if rpe_distance is not None else None,
            "max_drift": max_drift,
            "min_coverage": min_coverage,
        },
    )

    if report:
        try:
            save_run_report(result, report)
        except ValueError as e:
            _handle_error(e)

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        map_result = result["map"]
        trajectory_result = result["trajectory"]
        best_f1 = map_result["best_f1"]
        matching = trajectory_result["matching"]
        typer.echo(
            f"Map: {map_result['source_points']} pts | Ref: {map_result['target_points']} pts"
        )
        typer.echo(
            f"  Chamfer={map_result['chamfer_distance']:.4f}  "
            f"Hausdorff={map_result['hausdorff_distance']:.4f}  "
            f"AUC={map_result['auc']:.4f}"
        )
        typer.echo(
            f"  Best F1={best_f1['f1']:.4f} @ d={best_f1['threshold']:.2f}"
        )
        trajectory_rpe_rotation = trajectory_result.get("rpe_rotation")
        trajectory_rpe_rotation_label = (
            "n/a"
            if trajectory_rpe_rotation is None
            else f"{trajectory_rpe_rotation['rmse']:.4f}deg"
        )
        typer.echo(
            f"Trajectory: matched={matching['matched_poses']} ({matching['coverage_ratio']:.1%})  "
            f"ATE={trajectory_result['ate']['rmse']:.4f}  "
            f"RPE={trajectory_result['rpe_translation']['rmse']:.4f}  "
            f"RPErot={trajectory_rpe_rotation_label}  "
            f"Drift={trajectory_result['drift']['endpoint']:.4f}  "
            f"Align={trajectory_result['alignment']['mode']}"
        )
        overall_gate = result["overall_quality_gate"]
        if overall_gate is not None:
            typer.echo("")
            typer.echo(f"Overall Quality Gate: {'PASS' if overall_gate['passed'] else 'FAIL'}")
            for reason in overall_gate["reasons"]:
                typer.echo(f"  - {reason}")
        if report:
            typer.echo(f"Report: {report}")

    if output_json:
        _dump_json(result, output_json)
    if result["overall_quality_gate"] is not None and not result["overall_quality_gate"]["passed"]:
        raise typer.Exit(code=1)


@app.command("run-batch")
def run_batch_cmd(
    map_dir: str = typer.Argument(..., help="Directory containing estimated map point clouds"),
    map_reference_dir: str = typer.Option(
        ..., "--map-reference-dir",
        help="Directory containing reference maps matched by relative path or stem",
    ),
    trajectory_dir: str = typer.Option(
        ..., "--trajectory-dir",
        help="Directory containing estimated trajectories matched to the map outputs",
    ),
    trajectory_reference_dir: str = typer.Option(
        ..., "--trajectory-reference-dir",
        help="Directory containing reference trajectories matched by relative path or stem",
    ),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Scan subdirectories"),
    thresholds: Optional[str] = typer.Option(
        None, "--thresholds", "-t",
        help="Comma-separated distance thresholds for map F1/AUC evaluation",
    ),
    max_time_delta: float = typer.Option(
        0.05, "--max-time-delta",
        help="Max timestamp gap allowed for trajectory matching/interpolation (seconds)",
    ),
    align_origin: bool = typer.Option(
        False, "--align-origin",
        help="Translate each estimated trajectory so its first matched pose aligns to the reference",
    ),
    align_rigid: bool = typer.Option(
        False, "--align-rigid",
        help="Fit a rigid transform (rotation + translation) from each estimated trajectory to its reference",
    ),
    min_auc: Optional[float] = typer.Option(
        None, "--min-auc",
        help="Minimum map AUC required; contributes to the overall quality gate",
    ),
    max_chamfer: Optional[float] = typer.Option(
        None, "--max-chamfer",
        help="Maximum map Chamfer distance allowed; contributes to the overall quality gate",
    ),
    max_ate: Optional[float] = typer.Option(
        None, "--max-ate",
        help="Maximum trajectory ATE RMSE allowed; contributes to the overall quality gate",
    ),
    max_rpe: Optional[float] = typer.Option(
        None, "--max-rpe",
        help="Maximum trajectory translational RPE RMSE allowed; contributes to the overall quality gate",
    ),
    max_rpe_rotation_deg: Optional[float] = typer.Option(
        None, "--max-rpe-rotation-deg",
        help="Maximum trajectory rotational RPE RMSE in degrees; requires quaternion orientations",
    ),
    rpe_distance: Optional[List[float]] = typer.Option(
        None,
        "--rpe-distance",
        help="Distance window in meters for KITTI-style RPE (repeatable)",
    ),
    max_drift: Optional[float] = typer.Option(
        None, "--max-drift",
        help="Maximum trajectory endpoint drift allowed; contributes to the overall quality gate",
    ),
    min_coverage: Optional[float] = typer.Option(
        None, "--min-coverage",
        help="Minimum trajectory matched-pose coverage ratio required; contributes to the overall quality gate",
    ),
    report: Optional[str] = typer.Option(
        None, "--report",
        help="Write combined run-batch report (.md or .html)",
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump results as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Evaluate multiple map + trajectory runs together."""
    if format_json:
        import logging

        logging.getLogger("ca").setLevel(logging.ERROR)

    thresh_list = _parse_thresholds(thresholds)

    try:
        results = evaluate_run_batch(
            map_dir,
            map_reference_dir,
            trajectory_dir,
            trajectory_reference_dir,
            recursive=recursive,
            thresholds=thresh_list,
            max_time_delta=max_time_delta,
            align_origin=align_origin,
            align_rigid=align_rigid,
            min_auc=min_auc,
            max_chamfer=max_chamfer,
            max_ate=max_ate,
            max_rpe=max_rpe,
            max_rpe_rotation_deg=max_rpe_rotation_deg,
            rpe_distances_m=rpe_distance,
            max_drift=max_drift,
            min_coverage=min_coverage,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    if report:
        try:
            save_run_batch_report(
                results,
                map_reference_dir,
                trajectory_reference_dir,
                report,
                min_auc=min_auc,
                max_chamfer=max_chamfer,
                max_ate=max_ate,
                max_rpe=max_rpe,
                max_rpe_rotation_deg=max_rpe_rotation_deg,
                max_drift=max_drift,
                min_coverage=min_coverage,
            )
        except ValueError as e:
            _handle_error(e)

    summary = make_run_batch_summary(
        results,
        map_reference_dir,
        trajectory_reference_dir,
        min_auc=min_auc,
        max_chamfer=max_chamfer,
        max_ate=max_ate,
        max_rpe=max_rpe,
        max_rpe_rotation_deg=max_rpe_rotation_deg,
        max_drift=max_drift,
        min_coverage=min_coverage,
    )
    gate = summary["quality_gate"]
    should_fail = gate is not None and gate["fail_count"] > 0

    if format_json:
        typer.echo(json.dumps(results, indent=2))
    else:
        for item in results:
            status = ""
            map_status = ""
            trajectory_status = ""
            if item["overall_quality_gate"] is not None:
                status = "  PASS" if item["overall_quality_gate"]["passed"] else "  FAIL"
                map_status = (
                    "  Map=PASS"
                    if item["overall_quality_gate"]["map_passed"] is not False
                    else "  Map=FAIL"
                )
                trajectory_status = (
                    "  Trajectory=PASS"
                    if item["overall_quality_gate"]["trajectory_passed"] is not False
                    else "  Trajectory=FAIL"
                )
            typer.echo(
                f"  {item['id']}: map_auc={item['map']['auc']:.4f}  "
                f"map_chamfer={item['map']['chamfer_distance']:.4f}  "
                f"traj_ate={item['trajectory']['ate']['rmse']:.4f}  "
                f"traj_drift={item['trajectory']['drift']['endpoint']:.4f}  "
                f"coverage={item['trajectory']['matching']['coverage_ratio']:.1%}"
                f"{map_status}{trajectory_status}"
                f"{status}"
            )
        typer.echo(f"Mean Map AUC: {summary['mean_map_auc']:.4f}")
        typer.echo(f"Mean Map Chamfer: {summary['mean_map_chamfer']:.4f}")
        typer.echo(f"Mean Trajectory ATE RMSE: {summary['mean_traj_ate_rmse']:.4f}")
        typer.echo(f"Mean Trajectory Drift: {summary['mean_traj_drift']:.4f}")
        typer.echo(f"Mean Trajectory Coverage: {summary['mean_traj_coverage']:.1%}")
        if gate is not None:
            typer.echo(
                f"Quality Gate: pass={gate['pass_count']} fail={gate['fail_count']}"
            )
            typer.echo(f"Map Failures: {gate['map_fail_count']}")
            typer.echo(f"Trajectory Failures: {gate['trajectory_fail_count']}")
        typer.echo(f"Map Reference Dir: {map_reference_dir}")
        typer.echo(f"Trajectory Reference Dir: {trajectory_reference_dir}")
        typer.echo(f"Total runs: {len(results)}")
        if report:
            typer.echo(f"Report: {report}")

    if output_json:
        _dump_json(results, output_json)
    if should_fail:
        raise typer.Exit(code=1)
