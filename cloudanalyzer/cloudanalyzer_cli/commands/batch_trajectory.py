"""Batch, map and trajectory evaluation commands."""

import json
from typing import List, Optional
import typer
from ca.batch import batch_evaluate, batch_info, trajectory_batch_evaluate
from ca.density_map import density_map
from ca.evaluate import evaluate, plot_f1_curve
from ca.report import (
    make_batch_summary,
    make_trajectory_batch_summary,
    save_batch_report,
    save_trajectory_batch_report,
    save_trajectory_report,
)
from ca.trajectory import evaluate_trajectory
from cloudanalyzer_cli._app import (
    app,
    _dump_json,
    _handle_error,
    _attach_protocol,
    _parse_thresholds,
)


@app.command("batch")
def batch_cmd(
    directory: str = typer.Argument(..., help="Directory containing point cloud files"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Scan subdirectories"),
    evaluate_against: Optional[str] = typer.Option(
        None, "--evaluate",
        help="Evaluate each file against this reference point cloud",
    ),
    report: Optional[str] = typer.Option(
        None, "--report",
        help="Write batch evaluation report (.md or .html; requires --evaluate)",
    ),
    min_auc: Optional[float] = typer.Option(
        None, "--min-auc",
        help="Minimum AUC required to pass; exits with code 1 if any file fails",
    ),
    max_chamfer: Optional[float] = typer.Option(
        None, "--max-chamfer",
        help="Maximum Chamfer distance allowed; exits with code 1 if any file fails",
    ),
    compressed_dir: Optional[str] = typer.Option(
        None, "--compressed-dir",
        help="Directory containing compressed artifacts to compare by size",
    ),
    baseline_dir: Optional[str] = typer.Option(
        None, "--baseline-dir",
        help="Directory containing original uncompressed files for size ratio baseline",
    ),
    thresholds: Optional[str] = typer.Option(
        None, "--thresholds", "-t",
        help="Comma-separated distance thresholds for --evaluate",
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump results as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Run info on all point cloud files in a directory."""
    thresh_list = _parse_thresholds(thresholds)
    if thresh_list and not evaluate_against:
        typer.echo("Error: --thresholds requires --evaluate", err=True)
        raise typer.Exit(code=1)
    if report and not evaluate_against:
        typer.echo("Error: --report requires --evaluate", err=True)
        raise typer.Exit(code=1)
    if (min_auc is not None or max_chamfer is not None) and not evaluate_against:
        typer.echo("Error: --min-auc/--max-chamfer require --evaluate", err=True)
        raise typer.Exit(code=1)
    if (compressed_dir is not None or baseline_dir is not None) and not evaluate_against:
        typer.echo("Error: --compressed-dir/--baseline-dir require --evaluate", err=True)
        raise typer.Exit(code=1)

    if format_json:
        import logging
        logging.getLogger("ca").setLevel(logging.ERROR)
    try:
        if evaluate_against:
            results = batch_evaluate(
                directory,
                evaluate_against,
                recursive=recursive,
                thresholds=thresh_list,
                min_auc=min_auc,
                max_chamfer=max_chamfer,
                compressed_dir=compressed_dir,
                baseline_dir=baseline_dir,
            )
        else:
            results = batch_info(directory, recursive=recursive)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    if report and evaluate_against:
        try:
            save_batch_report(
                results,
                evaluate_against,
                report,
                min_auc=min_auc,
                max_chamfer=max_chamfer,
            )
        except ValueError as e:
            _handle_error(e)

    summary = None
    should_fail = False
    if evaluate_against and results:
        summary = make_batch_summary(
            results,
            evaluate_against,
            min_auc=min_auc,
            max_chamfer=max_chamfer,
        )
        gate = summary["quality_gate"]
        should_fail = gate is not None and gate["fail_count"] > 0

    if format_json:
        typer.echo(json.dumps(results, indent=2))
    else:
        if evaluate_against:
            for item in results:
                best_f1 = item["best_f1"]
                status = ""
                if item["quality_gate"] is not None:
                    status = "  PASS" if item["quality_gate"]["passed"] else "  FAIL"
                compression_text = ""
                compression = item.get("compression")
                if compression is not None:
                    compression_text = f"  Size={compression['size_ratio']:.4f}"
                    if compression.get("pareto_optimal"):
                        compression_text += "  Pareto"
                    if compression.get("recommended"):
                        compression_text += "  Recommended"
                typer.echo(
                    f"  {item['path']}: {item['num_points']} pts  "
                    f"Chamfer={item['chamfer_distance']:.4f}  "
                    f"AUC={item['auc']:.4f}  "
                    f"Best F1={best_f1['f1']:.4f} @ d={best_f1['threshold']:.2f}"
                    f"{compression_text}"
                    f"{status}"
                )
            if summary is not None:
                typer.echo(f"Mean AUC: {summary['mean_auc']:.4f}")
                typer.echo(f"Mean Chamfer: {summary['mean_chamfer_distance']:.4f}")
                compression = summary["compression"]
                if compression is not None:
                    typer.echo(f"Mean Size Ratio: {compression['mean_size_ratio']:.4f}")
                    typer.echo(
                        f"Mean Space Saving: {compression['mean_space_saving_ratio']:.1%}"
                    )
                    typer.echo(
                        f"Pareto Candidates: {compression['pareto_optimal_count']}"
                    )
                    recommended = compression["recommended_item"]
                    if recommended is not None and recommended.get("compression") is not None:
                        typer.echo(
                            "Recommended: "
                            f"{recommended['path']}  "
                            f"Size={recommended['compression']['size_ratio']:.4f}  "
                            f"AUC={recommended['auc']:.4f}"
                        )
                    else:
                        typer.echo("Recommended: none")
                gate = summary["quality_gate"]
                if gate is not None:
                    typer.echo(
                        f"Quality Gate: pass={gate['pass_count']} fail={gate['fail_count']}"
                    )
            typer.echo(f"Reference: {evaluate_against}")
        else:
            for info in results:
                typer.echo(f"  {info['path']}: {info['num_points']} pts")
        typer.echo(f"Total files: {len(results)}")
        if report:
            typer.echo(f"Report: {report}")
    if output_json:
        _dump_json(results, output_json)
    if should_fail:
        raise typer.Exit(code=1)


@app.command("density-map")
def density_map_cmd(
    input_path: str = typer.Argument(..., help="Input point cloud file"),
    output: str = typer.Option(..., "--output", "-o", help="Output image path (png)"),
    resolution: float = typer.Option(0.5, "--resolution", "-r", help="Grid cell size"),
    axis: str = typer.Option("z", "--axis", "-a", help="Projection axis: x, y, or z"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Generate a 2D density heatmap of a point cloud."""
    try:
        result = density_map(input_path, output, resolution=resolution, axis=axis)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    typer.echo(f"Points:       {result['num_points']}")
    typer.echo(f"Projection:   {result['projection_axis']}-axis")
    typer.echo(f"Grid:         {result['grid_size'][0]}x{result['grid_size'][1]}")
    typer.echo(f"Occupied:     {result['occupied_cells']} cells")
    typer.echo(f"Max density:  {result['max_density']}")
    typer.echo(f"Mean density: {result['mean_density']:.1f}")
    typer.echo(f"Saved:        {result['output']}")
    if output_json:
        _dump_json(result, output_json)


@app.command("evaluate")
def evaluate_cmd(
    source: str = typer.Argument(..., help="Source (estimated) point cloud"),
    target: str = typer.Argument(..., help="Target (reference) point cloud"),
    thresholds: Optional[str] = typer.Option(
        None, "--thresholds", "-t",
        help="Comma-separated distance thresholds (default: 0.05,0.1,0.2,0.3,0.5,1.0)",
    ),
    plot: Optional[str] = typer.Option(None, "--plot", help="Output path for F1 curve plot (png)"),
    protocol_path: Optional[str] = typer.Option(
        None,
        "--protocol",
        help="Versioned cloudanalyzer.protocol.v1 file to attach to the result.",
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Evaluate point cloud similarity (F1, Chamfer, Hausdorff, AUC)."""
    thresh_list = _parse_thresholds(thresholds)

    try:
        result = evaluate(source, target, thresholds=thresh_list)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    _attach_protocol(
        result,
        protocol_path,
        command="ca evaluate",
        kind="point_cloud",
        inputs=(source, target),
        options={"thresholds_m": list(thresh_list) if thresh_list is not None else None},
    )

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(f"Source: {result['source_points']} pts | Target: {result['target_points']} pts")
        typer.echo("")
        typer.echo(f"Chamfer Distance:  {result['chamfer_distance']:.4f}")
        typer.echo(f"Hausdorff Distance: {result['hausdorff_distance']:.4f}")
        typer.echo(f"AUC (F1):          {result['auc']:.4f}")
        typer.echo("")
        typer.echo("F1 Scores:")
        for s in result["f1_scores"]:
            typer.echo(f"  d={s['threshold']:.2f}  P={s['precision']:.4f}  R={s['recall']:.4f}  F1={s['f1']:.4f}")
        typer.echo("")
        ds = result["distance_stats"]
        typer.echo(f"S->T  mean={ds['source_to_target']['mean']:.4f}  median={ds['source_to_target']['median']:.4f}  max={ds['source_to_target']['max']:.4f}")
        typer.echo(f"T->S  mean={ds['target_to_source']['mean']:.4f}  median={ds['target_to_source']['median']:.4f}  max={ds['target_to_source']['max']:.4f}")
    if plot:
        plot_f1_curve(result, plot)
        typer.echo(f"Plot: {plot}")
    if output_json:
        _dump_json(result, output_json)


@app.command("traj-evaluate")
def traj_evaluate_cmd(
    estimated: str = typer.Argument(..., help="Estimated trajectory (.csv/.tum/.txt or .bag/.mcap/.db3)"),
    reference: str = typer.Argument(..., help="Reference trajectory (.csv/.tum/.txt)"),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        help="ROS topic when estimated (or reference) input is a bag recording",
    ),
    frame: Optional[str] = typer.Option(
        None,
        "--frame",
        help="Child frame id when the bag topic carries tf2_msgs/msg/TFMessage",
    ),
    max_time_delta: float = typer.Option(
        0.05, "--max-time-delta",
        help="Max timestamp gap allowed for matching/interpolation (seconds)",
    ),
    align_origin: bool = typer.Option(
        False, "--align-origin",
        help="Translate the estimated trajectory so its first matched pose aligns to the reference",
    ),
    align_rigid: bool = typer.Option(
        False, "--align-rigid",
        help="Fit a rigid transform (rotation + translation) from estimated to reference positions",
    ),
    max_ate: Optional[float] = typer.Option(
        None, "--max-ate",
        help="Maximum ATE RMSE allowed; exits with code 1 if exceeded",
    ),
    max_rpe: Optional[float] = typer.Option(
        None, "--max-rpe",
        help="Maximum translational RPE RMSE allowed; exits with code 1 if exceeded",
    ),
    max_rpe_rotation_deg: Optional[float] = typer.Option(
        None, "--max-rpe-rotation-deg",
        help="Maximum rotational RPE RMSE in degrees; requires quaternion orientations",
    ),
    max_drift: Optional[float] = typer.Option(
        None, "--max-drift",
        help="Maximum endpoint drift allowed; exits with code 1 if exceeded",
    ),
    min_coverage: Optional[float] = typer.Option(
        None, "--min-coverage",
        help="Minimum matched-pose coverage ratio required (0-1); exits with code 1 if not met",
    ),
    max_lateral: Optional[float] = typer.Option(
        None, "--max-lateral",
        help="Maximum lateral RMSE allowed; exits with code 1 if exceeded",
    ),
    max_longitudinal: Optional[float] = typer.Option(
        None, "--max-longitudinal",
        help="Maximum longitudinal RMSE allowed; exits with code 1 if exceeded",
    ),
    rpe_distance: Optional[List[float]] = typer.Option(
        None,
        "--rpe-distance",
        help="Distance window in meters for KITTI-style RPE (repeatable)",
    ),
    report: Optional[str] = typer.Option(
        None, "--report",
        help="Write trajectory report (.md or .html)",
    ),
    protocol_path: Optional[str] = typer.Option(
        None,
        "--protocol",
        help="Versioned cloudanalyzer.protocol.v1 file to attach to the result.",
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Evaluate a trajectory against a reference trajectory."""
    try:
        result = evaluate_trajectory(
            estimated,
            reference,
            topic=topic,
            frame=frame,
            max_time_delta=max_time_delta,
            align_origin=align_origin,
            align_rigid=align_rigid,
            max_ate=max_ate,
            max_rpe=max_rpe,
            max_rpe_rotation_deg=max_rpe_rotation_deg,
            max_drift=max_drift,
            min_coverage=min_coverage,
            max_lateral=max_lateral,
            max_longitudinal=max_longitudinal,
            rpe_distances_m=rpe_distance,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    _attach_protocol(
        result,
        protocol_path,
        command="ca traj-evaluate",
        kind="trajectory",
        inputs=(estimated, reference),
        options={
            "max_time_delta_s": max_time_delta,
            "align_origin": align_origin,
            "align_rigid": align_rigid,
            "rpe_distance_m": list(rpe_distance) if rpe_distance is not None else None,
            "max_ate": max_ate,
            "max_rpe": max_rpe,
            "max_rpe_rotation_deg": max_rpe_rotation_deg,
            "max_drift": max_drift,
            "min_coverage": min_coverage,
            "max_lateral": max_lateral,
            "max_longitudinal": max_longitudinal,
            "topic": topic,
            "frame": frame,
        },
    )

    if report:
        try:
            save_trajectory_report(result, report)
        except ValueError as e:
            _handle_error(e)

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        alignment = result["alignment"]
        matching = result["matching"]
        ate = result["ate"]
        rpe = result["rpe_translation"]
        drift = result["drift"]
        typer.echo(
            f"Estimated: {matching['estimated_poses']} poses | "
            f"Reference: {matching['reference_poses']} poses"
        )
        typer.echo(
            f"Matched:   {matching['matched_poses']} "
            f"({matching['coverage_ratio']:.1%})  "
            f"Time delta mean/max={matching['mean_abs_time_delta']:.4f}/{matching['max_abs_time_delta']:.4f}s"
        )
        if alignment["mode"] != "none":
            translation = alignment["translation"]
            typer.echo(
                f"Alignment: {alignment['mode']}  "
                f"translation=[{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}]"
            )
        typer.echo("")
        typer.echo(
            f"ATE RMSE:  {ate['rmse']:.4f}  "
            f"Mean={ate['mean']:.4f}  Max={ate['max']:.4f}"
        )
        typer.echo(
            f"RPE RMSE:  {rpe['rmse']:.4f}  "
            f"Mean={rpe['mean']:.4f}  Max={rpe['max']:.4f}"
        )
        if result["rpe_rotation"] is not None:
            rotation_rpe = result["rpe_rotation"]
            typer.echo(
                f"RPE Rot.:  {rotation_rpe['rmse']:.4f} deg  "
                f"Mean={rotation_rpe['mean']:.4f}  Max={rotation_rpe['max']:.4f}"
            )
        for distance_result in result["rpe_distance"]:
            translation_result = distance_result["translation"]
            if translation_result is not None:
                typer.echo(
                    f"RPE @{distance_result['delta_m']:.1f}m: "
                    f"{translation_result['rmse']:.4f}m "
                    f"({distance_result['pairs']} pairs)"
                )
        lateral = result["lateral"]
        longitudinal = result["longitudinal"]
        typer.echo(
            f"Lateral:   {lateral['rmse']:.4f}  "
            f"Mean={lateral['mean']:.4f}  Max={lateral['max']:.4f}"
        )
        typer.echo(
            f"Longitudinal: {longitudinal['rmse']:.4f}  "
            f"Mean={longitudinal['mean']:.4f}  Max={longitudinal['max']:.4f}"
        )
        typer.echo(
            f"Drift:     {drift['endpoint']:.4f}  "
            f"Ratio={drift['ratio_to_reference_path_length']:.4f}"
            if drift["ratio_to_reference_path_length"] is not None
            else f"Drift:     {drift['endpoint']:.4f}"
        )
        gate = result["quality_gate"]
        if gate is not None:
            typer.echo("")
            typer.echo(f"Quality Gate: {'PASS' if gate['passed'] else 'FAIL'}")
            for reason in gate["reasons"]:
                typer.echo(f"  - {reason}")
        if report:
            typer.echo(f"Report: {report}")

    if output_json:
        _dump_json(result, output_json)
    if result["quality_gate"] is not None and not result["quality_gate"]["passed"]:
        raise typer.Exit(code=1)


@app.command("uncertainty-evaluate")
def uncertainty_evaluate_cmd(
    estimated: str = typer.Argument(..., help="JSON trajectory with position covariance"),
    reference: str = typer.Argument(..., help="Ground-truth CSV/TUM trajectory"),
    max_time_delta: float = typer.Option(0.05, "--max-time-delta"),
    align_mode: str = typer.Option("none", "--align-mode", help="none, origin, or rigid"),
    confidence: float = typer.Option(0.95, "--confidence"),
    output_json: Optional[str] = typer.Option(None, "--output-json"),
    format_json: bool = typer.Option(False, "--format-json"),
) -> None:
    """Evaluate position NEES and chi-square coverage against ground truth."""
    from ca.core.uncertainty_evaluate import evaluate_uncertainty

    try:
        result = evaluate_uncertainty(
            estimated,
            reference,
            max_time_delta=max_time_delta,
            align_mode=align_mode,
            confidence=confidence,
        )
    except (FileNotFoundError, ValueError) as exc:
        _handle_error(exc)
    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(f"Matched states:       {result['num_matched_states']}")
        typer.echo(f"Mean position NEES (DoF={result['dof']}): {result['mean_position_nees']:.4f}")
        typer.echo(f"Normalized mean position NEES: {result['normalized_mean_position_nees']:.4f}")
        typer.echo(f"Coverage:             {result['coverage']:.1%}")
        typer.echo(f"Interpretation:       {result['statistical_interpretation']}")
    if output_json:
        _dump_json(result, output_json)


@app.command("traj-batch")
def traj_batch_cmd(
    directory: str = typer.Argument(..., help="Directory containing estimated trajectory files"),
    reference_dir: str = typer.Option(
        ..., "--reference-dir",
        help="Directory containing reference trajectory files matched by relative path or stem",
    ),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Scan subdirectories"),
    max_time_delta: float = typer.Option(
        0.05, "--max-time-delta",
        help="Max timestamp gap allowed for matching/interpolation (seconds)",
    ),
    align_origin: bool = typer.Option(
        False, "--align-origin",
        help="Translate each estimated trajectory so its first matched pose aligns to the reference",
    ),
    align_rigid: bool = typer.Option(
        False, "--align-rigid",
        help="Fit a rigid transform (rotation + translation) from each estimated trajectory to its reference",
    ),
    max_ate: Optional[float] = typer.Option(
        None, "--max-ate",
        help="Maximum ATE RMSE allowed; exits with code 1 if any file fails",
    ),
    max_rpe: Optional[float] = typer.Option(
        None, "--max-rpe",
        help="Maximum translational RPE RMSE allowed; exits with code 1 if any file fails",
    ),
    max_rpe_rotation_deg: Optional[float] = typer.Option(
        None, "--max-rpe-rotation-deg",
        help="Maximum rotational RPE RMSE in degrees; requires quaternion orientations",
    ),
    max_drift: Optional[float] = typer.Option(
        None, "--max-drift",
        help="Maximum endpoint drift allowed; exits with code 1 if any file fails",
    ),
    min_coverage: Optional[float] = typer.Option(
        None, "--min-coverage",
        help="Minimum matched-pose coverage ratio required (0-1); exits with code 1 if any file fails",
    ),
    rpe_distance: Optional[List[float]] = typer.Option(
        None,
        "--rpe-distance",
        help="Distance window in meters for KITTI-style RPE (repeatable)",
    ),
    report: Optional[str] = typer.Option(
        None, "--report",
        help="Write trajectory batch report (.md or .html)",
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump results as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Evaluate all trajectory files in a directory against matched references."""
    if format_json:
        import logging

        logging.getLogger("ca").setLevel(logging.ERROR)

    try:
        results = trajectory_batch_evaluate(
            directory,
            reference_dir,
            recursive=recursive,
            max_time_delta=max_time_delta,
            align_origin=align_origin,
            align_rigid=align_rigid,
            max_ate=max_ate,
            max_rpe=max_rpe,
            max_rpe_rotation_deg=max_rpe_rotation_deg,
            max_drift=max_drift,
            min_coverage=min_coverage,
            rpe_distances_m=rpe_distance,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    if report:
        try:
            save_trajectory_batch_report(
                results,
                reference_dir,
                report,
                max_ate=max_ate,
                max_rpe=max_rpe,
                max_rpe_rotation_deg=max_rpe_rotation_deg,
                max_drift=max_drift,
                min_coverage=min_coverage,
            )
        except ValueError as e:
            _handle_error(e)

    summary = make_trajectory_batch_summary(
        results,
        reference_dir,
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
            if item["quality_gate"] is not None:
                status = "  PASS" if item["quality_gate"]["passed"] else "  FAIL"
            rpe_rotation = item.get("rpe_rotation")
            rpe_rotation_label = (
                "n/a" if rpe_rotation is None else f"{rpe_rotation['rmse']:.4f}deg"
            )
            typer.echo(
                f"  {item['path']}: matched={item['matched_poses']}  "
                f"coverage={item['coverage_ratio']:.1%}  "
                f"ATE={item['ate']['rmse']:.4f}  "
                f"RPE={item['rpe_translation']['rmse']:.4f}  "
                f"RPErot={rpe_rotation_label}  "
                f"Drift={item['drift']['endpoint']:.4f}  "
                f"Align={item['alignment']['mode']}"
                f"{status}"
            )
        typer.echo(f"Mean ATE RMSE: {summary['mean_ate_rmse']:.4f}")
        typer.echo(f"Mean RPE RMSE: {summary['mean_rpe_rmse']:.4f}")
        mean_rpe_rotation = summary.get("mean_rpe_rotation_rmse_deg")
        typer.echo(
            "Mean RPE Rotation RMSE: "
            + (f"{mean_rpe_rotation:.4f} deg" if mean_rpe_rotation is not None else "n/a")
        )
        typer.echo(f"Mean Coverage: {summary['mean_coverage_ratio']:.1%}")
        if gate is not None:
            typer.echo(
                f"Quality Gate: pass={gate['pass_count']} fail={gate['fail_count']}"
            )
        typer.echo(f"Reference Dir: {reference_dir}")
        typer.echo(f"Total files: {len(results)}")
        if report:
            typer.echo(f"Report: {report}")

    if output_json:
        _dump_json(results, output_json)
    if should_fail:
        raise typer.Exit(code=1)
