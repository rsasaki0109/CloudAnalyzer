"""CloudAnalyzer CLI entry point."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import typer

from ca.core import load_check_suite, render_check_scaffold, run_check_suite
from ca.compare import run_compare
from ca.info import get_info
from ca.diff import run_diff
from ca.view import view
from ca.downsample import downsample
from ca.merge import merge
from ca.convert import convert
from ca.crop import crop
from ca.stats import compute_stats
from ca.normals import estimate_normals
from ca.filter import filter_outliers
from ca.sample import random_sample
from ca.align import align
from ca.batch import batch_evaluate, batch_info, trajectory_batch_evaluate
from ca.density_map import density_map
from ca.evaluate import evaluate, plot_f1_curve
from ca.split import split
from ca.pipeline import run_pipeline
from ca.plot import heatmap3d
from ca.report import (
    make_batch_summary,
    make_run_batch_summary,
    make_trajectory_batch_summary,
    save_batch_report,
    save_run_batch_report,
    save_run_report,
    save_trajectory_batch_report,
    save_trajectory_report,
)
from ca.run_evaluate import evaluate_run, evaluate_run_batch
from ca.trajectory import evaluate_trajectory
from ca.web import export_static_bundle as web_export_static_bundle
from ca.web import serve as web_serve
from ca.io import SUPPORTED_EXTENSIONS
from ca.log import setup_logging

app = typer.Typer(
    name="ca",
    help="CloudAnalyzer - AI-friendly CLI tool for point cloud analysis.",
)


def _dump_json(data, path: str) -> None:
    """Write result dict/list to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    typer.echo(f"JSON: {path}")


def _run_evaluate(source: str, target: str, result: dict, plot_path: Optional[str] = None) -> dict:
    """Run evaluate and print summary inline. Returns updated result dict."""
    eval_result = evaluate(source, target)
    typer.echo(f"  Chamfer={eval_result['chamfer_distance']:.4f}  AUC={eval_result['auc']:.4f}")
    best_f1 = max(eval_result["f1_scores"], key=lambda s: s["f1"])
    typer.echo(f"  Best F1={best_f1['f1']:.4f} @ d={best_f1['threshold']:.2f}")
    if plot_path:
        plot_f1_curve(eval_result, plot_path)
        typer.echo(f"  Plot: {plot_path}")
    result["evaluation"] = {
        "chamfer": eval_result["chamfer_distance"],
        "hausdorff": eval_result["hausdorff_distance"],
        "auc": eval_result["auc"],
        "f1_scores": eval_result["f1_scores"],
    }
    return result


def _handle_error(e: Exception) -> None:
    """Print error with hints and exit."""
    msg = str(e)
    typer.echo(f"Error: {msg}", err=True)

    # Provide helpful hints
    if isinstance(e, FileNotFoundError):
        typer.echo("Hint: Check the file path. Supported formats: .pcd, .ply, .las", err=True)
    elif "Unsupported format" in msg:
        typer.echo(f"Hint: Supported formats are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}", err=True)
    elif "Unsupported method" in msg:
        typer.echo("Hint: Supported methods are: icp, gicp", err=True)
    elif "empty" in msg.lower():
        typer.echo("Hint: The file exists but contains no points. Check the file integrity.", err=True)

    raise typer.Exit(code=1)


def _parse_thresholds(thresholds: Optional[str]) -> Optional[list[float]]:
    """Parse comma-separated threshold values."""
    if not thresholds:
        return None

    try:
        return [float(x.strip()) for x in thresholds.split(",")]
    except ValueError:
        typer.echo("Error: --thresholds must be comma-separated numbers", err=True)
        raise typer.Exit(code=1)


def _check_status_label(passed: bool | None) -> str:
    """Render a compact status label for config-driven checks."""
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    return "INFO"


def _print_check_suite_result(result: dict) -> None:
    """Print a concise human summary for `ca check`."""
    if result.get("project"):
        typer.echo(f"Project: {result['project']}")
    typer.echo(f"Config:   {result['config_path']}")
    for item in result["checks"]:
        summary = item["summary"]
        status = _check_status_label(item["passed"])
        if item["kind"] == "artifact":
            typer.echo(
                f"[{status}] {item['id']} ({item['kind']}): "
                f"auc={summary['auc']:.4f}  "
                f"chamfer={summary['chamfer_distance']:.4f}"
            )
        elif item["kind"] == "artifact_batch":
            typer.echo(
                f"[{status}] {item['id']} ({item['kind']}): "
                f"files={summary['total_files']}  "
                f"mean_auc={summary['mean_auc']:.4f}  "
                f"mean_chamfer={summary['mean_chamfer_distance']:.4f}"
            )
        elif item["kind"] == "trajectory":
            typer.echo(
                f"[{status}] {item['id']} ({item['kind']}): "
                f"matched={summary['matched_poses']}  "
                f"coverage={summary['coverage_ratio']:.1%}  "
                f"ate={summary['ate_rmse']:.4f}  "
                f"rpe={summary['rpe_rmse']:.4f}"
            )
        elif item["kind"] == "trajectory_batch":
            typer.echo(
                f"[{status}] {item['id']} ({item['kind']}): "
                f"files={summary['total_files']}  "
                f"mean_ate={summary['mean_ate_rmse']:.4f}  "
                f"mean_rpe={summary['mean_rpe_rmse']:.4f}  "
                f"mean_coverage={summary['mean_coverage_ratio']:.1%}"
            )
        elif item["kind"] == "run":
            typer.echo(
                f"[{status}] {item['id']} ({item['kind']}): "
                f"map_auc={summary['map_auc']:.4f}  "
                f"traj_ate={summary['trajectory_ate_rmse']:.4f}  "
                f"coverage={summary['coverage_ratio']:.1%}"
            )
        else:
            typer.echo(
                f"[{status}] {item['id']} ({item['kind']}): "
                f"runs={summary['total_runs']}  "
                f"mean_map_auc={summary['mean_map_auc']:.4f}  "
                f"mean_traj_ate={summary['mean_traj_ate_rmse']:.4f}  "
                f"mean_coverage={summary['mean_traj_coverage']:.1%}"
            )
        if item.get("report_path"):
            typer.echo(f"  Report: {item['report_path']}")
        if item.get("json_path"):
            typer.echo(f"  JSON:   {item['json_path']}")

    summary = result["summary"]
    typer.echo("")
    typer.echo(
        f"Checks: total={summary['total_checks']}  "
        f"gated={summary['gated_checks']}  "
        f"pass={summary['passed_checks']}  "
        f"fail={summary['failed_checks']}  "
        f"info={summary['unchecked_checks']}"
    )
    triage = summary.get("triage")
    if isinstance(triage, dict) and triage.get("items"):
        typer.echo("")
        typer.echo(
            f"Triage: {triage['strategy']}  failed={triage['failed_count']}"
        )
        for item in triage["items"][:3]:
            failed_dims = ", ".join(item["failed_dimensions"]) or "unknown"
            typer.echo(
                f"  {item['rank']}. {item['check_id']} ({item['kind']}): "
                f"score={item['severity_score']:.4f}  dims={failed_dims}"
            )
            if item.get("report_path"):
                typer.echo(f"     Report: {item['report_path']}")

@app.callback()
def common_options(
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Enable verbose (debug) output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-error output"),
) -> None:
    """CloudAnalyzer - AI-friendly CLI tool for point cloud analysis."""
    setup_logging(verbose=verbose, quiet=quiet)


@app.command("compare")
def compare_cmd(
    source: str = typer.Argument(..., help="Path to source point cloud (pcd/ply/las)"),
    target: str = typer.Argument(..., help="Path to target point cloud (pcd/ply/las)"),
    method: Optional[str] = typer.Option(
        "gicp", "--register",
        help="Registration method: icp, gicp, or 'none' to skip",
    ),
    json_out: Optional[str] = typer.Option(None, "--json", help="Output path for JSON report"),
    report: Optional[str] = typer.Option(None, "--report", help="Output path for Markdown report"),
    snapshot: Optional[str] = typer.Option(None, "--snapshot", help="Output path for snapshot image (png)"),
    threshold: Optional[float] = typer.Option(None, "--threshold", help="Distance threshold; report how many points exceed it"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump full result as JSON"),
) -> None:
    """Compare two point clouds with optional registration."""
    reg_method = method if method and method.lower() != "none" else None
    try:
        result = run_compare(
            source_path=source, target_path=target, method=reg_method,
            json_path=json_out, report_path=report, snapshot_path=snapshot,
            threshold=threshold,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)
    if output_json:
        _dump_json(result, output_json)


@app.command("info")
def info_cmd(
    path: str = typer.Argument(..., help="Path to point cloud file (pcd/ply/las)"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Show basic information about a point cloud file."""
    try:
        info = get_info(path)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    if format_json:
        typer.echo(json.dumps(info, indent=2))
    else:
        typer.echo(f"File:      {info['path']}")
        typer.echo(f"Points:    {info['num_points']}")
        typer.echo(f"Colors:    {info['has_colors']}")
        typer.echo(f"Normals:   {info['has_normals']}")
        typer.echo(f"BBox min:  [{info['bbox_min'][0]:.4f}, {info['bbox_min'][1]:.4f}, {info['bbox_min'][2]:.4f}]")
        typer.echo(f"BBox max:  [{info['bbox_max'][0]:.4f}, {info['bbox_max'][1]:.4f}, {info['bbox_max'][2]:.4f}]")
        typer.echo(f"Extent:    [{info['extent'][0]:.4f}, {info['extent'][1]:.4f}, {info['extent'][2]:.4f}]")
        typer.echo(f"Centroid:  [{info['centroid'][0]:.4f}, {info['centroid'][1]:.4f}, {info['centroid'][2]:.4f}]")
    if output_json:
        _dump_json(info, output_json)


@app.command("diff")
def diff_cmd(
    source: str = typer.Argument(..., help="Path to source point cloud"),
    target: str = typer.Argument(..., help="Path to target point cloud"),
    threshold: Optional[float] = typer.Option(None, "--threshold", help="Distance threshold; report how many points exceed it"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Quick distance stats between two point clouds (no registration)."""
    try:
        result = run_diff(source, target, threshold=threshold)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        stats = result["distance_stats"]
        typer.echo(f"Source: {result['source_points']} pts | Target: {result['target_points']} pts")
        typer.echo(f"Mean:   {stats['mean']:.4f}")
        typer.echo(f"Median: {stats['median']:.4f}")
        typer.echo(f"Max:    {stats['max']:.4f}")
        typer.echo(f"Min:    {stats['min']:.4f}")
        typer.echo(f"Std:    {stats['std']:.4f}")
        if "threshold" in result:
            t = result["threshold"]
            typer.echo(f"Exceed: {t['exceed_count']}/{t['total']} ({t['exceed_ratio']:.1%}) > {t['threshold']}")
    if output_json:
        _dump_json(result, output_json)


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
        s = result["spacing"]
        typer.echo(f"Spacing mean:   {s['mean']:.4f}")
        typer.echo(f"Spacing median: {s['median']:.4f}")
        typer.echo(f"Spacing min:    {s['min']:.4f}")
        typer.echo(f"Spacing max:    {s['max']:.4f}")
        typer.echo(f"Spacing std:    {s['std']:.4f}")
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


@app.command("downsample")
def downsample_cmd(
    path: str = typer.Argument(..., help="Input point cloud file"),
    output: str = typer.Option(..., "--output", "-o", help="Output file path"),
    voxel_size: float = typer.Option(0.05, "--voxel-size", "-v", help="Voxel size"),
    eval_quality: bool = typer.Option(False, "--evaluate", "-e", help="Evaluate quality against original"),
    plot: Optional[str] = typer.Option(None, "--plot", help="F1 curve plot (requires --evaluate)"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Downsample a point cloud using voxel grid filtering."""
    try:
        result = downsample(path, voxel_size, output)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    typer.echo(f"Original:     {result['original_points']} pts")
    typer.echo(f"Downsampled:  {result['downsampled_points']} pts")
    typer.echo(f"Reduction:    {result['reduction_ratio']:.1%}")
    typer.echo(f"Saved:        {result['output']}")
    if eval_quality or plot:
        result = _run_evaluate(output, path, result, plot)
    if output_json:
        _dump_json(result, output_json)


@app.command("merge")
def merge_cmd(
    paths: List[str] = typer.Argument(..., help="Input point cloud files to merge"),
    output: str = typer.Option(..., "--output", "-o", help="Output file path"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Merge multiple point clouds into one file."""
    try:
        result = merge(paths, output)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    for inp in result["inputs"]:
        typer.echo(f"  {inp['path']}: {inp['points']} pts")
    typer.echo(f"Total:  {result['total_points']} pts")
    typer.echo(f"Saved:  {result['output']}")
    if output_json:
        _dump_json(result, output_json)


@app.command("convert")
def convert_cmd(
    input_path: str = typer.Argument(..., help="Input point cloud file"),
    output_path: str = typer.Argument(..., help="Output file path (format from extension)"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Convert point cloud between formats (pcd/ply/las)."""
    try:
        result = convert(input_path, output_path)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    typer.echo(f"{result['input_format']} -> {result['output_format']}")
    typer.echo(f"Points: {result['num_points']}")
    typer.echo(f"Saved:  {result['output']}")
    if output_json:
        _dump_json(result, output_json)


@app.command("crop")
def crop_cmd(
    input_path: str = typer.Argument(..., help="Input point cloud file"),
    output: str = typer.Option(..., "--output", "-o", help="Output file path"),
    x_min: float = typer.Option(..., "--x-min", help="Bounding box X min"),
    y_min: float = typer.Option(..., "--y-min", help="Bounding box Y min"),
    z_min: float = typer.Option(..., "--z-min", help="Bounding box Z min"),
    x_max: float = typer.Option(..., "--x-max", help="Bounding box X max"),
    y_max: float = typer.Option(..., "--y-max", help="Bounding box Y max"),
    z_max: float = typer.Option(..., "--z-max", help="Bounding box Z max"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Crop point cloud to an axis-aligned bounding box."""
    try:
        result = crop(input_path, output, x_min, y_min, z_min, x_max, y_max, z_max)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    typer.echo(f"Original: {result['original_points']} pts")
    typer.echo(f"Cropped:  {result['cropped_points']} pts")
    typer.echo(f"Saved:    {result['output']}")
    if output_json:
        _dump_json(result, output_json)


@app.command("normals")
def normals_cmd(
    input_path: str = typer.Argument(..., help="Input point cloud file"),
    output: str = typer.Option(..., "--output", "-o", help="Output file path"),
    radius: float = typer.Option(0.5, "--radius", "-r", help="Search radius"),
    max_nn: int = typer.Option(30, "--max-nn", help="Max neighbors for estimation"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Estimate normals and save to file."""
    try:
        result = estimate_normals(input_path, output, radius=radius, max_nn=max_nn)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    typer.echo(f"Points:  {result['num_points']}")
    typer.echo(f"Radius:  {result['radius']}")
    typer.echo(f"Max NN:  {result['max_nn']}")
    typer.echo(f"Saved:   {result['output']}")
    if output_json:
        _dump_json(result, output_json)


@app.command("filter")
def filter_cmd(
    input_path: str = typer.Argument(..., help="Input point cloud file"),
    output: str = typer.Option(..., "--output", "-o", help="Output file path"),
    nb_neighbors: int = typer.Option(20, "--neighbors", "-n", help="Number of neighbors"),
    std_ratio: float = typer.Option(2.0, "--std-ratio", "-s", help="Std deviation ratio threshold"),
    eval_quality: bool = typer.Option(False, "--evaluate", "-e", help="Evaluate quality against original"),
    plot: Optional[str] = typer.Option(None, "--plot", help="F1 curve plot (requires --evaluate)"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Remove statistical outliers from a point cloud."""
    try:
        result = filter_outliers(input_path, output, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    typer.echo(f"Original: {result['original_points']} pts")
    typer.echo(f"Filtered: {result['filtered_points']} pts")
    typer.echo(f"Removed:  {result['removed_points']} pts")
    typer.echo(f"Saved:    {result['output']}")
    if eval_quality or plot:
        result = _run_evaluate(output, input_path, result, plot)
    if output_json:
        _dump_json(result, output_json)


@app.command("sample")
def sample_cmd(
    input_path: str = typer.Argument(..., help="Input point cloud file"),
    output: str = typer.Option(..., "--output", "-o", help="Output file path"),
    num_points: int = typer.Option(..., "--num", "-n", help="Number of points to keep"),
    eval_quality: bool = typer.Option(False, "--evaluate", "-e", help="Evaluate quality against original"),
    plot: Optional[str] = typer.Option(None, "--plot", help="F1 curve plot (requires --evaluate)"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Randomly sample a fixed number of points."""
    try:
        result = random_sample(input_path, output, num_points)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    typer.echo(f"Original: {result['original_points']} pts")
    typer.echo(f"Sampled:  {result['sampled_points']} pts")
    typer.echo(f"Saved:    {result['output']}")
    if eval_quality or plot:
        result = _run_evaluate(output, input_path, result, plot)
    if output_json:
        _dump_json(result, output_json)


@app.command("align")
def align_cmd(
    paths: List[str] = typer.Argument(..., help="Point cloud files to align (first is reference)"),
    output: str = typer.Option(..., "--output", "-o", help="Output file path"),
    method: str = typer.Option("gicp", "--method", "-m", help="Registration method: icp or gicp"),
    max_dist: float = typer.Option(1.0, "--max-dist", help="Max correspondence distance"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Align multiple point clouds sequentially and merge."""
    try:
        result = align(paths, output, method=method, max_correspondence_distance=max_dist)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    for step in result["steps"]:
        typer.echo(f"  Step {step['step']}: {step['path']} (fitness={step['fitness']:.4f}, rmse={step['rmse']:.4f})")
    typer.echo(f"Total:  {result['total_points']} pts")
    typer.echo(f"Saved:  {result['output']}")
    if output_json:
        _dump_json(result, output_json)


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
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Evaluate point cloud similarity (F1, Chamfer, Hausdorff, AUC)."""
    thresh_list = _parse_thresholds(thresholds)

    try:
        result = evaluate(source, target, thresholds=thresh_list)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

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
    estimated: str = typer.Argument(..., help="Estimated trajectory (.csv/.tum/.txt)"),
    reference: str = typer.Argument(..., help="Reference trajectory (.csv/.tum/.txt)"),
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
    max_drift: Optional[float] = typer.Option(
        None, "--max-drift",
        help="Maximum endpoint drift allowed; exits with code 1 if exceeded",
    ),
    min_coverage: Optional[float] = typer.Option(
        None, "--min-coverage",
        help="Minimum matched-pose coverage ratio required (0-1); exits with code 1 if not met",
    ),
    report: Optional[str] = typer.Option(
        None, "--report",
        help="Write trajectory report (.md or .html)",
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Evaluate a trajectory against a reference trajectory."""
    try:
        result = evaluate_trajectory(
            estimated,
            reference,
            max_time_delta=max_time_delta,
            align_origin=align_origin,
            align_rigid=align_rigid,
            max_ate=max_ate,
            max_rpe=max_rpe,
            max_drift=max_drift,
            min_coverage=min_coverage,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

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
    max_drift: Optional[float] = typer.Option(
        None, "--max-drift",
        help="Maximum endpoint drift allowed; exits with code 1 if any file fails",
    ),
    min_coverage: Optional[float] = typer.Option(
        None, "--min-coverage",
        help="Minimum matched-pose coverage ratio required (0-1); exits with code 1 if any file fails",
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
            max_drift=max_drift,
            min_coverage=min_coverage,
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
            typer.echo(
                f"  {item['path']}: matched={item['matched_poses']}  "
                f"coverage={item['coverage_ratio']:.1%}  "
                f"ATE={item['ate']['rmse']:.4f}  "
                f"RPE={item['rpe_translation']['rmse']:.4f}  "
                f"Drift={item['drift']['endpoint']:.4f}  "
                f"Align={item['alignment']['mode']}"
                f"{status}"
            )
        typer.echo(f"Mean ATE RMSE: {summary['mean_ate_rmse']:.4f}")
        typer.echo(f"Mean RPE RMSE: {summary['mean_rpe_rmse']:.4f}")
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
            max_drift=max_drift,
            min_coverage=min_coverage,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

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
        typer.echo(
            f"Trajectory: matched={matching['matched_poses']} ({matching['coverage_ratio']:.1%})  "
            f"ATE={trajectory_result['ate']['rmse']:.4f}  "
            f"RPE={trajectory_result['rpe_translation']['rmse']:.4f}  "
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


@app.command("check")
def check_cmd(
    config_path: str = typer.Argument(
        "cloudanalyzer.yaml",
        help="Path to cloudanalyzer.yaml (or JSON) config file",
    ),
    output_json: Optional[str] = typer.Option(
        None,
        "--output-json",
        help="Dump aggregated check summary as JSON",
    ),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Run config-driven artifact / trajectory / run QA checks."""
    if format_json:
        import logging

        logging.getLogger("ca").setLevel(logging.ERROR)

    try:
        suite = load_check_suite(config_path)
        if output_json is not None:
            suite = replace(suite, summary_output_json=str(Path(output_json).resolve()))
        result = run_check_suite(suite)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)
        return

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        _print_check_suite_result(result)
        if suite.summary_output_json:
            typer.echo(f"Summary JSON: {suite.summary_output_json}")

    if result["summary"]["failed_checks"] > 0:
        raise typer.Exit(code=1)


@app.command("init-check")
def init_check_cmd(
    output_path: str = typer.Argument(
        "cloudanalyzer.yaml",
        help="Path to write the starter cloudanalyzer.yaml",
    ),
    profile: str = typer.Option(
        "integrated",
        "--profile",
        help="Starter template profile: mapping, localization, perception, integrated",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite an existing config file",
    ),
) -> None:
    """Write a starter cloudanalyzer.yaml for a common QA workflow."""
    destination = Path(output_path).resolve()
    if destination.exists() and not force:
        _handle_error(
            ValueError(f"Refusing to overwrite existing file: {destination} (use --force)")
        )
        return

    try:
        template = render_check_scaffold(profile=profile).yaml_text
    except ValueError as e:
        _handle_error(e)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template, encoding="utf-8")
    typer.echo(f"Created: {destination}")
    typer.echo(f"Profile: {profile.strip().lower()}")
    typer.echo(f"Next:    ca check {destination}")


@app.command("split")
def split_cmd(
    input_path: str = typer.Argument(..., help="Input point cloud file"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Output directory for tiles"),
    grid_size: float = typer.Option(..., "--grid-size", "-g", help="Grid cell size"),
    axis: str = typer.Option("xy", "--axis", "-a", help="Split axes: xy, xz, or yz"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Split a point cloud into grid tiles."""
    try:
        result = split(input_path, output_dir, grid_size, axis)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    typer.echo(f"Original: {result['total_points']} pts")
    typer.echo(f"Tiles:    {result['num_tiles']}")
    typer.echo(f"Grid:     {result['grid_size']}m ({result['axis']})")
    typer.echo(f"Dir:      {result['output_dir']}")
    if output_json:
        _dump_json(result, output_json)


@app.command("pipeline")
def pipeline_cmd(
    input_path: str = typer.Argument(..., help="Input point cloud to process"),
    reference: str = typer.Argument(..., help="Reference point cloud for evaluation"),
    output: str = typer.Option(..., "--output", "-o", help="Output file path"),
    voxel_size: float = typer.Option(0.1, "--voxel-size", "-v", help="Voxel size for downsampling"),
    nb_neighbors: int = typer.Option(20, "--neighbors", "-n", help="Neighbors for outlier filter"),
    std_ratio: float = typer.Option(2.0, "--std-ratio", "-s", help="Std ratio for outlier filter"),
    thresholds: Optional[str] = typer.Option(
        None, "--thresholds", "-t",
        help="Comma-separated distance thresholds for evaluation",
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Run filter -> downsample -> evaluate pipeline."""
    thresh_list = _parse_thresholds(thresholds)

    try:
        result = run_pipeline(
            input_path, reference, output,
            voxel_size=voxel_size, nb_neighbors=nb_neighbors,
            std_ratio=std_ratio, thresholds=thresh_list,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    typer.echo(f"Filter:     {result['filter']['original']} -> {result['filter']['filtered']} pts (removed {result['filter']['removed']})")
    typer.echo(f"Downsample: {result['downsample']['input']} -> {result['downsample']['output']} pts ({result['downsample']['reduction']:.1%})")
    typer.echo(f"Chamfer:    {result['evaluation']['chamfer']:.4f}")
    typer.echo(f"AUC (F1):   {result['evaluation']['auc']:.4f}")
    typer.echo(f"Saved:      {result['output']}")
    if output_json:
        _dump_json(result, output_json)


@app.command("heatmap3d")
def heatmap3d_cmd(
    source: str = typer.Argument(..., help="Source point cloud"),
    target: str = typer.Argument(..., help="Target (reference) point cloud"),
    output: str = typer.Option(..., "--output", "-o", help="Output snapshot image (png)"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Render source colored by distance to target as 3D snapshot."""
    try:
        result = heatmap3d(source, target, output)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    typer.echo(f"Points:        {result['num_points']}")
    typer.echo(f"Mean distance: {result['mean_distance']:.4f}")
    typer.echo(f"Max distance:  {result['max_distance']:.4f}")
    typer.echo(f"Saved:         {result['output']}")
    if output_json:
        _dump_json(result, output_json)


@app.command("web")
def web_cmd(
    paths: List[str] = typer.Argument(..., help="Point cloud file(s) to view"),
    port: int = typer.Option(8080, "--port", "-p", help="HTTP port"),
    max_points: int = typer.Option(2_000_000, "--max-points", help="Max points for display"),
    heatmap: bool = typer.Option(
        False, "--heatmap",
        help="With 2 files, color the first by distance to the second",
    ),
    trajectory: Optional[str] = typer.Option(
        None, "--trajectory",
        help="Estimated trajectory to overlay on top of the point cloud view",
    ),
    trajectory_reference: Optional[str] = typer.Option(
        None, "--trajectory-reference",
        help="Reference trajectory to overlay alongside --trajectory",
    ),
    trajectory_max_time_delta: float = typer.Option(
        0.05, "--trajectory-max-time-delta",
        help="Max timestamp gap allowed when matching --trajectory to --trajectory-reference",
    ),
    trajectory_align_origin: bool = typer.Option(
        False, "--trajectory-align-origin",
        help="Translate --trajectory so its first matched pose aligns to --trajectory-reference",
    ),
    trajectory_align_rigid: bool = typer.Option(
        False, "--trajectory-align-rigid",
        help="Fit a rigid transform from --trajectory to --trajectory-reference before display",
    ),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't auto-open browser"),
) -> None:
    """Open point cloud in web browser (Three.js viewer)."""
    try:
        web_serve(
            paths,
            port=port,
            max_points=max_points,
            open_browser=not no_browser,
            heatmap=heatmap,
            trajectory_path=trajectory,
            trajectory_reference_path=trajectory_reference,
            trajectory_max_time_delta=trajectory_max_time_delta,
            trajectory_align_origin=trajectory_align_origin,
            trajectory_align_rigid=trajectory_align_rigid,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)


@app.command("web-export")
def web_export_cmd(
    paths: List[str] = typer.Argument(..., help="Point cloud file(s) to export"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Static bundle output directory"),
    max_points: int = typer.Option(2_000_000, "--max-points", help="Max points for display"),
    heatmap: bool = typer.Option(
        False, "--heatmap",
        help="With 2 files, color the first by distance to the second",
    ),
    trajectory: Optional[str] = typer.Option(
        None, "--trajectory",
        help="Estimated trajectory to overlay on top of the point cloud view",
    ),
    trajectory_reference: Optional[str] = typer.Option(
        None, "--trajectory-reference",
        help="Reference trajectory to overlay alongside --trajectory",
    ),
    trajectory_max_time_delta: float = typer.Option(
        0.05, "--trajectory-max-time-delta",
        help="Max timestamp gap allowed when matching --trajectory to --trajectory-reference",
    ),
    trajectory_align_origin: bool = typer.Option(
        False, "--trajectory-align-origin",
        help="Translate --trajectory so its first matched pose aligns to --trajectory-reference",
    ),
    trajectory_align_rigid: bool = typer.Option(
        False, "--trajectory-align-rigid",
        help="Fit a rigid transform from --trajectory to --trajectory-reference before display",
    ),
) -> None:
    """Export a static web viewer bundle for GitHub Pages or any static host."""
    try:
        result = web_export_static_bundle(
            paths,
            output_dir=output_dir,
            max_points=max_points,
            heatmap=heatmap,
            trajectory_path=trajectory,
            trajectory_reference_path=trajectory_reference,
            trajectory_max_time_delta=trajectory_max_time_delta,
            trajectory_align_origin=trajectory_align_origin,
            trajectory_align_rigid=trajectory_align_rigid,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)
        return

    typer.echo(f"Exported:     {result['output_dir']}")
    typer.echo(f"Viewer mode:  {result['viewer_mode']}")
    typer.echo(f"Data:         {result['data_json']}")
    typer.echo(f"Chunks:       {result['chunk_count']}")
    typer.echo(f"Display pts:  {result['display_points']}")


@app.command("version")
def version_cmd() -> None:
    """Show CloudAnalyzer version."""
    from ca import __version__
    typer.echo(f"CloudAnalyzer v{__version__}")


def main():
    app()


if __name__ == "__main__":
    main()
