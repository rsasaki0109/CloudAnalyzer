"""CloudAnalyzer CLI entry point."""

import json
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Any, List, Optional

import typer

from ca.core import (
    load_check_suite,
    render_check_scaffold,
    run_check_suite,
    summarize_baseline_evolution,
)
from ca.baseline_history import discover_history, list_baselines, rotate_history, save_baseline
from ca.detection import evaluate_detection
from ca.ground_evaluate import evaluate_ground_segmentation
from ca.compare import run_compare
from ca.scan_match_debug import run_scan_match_debug
from ca.slam_debug import analyze_slam_run, write_slam_debug_markdown
from ca.core.slam_run import (
    SlamRunRequest,
    discover_frame_paths,
    get_driver,
    write_map_ply,
    write_tum_trajectory,
)
from ca.info import get_info
from ca.diff import run_diff
from ca.view import view
from ca.downsample import downsample
from ca.merge import merge
from ca.mme import compute_mme
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
    save_detection_report,
    make_run_batch_summary,
    make_trajectory_batch_summary,
    save_batch_report,
    save_ground_report,
    save_run_batch_report,
    save_run_report,
    save_tracking_report,
    save_trajectory_batch_report,
    save_trajectory_report,
)
from ca.benchmark import (
    BenchmarkSuite,
    GATE_KEYS,
    evaluate_benchmark_run,
    load_benchmark_suite,
    materialize_suite,
)
from ca.bundle import (
    diff_bundles,
    pack_bundle,
    render_diff_markdown,
    show_bundle,
    unpack_bundle,
)
from ca.history import (
    build_history_series,
    discover_bundles,
    render_history_json,
    render_history_markdown,
)
from ca.geometry import (
    DEFAULT_MESH_SAMPLES,
    DEFAULT_SPLAT_SAMPLES,
    MESH_SAMPLE_METHODS,
    REPRESENTATIONS,
    SPLAT_METHODS,
    evaluate_geometry,
)
from ca.pr_comment import build_pr_comment
from ca.run_evaluate import evaluate_run, evaluate_run_batch
from ca.tracking import evaluate_tracking
from ca.trajectory import evaluate_trajectory
from ca.posegraph import discover_session_paths, validate_posegraph_session
from ca.loop_closure_report import LoopClosureGate, build_loop_closure_report
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


def _load_json_mapping(path: str) -> dict:
    """Load a JSON object from disk."""

    resolved = Path(path).resolve()
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {resolved}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level JSON object in: {resolved}")
    return data


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
        typer.echo(
            "Hint: Check the file path. Supported formats: .pcd, .ply, .las, .laz, .csv",
            err=True,
        )
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


def _parse_matrix16(matrix: Optional[str]) -> Optional[list[float]]:
    """Parse a 4x4 matrix from 16 comma-separated floats (row-major)."""
    if not matrix:
        return None
    try:
        values = [float(x.strip()) for x in matrix.split(",")]
    except ValueError:
        typer.echo("Error: --initial-matrix must be 16 comma-separated numbers", err=True)
        raise typer.Exit(code=1)
    if len(values) != 16:
        typer.echo("Error: --initial-matrix must contain exactly 16 numbers", err=True)
        raise typer.Exit(code=1)
    return values


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
        elif item["kind"] == "detection":
            typer.echo(
                f"[{status}] {item['id']} ({item['kind']}): "
                f"mAP={summary['map']:.4f}  "
                f"precision={summary['precision']:.4f}  "
                f"recall={summary['recall']:.4f}  "
                f"f1={summary['f1']:.4f}"
            )
        elif item["kind"] == "tracking":
            typer.echo(
                f"[{status}] {item['id']} ({item['kind']}): "
                f"mota={summary['mota']:.4f}  "
                f"recall={summary['recall']:.4f}  "
                f"id_switches={summary['id_switches']}  "
                f"mean_iou={summary['mean_iou']:.4f}"
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
        elif item["kind"] == "loop_closure":
            ate = summary.get("after_trajectory_ate_rmse")
            ate_text = f"  after_ate={ate:.4f}" if isinstance(ate, (int, float)) else ""
            typer.echo(
                f"[{status}] {item['id']} ({item['kind']}): "
                f"auc_gain={summary['map_auc_gain']:.4f}  "
                f"after_chamfer={summary['after_chamfer_distance']:.4f}"
                f"{ate_text}"
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


def _print_baseline_evolution_result(result: dict) -> None:
    """Print a concise human summary for baseline promotion decisions."""

    typer.echo(f"Candidate: {result['candidate_label']}")
    typer.echo(f"History:   {len(result['history_labels'])} summaries")
    typer.echo(
        f"Decision:  {result['decision']} ({result['strategy']}, confidence={result['confidence']:.2f})"
    )
    typer.echo(f"Reasons:   {', '.join(result['reasons'])}")
    if result["history_labels"]:
        typer.echo(f"Labels:    {', '.join(result['history_labels'])}")
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        if "margin_gain" in metadata:
            typer.echo(f"Margin:    gain={metadata['margin_gain']}")
        if "consecutive_passes" in metadata:
            typer.echo(f"Window:    consecutive_passes={metadata['consecutive_passes']}")

@app.callback()
def common_options(
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Enable verbose (debug) output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-error output"),
) -> None:
    """CloudAnalyzer - AI-friendly CLI tool for point cloud analysis."""
    setup_logging(verbose=verbose, quiet=quiet)


@app.command("compare")
def compare_cmd(
    source: str = typer.Argument(..., help="Path to source point cloud (pcd/ply/las/laz/csv)"),
    target: str = typer.Argument(..., help="Path to target point cloud (pcd/ply/las/laz/csv)"),
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


@app.command("scan-match-debug")
def scan_match_debug_cmd(
    scan: str = typer.Argument(..., help="Path to source scan point cloud (pcd/ply/las/laz/csv)"),
    map_cloud: str = typer.Argument(..., help="Path to map/reference point cloud (pcd/ply/las/laz/csv)"),
    method: str = typer.Option("gicp", "--method", help="Registration method: icp or gicp"),
    max_correspondence_distance: float = typer.Option(
        1.0,
        "--max-correspondence-distance",
        help="Maximum correspondence distance for ICP/GICP.",
    ),
    initial_matrix: Optional[str] = typer.Option(
        None,
        "--initial-matrix",
        help="Initial 4x4 scan-to-map transform as 16 comma-separated floats (row-major).",
    ),
    scan_voxel_size: Optional[float] = typer.Option(
        None,
        "--scan-voxel-size",
        help="Optional voxel size used to downsample the scan before matching.",
    ),
    map_voxel_size: Optional[float] = typer.Option(
        None,
        "--map-voxel-size",
        help="Optional voxel size used to downsample the map before matching.",
    ),
    crop_margin: Optional[float] = typer.Option(
        None,
        "--crop-margin",
        help="Crop map to the initial scan bounding box plus this margin.",
    ),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        help="Distance threshold for before/after exceed counts.",
    ),
    artifact_dir: Optional[str] = typer.Option(
        None,
        "--artifact-dir",
        help="Optional directory for colored before/after scan PLY artifacts.",
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump full result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Debug one scan-to-map ICP/GICP matching attempt."""
    matrix16 = _parse_matrix16(initial_matrix)
    try:
        result = run_scan_match_debug(
            scan_path=scan,
            map_path=map_cloud,
            method=method,
            max_correspondence_distance=max_correspondence_distance,
            initial_transform=matrix16,
            scan_voxel_size=scan_voxel_size,
            map_voxel_size=map_voxel_size,
            crop_margin=crop_margin,
            threshold=threshold,
            artifact_dir=artifact_dir,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        prep = result["preprocess"]
        reg = result["registration"]
        before = result["distance_before"]["stats"]
        after = result["distance_after"]["stats"]
        imp = result["improvement"]
        typer.echo(f"Scan:   {scan}")
        typer.echo(f"Map:    {map_cloud}")
        typer.echo(
            "Points: "
            f"scan {prep['scan_points_used']}/{prep['scan_points_raw']}  "
            f"map {prep['map_points_used']}/{prep['map_points_raw']}"
        )
        typer.echo(
            f"Reg:    {result['method']}  fitness={reg['fitness']:.4f}  "
            f"rmse={reg['inlier_rmse']:.4f}"
        )
        typer.echo(
            f"Before: mean={before['mean']:.4f}  median={before['median']:.4f}  "
            f"max={before['max']:.4f}"
        )
        typer.echo(
            f"After:  mean={after['mean']:.4f}  median={after['median']:.4f}  "
            f"max={after['max']:.4f}"
        )
        typer.echo(
            f"Delta:  mean={imp['mean']:.4f}  median={imp['median']:.4f}  "
            f"max={imp['max']:.4f}"
        )
        if result["artifacts"]:
            typer.echo("Artifacts:")
            for path in result["artifacts"].values():
                typer.echo(f"  {path}")

    if output_json:
        _dump_json(result, output_json)


@app.command("slam-debug")
def slam_debug_cmd(
    metrics_csv: str = typer.Argument(..., help="SLAM run metrics CSV, e.g. glim_mapping metrics.csv"),
    scans_manifest_csv: Optional[str] = typer.Option(
        None,
        "--scans-manifest-csv",
        help="Optional scan manifest CSV with scan_id,timestamp_sec,points_csv.",
    ),
    trajectory_csv: Optional[str] = typer.Option(
        None,
        "--trajectory-csv",
        help="Optional estimated trajectory CSV for web inspection commands.",
    ),
    map_cloud: Optional[str] = typer.Option(
        None,
        "--map",
        help="Optional map/reference cloud used to generate scan-match-debug commands.",
    ),
    top_k: int = typer.Option(10, "--top-k", help="Number of suspicious frames to report."),
    sort_by: str = typer.Option(
        "auto",
        "--sort-by",
        help="Ranking key: auto, rmse, cost, rejection, prediction-delta, initial-delta, failure.",
    ),
    artifact_dir: Optional[str] = typer.Option(
        None,
        "--artifact-dir",
        help="Optional base directory for scan-match-debug artifacts and web-export command.",
    ),
    run_scan_match_debug: bool = typer.Option(
        False,
        "--run-scan-match-debug",
        help="Run scan-match-debug automatically for selected suspicious frames.",
    ),
    scan_match_method: str = typer.Option(
        "gicp",
        "--scan-match-method",
        help="Registration method for automatic scan-match-debug: icp or gicp.",
    ),
    scan_match_max_correspondence_distance: float = typer.Option(
        1.0,
        "--scan-match-max-correspondence-distance",
        help="Maximum correspondence distance for automatic scan-match-debug.",
    ),
    scan_match_scan_voxel_size: Optional[float] = typer.Option(
        None,
        "--scan-match-scan-voxel-size",
        help="Optional scan voxel size for automatic scan-match-debug.",
    ),
    scan_match_map_voxel_size: Optional[float] = typer.Option(
        None,
        "--scan-match-map-voxel-size",
        help="Optional map voxel size for automatic scan-match-debug.",
    ),
    scan_match_crop_margin: Optional[float] = typer.Option(
        None,
        "--scan-match-crop-margin",
        help="Optional map crop margin for automatic scan-match-debug.",
    ),
    scan_match_threshold: Optional[float] = typer.Option(
        None,
        "--scan-match-threshold",
        help="Optional NN distance threshold for automatic scan-match-debug.",
    ),
    output_markdown: Optional[str] = typer.Option(
        None,
        "--output-markdown",
        help="Write a Markdown SLAM debug report.",
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump full result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Rank suspicious SLAM frames and print CloudAnalyzer drill-down commands."""

    try:
        result = analyze_slam_run(
            metrics_csv=metrics_csv,
            scans_manifest_csv=scans_manifest_csv,
            trajectory_csv=trajectory_csv,
            map_path=map_cloud,
            top_k=top_k,
            sort_by=sort_by,
            artifact_dir=artifact_dir,
            run_scan_match_debug_frames=run_scan_match_debug,
            scan_match_method=scan_match_method,
            scan_match_max_correspondence_distance=scan_match_max_correspondence_distance,
            scan_match_scan_voxel_size=scan_match_scan_voxel_size,
            scan_match_map_voxel_size=scan_match_map_voxel_size,
            scan_match_crop_margin=scan_match_crop_margin,
            scan_match_threshold=scan_match_threshold,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo(f"Metrics: {result['metrics_csv']}")
        typer.echo(f"Frames:  {result['total_frames']}  selected={len(result['selected_frames'])}")
        typer.echo(f"Sort:    {result['sort_by']}")
        for frame in result["selected_frames"]:
            typer.echo(
                f"#{frame['rank']:02d} scan={frame['scan_id']} "
                f"t={frame['timestamp_sec']} score={frame['score']:.3f}"
            )
            if frame["reasons"]:
                typer.echo(f"    reasons: {', '.join(frame['reasons'])}")
            if frame.get("diagnosis"):
                diagnosis = frame["diagnosis"]
                typer.echo(
                    "    diagnosis: "
                    f"{diagnosis['label']} ({diagnosis['confidence']})"
                )
                typer.echo(f"    action:  {diagnosis['suggested_action']}")
            if frame["scan_path"]:
                typer.echo(f"    scan:    {frame['scan_path']}")
            if frame["scan_match_debug_command"]:
                typer.echo(f"    debug:   {frame['scan_match_debug_command']}")
            if frame["scan_match_debug_result"]:
                registration = frame["scan_match_debug_result"]["registration"]
                improvement = frame["scan_match_debug_result"]["improvement"]
                typer.echo(
                    "    ca:      "
                    f"fitness={registration['fitness']:.4f} "
                    f"inlier_rmse={registration['inlier_rmse']:.4f} "
                    f"mean_delta={improvement['mean']:.4f}"
                )
            if frame["scan_match_debug_error"]:
                typer.echo(f"    ca_err:  {frame['scan_match_debug_error']}")
        commands = result.get("commands", {})
        if commands.get("web"):
            typer.echo(f"Web:     {commands['web']}")
        if commands.get("web_export"):
            typer.echo(f"Export:  {commands['web_export']}")

    if output_json:
        _dump_json(result, output_json)
    if output_markdown:
        write_slam_debug_markdown(result, output_markdown)
        typer.echo(f"Markdown: {output_markdown}")


@app.command("info")
def info_cmd(
    path: str = typer.Argument(..., help="Path to point cloud file (pcd/ply/las/laz/csv)"),
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
        typer.echo(
            "Robust extent p01-p99: "
            f"[{info['robust_extent'][0]:.4f}, {info['robust_extent'][1]:.4f}, "
            f"{info['robust_extent'][2]:.4f}]"
        )
        typer.echo(
            "Outside robust bbox: "
            f"{info['outside_robust_bbox_count']} ({info['outside_robust_bbox_ratio']:.2%})"
        )
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


@app.command("map-evaluate")
def map_evaluate_cmd(
    estimated: str = typer.Argument(..., help="Path to estimated map point cloud (pcd/ply/las/laz)"),
    reference: str = typer.Argument(..., help="Path to reference/GT map point cloud (pcd/ply/las/laz)"),
    thresholds: Optional[str] = typer.Option(
        None,
        "--thresholds",
        help="Comma-separated distance thresholds in meters (MapEval-style accuracy levels).",
    ),
    align_mode: str = typer.Option(
        "none",
        "--align-mode",
        help="Alignment mode: none | initial (apply --initial-matrix to estimated points).",
    ),
    initial_matrix: Optional[str] = typer.Option(
        None,
        "--initial-matrix",
        help="Initial 4x4 transform as 16 comma-separated floats (row-major).",
    ),
    artifact_dir: Optional[str] = typer.Option(
        None,
        "--artifact-dir",
        help="Optional output dir for visualization artifacts (colored PLY error maps).",
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump full result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Experimental MapEval-inspired map-to-map evaluation (GT-based)."""
    try:
        import numpy as np

        from ca.io import load_point_cloud
        from ca.core.map_evaluate import (
            MapEvaluateRequest,
            NNThresholdMapEvaluateStrategy,
        )
    except Exception as e:
        _handle_error(e)

    t_list = _parse_thresholds(thresholds)
    matrix16 = _parse_matrix16(initial_matrix)
    init_4x4 = None
    if matrix16 is not None:
        init_4x4 = np.array(matrix16, dtype=np.float64).reshape(4, 4)

    try:
        est_pcd = load_point_cloud(estimated)
        ref_pcd = load_point_cloud(reference)
        req = MapEvaluateRequest(
            estimated_points=np.asarray(est_pcd.points),
            reference_points=np.asarray(ref_pcd.points),
            thresholds_m=tuple(t_list) if t_list is not None else (0.2, 0.1, 0.08, 0.05, 0.01),
            align_mode=align_mode,
            initial_transform_4x4=init_4x4,
            artifact_dir=artifact_dir,
        )
        result = NNThresholdMapEvaluateStrategy().evaluate(req)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    payload = {
        "estimated": estimated,
        "reference": reference,
        "strategy": result.strategy,
        "design": result.design,
        "metrics": result.metrics,
        "artifacts": result.artifacts,
    }

    if format_json:
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        t0 = result.artifacts.get("thresholds_m", [0.2])[0]
        typer.echo(f"Estimated:  {estimated}")
        typer.echo(f"Reference:  {reference}")
        typer.echo(f"Align:      {result.artifacts.get('align_mode')}")
        typer.echo(f"Chamfer:    {result.metrics.get('chamfer_m'):.6f} m")
        typer.echo(f"F-score:    {result.metrics.get(f'fscore@{t0:.3f}m'):.6f} @ {t0:.3f} m")
        typer.echo(f"Accuracy:   {result.metrics.get(f'accuracy@{t0:.3f}m'):.6f} @ {t0:.3f} m")
        typer.echo(f"Complete:   {result.metrics.get(f'completeness@{t0:.3f}m'):.6f} @ {t0:.3f} m")
        if "estimated_error_raw_ply" in result.artifacts:
            typer.echo(f"Artifacts:  {result.artifacts['estimated_error_raw_ply']}")

    if output_json:
        _dump_json(payload, output_json)


@app.command("image-evaluate")
def image_evaluate_cmd(
    rendered_dir: str = typer.Argument(
        ...,
        help="Directory of rendered (candidate) images.",
    ),
    reference_dir: str = typer.Argument(
        ...,
        help="Directory of reference (ground-truth) images. Pairs match by filename.",
    ),
    metrics: str = typer.Option(
        "psnr,ssim",
        "--metrics",
        help="Comma-separated metrics to compute: psnr, ssim.",
    ),
    extensions: str = typer.Option(
        ".png,.jpg,.jpeg",
        "--extensions",
        help="Comma-separated image extensions to discover under <rendered_dir>.",
    ),
    ssim_window: int = typer.Option(
        11,
        "--ssim-window",
        help="Gaussian-window side length used in SSIM (Wang & Bovik 2004 use 11).",
    ),
    ssim_sigma: float = typer.Option(
        1.5, "--ssim-sigma", help="Gaussian-window sigma used in SSIM."
    ),
    max_pairs: Optional[int] = typer.Option(
        None,
        "--max-pairs",
        help="Cap on number of pairs evaluated. Useful for smoke-testing large render sets.",
    ),
    output_json: Optional[str] = typer.Option(
        None, "--output-json", help="Write the full result (per-pair + summary) as JSON."
    ),
    format_json: bool = typer.Option(
        False, "--format-json", help="Print the result as JSON to stdout."
    ),
) -> None:
    """Score rendered images against a reference set on PSNR / SSIM.

    The first photometric eval surface in CloudAnalyzer. Pairs images by
    filename across the two directories and emits per-pair metrics plus
    a per-metric summary (mean / median / min / max). Useful today for
    comparing two image dirs; future ``ca rendered-evaluate`` will
    render a 3DGS PLY into images at given camera poses and chain into
    this same scoring function.

    Example::

        ca image-evaluate renders/seq00 references/seq00 \\
            --metrics psnr,ssim --format-json
    """

    try:
        from ca.core.image_evaluate import ImageEvalRequest, image_evaluate
    except ImportError as exc:
        _handle_error(exc)

    metric_tuple = tuple(m.strip() for m in metrics.split(",") if m.strip())
    ext_tuple = tuple(e.strip() for e in extensions.split(",") if e.strip())
    if not metric_tuple:
        typer.echo("Error: --metrics cannot be empty.", err=True)
        raise typer.Exit(code=2)

    try:
        request = ImageEvalRequest(
            rendered_dir=Path(rendered_dir),
            reference_dir=Path(reference_dir),
            metrics=metric_tuple,
            extensions=ext_tuple,
            ssim_window_size=ssim_window,
            ssim_sigma=ssim_sigma,
            max_pairs=max_pairs,
        )
        result = image_evaluate(request)
    except (FileNotFoundError, ValueError) as exc:
        _handle_error(exc)

    payload = {
        "summary": result.summary,
        "pairs": result.pairs,
        "metadata": result.metadata,
    }

    if format_json:
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        s = result.summary
        typer.echo(
            f"image-evaluate: {s['pairs_evaluated']} pair(s) scored "
            f"({s['pairs_missing_in_reference']} missing in reference, "
            f"{s['pairs_size_mismatch']} size-mismatch)"
        )
        for m in metric_tuple:
            mean = s.get(f"{m}_mean")
            median = s.get(f"{m}_median")
            if mean is None:
                typer.echo(f"  {m.upper():<5} -- no finite values")
            else:
                lo = s.get(f"{m}_min")
                hi = s.get(f"{m}_max")
                unit = " dB" if m == "psnr" else ""
                typer.echo(
                    f"  {m.upper():<5} mean={mean:.4f}{unit} "
                    f"median={median:.4f}{unit} "
                    f"min={lo:.4f}{unit} max={hi:.4f}{unit}"
                )

    if output_json:
        _dump_json(payload, output_json)


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
    max_lateral: Optional[float] = typer.Option(
        None, "--max-lateral",
        help="Maximum lateral RMSE allowed; exits with code 1 if exceeded",
    ),
    max_longitudinal: Optional[float] = typer.Option(
        None, "--max-longitudinal",
        help="Maximum longitudinal RMSE allowed; exits with code 1 if exceeded",
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
            max_lateral=max_lateral,
            max_longitudinal=max_longitudinal,
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


@app.command("slam-run")
def slam_run_cmd(
    input_path: str = typer.Argument(
        ...,
        help=(
            "Directory of LiDAR scans (.bin/.pcd/.ply) or a frames-list .txt "
            "(one path per line)."
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
            "SLAM driver to run. Built-in: 'kiss-icp' (default, adopted), "
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

    try:
        frame_paths = discover_frame_paths(in_path)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)

    request = SlamRunRequest(
        frame_paths=tuple(frame_paths),
        timestamps_s=None,
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


benchmark_app = typer.Typer(
    name="benchmark",
    help="SLAM benchmark suite runner (fixed reference + gate).",
    no_args_is_help=True,
)
app.add_typer(benchmark_app, name="benchmark")


bundle_app = typer.Typer(
    name="bundle",
    help="Pack / unpack / inspect CloudAnalyzer QA result bundles.",
    no_args_is_help=True,
)
app.add_typer(bundle_app, name="bundle")


def _parse_notes(values: Optional[List[str]]) -> dict[str, str]:
    notes: dict[str, str] = {}
    if not values:
        return notes
    for raw in values:
        if "=" not in raw:
            typer.echo(
                f"Error: --note expects key=value; got {raw!r}",
                err=True,
            )
            raise typer.Exit(code=1)
        key, _, value = raw.partition("=")
        key = key.strip()
        if not key:
            typer.echo("Error: --note key cannot be empty", err=True)
            raise typer.Exit(code=1)
        notes[key] = value.strip()
    return notes


@bundle_app.command("pack")
def bundle_pack_cmd(
    summary_path: str = typer.Argument(..., help="Path to summary JSON (ca check / run-evaluate / benchmark eval)"),
    output: str = typer.Option(..., "--output", "-o", help="Output bundle ZIP path"),
    baseline: Optional[str] = typer.Option(None, "--baseline", help="Optional baseline summary JSON of the same shape"),
    project: Optional[str] = typer.Option(None, "--project", help="Project label written into bundle metadata"),
    commit: Optional[str] = typer.Option(None, "--commit", help="Git commit SHA to record in metadata"),
    pr_number: Optional[str] = typer.Option(None, "--pr-number", help="PR number to record in metadata"),
    runner_id: Optional[str] = typer.Option(None, "--runner-id", help="CI runner identifier to record"),
    note: Optional[List[str]] = typer.Option(
        None,
        "--note",
        help="Extra metadata as key=value (repeatable), e.g. --note dataset=newer-college",
    ),
) -> None:
    """Bundle a QA summary plus referenced reports into a single qa_bundle.zip."""
    try:
        notes = _parse_notes(note)
        metadata = pack_bundle(
            summary_path,
            output,
            baseline_path=baseline,
            project=project,
            git_commit=commit,
            pr_number=pr_number,
            runner_id=runner_id,
            notes=notes,
        )
    except (FileNotFoundError, ValueError) as exc:
        _handle_error(exc)

    typer.echo(f"Bundle:   {output}")
    typer.echo(f"Kind:     {metadata.summary_kind}")
    typer.echo(f"Artifacts: {len(metadata.artifacts)} report file(s) included")
    if metadata.has_baseline:
        typer.echo("Baseline:  included")
    if metadata.project:
        typer.echo(f"Project:   {metadata.project}")
    if metadata.git_commit:
        typer.echo(f"Commit:    {metadata.git_commit}")
    if metadata.pr_number:
        typer.echo(f"PR:        {metadata.pr_number}")
    if metadata.runner_id:
        typer.echo(f"Runner:    {metadata.runner_id}")
    if metadata.notes:
        typer.echo("Notes:")
        for key, value in metadata.notes.items():
            typer.echo(f"  {key}={value}")


@bundle_app.command("unpack")
def bundle_unpack_cmd(
    bundle_path: str = typer.Argument(..., help="Path to a CloudAnalyzer qa_bundle.zip"),
    output_dir: str = typer.Option(..., "--output", "-o", help="Directory to extract into"),
) -> None:
    """Extract a bundle to a directory."""
    try:
        metadata = unpack_bundle(bundle_path, output_dir)
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        _handle_error(exc)
    typer.echo(f"Extracted: {output_dir}")
    typer.echo(f"Kind:      {metadata.summary_kind}")
    typer.echo(f"Artifacts: {len(metadata.artifacts)} report file(s) restored")


@bundle_app.command("show")
def bundle_show_cmd(
    bundle_path: str = typer.Argument(..., help="Path to a CloudAnalyzer qa_bundle.zip"),
    format_json: bool = typer.Option(False, "--format-json", help="Print as JSON"),
) -> None:
    """Show bundle metadata and table of contents without extracting."""
    try:
        info = show_bundle(bundle_path)
    except (FileNotFoundError, ValueError) as exc:
        _handle_error(exc)

    if format_json:
        typer.echo(json.dumps(info, indent=2))
        return

    metadata = info["metadata"]
    typer.echo(f"Bundle:               {info['bundle_path']}")
    typer.echo(f"Bundle version:       {metadata.get('bundle_version')}")
    typer.echo(f"Created at:           {metadata.get('created_at')}")
    typer.echo(f"CloudAnalyzer:        {metadata.get('cloudanalyzer_version')}")
    typer.echo(f"Summary kind:         {metadata.get('summary_kind')}")
    if metadata.get("project"):
        typer.echo(f"Project:              {metadata['project']}")
    if metadata.get("git_commit"):
        typer.echo(f"Git commit:           {metadata['git_commit']}")
    if metadata.get("pr_number"):
        typer.echo(f"PR number:            {metadata['pr_number']}")
    if metadata.get("runner_id"):
        typer.echo(f"Runner id:            {metadata['runner_id']}")
    if metadata.get("has_baseline"):
        typer.echo("Baseline:             included")
    if metadata.get("notes"):
        typer.echo("Notes:")
        for key, value in metadata["notes"].items():
            typer.echo(f"  {key}={value}")
    typer.echo("")
    typer.echo("Contents:")
    for entry in info["contents"]:
        typer.echo(
            f"  {entry['path']}  ({entry['size_bytes']} B / {entry['compressed_bytes']} B compressed)"
        )


@bundle_app.command("diff")
def bundle_diff_cmd(
    old_bundle: str = typer.Argument(..., help="Path to the older qa_bundle.zip"),
    new_bundle: str = typer.Argument(..., help="Path to the newer qa_bundle.zip"),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the diff Markdown to a file instead of stdout",
    ),
    format_json: bool = typer.Option(
        False,
        "--format-json",
        help="Print the structured diff dict as JSON (suitable for dashboards / scripting)",
    ),
) -> None:
    """Compare two QA bundles and render a Markdown report.

    Reuses the same metric layout as `ca report-pr-comment`, so the diff
    looks identical to what PRs already render — old becomes the baseline,
    new is the "current" set of numbers.
    """
    try:
        diff = diff_bundles(old_bundle, new_bundle)
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        _handle_error(exc)

    if format_json:
        typer.echo(json.dumps(diff, indent=2))
        return

    markdown = render_diff_markdown(diff)
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        typer.echo(f"Diff written: {output}")
    else:
        typer.echo(markdown, nl=False)


@app.command("history")
def history_cmd(
    bundles: List[str] = typer.Argument(
        None,
        help="Paths to one or more qa_bundle.zip archives (oldest-to-newest is determined "
        "from metadata.created_at, not the argument order).",
    ),
    from_dir: Optional[str] = typer.Option(
        None,
        "--from-dir",
        help="Discover bundles in this directory (matches `*.zip` by default).",
    ),
    pattern: str = typer.Option(
        "*.zip",
        "--pattern",
        help="Glob pattern used with --from-dir (default: *.zip)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the Markdown history to a file instead of stdout.",
    ),
    format_json: bool = typer.Option(
        False,
        "--format-json",
        help="Emit the structured history payload as JSON (suitable for dashboards / scripting).",
    ),
) -> None:
    """Build a time-series view of QA gate metrics across many bundles.

    Reads N `qa_bundle.zip` archives, sorts them by their metadata's
    `created_at` stamp, and renders a per-metric trend table:

    - `check_suite` bundles: one table per `check_id` with that check's
      relevant metrics across the timeline.
    - `single_run` (`ca run-evaluate` / `ca benchmark eval`) bundles: one
      table covering map AUC / Chamfer / F1 + trajectory ATE / RPE /
      Drift / Coverage.

    Mixing bundle shapes across the input set is rejected.
    """
    paths: list[str] = list(bundles or [])
    if from_dir:
        try:
            discovered = discover_bundles(from_dir, pattern=pattern)
        except NotADirectoryError as exc:
            _handle_error(exc)
        paths.extend(str(p) for p in discovered)
    if not paths:
        typer.echo(
            "Error: provide at least one bundle path or use --from-dir.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        entries = build_history_series(paths)
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        _handle_error(exc)

    if format_json:
        typer.echo(json.dumps(render_history_json(entries), indent=2))
        return

    markdown = render_history_markdown(entries)
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        typer.echo(f"History written: {output}")
    else:
        typer.echo(markdown, nl=False)


def _parse_gate_overrides(values: Optional[List[str]]) -> dict[str, float | None]:
    """Parse repeated --gate key=value (or key=none) overrides."""
    overrides: dict[str, float | None] = {}
    if not values:
        return overrides
    for raw in values:
        if "=" not in raw:
            typer.echo(
                f"Error: --gate expects key=value; got {raw!r}",
                err=True,
            )
            raise typer.Exit(code=1)
        key, _, value = raw.partition("=")
        key = key.strip()
        value = value.strip()
        if key not in GATE_KEYS:
            typer.echo(
                f"Error: unknown gate key {key!r}. Allowed: {', '.join(GATE_KEYS)}",
                err=True,
            )
            raise typer.Exit(code=1)
        if value.lower() in {"none", "null", ""}:
            overrides[key] = None
        else:
            try:
                overrides[key] = float(value)
            except ValueError:
                typer.echo(
                    f"Error: --gate {key} must be numeric or 'none'; got {value!r}",
                    err=True,
                )
                raise typer.Exit(code=1)
    return overrides


def _print_suite_info(suite: BenchmarkSuite) -> None:
    typer.echo(f"Suite:       {suite.name} (v{suite.version})")
    typer.echo(f"Description: {suite.description}")
    if suite.license:
        typer.echo(f"License:     {suite.license}")
    typer.echo(f"Source:      {suite.source_path}")
    typer.echo("Sequences:")
    for name, seq in suite.sequences.items():
        typer.echo(f"  - {name}: {seq.description}")
        typer.echo(f"      reference_map:        {seq.reference_map_path}")
        typer.echo(f"      reference_trajectory: {seq.reference_trajectory_path}")
        if seq.sample_map_path or seq.sample_trajectory_path:
            typer.echo("      sample_outputs:")
            if seq.sample_map_path:
                typer.echo(f"        map:        {seq.sample_map_path}")
            if seq.sample_trajectory_path:
                typer.echo(f"        trajectory: {seq.sample_trajectory_path}")
    if suite.gate:
        typer.echo("Gate:")
        for key in GATE_KEYS:
            if key in suite.gate:
                typer.echo(f"  {key}: {suite.gate[key]}")


@benchmark_app.command("info")
def benchmark_info_cmd(
    suite_path: str = typer.Argument(..., help="Path to a benchmark suite YAML manifest"),
    format_json: bool = typer.Option(False, "--format-json", help="Print suite metadata as JSON"),
) -> None:
    """Show the sequences, references, and gate for a benchmark suite."""
    try:
        suite = load_benchmark_suite(suite_path)
    except (FileNotFoundError, ValueError) as exc:
        _handle_error(exc)

    if format_json:
        payload = {
            "name": suite.name,
            "version": suite.version,
            "description": suite.description,
            "license": suite.license,
            "source_path": str(suite.source_path),
            "sequences": {
                name: {
                    "description": seq.description,
                    "reference_map": str(seq.reference_map_path),
                    "reference_trajectory": str(seq.reference_trajectory_path),
                    "sample_map": (
                        str(seq.sample_map_path) if seq.sample_map_path else None
                    ),
                    "sample_trajectory": (
                        str(seq.sample_trajectory_path)
                        if seq.sample_trajectory_path
                        else None
                    ),
                }
                for name, seq in suite.sequences.items()
            },
            "gate": dict(suite.gate),
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        _print_suite_info(suite)


@benchmark_app.command("eval")
def benchmark_eval_cmd(
    suite_path: str = typer.Argument(..., help="Path to a benchmark suite YAML manifest"),
    map_path: str = typer.Option(..., "--map", help="Estimated map point cloud"),
    trajectory_path: str = typer.Option(..., "--trajectory", help="Estimated trajectory"),
    sequence: Optional[str] = typer.Option(
        None,
        "--sequence",
        help="Sequence name to evaluate against (defaults to the first sequence)",
    ),
    thresholds: Optional[str] = typer.Option(
        None,
        "--thresholds",
        "-t",
        help="Comma-separated distance thresholds for map F1/AUC evaluation",
    ),
    max_time_delta: float = typer.Option(
        0.05,
        "--max-time-delta",
        help="Max timestamp gap allowed for trajectory matching (seconds)",
    ),
    align_origin: bool = typer.Option(False, "--align-origin"),
    align_rigid: bool = typer.Option(False, "--align-rigid"),
    gate_overrides: Optional[List[str]] = typer.Option(
        None,
        "--gate",
        help="Override suite gate (repeatable): --gate min_auc=0.97 --gate max_rpe=none",
    ),
    report: Optional[str] = typer.Option(None, "--report", help="Write combined run report"),
    output_json: Optional[str] = typer.Option(
        None,
        "--output-json",
        help="Dump benchmark result as JSON",
    ),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Evaluate a SLAM run against a benchmark suite's fixed reference and gate."""
    try:
        suite = load_benchmark_suite(suite_path)
        overrides = _parse_gate_overrides(gate_overrides)
        result = evaluate_benchmark_run(
            suite,
            map_path,
            trajectory_path,
            sequence=sequence,
            gate_overrides=overrides,
            thresholds=_parse_thresholds(thresholds),
            max_time_delta=max_time_delta,
            align_origin=align_origin,
            align_rigid=align_rigid,
        )
    except (FileNotFoundError, ValueError) as exc:
        _handle_error(exc)

    if report:
        try:
            save_run_report(result, report)
        except ValueError as exc:
            _handle_error(exc)

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        info = result["benchmark"]
        map_result = result["map"]
        trajectory_result = result["trajectory"]
        typer.echo(
            f"Benchmark:  {info['suite']} v{info['version']} / sequence={info['sequence']}"
        )
        typer.echo(
            f"Map:        AUC={map_result['auc']:.4f}  Chamfer={map_result['chamfer_distance']:.4f}"
        )
        typer.echo(
            "Trajectory: ATE={ate:.4f}  RPE={rpe:.4f}  Drift={drift:.4f}  Coverage={cov:.1%}".format(
                ate=trajectory_result["ate"]["rmse"],
                rpe=trajectory_result["rpe_translation"]["rmse"],
                drift=trajectory_result["drift"]["endpoint"],
                cov=trajectory_result["matching"]["coverage_ratio"],
            )
        )
        overall_gate = result["overall_quality_gate"]
        if overall_gate is not None:
            typer.echo("")
            typer.echo(
                f"Overall Quality Gate: {'PASS' if overall_gate['passed'] else 'FAIL'}"
            )
            for reason in overall_gate["reasons"]:
                typer.echo(f"  - {reason}")
        if report:
            typer.echo(f"Report: {report}")

    if output_json:
        _dump_json(result, output_json)
    overall = result["overall_quality_gate"]
    if overall is not None and not overall["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("init")
def benchmark_init_cmd(
    suite_dir: str = typer.Argument(..., help="Directory to materialize the suite into (will be created)"),
    name: str = typer.Option(..., "--name", help="Suite name written to suite.yaml"),
    description: str = typer.Option(..., "--description", help="Suite description"),
    reference_map: str = typer.Option(..., "--reference-map", help="Path to the reference (GT) point cloud map"),
    reference_trajectory: str = typer.Option(
        ...,
        "--reference-trajectory",
        help="Path to the reference (GT) trajectory in TUM format",
    ),
    sequence: str = typer.Option(
        "default",
        "--sequence",
        help="Sequence name inside the suite manifest",
    ),
    sequence_description: Optional[str] = typer.Option(
        None,
        "--sequence-description",
        help="Per-sequence description (defaults to the suite description)",
    ),
    license: Optional[str] = typer.Option(
        None,
        "--license",
        help="License string written into suite.yaml (e.g. dataset attribution)",
    ),
    voxel: float = typer.Option(
        0.0,
        "--voxel",
        help="Voxel size (meters) for downsampling the reference map; 0 = no downsample",
    ),
    max_poses: Optional[int] = typer.Option(
        None,
        "--max-poses",
        help="Keep at most this many evenly-spaced trajectory poses; default = keep all",
    ),
    sample_map: Optional[str] = typer.Option(
        None,
        "--sample-map",
        help="Optional sample-output map to bundle for smoke tests",
    ),
    sample_trajectory: Optional[str] = typer.Option(
        None,
        "--sample-trajectory",
        help="Optional sample-output trajectory to bundle for smoke tests",
    ),
    gate_values: Optional[List[str]] = typer.Option(
        None,
        "--gate",
        help="Gate threshold as key=value (repeatable): --gate min_auc=0.97 --gate max_chamfer=0.05",
    ),
) -> None:
    """Materialize a SLAM benchmark suite from raw GT data on disk."""
    gate: dict[str, float] = {}
    if gate_values:
        for raw in gate_values:
            if "=" not in raw:
                typer.echo(f"Error: --gate expects key=value; got {raw!r}", err=True)
                raise typer.Exit(code=1)
            key, _, value = raw.partition("=")
            key = key.strip()
            if key not in GATE_KEYS:
                typer.echo(
                    f"Error: unknown gate key {key!r}. Allowed: {', '.join(GATE_KEYS)}",
                    err=True,
                )
                raise typer.Exit(code=1)
            try:
                gate[key] = float(value)
            except ValueError:
                typer.echo(f"Error: --gate {key} must be numeric; got {value!r}", err=True)
                raise typer.Exit(code=1)

    try:
        suite = materialize_suite(
            suite_dir,
            name=name,
            description=description,
            reference_map=reference_map,
            reference_trajectory=reference_trajectory,
            sequence_name=sequence,
            sequence_description=sequence_description,
            license=license,
            voxel_size=voxel,
            max_poses=max_poses,
            gate=gate or None,
            sample_map=sample_map,
            sample_trajectory=sample_trajectory,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _handle_error(exc)

    seq = suite.resolve_sequence(sequence)
    typer.echo(f"Suite:               {suite.name}")
    typer.echo(f"Manifest:            {suite.source_path}")
    typer.echo(f"Sequence:            {seq.name}")
    typer.echo(f"Reference map:       {seq.reference_map_path}")
    typer.echo(f"Reference trajectory: {seq.reference_trajectory_path}")
    if voxel > 0:
        typer.echo(f"Voxel downsample:    {voxel} m")
    if max_poses is not None:
        typer.echo(f"Max poses kept:      {max_poses}")
    if suite.gate:
        typer.echo("Gate:")
        for gate_key, gate_value in suite.gate.items():
            typer.echo(f"  {gate_key}: {gate_value}")


@app.command("report-pr-comment")
def report_pr_comment_cmd(
    summary_path: str = typer.Argument(
        ...,
        help="Path to a `ca check` summary or `ca run-evaluate` / `ca benchmark eval` JSON",
    ),
    baseline: Optional[str] = typer.Option(
        None,
        "--baseline",
        help="Baseline JSON of the same shape; renders metric deltas",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project",
        help="Project label shown in the header (overrides the JSON's `project` field)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the Markdown to a file instead of stdout",
    ),
) -> None:
    """Render a PR-comment Markdown blob from a CloudAnalyzer summary JSON."""
    try:
        summary = _load_json_mapping(summary_path)
    except FileNotFoundError as exc:
        _handle_error(exc)
    except ValueError as exc:
        _handle_error(exc)

    baseline_data = None
    if baseline:
        try:
            baseline_data = _load_json_mapping(baseline)
        except FileNotFoundError as exc:
            _handle_error(exc)
        except ValueError as exc:
            _handle_error(exc)

    try:
        markdown = build_pr_comment(summary, baseline=baseline_data, project=project)
    except ValueError as exc:
        _handle_error(exc)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        typer.echo(f"PR comment: {output}")
    else:
        typer.echo(markdown, nl=False)


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


@app.command("baseline-decision")
def baseline_decision_cmd(
    candidate_json: str = typer.Argument(
        ...,
        help="Candidate summary JSON emitted by `ca check --output-json`",
    ),
    history_json: Optional[List[str]] = typer.Option(
        None,
        "--history",
        help="Historical summary JSON files in oldest-to-newest order",
    ),
    history_dir: Optional[str] = typer.Option(
        None,
        "--history-dir",
        help="Auto-discover history JSONs from a directory (alternative to --history)",
    ),
    output_json: Optional[str] = typer.Option(
        None,
        "--output-json",
        help="Dump baseline decision summary as JSON",
    ),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Decide whether a candidate QA result should promote, keep, or reject a baseline."""

    try:
        candidate_result = _load_json_mapping(candidate_json)
        if history_dir and not history_json:
            history_paths = discover_history(history_dir)
        else:
            history_paths = list(history_json or [])
        history_results = [_load_json_mapping(path) for path in history_paths]
        result = summarize_baseline_evolution(candidate_result, history_results)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)
        return

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        _print_baseline_evolution_result(result)

    if output_json:
        if format_json:
            destination = Path(output_json)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
        else:
            _dump_json(result, output_json)
    if result["decision"] == "reject":
        raise typer.Exit(code=1)


@app.command("baseline-save")
def baseline_save_cmd(
    summary_json: str = typer.Argument(
        ..., help="QA summary JSON to save (from `ca check --output-json`)",
    ),
    history_dir: str = typer.Option(
        "qa/history", "--history-dir", help="Directory to store baseline history",
    ),
    label: Optional[str] = typer.Option(
        None, "--label", help="Custom label instead of auto-generated timestamp",
    ),
    keep: Optional[int] = typer.Option(
        None, "--keep", help="Rotate history to keep only this many baselines",
    ),
) -> None:
    """Save a QA summary as a baseline in the history directory."""
    try:
        dest = save_baseline(summary_json, history_dir, label=label)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)
        return
    typer.echo(f"Saved: {dest}")
    if keep is not None:
        removed = rotate_history(history_dir, keep=keep)
        if removed:
            typer.echo(f"Rotated: removed {len(removed)} old baseline(s)")


@app.command("baseline-list")
def baseline_list_cmd(
    history_dir: str = typer.Option(
        "qa/history", "--history-dir", help="Directory containing baseline history",
    ),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """List saved baselines in the history directory."""
    entries = list_baselines(history_dir)
    if format_json:
        typer.echo(json.dumps(entries, indent=2))
    else:
        if not entries:
            typer.echo(f"No baselines found in {history_dir}")
            return
        typer.echo(f"Baselines: {len(entries)} in {history_dir}")
        for entry in entries:
            status = "PASS" if entry["passed"] else "FAIL" if entry["passed"] is False else "?"
            typer.echo(f"  [{status}] {entry['name']}")


@app.command("geometry-evaluate")
def geometry_evaluate_cmd(
    source: str = typer.Argument(..., help="Source artifact (.ply/.pcd/.las/...). 3DGS PLY auto-detected."),
    reference: str = typer.Argument(..., help="Reference point cloud (.pcd/.ply/.las/...)"),
    representation: str = typer.Option(
        "auto",
        "--representation",
        help=f"Source representation: {', '.join(REPRESENTATIONS)}",
    ),
    opacity_threshold: Optional[float] = typer.Option(
        None,
        "--opacity-threshold",
        help="Drop splats whose rendered alpha (sigmoid of opacity) is below this; gaussian-points only",
    ),
    voxel: Optional[float] = typer.Option(
        None,
        "--voxel",
        help="Voxel-downsample the adapted source before evaluation (meters)",
    ),
    mesh_samples: int = typer.Option(
        DEFAULT_MESH_SAMPLES,
        "--mesh-samples",
        help=f"Surface-sample this many points from a mesh (default: {DEFAULT_MESH_SAMPLES}); mesh representation only",
    ),
    mesh_method: str = typer.Option(
        "uniform",
        "--mesh-method",
        help=f"Mesh sampling strategy ({', '.join(MESH_SAMPLE_METHODS)}); mesh representation only",
    ),
    splat_method: str = typer.Option(
        "centers",
        "--splat-method",
        help=(
            f"3DGS sampling strategy ({', '.join(SPLAT_METHODS)}); gaussian-points only. "
            "`centers` uses splat centers only; `ellipsoid` surface-samples each splat "
            "using scale_*/rot_* properties for a better proxy of the rendered surface."
        ),
    ),
    splat_samples: int = typer.Option(
        DEFAULT_SPLAT_SAMPLES,
        "--splat-samples",
        help=(
            f"Points sampled per splat in ellipsoid mode (default: {DEFAULT_SPLAT_SAMPLES}); "
            "ignored unless --splat-method=ellipsoid"
        ),
    ),
    thresholds: Optional[str] = typer.Option(
        None,
        "--thresholds",
        "-t",
        help="Comma-separated distance thresholds for F1/AUC evaluation",
    ),
    plot: Optional[str] = typer.Option(None, "--plot", help="Write the F1 curve to this PNG"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Cross-representation geometry QA (3DGS PLY, triangle meshes, point clouds, ...)."""
    if representation not in REPRESENTATIONS:
        typer.echo(
            f"Error: --representation must be one of {', '.join(REPRESENTATIONS)}",
            err=True,
        )
        raise typer.Exit(code=1)
    if mesh_method not in MESH_SAMPLE_METHODS:
        typer.echo(
            f"Error: --mesh-method must be one of {', '.join(MESH_SAMPLE_METHODS)}",
            err=True,
        )
        raise typer.Exit(code=1)
    if splat_method not in SPLAT_METHODS:
        typer.echo(
            f"Error: --splat-method must be one of {', '.join(SPLAT_METHODS)}",
            err=True,
        )
        raise typer.Exit(code=1)
    if splat_samples < 2:
        typer.echo(
            "Error: --splat-samples must be >= 2 for ellipsoid sampling",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        result = evaluate_geometry(
            source,
            reference,
            representation=representation,
            opacity_threshold=opacity_threshold,
            voxel_size=voxel,
            thresholds=_parse_thresholds(thresholds),
            mesh_samples=mesh_samples,
            mesh_method=mesh_method,
            splat_method=splat_method,
            splat_samples=splat_samples,
        )
    except (FileNotFoundError, ValueError) as exc:
        _handle_error(exc)

    if plot:
        try:
            plot_f1_curve(result, plot)
        except ValueError as exc:
            _handle_error(exc)

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        rep = result["representation"]
        best_f1 = max(result["f1_scores"], key=lambda s: s["f1"])
        typer.echo(
            f"Source:        {result['source_points']} pts "
            f"(representation={rep['detected']}, original={rep['original_count']})"
        )
        if rep["applied_filters"]:
            typer.echo("Filters:       " + "; ".join(rep["applied_filters"]))
        typer.echo(f"Reference:     {result['target_points']} pts")
        typer.echo(
            f"  Chamfer={result['chamfer_distance']:.4f}  "
            f"Hausdorff={result['hausdorff_distance']:.4f}  "
            f"AUC={result['auc']:.4f}"
        )
        typer.echo(
            f"  Best F1={best_f1['f1']:.4f} @ d={best_f1['threshold']:.2f}"
        )
        if plot:
            typer.echo(f"Plot: {plot}")

    if output_json:
        _dump_json(result, output_json)


@app.command("ground-evaluate")
def ground_evaluate_cmd(
    estimated_ground: str = typer.Argument(..., help="Estimated ground points (pcd/ply/las)"),
    estimated_nonground: str = typer.Argument(..., help="Estimated non-ground points (pcd/ply/las)"),
    reference_ground: str = typer.Argument(..., help="Reference ground points (pcd/ply/las)"),
    reference_nonground: str = typer.Argument(..., help="Reference non-ground points (pcd/ply/las)"),
    voxel_size: float = typer.Option(0.2, "--voxel-size", help="Voxel grid resolution for comparison (meters)"),
    min_precision: Optional[float] = typer.Option(None, "--min-precision", help="Minimum precision required"),
    min_recall: Optional[float] = typer.Option(None, "--min-recall", help="Minimum recall required"),
    min_f1: Optional[float] = typer.Option(None, "--min-f1", help="Minimum F1 score required"),
    min_iou: Optional[float] = typer.Option(None, "--min-iou", help="Minimum IoU required"),
    report: Optional[str] = typer.Option(
        None, "--report",
        help="Write ground segmentation report (.md or .html)",
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Evaluate ground segmentation quality against reference labels."""
    try:
        result = evaluate_ground_segmentation(
            estimated_ground,
            estimated_nonground,
            reference_ground,
            reference_nonground,
            voxel_size=voxel_size,
            min_precision=min_precision,
            min_recall=min_recall,
            min_f1=min_f1,
            min_iou=min_iou,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)
        return

    if report:
        try:
            save_ground_report(result, report)
        except ValueError as e:
            _handle_error(e)
            return

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        counts = result["counts"]
        cm = result["confusion_matrix"]
        typer.echo(
            f"Estimated: ground={counts['estimated_ground_points']}  "
            f"nonground={counts['estimated_nonground_points']}"
        )
        typer.echo(
            f"Reference: ground={counts['reference_ground_points']}  "
            f"nonground={counts['reference_nonground_points']}"
        )
        typer.echo(f"Voxel:     {result['voxel_size']}m")
        typer.echo("")
        typer.echo(f"TP={cm['tp']}  FP={cm['fp']}  FN={cm['fn']}  TN={cm['tn']}")
        typer.echo(
            f"Precision: {result['precision']:.4f}  "
            f"Recall: {result['recall']:.4f}  "
            f"F1: {result['f1']:.4f}"
        )
        typer.echo(f"IoU:       {result['iou']:.4f}  Accuracy: {result['accuracy']:.4f}")
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


@app.command("detection-evaluate")
def detection_evaluate_cmd(
    estimated: str = typer.Argument(..., help="Estimated detection sequence (.json)"),
    reference: str = typer.Argument(..., help="Reference detection sequence (.json)"),
    iou_thresholds: Optional[str] = typer.Option(
        None,
        "--iou-thresholds",
        help="Comma-separated IoU thresholds (default: 0.25,0.50)",
    ),
    primary_iou_threshold: Optional[float] = typer.Option(
        None,
        "--primary-iou-threshold",
        help="Threshold from --iou-thresholds used for precision/recall/F1 gating",
    ),
    min_map: Optional[float] = typer.Option(None, "--min-map", help="Minimum mAP required"),
    min_precision: Optional[float] = typer.Option(
        None, "--min-precision", help="Minimum precision at the primary IoU threshold required"
    ),
    min_recall: Optional[float] = typer.Option(
        None, "--min-recall", help="Minimum recall at the primary IoU threshold required"
    ),
    min_f1: Optional[float] = typer.Option(
        None, "--min-f1", help="Minimum F1 at the primary IoU threshold required"
    ),
    report: Optional[str] = typer.Option(
        None, "--report", help="Write detection report (.md or .html)"
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Evaluate 3D object detections against reference boxes."""
    threshold_list = _parse_thresholds(iou_thresholds)
    try:
        result = evaluate_detection(
            estimated,
            reference,
            iou_thresholds=threshold_list,
            primary_iou_threshold=primary_iou_threshold,
            min_map=min_map,
            min_precision=min_precision,
            min_recall=min_recall,
            min_f1=min_f1,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)
        return

    if report:
        try:
            save_detection_report(result, report)
        except ValueError as e:
            _handle_error(e)
            return

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        counts = result["counts"]
        primary = result["primary_threshold_result"]
        typer.echo(
            f"Estimated: {counts['estimated_frames']} frames / {counts['estimated_boxes']} boxes | "
            f"Reference: {counts['reference_frames']} frames / {counts['reference_boxes']} boxes"
        )
        typer.echo(f"Shared frames: {counts['shared_frames']}")
        typer.echo("")
        typer.echo(f"mAP:         {result['mAP']:.4f}")
        typer.echo(
            f"Primary IoU: {primary['iou_threshold']:.2f}  "
            f"Precision={primary['precision']:.4f}  "
            f"Recall={primary['recall']:.4f}  "
            f"F1={primary['f1']:.4f}"
        )
        typer.echo(
            f"Mean IoU:    {primary['mean_iou']:.4f}  "
            f"Mean center distance={primary['mean_center_distance']:.4f}"
        )
        typer.echo("Per-class AP:")
        for label, summary in sorted(result["per_class"].items()):
            typer.echo(
                f"  {label}: mean_ap={summary['mean_ap']:.4f}"
                if summary["mean_ap"] is not None
                else f"  {label}: mean_ap=n/a"
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


@app.command("tracking-evaluate")
def tracking_evaluate_cmd(
    estimated: str = typer.Argument(..., help="Estimated tracking sequence (.json)"),
    reference: str = typer.Argument(..., help="Reference tracking sequence (.json)"),
    iou_threshold: float = typer.Option(0.5, "--iou-threshold", help="IoU threshold for frame-wise box matching"),
    min_mota: Optional[float] = typer.Option(None, "--min-mota", help="Minimum MOTA required"),
    min_recall: Optional[float] = typer.Option(None, "--min-recall", help="Minimum recall required"),
    max_id_switches: Optional[int] = typer.Option(
        None, "--max-id-switches", help="Maximum ID switches allowed"
    ),
    report: Optional[str] = typer.Option(
        None, "--report", help="Write tracking report (.md or .html)"
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Evaluate 3D multi-object tracking against reference tracks."""
    try:
        result = evaluate_tracking(
            estimated,
            reference,
            iou_threshold=iou_threshold,
            min_mota=min_mota,
            min_recall=min_recall,
            max_id_switches=max_id_switches,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)
        return

    if report:
        try:
            save_tracking_report(result, report)
        except ValueError as e:
            _handle_error(e)
            return

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        counts = result["counts"]
        detection = result["detection"]
        tracking = result["tracking"]
        typer.echo(
            f"Estimated: {counts['estimated_frames']} frames / {counts['estimated_detections']} detections / "
            f"{counts['estimated_tracks']} tracks"
        )
        typer.echo(
            f"Reference: {counts['reference_frames']} frames / {counts['reference_detections']} detections / "
            f"{counts['reference_tracks']} tracks"
        )
        typer.echo(f"Shared frames: {counts['shared_frames']}")
        typer.echo("")
        typer.echo(
            f"Precision: {detection['precision']:.4f}  "
            f"Recall: {detection['recall']:.4f}  "
            f"F1: {detection['f1']:.4f}"
        )
        typer.echo(
            f"MOTA:      {tracking['mota']:.4f}  "
            f"ID switches={tracking['id_switches']}  "
            f"Fragments={tracking['track_fragmentations']}"
        )
        typer.echo(
            f"Mean IoU:  {tracking['mean_iou']:.4f}  "
            f"Mean center distance={tracking['mean_center_distance']:.4f}"
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


@app.command("lidar-odometry-view")
def lidar_odometry_view_cmd(
    trajectory: str = typer.Argument(..., help="LiDAR odometry trajectory (.csv/.tum/.txt)"),
    map_clouds: Optional[List[str]] = typer.Option(
        None,
        "--map",
        "-m",
        help="Optional map/point-cloud file to show behind the odometry trajectory. Repeatable.",
    ),
    trajectory_reference: Optional[str] = typer.Option(
        None,
        "--trajectory-reference",
        help="Optional reference trajectory to overlay alongside the odometry result.",
    ),
    trajectory_max_time_delta: float = typer.Option(
        0.05,
        "--trajectory-max-time-delta",
        help="Max timestamp gap allowed when matching to --trajectory-reference.",
    ),
    trajectory_align_origin: bool = typer.Option(
        False,
        "--trajectory-align-origin",
        help="Translate the odometry trajectory so its first matched pose aligns to the reference.",
    ),
    trajectory_align_rigid: bool = typer.Option(
        False,
        "--trajectory-align-rigid",
        help="Fit a rigid transform from odometry trajectory to reference before display.",
    ),
    slam_debug_report: Optional[str] = typer.Option(
        None,
        "--slam-debug-report",
        help="Optional ca slam-debug JSON report; selected frames are marked on the trajectory.",
    ),
    port: int = typer.Option(8080, "--port", "-p", help="HTTP port"),
    max_points: int = typer.Option(2_000_000, "--max-points", help="Max map points for display"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't auto-open browser"),
) -> None:
    """Open a LiDAR odometry result in the web viewer."""

    try:
        web_serve(
            list(map_clouds or []),
            port=port,
            max_points=max_points,
            open_browser=not no_browser,
            heatmap=False,
            trajectory_path=trajectory,
            trajectory_reference_path=trajectory_reference,
            trajectory_max_time_delta=trajectory_max_time_delta,
            trajectory_align_origin=trajectory_align_origin,
            trajectory_align_rigid=trajectory_align_rigid,
            slam_debug_report_path=slam_debug_report,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)


@app.command("lidar-odometry-export")
def lidar_odometry_export_cmd(
    trajectory: str = typer.Argument(..., help="LiDAR odometry trajectory (.csv/.tum/.txt)"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Static bundle output directory"),
    map_clouds: Optional[List[str]] = typer.Option(
        None,
        "--map",
        "-m",
        help="Optional map/point-cloud file to show behind the odometry trajectory. Repeatable.",
    ),
    trajectory_reference: Optional[str] = typer.Option(
        None,
        "--trajectory-reference",
        help="Optional reference trajectory to overlay alongside the odometry result.",
    ),
    trajectory_max_time_delta: float = typer.Option(
        0.05,
        "--trajectory-max-time-delta",
        help="Max timestamp gap allowed when matching to --trajectory-reference.",
    ),
    trajectory_align_origin: bool = typer.Option(
        False,
        "--trajectory-align-origin",
        help="Translate the odometry trajectory so its first matched pose aligns to the reference.",
    ),
    trajectory_align_rigid: bool = typer.Option(
        False,
        "--trajectory-align-rigid",
        help="Fit a rigid transform from odometry trajectory to reference before display.",
    ),
    slam_debug_report: Optional[str] = typer.Option(
        None,
        "--slam-debug-report",
        help="Optional ca slam-debug JSON report; selected frames are marked on the trajectory.",
    ),
    max_points: int = typer.Option(2_000_000, "--max-points", help="Max map points for display"),
) -> None:
    """Export a static LiDAR odometry viewer bundle."""

    try:
        result = web_export_static_bundle(
            list(map_clouds or []),
            output_dir=output_dir,
            max_points=max_points,
            heatmap=False,
            trajectory_path=trajectory,
            trajectory_reference_path=trajectory_reference,
            trajectory_max_time_delta=trajectory_max_time_delta,
            trajectory_align_origin=trajectory_align_origin,
            trajectory_align_rigid=trajectory_align_rigid,
            slam_debug_report_path=slam_debug_report,
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


@app.command("convert-labels")
def convert_labels_cmd(
    format: str = typer.Option(..., "--format", help="Label format (kitti)"),
    input: str = typer.Option(..., "--input", help="Input label directory"),
    output: str = typer.Option(..., "--output", help="Output JSON path"),
    no_camera_to_lidar: bool = typer.Option(
        False, "--no-camera-to-lidar", help="Skip KITTI camera-to-lidar transform"
    ),
):
    """Convert external label formats to CloudAnalyzer JSON."""
    if format != "kitti":
        typer.echo(f"Unsupported format: {format}. Supported: kitti", err=True)
        raise typer.Exit(code=1)

    from ca.kitti import convert_kitti_labels

    try:
        result = convert_kitti_labels(
            input, output, camera_to_lidar=not no_camera_to_lidar
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)
        return
    typer.echo(json.dumps(result, indent=2))


def main():
    app()


if __name__ == "__main__":
    main()
