"""CloudAnalyzer CLI entry point."""

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer

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
from ca.batch import batch_info
from ca.density_map import density_map
from ca.evaluate import evaluate, plot_f1_curve
from ca.split import split
from ca.pipeline import run_pipeline
from ca.plot import heatmap3d
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
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump results as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Run info on all point cloud files in a directory."""
    if format_json:
        import logging
        logging.getLogger("ca").setLevel(logging.ERROR)
    try:
        results = batch_info(directory, recursive=recursive)
    except FileNotFoundError as e:
        _handle_error(e)

    if format_json:
        typer.echo(json.dumps(results, indent=2))
    else:
        for info in results:
            typer.echo(f"  {info['path']}: {info['num_points']} pts")
        typer.echo(f"Total files: {len(results)}")
    if output_json:
        _dump_json(results, output_json)


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
    thresh_list = None
    if thresholds:
        try:
            thresh_list = [float(x.strip()) for x in thresholds.split(",")]
        except ValueError:
            typer.echo("Error: --thresholds must be comma-separated numbers", err=True)
            raise typer.Exit(code=1)

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
    thresh_list = None
    if thresholds:
        try:
            thresh_list = [float(x.strip()) for x in thresholds.split(",")]
        except ValueError:
            typer.echo("Error: --thresholds must be comma-separated numbers", err=True)
            raise typer.Exit(code=1)

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
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't auto-open browser"),
) -> None:
    """Open point cloud in web browser (Three.js viewer)."""
    try:
        web_serve(paths, port=port, max_points=max_points, open_browser=not no_browser)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)


@app.command("version")
def version_cmd() -> None:
    """Show CloudAnalyzer version."""
    from ca import __version__
    typer.echo(f"CloudAnalyzer v{__version__}")


def main():
    app()


if __name__ == "__main__":
    main()
