"""CloudAnalyzer CLI entry point."""

import json
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

app = typer.Typer(
    name="ca",
    help="CloudAnalyzer - AI-friendly CLI tool for point cloud analysis.",
)


def _dump_json(data: dict, path: str) -> None:
    """Write result dict to JSON file."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    typer.echo(f"JSON: {path}")


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
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    if output_json:
        _dump_json(result, output_json)


@app.command("info")
def info_cmd(
    path: str = typer.Argument(..., help="Path to point cloud file (pcd/ply/las)"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Show basic information about a point cloud file."""
    try:
        info = get_info(path)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

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
) -> None:
    """Quick distance stats between two point clouds (no registration)."""
    try:
        result = run_diff(source, target, threshold=threshold)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

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


@app.command("view")
def view_cmd(
    paths: List[str] = typer.Argument(..., help="Point cloud file(s) to view"),
) -> None:
    """Open interactive 3D viewer for point cloud(s)."""
    try:
        view(paths)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("downsample")
def downsample_cmd(
    path: str = typer.Argument(..., help="Input point cloud file"),
    output: str = typer.Option(..., "--output", "-o", help="Output file path"),
    voxel_size: float = typer.Option(0.05, "--voxel-size", "-v", help="Voxel size"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Downsample a point cloud using voxel grid filtering."""
    try:
        result = downsample(path, voxel_size, output)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Original:     {result['original_points']} pts")
    typer.echo(f"Downsampled:  {result['downsampled_points']} pts")
    typer.echo(f"Reduction:    {result['reduction_ratio']:.1%}")
    typer.echo(f"Voxel size:   {result['voxel_size']}")
    typer.echo(f"Saved:        {result['output']}")
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
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

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
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

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
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Original: {result['original_points']} pts")
    typer.echo(f"Cropped:  {result['cropped_points']} pts")
    typer.echo(f"Saved:    {result['output']}")
    if output_json:
        _dump_json(result, output_json)


@app.command("stats")
def stats_cmd(
    path: str = typer.Argument(..., help="Point cloud file"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Show detailed statistics (density, spacing distribution)."""
    try:
        result = compute_stats(path)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"File:      {result['path']}")
    typer.echo(f"Points:    {result['num_points']}")
    typer.echo(f"Volume:    {result['volume']:.4f}")
    typer.echo(f"Density:   {result['density']:.2f} pts/unit³")
    s = result["spacing"]
    typer.echo(f"Spacing mean:   {s['mean']:.4f}")
    typer.echo(f"Spacing median: {s['median']:.4f}")
    typer.echo(f"Spacing min:    {s['min']:.4f}")
    typer.echo(f"Spacing max:    {s['max']:.4f}")
    typer.echo(f"Spacing std:    {s['std']:.4f}")
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
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

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
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Remove statistical outliers from a point cloud."""
    try:
        result = filter_outliers(input_path, output, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Original: {result['original_points']} pts")
    typer.echo(f"Filtered: {result['filtered_points']} pts")
    typer.echo(f"Removed:  {result['removed_points']} pts")
    typer.echo(f"Saved:    {result['output']}")
    if output_json:
        _dump_json(result, output_json)


@app.command("sample")
def sample_cmd(
    input_path: str = typer.Argument(..., help="Input point cloud file"),
    output: str = typer.Option(..., "--output", "-o", help="Output file path"),
    num_points: int = typer.Option(..., "--num", "-n", help="Number of points to keep"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
) -> None:
    """Randomly sample a fixed number of points."""
    try:
        result = random_sample(input_path, output, num_points)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Original: {result['original_points']} pts")
    typer.echo(f"Sampled:  {result['sampled_points']} pts")
    typer.echo(f"Saved:    {result['output']}")
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
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    for step in result["steps"]:
        typer.echo(f"  Step {step['step']}: {step['path']} (fitness={step['fitness']:.4f}, rmse={step['rmse']:.4f})")
    typer.echo(f"Total:  {result['total_points']} pts")
    typer.echo(f"Saved:  {result['output']}")
    if output_json:
        _dump_json(result, output_json)


@app.command("version")
def version_cmd() -> None:
    """Show CloudAnalyzer version."""
    typer.echo("CloudAnalyzer v0.1.0")


def main():
    app()


if __name__ == "__main__":
    main()
