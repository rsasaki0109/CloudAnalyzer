"""COPC tiling, pipeline and web viewer commands."""

import json
from typing import List, Optional
import typer
from ca.split import split
from ca.pipeline import run_pipeline
from ca.plot import heatmap3d
from ca.web import export_static_bundle as web_export_static_bundle
from ca.web import serve as web_serve
from cloudanalyzer_cli._app import app, _dump_json, _handle_error, _parse_thresholds


@app.command("copc-tile")
def copc_tile_cmd(
    source: str = typer.Argument(..., help="Local or HTTP(S) COPC source"),
    out: str = typer.Option(..., "--out", help="New job directory, or existing job with --resume"),
    grid: float = typer.Option(..., "--grid", help="XY tile width in source coordinate units"),
    halo: float = typer.Option(0, "--halo", help="Neighbor support width in source units"),
    box: Optional[str] = typer.Option(None, "--box", help="Inclusive original-coordinate xmin,ymin,zmin,xmax,ymax,zmax"),
    origin: str = typer.Option("0,0", "--origin", help="XY grid origin in source coordinates"),
    chunk_size: int = typer.Option(10000, "--chunk-size", help="Requested points per batch; frames also have an 8 MiB cap"),
    resume: bool = typer.Option(False, "--resume", help="Verify source/options and continue committed prefix"),
    stop_after_nodes: Optional[int] = typer.Option(None, "--stop-after-nodes", help="Commit this many new nodes then pause"),
    max_node_output_mib: int = typer.Option(256, "--max-node-output-mib", help="Per-node output quota; may be increased on resume"),
    max_fragments_per_node: int = typer.Option(65536, "--max-fragments-per-node", help="Per-node fragment quota; may be increased on resume"),
) -> None:
    """Create bounded full-density tile packs with halo and durable checkpoints."""
    from ca.copc_tiles import tile_copc
    from sqlite3 import Error as SqliteError

    try:
        report = tile_copc(source, out, grid, halo=halo,
                           bounds=[float(v) for v in box.split(",")] if box else None,
                           origin=[float(v) for v in origin.split(",")], chunk_size=chunk_size,
                           resume=resume, stop_after_nodes=stop_after_nodes,
                           max_node_output_bytes=max_node_output_mib << 20,
                           max_fragments_per_node=max_fragments_per_node)
    except (OSError, ValueError, RuntimeError, SqliteError) as error:
        _handle_error(error)
        return
    typer.echo(json.dumps(report, indent=2))


@app.command("copc-tile-export")
def copc_tile_export_cmd(
    job: str = typer.Argument(..., help="COPC tile job directory"),
    i: int = typer.Option(..., "--i", help="Tile X grid index"),
    j: int = typer.Option(..., "--j", help="Tile Y grid index"),
    out: str = typer.Option(..., "--out", help="New .las/.laz artifact"),
    include_halo: bool = typer.Option(False, "--include-halo", help="Export halo with explicit source identity/halo dimensions"),
) -> None:
    """Stream one committed tile to LAS/LAZ with original attributes and CRS."""
    from ca.copc_tiles import export_copc_tile
    from sqlite3 import Error as SqliteError

    try:
        report = export_copc_tile(job, i, j, out, include_halo=include_halo)
    except (OSError, ValueError, RuntimeError, SqliteError) as error:
        _handle_error(error)
        return
    typer.echo(json.dumps(report, indent=2))


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
