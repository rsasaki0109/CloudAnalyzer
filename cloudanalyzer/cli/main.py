"""CloudAnalyzer CLI entry point."""

from typing import Optional

import typer

from ca.compare import run_compare
from ca.info import get_info
from ca.diff import run_diff

app = typer.Typer(
    name="ca",
    help="CloudAnalyzer - AI-friendly CLI tool for point cloud analysis.",
)


@app.command("compare")
def compare_cmd(
    source: str = typer.Argument(..., help="Path to source point cloud (pcd/ply/las)"),
    target: str = typer.Argument(..., help="Path to target point cloud (pcd/ply/las)"),
    method: Optional[str] = typer.Option(
        "gicp",
        "--register",
        help="Registration method: icp, gicp, or 'none' to skip",
    ),
    json_out: Optional[str] = typer.Option(
        None,
        "--json",
        help="Output path for JSON report",
    ),
    report: Optional[str] = typer.Option(
        None,
        "--report",
        help="Output path for Markdown report",
    ),
    snapshot: Optional[str] = typer.Option(
        None,
        "--snapshot",
        help="Output path for snapshot image (png)",
    ),
) -> None:
    """Compare two point clouds with optional registration."""
    reg_method = method if method and method.lower() != "none" else None

    try:
        run_compare(
            source_path=source,
            target_path=target,
            method=reg_method,
            json_path=json_out,
            report_path=report,
            snapshot_path=snapshot,
        )
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("info")
def info_cmd(
    path: str = typer.Argument(..., help="Path to point cloud file (pcd/ply/las)"),
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


@app.command("diff")
def diff_cmd(
    source: str = typer.Argument(..., help="Path to source point cloud"),
    target: str = typer.Argument(..., help="Path to target point cloud"),
) -> None:
    """Quick distance stats between two point clouds (no registration)."""
    try:
        result = run_diff(source, target)
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


@app.command("version")
def version_cmd() -> None:
    """Show CloudAnalyzer version."""
    typer.echo("CloudAnalyzer v0.1.0")


def main():
    app()


if __name__ == "__main__":
    main()
