"""Version and label conversion commands."""

import json
import typer
from cloudanalyzer_cli._app import app, _handle_error


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
