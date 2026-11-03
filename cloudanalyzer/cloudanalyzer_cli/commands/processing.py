"""Point-cloud processing and bridge inspection commands."""

from typing import List, Optional
import typer
from ca.downsample import downsample
from ca.merge import merge
from ca.convert import convert
from ca.crop import crop
from ca.normals import estimate_normals
from ca.filter import filter_outliers
from ca.sample import random_sample
from ca.align import align
from cloudanalyzer_cli._app import app, _dump_json, _run_evaluate, _handle_error


@app.command("web-view")
def web_view_cmd(
    paths: List[str] = typer.Argument(..., help="Files, or folders of results (their .ply, .pcd, .las/.laz, .tum ...)"),
    port: int = typer.Option(0, "--port", help="Serve on this port (0: any free port)"),
    app_url: str = typer.Option(
        "https://rsasaki0109.github.io/CloudAnalyzer/app/", "--app", help="The web app to open them in"
    ),
    no_browser: bool = typer.Option(False, "--no-browser", help="Print the link only"),
) -> None:
    """Open results in the CloudAnalyzer web app: serve them from this machine and open one link."""
    import webbrowser

    from ca.web_view import Viewer

    try:
        viewer = Viewer(paths, port=port, app=app_url)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)
    for f in viewer.files:
        typer.echo(f"serving {f}")
    typer.echo(viewer.link)
    if not no_browser:
        webbrowser.open(viewer.link)
    typer.echo("Serving on 127.0.0.1 only; Ctrl-C to stop.", err=True)
    try:
        viewer.serve_forever()
    except KeyboardInterrupt:
        viewer.close()


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


def _int_list(value: Optional[str]) -> Optional[list[int]]:
    return [int(v) for v in value.split(",")] if value else None


@app.command("bridge-sections")
def bridge_sections_cmd(
    input_path: str = typer.Argument(..., help="Bridge point cloud (LAS/LAZ, or any format ca reads)"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Directory for JSON, CSV table and SVG section"),
    spacing: float = typer.Option(1.0, "--spacing", help="Distance between sections along the axis (m)"),
    thickness: float = typer.Option(0.1, "--thickness", help="Section slab thickness (m)"),
    classes: Optional[str] = typer.Option(None, "--classes", help="Measure only these classification codes, e.g. 0,1,2,3"),
    axis_classes: Optional[str] = typer.Option(None, "--axis-classes", help="Codes whose points define the axis and deck ends, e.g. the deck"),
    axis: Optional[str] = typer.Option(None, "--axis", help="Explicit axis as x1,y1,x2,y2"),
) -> None:
    """Cut sections along a bridge and write a member dimension table.

    Values are measured from the point cloud only: visible depths are lower
    bounds and items without supporting returns are reported as unobserved.
    """
    from ca.bridge import bridge_sections

    try:
        ends = [float(v) for v in axis.split(",")] if axis else None
        if ends is not None and len(ends) != 4:
            raise ValueError("--axis needs x1,y1,x2,y2")
        result = bridge_sections(
            input_path, output_dir, spacing=spacing, thickness=thickness,
            classes=_int_list(classes), axis_classes=_int_list(axis_classes),
            axis=(ends[:2], ends[2:]) if ends else None,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    measured = sum(s["status"] == "measured" for s in result["sections"])
    typer.echo(f"Sections:  {measured}/{len(result['sections'])} measured")
    for row in result["table"]:
        value = "-" if row["value"] is None else f"{row['value']:.3f}"
        typer.echo(f"  {row['label_ja']}: {value} ({row['status']})")
    for name, path in result["outputs"].items():
        typer.echo(f"Saved {name}: {path}")


@app.command("bridge-plan")
def bridge_plan_cmd(
    input_path: str = typer.Argument(..., help="Bridge point cloud (LAS/LAZ, or any format ca reads)"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Directory for JSON, member CSV and SVG plan"),
    spacing: float = typer.Option(0.25, "--spacing", help="Distance between sections searched for girders (m)"),
    classes: Optional[str] = typer.Option(None, "--classes", help="Measure only these classification codes, e.g. 0,1,2,3"),
    axis_classes: Optional[str] = typer.Option(None, "--axis-classes", help="Codes whose points define the axis and deck ends, e.g. the deck"),
    member_classes: Optional[str] = typer.Option(None, "--member-classes", help="Codes searched for girders and cross beams, e.g. 1,2 without abutments"),
    axis: Optional[str] = typer.Option(None, "--axis", help="Explicit axis as x1,y1,x2,y2"),
    supports: Optional[str] = typer.Option(None, "--supports", help="Piers as distances from the start along the centreline (m), e.g. 18.5,37.0; detected when omitted"),
) -> None:
    """Draw a bridge plan with numbered deck, girder and cross-beam elements.

    Girders and cross beams are drawn only where returns beneath the deck show
    them; without such returns nothing below the deck is drawn or assumed.
    """
    from ca.bridge_plan import bridge_plan

    try:
        ends = [float(v) for v in axis.split(",")] if axis else None
        if ends is not None and len(ends) != 4:
            raise ValueError("--axis needs x1,y1,x2,y2")
        result = bridge_plan(
            input_path, output_dir, spacing=spacing, classes=_int_list(classes),
            axis_classes=_int_list(axis_classes), member_classes=_int_list(member_classes),
            axis=(ends[:2], ends[2:]) if ends else None,
            supports=[float(v) for v in supports.split(",")] if supports else None,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    deck = result["deck"]
    typer.echo(f"Deck:      {deck['length_m']:.3f} m x {deck['width_m']:.3f} m, skew {deck['skew_deg']:.1f} deg")
    typer.echo(f"Underside: {result['underside']}")
    typer.echo(f"Girders:   {len(result['girders'])}, cross beams: {len(result['cross_beams'])}, "
               f"elements: {len(result['elements'])}")
    lengths = ", ".join(f"{sp['length_m']:.2f} m" for sp in result["spans"])
    typer.echo(f"Spans:     {len(result['spans'])} ({lengths})")
    for warning in result["warnings"]:
        typer.echo(f"Warning:   {warning}")
    for name, path in result["outputs"].items():
        typer.echo(f"Saved {name}: {path}")


@app.command("bridge-damage")
def bridge_damage_cmd(
    input_path: str = typer.Argument(..., help="Bridge point cloud with colour (LAS/LAZ, or any format ca reads)"),
    output_dir: str = typer.Option(..., "--output-dir", "-o", help="Directory for JSON, CSV, soffit orthophoto and SVG"),
    classes: Optional[str] = typer.Option(None, "--classes", help="Codes measured for the plan, e.g. 0,1,2,3"),
    axis_classes: Optional[str] = typer.Option(None, "--axis-classes", help="Codes whose points define the axis and deck ends"),
    member_classes: Optional[str] = typer.Option(None, "--member-classes", help="Codes searched for girders and cross beams"),
    surface_classes: Optional[str] = typer.Option(None, "--surface-classes", help="Codes the soffit is taken from, e.g. 1,2"),
    supports: Optional[str] = typer.Option(None, "--supports", help="Piers as distances from the start (m)"),
) -> None:
    """Find damage candidates on a deck soffit and place them on the numbered plan.

    Candidates are colour or height departures from the surrounding soffit for an
    inspector to confirm; nothing is diagnosed.
    """
    from ca.bridge_damage import bridge_damage

    try:
        result = bridge_damage(
            input_path, output_dir, classes=_int_list(classes), axis_classes=_int_list(axis_classes),
            member_classes=_int_list(member_classes), surface_classes=_int_list(surface_classes),
            supports=[float(v) for v in supports.split(",")] if supports else None,
        )
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    typer.echo(f"Underside:  {result['underside']}")
    if "soffit" in result:
        typer.echo(f"Searched:   {result['soffit']['area_searched_m2']:.1f} m2 of soffit")
    kinds: dict[str, int] = {}
    for c in result["candidates"]:
        kinds[c["label_ja"]] = kinds.get(c["label_ja"], 0) + 1
    typer.echo(f"Candidates: {len(result['candidates'])}")
    for label, n in kinds.items():
        typer.echo(f"  {label}: {n}")
    typer.echo(f"Note:       {result['note']}")
    for name, path in result["outputs"].items():
        typer.echo(f"Saved {name}: {path}")


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
