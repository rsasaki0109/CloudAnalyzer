"""Comparison, evaluation and SLAM debugging commands."""

import json
from pathlib import Path
from typing import Optional
import typer
from ca.compare import run_compare
from ca.scan_match_debug import run_scan_match_debug
from ca.slam_debug import analyze_slam_run, write_slam_debug_markdown
from ca.info import get_info
from ca.diff import run_diff
from cloudanalyzer_cli._app import (
    app,
    _dump_json,
    _handle_error,
    _attach_protocol,
    _parse_thresholds,
    _parse_matrix16,
)


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
    path: str = typer.Argument(..., help="Path to point cloud or ROS bag file (pcd/ply/las/laz/csv/bag/mcap/db3)"),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Show basic information about a point cloud or ROS bag file."""
    try:
        info = get_info(path)
    except (FileNotFoundError, ValueError) as e:
        _handle_error(e)

    if format_json:
        typer.echo(json.dumps(info, indent=2))
    elif info.get("kind") == "rosbag":
        typer.echo(f"File:      {info['path']}")
        typer.echo("Kind:      ROS bag / MCAP")
        typer.echo(f"Messages:  {info['message_count']}")
        typer.echo(f"Duration:  {info['duration_ns'] / 1e9:.3f} s")
        typer.echo("Topics:")
        for row in info["topics"]:
            typer.echo(f"  {row['topic']} ({row['type']}): {row['count']}")
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
    structure_voxel_size: float = typer.Option(
        3.0,
        "--structure-voxel-size",
        help="Voxel size in meters for AWD/SCS (MapEval uses 3.0 m). Set 0 to disable.",
    ),
    map_eval_min_voxel_points: int = typer.Option(
        100,
        "--map-eval-min-voxel-points",
        help="Minimum points in each voxel for AWD/SCS.",
    ),
    map_eval_neighbor_radius: int = typer.Option(
        5,
        "--map-eval-neighbor-radius",
        help="Voxel neighbor radius for SCS.",
    ),
    streaming: bool = typer.Option(
        False,
        "--streaming",
        help="Use chunked AWD/SCS evaluation and avoid materializing all points; NN metrics are omitted.",
    ),
    chunk_size: int = typer.Option(
        100_000,
        "--chunk-size",
        help="Points per input chunk when --streaming is enabled.",
    ),
    protocol_path: Optional[str] = typer.Option(
        None,
        "--protocol",
        help="Versioned cloudanalyzer.protocol.v1 file to attach to the result.",
    ),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Dump full result as JSON"),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Experimental MapEval-inspired map-to-map evaluation (GT-based)."""
    try:
        import numpy as np

        from ca.io import load_point_cloud
        from ca.core.map_evaluate import (
            MapEvalProtocol,
            MapEvaluateRequest,
            NNThresholdMapEvaluateStrategy,
            evaluate_map_streaming,
        )
    except Exception as e:
        _handle_error(e)

    t_list = _parse_thresholds(thresholds)
    matrix16 = _parse_matrix16(initial_matrix)
    init_4x4 = None
    if matrix16 is not None:
        init_4x4 = np.array(matrix16, dtype=np.float64).reshape(4, 4)

    accuracy_levels = (
        tuple(t_list) if t_list is not None else (0.2, 0.1, 0.08, 0.05, 0.01)
    )
    protocol = (
        MapEvalProtocol(
            voxel_size_m=structure_voxel_size,
            min_voxel_points=map_eval_min_voxel_points,
            neighbor_radius=map_eval_neighbor_radius,
            accuracy_level_m=accuracy_levels,
        )
        if structure_voxel_size > 0
        else None
    )

    try:
        if streaming:
            if protocol is None:
                raise ValueError("--streaming requires --structure-voxel-size > 0")
            if align_mode != "none" or matrix16 is not None:
                raise ValueError(
                    "--streaming currently supports only --align-mode none; "
                    "alignment must be applied before chunked evaluation"
                )
            if artifact_dir is not None:
                raise ValueError(
                    "--artifact-dir is not supported with --streaming; "
                    "streaming mode emits metrics/provenance only"
                )
            from ca.io import PointChunkReader

            result = evaluate_map_streaming(
                PointChunkReader(estimated, chunk_size=chunk_size),
                PointChunkReader(reference, chunk_size=chunk_size),
                protocol=protocol,
            )
        else:
            est_pcd = load_point_cloud(estimated)
            ref_pcd = load_point_cloud(reference)
            req = MapEvaluateRequest(
                estimated_points=np.asarray(est_pcd.points),
                reference_points=np.asarray(ref_pcd.points),
                thresholds_m=accuracy_levels,
                align_mode=align_mode,
                initial_transform_4x4=init_4x4,
                artifact_dir=artifact_dir,
                structure_voxel_size=structure_voxel_size,
                map_eval_protocol=protocol,
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
        "metric_family": result.metric_family,
        "reference_required": result.reference_required,
        "mode": result.mode,
        "sampling_policy": result.sampling_policy,
    }
    _attach_protocol(
        payload,
        protocol_path,
        command="ca map-evaluate",
        kind="map",
        inputs=(estimated, reference),
        options={
            "thresholds_m": list(accuracy_levels),
            "align_mode": align_mode,
            "structure_voxel_size_m": structure_voxel_size,
            "map_eval_min_voxel_points": map_eval_min_voxel_points,
            "map_eval_neighbor_radius": map_eval_neighbor_radius,
            "streaming": streaming,
            "chunk_size": chunk_size,
        },
    )

    if format_json:
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        t0 = result.artifacts.get("thresholds_m", [0.2])[0]
        typer.echo(f"Estimated:  {estimated}")
        typer.echo(f"Reference:  {reference}")
        typer.echo(f"Align:      {result.artifacts.get('align_mode')}")
        if result.metrics.get("chamfer_m") is not None:
            typer.echo(f"Chamfer:    {result.metrics['chamfer_m']:.6f} m")
        if "awd_m" in result.metrics:
            typer.echo(f"AWD:        {result.metrics['awd_m']:.6f} m")
            typer.echo(f"SCS:        {result.metrics['scs']:.6f}")
        fscore = result.metrics.get(f"fscore@{t0:.3f}m")
        accuracy = result.metrics.get(f"accuracy@{t0:.3f}m")
        completeness = result.metrics.get(f"completeness@{t0:.3f}m")
        if fscore is not None:
            typer.echo(f"F-score:    {fscore:.6f} @ {t0:.3f} m")
            typer.echo(f"Accuracy:   {accuracy:.6f} @ {t0:.3f} m")
            typer.echo(f"Complete:   {completeness:.6f} @ {t0:.3f} m")
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
        help=(
            "Comma-separated metrics: psnr, ssim, lpips, dreamsim_distance, "
            "frequency_consistency (learned metrics need an optional extra)."
        ),
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
    protocol_path: Optional[str] = typer.Option(
        None,
        "--protocol",
        help="Versioned cloudanalyzer.protocol.v1 file to attach to the result.",
    ),
    format_json: bool = typer.Option(
        False, "--format-json", help="Print the result as JSON to stdout."
    ),
) -> None:
    """Score rendered images against references, including 3DGS artifact FCM.

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
    _attach_protocol(
        payload,
        protocol_path,
        command="ca image-evaluate",
        kind="image",
        inputs=(rendered_dir, reference_dir),
        options={
            "metrics": list(metric_tuple),
            "extensions": list(ext_tuple),
            "ssim_window_size": ssim_window,
            "ssim_sigma": ssim_sigma,
            "max_pairs": max_pairs,
        },
    )

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


@app.command("rendered-evaluate")
def rendered_evaluate_cmd(
    splat_path: str = typer.Argument(..., help="3D Gaussian Splatting PLY to render."),
    reference_dir: str = typer.Argument(
        ...,
        help="Directory of reference (ground-truth) images. Pairs match by filename.",
    ),
    cameras: str = typer.Option(
        ...,
        "--cameras",
        help="Camera bundle: transforms.json, COLMAP images.txt, or a directory containing either.",
    ),
    reference_pointcloud: Optional[str] = typer.Option(
        None,
        "--reference-pointcloud",
        help="Optional reference scan for combined geometry QA (Chamfer / AUC / F1).",
    ),
    metrics: str = typer.Option(
        "psnr,ssim",
        "--metrics",
        help=(
            "Comma-separated metrics: psnr, ssim, lpips, dreamsim_distance, "
            "frequency_consistency."
        ),
    ),
    opacity_threshold: Optional[float] = typer.Option(
        None,
        "--opacity-threshold",
        help="Drop splats below this rendered alpha before rasterization.",
    ),
    geometry_opacity_threshold: Optional[float] = typer.Option(
        None,
        "--geometry-opacity-threshold",
        help="Opacity filter for the optional geometry pass (defaults to --opacity-threshold).",
    ),
    geometry_voxel: Optional[float] = typer.Option(
        None,
        "--geometry-voxel",
        help="Voxel-downsample the adapted source before geometry QA.",
    ),
    geometry_splat_method: str = typer.Option(
        "centers",
        "--geometry-splat-method",
        help="3DGS geometry adapter: centers or ellipsoid.",
    ),
    geometry_splat_samples: int = typer.Option(
        8,
        "--geometry-splat-samples",
        help="Samples per splat when --geometry-splat-method=ellipsoid.",
    ),
    render_device: Optional[str] = typer.Option(
        None,
        "--render-device",
        help="Torch device for gsplat rendering (default: cuda when available).",
    ),
    rendered_dir: Optional[str] = typer.Option(
        None,
        "--rendered-dir",
        help="Keep rendered PNGs in this directory instead of a temp folder.",
    ),
    max_pairs: Optional[int] = typer.Option(
        None,
        "--max-pairs",
        help="Cap on number of photometric pairs evaluated.",
    ),
    ssim_window_size: int = typer.Option(
        11,
        "--ssim-window-size",
        help="SSIM Gaussian window side length.",
    ),
    ssim_sigma: float = typer.Option(
        1.5,
        "--ssim-sigma",
        help="SSIM Gaussian window sigma.",
    ),
    report: Optional[str] = typer.Option(
        None,
        "--report",
        help="Write a combined HTML/Markdown report to this path.",
    ),
    output_json: Optional[str] = typer.Option(
        None, "--output-json", help="Write the full result (photometric + geometry) as JSON."
    ),
    protocol_path: Optional[str] = typer.Option(
        None,
        "--protocol",
        help="Versioned cloudanalyzer.protocol.v1 file to attach to the result.",
    ),
    format_json: bool = typer.Option(
        False, "--format-json", help="Print the result as JSON to stdout."
    ),
) -> None:
    """Render a 3DGS PLY and score photometric + optional geometry QA.

    Requires ``pip install 'cloudanalyzer[gs]'`` for gsplat rendering and
    optional LPIPS. Camera poses come from nerfstudio ``transforms.json`` or
    COLMAP ``cameras.txt`` + ``images.txt``.

    Example::

        ca rendered-evaluate scene.ply references/ \\
            --cameras transforms.json \\
            --reference-pointcloud reference.pcd \\
            --metrics psnr,ssim,lpips --report report.html
    """

    try:
        from ca.core.rendered_evaluate import (
            RenderedEvalRequest,
            rendered_evaluate,
            rendered_evaluate_to_dict,
        )
    except ImportError as exc:
        _handle_error(exc)

    metric_tuple = tuple(m.strip() for m in metrics.split(",") if m.strip())
    if not metric_tuple:
        typer.echo("Error: --metrics cannot be empty.", err=True)
        raise typer.Exit(code=2)

    geom_opacity = geometry_opacity_threshold
    if geom_opacity is None:
        geom_opacity = opacity_threshold

    try:
        result = rendered_evaluate(
            RenderedEvalRequest(
                splat_path=Path(splat_path),
                cameras_path=Path(cameras),
                reference_dir=Path(reference_dir),
                metrics=metric_tuple,
                reference_pointcloud=(
                    Path(reference_pointcloud) if reference_pointcloud else None
                ),
                opacity_threshold=opacity_threshold,
                geometry_opacity_threshold=geom_opacity,
                geometry_voxel=geometry_voxel,
                geometry_splat_method=geometry_splat_method,
                geometry_splat_samples=geometry_splat_samples,
                render_device=render_device,
                keep_rendered_dir=Path(rendered_dir) if rendered_dir else None,
                max_pairs=max_pairs,
                ssim_window_size=ssim_window_size,
                ssim_sigma=ssim_sigma,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        _handle_error(exc)

    payload = rendered_evaluate_to_dict(result)
    _attach_protocol(
        payload,
        protocol_path,
        command="ca rendered-evaluate",
        kind="rendered",
        inputs=(splat_path, cameras, reference_dir, reference_pointcloud),
        options={
            "metrics": list(metric_tuple),
            "opacity_threshold": opacity_threshold,
            "geometry_opacity_threshold": geom_opacity,
            "geometry_voxel": geometry_voxel,
            "geometry_splat_method": geometry_splat_method,
            "geometry_splat_samples": geometry_splat_samples,
            "render_device": render_device,
            "max_pairs": max_pairs,
            "ssim_window_size": ssim_window_size,
            "ssim_sigma": ssim_sigma,
        },
    )

    if report:
        from ca.report import save_rendered_report

        save_rendered_report(payload, report)

    if format_json:
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        s = result.photometric["summary"]
        typer.echo(
            f"rendered-evaluate: {s['pairs_evaluated']} pair(s) scored "
            f"({s['pairs_missing_in_reference']} missing in reference, "
            f"{s['pairs_size_mismatch']} size-mismatch)"
        )
        typer.echo(
            f"  renderer: {result.renderer.get('backend')} "
            f"frames={result.renderer.get('frames_rendered')} "
            f"dir={result.renderer.get('rendered_dir')}"
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
        if result.geometry is not None:
            typer.echo(
                f"  geometry Chamfer={result.geometry.get('chamfer_distance', 0):.4f} "
                f"AUC={result.geometry.get('auc', 0):.4f}"
            )

    if output_json:
        _dump_json(payload, output_json)
