"""Geometry, ground, detection and tracking evaluation commands."""

import json
from typing import Optional
import typer
from ca.detection import evaluate_detection
from ca.ground_evaluate import evaluate_ground_segmentation
from ca.evaluate import plot_f1_curve
from ca.report import save_detection_report, save_ground_report, save_tracking_report
from ca.geometry import (
    DEFAULT_MESH_SAMPLES,
    DEFAULT_SPLAT_SAMPLES,
    MESH_SAMPLE_METHODS,
    REPRESENTATIONS,
    SPLAT_METHODS,
    evaluate_geometry,
)
from ca.tracking import evaluate_tracking
from cloudanalyzer_cli._app import (
    app,
    _dump_json,
    _handle_error,
    _attach_protocol,
    _parse_thresholds,
)


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
    protocol_path: Optional[str] = typer.Option(
        None,
        "--protocol",
        help="Versioned cloudanalyzer.protocol.v1 file to attach to the result.",
    ),
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

    _attach_protocol(
        result,
        protocol_path,
        command="ca geometry-evaluate",
        kind="geometry",
        inputs=(source, reference),
        options={
            "representation": representation,
            "opacity_threshold": opacity_threshold,
            "voxel_size_m": voxel,
            "mesh_samples": mesh_samples,
            "mesh_method": mesh_method,
            "splat_method": splat_method,
            "splat_samples": splat_samples,
            "thresholds_m": _parse_thresholds(thresholds),
        },
    )

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
