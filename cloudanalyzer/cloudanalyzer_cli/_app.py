"""The ``ca`` Typer application, its sub-apps and helpers shared by command modules."""

import json
from pathlib import Path
from typing import Any, Optional

import typer

from ca.evaluate import evaluate, plot_f1_curve
from ca.protocol import (
    ProtocolValidationError,
    attach_evaluation_protocol,
    load_protocol,
)
from ca.io import SUPPORTED_EXTENSIONS
from ca.log import setup_logging

app = typer.Typer(
    name="ca",
    help="CloudAnalyzer - AI-friendly CLI tool for point cloud analysis.",
)

protocol_app = typer.Typer(
    name="protocol",
    help="Validate and inspect versioned evaluation protocol files.",
    no_args_is_help=True,
)
app.add_typer(protocol_app, name="protocol")


@protocol_app.command("validate")
def protocol_validate_cmd(
    protocol_path: str = typer.Argument(..., help="JSON/YAML protocol file to validate."),
    format_json: bool = typer.Option(False, "--format-json", help="Print normalized JSON."),
) -> None:
    """Validate a ``cloudanalyzer.protocol.v1`` document and show its identity."""

    try:
        document = load_protocol(protocol_path)
    except (FileNotFoundError, ProtocolValidationError, ValueError) as exc:
        _handle_error(exc)

    if format_json:
        typer.echo(json.dumps(document, indent=2, ensure_ascii=False))
        return

    typer.echo(f"Schema:  {document['schema_version']}")
    typer.echo(f"Name:    {document['name']}")
    typer.echo(f"Kind:    {document['kind']}")
    typer.echo(f"SHA256:  {document['protocol_sha256']}")
    typer.echo("Status:  VALID")


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
    missing = getattr(e, "filename", None)
    if isinstance(e, FileNotFoundError) and isinstance(missing, str) and Path(missing).name in {"run.json", "job.json"}:
        typer.echo(
            f"Hint: {Path(missing).parent} is not a mapping run directory. Pass the --out directory "
            "given to mapping-run; a relative path resolves from the current directory.",
            err=True,
        )
    elif isinstance(e, FileNotFoundError):
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


def _attach_protocol(result: dict, protocol_path: Optional[str], **kwargs: Any) -> dict:
    """Attach a protocol and turn validation failures into normal CLI errors."""

    try:
        return attach_evaluation_protocol(result, protocol_path, **kwargs)
    except (FileNotFoundError, ProtocolValidationError, ValueError) as exc:
        _handle_error(exc)
    return result


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


def _check_status_label(passed: bool | None, gate_status: str | None = None) -> str:
    """Render a compact status label for config-driven checks."""
    if gate_status in {"warn", "soft_fail", "skip", "not_applicable"}:
        return gate_status.upper()
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
        status = _check_status_label(item["passed"], item.get("gate_status"))
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
        elif item["kind"] == "image":
            psnr = summary.get("psnr_mean")
            psnr_text = f"{psnr:.4f}" if isinstance(psnr, (int, float)) else "inf"
            ssim = summary.get("ssim_mean")
            ssim_text = f"{ssim:.4f}" if isinstance(ssim, (int, float)) else "n/a"
            typer.echo(
                f"[{status}] {item['id']} ({item['kind']}): "
                f"pairs={summary['pairs_evaluated']}  "
                f"psnr={psnr_text}  "
                f"ssim={ssim_text}"
            )
        elif item["kind"] == "rendered":
            psnr = summary.get("psnr_mean")
            psnr_text = f"{psnr:.4f}" if isinstance(psnr, (int, float)) else "inf"
            ssim = summary.get("ssim_mean")
            ssim_text = f"{ssim:.4f}" if isinstance(ssim, (int, float)) else "n/a"
            geom = ""
            chamfer = summary.get("chamfer_distance")
            if isinstance(chamfer, (int, float)):
                geom = f"  chamfer={chamfer:.4f}"
            typer.echo(
                f"[{status}] {item['id']} ({item['kind']}): "
                f"pairs={summary['pairs_evaluated']}  "
                f"psnr={psnr_text}  "
                f"ssim={ssim_text}{geom}"
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
    gate_summary = result.get("gate_summary")
    if isinstance(gate_summary, dict):
        typer.echo(
            "Gate: "
            f"mode={gate_summary['mode']}  "
            f"blocking_fail={len(gate_summary['blocking_failed_ids'])}  "
            f"warn={gate_summary['warn_count']}  "
            f"soft_fail={gate_summary['soft_fail_count']}  "
            f"skip={gate_summary['skip_count']}  "
            f"not_applicable={gate_summary['not_applicable_count']}  "
            f"exit={gate_summary['exit_code']}"
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
