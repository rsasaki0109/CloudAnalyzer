"""Quality gate, PR comment and baseline commands."""

import json
from dataclasses import replace
from pathlib import Path
from typing import List, Optional
import typer
from ca.core import (
    load_check_suite,
    render_check_scaffold,
    run_check_suite,
    summarize_baseline_evolution,
)
from ca.core.gate import GateMode
from ca.baseline_history import discover_history, list_baselines, rotate_history, save_baseline
from ca.pr_comment import build_pr_comment
from cloudanalyzer_cli._app import (
    app,
    _dump_json,
    _load_json_mapping,
    _handle_error,
    _print_check_suite_result,
    _print_baseline_evolution_result,
)


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
    warn_only: bool = typer.Option(
        False,
        "--warn-only",
        help="Never fail the process for gate failures; record failures in gate_summary.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Fail on fail/warn/soft_fail/skip/not_applicable/ungated outcomes.",
    ),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Run config-driven artifact / trajectory / run QA checks."""
    if format_json:
        import logging

        logging.getLogger("ca").setLevel(logging.ERROR)

    if warn_only and strict:
        typer.echo("Error: --warn-only and --strict are mutually exclusive", err=True)
        raise typer.Exit(code=2)

    gate_mode: GateMode = "strict" if strict else "warn_only" if warn_only else "default"

    try:
        suite = load_check_suite(config_path)
        if output_json is not None:
            suite = replace(suite, summary_output_json=str(Path(output_json).resolve()))
        result = run_check_suite(suite, gate_mode=gate_mode)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo("Hint: Check config and input file paths.", err=True)
        raise typer.Exit(code=3)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)
        return

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        _print_check_suite_result(result)
        if suite.summary_output_json:
            typer.echo(f"Summary JSON: {suite.summary_output_json}")

    gate_summary = result.get("gate_summary", {})
    exit_code = int(gate_summary.get("exit_code", 0) or 0)
    if exit_code:
        raise typer.Exit(code=exit_code)


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
