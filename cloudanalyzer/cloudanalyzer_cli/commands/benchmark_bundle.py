"""Benchmark, bundle, leaderboard and history commands."""

import json
import zipfile
from pathlib import Path
from typing import List, Optional
import typer
from ca.report import save_run_report
from ca.benchmark import (
    BenchmarkSuite,
    GATE_KEYS,
    evaluate_benchmark_run,
    load_benchmark_suite,
    materialize_suite,
    write_benchmark_report_bundle,
)
from ca.bundle import diff_bundles, pack_bundle, render_diff_markdown, show_bundle, unpack_bundle
from ca.history import (
    build_history_series,
    discover_bundles,
    render_history_json,
    render_history_markdown,
)
from ca.leaderboard import build_leaderboard_from_bundles
from cloudanalyzer_cli._app import (
    app,
    _dump_json,
    _handle_error,
    _attach_protocol,
    _parse_thresholds,
)


benchmark_app = typer.Typer(
    name="benchmark",
    help="SLAM benchmark suite runner (fixed reference + gate).",
    no_args_is_help=True,
)
app.add_typer(benchmark_app, name="benchmark")


bundle_app = typer.Typer(
    name="bundle",
    help="Pack / unpack / inspect CloudAnalyzer QA result bundles.",
    no_args_is_help=True,
)
app.add_typer(bundle_app, name="bundle")


leaderboard_app = typer.Typer(
    name="leaderboard",
    help="Build static leaderboard sites from benchmark report bundles.",
    no_args_is_help=True,
)
app.add_typer(leaderboard_app, name="leaderboard")


def _parse_notes(values: Optional[List[str]]) -> dict[str, str]:
    notes: dict[str, str] = {}
    if not values:
        return notes
    for raw in values:
        if "=" not in raw:
            typer.echo(
                f"Error: --note expects key=value; got {raw!r}",
                err=True,
            )
            raise typer.Exit(code=1)
        key, _, value = raw.partition("=")
        key = key.strip()
        if not key:
            typer.echo("Error: --note key cannot be empty", err=True)
            raise typer.Exit(code=1)
        notes[key] = value.strip()
    return notes


@leaderboard_app.command("build")
def leaderboard_build_cmd(
    bundle_dirs: List[str] = typer.Argument(
        ...,
        help="One or more `ca benchmark eval --out` report bundle directories",
    ),
    out: str = typer.Option(
        ...,
        "--out",
        help="Output directory for static leaderboard site",
    ),
    title: str = typer.Option(
        "CloudAnalyzer Benchmark Leaderboard",
        "--title",
        help="Leaderboard title written into results.json and index.html",
    ),
    format_json: bool = typer.Option(False, "--format-json", help="Print results JSON to stdout"),
) -> None:
    """Build a static leaderboard from benchmark report bundle directories."""
    try:
        payload = build_leaderboard_from_bundles(bundle_dirs, out, title=title)
    except (FileNotFoundError, ValueError) as exc:
        _handle_error(exc)

    if format_json:
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(f"Leaderboard: {out}")
        typer.echo(f"Rows: {len(payload['rows'])}")
        typer.echo(f"Warnings: {len(payload['warnings'])}")
        typer.echo(f"Errors: {len(payload['errors'])}")
        typer.echo(f"HTML: {Path(out) / 'index.html'}")
        typer.echo(f"JSON: {Path(out) / 'results.json'}")

    if payload["errors"]:
        raise typer.Exit(code=1)


@bundle_app.command("pack")
def bundle_pack_cmd(
    summary_path: str = typer.Argument(..., help="Path to summary JSON (ca check / run-evaluate / benchmark eval)"),
    output: str = typer.Option(..., "--output", "-o", help="Output bundle ZIP path"),
    baseline: Optional[str] = typer.Option(None, "--baseline", help="Optional baseline summary JSON of the same shape"),
    project: Optional[str] = typer.Option(None, "--project", help="Project label written into bundle metadata"),
    commit: Optional[str] = typer.Option(None, "--commit", help="Git commit SHA to record in metadata"),
    pr_number: Optional[str] = typer.Option(None, "--pr-number", help="PR number to record in metadata"),
    runner_id: Optional[str] = typer.Option(None, "--runner-id", help="CI runner identifier to record"),
    note: Optional[List[str]] = typer.Option(
        None,
        "--note",
        help="Extra metadata as key=value (repeatable), e.g. --note dataset=newer-college",
    ),
) -> None:
    """Bundle a QA summary plus referenced reports into a single qa_bundle.zip."""
    try:
        notes = _parse_notes(note)
        metadata = pack_bundle(
            summary_path,
            output,
            baseline_path=baseline,
            project=project,
            git_commit=commit,
            pr_number=pr_number,
            runner_id=runner_id,
            notes=notes,
        )
    except (FileNotFoundError, ValueError) as exc:
        _handle_error(exc)

    typer.echo(f"Bundle:   {output}")
    typer.echo(f"Kind:     {metadata.summary_kind}")
    typer.echo(f"Artifacts: {len(metadata.artifacts)} report file(s) included")
    if metadata.has_baseline:
        typer.echo("Baseline:  included")
    if metadata.project:
        typer.echo(f"Project:   {metadata.project}")
    if metadata.git_commit:
        typer.echo(f"Commit:    {metadata.git_commit}")
    if metadata.pr_number:
        typer.echo(f"PR:        {metadata.pr_number}")
    if metadata.runner_id:
        typer.echo(f"Runner:    {metadata.runner_id}")
    if metadata.notes:
        typer.echo("Notes:")
        for key, value in metadata.notes.items():
            typer.echo(f"  {key}={value}")


@bundle_app.command("unpack")
def bundle_unpack_cmd(
    bundle_path: str = typer.Argument(..., help="Path to a CloudAnalyzer qa_bundle.zip"),
    output_dir: str = typer.Option(..., "--output", "-o", help="Directory to extract into"),
) -> None:
    """Extract a bundle to a directory."""
    try:
        metadata = unpack_bundle(bundle_path, output_dir)
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        _handle_error(exc)
    typer.echo(f"Extracted: {output_dir}")
    typer.echo(f"Kind:      {metadata.summary_kind}")
    typer.echo(f"Artifacts: {len(metadata.artifacts)} report file(s) restored")


@bundle_app.command("show")
def bundle_show_cmd(
    bundle_path: str = typer.Argument(..., help="Path to a CloudAnalyzer qa_bundle.zip"),
    format_json: bool = typer.Option(False, "--format-json", help="Print as JSON"),
) -> None:
    """Show bundle metadata and table of contents without extracting."""
    try:
        info = show_bundle(bundle_path)
    except (FileNotFoundError, ValueError) as exc:
        _handle_error(exc)

    if format_json:
        typer.echo(json.dumps(info, indent=2))
        return

    metadata = info["metadata"]
    typer.echo(f"Bundle:               {info['bundle_path']}")
    typer.echo(f"Bundle version:       {metadata.get('bundle_version')}")
    typer.echo(f"Created at:           {metadata.get('created_at')}")
    typer.echo(f"CloudAnalyzer:        {metadata.get('cloudanalyzer_version')}")
    typer.echo(f"Summary kind:         {metadata.get('summary_kind')}")
    if metadata.get("project"):
        typer.echo(f"Project:              {metadata['project']}")
    if metadata.get("git_commit"):
        typer.echo(f"Git commit:           {metadata['git_commit']}")
    if metadata.get("pr_number"):
        typer.echo(f"PR number:            {metadata['pr_number']}")
    if metadata.get("runner_id"):
        typer.echo(f"Runner id:            {metadata['runner_id']}")
    if metadata.get("has_baseline"):
        typer.echo("Baseline:             included")
    if metadata.get("notes"):
        typer.echo("Notes:")
        for key, value in metadata["notes"].items():
            typer.echo(f"  {key}={value}")
    typer.echo("")
    typer.echo("Contents:")
    for entry in info["contents"]:
        typer.echo(
            f"  {entry['path']}  ({entry['size_bytes']} B / {entry['compressed_bytes']} B compressed)"
        )


@bundle_app.command("diff")
def bundle_diff_cmd(
    old_bundle: str = typer.Argument(..., help="Path to the older qa_bundle.zip"),
    new_bundle: str = typer.Argument(..., help="Path to the newer qa_bundle.zip"),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the diff Markdown to a file instead of stdout",
    ),
    format_json: bool = typer.Option(
        False,
        "--format-json",
        help="Print the structured diff dict as JSON (suitable for dashboards / scripting)",
    ),
) -> None:
    """Compare two QA bundles and render a Markdown report.

    Reuses the same metric layout as `ca report-pr-comment`, so the diff
    looks identical to what PRs already render — old becomes the baseline,
    new is the "current" set of numbers.
    """
    try:
        diff = diff_bundles(old_bundle, new_bundle)
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        _handle_error(exc)

    if format_json:
        typer.echo(json.dumps(diff, indent=2))
        return

    markdown = render_diff_markdown(diff)
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        typer.echo(f"Diff written: {output}")
    else:
        typer.echo(markdown, nl=False)


@app.command("history")
def history_cmd(
    bundles: List[str] = typer.Argument(
        None,
        help="Paths to one or more qa_bundle.zip archives (oldest-to-newest is determined "
        "from metadata.created_at, not the argument order).",
    ),
    from_dir: Optional[str] = typer.Option(
        None,
        "--from-dir",
        help="Discover bundles in this directory (matches `*.zip` by default).",
    ),
    pattern: str = typer.Option(
        "*.zip",
        "--pattern",
        help="Glob pattern used with --from-dir (default: *.zip)",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the Markdown history to a file instead of stdout.",
    ),
    format_json: bool = typer.Option(
        False,
        "--format-json",
        help="Emit the structured history payload as JSON (suitable for dashboards / scripting).",
    ),
) -> None:
    """Build a time-series view of QA gate metrics across many bundles.

    Reads N `qa_bundle.zip` archives, sorts them by their metadata's
    `created_at` stamp, and renders a per-metric trend table:

    - `check_suite` bundles: one table per `check_id` with that check's
      relevant metrics across the timeline.
    - `single_run` (`ca run-evaluate` / `ca benchmark eval`) bundles: one
      table covering map AUC / Chamfer / F1 + trajectory ATE / RPE /
      Drift / Coverage.

    Mixing bundle shapes across the input set is rejected.
    """
    paths: list[str] = list(bundles or [])
    if from_dir:
        try:
            discovered = discover_bundles(from_dir, pattern=pattern)
        except NotADirectoryError as exc:
            _handle_error(exc)
        paths.extend(str(p) for p in discovered)
    if not paths:
        typer.echo(
            "Error: provide at least one bundle path or use --from-dir.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        entries = build_history_series(paths)
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        _handle_error(exc)

    if format_json:
        typer.echo(json.dumps(render_history_json(entries), indent=2))
        return

    markdown = render_history_markdown(entries)
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        typer.echo(f"History written: {output}")
    else:
        typer.echo(markdown, nl=False)


def _parse_gate_overrides(values: Optional[List[str]]) -> dict[str, float | None]:
    """Parse repeated --gate key=value (or key=none) overrides."""
    overrides: dict[str, float | None] = {}
    if not values:
        return overrides
    for raw in values:
        if "=" not in raw:
            typer.echo(
                f"Error: --gate expects key=value; got {raw!r}",
                err=True,
            )
            raise typer.Exit(code=1)
        key, _, value = raw.partition("=")
        key = key.strip()
        value = value.strip()
        if key not in GATE_KEYS:
            typer.echo(
                f"Error: unknown gate key {key!r}. Allowed: {', '.join(GATE_KEYS)}",
                err=True,
            )
            raise typer.Exit(code=1)
        if value.lower() in {"none", "null", ""}:
            overrides[key] = None
        else:
            try:
                overrides[key] = float(value)
            except ValueError:
                typer.echo(
                    f"Error: --gate {key} must be numeric or 'none'; got {value!r}",
                    err=True,
                )
                raise typer.Exit(code=1)
    return overrides


def _print_suite_info(suite: BenchmarkSuite) -> None:
    typer.echo(f"Suite:       {suite.name} (v{suite.version})")
    typer.echo(f"Description: {suite.description}")
    if suite.license:
        typer.echo(f"License:     {suite.license}")
    if suite.dataset:
        typer.echo(f"Dataset:     {json.dumps(suite.dataset, sort_keys=True)}")
    typer.echo(f"Source:      {suite.source_path}")
    typer.echo("Sequences:")
    for name, seq in suite.sequences.items():
        typer.echo(f"  - {name}: {seq.description}")
        typer.echo(f"      reference_map:        {seq.reference_map_path}")
        typer.echo(f"      reference_trajectory: {seq.reference_trajectory_path}")
        if seq.sample_map_path or seq.sample_trajectory_path:
            typer.echo("      sample_outputs:")
            if seq.sample_map_path:
                typer.echo(f"        map:        {seq.sample_map_path}")
            if seq.sample_trajectory_path:
                typer.echo(f"        trajectory: {seq.sample_trajectory_path}")
    if suite.gate:
        typer.echo("Gate:")
        for key in GATE_KEYS:
            if key in suite.gate:
                typer.echo(f"  {key}: {suite.gate[key]}")


@benchmark_app.command("info")
def benchmark_info_cmd(
    suite_path: str = typer.Argument(..., help="Path to a benchmark suite YAML manifest"),
    format_json: bool = typer.Option(False, "--format-json", help="Print suite metadata as JSON"),
) -> None:
    """Show the sequences, references, and gate for a benchmark suite."""
    try:
        suite = load_benchmark_suite(suite_path)
    except (FileNotFoundError, ValueError) as exc:
        _handle_error(exc)

    if format_json:
        payload = {
            "name": suite.name,
            "version": suite.version,
            "description": suite.description,
            "license": suite.license,
            "dataset": dict(suite.dataset),
            "source_path": str(suite.source_path),
            "sequences": {
                name: {
                    "description": seq.description,
                    "reference_map": str(seq.reference_map_path),
                    "reference_trajectory": str(seq.reference_trajectory_path),
                    "sample_map": (
                        str(seq.sample_map_path) if seq.sample_map_path else None
                    ),
                    "sample_trajectory": (
                        str(seq.sample_trajectory_path)
                        if seq.sample_trajectory_path
                        else None
                    ),
                }
                for name, seq in suite.sequences.items()
            },
            "gate": dict(suite.gate),
        }
        typer.echo(json.dumps(payload, indent=2))
    else:
        _print_suite_info(suite)


@benchmark_app.command("eval")
def benchmark_eval_cmd(
    suite_path: str = typer.Argument(..., help="Path to a benchmark suite YAML manifest"),
    map_path: str = typer.Option(..., "--map", help="Estimated map point cloud"),
    trajectory_path: str = typer.Option(..., "--trajectory", help="Estimated trajectory"),
    sequence: Optional[str] = typer.Option(
        None,
        "--sequence",
        help="Sequence name to evaluate against (defaults to the first sequence)",
    ),
    thresholds: Optional[str] = typer.Option(
        None,
        "--thresholds",
        "-t",
        help="Comma-separated distance thresholds for map F1/AUC evaluation",
    ),
    max_time_delta: float = typer.Option(
        0.05,
        "--max-time-delta",
        help="Max timestamp gap allowed for trajectory matching (seconds)",
    ),
    align_origin: bool = typer.Option(False, "--align-origin"),
    align_rigid: bool = typer.Option(False, "--align-rigid"),
    gate_overrides: Optional[List[str]] = typer.Option(
        None,
        "--gate",
        help="Override suite gate (repeatable): --gate min_auc=0.97 --gate max_rpe=none",
    ),
    report: Optional[str] = typer.Option(None, "--report", help="Write combined run report"),
    out: Optional[str] = typer.Option(
        None,
        "--out",
        help="Write standard report bundle directory (metrics.json, summary.md, report.html, manifest.lock.yaml, provenance.json)",
    ),
    output_json: Optional[str] = typer.Option(
        None,
        "--output-json",
        help="Dump benchmark result as JSON",
    ),
    protocol_path: Optional[str] = typer.Option(
        None,
        "--protocol",
        help="Versioned cloudanalyzer.protocol.v1 file to attach to the result.",
    ),
    format_json: bool = typer.Option(False, "--format-json", help="Print JSON to stdout"),
) -> None:
    """Evaluate a SLAM run against a benchmark suite's fixed reference and gate."""
    try:
        suite = load_benchmark_suite(suite_path)
        overrides = _parse_gate_overrides(gate_overrides)
        result = evaluate_benchmark_run(
            suite,
            map_path,
            trajectory_path,
            sequence=sequence,
            gate_overrides=overrides,
            thresholds=_parse_thresholds(thresholds),
            max_time_delta=max_time_delta,
            align_origin=align_origin,
            align_rigid=align_rigid,
        )
    except (FileNotFoundError, ValueError) as exc:
        _handle_error(exc)

    benchmark_reference = result.get("benchmark", {}).get("reference", {})
    _attach_protocol(
        result,
        protocol_path,
        command="ca benchmark eval",
        kind="run",
        inputs=(
            suite_path,
            map_path,
            trajectory_path,
            benchmark_reference.get("map"),
            benchmark_reference.get("trajectory"),
        ),
        options={
            "sequence": sequence,
            "thresholds_m": _parse_thresholds(thresholds),
            "max_time_delta_s": max_time_delta,
            "align_origin": align_origin,
            "align_rigid": align_rigid,
            "gate_overrides": overrides,
        },
    )

    if report:
        try:
            save_run_report(result, report)
        except ValueError as exc:
            _handle_error(exc)

    bundle_paths: dict[str, str] | None = None
    if out:
        try:
            bundle_paths = write_benchmark_report_bundle(
                result,
                suite,
                out,
                map_path=map_path,
                trajectory_path=trajectory_path,
                sequence=sequence,
            )
        except (FileNotFoundError, ValueError) as exc:
            _handle_error(exc)

    if format_json:
        typer.echo(json.dumps(result, indent=2))
    else:
        info = result["benchmark"]
        map_result = result["map"]
        trajectory_result = result["trajectory"]
        typer.echo(
            f"Benchmark:  {info['suite']} v{info['version']} / sequence={info['sequence']}"
        )
        typer.echo(
            f"Map:        AUC={map_result['auc']:.4f}  Chamfer={map_result['chamfer_distance']:.4f}"
        )
        typer.echo(
            "Trajectory: ATE={ate:.4f}  RPE={rpe:.4f}  Drift={drift:.4f}  Coverage={cov:.1%}".format(
                ate=trajectory_result["ate"]["rmse"],
                rpe=trajectory_result["rpe_translation"]["rmse"],
                drift=trajectory_result["drift"]["endpoint"],
                cov=trajectory_result["matching"]["coverage_ratio"],
            )
        )
        overall_gate = result["overall_quality_gate"]
        if overall_gate is not None:
            typer.echo("")
            typer.echo(
                f"Overall Quality Gate: {'PASS' if overall_gate['passed'] else 'FAIL'}"
            )
            for reason in overall_gate["reasons"]:
                typer.echo(f"  - {reason}")
        if report:
            typer.echo(f"Report: {report}")
        if bundle_paths is not None:
            typer.echo(f"Bundle: {bundle_paths['bundle_dir']}")
            typer.echo(f"Report: {bundle_paths['report']}")
            typer.echo(f"JSON: {bundle_paths['metrics']}")

    if output_json:
        _dump_json(result, output_json)
    overall = result["overall_quality_gate"]
    if overall is not None and not overall["passed"]:
        raise typer.Exit(code=1)


@benchmark_app.command("init")
def benchmark_init_cmd(
    suite_dir: str = typer.Argument(..., help="Directory to materialize the suite into (will be created)"),
    name: str = typer.Option(..., "--name", help="Suite name written to suite.yaml"),
    description: str = typer.Option(..., "--description", help="Suite description"),
    reference_map: str = typer.Option(..., "--reference-map", help="Path to the reference (GT) point cloud map"),
    reference_trajectory: str = typer.Option(
        ...,
        "--reference-trajectory",
        help="Path to the reference (GT) trajectory in TUM format",
    ),
    sequence: str = typer.Option(
        "default",
        "--sequence",
        help="Sequence name inside the suite manifest",
    ),
    sequence_description: Optional[str] = typer.Option(
        None,
        "--sequence-description",
        help="Per-sequence description (defaults to the suite description)",
    ),
    license: Optional[str] = typer.Option(
        None,
        "--license",
        help="License string written into suite.yaml (e.g. dataset attribution)",
    ),
    voxel: float = typer.Option(
        0.0,
        "--voxel",
        help="Voxel size (meters) for downsampling the reference map; 0 = no downsample",
    ),
    max_poses: Optional[int] = typer.Option(
        None,
        "--max-poses",
        help="Keep at most this many evenly-spaced trajectory poses; default = keep all",
    ),
    sample_map: Optional[str] = typer.Option(
        None,
        "--sample-map",
        help="Optional sample-output map to bundle for smoke tests",
    ),
    sample_trajectory: Optional[str] = typer.Option(
        None,
        "--sample-trajectory",
        help="Optional sample-output trajectory to bundle for smoke tests",
    ),
    gate_values: Optional[List[str]] = typer.Option(
        None,
        "--gate",
        help="Gate threshold as key=value (repeatable): --gate min_auc=0.97 --gate max_chamfer=0.05",
    ),
) -> None:
    """Materialize a SLAM benchmark suite from raw GT data on disk."""
    gate: dict[str, float] = {}
    if gate_values:
        for raw in gate_values:
            if "=" not in raw:
                typer.echo(f"Error: --gate expects key=value; got {raw!r}", err=True)
                raise typer.Exit(code=1)
            key, _, value = raw.partition("=")
            key = key.strip()
            if key not in GATE_KEYS:
                typer.echo(
                    f"Error: unknown gate key {key!r}. Allowed: {', '.join(GATE_KEYS)}",
                    err=True,
                )
                raise typer.Exit(code=1)
            try:
                gate[key] = float(value)
            except ValueError:
                typer.echo(f"Error: --gate {key} must be numeric; got {value!r}", err=True)
                raise typer.Exit(code=1)

    try:
        suite = materialize_suite(
            suite_dir,
            name=name,
            description=description,
            reference_map=reference_map,
            reference_trajectory=reference_trajectory,
            sequence_name=sequence,
            sequence_description=sequence_description,
            license=license,
            voxel_size=voxel,
            max_poses=max_poses,
            gate=gate or None,
            sample_map=sample_map,
            sample_trajectory=sample_trajectory,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _handle_error(exc)

    seq = suite.resolve_sequence(sequence)
    typer.echo(f"Suite:               {suite.name}")
    typer.echo(f"Manifest:            {suite.source_path}")
    typer.echo(f"Sequence:            {seq.name}")
    typer.echo(f"Reference map:       {seq.reference_map_path}")
    typer.echo(f"Reference trajectory: {seq.reference_trajectory_path}")
    if voxel > 0:
        typer.echo(f"Voxel downsample:    {voxel} m")
    if max_poses is not None:
        typer.echo(f"Max poses kept:      {max_poses}")
    if suite.gate:
        typer.echo("Gate:")
        for gate_key, gate_value in suite.gate.items():
            typer.echo(f"  {gate_key}: {gate_value}")
