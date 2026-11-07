"""Mapping job and agent-run commands."""

import json
import zipfile
from pathlib import Path
from typing import Optional
import typer
from cloudanalyzer_cli._app import app, _load_json_mapping, _handle_error


@app.command("mapping-start")
def mapping_start_cmd(
    source: str = typer.Argument(..., help="Raw MCAP, ROS1 bag or rosbag2 SQLite recording"),
    out: str = typer.Option(..., "--out", help="New mapping job directory"),
    max_attempts: int = typer.Option(4, "--max-attempts", min=1, max=8),
    keyframe_spacing: float = typer.Option(1.0, "--keyframe-spacing"),
    keep_dynamic: bool = typer.Option(False, "--keep-dynamic"),
    minimum_retained_fraction: float = typer.Option(0.9, "--minimum-retained-fraction", min=0, max=1),
) -> None:
    """Generate a point-cloud map and trajectory for an agent-controlled mapping job."""
    from ca.mapping_job import start_mapping_job
    try:
        result = start_mapping_job(source, out, keyframe_spacing=keyframe_spacing,
                                   remove_dynamic=not keep_dynamic, max_attempts=max_attempts,
                                   minimum_retained_fraction=minimum_retained_fraction)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))
    if result["status"] == "pointcloud_failed":
        raise typer.Exit(1)


@app.command("mapping-candidate")
def mapping_candidate_cmd(
    job: str = typer.Argument(...),
    options: str = typer.Option(..., "--options", help="JSON file of explicit road assumptions and fitting options"),
    reason: str = typer.Option(..., "--reason", help="Why the agent is testing this hypothesis"),
) -> None:
    """Generate and audit one HD-map candidate, recording failures without overwriting others."""
    from ca.mapping_job import generate_mapping_candidate
    try:
        parameters = json.loads(Path(options).read_text(encoding="utf-8"))
        if not isinstance(parameters, dict):
            raise ValueError("options must be a JSON object")
        result = generate_mapping_candidate(job, parameters, reason)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))
    if result["attempts"][-1]["status"] == "failed":
        raise typer.Exit(1)


@app.command("mapping-corridors")
def mapping_corridors_cmd(
    job: str = typer.Argument(...),
    search_radius: float = typer.Option(8.0, "--search-radius", min=1, max=20),
) -> None:
    """Generate lane-free surface corridor proposals before assigning road assumptions."""
    from ca.mapping_job import propose_mapping_corridors
    try:
        result = propose_mapping_corridors(job, search_radius_m=search_radius)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))
    if result["status"] == "failed":
        raise typer.Exit(1)


@app.command("mapping-corridors-inspect")
def mapping_corridors_inspect_cmd(
    job: str = typer.Argument(...),
    candidate: Optional[int] = typer.Option(None, "--candidate", min=1),
    offset: int = typer.Option(0, "--offset", min=0),
) -> None:
    """Read paged corridor evidence or candidate geometry without native processing."""
    from ca.mapping_job import inspect_mapping_corridors
    try:
        result = inspect_mapping_corridors(job, candidate_id=candidate, offset=offset)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))


@app.command("mapping-geometry")
def mapping_geometry_cmd(
    job: str = typer.Argument(...),
    decisions: str = typer.Option(..., "--decisions", help="JSON list of corridor include/defer decisions and reasons"),
    reason: str = typer.Option(..., "--reason"),
) -> None:
    """Adopt source curves as an editable geometry draft with unresolved lane semantics."""
    from ca.mapping_job import generate_mapping_geometry
    try:
        choices = json.loads(Path(decisions).read_text(encoding="utf-8"))
        result = generate_mapping_geometry(job, choices, reason)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))
    if result["attempts"][-1]["status"] == "failed":
        raise typer.Exit(1)


@app.command("mapping-geometry-inspect")
def mapping_geometry_inspect_cmd(
    job: str = typer.Argument(...),
    candidate: int = typer.Option(..., "--candidate", min=1),
    offset: int = typer.Option(0, "--offset", min=0),
) -> None:
    """Read geometry decisions, source curves and unresolved extent without processing."""
    from ca.mapping_job import inspect_mapping_geometry
    try:
        result = inspect_mapping_geometry(job, candidate, offset)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))


@app.command("mapping-run")
def mapping_run_cmd(
    source: str = typer.Argument(...),
    out: str = typer.Option(..., "--out"),
    layout: str = typer.Option(..., "--layout", help="JSON object with fixed, explicit unverified lane assumptions"),
    max_attempts: int = typer.Option(6, "--max-attempts", min=2, max=8),
    minimum_retained_fraction: float = typer.Option(.9, "--minimum-retained-fraction", min=0, max=1),
    sample_spacing_m: float | None = typer.Option(None, "--corridor-sample-spacing", min=.5, max=2., help="Initial corridor station spacing in metres; omission uses the native default"),
) -> None:
    """Start the MCP agent's mapping loop and return its next decision contract."""
    from ca.mapping_run import start_mapping_run
    try:
        result = start_mapping_run(source, out, json.loads(Path(layout).read_text(encoding="utf-8")),
                                   max_attempts=max_attempts, minimum_retained_fraction=minimum_retained_fraction,
                                   sample_spacing_m=sample_spacing_m)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))
    if result["status"] == "processing_failed":
        raise typer.Exit(1)


@app.command("mapping-scorecard")
def mapping_scorecard_cmd(
    run: str = typer.Argument(..., help="Saved mapping run directory"),
    baseline: str | None = typer.Option(None, "--baseline", help="Saved run to compare against; requires identical source/layout hashes"),
    text_output: bool = typer.Option(False, "--text", help="Show a readable summary instead of JSON"),
    output: str | None = typer.Option(None, "--output", help="Save to a new file instead of stdout; parent directory must exist"),
) -> None:
    """Summarize saved generation and measurement evidence without running mapping."""
    from ca.mapping_scorecard import scorecard, compare, format_text, save_output
    try:
        current = scorecard(run)
        result = compare(scorecard(baseline), current) if baseline is not None else current
        encoded = format_text(result) if text_output else json.dumps(result, indent=2, allow_nan=False)
        if output is not None:
            save_output(output, encoded)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    if output is None:
        typer.echo(encoded)


@app.command("mapping-measure")
def mapping_measure_cmd(
    command: list[str] = typer.Argument(..., help="Executable and arguments after --; no shell expansion"),
    run_dir: str = typer.Option(..., "--run-dir", help="New run directory used by the child command"),
    interval: float = typer.Option(.05, "--interval", min=.01, max=1, help="Memory sampling interval in seconds"),
    timeout: float | None = typer.Option(None, "--timeout", help="Positive command duration limit in seconds; omitted means unlimited"),
) -> None:
    """Measure a synchronous command and save its receipt in a new run directory."""
    from ca.mapping_measure import measure
    try:
        result = measure(command, run_dir, interval, timeout=timeout)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2, allow_nan=False))
    if result['status'] != 'process_succeeded':
        raise typer.Exit(1)


@app.command("mapping-prepare-glim-recording")
def mapping_prepare_glim_recording_cmd(
    lidar: str = typer.Argument(..., help="ROS2 point cloud MCAP"),
    imu: str = typer.Option(..., "--imu", help="ROS2 raw IMU MCAP"),
    points_topic: str = typer.Option(..., "--points-topic"),
    imu_topic: str = typer.Option(..., "--imu-topic"),
    max_imu_gap: float = typer.Option(.05, "--max-imu-gap", min=1e-9, max=3600, help="Maximum IMU header interval in seconds; default 50 ms"),
    out: str = typer.Option(..., "--out", help="New recording and receipt directory"),
) -> None:
    """Join selected point/IMU streams unchanged and run the GLIM input diagnostic."""
    from ca.glim_recording import prepare_glim_recording
    try:
        result = prepare_glim_recording(lidar, imu, out, points_topic, imu_topic, max_imu_gap_s=max_imu_gap)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))
    if result['status'] != 'basic_checks_passed':
        raise typer.Exit(1)


@app.command("mapping-check-glim")
def mapping_check_glim_cmd(
    bag: str = typer.Argument(..., help="ROS2 MCAP recording"),
    points_topic: str = typer.Option(..., "--points-topic"),
    imu_topic: str = typer.Option(..., "--imu-topic"),
    max_imu_gap: float = typer.Option(.05, "--max-imu-gap", min=1e-9, max=3600, help="Maximum IMU header interval in seconds; default 50 ms"),
) -> None:
    """Inspect GLIM input sensor availability and timestamps without running SLAM."""
    from ca.glim_preflight import inspect_glim_bag
    try:
        result = inspect_glim_bag(bag, points_topic, imu_topic, max_imu_gap_s=max_imu_gap)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))
    if result['status'] != 'basic_checks_passed':
        raise typer.Exit(1)


@app.command("mapping-inspect-glim")
def mapping_inspect_glim_cmd(
    directory: str = typer.Argument(..., help="GLIM run directory containing glim-run.json"),
) -> None:
    """Recheck recorded GLIM inputs, configuration and trajectory without running SLAM."""
    from ca.glim_inspect import inspect_glim_run
    try:
        result = inspect_glim_run(directory)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))
    if result['status'] != 'recorded_files_match':
        raise typer.Exit(1)


@app.command("mapping-run-glim")
def mapping_run_glim_cmd(
    bag: str = typer.Argument(..., help="ROS2 MCAP file or bag directory"),
    config: str = typer.Option(..., "--config", help="Complete sensor-specific GLIM configuration directory"),
    out: str = typer.Option(..., "--out", help="New run directory"),
    image: str = typer.Option(..., "--image", help="Already installed GLIM ROS2 Docker image"),
    points_topic: str | None = typer.Option(None, "--points-topic", help="Enable input preflight for this MCAP point topic; also supply --imu-topic"),
    imu_topic: str | None = typer.Option(None, "--imu-topic", help="MCAP IMU topic to preflight before starting Docker"),
    max_imu_gap: float = typer.Option(.05, "--max-imu-gap", min=1e-9, max=3600),
    timeout: float = typer.Option(3600, "--timeout", min=1, help="Maximum execution seconds"),
) -> None:
    """Execute GLIM in Docker and retain input hashes, logs and dump artifacts."""
    from ca.glim_run import run_glim
    import subprocess
    try:
        result = run_glim(bag, config, out, image, timeout, points_topic=points_topic, imu_topic=imu_topic, max_imu_gap_s=max_imu_gap)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))
    if result['status'] != 'process_succeeded':
        raise typer.Exit(1)


@app.command("mapping-prepare-glim")
def mapping_prepare_glim_cmd(
    dump: str = typer.Argument(..., help="GLIM dump containing global traj_lidar.txt"),
    point_map: str = typer.Option(..., "--map", help="PLY exported from the same optimized GLIM state"),
    out: str = typer.Option(..., "--out", help="New directory for trajectory and provenance"),
    run_report: str | None = typer.Option(None, "--run-report", help="Successful GLIM run report to bind to this exact trajectory"),
    provenance: str = typer.Option(..., "--provenance", help="Explicit GLIM version, source, frame, time and configuration JSON"),
) -> None:
    """Prepare a GLIM global LiDAR export for mapping-import, without running SLAM."""
    from ca.glim_import import prepare_glim_export
    try:
        result = prepare_glim_export(dump, point_map, out, _load_json_mapping(provenance), run_report=run_report)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))


@app.command("mapping-import")
def mapping_import_cmd(
    point_map: str = typer.Argument(..., help="Existing PLY map; referenced without copying"),
    trajectory: str = typer.Option(..., "--trajectory", help="TUM world-from-sensor poses in the same metre frame"),
    out: str = typer.Option(..., "--out"),
    provenance: str = typer.Option(..., "--provenance", help="JSON producer/frame/time/configuration declaration"),
    layout: str = typer.Option(..., "--layout", help="JSON fixed, explicit unverified lane assumptions"),
    max_attempts: int = typer.Option(6, "--max-attempts", min=2, max=8),
    minimum_retained_fraction: float = typer.Option(.9, "--minimum-retained-fraction", min=0, max=1),
    sample_spacing_m: float | None = typer.Option(None, "--corridor-sample-spacing", min=.5, max=2., help="Initial corridor station spacing in metres; omission uses the native default"),
) -> None:
    """Start HD drafting from an external SLAM map and trajectory."""
    from ca.mapping_run import start_mapping_run_from_slam
    try:
        result = start_mapping_run_from_slam(point_map, trajectory, out,
            _load_json_mapping(provenance), _load_json_mapping(layout),
            max_attempts=max_attempts, minimum_retained_fraction=minimum_retained_fraction,
            sample_spacing_m=sample_spacing_m)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))
    if result["status"] == "processing_failed":
        raise typer.Exit(1)


@app.command("mapping-run-inspect")
def mapping_run_inspect_cmd(job: str = typer.Argument(...), offset: int = typer.Option(0, "--offset", min=0)) -> None:
    """Read saved agent-run state, paged observations and output paths."""
    from ca.mapping_run import inspect_mapping_run
    try:
        result = inspect_mapping_run(job, offset)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))


@app.command("mapping-trajectory-evaluate")
def mapping_trajectory_evaluate_cmd(
    job: str = typer.Argument(...),
    reference: str = typer.Option(..., "--reference", help="Timestamped metre-frame TUM/CSV reference"),
    provenance: str = typer.Option(..., "--provenance", help="JSON source/license/frame/time basis and generation-use declaration"),
    out: str = typer.Option(..., "--out", help="New report file outside the mapping job"),
    max_time_delta: float = typer.Option(.05, "--max-time-delta"),
    alignment_prefix_fraction: float = typer.Option(1., "--alignment-prefix-fraction", help="Fit only this prefix and evaluate the disjoint suffix; 1 fits/evaluates all samples"),
) -> None:
    """Compare saved original/corrected motion without changing maps or attempts."""
    from ca.mapping_trajectory import evaluate_mapping_trajectory
    try:
        result = evaluate_mapping_trajectory(job, reference, json.loads(Path(provenance).read_text()), out, max_time_delta, alignment_prefix_fraction)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))


@app.command("mapping-trajectory-inspect")
def mapping_trajectory_inspect_cmd(
    report: str = typer.Argument(..., help="Saved comparison JSON"),
    sha256: str = typer.Option(..., "--sha256", help="Expected SHA-256 from evaluation's report artifact"),
    size: int = typer.Option(..., "--bytes", help="Expected byte count from evaluation's report artifact"),
    window_poses: int = typer.Option(12, "--window-poses"),
    ranking: str = typer.Option("regression", "--ranking"),
    offset: int = typer.Option(0, "--offset"),
) -> None:
    """Locate evaluated trajectory error windows in the original point-map frame."""
    from ca.mapping_trajectory_review import inspect_mapping_trajectory_comparison
    try:
        result = inspect_mapping_trajectory_comparison(
            {"path": str(Path(report).resolve()), "sha256": sha256, "bytes": size}, window_poses, ranking, offset)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))


@app.command("mapping-motion-trial")
def mapping_motion_trial_cmd(
    job: str = typer.Argument(...),
    out: str = typer.Option(..., "--out"),
    policy: str = typer.Option(..., "--policy", help="JSON file with explicit find_loops/use_gravity booleans"),
    reason: str = typer.Option(..., "--reason"),
    max_attempts: int = typer.Option(4, "--max-attempts", min=1, max=8),
) -> None:
    """Generate one alternative motion/point-map candidate in a new job."""
    from ca.mapping_motion_trial import trial_mapping_motion
    try:
        value = json.loads(Path(policy).read_text())
        if not isinstance(value, dict) or set(value) != {"find_loops", "use_gravity"}:
            raise ValueError("policy needs exactly find_loops and use_gravity")
        result = trial_mapping_motion(job, out, value["find_loops"], value["use_gravity"], reason, max_attempts)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))
    if result["status"] == "failed":
        raise typer.Exit(1)


@app.command("mapping-run-advance")
def mapping_run_advance_cmd(
    job: str = typer.Argument(...),
    action: str = typer.Option(..., "--action", help="JSON decision from inspect_mapping_run.action_contract, including point-cloud retry and comparison"),
    revision: int = typer.Option(..., "--revision", min=0),
    reason: str = typer.Option(..., "--reason"),
) -> None:
    """Execute and record an agent decision using the inspected run revision."""
    from ca.mapping_run import advance_mapping_run
    try:
        result = advance_mapping_run(job, json.loads(Path(action).read_text(encoding="utf-8")), reason, revision)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))
    if any(result.get(key, {}).get("status") == "failed" for key in ("draft_result", "refine_result", "connect_result", "pointcloud_retry_result")):
        raise typer.Exit(1)


@app.command("mapping-run-export")
def mapping_run_export_cmd(
    job: str = typer.Argument(...),
    out: str = typer.Option(..., "--out", help="New portable review ZIP; existing files are never overwritten"),
    attribution: str = typer.Option(..., "--attribution", help="Source-data license and credit text"),
    max_bundle_bytes: int = typer.Option(1024**3, "--max-bundle-bytes", min=1024, max=4 * 1024**3),
) -> None:
    """Package the exact finished point/HD pair and final evidence for portable review."""
    from ca.mapping_bundle import export_mapping_run
    try:
        result = export_mapping_run(job, out, attribution, max_bundle_bytes)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))


@app.command("mapping-run-preview")
def mapping_run_preview_cmd(
    job: str = typer.Argument(...),
    out: str = typer.Option(..., "--out", help="New display-preview review ZIP"),
    attribution: str = typer.Option(
        ..., "--attribution", help="Source-data license and credit text"
    ),
    max_preview_points: int = typer.Option(
        200000, "--max-preview-points", min=1, max=1000000
    ),
    max_bundle_bytes: int = typer.Option(
        64 * 1024**2, "--max-bundle-bytes", min=1024, max=4 * 1024**3
    ),
) -> None:
    """Package bounded display-only point records and the exact HD map with original audits."""
    from ca.mapping_bundle import export_mapping_preview

    try:
        result = export_mapping_preview(
            job, out, attribution, max_preview_points, max_bundle_bytes
        )
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))


@app.command("mapping-bundle-inspect")
def mapping_bundle_inspect_cmd(
    bundle: str = typer.Argument(...),
    max_bundle_bytes: int = typer.Option(1024**3, "--max-bundle-bytes", min=1024, max=4 * 1024**3),
) -> None:
    """Verify a portable review ZIP and read its map paths and remaining holds."""
    from ca.mapping_bundle import inspect_mapping_bundle
    try:
        result = inspect_mapping_bundle(bundle, max_bundle_bytes)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))


@app.command("mapping-lanes")
def mapping_lanes_cmd(
    job: str = typer.Argument(...),
    geometry: int = typer.Option(..., "--geometry", min=1, help="Retained geometry attempt ID"),
    specs: str = typer.Option(..., "--specs", help="JSON list of explicit per-centre-curve lane hypotheses"),
    boundary_policy: str = typer.Option(..., "--boundary-policy"),
    reason: str = typer.Option(..., "--reason"),
) -> None:
    """Export source-span lane hypotheses with preserved geometry and IR/OSM source audits."""
    from ca.mapping_job import generate_mapping_corridor_lanes
    try:
        hypotheses = json.loads(Path(specs).read_text(encoding="utf-8"))
        result = generate_mapping_corridor_lanes(job, geometry, hypotheses, boundary_policy, reason)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))
    if result["attempts"][-1]["status"] == "failed":
        raise typer.Exit(1)


@app.command("mapping-status")
def mapping_status_cmd(job: str = typer.Argument(...)) -> None:
    """Inspect mapping evidence, artifact paths and remaining attempt budget as JSON."""
    from ca.mapping_job import inspect_mapping_job
    try:
        result = inspect_mapping_job(job)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))


@app.command("mapping-select")
def mapping_select_cmd(
    job: str = typer.Argument(...),
    candidate: int = typer.Option(..., "--candidate"),
    reason: str = typer.Option(..., "--reason"),
) -> None:
    """Select a retained HD draft with a recorded reason and visible quality holds."""
    from ca.mapping_job import select_mapping_candidate
    try:
        result = select_mapping_candidate(job, candidate, reason)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))


@app.command("mapping-diagnose")
def mapping_diagnose_cmd(
    job: str = typer.Argument(...),
    candidate: int = typer.Option(..., "--candidate", min=1),
) -> None:
    """Explain saved source-review holds without rerunning generation or spending attempts."""
    from ca.mapping_job import diagnose_mapping_candidate
    try:
        result = diagnose_mapping_candidate(job, candidate)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
    typer.echo(json.dumps(result, indent=2))
