"""Vector-map drafting commands."""

import json
from pathlib import Path
from typing import List, Literal, Optional, cast
import typer
from cloudanalyzer_cli._app import app, _handle_error


@app.command("vectormap-build")
def vectormap_build_cmd(
    cloud: str = typer.Argument(..., help="Surveyed point cloud, in the trajectory's metre frame"),
    trajectory: str = typer.Argument(..., help="TUM, KITTI or timestamped XYZ CSV; outside forward lane"),
    out: str = typer.Option(..., "--out", help="New output directory; existing directories are not replaced"),
    forward_lanes: int = typer.Option(1, "--forward-lanes"),
    backward_lanes: int = typer.Option(1, "--backward-lanes"),
    right_hand: bool = typer.Option(False, "--right-hand", help="Traffic keeps right (default: left)"),
    lane_width: float = typer.Option(3.5, "--lane-width", help="Width prior in metres"),
    speed_limit: float = typer.Option(40.0, "--speed-limit", help="Speed in km/h"),
    segment_length: float = typer.Option(50.0, "--segment-length", help="Lane piece length in metres; 0 keeps whole roads"),
    no_anchor_width_prior: bool = typer.Option(False, "--no-anchor-width-prior", help="Keep inferred lines relative to the trajectory instead of detected outer edges"),
    physical_anchors_only: bool = typer.Option(False, "--physical-anchors-only", help="Exclude scan limits from inferred offset anchors; retain curb/paint observations"),
    align_trace_to_curbs: bool = typer.Option(False, "--align-trace-to-curbs", help="Translate straight traces using stable paired source curbs; preserve lane count and width priors"),
    infer_lane_edges: bool = typer.Option(False, "--infer-lane-edges", help="Infer configured-width lane edges inside distant curbs after paint-divider correction; retain road-edge candidates separately"),
    fit_paint_divider: bool = typer.Option(False, "--fit-paint-divider", help="Correct a two-lane interior line from strong source paint guarded by paired curbs; retain outside boundaries"),
    fit_paint_corridor: bool = typer.Option(False, "--fit-paint-corridor", help="Fit straight source paint bundles; report observed intervals and inferred gaps, preserving explicit lane roles"),
    paint_channel: str = typer.Option("rgb", "--paint-channel", help="Explicit paint-fit source: rgb (default) or retained intensity; no automatic fallback"),
    no_track_boundaries: bool = typer.Option(False, "--no-track-boundaries", help="Choose each boundary slice independently without continuity tracking"),
    no_fit_boundaries: bool = typer.Option(False, "--no-fit-boundaries", help="Keep selected source positions without trajectory-relative curve fitting"),
    fit_source_surface: bool = typer.Option(False, "--fit-source-surface", help="Fit widths/heights to source ground; defer unsupported intervals; lane counts remain explicit"),
    local_ground_height: bool = typer.Option(False, "--local-ground-height", help="Estimate seed-road height from the lowest spatially supported layer; defer missing support and review level ambiguity"),
    no_verify_curb_profiles: bool = typer.Option(False, "--no-verify-curb-profiles", help="Keep unchecked height-step candidates, including tall objects and isolated returns"),
    no_merge_repeated_passes: bool = typer.Option(False, "--no-merge-repeated-passes", help="Add roads without reusing matching existing intervals"),
    existing_map: Optional[str] = typer.Option(None, "--existing-map", help="Keep this IR JSON or Lanelet2 map and add uncovered intervals"),
    reference_map: Optional[str] = typer.Option(None, "--reference-map", help="Lanelet2 map supplying coordinate metadata only"),
    projection: Optional[str] = typer.Option(None, "--projection", help="mgrs, utm or transverse_mercator; omit for Local"),
    origin_lat: Optional[float] = typer.Option(None, "--origin-lat", help="Latitude of origin or MGRS tile representative"),
    origin_lon: Optional[float] = typer.Option(None, "--origin-lon", help="Longitude of origin or MGRS tile representative"),
) -> None:
    """Draft Autoware Lanelet2 roads and print their evidence and validation report as JSON."""
    from ca.vector_map import build_vector_map

    try:
        report = build_vector_map(
            cloud, trajectory, out, forward_lanes=forward_lanes, backward_lanes=backward_lanes,
            left_hand_traffic=not right_hand, lane_width=lane_width, speed_limit=speed_limit,
            segment_length=segment_length, reference_map=reference_map, projection=projection,
            anchor_width_prior=not no_anchor_width_prior,
            physical_anchors_only=physical_anchors_only,
            align_trace_to_curbs=align_trace_to_curbs,
            infer_lane_edges=infer_lane_edges,
            fit_paint_divider=fit_paint_divider,
            fit_paint_corridor=fit_paint_corridor,
            paint_channel=cast(Literal["rgb", "intensity"], paint_channel),
            track_boundaries=not no_track_boundaries, fit_boundaries=not no_fit_boundaries,
            fit_source_surface=fit_source_surface,
            local_ground_height=local_ground_height,
            verify_curb_profiles=not no_verify_curb_profiles,
            merge_repeated_passes=not no_merge_repeated_passes, existing_map=existing_map,
            origin_lat=origin_lat, origin_lon=origin_lon,
        )
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
        return
    typer.echo(json.dumps(report, indent=2))


@app.command("vectormap-connect")
def vectormap_connect_cmd(
    cloud: str = typer.Argument(..., help="Point cloud in the map's metre frame"),
    vector_map: str = typer.Argument(..., help="Editable IR JSON or Lanelet2 map with open road ends"),
    out: str = typer.Option(..., "--out", help="New output directory"),
    max_gap: float = typer.Option(30.0, "--max-gap", help="Maximum XY gap in metres, 1..100"),
    min_ground_support: float = typer.Option(0.9, "--min-ground-support", help="Supported centre sample fraction, 0.5..1"),
    check_boundary_support: bool = typer.Option(False, "--check-boundary-support", help="Also require ground along both boundaries and all endpoints"),
    preview: bool = typer.Option(False, "--preview", help="Write candidates and the unchanged input map for review"),
    lane_pairs: Optional[List[str]] = typer.Option(None, "--pair", help="Select FROM:TO; repeat for branches. Omit to add all proposals"),
) -> None:
    """Draft junction connections; ground support does not establish permitted turns."""
    from ca.vector_map import connect_vector_map_junctions

    try:
        pairs: Optional[List[tuple[int, int]]] = None
        if lane_pairs is not None:
            pairs = []
            for value in lane_pairs:
                parts = value.split(":")
                if len(parts) != 2:
                    raise ValueError("--pair requires FROM:TO lane IDs")
                pairs.append((int(parts[0]), int(parts[1])))
        report = connect_vector_map_junctions(
            cloud, vector_map, out, max_gap=max_gap, min_ground_support=min_ground_support,
            check_boundary_support=check_boundary_support,
            lane_pairs=pairs, preview_only=preview,
        )
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
        return
    typer.echo(json.dumps(report, indent=2))


@app.command("vectormap-signal")
def vectormap_signal_cmd(
    cloud: str = typer.Argument(..., help="Local point cloud or HTTP(S) COPC .laz URL in the map's metre frame"),
    vector_map: str = typer.Argument(..., help="Editable IR JSON or Lanelet2 map"),
    out: str = typer.Option(..., "--out", help="New output directory"),
    box: str = typer.Option(..., "--box", help="xmin,ymin,zmin,xmax,ymax,zmax enclosing the identified head"),
    lanes: List[int] = typer.Option(..., "--lane", help="Confirmed controlled lane ID; repeat for multiple lanes"),
    kind: str = typer.Option("vehicle", "--kind", help="vehicle or pedestrian (user identification)"),
    add: bool = typer.Option(False, "--add", help="Add measured geometry; default only previews"),
) -> None:
    """Measure a signal head from points; identify the object and controlled lanes first."""
    from ca.vector_map import measure_vector_map_signal

    try:
        report = measure_vector_map_signal(cloud, vector_map, out, bounds=[float(v) for v in box.split(",")],
                                          lanes=lanes, kind=kind, preview_only=not add)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
        return
    typer.echo(json.dumps(report, indent=2))


@app.command("vectormap-crosswalk")
def vectormap_crosswalk_cmd(
    cloud: str = typer.Argument(..., help="Local cloud retaining RGB or intensity; whole-file compatibility reader"),
    vector_map: str = typer.Argument(..., help="Editable IR JSON or Lanelet2 map in the same metre frame"),
    out: str = typer.Option(..., "--out", help="New output directory"),
    box: str = typer.Option(..., "--box", help="xmin,ymin,zmin,xmax,ymax,zmax enclosing road paint"),
    lanes: List[int] = typer.Option([], "--lane", help="Confirmed crossing lane ID; repeat as needed, optional for preview"),
    candidate: int = typer.Option(0, "--candidate", help="Reviewed candidate index (zero based)"),
    brightness_fraction: float = typer.Option(0.75, "--brightness-fraction", help="0.4–0.9 of ground contrast; lower includes dim paint and more background"),
    add: bool = typer.Option(False, "--add", help="Add the reviewed crossing; default only previews bands"),
) -> None:
    """Measure paint bands; confirm the object and crossing lanes before adding."""
    from ca.vector_map import measure_vector_map_crosswalk

    try:
        report = measure_vector_map_crosswalk(cloud, vector_map, out,
            bounds=[float(v) for v in box.split(",")], lanes=lanes,
            candidate=candidate, brightness_fraction=brightness_fraction, preview_only=not add)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
        return
    typer.echo(json.dumps(report, indent=2))


@app.command("vectormap-discover")
def vectormap_discover_cmd(
    cloud: str = typer.Argument(..., help="Local attribute-bearing scene; whole-file compatibility reader"),
    out: str = typer.Option(..., "--out", help="New output directory"),
    vector_map: Optional[str] = typer.Option(None, "--map", help="Roads generated from cloud/trajectory, IR JSON or OSM"),
    scope: str = typer.Option("road_corridor", "--scope", help="road_corridor or ground_surface (no map required)"),
    radius: float = typer.Option(12.0, "--radius", help="4–18 metre corridor radius"),
    brightness_fraction: float = typer.Option(0.65, "--brightness-fraction", help="0.4–0.9 of ground contrast"),
    confirmations: Optional[str] = typer.Option(None, "--confirmations", help="JSON file of explicitly reviewed candidate/key/classification/lanes; default previews"),
) -> None:
    """Automatically find equipment proposals; inspect and confirm semantics before adding."""
    from ca.vector_map import discover_vector_map_features
    try:
        confirmed = json.loads(Path(confirmations).read_text(encoding="utf-8")) if confirmations else None
        report = discover_vector_map_features(cloud, out, vector_map=vector_map, scope=scope, corridor_radius=radius, brightness_fraction=brightness_fraction, confirmations=confirmed)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
        return
    typer.echo(json.dumps(report, indent=2))


@app.command("vectormap-suggest")
def vectormap_suggest_cmd(
    vector_map: str = typer.Argument(..., help="Existing map JSON or Lanelet2 OSM"),
    rule_id: int = typer.Option(..., "--rule", help="Signal regulatory element ID"),
    candidate_key: str | None = typer.Option(None, "--candidate", help="Explicit candidate key from preview"),
    map_snapshot: str | None = typer.Option(None, "--snapshot", help="Current map snapshot from preview"),
    out: str | None = typer.Option(None, "--out", help="NEW output directory for adoption; omit for preview"),
):
    """Preview signal target evidence or adopt an explicitly reviewed candidate."""
    from ca.vector_map import propose_vector_map_relations
    try:
        report = propose_vector_map_relations(vector_map, rule_id, candidate_key=candidate_key, map_snapshot=map_snapshot, out_dir=out)
        typer.echo(json.dumps(report, indent=2))
    except (ValueError, RuntimeError, OSError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)


@app.command("vectormap-relate")
def vectormap_relate_cmd(
    vector_map: str = typer.Argument(..., help="Existing editable IR JSON or Lanelet2 OSM"),
    rule_id: Optional[int] = typer.Option(None, "--rule", help="Equipment rule ID; omit for read-only inspection"),
    lanes: List[int] = typer.Option([], "--lane", help="Reviewed vehicle/crossing lane ID; repeat for each"),
    crosswalks: List[int] = typer.Option([], "--crosswalk", help="Pedestrian controlled crossing ID; repeat"),
    stops: List[int] = typer.Option([], "--stop", help="Reviewed stop-line ID; repeat"),
    out: Optional[str] = typer.Option(None, "--out", help="New output directory, required for edits"),
) -> None:
    """Inspect or explicitly review equipment associations, retaining physical geometry."""
    from ca.vector_map import edit_vector_map_relations
    try:
        report = edit_vector_map_relations(vector_map, rule_id=rule_id, lanes=lanes, controlled_crosswalks=crosswalks, stop_lines=stops, out_dir=out)
    except (OSError, ValueError, RuntimeError) as error:
        _handle_error(error)
        return
    typer.echo(json.dumps(report, indent=2))
