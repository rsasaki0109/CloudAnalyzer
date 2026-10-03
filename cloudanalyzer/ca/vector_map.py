"""Draft Autoware Lanelet2 roads from a surveyed cloud and a measured drive."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from ca._rust import core


def build_vector_map(
    cloud: str,
    trajectory: str,
    out_dir: str,
    *,
    forward_lanes: int = 1,
    backward_lanes: int = 1,
    left_hand_traffic: bool = True,
    lane_width: float = 3.5,
    speed_limit: float = 40.0,
    segment_length: float = 50.0,
    anchor_width_prior: bool = True,
    physical_anchors_only: bool = False,
    align_trace_to_curbs: bool = False,
    infer_lane_edges: bool = False,
    fit_paint_divider: bool = False,
    fit_paint_corridor: bool = False,
    paint_channel: Literal["rgb", "intensity"] = "rgb",
    track_boundaries: bool = True,
    fit_boundaries: bool = True,
    fit_source_surface: bool = False,
    verify_curb_profiles: bool = True,
    merge_repeated_passes: bool = True,
    existing_map: str | None = None,
    reference_map: str | None = None,
    projection: str | None = None,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
) -> dict[str, Any]:
    """Write a draft, projector metadata, editable IR and evidence report in a NEW directory.

    Inputs must already share a metre coordinate frame. The trajectory follows the outside
    forward lane. existing_map retains its geometry, IDs, rules and coordinates; matching
    intervals are reused and uncovered intervals are added. It can be editable IR JSON or OSM.
    Reference maps supply coordinate metadata only, never geometry. Explicit
    projections are mgrs, utm or transverse_mercator; an origin selects the MGRS tile or
    defines the local origin of the other projections. Omitted metadata uses Autoware Local.
    Candidate tracking rejects isolated peaks; trajectory-relative curve fitting moves XY by at most
    0.5 m and preserves ground heights. Evidence counts describe selected sources before
    fitting, including explicit width priors. physical_anchors_only excludes point
    coverage limits from offsets for inferred lines; curb/intensity observations
    remain anchors. Coverage-edge geometry can still be selected and needs review.
    The ignored-candidate counter is before tracking and surface deferral.
    align_trace_to_curbs optionally translates straight traces into a stable pair
    of source curbs enclosing the configured total width. It preserves lane counts,
    requires a majority of sections and holds curved or unconfirmed traces.
    The report records the translation; this does not certify lane identity.
    infer_lane_edges optionally places a configured-width outer lane prior
    inside distant verified curbs after an applied paint-divider correction.
    Original curb candidates stay in the report; outer paint is not observed.
    fit_paint_divider optionally corrects only a two-lane interior boundary
    from one strong source-paint track guarded by paired physical curbs. Outside geometry
    remains unchanged; missing paint is inferred and lane roles remain manual.
    paint_channel explicitly selects retained RGB (default) or intensity for paint fits.
    Intensity is normalized to ROI P10/P99.9; no automatic fallback or channel mutation.
    fit_paint_corridor optionally fits straight parallel boundaries from thin source
    paint with dark source returns on both sides. It measures heading and spacing
    but keeps lane counts/directions manual. Sparse outer paint and dash gaps can
    be extended; those vertices remain inferred. The report separates observed
    component intervals, interpolation and extrapolation before footprint trimming.
    Ambiguous bundles, missing contrast and exhausted scan budgets hold the fit.
    Curb profile checks reject tall raised
    surfaces and isolated low returns; this can leave more width assumptions and does
    not guarantee better lane geometry. Stages can be disabled independently.
    fit_source_surface instead fits inferred widths/heights to a coherent low-surface
    band, keeps lane counts explicit and defers unobserved intervals/short fragments.
    Coverage edges can be occlusion, not road boundaries; reduced extent is not an
    accuracy improvement by itself. Existing maps are retained, not automatically refitted.
    Source-backed candidates are retained and clipped first. If less than 60% of their
    length is supported, low-surface footprint fitting replaces those candidates.
    Review boundaries, repeated passes, travel directions and junctions before using the map.
    """
    if paint_channel not in ("rgb", "intensity"):
        raise ValueError("paint_channel must be rgb or intensity")
    module = core()
    if module is None or not hasattr(module, "build_vector_map"):
        raise RuntimeError(
            'vector map drafting needs an updated Rust core: pip install "cloudanalyzer[fast]" (or build rust/crates/ca-py with maturin)'
        )
    inputs = (
        [Path(cloud), Path(trajectory)]
        + ([Path(reference_map)] if reference_map else [])
        + ([Path(existing_map)] if existing_map else [])
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(str(path))
    out = Path(out_dir).resolve()
    if out.exists():
        raise FileExistsError(
            f"output directory already exists: {out}; choose a new directory"
        )
    geo = None
    if existing_map and (
        reference_map or projection or origin_lat is not None or origin_lon is not None
    ):
        raise ValueError(
            "existing_map retains its coordinates; omit reference_map, projection and origin"
        )
    if reference_map and (
        projection or origin_lat is not None or origin_lon is not None
    ):
        raise ValueError("choose reference_map or explicit projection and origin")
    if projection is not None:
        if projection not in {"mgrs", "utm", "transverse_mercator"}:
            raise ValueError(
                "projection must be mgrs, utm or transverse_mercator; omit it for Local"
            )
        if origin_lat is None or origin_lon is None:
            raise ValueError("a projection requires origin_lat and origin_lon")
        if (
            not math.isfinite(origin_lat)
            or not -80 <= origin_lat <= 84
            or not math.isfinite(origin_lon)
            or not -180 <= origin_lon <= 180
        ):
            raise ValueError(
                "origin requires finite latitude [-80,84] and longitude [-180,180]"
            )
        geo = json.dumps(
            {"projection": projection, "origin": {"lat": origin_lat, "lon": origin_lon}}
        )
    elif origin_lat is not None or origin_lon is not None:
        raise ValueError("an origin requires a projection")
    options = json.dumps(
        {
            "forward_lanes": forward_lanes,
            "backward_lanes": backward_lanes,
            "left_hand_traffic": left_hand_traffic,
            "lane_width": lane_width,
            "speed_limit": speed_limit,
            "segment_length": segment_length,
            "anchor_width_prior": anchor_width_prior,
            **({"physical_anchors_only": True} if physical_anchors_only else {}),
            **({"align_trace_to_curbs": True} if align_trace_to_curbs else {}),
            **({"infer_lane_edges": True} if infer_lane_edges else {}),
            **({"fit_paint_divider": True} if fit_paint_divider else {}),
            **({"fit_paint_corridor": True} if fit_paint_corridor else {}),
            **({"paint_channel": "intensity"} if paint_channel == "intensity" else {}),
            "track_boundaries": track_boundaries,
            "fit_boundaries": fit_boundaries,
            **({"fit_source_surface": True} if fit_source_surface else {}),
            "verify_curb_profiles": verify_curb_profiles,
            "merge_repeated_passes": merge_repeated_passes,
        },
        allow_nan=False,
    )
    payload = json.loads(
        module.build_vector_map(
            str(inputs[0]), str(inputs[1]), options, reference_map, geo, existing_map
        )
    )
    report: dict[str, Any] = payload["report"]
    report["inputs"] = {
        "cloud": str(inputs[0].resolve()),
        "trajectory": str(inputs[1].resolve()),
        "reference_map": str(Path(reference_map).resolve()) if reference_map else None,
        "existing_map": str(Path(existing_map).resolve()) if existing_map else None,
    }
    report["options"] = json.loads(options)
    return _publish(payload, out)


def connect_vector_map_junctions(
    cloud: str,
    vector_map: str,
    out_dir: str,
    *,
    max_gap: float = 30.0,
    min_ground_support: float = 0.9,
    check_boundary_support: bool = False,
    lane_pairs: list[tuple[int, int]] | None = None,
    preview_only: bool = False,
) -> dict[str, Any]:
    """Preview or add junction drafts and write four artifacts in a NEW directory.

    Inputs share a metre frame. Existing IR geometry, IDs, rules and projector remain
    fixed; OSM imports report unsupported members. Open driving road ends generate all
    ground-supported choices, including branching junctions. Geometry cannot establish
    permitted turns, obstacle clearance or signal rules: review every candidate.
    preview_only keeps the map unchanged and returns candidate geometry in the report.
    lane_pairs selects (from,to) pairs; omit to add all proposals, or [] for a no-op.
    Use the original input map and a new output directory after reviewing a preview.
    check_boundary_support also checks both actual boundary curves and all endpoints
    with the source-audit protocol. Use min_ground_support=1.0 to defer every unsupported
    sampled interval. Full support is not certification of legal turns or road semantics.
    """
    module = core()
    if module is None or not hasattr(module, "connect_vector_map_junctions"):
        raise RuntimeError("junction drafting needs an updated Rust core")
    inputs = [Path(cloud), Path(vector_map)]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(str(path))
    out = Path(out_dir).resolve()
    if out.exists():
        raise FileExistsError(
            f"output directory already exists: {out}; choose a new directory"
        )
    parameters = {"max_gap": max_gap, "min_ground_support": min_ground_support}
    if check_boundary_support:
        parameters["check_boundary_support"] = True
    options = json.dumps(parameters, allow_nan=False)
    pairs = json.dumps(lane_pairs, allow_nan=False) if lane_pairs is not None else None
    payload = json.loads(
        module.connect_vector_map_junctions(
            str(inputs[0]), str(inputs[1]), options, pairs, preview_only
        )
    )
    report: dict[str, Any] = payload["report"]
    report["inputs"] = {
        "cloud": str(inputs[0].resolve()),
        "vector_map": str(inputs[1].resolve()),
    }
    report["options"] = json.loads(options)
    report["lane_pairs"] = json.loads(pairs) if pairs is not None else None
    return _publish(payload, out)


def measure_vector_map_signal(
    cloud: str,
    vector_map: str,
    out_dir: str,
    *,
    bounds: list[float],
    lanes: list[int],
    kind: str = "vehicle",
    preview_only: bool = True,
) -> dict[str, Any]:
    """Measure a user-identified head from an original-coordinate 3D box.

    bounds is xmin,ymin,zmin,xmax,ymax,zmax. Classification and controlled lanes
    are user supplied. Preview writes unchanged map artifacts; adding recomputes
    support and uses measured housing geometry without invented lamps/stop lines.
    LAS/LAZ/CSV stream selected points; local/HTTP COPC reads full-density boxes.
    Other formats retain the whole-file reader. No coordinate conversion occurs.
    """
    module = core()
    if module is None or not hasattr(module, "measure_vector_map_signal"):
        raise RuntimeError("signal measurement needs an updated Rust core")
    if not Path(vector_map).is_file():
        raise FileNotFoundError(vector_map)
    parsed = urlsplit(cloud)
    remote = parsed.scheme in {"http", "https"}
    if not remote and not Path(cloud).is_file():
        raise FileNotFoundError(cloud)
    out = Path(out_dir).resolve()
    if out.exists():
        raise FileExistsError(
            f"output directory already exists: {out}; choose a new directory"
        )
    if len(bounds) != 6 or not all(math.isfinite(v) for v in bounds) or any(
        bounds[i] >= bounds[i + 3] or bounds[i + 3] - bounds[i] > 10
        for i in range(3)
    ):
        raise ValueError("signal box requires six finite increasing bounds, at most 10 m per axis")
    if kind not in {"vehicle", "pedestrian"}:
        raise ValueError("kind must be vehicle or pedestrian")
    if (not lanes or any(type(lane) is not int or lane < 1 for lane in lanes)
            or len(set(lanes)) != len(lanes)):
        raise ValueError("select distinct positive controlled lane IDs explicitly")
    if remote and not parsed.path.lower().endswith(".laz"):
        raise ValueError("HTTP signal input must be a range-capable COPC .laz URL")
    options = {"min": bounds[:3], "max": bounds[3:], "lanes": lanes, "kind": kind}
    streaming = remote or Path(cloud).suffix.lower() in {".las", ".laz", ".csv"}
    encoded = json.dumps(options, allow_nan=False)
    if streaming:
        if not hasattr(module, "measure_vector_map_signal_points"):
            raise RuntimeError("spatial signal measurement needs an updated Rust core")
        import numpy as np

        from ca.io import iter_point_chunks

        box = (bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
        chunks = iter_point_chunks(cloud, chunk_size=10_000, bounds=box)
        selected = []
        count = 0
        try:
            for chunk in chunks:
                count += len(chunk)
                if count > 200_000:
                    raise ValueError("signal box exceeds 200000 points; isolate a smaller head")
                selected.append(chunk)
        finally:
            chunks.close()
        points = np.concatenate(selected) if selected else np.empty((0, 3), dtype=np.float64)
        payload = json.loads(module.measure_vector_map_signal_points(
            points, vector_map, encoded, preview_only
        ))
        strategy = ("copc-full-density-box" if remote or cloud.lower().endswith(".copc.laz")
                    else "sequential-filtered-chunks")
        payload["report"]["processing"] = {
            "strategy": strategy, "selected_points": count,
            "selected_limit": 200_000, "chunk_points": 10_000,
        }
    else:
        payload = json.loads(module.measure_vector_map_signal(cloud, vector_map, encoded, preview_only))
        payload["report"]["processing"] = {"strategy": "whole-file-compatibility", "selected_limit": 200_000}
    source_input = (urlunsplit((parsed.scheme, parsed.netloc.split("@")[-1], parsed.path, "", ""))
                    if remote else str(Path(cloud).resolve()))
    payload["report"]["inputs"] = {
        "cloud": source_input,
        "vector_map": str(Path(vector_map).resolve()),
    }
    payload["report"]["options"] = options
    return _publish(payload, out)


def measure_vector_map_crosswalk(
    cloud: str,
    vector_map: str,
    out_dir: str,
    *,
    bounds: list[float],
    lanes: list[int] | None = None,
    candidate: int = 0,
    brightness_fraction: float = 0.75,
    preview_only: bool = True,
) -> dict[str, Any]:
    """Propose ground-paint bands, then add an explicitly confirmed crossing.

    bounds is xmin,ymin,zmin,xmax,ymax,zmax in the map's metre frame. Retained
    RGB/intensity determines the band geometry; the map is used only to check
    user-supplied lane IDs. Preview is read-only and may omit lanes. Adding
    requires confirmed crossing lanes and a reviewed candidate index from the
    preview. No stop line or legal priority is inferred. Missing paint may
    shorten the measured footprint. Use the original map when adding.

    This local compatibility reader loads the complete attribute-bearing input;
    the 200000-point cap applies to the selected box, not total reader memory.
    Export an attribute-preserving ROI first for a large source. XYZ-only and
    HTTP inputs are not supported for paint measurement.
    """
    selected_lanes = [] if lanes is None else lanes
    if len(bounds) != 6 or not all(math.isfinite(v) for v in bounds) or any(
        bounds[i] >= bounds[i + 3] or bounds[i + 3] - bounds[i] > (5 if i == 2 else 40)
        for i in range(3)
    ):
        raise ValueError("crosswalk box requires six finite increasing bounds, XY at most 40 m and Z at most 5 m")
    if (any(type(lane) is not int or lane < 1 for lane in selected_lanes)
            or len(set(selected_lanes)) != len(selected_lanes)
            or (not preview_only and not selected_lanes)):
        raise ValueError("confirm distinct positive crossing lane IDs before adding")
    if type(candidate) is not int or candidate < 0:
        raise ValueError("candidate must be a nonnegative preview index")
    if not math.isfinite(brightness_fraction) or not 0.4 <= brightness_fraction <= 0.9:
        raise ValueError("brightness_fraction must be finite and between 0.4 and 0.9")
    if urlsplit(cloud).scheme in {"http", "https"}:
        raise ValueError("paint measurement requires a local attribute-preserving ROI or cloud")
    for path in (cloud, vector_map):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    out = Path(out_dir).resolve()
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}; choose a new directory")
    module = core()
    if module is None or not hasattr(module, "measure_vector_map_crosswalk"):
        raise RuntimeError("crosswalk paint measurement needs an updated Rust core")
    options = {"min": bounds[:3], "max": bounds[3:], "lanes": selected_lanes, "candidate": candidate, "brightness_fraction": brightness_fraction}
    payload = json.loads(module.measure_vector_map_crosswalk(
        cloud, vector_map, json.dumps(options, allow_nan=False), preview_only
    ))
    payload["report"]["options"] = options
    payload["report"]["inputs"] = {"cloud": str(Path(cloud).resolve()), "vector_map": str(Path(vector_map).resolve())}
    payload["report"]["processing"] = {"strategy": "whole-file-compatibility-with-attributes", "selected_limit": 200_000}
    return _publish(payload, out)


def discover_vector_map_features(
    cloud: str,
    out_dir: str,
    *,
    vector_map: str | None = None,
    scope: str = "road_corridor",
    corridor_radius: float = 12.0,
    brightness_fraction: float = 0.65,
    confirmations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Find paint and elevated-panel proposals without manually placed feature boxes.

    Use roads generated from the cloud and trajectory as vector_map, or scan all
    supported lower surfaces with scope='ground_surface' and no map. Surface
    anchors may include other levels; shapes do not establish object identity.
    Preview writes the unchanged map plus proposals to a NEW output directory.
    To add, pass explicit candidate/key/classification/lanes confirmations from
    the preview against the original map/source. Native code rechecks support
    and adds atomically. Nearby lanes are suggestions, never traffic semantics.
    Signal lamps, stop signs and signal/stop-line relationships are not inferred.

    This local compatibility reader loads the whole attribute-bearing source.
    Whole-ground search supports at most 2M source points; road corridors at
    most 2M selected points. Export an attribute-preserving scene for larger
    inputs. These caps do not bound the reader's peak memory.
    """
    if scope not in {"road_corridor", "ground_surface"}:
        raise ValueError("scope must be road_corridor or ground_surface")
    if scope == "road_corridor" and vector_map is None:
        raise ValueError("road_corridor search requires generated roads")
    if not math.isfinite(corridor_radius) or not 4 <= corridor_radius <= 18:
        raise ValueError("corridor_radius must be finite and between 4 and 18 metres")
    if not math.isfinite(brightness_fraction) or not 0.4 <= brightness_fraction <= 0.9:
        raise ValueError("brightness_fraction must be finite and between 0.4 and 0.9")
    if urlsplit(cloud).scheme in {"http", "https"}:
        raise ValueError("feature discovery requires a local attribute-preserving scene")
    for path in [cloud] + ([vector_map] if vector_map else []):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    out = Path(out_dir).resolve()
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}; choose a new directory")
    module = core()
    if module is None or not hasattr(module, "discover_vector_map_features"):
        raise RuntimeError("automatic feature discovery needs an updated Rust core")
    options = {"scope": scope, "corridor_radius": corridor_radius, "brightness_fraction": brightness_fraction}
    encoded = None if confirmations is None else json.dumps(confirmations, allow_nan=False)
    payload = json.loads(module.discover_vector_map_features(cloud, vector_map, json.dumps(options, allow_nan=False), encoded))
    payload["report"]["options"] = options
    payload["report"]["inputs"] = {"cloud": str(Path(cloud).resolve()), "vector_map": str(Path(vector_map).resolve()) if vector_map else None}
    payload["report"]["processing"] = {"strategy": "whole-file-compatibility-with-attributes", "corridor_point_limit": 2_000_000}
    return _publish(payload, out)


def _publish(payload: dict[str, Any], out: Path) -> dict[str, Any]:
    report: dict[str, Any] = payload["report"]
    names = {
        "map": "lanelet2_map.osm",
        "projector": "map_projector_info.yaml",
        "editable_map": "vector_map.json",
        "report": "report.json",
    }
    report["files"] = {key: str(out / name) for key, name in names.items()}
    report["editing"] = {
        "command": "vectormap",
        "args": ["mcp", str(out / names["map"])],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # Stage all artifacts together; failed generation never leaves half a result.
    with tempfile.TemporaryDirectory(
        prefix=".vector-map-", dir=out.parent
    ) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        content = {
            names["map"]: payload["osm"],
            names["projector"]: payload["projector_info"],
            names["editable_map"]: payload["map_json"],
            names["report"]: json.dumps(report, indent=2) + "\n",
        }
        for name, text in content.items():
            with (stage / name).open("w", encoding="utf-8", newline="\n") as file:
                file.write(text)
        os.rename(stage, out)
    return report


def edit_vector_map_relations(
    vector_map: str,
    *,
    rule_id: int | None = None,
    lanes: list[int] | None = None,
    controlled_crosswalks: list[int] | None = None,
    stop_lines: list[int] | None = None,
    out_dir: str | None = None,
) -> dict[str, Any]:
    """Inspect equipment associations, or explicitly replace targets into a NEW directory.

    Omit rule_id and out_dir for read-only inspection. Editing requires both.
    Pedestrian signals control crosswalks; vehicle signals control lanes and an
    optional previously reviewed transverse stop marking. Empty lists retain
    unresolved targets. Physical geometry, lamps and source provenance stay fixed.
    These are operator-reviewed drafts, not inferred legal control or phases.
    """
    source = Path(vector_map)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    values = {"lanes": lanes or [], "controlled_crosswalks": controlled_crosswalks or [], "stop_lines": stop_lines or []}
    if rule_id is None:
        if out_dir is not None or any(values.values()):
            raise ValueError("inspection omits output and edit targets")
        encoded = None
    else:
        if type(rule_id) is not int or rule_id <= 0 or not out_dir:
            raise ValueError("editing requires a positive rule_id and a new output directory")
        if any(len(v) > 128 or any(type(i) is not int or i <= 0 for i in v) for v in values.values()):
            raise ValueError("targets must contain at most 128 positive integer IDs")
        out = Path(out_dir).resolve()
        if out.exists():
            raise FileExistsError(str(out))
        encoded = json.dumps({"rule_id": rule_id, **values}, allow_nan=False)
    module = core()
    if module is None or not hasattr(module, "edit_vector_map_relations"):
        raise RuntimeError("association review needs an updated CloudAnalyzer Rust core")
    payload = json.loads(module.edit_vector_map_relations(str(source), encoded))
    payload["report"]["input"] = str(source.resolve())
    return _publish(payload, out) if rule_id is not None else payload["report"]


def propose_vector_map_relations(
    vector_map: str,
    rule_id: int,
    *,
    candidate_key: str | None = None,
    map_snapshot: str | None = None,
    out_dir: str | None = None,
) -> dict[str, Any]:
    """Preview geometric signal targets; explicitly adopt one into a NEW directory.

    Preview is read-only and selects nothing. Adoption requires candidate_key,
    map_snapshot from a fresh preview and out_dir together. Rust recomputes the
    candidate and rejects stale, incomplete or unsupported evidence. Distance,
    unsigned housing orientation, road context and elevation support geometric
    drafts only; they do not prove legal control, front face or signal phases.
    """
    source = Path(vector_map)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    if type(rule_id) is not int or not 0 < rule_id <= 2**64 - 1:
        raise ValueError("rule_id must be a positive integer ID")
    adopting = any(v is not None for v in (candidate_key, map_snapshot, out_dir))
    encoded = None
    if adopting:
        if not isinstance(candidate_key, str) or not candidate_key or not isinstance(map_snapshot, str) or not map_snapshot or not isinstance(out_dir, str) or not out_dir:
            raise ValueError("adoption requires candidate_key, map_snapshot and a new output directory together")
        out = Path(out_dir).resolve()
        if out.exists():
            raise FileExistsError(str(out))
        encoded = json.dumps({"rule_id": rule_id, "candidate_key": candidate_key, "map_snapshot": map_snapshot})
    module = core()
    if module is None or not hasattr(module, "propose_vector_map_relations"):
        raise RuntimeError("target proposals need an updated CloudAnalyzer Rust core")
    payload = json.loads(module.propose_vector_map_relations(str(source), rule_id, encoded))
    payload["report"]["input"] = str(source.resolve())
    return _publish(payload, out) if adopting else payload["report"]
