"""Preserve chosen source curves as editable geometry, without inventing lanes."""

from __future__ import annotations

import math
from typing import Any


def assemble_geometry(report: dict[str, Any], decisions: list[dict[str, Any]], reason: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Make an IR and full input-station disposition from explicit bounded choices."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("geometry drafting needs a reason")
    if not isinstance(decisions, list) or not 1 <= len(decisions) <= 256:
        raise ValueError("supply 1..256 corridor decisions")
    candidates = {c["id"]: c for c in report["candidates"]}
    choices: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    for item in decisions:
        if not isinstance(item, dict) or set(item) - {"candidate_id", "action", "reason", "from_m", "to_m"}:
            raise ValueError("decisions allow candidate_id, action, reason and optional from_m/to_m only")
        cid = item.get("candidate_id")
        if type(cid) is not int or cid not in candidates:
            raise ValueError("decision needs a known corridor candidate_id")
        if item.get("action") not in ("include", "defer") or not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("each decision needs include/defer and a nonempty reason")
        candidate = candidates[cid]
        start, stop = item.get("from_m", candidate["from_m"]), item.get("to_m", candidate["to_m"])
        stations = [s["station_m"] for s in candidate["sections"]]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, stop)) or start >= stop or start not in stations or stop not in stations:
            raise ValueError("decision range must use increasing observed candidate stations; no extrapolation")
        choice = {"candidate_id": cid, "action": item["action"], "reason": item["reason"].strip(), "from_m": start, "to_m": stop}
        if any(c["candidate_id"] == cid and max(start, c["from_m"]) < min(stop, c["to_m"]) for c in choices):
            raise ValueError("overlapping decisions for the same candidate")
        choices.append(choice)
        if item["action"] == "defer":
            continue
        if any(max(start, s["from_m"]) < min(stop, s["to_m"]) for s in segments):
            raise ValueError("included candidates overlap input stations; choose one band per interval")
        sections = [s for s in candidate["sections"] if start <= s["station_m"] <= stop]
        references = {}
        for role in ("center", "left", "right"):
            bid = len(boundaries) + 1
            references[role] = bid
            geometry = [s[role] for s in sections]
            if any(len(p) != 3 or any(not math.isfinite(v) for v in p) for p in geometry):
                raise ValueError("source curve contains invalid XYZ")
            boundaries.append({"id": bid, "kind": {"type": "other"}, "geometry": geometry,
                "attributes": {"cloudanalyzer:role": f"source_{role}", "cloudanalyzer:corridor": str(cid),
                               "cloudanalyzer:review_required": "true", "cloudanalyzer:road_semantics": "unresolved"}})
        segments.append({**choice, "curve_ids": references, "sections": sections, "review_required": True,
                         "complete_width_resolved": False, "lane_count": None, "traffic_direction": None, "speed_limit": None})
    if not segments:
        raise ValueError("include at least one observed corridor interval")
    segments.sort(key=lambda s: s["from_m"])
    # Partition the entire request, including unreviewed and source-deferred extent.
    total = report["trajectory_length_m"]
    breaks = sorted({0.0, total, *[v for c in candidates.values() for v in (c["from_m"], c["to_m"])],
                     *[v for c in choices for v in (c["from_m"], c["to_m"])],
                     *[v for c in report["deferred_intervals"] for v in (c["from_m"], c["to_m"])],
                     *[v for c in report["ambiguous_intervals"] for v in (c["from_m"], c["to_m"])]})
    disposition = []
    for start, stop in zip(breaks, breaks[1:]):
        offered = [cid for cid, c in candidates.items() if c["from_m"] <= start and stop <= c["to_m"]]
        included = [s["candidate_id"] for s in segments if s["from_m"] <= start and stop <= s["to_m"]]
        deferred = [c["candidate_id"] for c in choices if c["action"] == "defer" and c["from_m"] <= start and stop <= c["to_m"]]
        source = [i["reason"] for i in report["deferred_intervals"] if i["from_m"] <= start and stop <= i["to_m"]]
        status = "included_geometry" if included else "source_deferred" if not offered else "agent_deferred" if set(offered) <= set(deferred) else "not_reviewed"
        disposition.append({"from_m": start, "to_m": stop, "status": status, "included_candidate_id": included[0] if included else None,
            "available_candidate_ids": offered, "deferred_candidate_ids": deferred, "source_reasons": source,
            "source_ambiguous": any(i["from_m"] <= start and stop <= i["to_m"] for i in report["ambiguous_intervals"])})
    included_length = sum(s["to_m"] - s["from_m"] for s in segments)
    ir = {"format": "vectormap-ir", "version": 1, "metadata": {"name": "Source corridor geometry draft",
          "attributes": {"cloudanalyzer:draft": "surface_geometry", "cloudanalyzer:coordinate_frame": report["coordinate_frame"],
                         "cloudanalyzer:road_semantics": "unresolved", "cloudanalyzer:deployment_ready": "false"}},
          "boundaries": boundaries, "lanes": [], "roads": [], "topology": []}
    evidence = {"schema": "cloudanalyzer.mapping_geometry.v1", "reason": reason.strip(), "coordinate_frame": report["coordinate_frame"],
        "protocol": report["protocol"], "source_proposal_limited": report["limited"], "decisions": choices, "segments": segments,
        "station_disposition": disposition,
        "summary": {"segments": len(segments), "reference_curves": len(boundaries), "trajectory_length_m": total,
                    "included_station_length_m": included_length, "unresolved_station_length_m": total - included_length,
                    "lane_count": None, "complete_width_resolved": False, "road_semantics_inferred": False,
                    "source_quality_passed": False, "deployment_ready": False},
        "warnings": ["Included source curves remain review geometry. No complete width, lane, direction, speed, legal use, connectivity, georeference or full-width interior is established.",
                     "Other reference curves use the editable IR; standalone unknown ways are discarded by the Lanelet2 reader, so this stage does not publish OSM.",
                     "Station coverage is input-path extent, not unique road length or a source-quality pass. Separate candidates are not joined or smoothed."],
        "source_warnings": report["warnings"]}
    return ir, evidence


def corridor_lane_requests(geometry: dict[str, Any], lane_specs: list[dict[str, Any]], boundary_policy: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate explicit layout hypotheses and bind them to saved segment curves."""
    if boundary_policy != "source_span_hypothesis":
        raise ValueError("explicitly choose boundary_policy=source_span_hypothesis; complete width remains unresolved")
    if not isinstance(lane_specs, list) or not 1 <= len(lane_specs) <= 256:
        raise ValueError("supply 1..256 lane specifications")
    segments = {s["curve_ids"]["center"]: s for s in geometry["segments"]}
    requests, chosen = [], []
    seen = set()
    # The saved source-audit protocol currently evaluates driving lanes only.
    kinds = {"driving"}
    def number(value: Any, low: float, high: float) -> bool:
        return type(value) in (int, float) and math.isfinite(value) and low <= value <= high
    for spec in lane_specs:
        if not isinstance(spec, dict) or set(spec) != {"center_curve_id", "lanes", "speed_limit_kmh", "reason"}:
            raise ValueError("each specification needs center_curve_id, lanes, explicit speed_limit_kmh and reason")
        cid = spec["center_curve_id"]
        if type(cid) is not int or cid not in segments or cid in seen:
            raise ValueError("use distinct centre curve IDs from the retained geometry draft")
        if not isinstance(spec["reason"], str) or not spec["reason"].strip():
            raise ValueError("each lane specification needs a reason")
        lanes = spec["lanes"]
        if not isinstance(lanes, list) or not 1 <= len(lanes) <= 16:
            raise ValueError("supply 1..16 ordered lane hypotheses per source segment")
        for lane in lanes:
            if not isinstance(lane, dict) or set(lane) != {"direction", "kind", "one_way", "fraction", "minimum_width_m"}:
                raise ValueError("lanes need explicit direction, kind, one_way, fraction and minimum_width_m")
            if lane["direction"] not in ("forward", "backward") or not isinstance(lane["kind"], str) or lane["kind"] not in kinds or type(lane["one_way"]) is not bool:
                raise ValueError("use supported lane kind/direction and boolean one_way")
            if not number(lane["fraction"], 1e-9, 1) or not number(lane["minimum_width_m"], .5, 10):
                raise ValueError("lane fractions must be positive; minimum_width_m must be within 0.5..10")
        if abs(sum(lane["fraction"] for lane in lanes) - 1) > 1e-9:
            raise ValueError("lane fractions must sum to 1")
        speed = spec["speed_limit_kmh"]
        if (speed is not None and not number(speed, .1, 200)) or (speed is None and any(l["kind"] in {"driving", "bus", "emergency"} for l in lanes)):
            raise ValueError("vehicle lanes require an explicit speed_limit_kmh within 0.1..200")
        segment = segments[cid]
        requests.append({"center_curve_id": cid, "left_curve_id": segment["curve_ids"]["left"],
                         "right_curve_id": segment["curve_ids"]["right"], "lanes": lanes, "speed_limit_kmh": speed})
        chosen.append(segment)
        seen.add(cid)
    return requests, chosen


def verify_lane_roundtrip(original: dict[str, Any], reopened: dict[str, Any]) -> None:
    """Require OSM reload to retain lane membership, geometry and assigned semantics."""
    source = {l["id"]: l for l in original.get("lanes", [])}
    target = {l["id"]: l for l in reopened.get("lanes", [])}
    if not source or source.keys() != target.keys():
        raise ValueError("OSM reload did not retain every hypothesized lane")
    def ref(value: Any) -> tuple[int, bool]:
        return (value, False) if type(value) is int else (value["boundary"], value.get("reversed", False))
    boundaries = [{b["id"]: b for b in m.get("boundaries", [])} for m in (original, reopened)]
    for lid, lane in source.items():
        other = target[lid]
        if lane.get("kind", "driving") != other.get("kind", "driving") or lane.get("one_way", True) != other.get("one_way", True):
            raise ValueError("OSM reload changed lane kind or one-way hypothesis")
        a, b = lane.get("speed_limit", {}).get("kmh"), other.get("speed_limit", {}).get("kmh")
        if (a is None) != (b is None) or (a is not None and not math.isclose(a, b, abs_tol=1e-6)):
            raise ValueError("OSM reload changed the speed hypothesis")
        for side in ("left", "right"):
            if ref(lane[side]) != ref(other[side]):
                raise ValueError("OSM reload changed lane orientation or shared boundaries")
            bid, _ = ref(lane[side])
            a_points, b_points = boundaries[0][bid]["geometry"], boundaries[1][bid]["geometry"]
            if len(a_points) != len(b_points) or any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-9) for p, q in zip(a_points, b_points) for a, b in zip(p, q)):
                raise ValueError("OSM reload changed source-derived boundary geometry")
