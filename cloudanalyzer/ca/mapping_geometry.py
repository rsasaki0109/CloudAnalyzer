"""Preserve chosen source curves as editable geometry, without inventing lanes."""

from __future__ import annotations

import json
import math
from typing import Any, Callable


def _span_minimum(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    norm = dx * dx + dy * dy
    t = max(0., min(1., -(a[0] * dx + a[1] * dy) / norm)) if norm else 0.
    return math.hypot(a[0] + t * dx, a[1] + t * dy), t


def _append_width_range(ranges: list[dict[str, Any]], lo: float, hi: float,
                        width: float, contains_path: bool) -> None:
    if lo >= hi:
        return
    if ranges and ranges[-1]["to_m"] == lo:
        active = ranges[-1]
        active.update(to_m=hi, minimum_interpolated_span_m=min(active["minimum_interpolated_span_m"], width),
                      section_pairs=active["section_pairs"] + 1,
                      path_contained_at_all_sections=active["path_contained_at_all_sections"] and contains_path)
    else:
        ranges.append({"from_m": lo, "to_m": hi, "minimum_interpolated_span_m": width,
                       "section_pairs": 1, "path_contained_at_all_sections": contains_path})


def preview_layout_width(candidate: dict[str, Any], layout: dict[str, Any]) -> dict[str, Any]:
    """Screen inspected section pairs against the fixed lane fractions and widths.

    A narrow section does not discard the rest of a long candidate. Ranges use
    observed endpoints only. Additional interpolated_ranges can end within an
    inspected section pair, with explicit endpoint_policy=linear_interpolation.
    Check the minimum XY span between endpoints,
    including a possible pinch between two wide sections. This is a width screen,
    not a ground audit, lane-generation result or adoption decision. Only the
    bounded section preview is checked; uninspected tails remain unresolved.
    """
    required = max(lane["minimum_width_m"] / lane["fraction"] for lane in layout["lanes"])
    sections = candidate["sections"]
    spans = []
    for section in sections:
        left, right = section["left"], section["right"]
        if any(len(p) != 3 or any(not math.isfinite(v) or abs(v) > 1e15 for v in p) for p in (left, right)):
            raise ValueError("width preview needs finite observed boundary sections")
        spans.append((left[0] - right[0], left[1] - right[1]))
    ranges: list[dict[str, Any]] = []
    interpolated: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(zip(spans, spans[1:])):
        lo, hi = sections[i]["station_m"], sections[i + 1]["station_m"]
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            raise ValueError("width preview needs increasing observed stations")
        width, t = _span_minimum(a, b)
        contains_path = all(s.get("intersects_trajectory") is True for s in sections[i:i + 2])
        if all(width * lane["fraction"] + 1e-6 >= lane["minimum_width_m"] for lane in layout["lanes"]):
            _append_width_range(ranges, lo, hi, width, contains_path)
            _append_width_range(interpolated, lo, hi, width, contains_path)
        else:
            # The XY span is convex on each side of its minimum. Keep the good
            # side of each bracket; use the nominal width, not the lane tolerance,
            # at newly derived endpoints so rounding does not consume that margin.
            def span(at: float) -> tuple[float, float]:
                return (a[0] + at * (b[0] - a[0]), a[1] + at * (b[1] - a[1]))
            for edge in (0., 1.):
                if math.hypot(*span(edge)) < required:
                    continue
                good, bad = edge, t
                for _ in range(60):
                    middle = (good + bad) / 2
                    if math.hypot(*span(middle)) >= required:
                        good = middle
                    else:
                        bad = middle
                start, stop = sorted((edge, good))
                minimum, _ = _span_minimum(span(start), span(stop))
                _append_width_range(interpolated, lo if start == 0 else lo + (hi - lo) * start,
                                    hi if stop == 1 else lo + (hi - lo) * stop, minimum, contains_path)
    checked = max(0, len(sections) - 1)
    total = max(0, candidate.get("total_sections", len(sections)) - 1)
    return {"required_span_m": required, "lane_width_tolerance_m": 1e-6,
            "measurement": "minimum_xy_span_of_linearly_interpolated_observed_boundaries",
            "checked_section_pairs": checked, "total_section_pairs": total,
            "uninspected_section_pairs": total - checked,
            "width_matching_station_length_m": sum(r["to_m"] - r["from_m"] for r in ranges),
            "ranges": ranges[:16], "ranges_total": len(ranges), "ranges_limited_in_response": len(ranges) > 16,
            "interpolated_ranges": [{**r, "endpoint_policy": "linear_interpolation"} for r in interpolated[:16]],
            "interpolated_ranges_total": len(interpolated),
            "interpolated_ranges_limited_in_response": len(interpolated) > 16,
            "interpolated_width_matching_station_length_m": sum(r["to_m"] - r["from_m"] for r in interpolated),
            "interpolated_path_containment_checked": False,
            "source_support_checked": False, "lane_generation_checked": False,
            "adopted": False}


def _clipped_sections(sections: list[dict[str, Any]], start: float, stop: float) -> list[dict[str, Any]]:
    """Clip existing piecewise linear curves, recording both source brackets."""
    def endpoint(station: float) -> dict[str, Any]:
        for section in sections:
            if section["station_m"] == station:
                return section
        for a, b in zip(sections, sections[1:]):
            if a["station_m"] < station < b["station_m"]:
                t = (station - a["station_m"]) / (b["station_m"] - a["station_m"])
                points = {}
                for role in ("center", "left", "right"):
                    if any(len(s[role]) != 3 or any(not math.isfinite(v) for v in s[role]) for s in (a, b)):
                        raise ValueError("source curve contains invalid XYZ")
                    points[role] = [x + t * (y - x) for x, y in zip(a[role], b[role])]
                return {"station_m": station, **points, "interpolated": True,
                        "source_stations_m": [a["station_m"], b["station_m"]]}
        raise ValueError("interpolated range must stay within observed candidate stations; no extrapolation")
    return [endpoint(start), *[s for s in sections if start < s["station_m"] < stop], endpoint(stop)]


def preview_lane_directions(candidate: dict[str, Any], layout: dict[str, Any],
                            preflight: Callable[[str, str], str] | None) -> dict[str, Any]:
    """Apply the builder's direction planning to inspected width ranges, without lanes.

    This checks each clipped range as a whole and each adjacent section pair
    separately. An adjacent pair alone need not predict a longer range. Missing
    native support stays explicitly unavailable; inspection remains read-only.
    """
    width = candidate["layout_width_preview"]
    limited = width["interpolated_ranges_limited_in_response"] or width["uninspected_section_pairs"] > 0
    answer: dict[str, Any] = {"available": preflight is not None, "ranges": [], "adjacent_interval_checks": [],
        "native_check_completed": False,
        "limited": limited, "geometry_generated": False, "source_support_checked": False,
        "roundtrip_checked": False, "adopted": False}
    if preflight is None:
        answer["reason"] = "Native corridor direction preflight is unavailable"
        return answer
    boundaries: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []
    checks: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    ranges = width["interpolated_ranges"]
    if not ranges:
        return answer
    for interval in ranges:
        sections = _clipped_sections(candidate["sections"], interval["from_m"], interval["to_m"])
        checks.append(("ranges", interval, sections))
        checks.extend(("adjacent_interval_checks", {"from_m": a["station_m"], "to_m": b["station_m"]}, [a, b])
                      for a, b in zip(sections, sections[1:]))
    for _, _, sections in checks:
        cid = len(boundaries) + 1
        for role in ("center", "left", "right"):
            boundaries.append({"id": len(boundaries) + 1, "kind": {"type": "other"},
                "geometry": [s[role] for s in sections],
                "attributes": {"cloudanalyzer:role": f"source_{role}"}})
        specs.append({"center_curve_id": cid, "left_curve_id": cid + 1, "right_curve_id": cid + 2,
            "lanes": layout["lanes"], "speed_limit_kmh": layout["speed_limit_kmh"]})
    geometry = {"format": "vectormap-ir", "version": 1,
        "metadata": {"attributes": {"cloudanalyzer:draft": "surface_geometry"}},
        "boundaries": boundaries, "lanes": [], "roads": [], "topology": []}
    try:
        encoded = preflight(json.dumps(geometry, allow_nan=False), json.dumps(specs, allow_nan=False))
    except ValueError as error:
        answer["error"] = str(error)
        return answer
    result = json.loads(encoded)
    if result["schema"] != "cloudanalyzer.corridor_lane_direction_preview.v1":
        raise ValueError("unexpected native corridor direction preview schema")
    by_id = {s["center_curve_id"]: s for s in result["segments"]}
    if set(by_id) != {s["center_curve_id"] for s in specs}:
        raise ValueError("native direction preview lost inspected ranges")
    for spec, (scope, interval, sections) in zip(specs, checks):
        segment = by_id[spec["center_curve_id"]]
        def station_range(indices: list[int]) -> list[float]:
            a, b = indices
            if not 0 <= a < b < len(sections):
                raise ValueError("native direction preview returned an unseen section")
            return [sections[a]["station_m"], sections[b]["station_m"]]
        answer[scope].append({"from_m": interval["from_m"], "to_m": interval["to_m"],
            "endpoint_policy": "linear_interpolation", "direction_checks_passed": segment["direction_checks_passed"],
            "planned_pieces_total": segment["planned_pieces_total"],
            "planned_pieces_limited": segment["planned_pieces_limited"],
            "planned_station_pieces": [station_range(piece) for piece in segment["planned_pieces"]],
            "unresolved_intervals": segment["unresolved_intervals"], "issues_limited": segment["issues_limited"],
            "issues": [{**issue, "station_range_m": station_range(issue["section_range"])} for issue in segment["issues"]]})
    answer["limited"] |= result["limited"]
    answer["native_check_completed"] = True
    return answer


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
        if not isinstance(item, dict) or set(item) - {"candidate_id", "action", "reason", "from_m", "to_m", "endpoint_policy"}:
            raise ValueError("decisions allow candidate_id, action, reason, optional from_m/to_m and endpoint_policy only")
        cid = item.get("candidate_id")
        if type(cid) is not int or cid not in candidates:
            raise ValueError("decision needs a known corridor candidate_id")
        if item.get("action") not in ("include", "defer") or not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("each decision needs include/defer and a nonempty reason")
        candidate = candidates[cid]
        start, stop = item.get("from_m", candidate["from_m"]), item.get("to_m", candidate["to_m"])
        stations = [s["station_m"] for s in candidate["sections"]]
        policy = item.get("endpoint_policy", "observed")
        if policy not in ("observed", "linear_interpolation"):
            raise ValueError("endpoint_policy must be observed or linear_interpolation")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, stop)) or start >= stop:
            raise ValueError("decision range must use increasing observed candidate stations; no extrapolation")
        if policy == "observed" and (start not in stations or stop not in stations):
            raise ValueError("decision range must use increasing observed candidate stations; no extrapolation")
        if policy == "linear_interpolation" and (not stations or any(not math.isfinite(s) for s in stations)
                or any(a >= b for a, b in zip(stations, stations[1:]))
                or not stations[0] <= start < stop <= stations[-1]):
            raise ValueError("interpolated range must stay within increasing observed candidate stations; no extrapolation")
        choice = {"candidate_id": cid, "action": item["action"], "reason": item["reason"].strip(), "from_m": start, "to_m": stop}
        if "endpoint_policy" in item:
            choice["endpoint_policy"] = policy
        if any(c["candidate_id"] == cid and max(start, c["from_m"]) < min(stop, c["to_m"]) for c in choices):
            raise ValueError("overlapping decisions for the same candidate")
        choices.append(choice)
        if item["action"] == "defer":
            continue
        if any(max(start, s["from_m"]) < min(stop, s["to_m"]) for s in segments):
            raise ValueError("included candidates overlap input stations; choose one band per interval")
        sections = (_clipped_sections(candidate["sections"], start, stop) if policy == "linear_interpolation"
                    else [s for s in candidate["sections"] if start <= s["station_m"] <= stop])
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
        a_center, b_center = lane.get("centerline"), other.get("centerline")
        if (a_center is None) != (b_center is None) or (a_center is not None and (
            len(a_center) != len(b_center) or any(
                len(p) != 3 or len(q) != 3 or any(
                    not math.isclose(a, b, rel_tol=0, abs_tol=1e-9) for a, b in zip(p, q)
                ) for p, q in zip(a_center, b_center)
            )
        )):
            raise ValueError("OSM reload changed the explicit source-section centerline")
        for side in ("left", "right"):
            if ref(lane[side]) != ref(other[side]):
                raise ValueError("OSM reload changed lane orientation or shared boundaries")
            bid, _ = ref(lane[side])
            a_points, b_points = boundaries[0][bid]["geometry"], boundaries[1][bid]["geometry"]
            if len(a_points) != len(b_points) or any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-9) for p, q in zip(a_points, b_points) for a, b in zip(p, q)):
                raise ValueError("OSM reload changed source-derived boundary geometry")
