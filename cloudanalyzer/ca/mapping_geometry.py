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
