"""Explicit, source-checked short connections between consecutive drive pieces."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ca import mapping_job as jobs
from ca.mapping_geometry import verify_lane_roundtrip
from ca.vector_map import _publish

OPTIONS = {"max_gap": 10., "min_ground_support": 1., "check_boundary_support": True}


def edges(ir: dict[str, Any]) -> set[tuple[int, int]]:
    """Read directed topology from both successor and predecessor declarations."""
    result: set[tuple[int, int]] = set()
    ids = {lane["id"] for lane in ir["lanes"]}
    for link in ir.get("topology", []):
        result.update((link["lane"], to) for to in link.get("successors", []))
        result.update((frm, link["lane"]) for frm in link.get("predecessors", []))
    if any(frm not in ids or to not in ids or frm == to for frm, to in result):
        raise ValueError("route topology contains invalid lane references")
    return result


def route_metrics(ir: dict[str, Any], intervals: dict[int, tuple[float, float]]) -> dict[str, Any]:
    """Measure explicit forward chains in ORIGINAL drive stations, not map length."""
    ids = {lane["id"] for lane in ir["lanes"]}
    if ids != intervals.keys():
        raise ValueError("every route lane needs its original drive interval")
    successors: dict[int, int] = {}
    predecessors: dict[int, int] = {}
    for frm, to in edges(ir):
        if frm in successors or to in predecessors or abs(intervals[frm][1] - intervals[to][0]) > 1e-6:
            raise ValueError("connections must form unambiguous station-contiguous forward chains")
        successors[frm], predecessors[to] = to, frm
    routes: list[dict[str, Any]] = []
    visited = set()
    for start in sorted(ids - predecessors.keys(), key=lambda lid: intervals[lid][0]):
        chain = [start]
        while chain[-1] in successors:
            if successors[chain[-1]] in chain:
                raise ValueError("route contains a cycle")
            chain.append(successors[chain[-1]])
        visited.update(chain)
        lo, hi = intervals[start][0], intervals[chain[-1]][1]
        routes.append({"lane_ids": chain, "from_m": lo, "to_m": hi, "station_span_m": hi - lo})
    if visited != ids:
        raise ValueError("route contains a cycle or unreachable topology")
    return {"measurement": "original_recorded_xy_station_span_along_explicit_lane_graph",
            "connected_components": len(routes), "longest_route_station_span_m": max(r["station_span_m"] for r in routes),
            "routes": routes, "legal_routing_verified": False}


def _parent(job: dict[str, Any], cid: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    from ca import mapping_retry as retries, mapping_patch as patches
    parent = next((a for a in job["attempts"] if a["id"] == cid), None)
    if parent is None or parent.get("kind") not in {"corridor_lanes", "connected_corridor_lanes", "patched_corridor_lanes"} or parent["status"] != "audited_draft":
        raise ValueError("inspect connections on an audited corridor lane draft")
    parent = retries._parent(job, cid)
    specs = parent["road_options"]["lane_specs"]
    if any(len(s["lanes"]) != 1 or s["lanes"][0]["direction"] != "forward" or not s["lanes"][0]["one_way"] for s in specs):
        raise ValueError("short connections currently require one forward one-way driving lane per piece")
    ir = json.loads(Path(parent["files"]["editable_map"]["path"]).read_text())
    if not 1 <= len(ir["lanes"]) <= 256:
        raise ValueError("short connections need at most 256 lanes")
    report = json.loads(Path(parent["files"]["report"]["path"]).read_text())
    intervals = patches._intervals(parent)
    if "lane_minimum_width_m" in report:
        widths = {int(k): v for k, v in report["lane_minimum_width_m"].items()}
    elif parent["kind"] == "patched_corridor_lanes":
        # Patch validation froze one uniform run layout across retained/addition lanes.
        fixed = {s["lanes"][0]["minimum_width_m"] for s in specs}
        if len(fixed) != 1:
            raise ValueError("patched lanes need their fixed uniform minimum width")
        widths = {lid: next(iter(fixed)) for lid in intervals}
    else:
        widths = {b["lane_ids"][0]: next(s["lanes"][0]["minimum_width_m"] for s in specs if s["center_curve_id"] == b["center_curve_id"])
                  for b in report["built_segments"] if len(b["lane_ids"]) == 1}
        for added in parent.get("routes", {}).get("added", []):
            widths[added["lane"]] = max(widths[added["from"]], widths[added["to"]])
    if widths.keys() != intervals.keys() or any(not math.isfinite(v) or v <= 0 for v in widths.values()):
        raise ValueError("every lane needs its fixed minimum width")
    pieces = sorted([{"lane": lid, "from_m": lo, "to_m": hi, "minimum_width_m": widths[lid]}
                     for lid, (lo, hi) in intervals.items()], key=lambda p: (p["from_m"], p["lane"]))
    if {p["lane"] for p in pieces} != {l["id"] for l in ir["lanes"]}:
        raise ValueError("source pieces do not account for all draft lanes")
    if any(not math.isfinite(p[k]) for p in pieces for k in ("from_m", "to_m")) or any(p["from_m"] >= p["to_m"] for p in pieces):
        raise ValueError("invalid original lane stations")
    if any(a["to_m"] > b["from_m"] + 1e-6 for a, b in zip(pieces, pieces[1:])):
        raise ValueError("source lane intervals overlap")
    route_metrics(ir, intervals)
    return parent, ir, report, pieces


def _inside(points: np.ndarray, polygon: np.ndarray) -> bool:
    """Even-odd containment including boundary points within numerical tolerance."""
    a, b = polygon, np.roll(polygon, -1, axis=0)
    delta = b - a
    for point in points:
        length = np.sum(delta * delta, axis=1)
        fraction = np.clip(np.sum((point - a) * delta, axis=1) / np.maximum(length, 1e-30), 0, 1)
        if np.min(np.linalg.norm(point - (a + fraction[:, None] * delta), axis=1)) <= 1e-7:
            continue
        crosses = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
        selected = np.flatnonzero(crosses)
        x = a[selected, 0] + (point[1] - a[selected, 1]) * delta[selected, 0] / delta[selected, 1]
        if np.count_nonzero(point[0] < x) % 2 != 1:
            return False
    return True


def _width(candidate: dict[str, Any]) -> float:
    # Exact minimum between equal normalized XY-arc positions on linear sides.
    sides = []
    for side in ("left", "right"):
        curve = np.asarray(candidate[side], dtype=float)[:, :2]
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        if not np.isfinite(curve).all() or distance[-1] <= 0:
            raise ValueError("invalid connector boundary")
        sides.append((curve, distance / distance[-1]))
    ts = np.unique(np.concatenate([t for _, t in sides]))
    samples = [np.column_stack([np.interp(ts, t, curve[:, k]) for k in (0, 1)]) for curve, t in sides]
    width = samples[0] - samples[1]
    delta = np.diff(width, axis=0)
    fraction = np.clip(-np.sum(width[:-1] * delta, axis=1) / np.maximum(np.sum(delta * delta, axis=1), 1e-30), 0, 1)
    return float(np.min(np.linalg.norm(width[:-1] + fraction[:, None] * delta, axis=1)))


def inspect_connections(root: Path, cid: int, offset: int) -> dict[str, Any]:
    if type(offset) is not int or offset < 0:
        raise ValueError("connection offset must be a nonnegative integer")
    with jobs._locked(root):
        job = jobs._load(root)
        jobs._inputs(job)
        parent, ir, _, pieces = _parent(job, cid)
        path = root / f"connections-{cid:02d}.json"
        inputs = {**{f"parent_{k}": v for k, v in parent["files"].items()}, "parent_audits": parent["quality_report"],
                  **{f"geometry_{k}": v for k, v in parent["geometry_inputs"].items()}, "proposal": parent["corridor_proposal"],
                  **{f"pointcloud_{k}": v for k, v in job["pointcloud"]["files"].items()}, "source": job["source"],
                  "native": job["runtime"]["native"]["extension"]}
        # Freeze the complete inherited lineage, including the retained patch checks.
        for key in ("connection_inputs", "patch_inputs"):
            inputs.update({f"inherited_{key}_{k}": v for k, v in parent.get(key, {}).items()})
        for key in ("connection_proposal", "connection_checks", "connection_audits", "patch_preview", "patch_checks", "patch_audits"):
            if key in parent:
                inputs[f"inherited_{key}"] = parent[key]
        if path.exists():
            proposal = json.loads(path.read_text())
            if proposal["inputs"] != inputs:
                raise ValueError("connection proposal inputs changed")
        else:
            module = jobs.core()
            assert module is not None
            payload = json.loads(module.connect_vector_map_junctions(job["pointcloud"]["files"]["map"]["path"],
                                parent["files"]["editable_map"]["path"], json.dumps(OPTIONS), None, True))
            offered = {(c["from"], c["to"]): c for c in payload["report"]["junctions"]["candidates"]}
            poses = np.loadtxt(job["pointcloud"]["files"]["trajectory"]["path"]).reshape(-1, 3, 4)[:, :, 3]
            stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1))]
            if not np.isfinite(poses).all() or stations[-1] < pieces[-1]["to_m"] - 1e-6:
                raise ValueError("trajectory does not cover the source stations")
            candidates, rejected = [], []
            from ca import mapping_local_points as local
            box = local.bounds(job)
            for first, second in zip(pieces, pieces[1:]):
                pair = (first["lane"], second["lane"])
                gap = second["from_m"] - first["to_m"]
                if not 0 < gap <= OPTIONS["max_gap"] or pair not in offered:
                    continue
                c = offered[pair]
                ts = np.linspace(first["to_m"], second["from_m"], math.ceil(gap / .5) + 1)
                xy = np.column_stack([np.interp(ts, stations, poses[:, k]) for k in (0, 1)])
                minimum = _width(c)
                holds = []
                if box is not None and any(not local.inside_geometry(c[k], box) for k in ('left', 'right', 'center')):
                    holds.append('outside_local_point_update_bounds')
                if any(frm == pair[0] or to == pair[1] for frm, to in edges(ir)):
                    holds.append("existing_endpoint_already_connected")
                if c["ambiguous"]:
                    holds.append("ambiguous_native_branch")
                if not _inside(xy, np.asarray(c["left"] + c["right"][::-1])[:, :2]):
                    holds.append("recorded_path_outside_connection")
                if minimum + 1e-7 < max(first["minimum_width_m"], second["minimum_width_m"]):
                    holds.append("below_fixed_minimum_width")
                if max(len(c[k]) for k in ("center", "left", "right")) > 128:
                    holds.append("geometry_exceeds_inspection_bound")
                if holds:
                    rejected.append({"from": pair[0], "to": pair[1], "holds": holds})
                else:
                    candidates.append({**c, "from_m": first["to_m"], "to_m": second["from_m"], "station_gap_m": gap,
                                       "trajectory_xy": xy.tolist(), "trajectory_step_max_m": .5,
                                       "minimum_interpolated_xy_width_m": minimum,
                                       "width_measurement": "equal_normalized_xy_arc_positions_on_linear_boundaries",
                                       "both_estimators_verified": False})
            proposal = {"candidate_id": cid, "inputs": inputs, "options": OPTIONS, "pieces": pieces,
                        "baseline_routes": route_metrics(ir, {p["lane"]: (p["from_m"], p["to_m"]) for p in pieces}),
                        "candidates": candidates, "rejected": rejected,
                        "native_summary": {k: v for k, v in payload["report"]["junctions"].items() if k not in {"candidates", "added"}},
                        "traffic_rules_inferred": False}
            jobs._inputs(job)
            for artifact in inputs.values():
                jobs._verify(artifact)
            jobs._save(path, proposal)
        for artifact in proposal["inputs"].values():
            jobs._verify(artifact)
        return {"proposal_file": jobs._artifact(path), "candidate_id": cid, "options": proposal["options"],
                "baseline_routes": proposal["baseline_routes"], "candidates_total": len(proposal["candidates"]),
                "candidates": proposal["candidates"][offset:offset + 8],
                "next_offset": offset + 8 if offset + 8 < len(proposal["candidates"]) else None,
                "rejected": proposal["rejected"][:8], "rejected_total": len(proposal["rejected"]),
                "native_summary": proposal["native_summary"], "deployment_ready": False}


def validate_pairs(proposal_file: dict[str, Any], pairs: Any) -> list[tuple[int, int]]:
    jobs._verify(proposal_file)
    proposal = json.loads(Path(proposal_file["path"]).read_text())
    for artifact in proposal["inputs"].values():
        jobs._verify(artifact)
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= 32:
        raise ValueError("choose 1..32 inspected connection pairs with individual reasons")
    chosen = []
    offered = {(c["from"], c["to"]) for c in proposal["candidates"]}
    for pair in pairs:
        if not isinstance(pair, dict) or set(pair) != {"from", "to", "reason"} or any(type(pair[k]) is not int for k in ("from", "to")) or not isinstance(pair["reason"], str) or not pair["reason"].strip():
            raise ValueError("each pair needs integer from/to and a reason")
        key = pair["from"], pair["to"]
        if key not in offered or key in chosen:
            raise ValueError("choose distinct source-checked offered connection pairs")
        chosen.append(key)
    return chosen


def connect(root: Path, cid: int, pairs: list[dict[str, Any]], proposal_file: dict[str, Any], reason: str) -> dict[str, Any]:
    with jobs._locked(root):
        job = jobs._load(root)
        jobs._inputs(job)
        parent, original, parent_report, _ = _parent(job, cid)
        chosen = validate_pairs(proposal_file, pairs)
        proposal = json.loads(Path(proposal_file["path"]).read_text())
        if proposal["candidate_id"] != cid:
            raise ValueError("connection proposal belongs to another parent")
        if len(original["lanes"]) + len(chosen) > 256:
            raise ValueError("short connections allow at most 256 total lanes")
        if jobs._remaining(job) <= 0:
            raise ValueError("mapping attempt budget exhausted")
        attempt: dict[str, Any] = {"id": len(job["attempts"]) + 1, "kind": "connected_corridor_lanes", "status": "running",
            "reason": reason, "road_options": parent["road_options"], "geometry_inputs": parent["geometry_inputs"],
            "corridor_proposal": parent["corridor_proposal"], "connection_inputs": proposal["inputs"],
            "connection_proposal": proposal_file, "connection_pairs": pairs, "parent_candidate_id": cid}
        for key in ("patch_inputs", "patch_preview", "patch_checks", "patch_audits"):
            if key in parent:
                attempt[key] = parent[key]
        job["attempts"].append(attempt)
        jobs._save(root / "job.json", job)
        target = root / f"candidate-{attempt['id']:02d}"
        try:
            if target.exists():
                raise FileExistsError(f"output directory already exists: {target}")
            module = jobs.core()
            assert module is not None
            cloud = job["pointcloud"]["files"]["map"]["path"]
            payload = json.loads(module.connect_vector_map_junctions(cloud, parent["files"]["editable_map"]["path"],
                                json.dumps(OPTIONS), json.dumps(chosen), False))
            ir = json.loads(payload["map_json"])
            for key in ("metadata", "lanes", "boundaries", "roads", "rules"):
                if isinstance(original.get(key), list):
                    if any(item not in ir.get(key, []) for item in original[key]):
                        raise ValueError("connection changed existing map geometry or semantics")
                elif original.get(key) != ir.get(key):
                    raise ValueError("connection changed existing map metadata")
            intervals = {p["lane"]: (p["from_m"], p["to_m"]) for p in proposal["pieces"]}
            candidates = {(c["from"], c["to"]): c for c in proposal["candidates"]}
            added_ids = payload["report"]["junctions"]["added"]
            graph = edges(ir)
            added = []
            for lid in added_ids:
                frm = [a for a, b in graph if b == lid]
                to = [b for a, b in graph if a == lid]
                if len(frm) != 1 or len(to) != 1:
                    raise ValueError("connector must have exactly one predecessor and successor")
                added.append({"from": frm[0], "to": to[0], "lane": lid})
            if {(a["from"], a["to"]) for a in added} != set(chosen) or len(added) != len(chosen):
                raise ValueError("native connection result differs from the explicit decision")
            connector_ids = set()
            expected_edges = edges(original).copy()
            widths = {p["lane"]: p["minimum_width_m"] for p in proposal["pieces"]}
            boundaries = {b["id"]: b for b in ir["boundaries"]}
            ir_lanes = {l["id"]: l for l in ir["lanes"]}
            for a in added:
                c = candidates[a["from"], a["to"]]
                lid = a["lane"]
                connector_ids.add(lid)
                intervals[lid] = c["from_m"], c["to_m"]
                widths[lid] = max(widths[a["from"]], widths[a["to"]])
                expected_edges.update(((a["from"], lid), (lid, a["to"])))
                lane = ir_lanes[lid]
                for side in ("left", "right"):
                    ref = lane[side]
                    bid = ref if type(ref) is int else ref["boundary"]
                    points = boundaries[bid]["geometry"]
                    if isinstance(ref, dict) and ref.get("reversed", False):
                        points = points[::-1]
                    if np.asarray(points).shape != np.asarray(c[side]).shape or not np.allclose(points, c[side], atol=1e-9, rtol=0):
                        raise ValueError("connector boundary differs from the inspected geometry")
                if (lane.get("kind", "driving") != "driving" or not lane.get("one_way", True)
                    or lane.get("speed_limit") != ir_lanes[a["from"]].get("speed_limit")):
                    raise ValueError("connector changed the fixed driving layout")
            if edges(ir) != expected_edges:
                raise ValueError("connection topology differs from the explicit decision")
            from ca import mapping_patch as patches
            patches._preserved(original, ir, expected_edges - edges(original))
            if set(ir_lanes) != {l["id"] for l in original["lanes"]} | connector_ids:
                raise ValueError("connection added unexpected lanes")
            routes = {"before": proposal["baseline_routes"], "after": route_metrics(ir, intervals),
                      "added": added, "connection_pairs": pairs, "source_extent_unchanged": True,
                      "retained_edges": sorted(edges(original)), "legal_routing_verified": False}
            evidence = payload["report"]
            evidence.update({"options": parent["road_options"], "reason": reason, "routes": routes,
                "connection_proposal": proposal_file, "connection_inputs": proposal["inputs"],
                "station_disposition": parent_report["station_disposition"], "extraction": parent["extraction"],
                "extent": parent["extent"], "built_segments": parent_report["built_segments"],
                "lane_intervals": {str(k): v for k, v in intervals.items()},
                "lane_minimum_width_m": {str(k): v for k, v in widths.items()},
                "retained_geometry_and_connections": True})
            with tempfile.TemporaryDirectory(prefix=".mapping-connections-", dir=root) as temporary:
                draft = Path(temporary) / "result"
                files = _publish(payload, draft)["files"]
                reopened = json.loads(module.edit_vector_map_relations(files["map"]))
                reopened_ir = json.loads(reopened["map_json"])
                verify_lane_roundtrip(ir, reopened_ir)
                if {l["id"]: l.get("turn_direction") for l in ir["lanes"]} != {
                    l["id"]: l.get("turn_direction") for l in reopened_ir["lanes"]}:
                    raise ValueError("OSM reload changed geometric turn labels")
                if edges(reopened_ir) != expected_edges or route_metrics(reopened_ir, intervals) != routes["after"]:
                    raise ValueError("OSM reload changed route reachability")
                audits = {key: json.loads(module.audit_vector_map_quality_details(cloud, files[fkey]))
                          for key, fkey in (("editable", "editable_map"), ("reopened_osm", "map"))}
                audits["ground_consensus"] = {key: json.loads(module.audit_vector_map_ground_consensus_details(cloud, files[fkey]))
                                             for key, fkey in (("editable", "editable_map"), ("reopened_osm", "map"))}
                # Keep failed audit evidence too; it must not silently become a published connection.
                failed_audit = root / f"candidate-{attempt['id']:02d}-connection-audits.json"
                jobs._save(failed_audit, audits)
                attempt["connection_audits"] = jobs._artifact(failed_audit)
                before_audits = json.loads(Path(parent["quality_report"]["path"]).read_text())
                checks = patches._checks(before_audits, audits, {l["id"] for l in original["lanes"]}, connector_ids)
                check_path = root / f"candidate-{attempt['id']:02d}-connection-checks.json"
                jobs._save(check_path, checks)
                attempt["connection_checks"] = jobs._artifact(check_path)
                if not checks["passes"]:
                    raise ValueError("connection source checks failed: " + ", ".join(checks["holds"]))
                for audit in [audits["editable"], audits["reopened_osm"], *audits["ground_consensus"].values()]:
                    diagnosis = jobs._diagnose_audit(audit)
                    lanes = [l for l in audit["quality"]["lanes"] if l["lane"] in connector_ids]
                    if not diagnosis["complete"] or {l["lane"] for l in lanes} != connector_ids or any(
                        l[k]["fraction"] < 1. or not l[k]["start_supported"] or not l[k]["end_supported"]
                        for l in lanes for k in ("center", "left", "right")):
                        raise ValueError("every connector trace needs complete support from both estimators in IR and reopened OSM")
                if any(i["severity"] == "error" for i in evidence["autoware_issues"]):
                    raise ValueError("connection has export errors")
                jobs._save(draft / "source-quality.json", audits)
                evidence["lane_roundtrip_verified"] = True
                evidence["topology_roundtrip_verified"] = True
                evidence["both_connector_estimators_verified"] = True
                evidence["files"] = {k: str(target / Path(v).name) for k, v in files.items()}
                evidence["editing"] = {"command": "vectormap", "args": ["mcp", evidence["files"]["map"]]}
                jobs._save(draft / "report.json", evidence)
                result = {"files": {k: {**jobs._artifact(files[k]), "path": v} for k, v in evidence["files"].items()},
                    "quality_report": {**jobs._artifact(draft / "source-quality.json"), "path": str(target / "source-quality.json")},
                    "quality": jobs._quality_summary(audits["editable"]), "reopened_quality": jobs._quality_summary(audits["reopened_osm"]),
                    "ground_consensus_quality": {k: jobs._quality_summary(v) for k, v in audits["ground_consensus"].items()},
                    "extraction": parent["extraction"], "extent": parent["extent"], "export_issues": evidence["autoware_issues"],
                    "routes": routes, "status": "audited_draft"}
                jobs._inputs(job)
                jobs._verify(proposal_file)
                for artifact in proposal["inputs"].values():
                    jobs._verify(artifact)
                os.rename(draft, target)
            attempt.update(result)
            job["status"] = "candidates_ready" if job["selected"] is None else "selected_draft"
        except BaseException as error:
            attempt.update({"status": "failed", "error": str(error), "error_type": type(error).__name__})
            if not isinstance(error, Exception) and not jobs._native_panic(error):
                jobs._save(root / "job.json", job)
                raise
        jobs._save(root / "job.json", job)
    return jobs.inspect_mapping_job(str(root))
