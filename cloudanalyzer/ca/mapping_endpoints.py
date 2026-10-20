"""Explicit endpoint revisions of audited, station-adjacent lane drafts."""
from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ca import mapping_connections as links, mapping_job as jobs, mapping_patch as patches
from ca import mapping_retry as retries
from ca.mapping_geometry import verify_lane_roundtrip
from ca.vector_map import _publish

PROTOCOL: dict[str, Any] = {
    "schema": "cloudanalyzer.endpoint_alignment.v1",
    "maximum_displacement_xyz_m": .75,
    "displacement_basis": "at most one unchanged source-audit ground radius",
    "recorded_path_tolerance_m": links.PATH_TOLERANCE_M,
    "maximum_vertices_per_boundary": 256,
    "one_explicit_pair_per_attempt": True,
    "changed_geometry_not_claimed_preserved": True,
}
EVIDENCE = ("endpoint_preview", "endpoint_checks", "endpoint_audits")


def _side(ir: dict[str, Any], lid: int, side: str) -> tuple[dict[str, Any], np.ndarray, bool]:
    lane = next(l for l in ir["lanes"] if l["id"] == lid)
    ref = lane[side]
    bid = ref if type(ref) is int else ref["boundary"]
    boundary = next(b for b in ir["boundaries"] if b["id"] == bid)
    reverse = isinstance(ref, dict) and ref.get("reversed", False)
    points = np.asarray(boundary["geometry"], dtype=float)
    return boundary, points[::-1] if reverse else points, reverse


def _context(root: Path, cid: int) -> tuple[Any, Any, Any, Any, Any]:
    job = jobs._load(root)
    jobs._inputs(job)
    if "retry_parent" in job:
        raise ValueError("endpoint alignment requires a root run; finish the child repair first")
    parent, ir, report, pieces = links._parent(job, cid)
    return job, parent, ir, report, pieces


def _inputs(root: Path, job: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    from ca.mapping_run import _load
    inputs = {"layout": _load(root)["layout_file"], "source": job["source"],
        "native": job["runtime"]["native"]["extension"], "proposal": parent["corridor_proposal"],
        "parent_audits": parent["quality_report"],
        **{f"parent_{k}": v for k, v in parent["files"].items()},
        **{f"geometry_{k}": v for k, v in parent["geometry_inputs"].items()},
        **{f"pointcloud_{k}": v for k, v in job["pointcloud"]["files"].items()},
        **{f"continuation_{k}": v for k, v in job.get("continuation_inputs", {}).items()}}
    for key in ("connection_inputs", "patch_inputs", "height_inputs", "endpoint_inputs"):
        inputs.update({f"inherited_{key}_{k}": v for k, v in parent.get(key, {}).items()})
    for key in (*EVIDENCE, "connection_proposal", "connection_checks", "connection_audits",
                "patch_preview", "patch_checks", "patch_audits", "height_checks", "height_audits", "height_trial"):
        if key in parent:
            inputs[f"inherited_{key}"] = parent[key]
    retries._verify_inputs(inputs)
    return inputs


def _case(ir: dict[str, Any], pieces: list[dict[str, Any]], frm: int, to: int, target: int) -> dict[str, Any]:
    if any(type(v) is not int for v in (frm, to, target)) or target not in (frm, to):
        raise ValueError("choose integer from/to and revise_lane equal to one of them")
    pair = next(((a, b) for a, b in zip(pieces, pieces[1:]) if (a["lane"], b["lane"]) == (frm, to)), None)
    if pair is None or abs(pair[0]["to_m"] - pair[1]["from_m"]) > 1e-6:
        raise ValueError("endpoint alignment needs consecutive pieces with a zero original-station gap")
    if any(a == frm or b == to for a, b in links.edges(ir)):
        raise ValueError("the selected exit or entry already has a connection")
    lane = next(l for l in ir["lanes"] if l["id"] == target)
    if lane.get("attributes", {}).get("cloudanalyzer:geometry_revision"):
        raise ValueError("a revised lane cannot accumulate further endpoint displacements")
    users: dict[int, set[int]] = {}
    for row in ir["lanes"]:
        if row.get("kind", "driving") != "driving" or not row.get("one_way", True):
            raise ValueError("endpoint alignment requires forward one-way driving lanes")
        for side in ("left", "right"):
            bid = _side(ir, row["id"], side)[0]["id"]
            users.setdefault(bid, set()).add(row["id"])
    sides = [_side(ir, target, s) for s in ("left", "right")]
    if (sides[0][0]["id"] == sides[1][0]["id"]
            or any(users[b["id"]] != {target} for b, _, _ in sides)):
        raise ValueError("the revised lane requires two private boundaries")
    curves = [np.asarray(lane.get("centerline", []), dtype=float), *(p for _, p, _ in sides)]
    if (len({len(p) for p in curves}) != 1 or any(p.ndim != 2 or p.shape[1] != 3
            or not 2 <= len(p) <= PROTOCOL["maximum_vertices_per_boundary"] or not np.isfinite(p).all() for p in curves)):
        raise ValueError("the revised lane requires bounded, explicit paired center/boundary vertices")
    if not np.allclose(curves[0], (curves[1] + curves[2]) * .5, atol=1e-9, rtol=0):
        raise ValueError("the explicit centerline must contain paired source midpoints")
    return {"from": frm, "to": to, "target_lane": target,
            "anchor_lane": to if target == frm else frm,
            "target_endpoint": "end" if target == frm else "start"}


def _revise(original: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    ir = copy.deepcopy(original)
    index = -1 if case["target_endpoint"] == "end" else 0
    anchor_index = 0 if index == -1 else -1
    for side in ("left", "right"):
        anchor, points, _ = _side(original, case["anchor_lane"], side)
        boundary, _, reverse = _side(ir, case["target_lane"], side)
        storage_index = (-1 if index == 0 else 0) if reverse else index
        boundary["geometry"][storage_index] = points[anchor_index].tolist()
        boundary.setdefault("attributes", {}).update({
            "cloudanalyzer:geometry_revision": "retained_endpoint_alignment_hypothesis",
            "cloudanalyzer:anchor_boundary": str(anchor["id"]),
            "cloudanalyzer:anchor_endpoint": "start" if anchor_index == 0 else "end"})
    lane = next(l for l in ir["lanes"] if l["id"] == case["target_lane"])
    lane["centerline"][index] = ((_side(ir, lane["id"], "left")[1][index]
                                   + _side(ir, lane["id"], "right")[1][index]) * .5).tolist()
    lane.setdefault("attributes", {}).update({
        "cloudanalyzer:geometry_revision": "retained_endpoint_alignment_hypothesis",
        "cloudanalyzer:endpoint_anchor_lane": str(case["anchor_lane"]),
        "cloudanalyzer:revised_endpoint": case["target_endpoint"]})
    graph = links.edges(original) | {(case["from"], case["to"])}
    old_links = {t["lane"]: t for t in ir.get("topology", [])}
    ir["topology"] = [{**old_links.get(lid, {}), "lane": lid,
                       "predecessors": sorted(a for a, b in graph if b == lid),
                       "successors": sorted(b for a, b in graph if a == lid)}
                      for lid in sorted(l["id"] for l in ir["lanes"])]
    return ir


def _geometry_check(original: dict[str, Any], expected: dict[str, Any], actual: dict[str, Any], case: dict[str, Any]) -> None:
    """Only the declared endpoint/midpoint/attributes and one directed edge may change."""
    if set(actual) != set(expected):
        raise ValueError("endpoint export changed map fields")
    for key in expected:
        if key in ("lanes", "boundaries"):
            if ({r["id"]: r for r in actual[key]} != {r["id"]: r for r in expected[key]}
                    or len(actual[key]) != len(expected[key])):
                raise ValueError("endpoint export changed geometry outside the explicit revision")
        elif key != "topology" and actual[key] != expected[key]:
            raise ValueError("endpoint export changed map metadata or semantics")
    if links.edges(actual) != links.edges(expected):
        raise ValueError("endpoint export changed the explicit graph")
    rows = {r["lane"]: r for r in actual.get("topology", [])}
    for row in original.get("topology", []):
        if any(rows.get(row["lane"], {}).get(k) != v for k, v in row.items() if k not in {"lane", "predecessors", "successors"}):
            raise ValueError("endpoint export changed lateral or other relationships")


def _anchor_check(audits: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    endpoint = "start_supported" if case["target_endpoint"] == "end" else "end_supported"
    flags = {name: {trace: next(l for l in a["quality"]["lanes"] if l["lane"] == case["anchor_lane"])[trace][endpoint]
                    for trace in ("center", "left", "right")} for name, a in patches._audits(audits).items()}
    return {"endpoint": endpoint, "flags": flags, "passes": all(all(f.values()) for f in flags.values())}


def _direction(ir: dict[str, Any], case: dict[str, Any], required: float) -> dict[str, Any]:
    lane = next(l for l in ir["lanes"] if l["id"] == case["target_lane"])
    curves = [lane["centerline"], *(_side(ir, lane["id"], s)[1].tolist() for s in ("left", "right"))]
    geometry = {"format": "vectormap-ir", "version": 1,
        "metadata": {"attributes": {"cloudanalyzer:draft": "surface_geometry"}}, "boundaries": [
        {"id": i, "kind": {"type": "other"}, "geometry": curve, "attributes": {"cloudanalyzer:role": role}}
        for i, (curve, role) in enumerate(zip(curves, ("source_center", "source_left", "source_right")), 1)],
        "lanes": [], "roads": [], "topology": []}
    spec = [{"center_curve_id": 1, "left_curve_id": 2, "right_curve_id": 3, "speed_limit_kmh": lane["speed_limit"]["kmh"], "lanes": [
        {"direction": "forward", "kind": "driving", "one_way": True, "fraction": 1., "minimum_width_m": required}]}]
    method = getattr(jobs.core(), "inspect_corridor_lane_directions", None)
    if method is None:
        return {"passes": False, "hold": "native_direction_preflight_unavailable"}
    try:
        result = json.loads(method(json.dumps(geometry), json.dumps(spec)))
        passes = (not result["limited"] and result["direction_checks_passed"] and len(result["segments"]) == 1
                  and result["segments"][0]["planned_pieces"] == [[0, len(curves[0]) - 1]])
        return {"passes": bool(passes), "native_preview": result}
    except ValueError as error:
        return {"passes": False, "error": str(error)}


def preview(root: Path, cid: int, frm: int, to: int, target: int) -> dict[str, Any]:
    with jobs._locked(root):
        job, parent, ir, _, pieces = _context(root, cid)
        case = _case(ir, pieces, frm, to, target)
        inputs = _inputs(root, job, parent)
        path = root / f"endpoints-{cid:02d}-{frm}-{to}-{target}.json"
        if path.exists():
            report = patches._read(jobs._artifact(path))
            if report["inputs"] != inputs or report["protocol"] != PROTOCOL or report["case"] != case:
                raise ValueError("endpoint preview inputs changed")
        else:
            audits = patches._read(parent["quality_report"])
            names = patches._audits(audits)
            ids = {l["id"] for l in ir["lanes"]}
            if len(names) != 4 or any(not jobs._diagnose_audit(a)["complete"] or a["quality"].get("problems_limited", True)
                    or {l["lane"] for l in a["quality"]["lanes"]} != ids for a in names.values()):
                raise ValueError("endpoint inspection needs four complete audits and failure locations")
            holds = []
            anchor = _anchor_check(audits, case)
            if not anchor["passes"]:
                holds.append("anchor_endpoint_not_supported_in_all_four_audits")
            radius = min(PROTOCOL["maximum_displacement_xyz_m"], *(a["quality"]["ground_radius_m"] for a in names.values()))
            distances = {s: float(np.linalg.norm(_side(ir, frm, s)[1][-1] - _side(ir, to, s)[1][0])) for s in ("left", "right")}
            if max(distances.values()) > radius:
                holds.append("endpoint_displacement_exceeds_fixed_limit")
            proposed = _revise(ir, case)
            required = next(p["minimum_width_m"] for p in pieces if p["lane"] == target)
            width = links._width_check({s: _side(proposed, target, s)[1].tolist() for s in ("left", "right")}, required)
            if not width["passes"]:
                holds.append("below_fixed_minimum_width")
            poses = np.loadtxt(job["pointcloud"]["files"]["trajectory"]["path"]).reshape(-1, 3, 4)[:, :, 3]
            stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1))]
            piece = next(p for p in pieces if p["lane"] == target)
            if not np.isfinite(poses).all() or stations[-1] < pieces[-1]["to_m"] - 1e-6:
                raise ValueError("trajectory does not cover the original source stations")
            ts = np.linspace(piece["from_m"], piece["to_m"], int(np.ceil((piece["to_m"] - piece["from_m"]) / .5)) + 1)
            xy = np.column_stack([np.interp(ts, stations, poses[:, k]) for k in (0, 1)])
            polygon = np.concatenate([_side(proposed, target, "left")[1], _side(proposed, target, "right")[1][::-1]])[:, :2]
            outside = links._outside_by(xy, polygon)
            if outside > PROTOCOL["recorded_path_tolerance_m"]:
                holds.append("recorded_path_outside_revised_lane")
            direction = _direction(proposed, case, required) if not holds else {"passes": False, "skipped_due_to_holds": True}
            if not holds and not direction["passes"]:
                holds.append("whole_revised_lane_requires_deferral_or_cuts")
            report = {"protocol": PROTOCOL, "candidate_id": cid, "case": case, "inputs": inputs,
                "holds": holds, "eligible": not holds, "anchor_check": anchor,
                "boundary_displacements_xyz_m": distances, "effective_displacement_limit_m": radius,
                "width_check": width, "recorded_path_outside_m": outside, "direction_check": direction,
                "endpoint_changes": {s: {"before": _side(ir, target, s)[1][-1 if target == frm else 0].tolist(),
                    "after": _side(proposed, target, s)[1][-1 if target == frm else 0].tolist()} for s in ("left", "right")},
                "four_fresh_audits_required": True, "source_extent_unchanged": True, "deployment_ready": False}
            jobs._inputs(job)
            retries._verify_inputs(inputs)
            jobs._save(path, report)
        return {"file": jobs._artifact(path), **{k: v for k, v in report.items() if k != "inputs"}}


def validate(root: Path, cid: int, preview_file: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any, Any]:
    job, parent, original, parent_report, pieces = _context(root, cid)
    report = patches._read(preview_file)
    c = report["case"]
    if (report["candidate_id"] != cid or report["protocol"] != PROTOCOL or not report["eligible"] or report["holds"]
            or report["case"] != _case(original, pieces, c["from"], c["to"], c["target_lane"])
            or report["inputs"] != _inputs(root, job, parent)):
        raise ValueError("choose an eligible inspected endpoint revision from this parent")
    return job, parent, original, parent_report, pieces, report


def _quality_checks(before: dict[str, Any], after: dict[str, Any], ids: set[int], target: int) -> dict[str, Any]:
    holds = []
    old, new = patches._audits(before), patches._audits(after)
    for name in ("editable", "reopened_osm", "consensus_editable", "consensus_reopened_osm"):
        a, b = old[name], new[name]
        da, db = jobs._diagnose_audit(a), jobs._diagnose_audit(b)
        if (not da["complete"] or not db["complete"] or da["protocol"] != db["protocol"]
                or a["quality"].get("problems_limited", True) or b["quality"].get("problems_limited", True)):
            holds.append(f"{name}:incomplete_or_changed_protocol")
        rows = [{l["lane"]: l for l in audit["quality"]["lanes"]} for audit in (a, b)]
        if any(r.keys() != ids for r in rows):
            holds.append(f"{name}:lane_membership_changed")
            continue
        if any(rows[0][lid] != rows[1][lid] for lid in ids - {target}):
            holds.append(f"{name}:unchanged_lane_audit_changed")
        if ([p for p in a["quality"].get("problems", []) if p["lane"] != target]
                != [p for p in b["quality"].get("problems", []) if p["lane"] != target]):
            holds.append(f"{name}:unchanged_failure_locations_changed")
    statuses = {}
    for low, consensus in (("editable", "consensus_editable"), ("reopened_osm", "consensus_reopened_osm")):
        if new[low]["quality"]["lanes"] != new["reopened_osm" if low == "editable" else "editable"]["quality"]["lanes"]:
            holds.append("IR_and_OSM_low_quantile_rows_differ")
        revised_rows: list[Any] = [next((l for l in new[n]["quality"]["lanes"] if l["lane"] == target), None) for n in (low, consensus)]
        if any(r is None for r in revised_rows):
            holds.append(f"{low}:missing_revised_lane")
            continue
        statuses[low] = {s: jobs.trace_support(revised_rows[0][s], revised_rows[1][s]) for s in ("center", "left", "right")}
        if "held" in statuses[low].values():
            holds.append(f"{low}:revised_trace_not_supported_or_explained")
    if new["consensus_editable"]["quality"]["lanes"] != new["consensus_reopened_osm"]["quality"]["lanes"]:
        holds.append("IR_and_OSM_consensus_rows_differ")
    return {"protocol": "fresh_revised_lane_exact_unchanged_lane_audits_v1", "passes": not holds,
            "holds": sorted(set(holds)), "revised_lane_id": target, "unchanged_lane_ids": sorted(ids - {target}),
            "revised_trace_statuses": statuses, "changed_geometry_not_claimed_preserved": True}


def align(root: Path, cid: int, preview_file: dict[str, Any], reason: str) -> dict[str, Any]:
    with jobs._locked(root):
        job, parent, original, parent_report, pieces, previewed = validate(root, cid, preview_file)
        if jobs._remaining(job) < 1 or not reason.strip():
            raise ValueError("endpoint revision needs a reason and one remaining shared HD attempt")
        case = previewed["case"]
        attempt: dict[str, Any] = {"id": len(job["attempts"]) + 1, "kind": "aligned_corridor_lanes", "status": "running",
            "reason": reason, "parent_candidate_id": cid, "road_options": parent["road_options"],
            "geometry_inputs": parent["geometry_inputs"], "corridor_proposal": parent["corridor_proposal"],
            "endpoint_inputs": previewed["inputs"], "endpoint_preview": preview_file}
        for key in ("connection_inputs", "connection_proposal", "connection_checks", "connection_audits",
                    "patch_inputs", "patch_preview", "patch_checks", "patch_audits", "height_inputs",
                    "height_edits", "height_checks", "height_audits", "height_trial"):
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
            proposed = _revise(original, case)
            intervals = {p["lane"]: (p["from_m"], p["to_m"]) for p in pieces}
            routes = {"before": links.route_metrics(original, intervals), "after": links.route_metrics(proposed, intervals),
                      "source_extent_unchanged": True, "revised_lane_id": case["target_lane"], "added_edge": [case["from"], case["to"]]}
            with tempfile.TemporaryDirectory(prefix=".mapping-endpoints-", dir=root) as temporary:
                hypothesis = Path(temporary) / "hypothesis.json"
                jobs._save(hypothesis, proposed)
                payload = json.loads(module.edit_vector_map_relations(str(hypothesis)))
                ir = json.loads(payload["map_json"])
                _geometry_check(original, proposed, ir, case)
                draft = Path(temporary) / "result"
                files = _publish(payload, draft)["files"]
                if Path(files["projector"]).read_bytes() != Path(parent["files"]["projector"]["path"]).read_bytes():
                    raise ValueError("endpoint revision changed the projector")
                reopened = json.loads(json.loads(module.edit_vector_map_relations(files["map"]))["map_json"])
                verify_lane_roundtrip(ir, reopened)
                old_osm = json.loads(json.loads(module.edit_vector_map_relations(parent["files"]["map"]["path"]))["map_json"])
                changed_boundaries = {_side(original, case["target_lane"], s)[0]["id"] for s in ("left", "right")}
                for key, excluded in (("lanes", {case["target_lane"]}), ("boundaries", changed_boundaries)):
                    before = {r["id"]: r for r in old_osm[key]}
                    after = {r["id"]: r for r in reopened[key]}
                    if before.keys() != after.keys() or any(before[i] != after[i] for i in before.keys() - excluded):
                        raise ValueError("OSM reload changed an untouched lane or boundary")
                if links.edges(reopened) != links.edges(proposed) or links.route_metrics(reopened, intervals) != routes["after"]:
                    raise ValueError("OSM reload changed endpoint route reachability")
                if any(i["severity"] == "error" for i in payload["report"]["autoware_issues"]):
                    raise ValueError("endpoint revision has export errors")
                cloud = job["pointcloud"]["files"]["map"]["path"]
                audits = {k: json.loads(module.audit_vector_map_quality_details(cloud, files[f]))
                          for k, f in (("editable", "editable_map"), ("reopened_osm", "map"))}
                audits["ground_consensus"] = {k: json.loads(module.audit_vector_map_ground_consensus_details(cloud, files[f]))
                                             for k, f in (("editable", "editable_map"), ("reopened_osm", "map"))}
                audit_path = root / f"candidate-{attempt['id']:02d}-endpoint-audits.json"
                jobs._save(audit_path, audits)
                attempt["endpoint_audits"] = jobs._artifact(audit_path)
                checks = _quality_checks(patches._read(parent["quality_report"]), audits, set(intervals), case["target_lane"])
                check_path = root / f"candidate-{attempt['id']:02d}-endpoint-checks.json"
                jobs._save(check_path, checks)
                attempt["endpoint_checks"] = jobs._artifact(check_path)
                if not checks["passes"]:
                    raise ValueError("endpoint revision failed fresh source or exact retained audit checks")
                evidence = payload["report"]
                for key in ("source_proposals", "append_inputs", "append_checks", "baseline_candidate_id", "addition_candidate_id",
                            "selected_gap_ids", "explicit_endpoint_pairs", "added_lane_id_map"):
                    if key in parent_report:
                        evidence[key] = parent_report[key]
                evidence.update(options=parent["road_options"], reason=reason, routes=routes,
                    station_disposition=parent_report["station_disposition"], extraction=parent["extraction"], extent=parent["extent"],
                    built_segments=parent_report.get("built_segments", []), lane_intervals={str(k): v for k, v in intervals.items()},
                    lane_minimum_width_m={str(p["lane"]): p["minimum_width_m"] for p in pieces},
                    endpoint_alignment={"preview": preview_file, "case": case, "checks": checks, "four_fresh_audits_complete": True,
                        "changed_geometry_not_claimed_preserved": True}, lane_roundtrip_verified=True, topology_roundtrip_verified=True)
                evidence["files"] = {k: str(target / Path(v).name) for k, v in files.items()}
                jobs._save(draft / "report.json", evidence)
                jobs._save(draft / "source-quality.json", audits)
                result = {"files": {k: {**jobs._artifact(files[k]), "path": v} for k, v in evidence["files"].items()},
                    "quality_report": {**jobs._artifact(draft / "source-quality.json"), "path": str(target / "source-quality.json")},
                    "quality": jobs._quality_summary(audits["editable"]), "reopened_quality": jobs._quality_summary(audits["reopened_osm"]),
                    "ground_consensus_quality": {k: jobs._quality_summary(v) for k, v in audits["ground_consensus"].items()},
                    "extraction": parent["extraction"], "extent": parent["extent"], "export_issues": evidence["autoware_issues"],
                    "routes": routes, "status": "audited_draft"}
                jobs._inputs(job)
                jobs._verify(preview_file)
                retries._verify_inputs(previewed["inputs"])
                os.rename(draft, target)
            attempt.update(result)
            job["status"] = "candidates_ready" if job["selected"] is None else "selected_draft"
        except BaseException as error:
            attempt.update(status="failed", error=str(error), error_type=type(error).__name__)
            if not isinstance(error, Exception) and not jobs._native_panic(error):
                jobs._save(root / "job.json", job)
                raise
        jobs._save(root / "job.json", job)
    return jobs.inspect_mapping_job(str(root))
