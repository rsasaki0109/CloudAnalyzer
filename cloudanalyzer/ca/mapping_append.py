"""Append an audited refined draft in seen gaps of an unchanged external map."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, cast

from ca import mapping_job as jobs, mapping_patch as patches, mapping_retry as retries
from ca.mapping_connections import edges, route_metrics
from ca.mapping_geometry import verify_lane_roundtrip
from ca.mapping_map_gaps import SCHEMA as GAP_SCHEMA
from ca.vector_map import _publish

MAX_OUTPUT_BYTES = 64 * 1024**2
PREVIEW_PROTOCOL = {"maximum_endpoint_pairs": 32, "maximum_bytes": 2 * 1024**2,
                    "endpoint_tolerance_m": 1e-8, "station_tolerance_m": 1e-6,
                    "native_processing_performed": False, "shared_hd_attempts_spent": 0}


def validate(root: Path, baseline_id: int, addition_id: int, gap_ids: list[int],
             evidence: dict[str, Any], layout: dict[str, Any]) -> dict[str, Any]:
    """Bind two own audited sources to the same fixed map, layout and seen gaps."""
    job = jobs._load(root)
    jobs._inputs(job)
    if job["pointcloud"].get("origin") != "external_slam":
        raise ValueError("draft appending requires a frozen imported external SLAM map")
    if "draft_append" in job:
        raise ValueError("one draft append is allowed per run; no recursive appends")
    if any(type(v) is not int or v < 1 for v in (baseline_id, addition_id)) or baseline_id == addition_id:
        raise ValueError("append needs distinct positive audited candidate IDs")
    if (not isinstance(gap_ids, list) or not 1 <= len(gap_ids) <= 8
            or any(type(v) is not int or v < 1 for v in gap_ids) or len(set(gap_ids)) != len(gap_ids)):
        raise ValueError("append needs 1..8 distinct seen gap IDs")
    baseline, addition = [retries._parent(job, cid) for cid in (baseline_id, addition_id)]
    if baseline["kind"] not in {"corridor_lanes", "connected_corridor_lanes", "aligned_corridor_lanes"} or addition["kind"] != "corridor_lanes":
        raise ValueError("append an unconnected refined source draft to the original audited baseline")
    refinement = job.get("corridor_refinement", {})
    if (refinement.get("status") != "ready" or baseline["corridor_proposal"] != refinement["previous"]["file"]
            or addition["corridor_proposal"] != refinement["file"]):
        raise ValueError("append requires the retained original and its own path-association refinement")
    fixed = patches._read(layout)
    lanes = fixed["lanes"]
    if len(lanes) != 1 or lanes[0]["direction"] != "forward" or not lanes[0]["one_way"] or lanes[0]["fraction"] != 1:
        raise ValueError("draft appending currently requires one forward one-way lane")
    for parent in (baseline, addition):
        options = parent["road_options"]
        if (options["boundary_policy"] != fixed["boundary_policy"] or not options["lane_specs"]
                or any(s["lanes"] != lanes or s["speed_limit_kmh"] != fixed["speed_limit_kmh"] for s in options["lane_specs"])):
            raise ValueError("append cannot change the frozen layout or minimum widths")
    proposals = [patches._read(p["corridor_proposal"]) for p in (baseline, addition)]
    protocols = [copy.deepcopy(p["protocol"]) for p in proposals]
    modes = [p["options"].pop("association", "all_supported_bands") for p in protocols]
    if (modes != ["all_supported_bands", "trajectory_containing"] or protocols[0] != protocols[1]
            or proposals[0]["input_artifacts"] != proposals[1]["input_artifacts"]
            or proposals[0]["coordinate_frame"] != proposals[1]["coordinate_frame"]
            or proposals[0]["trajectory_length_m"] != proposals[1]["trajectory_length_m"]
            or baseline["extent"]["trajectory_length_m"] != addition["extent"]["trajectory_length_m"]):
        raise ValueError("append changed source, stations or extraction thresholds beyond path association")
    saved = patches._read(evidence)
    if (saved.get("schema") != GAP_SCHEMA or saved.get("candidate_id") != baseline_id
            or saved["inputs"]["proposal"] != baseline["corridor_proposal"] or saved["inputs"]["layout"] != layout
            or saved["inputs"]["audits"] != baseline["quality_report"]
            or any(saved["inputs"].get(f"pointcloud_{k}") != v for k, v in job["pointcloud"]["files"].items())):
        raise ValueError("append needs the retained baseline's saved-map gap evidence")
    retries._verify_inputs(saved["inputs"])
    gaps = [g for g in saved["gaps"] if g["id"] in gap_ids]
    if {g["id"] for g in gaps} != set(gap_ids):
        raise ValueError("inspect every chosen append gap through this run")
    before, added = [patches._intervals(p) for p in (baseline, addition)]
    intervals = sorted(added.values())
    if (not intervals or any(not all(math.isfinite(v) for v in i) or i[0] >= i[1] for i in intervals)
            or any(a[1] > b[0] for a, b in zip(intervals, intervals[1:]))):
        raise ValueError("addition lane intervals must be finite, increasing and disjoint")
    if any(max(lo, a) < min(hi, b) for lo, hi in intervals for a, b in before.values()):
        raise ValueError("addition overlaps a retained lane or connector")
    if any(not any(g["from_m"] <= lo < hi <= g["to_m"] for g in gaps) for lo, hi in intervals):
        raise ValueError("every addition must stay within a chosen seen baseline gap")
    original, trial = [patches._read(p["files"]["editable_map"]) for p in (baseline, addition)]
    if (len(original["lanes"]) + len(trial["lanes"]) > 256
            or sum(len(b["geometry"]) for m in (original, trial) for b in m["boundaries"]) > 100000):
        raise ValueError("append exceeds the 256-lane or 100000-vertex budget")
    if trial.get("roads") or trial.get("rules") or trial.get("topology"):
        raise ValueError("addition must contain source lanes without routes, road groups or rules")
    if set(before) != {l["id"] for l in original["lanes"]} or set(added) != {l["id"] for l in trial["lanes"]}:
        raise ValueError("append intervals do not account for every retained and added lane")
    route_metrics(original, before)
    inputs = {"gap_evidence": evidence, "layout": layout, "native": job["runtime"]["native"]["extension"],
              **{f"pointcloud_{k}": v for k, v in job["pointcloud"]["files"].items()}}
    for label, parent in (("baseline", baseline), ("addition", addition)):
        inputs.update({f"{label}_{k}": v for k, v in parent["files"].items()})
        inputs.update({f"{label}_geometry_{k}": v for k, v in parent["geometry_inputs"].items()})
        inputs.update({f"{label}_proposal": parent["corridor_proposal"], f"{label}_audits": parent["quality_report"]})
        inputs.update({f"{label}_connection_{k}": v for k, v in parent.get("connection_inputs", {}).items()})
        inputs.update({f"{label}_endpoint_{k}": v for k, v in parent.get("endpoint_inputs", {}).items()})
        for key in ("endpoint_preview", "endpoint_checks", "endpoint_audits"):
            if key in parent:
                inputs[f"{label}_{key}"] = parent[key]
    retries._verify_inputs(inputs)
    if shutil.disk_usage(root).free < 10 * 1024**3 + 2 * MAX_OUTPUT_BYTES:
        raise ValueError("insufficient disk space: draft append must retain 10 GiB free")
    return dict(inputs=inputs, baseline=baseline, addition=addition, original=original, trial=trial,
                before_intervals=before, addition_intervals=added)


def _joins(prepared: dict[str, Any]) -> list[dict[str, Any]]:
    merged, ids = patches._merge(prepared["original"], prepared["trial"])
    bounds = {b["id"]: b["geometry"] for b in merged["boundaries"]}
    lanes = {l["id"]: l for l in merged["lanes"]}
    intervals = {**prepared["before_intervals"], **{ids[k]: v for k, v in prepared["addition_intervals"].items()}}
    added = set(ids.values())
    def point(lid: int, side: str, last: bool) -> list[float]:
        ref = lanes[lid][side]
        bid = ref if type(ref) is int else ref["boundary"]
        reverse = isinstance(ref, dict) and ref.get("reversed", False)
        return cast(list[float], bounds[bid][-1 if last != reverse else 0])
    result = []
    for frm, (lo, hi) in intervals.items():
        for to, (start, stop) in intervals.items():
            if frm == to or (frm not in added and to not in added) or abs(hi - start) > PREVIEW_PROTOCOL["station_tolerance_m"]:
                continue
            distances = [math.dist(point(frm, side, True), point(to, side, False)) for side in ("left", "right")]
            if max(distances) <= PREVIEW_PROTOCOL["endpoint_tolerance_m"]:
                result.append({"from": frm, "to": to, "station_m": hi, "boundary_endpoint_distances_m": distances})
                if len(result) > PREVIEW_PROTOCOL["maximum_endpoint_pairs"]:
                    raise ValueError("append exceeds 32 exact endpoint pairs; choose fewer additions")
    return sorted(result, key=lambda p: (p["station_m"], p["from"], p["to"]))


def preview(root: Path, baseline_id: int, addition_id: int, gap_ids: list[int], evidence: dict[str, Any],
            layout: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    with jobs._locked(root):
        prepared = validate(root, baseline_id, addition_id, gap_ids, evidence, layout)
        if prepared["inputs"] != expected:
            raise ValueError("append preview inputs changed after validation")
        value = dict(schema="cloudanalyzer.mapping_append_preview.v1", baseline_candidate_id=baseline_id,
            addition_candidate_id=addition_id, gap_ids=gap_ids, inputs=expected, protocol=PREVIEW_PROTOCOL,
            candidates=_joins(prepared), note="Exact endpoint observations only. Select every listed pair explicitly or retain the input drafts; source audits are checked during append.")
        key = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        path = root / f"append-preview-{baseline_id}-{addition_id}-{key[:16]}.json"
        if path.exists():
            if path.stat().st_size > PREVIEW_PROTOCOL["maximum_bytes"] or patches._read(jobs._artifact(path)) != value:
                raise ValueError("saved append preview changed")
        else:
            if len((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")) > PREVIEW_PROTOCOL["maximum_bytes"]:
                raise ValueError("append preview exceeds its 2 MiB budget")
            retries._verify_inputs(expected)
            jobs._save(path, value)
        retries._verify_inputs(expected)
        return {"file": jobs._artifact(path), **{k: v for k, v in value.items() if k != "inputs"}}


def chosen_pairs(prepared: dict[str, Any], preview_file: dict[str, Any], pairs: list[dict[str, Any]]) -> set[tuple[int, int]]:
    saved = patches._read(preview_file)
    offered = _joins(prepared)
    if saved.get("protocol") != PREVIEW_PROTOCOL or saved.get("inputs") != prepared["inputs"] or saved.get("candidates") != offered:
        raise ValueError("append endpoint preview belongs to different frozen inputs")
    if not isinstance(pairs, list) or len(pairs) > PREVIEW_PROTOCOL["maximum_endpoint_pairs"]:
        raise ValueError("append needs at most 32 explicit endpoint pairs")
    selected = set()
    for p in pairs:
        if (not isinstance(p, dict) or set(p) != {"from", "to", "reason"}
                or any(type(p[k]) is not int for k in ("from", "to"))
                or not isinstance(p["reason"], str) or not p["reason"].strip()):
            raise ValueError("append pairs need inspected from/to IDs and a reason")
        pair = (p["from"], p["to"])
        if pair in selected:
            raise ValueError("duplicate append endpoint pair")
        selected.add(pair)
    if selected != {(p["from"], p["to"]) for p in offered}:
        raise ValueError("select every exact inspected append endpoint pair or retain the input drafts")
    merged, ids = patches._merge(prepared["original"], prepared["trial"])
    expected_edges = edges(merged) | selected
    if expected_edges:
        merged["topology"] = [{"lane": lid, "predecessors": sorted(a for a, b in expected_edges if b == lid),
                               "successors": sorted(b for a, b in expected_edges if a == lid)} for lid in sorted({l["id"] for l in merged["lanes"]})]
    route_metrics(merged, {**prepared["before_intervals"], **{ids[k]: v for k, v in prepared["addition_intervals"].items()}})
    return selected


def append(root: Path, baseline_id: int, addition_id: int, gap_ids: list[int], evidence: dict[str, Any],
           layout: dict[str, Any], pairs: list[dict[str, Any]], preview_file: dict[str, Any],
           reason: str, expected: dict[str, Any]) -> dict[str, Any]:
    with jobs._locked(root):
        prepared = validate(root, baseline_id, addition_id, gap_ids, evidence, layout)
        if {**prepared["inputs"], "append_preview": preview_file} != expected:
            raise ValueError("draft append inputs changed after decision validation")
        selected = chosen_pairs(prepared, preview_file, pairs)
        job = jobs._load(root)
        if jobs._remaining(job) < 1:
            raise ValueError("draft append needs one remaining shared HD attempt")
        baseline, addition = prepared["baseline"], prepared["addition"]
        options = copy.deepcopy(baseline["road_options"])
        options["lane_specs"] = [{**s, "source_candidate_id": p["id"]} for p in (baseline, addition) for s in p["road_options"]["lane_specs"]]
        attempt: dict[str, Any] = dict(id=len(job["attempts"]) + 1, kind="appended_corridor_lanes", status="running",
            reason=reason, baseline_candidate_id=baseline_id, addition_candidate_id=addition_id, gap_ids=gap_ids,
            patch_inputs=expected, geometry_inputs=baseline["geometry_inputs"], corridor_proposal=baseline["corridor_proposal"],
            road_options=options, append_pairs=pairs)
        for key in ("endpoint_inputs", "endpoint_preview", "endpoint_checks", "endpoint_audits"):
            if key in baseline:
                attempt[key] = baseline[key]
        job["attempts"].append(attempt)
        job["draft_append"] = {"candidate_id": attempt["id"]}
        jobs._save(root / "job.json", job)
        target = root / f"candidate-{attempt['id']:02d}"
        try:
            if target.exists():
                raise FileExistsError(str(target))
            ir, ids = patches._merge(prepared["original"], prepared["trial"])
            expected_edges = edges(ir) | selected
            if expected_edges:
                old_rows = {r["lane"]: r for r in ir.get("topology", [])}
                ir["topology"] = [{**old_rows.get(lid, {}), "lane": lid,
                    "predecessors": sorted(a for a, b in expected_edges if b == lid),
                    "successors": sorted(b for a, b in expected_edges if a == lid)} for lid in sorted({l["id"] for l in ir["lanes"]})]
            patches._preserved(prepared["original"], ir, selected)
            intervals = {**prepared["before_intervals"], **{ids[k]: v for k, v in prepared["addition_intervals"].items()}}
            reports = [patches._read(p["files"]["report"]) for p in (baseline, addition)]
            cuts = sorted({v for report in reports for i in report["station_disposition"] for v in (i["from_m"], i["to_m"])})
            disposition = []
            for lo, hi in zip(cuts, cuts[1:]):
                rows = [next(i for i in report["station_disposition"] if i["from_m"] <= lo < hi <= i["to_m"]) for report in reports]
                source = 1 if rows[1]["status"] == "included_lane_hypothesis" else 0
                disposition.append({**rows[source], "from_m": lo, "to_m": hi,
                    "source_ambiguous": rows[0].get("source_ambiguous", False) or rows[source].get("source_ambiguous", False),
                    "baseline_source_reasons": rows[0].get("source_reasons", []),
                    "origin": "addition" if source else "baseline", "source_proposal_sha256": (addition if source else baseline)["corridor_proposal"]["sha256"]})
            extraction = {**baseline["extraction"], "generated_length": sum(i["to_m"] - i["from_m"] for i in disposition if i["status"] == "included_lane_hypothesis"),
                          "roads": baseline["extraction"]["roads"] + len(ids), "lanes": len(ir["lanes"])}
            routes = {"before": route_metrics(prepared["original"], prepared["before_intervals"]),
                      "after": route_metrics(ir, intervals), "added": [], "legal_routing_verified": False}
            with tempfile.TemporaryDirectory(prefix=".mapping-append-", dir=root) as temporary:
                input_path = Path(temporary) / "merged.json"
                jobs._save(input_path, ir)
                module = jobs.core()
                assert module is not None
                payload = json.loads(module.edit_vector_map_relations(str(input_path)))
                canonical = json.loads(payload["map_json"])
                patches._preserved(prepared["original"], canonical, selected)
                verify_lane_roundtrip(ir, canonical)
                if payload["projector_info"] != Path(baseline["files"]["projector"]["path"]).read_text(encoding="utf-8"):
                    raise ValueError("draft append changed baseline projection")
                if any(i["severity"] == "error" for i in payload["report"]["autoware_issues"]):
                    raise ValueError("draft append has export errors")
                draft = Path(temporary) / "result"
                files = _publish(payload, draft)["files"]
                reopened = json.loads(module.edit_vector_map_relations(files["map"]))
                reopened_ir = json.loads(reopened["map_json"])
                verify_lane_roundtrip(canonical, reopened_ir)
                baseline_osm = json.loads(json.loads(module.edit_vector_map_relations(baseline["files"]["map"]["path"]))["map_json"])
                patches._preserved(baseline_osm, reopened_ir, selected)
                if edges(reopened_ir) != edges(ir) or route_metrics(reopened_ir, intervals) != routes["after"]:
                    raise ValueError("draft append OSM reload changed retained routing")
                cloud = job["pointcloud"]["files"]["map"]["path"]
                audits = {k: json.loads(module.audit_vector_map_quality_details(cloud, files[f])) for k, f in (("editable", "editable_map"), ("reopened_osm", "map"))}
                audits["ground_consensus"] = {k: json.loads(module.audit_vector_map_ground_consensus_details(cloud, files[f])) for k, f in (("editable", "editable_map"), ("reopened_osm", "map"))}
                checks = patches._checks(patches._read(baseline["quality_report"]), audits, set(prepared["before_intervals"]), set(ids.values()))
                for name, audit in patches._audits(audits).items():
                    audited = {l["lane"]: l for l in audit["quality"]["lanes"]}
                    for frm, to in selected:
                        if any(not audited[lid][trace][endpoint] for lid, endpoint in ((frm, "end_supported"), (to, "start_supported")) for trace in ("center", "left", "right")):
                            checks["holds"].append(f"{name}:unsupported_append_join_endpoint:{frm}:{to}")
                checks["passes"] = not checks["holds"]
                if len(json.dumps(audits, allow_nan=False).encode()) > MAX_OUTPUT_BYTES // 2:
                    raise ValueError("draft append audit evidence exceeds its output budget")
                for name, value in (("audits", audits), ("checks", checks)):
                    path = root / f"candidate-{attempt['id']:02d}-append-{name}.json"
                    jobs._save(path, value)
                    attempt[f"patch_{name}"] = jobs._artifact(path)
                if not checks["passes"]:
                    raise ValueError("draft append source checks held the result: " + ", ".join(checks["holds"][:8]))
                jobs._save(draft / "source-quality.json", audits)
                report = payload["report"]
                report.update(options=options, reason=reason, append_inputs=expected, append_checks=checks,
                    baseline_candidate_id=baseline_id, addition_candidate_id=addition_id, selected_gap_ids=gap_ids,
                    source_proposals={"baseline": baseline["corridor_proposal"], "addition": addition["corridor_proposal"]},
                    explicit_endpoint_pairs=pairs,
                    added_lane_id_map={str(k): v for k, v in ids.items()}, lane_intervals={str(k): v for k, v in intervals.items()},
                    station_disposition=disposition, extraction=extraction, extent=jobs._extent(extraction, job["minimum_retained_fraction"]),
                    routes=routes, retained_geometry_and_connections=True, lane_roundtrip_verified=True,
                    note="Existing IDs, geometry, semantics and metadata remain exact. Directed routes retain original edges and add only explicitly inspected endpoint pairs. "
                         "The point cloud and trajectory are unchanged; original source and semantic holds remain visible.")
                report["files"] = {k: str(target / Path(v).name) for k, v in files.items()}
                report["editing"] = {"command": "vectormap", "args": ["mcp", report["files"]["map"]]}
                jobs._save(draft / "report.json", report)
                if sum(p.stat().st_size for p in draft.rglob("*") if p.is_file()) > MAX_OUTPUT_BYTES:
                    raise ValueError("draft append exceeds its output budget")
                result = dict(files={k: {**jobs._artifact(files[k]), "path": v} for k, v in report["files"].items()},
                    quality_report={**jobs._artifact(draft / "source-quality.json"), "path": str(target / "source-quality.json")},
                    quality=jobs._quality_summary(audits["editable"]), reopened_quality=jobs._quality_summary(audits["reopened_osm"]),
                    ground_consensus_quality={k: jobs._quality_summary(v) for k, v in audits["ground_consensus"].items()},
                    extraction=extraction, extent=report["extent"], export_issues=report["autoware_issues"], routes=routes, status="audited_draft")
                jobs._inputs(job)
                retries._verify_inputs(expected)
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
