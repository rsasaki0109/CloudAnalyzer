"""Explicit, bounded Z hypotheses for interior vertices of local HD additions."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ca import (
    mapping_job as jobs,
    mapping_retry as retries,
    mapping_local_points as local,
)
from ca.mapping_connections import edges
from ca.mapping_geometry import verify_lane_roundtrip
from ca.mapping_patch import _audits, _intervals, _read, validate as validate_patch
from ca.vector_map import _publish

PROTOCOL = {
    "maximum_delta_z_m": 0.1,
    "maximum_edits": 16,
    "maximum_vertices": 256,
    "boundary_xy_and_endpoints_fixed": True,
    "fully_supported_under_four_audits_required": True,
    "independent_accuracy_established": False,
}


def _parent(
    root: Path, cid: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    job = jobs._load(root)
    jobs._inputs(job)
    candidate = retries._parent(job, cid)
    if (
        "retry_parent" not in job
        or local.bounds(job) is None
        or "gap_patch" in job
        or candidate["kind"] != "corridor_lanes"
        or "height_inputs" in candidate
    ):
        raise ValueError("height edits require an unpatched local-retry addition draft")
    parent_root = Path(job["retry_parent"]["job_dir"])
    parent_job = jobs._load(parent_root)
    jobs._inputs(parent_job)
    stage = parent_job.get("pointcloud_retry", {})
    if (
        stage.get("status") != "ready"
        or Path(stage["child_job_dir"]).resolve() != root.resolve()
    ):
        raise ValueError("height draft does not belong to a ready point-map retry")
    # Reuse the combined patch's fixed layout, gap-only intervals, local geometry
    # and frozen reference checks before allowing any height hypothesis.
    point_preview = _read(job["retry_inputs"]["local_preview"])
    prepared = validate_patch(root, cid, point_preview["request"]["gap_ids"])
    return job, candidate, prepared["trial"]


def _inputs(job: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        **{f"parent_{k}": v for k, v in candidate["files"].items()},
        "parent_audits": candidate["quality_report"],
        **{f"geometry_{k}": v for k, v in candidate["geometry_inputs"].items()},
        "proposal": candidate["corridor_proposal"],
        "source": job["source"],
        "native": job["runtime"]["native"]["extension"],
        **{f"pointcloud_{k}": v for k, v in job["pointcloud"]["files"].items()},
        **{f"retry_{k}": v for k, v in job["retry_inputs"].items()},
    }


def preview(root: Path, cid: int, offset: int) -> dict[str, Any]:
    with jobs._locked(root):
        job, parent, ir = _parent(root, cid)
        inputs = _inputs(job, parent)
        retries._verify_inputs(inputs)
        path = root / f"heights-{cid:02d}.json"
        if path.exists():
            report = _read(jobs._artifact(path))
            if report["inputs"] != inputs or report["protocol"] != PROTOCOL:
                raise ValueError("height preview inputs or protocol changed")
        else:
            audits = _read(parent["quality_report"])
            names = _audits(audits)
            if any(
                not jobs._diagnose_audit(a)["complete"]
                or a["quality"].get("problems_limited", True)
                for a in names.values()
            ):
                raise ValueError(
                    "height inspection requires complete four audits and failure locations"
                )
            lanes = {l["id"]: l for l in ir["lanes"]}
            boundaries = {b["id"]: b for b in ir["boundaries"]}
            users: dict[int, set[int]] = {}
            for lane in lanes.values():
                for side in ("left", "right"):
                    ref = lane[side]
                    bid = ref if type(ref) is int else ref["boundary"]
                    users.setdefault(bid, set()).add(lane["id"])
            offered: dict[tuple[int, int], dict[str, Any]] = {}
            failures = []
            box = local.bounds(job)
            assert box is not None
            for name, audit in names.items():
                for p in audit["quality"]["problems"]:
                    failures.append({"estimator_format": name, **p})
                    if p["reason"] != "height_mismatch" or p["curve"] not in (
                        "left",
                        "right",
                    ):
                        continue
                    ref = lanes[p["lane"]][p["curve"]]
                    bid = ref if type(ref) is int else ref["boundary"]
                    if len(users[bid]) != 1:
                        continue
                    points = np.asarray(boundaries[bid]["geometry"], dtype=float)
                    xy = points[:, :2]
                    delta = np.diff(xy, axis=0)
                    norm = np.sum(delta * delta, axis=1)
                    for point, height in zip(p["points"], p["source_heights_m"]):
                        if height is None:
                            continue
                        t = np.clip(
                            np.sum((np.asarray(point[:2]) - xy[:-1]) * delta, axis=1)
                            / np.maximum(norm, 1e-30),
                            0,
                            1,
                        )
                        distance = np.linalg.norm(
                            xy[:-1] + t[:, None] * delta - np.asarray(point[:2]), axis=1
                        )
                        segment = int(np.argmin(distance))
                        if distance[segment] > 1e-6:
                            continue
                        for index in (segment, segment + 1):
                            if index in (
                                0,
                                len(points) - 1,
                            ) or not local.inside_geometry([points[index]], box):
                                continue
                            row = offered.setdefault(
                                (bid, index),
                                {
                                    "boundary_id": bid,
                                    "vertex_index": index,
                                    "xyz": points[index].tolist(),
                                    "lane_id": p["lane"],
                                    "observations": [],
                                },
                            )
                            row["observations"].append(
                                {
                                    "estimator_format": name,
                                    "failure_xyz": point,
                                    "observed_height_m": height,
                                    "height_difference_m": height - point[2],
                                }
                            )
            if len(offered) > PROTOCOL["maximum_vertices"]:
                raise ValueError(
                    "height preview exceeds 256 affected interior vertices"
                )
            report = {
                "schema": "cloudanalyzer.height_preview.v1",
                "candidate_id": cid,
                "inputs": inputs,
                "protocol": PROTOCOL,
                "effective_bounds_xy": box,
                "vertices": [offered[k] for k in sorted(offered)],
                "failures": failures,
                "note": "Observed heights remain estimator hypotheses. Specify individual Z deltas, at most 0.1 m, on seen affected interior vertices. Boundary XY/endpoints, point map, semantics and routes remain fixed. Four full audits must support every addition trace before publication. No automatic edits or estimator preference.",
            }
            jobs._inputs(job)
            retries._verify_inputs(inputs)
            jobs._save(path, report)
        page = copy.deepcopy(report["vertices"][offset : offset + 8])
        for row in page:
            row["observations_total"] = len(row["observations"])
            row["observations"] = row["observations"][:8]
        return {
            "file": jobs._artifact(path),
            "candidate_id": cid,
            "vertices": page,
            "vertices_total": len(report["vertices"]),
            "next_offset": offset + 8 if offset + 8 < len(report["vertices"]) else None,
            "protocol": PROTOCOL,
            "effective_bounds_xy": report["effective_bounds_xy"],
            "failures_total": len(report["failures"]),
            "note": report["note"],
        }


def validate(
    root: Path, cid: int, preview_file: dict[str, Any], edits: Any
) -> dict[str, Any]:
    job, parent, ir = _parent(root, cid)
    if "height_repair" in job or jobs._remaining(job) < 2:
        raise ValueError(
            "one height trial requires two remaining shared attempts: edit and combined patch"
        )
    report = _read(preview_file)
    retries._verify_inputs(report["inputs"])
    if (
        report["candidate_id"] != cid
        or report["protocol"] != PROTOCOL
        or report["inputs"] != _inputs(job, parent)
    ):
        raise ValueError("height preview belongs to another decision or protocol")
    if not isinstance(edits, list) or not 1 <= len(edits) <= PROTOCOL["maximum_edits"]:
        raise ValueError("choose 1..16 distinct inspected interior vertices")
    offered = {(r["boundary_id"], r["vertex_index"]): r for r in report["vertices"]}
    chosen = set()
    for edit in edits:
        if (
            not isinstance(edit, dict)
            or set(edit) != {"boundary_id", "vertex_index", "delta_z_m", "reason"}
            or any(type(edit[k]) is not int for k in ("boundary_id", "vertex_index"))
            or type(edit["delta_z_m"]) not in (int, float)
            or not math.isfinite(edit["delta_z_m"])
            or not 0 < abs(edit["delta_z_m"]) <= PROTOCOL["maximum_delta_z_m"]
            or not isinstance(edit["reason"], str)
            or not edit["reason"].strip()
        ):
            raise ValueError(
                "each height edit needs an inspected vertex, finite nonzero delta <=0.1 m and reason"
            )
        key = (edit["boundary_id"], edit["vertex_index"])
        if key not in offered or key in chosen:
            raise ValueError("choose distinct inspected affected interior vertices")
        chosen.add(key)
    return {
        "inputs": {**report["inputs"], "height_preview": preview_file},
        "parent": parent,
        "original": ir,
    }


def _preserved(
    original: dict[str, Any], after: dict[str, Any], edits: list[dict[str, Any]]
) -> None:
    expected = copy.deepcopy(original)
    rows = {b["id"]: b for b in expected["boundaries"]}
    for e in edits:
        rows[e["boundary_id"]]["geometry"][e["vertex_index"]][2] += e["delta_z_m"]
    if expected != after:
        raise ValueError(
            "height export changed boundary XY/endpoints, unchosen Z, metadata or lane relations"
        )


def _checks(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    holds = []
    old, new = _audits(before), _audits(after)
    for name in old:
        a, b = jobs._diagnose_audit(old[name]), jobs._diagnose_audit(new[name])
        if (
            not a["complete"]
            or not b["complete"]
            or a["protocol"] != b["protocol"]
            or new[name]["quality"].get("problems_limited", True)
        ):
            holds.append(f"{name}:incomplete_or_changed_audit_protocol")
            continue
        x, y = [
            {l["lane"]: l for l in v["quality"]["lanes"]}
            for v in (old[name], new[name])
        ]
        if x.keys() != y.keys():
            holds.append(f"{name}:lane_membership_changed")
            continue
        for lid, lane in y.items():
            for trace in ("center", "left", "right"):
                row = lane[trace]
                if row["samples"] != x[lid][trace]["samples"]:
                    holds.append(f"{name}:sample_count_changed:{lid}:{trace}")
                if (
                    not row["samples"]
                    or row["supported"] != row["samples"]
                    or not row["start_supported"]
                    or not row["end_supported"]
                ):
                    holds.append(f"{name}:addition_not_fully_supported:{lid}:{trace}")
    for left_name, right_name in (
        ("editable", "reopened_osm"),
        ("consensus_editable", "consensus_reopened_osm"),
    ):
        if (
            jobs._diagnose_audit(new[left_name])["lanes"]
            != jobs._diagnose_audit(new[right_name])["lanes"]
        ):
            holds.append(f"{left_name}:ir_and_reopened_source_differ")
    return {
        "protocol": "fixed_four_audits_all_addition_traces_fully_supported_after_bounded_z_hypothesis",
        "holds": sorted(set(holds)),
        "passes": not holds,
        "independent_accuracy_established": False,
    }


def edit(
    root: Path,
    cid: int,
    edits: list[dict[str, Any]],
    preview_file: dict[str, Any],
    reason: str,
    prepared: dict[str, Any],
) -> dict[str, Any]:
    with jobs._locked(root):
        job = jobs._load(root)
        validated = validate(root, cid, preview_file, edits)
        if validated["inputs"] != prepared["inputs"]:
            raise ValueError("height inputs changed after validation")
        parent, original = validated["parent"], validated["original"]
        attempt: dict[str, Any] = {
            "id": len(job["attempts"]) + 1,
            "kind": "corridor_lanes",
            "status": "running",
            "reason": reason,
            "parent_candidate_id": cid,
            "height_inputs": prepared["inputs"],
            "height_edits": edits,
            "road_options": parent["road_options"],
            "geometry_inputs": parent["geometry_inputs"],
            "corridor_proposal": parent["corridor_proposal"],
        }
        job["attempts"].append(attempt)
        job["height_repair"] = {"candidate_id": attempt["id"]}
        jobs._save(root / "job.json", job)
        target = root / f"candidate-{attempt['id']:02d}"
        try:
            if target.exists():
                raise FileExistsError(str(target))
            ir = copy.deepcopy(original)
            boundaries = {b["id"]: b for b in ir["boundaries"]}
            for e in edits:
                boundaries[e["boundary_id"]]["geometry"][e["vertex_index"]][2] += e[
                    "delta_z_m"
                ]
            trial_path = root / f"candidate-{attempt['id']:02d}-height-trial.json"
            jobs._save(trial_path, ir)
            attempt["height_trial"] = jobs._artifact(trial_path)
            module = jobs.core()
            assert module is not None
            payload = json.loads(module.edit_vector_map_relations(str(trial_path)))
            canonical = json.loads(payload["map_json"])
            _preserved(original, canonical, edits)
            if (
                payload["projector_info"]
                != Path(parent["files"]["projector"]["path"]).read_text()
            ):
                raise ValueError("height edit changed projection")
            with tempfile.TemporaryDirectory(
                prefix=".mapping-heights-", dir=root
            ) as temporary:
                draft = Path(temporary) / "result"
                files = _publish(payload, draft)["files"]
                reopened = json.loads(
                    json.loads(module.edit_vector_map_relations(files["map"]))[
                        "map_json"
                    ]
                )
                verify_lane_roundtrip(ir, reopened)
                if edges(ir) != edges(reopened) or {
                    l["id"]: l.get("turn_direction") for l in ir["lanes"]
                } != {l["id"]: l.get("turn_direction") for l in reopened["lanes"]}:
                    raise ValueError(
                        "height edit OSM reload changed routing or turn labels"
                    )
                cloud = job["pointcloud"]["files"]["map"]["path"]
                audits = {
                    k: json.loads(
                        module.audit_vector_map_quality_details(cloud, files[f])
                    )
                    for k, f in [("editable", "editable_map"), ("reopened_osm", "map")]
                }
                audits["ground_consensus"] = {
                    k: json.loads(
                        module.audit_vector_map_ground_consensus_details(
                            cloud, files[f]
                        )
                    )
                    for k, f in [("editable", "editable_map"), ("reopened_osm", "map")]
                }
                checks = _checks(_read(parent["quality_report"]), audits)
                for key, value in [("audits", audits), ("checks", checks)]:
                    path = root / f"candidate-{attempt['id']:02d}-height-{key}.json"
                    jobs._save(path, value)
                    attempt[f"height_{key}"] = jobs._artifact(path)
                if not checks["passes"]:
                    raise ValueError(
                        "height hypothesis held: " + ", ".join(checks["holds"][:8])
                    )
                if any(
                    i["severity"] == "error"
                    for i in payload["report"]["autoware_issues"]
                ):
                    raise ValueError("height hypothesis has export errors")
                report = _read(parent["files"]["report"])
                report.update(
                    height_inputs=prepared["inputs"],
                    height_edits=edits,
                    height_checks=checks,
                    height_hypothesis="Explicit bounded Z trial; original source curves remain observations, not edited source data",
                    validation=payload["report"]["validation"],
                    autoware_issues=payload["report"]["autoware_issues"],
                )
                report["files"] = {
                    k: str(target / Path(v).name) for k, v in files.items()
                }
                report["editing"] = {
                    "command": "vectormap",
                    "args": ["mcp", report["files"]["map"]],
                }
                jobs._save(draft / "report.json", report)
                jobs._save(draft / "source-quality.json", audits)
                result = {
                    "status": "audited_draft",
                    "files": {
                        k: {**jobs._artifact(files[k]), "path": v}
                        for k, v in report["files"].items()
                    },
                    "quality_report": {
                        **jobs._artifact(draft / "source-quality.json"),
                        "path": str(target / "source-quality.json"),
                    },
                    "quality": jobs._quality_summary(audits["editable"]),
                    "reopened_quality": jobs._quality_summary(audits["reopened_osm"]),
                    "export_issues": payload["report"]["autoware_issues"],
                    "ground_consensus_quality": {
                        k: jobs._quality_summary(v)
                        for k, v in audits["ground_consensus"].items()
                    },
                    **{k: parent[k] for k in ("extraction", "extent")},
                    "lane_intervals": _intervals(parent),
                }
                jobs._inputs(job)
                retries._verify_inputs(prepared["inputs"])
                jobs._verify(attempt["height_trial"])
                os.rename(draft, target)
            attempt.update(result)
            job["status"] = "candidates_ready"
        except BaseException as error:
            attempt.update(
                status="failed", error=str(error), error_type=type(error).__name__
            )
            if not isinstance(error, Exception) and not jobs._native_panic(error):
                jobs._save(root / "job.json", job)
                raise
        jobs._save(root / "job.json", job)
    return jobs.inspect_mapping_job(str(root))
