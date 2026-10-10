"""Gap-only HD trials over an exact retained point map and source proposal."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any, cast

from ca import mapping_job as jobs
from ca import mapping_retry as retries


def occupancy(parent: dict[str, Any], gap: dict[str, Any]) -> dict[str, Any]:
    """Source extent gaps can already be occupied by a retained connector."""
    from ca.mapping_patch import _intervals

    try:
        intervals = _intervals(parent)
    except ValueError as error:
        return {"available": False, "reason": str(error)}
    lo, hi = gap["from_m"], gap["to_m"]
    overlaps = [
        {"lane_id": i, "from_m": max(lo, a), "to_m": min(hi, b)}
        for i, (a, b) in intervals.items()
        if max(lo, a) < min(hi, b)
    ]
    cuts = sorted(
        {lo, hi, *[v for item in overlaps for v in (item["from_m"], item["to_m"])]}
    )
    free = [
        {"from_m": a, "to_m": b}
        for a, b in zip(cuts, cuts[1:])
        if not any(item["from_m"] <= a and b <= item["to_m"] for item in overlaps)
    ]
    return {
        "available": True,
        "overlaps": overlaps[:16],
        "overlaps_total": len(overlaps),
        "unoccupied_intervals": free[:16],
        "unoccupied_intervals_total": len(free),
        "limited_in_response": len(overlaps) > 16 or len(free) > 16,
    }


def validate(evidence: dict[str, Any], cid: int, gap_ids: Any) -> dict[str, Any]:
    jobs._verify(evidence)
    saved = json.loads(Path(evidence["path"]).read_text())
    retries._verify_inputs(saved["inputs"])
    jobs._verify(saved["raw_manifest"])
    if saved["candidate_id"] != cid:
        raise ValueError("HD repair evidence belongs to another baseline")
    if (
        not isinstance(gap_ids, list)
        or not 1 <= len(gap_ids) <= 8
        or any(type(i) is not int for i in gap_ids)
        or len(set(gap_ids)) != len(gap_ids)
        or not set(gap_ids) <= {g["id"] for g in saved["gaps"]}
    ):
        raise ValueError("choose 1..8 distinct inspected HD gap IDs")
    return cast(dict[str, Any], saved)


def validate_targets(
    parent: dict[str, Any], saved: dict[str, Any], gap_ids: list[int]
) -> None:
    for gap in saved["gaps"]:
        if gap["id"] in gap_ids:
            occupied = occupancy(parent, gap)
            if not occupied["available"] or not occupied["unoccupied_intervals_total"]:
                raise ValueError(
                    "choose HD gaps with unoccupied intervals; retained lanes and connectors cannot overlap additions"
                )


def verify(job: dict[str, Any]) -> None:
    """Reject redirected artifacts as well as edits to frozen artifact contents."""
    manifest = json.loads(
        Path(job["retry_inputs"]["hd_repair_manifest"]["path"]).read_text()
    )
    for key in (
        "pointcloud",
        "pointcloud_options",
        "source",
        "runtime",
        "scope",
        "minimum_retained_fraction",
        "retry_parent",
    ):
        if job[key] != manifest[key]:
            raise ValueError(f"HD-only repair changed frozen {key}")
    if job["max_attempts"] != manifest["shared_attempts"]:
        raise ValueError("HD-only repair changed the shared attempt budget")
    if job["corridor_proposal"]["file"] != manifest["proposal"]:
        raise ValueError("HD-only repair changed the frozen source proposal")


def start(
    root: Path, cid: int, evidence: dict[str, Any], gap_ids: list[int], reason: str
) -> dict[str, Any]:
    from ca.mapping_run import SCHEMA

    with jobs._locked(root):
        job = jobs._load(root)
        jobs._inputs(job)
        parent = retries._parent(job, cid)
        saved = validate(evidence, cid, gap_ids)
        if saved["inputs"] != retries._inputs(job, parent, saved["inputs"]["layout"]):
            raise ValueError("HD repair evidence does not match the retained map")
        stage = job.get("pointcloud_retry")
        if stage:
            if (
                stage.get("strategy"),
                stage["candidate_id"],
                stage["gap_ids"],
                stage["evidence"],
                stage["reason"],
            ) != ("hd_only", cid, gap_ids, evidence, reason):
                raise ValueError(
                    "the shared repair allocation belongs to another decision"
                )
            if stage["status"] == "running":
                raise RuntimeError(
                    "HD repair preparation was interrupted; retain its files before recovery"
                )
            return cast(dict[str, Any], stage)
        validate_targets(parent, saved, gap_ids)
        available = jobs._remaining(job)
        if available < 3 or "retry_inputs" in job:
            raise ValueError(
                "HD-only repair needs three shared HD attempts and a root run"
            )
        child = root / "hd-repair"
        if child.exists():
            raise FileExistsError(f"HD repair directory already exists: {child}")
        stage = {
            "strategy": "hd_only",
            "status": "running",
            "candidate_id": cid,
            "gap_ids": gap_ids,
            "evidence": evidence,
            "reason": reason,
            "child_job_dir": str(child),
            "allocated_attempts": available,
            "pointcloud_regenerated": False,
        }
        job["pointcloud_retry"] = stage
        jobs._save(root / "job.json", job)
        try:
            child.mkdir()
            shutil.copyfile(
                saved["inputs"]["layout"]["path"], child / "layout-hypothesis.json"
            )
            proposal = json.loads(Path(parent["corridor_proposal"]["path"]).read_text())
            child_job = {
                "schema": jobs.SCHEMA,
                "job_dir": str(child),
                "source": job["source"],
                "runtime": job["runtime"],
                "status": "pointcloud_ready",
                "pointcloud": copy.deepcopy(job["pointcloud"]),
                "pointcloud_options": copy.deepcopy(job["pointcloud_options"]),
                "attempts": [],
                "selected": None,
                "max_attempts": available,
                "scope": job["scope"],
                "minimum_retained_fraction": job["minimum_retained_fraction"],
                "retry_parent": {
                    "job_dir": str(root),
                    "candidate_id": cid,
                    "gap_ids": gap_ids,
                    "reason": reason,
                },
                "retry_inputs": {
                    **saved["inputs"],
                    "retry_evidence": evidence,
                    "raw_manifest": saved["raw_manifest"],
                },
                "corridor_proposal": {
                    "status": "ready",
                    "file": parent["corridor_proposal"],
                    "options": proposal["protocol"].get("options", {}),
                    "summary": jobs._corridor_summary(proposal),
                },
            }
            manifest = {
                "schema": "cloudanalyzer.hd_only_repair.v1",
                **{
                    k: copy.deepcopy(child_job[k])
                    for k in (
                        "pointcloud",
                        "pointcloud_options",
                        "source",
                        "runtime",
                        "scope",
                        "minimum_retained_fraction",
                        "retry_parent",
                    )
                },
                "proposal": parent["corridor_proposal"],
                "pointcloud_regenerated": False,
                "shared_attempts": available,
                "combined_gap_patch_required": True,
            }
            jobs._save(child / "hd-repair.json", manifest)
            child_job["retry_inputs"]["hd_repair_manifest"] = jobs._artifact(
                child / "hd-repair.json"
            )
            jobs._inputs(child_job)
            jobs._save(child / "job.json", child_job)
            run = {
                "schema": SCHEMA,
                "revision": 0,
                "status": "needs_agent",
                "layout_file": jobs._artifact(child / "layout-hypothesis.json"),
                "reviewed_candidates": [],
                "history": [],
                "output": None,
                "maximum_actions": 128,
                "pointcloud_retry_allowed": False,
            }
            jobs._save(child / "run.json", run)
            stage.update(
                status="ready",
                pointcloud=child_job["pointcloud"],
                corridor_summary=child_job["corridor_proposal"]["summary"],
                manifest=child_job["retry_inputs"]["hd_repair_manifest"],
            )
        except BaseException as error:
            stage.update(
                status="failed", error=str(error), error_type=type(error).__name__
            )
            jobs._save(root / "job.json", job)
            raise
        jobs._save(root / "job.json", job)
        return stage
