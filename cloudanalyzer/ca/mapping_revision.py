"""Explicit new repair sessions seeded by an immutable delivered map pair."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from ca import mapping_job as jobs, mapping_retry as retries


def spent(job: dict[str, Any]) -> int:
    """Count actual attempts, including failures, rather than transferred allocations."""
    count = len(job["attempts"]) - int("continuation" in job)
    stage = job.get("pointcloud_retry")
    if stage and (Path(stage["child_job_dir"]) / "job.json").exists():
        count += len(jobs._load(Path(stage["child_job_dir"]))["attempts"])
    return count


def _artifacts(value: Any, result: dict[str, Any]) -> None:
    if isinstance(value, dict):
        if set(value) == {"path", "sha256", "bytes"}:
            previous = result.get(value["path"])
            if previous is not None and previous != value:
                raise ValueError("continuation lineage has conflicting artifact hashes")
            result[value["path"]] = value
        else:
            for child in value.values():
                _artifacts(child, result)
    elif isinstance(value, list):
        for child in value:
            _artifacts(child, result)


def continue_mapping_run(
    finished_job_dir: str, out_dir: str, max_attempts: int, reason: str
) -> dict[str, Any]:
    """Start a NEW bounded repair session from the exact finished point/HD map pair.

    No odometry, fusion, proposal extraction or HD generation runs at startup.
    Candidate 1 is the adopted seed and spends zero new attempts. Supply an explicit
    new 3..8 attempt budget and reason; earlier budgets and finished runs remain fixed.
    Frozen source, motion, layout, audit thresholds and full-input extent are inherited.
    Inspect gaps, preview a new local-density box, retry_local_density, then draft a
    gap-only child patch and compare/finish explicitly. Connections can also extend
    the retained seed. Failed trials can finish candidate 1 with the exact prior pair.
    Full replacements and added-frame trials are unavailable in continuation roots.
    Artifacts are immutable references: keep earlier run directories accessible.
    Thinning settings describe the last fusion, not uniform density of a hybrid map;
    another density trial must actually reduce them within the existing lower bounds.
    Source support holds and unknown traffic rules remain visible. This does not
    establish independent accuracy or full-drive connectivity.
    """
    from ca import mapping_run as runs

    if type(max_attempts) is not int or not 3 <= max_attempts <= 8:
        raise ValueError(
            "a continuation needs an explicit 3..8 shared HD attempt budget"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("supply a continuation reason")
    source = Path(finished_job_dir).resolve()
    target = Path(out_dir).resolve()
    if target.exists():
        raise FileExistsError(f"continuation output directory already exists: {target}")
    with runs._locked(source):
        run = runs._load(source)
        if (
            run["status"] != "finished"
            or not run.get("output")
            or run["output"]["candidate_id"] is None
        ):
            raise ValueError(
                "continue only a finished run with a delivered audited HD map"
            )
        runs.inspect_mapping_run(str(source))
        root_job = jobs._load(source)
        jobs._inputs(root_job)
        output = run["output"]
        owner = Path(output.get("candidate_job_dir", str(source))).resolve()
        owner_run = runs._load(owner)
        if (
            owner_run["status"] != "finished"
            or owner_run["layout_file"]["sha256"] != run["layout_file"]["sha256"]
        ):
            raise ValueError(
                "delivered owner must be finished with the same frozen layout"
            )
        job = jobs._load(owner)
        jobs._inputs(job)
        candidate = retries._parent(job, output["candidate_id"])
        diagnosis = jobs.diagnose_mapping_candidate(str(owner), candidate["id"])
        audits = [
            diagnosis["editable"],
            diagnosis["reopened_osm"],
            *(diagnosis["ground_consensus"] or {}).values(),
        ]
        if (
            len(audits) != 4
            or any(not a["complete"] or a["errors"] for a in audits)
            or any(i["severity"] == "error" for i in diagnosis["export_issues"])
        ):
            raise ValueError(
                "continuation needs four complete audits and no structural/export errors"
            )
        if (
            job["pointcloud"].get("additional_frame_ids")
            or "frame_adoption" in job["pointcloud"]
        ):
            raise ValueError(
                "continuation currently requires the original retained frame set"
            )
        if job["source"] != root_job["source"] or job["runtime"] != root_job["runtime"]:
            raise ValueError("delivered pair changed source or native runtime")
        for key in ("map", "graph", "trajectory"):
            if output["artifacts"][key] != job["pointcloud"]["files"][key]:
                raise ValueError("seed point map is not the exact delivered pair")
        for key, value in candidate["files"].items():
            if output["artifacts"].get(f"hd_{key}") != value:
                raise ValueError("seed HD map is not the exact delivered pair")
        inputs: dict[str, Any] = {}
        for saved in (run, root_job, job, owner_run):
            _artifacts(saved, inputs)
        for path in (
            source / "run.json",
            source / "job.json",
            owner / "job.json",
            owner / "run.json",
            Path(job["pointcloud"]["reports"]["correction"]),
        ):
            _artifacts(jobs._artifact(path), inputs)
        retries._verify_inputs(inputs)
        seed = deepcopy(candidate)
        inherited = {"job_dir": str(owner), "candidate_id": candidate["id"]}
        seed.update(id=1, seeded_from=inherited)
        previous_spent = root_job.get("continuation", {}).get(
            "previous_spent_attempts", 0
        ) + spent(root_job)
        manifest = {
            "schema": "cloudanalyzer.mapping_continuation.v1",
            "parent_run": str(source),
            "seeded_from": inherited,
            "reason": reason.strip(),
            "new_max_attempts": max_attempts,
            "previous_spent_attempts": previous_spent,
            "inputs": inputs,
            "pointcloud_files": job["pointcloud"]["files"],
            "hd_files": candidate["files"],
            "previous_point_update_scope_archived": "local_update_report"
            in job["pointcloud"]["files"],
            "pointcloud_options_meaning": "last_full_fusion_settings_not_uniform_hybrid_map_resolution",
            "original_source_holds_retained": True,
            "independent_accuracy_established": False,
        }
        target.mkdir(parents=True)
        jobs._save(target / "continuation.json", manifest)
        continuation = {
            k: manifest[k]
            for k in ("parent_run", "seeded_from", "previous_spent_attempts")
        }
        continuation["file"] = jobs._artifact(target / "continuation.json")
        inherited_job = {
            k: deepcopy(job[k])
            for k in (
                "schema",
                "source",
                "runtime",
                "pointcloud",
                "pointcloud_options",
                "minimum_retained_fraction",
                "scope",
                "corridor_proposal",
            )
        }
        inherited_job["pointcloud"]["files"] = {
            k: job["pointcloud"]["files"][k] for k in ("map", "graph", "trajectory")
        }
        inherited_job.update(
            job_dir=str(target),
            status="candidates_ready",
            attempts=[seed],
            selected=None,
            max_attempts=max_attempts,
            continuation=continuation,
            continuation_inputs={
                **inputs,
                "continuation_manifest": continuation["file"],
            },
        )
        if "corridor_refinement" in job:
            inherited_job["corridor_refinement"] = deepcopy(job["corridor_refinement"])
        jobs._save(target / "job.json", inherited_job)
        jobs._save(
            target / "layout-hypothesis.json",
            json.loads(Path(run["layout_file"]["path"]).read_text()),
        )
        new_run = {
            "schema": runs.SCHEMA,
            "revision": 0,
            "status": "needs_agent",
            "layout_file": jobs._artifact(target / "layout-hypothesis.json"),
            "reviewed_candidates": [],
            "history": [
                {
                    "sequence": 1,
                    "action": {"type": "continue"},
                    "reason": reason.strip(),
                    "status": "seeded",
                    "lane_candidate_id": 1,
                    "seeded_from": inherited,
                }
            ],
            "output": None,
            "maximum_actions": 128,
        }
        jobs._save(target / "run.json", new_run)
    return runs.inspect_mapping_run(str(target))
