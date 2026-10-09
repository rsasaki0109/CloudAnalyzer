"""A persistent mapping loop driven by the calling MCP agent, without an embedded model."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ca import mapping_job as jobs
from ca import mapping_connections as connections
from ca import mapping_retry as retries
from ca.mapping_geometry import assemble_geometry, corridor_lane_requests

SCHEMA = "cloudanalyzer.mapping_run.v1"
GUIDANCE = """Continue this run autonomously within its fixed layout and attempt budget.
Page the candidate index; inspect relevant candidates before deciding include/defer
ranges. Compare path association, surface levels, spans, edge evidence and competing
bands. Source support does not prove road identity or complete width.
If off-path overlapping bands fragment the route, explicitly refine with
association=trajectory_containing once, then inspect the new proposal IDs afresh.
All original profiles and drafts remain available; missing source is never bridged.
After drafting, inspect_connections on an audited draft. For a single forward
one-way lane per piece, explicitly connect inspected consecutive drive pieces.
Short gaps need recorded-path containment, unchanged minimum widths and complete
center/boundary support from both ground estimators after export/reload. Inspect
local and global route station spans separately; connector reachability does not
increase original source-corridor extent or prove permitted turns.
Each connect trial supplies a complete pair set for its unconnected parent;
repeat earlier pairs explicitly to retain them in another trial.
For missing intervals, inspect_gaps to compare profiles and aligned raw-log
neighborhoods. Counts do not prove coherent ground or a thinning-related cause.
Explicit retry_pointcloud can reduce scan/map thinning once, keeping motion,
frame selection, dynamic filtering, thresholds and layout fixed. Continue the
returned child_run, inspect fresh candidates and draft using the transferred
shared HD budget. compare_retry shows gained AND lost source intervals, both
audits and route spans. More points do not establish improvement. Finish the
parent draft if the trial is worse, or finish_retry explicitly after comparing
and finishing the child. Earlier map artifacts are never replaced.
Draft decisions describe a complete replacement hypothesis, not additions to the previous lane map.
The runner binds the fixed layout to every included piece and executes geometry,
lane export and diagnosis. Read both ground estimators and full-input extent before
trying another hypothesis. Do not alter lane count, width requirements, speed or
extent goal to improve a score. Finish with a retained audited draft and explicit
holds, or no HD output if none can be generated. Never claim inferred traffic rules,
independent accuracy or deployment readiness. The calling agent makes decisions;
the runner has no candidate-ranking rules or embedded LLM."""


def _layout(value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != {"boundary_policy", "lanes", "speed_limit_kmh", "reason"}:
        raise ValueError("layout_hypothesis needs boundary_policy, lanes, speed_limit_kmh and reason")
    corridor_lane_requests({"segments": [{"curve_ids": {"center": 1, "left": 2, "right": 3}}]},
        [{"center_curve_id": 1, **{k: value[k] for k in ("lanes", "speed_limit_kmh", "reason")}}], value["boundary_policy"])


def _load(root: Path) -> dict[str, Any]:
    run: dict[str, Any] = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if run.get("schema") != SCHEMA:
        raise ValueError("unsupported mapping run schema")
    jobs._verify(run["layout_file"])
    return run


@contextmanager
def _locked(root: Path) -> Iterator[None]:
    # The underlying mapping job acquires its own lock for each processing stage.
    path = root / ".mapping-run-lock"
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError("mapping run is busy; inspect the recorded action before retrying") from error
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


def start_mapping_run(source: str, out_dir: str, layout_hypothesis: dict[str, Any],
    max_attempts: int = 6, minimum_retained_fraction: float = .9, keyframe_spacing: float = 1.,
    remove_dynamic: bool = True, pointcloud_topic: str | None = None, imu_topic: str | None = None,
) -> dict[str, Any]:
    """Start an agent-driven raw-log-to-both-maps loop with one fixed layout hypothesis.

    Creates a new job, point map and corridor proposals, then returns bounded observations
    and the next decision contract. No candidate IDs or per-piece lane JSON are required
    at startup. layout_hypothesis explicitly supplies boundary_policy=source_span_hypothesis,
    reason, speed_limit_kmh and ordered driving lanes with direction/one_way/fraction/
    minimum_width_m. These remain unverified assumptions. The calling MCP agent inspects
    candidates and uses advance_mapping_run to draft, retry and finish autonomously.
    No LLM or model API key is required by this runner. Existing outputs are never reused
    as new jobs; inspection resumes saved runs. max_attempts must allow at least one
    geometry/lane pair. Processing failures and partial outputs remain visible.
    """
    _layout(layout_hypothesis)
    if type(max_attempts) is not int or not 2 <= max_attempts <= 8:
        raise ValueError("mapping runs need 2..8 shared HD attempts")
    job = jobs.start_mapping_job(source, out_dir, keyframe_spacing=keyframe_spacing,
        remove_dynamic=remove_dynamic, max_attempts=max_attempts, pointcloud_topic=pointcloud_topic,
        imu_topic=imu_topic, minimum_retained_fraction=minimum_retained_fraction)
    root = Path(out_dir).resolve()
    jobs._save(root / "layout-hypothesis.json", layout_hypothesis)
    run = {"schema": SCHEMA, "revision": 0, "status": "preparing", "layout_file": jobs._artifact(root / "layout-hypothesis.json"),
        "reviewed_candidates": [], "history": [], "output": None, "maximum_actions": 128}
    jobs._save(root / "run.json", run)
    try:
        if job["status"] == "pointcloud_ready":
            stage = jobs.propose_mapping_corridors(str(root))
            run["status"] = "needs_agent" if stage["status"] == "ready" else "processing_failed"
        else:
            run["status"] = "processing_failed"
    except BaseException as error:
        run["status"] = "preparation_interrupted"
        run["error"] = str(error)
        raise
    finally:
        jobs._save(root / "run.json", run)
    return inspect_mapping_run(str(root))


def _diagnosis(root: Path, candidate_id: int) -> dict[str, Any]:
    diagnosis = jobs.diagnose_mapping_candidate(str(root), candidate_id)
    diagnosis["lane_specifications_total"] = len(diagnosis.pop("road_options")["lane_specs"])
    diagnosis.pop("effective_options")
    # Full bounded problem locations remain available through the existing diagnosis tool.
    for audit in [diagnosis["editable"], diagnosis["reopened_osm"], *(diagnosis["ground_consensus"] or {}).values()]:
        audit.pop("problems", None)
        audit["problem_locations_in_response"] = False
        audit["lanes_total"] = len(audit["lanes"])
        audit["lanes_limited_in_response"] = len(audit["lanes"]) > 16
        audit["lanes"] = audit["lanes"][:16]
    if diagnosis.get("routes"):
        for metrics in (diagnosis["routes"]["before"], diagnosis["routes"]["after"]):
            metrics["routes_total"] = len(metrics["routes"])
            metrics["routes_limited_in_response"] = len(metrics["routes"]) > 16
            metrics["routes"] = metrics["routes"][:16]
    return diagnosis


def inspect_mapping_run(job_dir: str, offset: int = 0) -> dict[str, Any]:
    """Resume the calling agent with bounded run observations, outputs and decision guidance.

    Reads saved state without native processing or spending attempts. Candidate index
    pages contain 16 items; advance_mapping_run inspect actions return original source
    sections. Summary history includes failures and artifact IDs, while run.json retains
    every action/reason and inspection receipt. Outputs remain drafts with full-input
    extent and unresolved semantics. Use the returned revision for the next action.
    """
    root = Path(job_dir).resolve()
    run = _load(root)
    job = jobs.inspect_mapping_job(str(root))
    jobs._verify(job["source"])
    if job["pointcloud"]:
        for artifact in job["pointcloud"]["files"].values():
            jobs._verify(artifact)
    proposal = job.get("corridor_proposal", {})
    if "corridor_refinement" in job:
        jobs._verify(job["corridor_refinement"]["previous"]["file"])
    index = jobs.inspect_mapping_corridors(str(root), offset=offset) if proposal.get("status") == "ready" else None
    if run["output"]:
        for artifact in run["output"]["artifacts"].values():
            jobs._verify(artifact)
    for h in run["history"]:
        for key in ("gap_observation", "retry_comparison"):
            if key in h:
                jobs._verify(h[key]["file"])
        if "connection_observation" in h:
            jobs._verify(h["connection_observation"]["proposal_file"])
    stage = job.get("pointcloud_retry")
    retry_child = None
    if stage and (Path(stage["child_job_dir"]) / "job.json").exists():
        cj = jobs.inspect_mapping_job(stage["child_job_dir"])
        retry_child = {"job_dir": stage["child_job_dir"], "status": cj["status"], "remaining_attempts": cj["remaining_attempts"]}
    return {"schema": SCHEMA, "job_dir": str(root), "revision": run["revision"], "status": run["status"],
        "layout_hypothesis": json.loads(Path(run["layout_file"]["path"]).read_text(encoding="utf-8")),
        "pointcloud": job["pointcloud"], "remaining_attempts": job["remaining_attempts"], "candidate_index": index,
        "reviewed_candidates": run["reviewed_candidates"], "history_total": len(run["history"]),
        "corridor_refinement": job.get("corridor_refinement"),
        "pointcloud_retry": stage, "retry_child": retry_child, "pointcloud_retry_allowed": run.get("pointcloud_retry_allowed", True),
        "history": [{k: v for k, v in a.items() if k not in {"observation", "connection_observation", "gap_observation", "action"}} for a in run["history"][-8:]],
        "history_limited": len(run["history"]) > 8, "output": run["output"], "guidance": GUIDANCE,
        "action_contract": {"inspect": {"type": "inspect", "candidate_ids": "1..8 IDs from the frozen proposal"},
            "refine": {"type": "refine", "association": "trajectory_containing (one source extraction experiment; inspect new IDs afterward)"},
            "draft": {"type": "draft", "decisions": "complete include/defer choices with reasons and optional observed ranges"},
            "inspect_connections": {"type": "inspect_connections", "candidate_id": "own audited unconnected lane draft ID", "offset": "nonnegative; pages of 8"},
            "connect": {"type": "connect", "candidate_id": "inspected parent lane draft ID", "pairs": "1..32 inspected {from,to,reason} pairs; one shared HD attempt"},
            "inspect_gaps": {"type": "inspect_gaps", "candidate_id": "own audited lane draft", "offset": "nonnegative; pages of 8"},
            "retry_pointcloud": {"type": "retry_pointcloud", "candidate_id": "inspected baseline", "gap_ids": "1..8 inspected IDs",
                "options": "explicit scan_voxel_m/map_voxel_m; bounded thinning reduction, one child with transferred HD budget"},
            "compare_retry": {"type": "compare_retry", "candidate_id": "own audited candidate in retry child"},
            "finish_retry": {"type": "finish_retry", "candidate_id": "compared, finished retry-child candidate"},
            "resume": {"type": "resume"}, "finish": {"type": "finish", "candidate_id": "audited draft ID, or null if none"}},
        "deployment_ready": False}


def _draft(root: Path, run: dict[str, Any], entry: dict[str, Any], layout: dict[str, Any]) -> dict[str, Any]:
    """Journal planned IDs so an interrupted caller cannot replay completed native stages."""
    job = jobs.inspect_mapping_job(str(root))
    gid = entry["geometry_candidate_id"]
    parent = next((a for a in job["attempts"] if a["id"] == gid), None)
    if parent is None:
        if len(job["attempts"]) + 1 != gid:
            raise ValueError("mapping job changed outside this run; inspect retained attempts")
        job = jobs.generate_mapping_geometry(str(root), entry["action"]["decisions"], entry["reason"])
        parent = job["attempts"][-1]
    if parent.get("kind") != "surface_geometry" or parent["reason"] != entry["reason"]:
        raise ValueError("planned geometry attempt belongs to a different action")
    if parent["status"] == "running":
        raise RuntimeError("geometry processing did not finish; inspect its retained state before recovery")
    if parent["status"] != "geometry_draft":
        return {"status": "failed", "stage": "geometry", "error": parent.get("error")}
    for artifact in [*parent["files"].values(), parent["corridor_proposal"]]:
        jobs._verify(artifact)
    geometry = json.loads(Path(parent["files"]["report"]["path"]).read_text(encoding="utf-8"))
    specs = [{"center_curve_id": s["curve_ids"]["center"], **{k: layout[k] for k in ("lanes", "speed_limit_kmh", "reason")}}
             for s in geometry["segments"]]
    lid = gid + 1
    entry["lane_candidate_id"] = lid
    jobs._save(root / "run.json", run)
    attempt = next((a for a in job["attempts"] if a["id"] == lid), None)
    if attempt is None:
        if len(job["attempts"]) + 1 != lid:
            raise ValueError("mapping job changed outside this run; inspect retained attempts")
        job = jobs.generate_mapping_corridor_lanes(str(root), gid, specs, layout["boundary_policy"], entry["reason"])
        attempt = job["attempts"][-1]
    if attempt.get("kind") != "corridor_lanes" or attempt["road_options"]["geometry_candidate_id"] != gid or attempt["reason"] != entry["reason"]:
        raise ValueError("planned lane attempt belongs to a different action")
    if attempt["status"] == "running":
        raise RuntimeError("lane processing did not finish; inspect its retained state before recovery")
    if attempt["status"] != "audited_draft":
        return {"status": "failed", "stage": "lanes", "error": attempt.get("error")}
    return {"status": "audited_draft", "diagnosis": _diagnosis(root, lid)}


def _connect(root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    job = jobs.inspect_mapping_job(str(root))
    lid = entry["lane_candidate_id"]
    attempt = next((a for a in job["attempts"] if a["id"] == lid), None)
    if attempt is None:
        if len(job["attempts"]) + 1 != lid:
            raise ValueError("mapping job changed outside this run")
        job = connections.connect(root, entry["action"]["candidate_id"], entry["action"]["pairs"],
                                  entry["connection_proposal"], entry["reason"])
        attempt = job["attempts"][-1]
    if (attempt.get("kind") != "connected_corridor_lanes" or attempt["reason"] != entry["reason"]
        or attempt["parent_candidate_id"] != entry["action"]["candidate_id"]
        or attempt["connection_pairs"] != entry["action"]["pairs"]
        or attempt["connection_proposal"] != entry["connection_proposal"]):
        raise ValueError("planned connection attempt belongs to a different action")
    if attempt["status"] == "running":
        raise RuntimeError("connection processing did not finish; inspect its retained state before recovery")
    if attempt["status"] != "audited_draft":
        return {"status": "failed", "stage": "connections", "error": attempt.get("error")}
    return {"status": "audited_draft", "diagnosis": _diagnosis(root, lid)}


def _retry(root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    a = entry["action"]
    stage = retries.retry(root, a["candidate_id"], entry["gap_evidence"], a["gap_ids"], a["options"], entry["reason"])
    return {"status": stage["status"], "stage": stage,
            "child_run": inspect_mapping_run(stage["child_job_dir"]) if stage["status"] == "ready" else None}


def advance_mapping_run(job_dir: str, action: dict[str, Any], reason: str, expected_revision: int) -> dict[str, Any]:
    """Execute the calling agent's next inspect/draft/resume/finish decision and persist it.

    Every action needs a reason and the inspected revision; stale retries cannot spend
    attempts twice. Inspect up to eight candidate IDs and read the returned evidence.
    Draft complete include/defer choices only for inspected candidates, using observed
    preview stations for ranges. One draft automatically saves geometry, binds the fixed
    layout, exports IR/OSM and reads both audits, spending up to two shared HD attempts.
    Each draft replaces the hypothesis; prior maps/selection stay intact. Retry based
    on evidence, or refine once with association=trajectory_containing to re-extract
    path-containing source bands at unchanged thresholds. Reinspect the new IDs;
    prior observations cannot authorize a different proposal. Refinement preserves
    the original report and drafts and spends no HD attempt. Its result is a
    geometric association hypothesis, not proof of road identity or permitted use.
    Keep the initial layout unchanged. Resume an interrupted draft/refinement
    without replaying completed stages. Finish with an audited candidate ID or
    null, returning both map paths and explicit source/extent holds; this never selects
    or certifies the map. No candidate IDs are hardcoded or ranked by the runner.
    inspect_connections returns at most eight short connections on an own audited
    single-forward-lane draft. Explicit connect pairs must have been inspected;
    one shared attempt retains existing lanes and verifies both source estimators
    and route topology after OSM reload. Source-corridor extent stays unchanged.
    Connection processing and its inspection also support interrupted resume.
    inspect_gaps returns raw-source neighborhoods and observed missing intervals.
    retry_pointcloud explicitly reduces thinning once with frozen motion, frame
    selection, filtering policy and thresholds, transferring remaining HD attempts to a
    child run. Inspect/draft there, compare_retry its actual audited map, and
    explicitly finish_retry or retain the earlier baseline. No automatic adoption.
    """
    if not isinstance(reason, str) or not reason.strip() or type(expected_revision) is not int:
        raise ValueError("supply a reason and the inspected integer revision")
    if not isinstance(action, dict) or not isinstance(action.get("type"), str) or action["type"] not in {
        "inspect", "refine", "draft", "inspect_connections", "connect", "inspect_gaps", "retry_pointcloud", "compare_retry", "finish_retry", "finish", "resume"}:
        raise ValueError("use an action type from action_contract")
    root = Path(job_dir).resolve()
    with _locked(root):
        run = _load(root)
        if run["revision"] != expected_revision:
            raise ValueError("stale run revision; inspect_mapping_run before deciding again")
        if run["status"] not in {"needs_agent", "interrupted", "processing_failed"}:
            raise ValueError("run is not awaiting an agent decision")
        layout = json.loads(Path(run["layout_file"]["path"]).read_text(encoding="utf-8"))
        _layout(layout)
        job = jobs.inspect_mapping_job(str(root))
        jobs._inputs(job)
        kind = action["type"]
        attempt: dict[str, Any] | None = None
        if run["status"] == "processing_failed" and kind != "finish":
            raise ValueError("finish with retained point-map outputs after preparation failed")
        keys = {"inspect": {"type", "candidate_ids"}, "refine": {"type", "association"}, "draft": {"type", "decisions"},
                "inspect_connections": {"type", "candidate_id", "offset"}, "connect": {"type", "candidate_id", "pairs"},
                "inspect_gaps": {"type", "candidate_id", "offset"}, "retry_pointcloud": {"type", "candidate_id", "gap_ids", "options"},
                "compare_retry": {"type", "candidate_id"}, "finish_retry": {"type", "candidate_id"},
                "finish": {"type", "candidate_id"}, "resume": {"type"}}
        if set(action) != keys[kind]:
            raise ValueError("supply only the required action fields from action_contract")
        if kind == "resume":
            if run["status"] != "interrupted" or not run["history"] or run["history"][-1]["action"]["type"] not in {
                "draft", "refine", "inspect_connections", "connect", "inspect_gaps", "retry_pointcloud", "compare_retry"}:
                raise ValueError("resume requires an interrupted processing action")
            entry = run["history"][-1]
        else:
            if run["status"] == "interrupted" and kind != "finish":
                raise ValueError("resume the interrupted action or finish with retained outputs")
            if len(run["history"]) >= run["maximum_actions"] and kind not in {"finish", "finish_retry"}:
                raise ValueError("action budget exhausted; finish with the retained outputs")
            if kind == "inspect":
                ids = action["candidate_ids"]
                if not isinstance(ids, list) or not 1 <= len(ids) <= 8 or any(type(v) is not int or v < 1 for v in ids) or len(set(ids)) != len(ids):
                    raise ValueError("inspect needs 1..8 distinct positive candidate IDs")
                observation = [jobs.inspect_mapping_corridors(str(root), candidate_id=cid)["candidate"] for cid in ids]
            elif kind == "refine":
                if action["association"] != "trajectory_containing":
                    raise ValueError("refine association must be trajectory_containing")
                if "corridor_refinement" in job:
                    raise ValueError("path-association refinement was already attempted")
                if job.get("corridor_proposal", {}).get("status") != "ready":
                    raise ValueError("refinement requires a ready original corridor proposal")
            elif kind == "draft":
                if job["remaining_attempts"] < 2:
                    raise ValueError("a new draft needs two remaining shared HD attempts")
                proposal = job["corridor_proposal"]["file"]
                jobs._verify(proposal)
                _, validated = assemble_geometry(json.loads(Path(proposal["path"]).read_text()), action["decisions"], reason)
                original_sha = job.get("corridor_refinement", {}).get("previous", {}).get("file", proposal)["sha256"]
                if run.get("reviewed_proposal_sha256", original_sha) != proposal["sha256"]:
                    raise ValueError("inspect candidates from the current proposal before drafting decisions")
                if any(d["candidate_id"] not in run["reviewed_candidates"] for d in validated["decisions"]):
                    raise ValueError("inspect candidates through this run before drafting decisions")
                for d in validated["decisions"]:
                    previews = [c for h in run["history"] if h.get("proposal_sha256", original_sha) == proposal["sha256"]
                                for c in h.get("observation", []) if c["id"] == d["candidate_id"]]
                    stations = {s["station_m"] for c in previews for s in c["sections"]}
                    if d["from_m"] not in stations or d["to_m"] not in stations:
                        raise ValueError("draft ranges must use inspected preview stations; retain unreviewed tails")
            elif kind in {"inspect_connections", "connect"}:
                cid = action["candidate_id"]
                if type(cid) is not int or cid not in [h.get("lane_candidate_id") for h in run["history"]]:
                    raise ValueError("use a lane draft generated by this run")
                connections._parent(job, cid)
                if kind == "inspect_connections":
                    if type(action["offset"]) is not int or action["offset"] < 0:
                        raise ValueError("connection offset must be a nonnegative integer")
                else:
                    if job["remaining_attempts"] < 1:
                        raise ValueError("a connection needs one remaining shared HD attempt")
                    receipts = [h["connection_observation"] for h in run["history"] if h.get("connection_observation", {}).get("candidate_id") == cid]
                    if not receipts:
                        raise ValueError("inspect connections through this run before adopting pairs")
                    proposal_file = receipts[-1]["proposal_file"]
                    chosen = connections.validate_pairs(proposal_file, action["pairs"])
                    seen = {(c["from"], c["to"]) for r in receipts if r["proposal_file"] == proposal_file for c in r["candidates"]}
                    if not set(chosen) <= seen:
                        raise ValueError("inspect every chosen connection pair before adoption")
            elif kind in {"inspect_gaps", "retry_pointcloud"}:
                cid = action["candidate_id"]
                if type(cid) is not int or cid not in [h.get("lane_candidate_id") for h in run["history"]]:
                    raise ValueError("use an audited lane draft generated by this run")
                retries._parent(job, cid)
                if kind == "inspect_gaps":
                    if type(action["offset"]) is not int or action["offset"] < 0:
                        raise ValueError("gap offset must be a nonnegative integer")
                else:
                    if not run.get("pointcloud_retry_allowed", True) or "pointcloud_retry" in job:
                        raise ValueError("one point-cloud retry is allowed per root run; no recursive retries")
                    if job["remaining_attempts"] < 2:
                        raise ValueError("retry needs two remaining shared HD attempts")
                    receipts_gaps = [h["gap_observation"] for h in run["history"] if h.get("gap_observation", {}).get("candidate_id") == cid]
                    if not receipts_gaps:
                        raise ValueError("inspect gaps through this run before retrying the point map")
                    gap_evidence = receipts_gaps[-1]["file"]
                    retries.validate(gap_evidence, action["gap_ids"], action["options"])
                    seen_gaps = {g["id"] for r in receipts_gaps if r["file"] == gap_evidence for g in r["gaps"]}
                    if not set(action["gap_ids"]) <= seen_gaps:
                        raise ValueError("inspect every chosen gap before the density trial")
            elif kind in {"compare_retry", "finish_retry"}:
                cid = action["candidate_id"]
                if type(cid) is not int or cid < 1 or job.get("pointcloud_retry", {}).get("status") != "ready":
                    raise ValueError("use an audited candidate in the ready retry child")
                child_root = Path(job["pointcloud_retry"]["child_job_dir"])
                child_run = _load(child_root)
                child_job = jobs.inspect_mapping_job(str(child_root))
                attempt = retries._parent(child_job, cid)
                if cid not in [h.get("lane_candidate_id") for h in child_run["history"]]:
                    raise ValueError("candidate was not generated by the retry run")
                if kind == "finish_retry":
                    if child_run["status"] != "finished" or child_run["output"]["candidate_id"] != cid:
                        raise ValueError("finish the child with this candidate before delivering the retry")
                    comparisons = [h["retry_comparison"] for h in run["history"] if h.get("retry_comparison", {}).get("retry_candidate_id") == cid]
                    if not comparisons:
                        raise ValueError("compare_retry before delivering a regenerated map")
                    comparison = retries.compare(root, cid)
                    if comparison["file"] != comparisons[-1]["file"]:
                        raise ValueError("retry comparison changed since inspection")
                    diagnosis = _diagnosis(child_root, cid)
            else:
                cid = action["candidate_id"]
                if cid is not None and (type(cid) is not int or cid < 1):
                    raise ValueError("finish needs an audited candidate ID or null")
                if cid is not None:
                    attempt = next((a for a in job["attempts"] if a["id"] == cid), None)
                    if attempt is None or attempt.get("kind") not in {"corridor_lanes", "connected_corridor_lanes"} or cid not in [h.get("lane_candidate_id") for h in run["history"]]:
                        raise ValueError("finish with an audited lane candidate generated by this run")
                    diagnosis = _diagnosis(root, cid)
            entry = {"sequence": len(run["history"]) + 1, "action": action, "reason": reason.strip(), "status": "running"}
            if "corridor_proposal" in job:
                entry["proposal_sha256"] = job["corridor_proposal"].get("file", {}).get("sha256")
            if kind == "draft":
                entry["geometry_candidate_id"] = len(job["attempts"]) + 1
            elif kind == "connect":
                entry["lane_candidate_id"] = len(job["attempts"]) + 1
                entry["connection_proposal"] = proposal_file
            elif kind == "retry_pointcloud":
                entry["gap_evidence"] = gap_evidence
            run["history"].append(entry)
        run["status"] = "action_running"
        run["revision"] += 1
        jobs._save(root / "run.json", run)
        try:
            if kind == "inspect":
                entry["observation"] = observation
                proposal_sha = job["corridor_proposal"]["file"]["sha256"]
                if run.get("reviewed_proposal_sha256", proposal_sha) != proposal_sha:
                    run["reviewed_candidates"] = []
                run["reviewed_proposal_sha256"] = proposal_sha
                run["reviewed_candidates"] = sorted(set(run["reviewed_candidates"]) | set(action["candidate_ids"]))
                entry["status"] = "inspected"
            elif entry["action"]["type"] == "refine":
                outcome = jobs._refine_mapping_corridors(str(root), entry["reason"])
                entry["status"] = outcome["status"]
                entry["outcome"] = outcome
                if outcome["status"] == "ready":
                    run["reviewed_candidates"] = []
                    run["reviewed_proposal_sha256"] = outcome["file"]["sha256"]
            elif entry["action"]["type"] == "inspect_connections":
                observation_connections = connections.inspect_connections(root, entry["action"]["candidate_id"], entry["action"]["offset"])
                entry["connection_observation"] = observation_connections
                entry["status"] = "inspected"
            elif entry["action"]["type"] == "connect":
                outcome = _connect(root, entry)
                entry["status"] = outcome["status"]
                entry["outcome"] = outcome
            elif entry["action"]["type"] == "inspect_gaps":
                gap_observation = retries.inspect_gaps(root, entry["action"]["candidate_id"], entry["action"]["offset"], run["layout_file"])
                entry["gap_observation"] = gap_observation
                entry["status"] = "inspected"
            elif entry["action"]["type"] == "retry_pointcloud":
                outcome = _retry(root, entry)
                entry["status"] = outcome["status"]
                entry["outcome"] = {k: v for k, v in outcome.items() if k != "child_run"}
            elif entry["action"]["type"] == "compare_retry":
                comparison = retries.compare(root, entry["action"]["candidate_id"])
                entry["retry_comparison"] = comparison
                entry["status"] = "compared"
            elif kind in {"draft", "resume"}:
                outcome = _draft(root, run, entry, layout)
                entry["status"] = outcome["status"]
                entry["outcome"] = outcome
            else:
                artifacts = dict((child_job if kind == "finish_retry" else job)["pointcloud"]["files"])
                if action["candidate_id"] is not None:
                    assert attempt is not None
                    artifacts.update({f"hd_{k}": v for k,v in attempt["files"].items()})
                    artifacts["hd_source_audits"] = attempt["quality_report"]
                run["output"] = {"status": "draft_needs_review" if action["candidate_id"] is not None else "hd_unavailable",
                    "candidate_id": action["candidate_id"], "artifacts": artifacts,
                    "diagnosis": diagnosis if action["candidate_id"] is not None else None,
                    "decision_history": str(root / "run.json"), "layout_hypothesis": run["layout_file"],
                    "road_semantics_inferred": False, "deployment_ready": False}
                if job.get("pointcloud_retry"):
                    stage = job["pointcloud_retry"]
                    run["output"]["pointcloud_retry_decision"] = {"adopted": kind == "finish_retry",
                        "status": stage["status"], "child_job_dir": stage["child_job_dir"],
                        "baseline_candidate_id": stage["candidate_id"], "reason": reason.strip()}
                    for h in run["history"]:
                        if "retry_comparison" in h:
                            c = h["retry_comparison"]
                            run["output"]["artifacts"][f"retry_comparison_{c['retry_candidate_id']}"] = c["file"]
                if kind == "finish_retry":
                    run["output"].update(candidate_job_dir=str(child_root), retry_comparison=comparison,
                        baseline_candidate_id=job["pointcloud_retry"]["candidate_id"], child_decision_history=str(child_root / "run.json"))
                    run["output"]["artifacts"]["retry_comparison"] = comparison["file"]
                entry["status"] = "finished"
            run["status"] = "finished" if kind in {"finish", "finish_retry"} else "needs_agent"
        except BaseException as error:
            run["status"] = "interrupted"
            entry["error"] = str(error)
            entry["error_type"] = type(error).__name__
            raise
        finally:
            jobs._save(root / "run.json", run)
    answer = inspect_mapping_run(str(root))
    if kind == "inspect":
        answer["observation"] = observation
    elif entry["action"]["type"] == "refine":
        answer["refine_result"] = outcome
    elif entry["action"]["type"] == "inspect_connections":
        answer["connection_observation"] = observation_connections
    elif entry["action"]["type"] == "connect":
        answer["connect_result"] = outcome
    elif entry["action"]["type"] == "inspect_gaps":
        answer["gap_observation"] = gap_observation
    elif entry["action"]["type"] == "retry_pointcloud":
        answer["pointcloud_retry_result"] = outcome
    elif entry["action"]["type"] == "compare_retry":
        answer["retry_comparison"] = comparison
    elif kind in {"draft", "resume"}:
        answer["draft_result"] = outcome
    return answer
