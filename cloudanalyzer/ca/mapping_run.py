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
from ca import mapping_frames as frames
from ca import mapping_patch as patches
from ca import mapping_local_points as local_points
from ca import mapping_heights as heights
from ca.mapping_geometry import assemble_geometry, corridor_lane_requests

SCHEMA = "cloudanalyzer.mapping_run.v1"
GUIDANCE = """Continue this run autonomously within its fixed layout and attempt budget.
Page the candidate index; inspect relevant candidates before deciding include/defer
ranges. Compare path association, surface levels, spans, edge evidence and competing
bands. Source support does not prove road identity or complete width.
If off-path overlapping bands fragment the route, explicitly refine with
association=trajectory_containing once, then inspect the new proposal IDs afresh.
All original profiles and drafts remain available; unresolved source stays visible
in corridor-extent accounting even when a short connector is added.
After drafting, inspect_connections on an audited draft. For a single forward
one-way lane per piece, explicitly connect inspected consecutive drive pieces.
Short gaps need recorded-path containment, unchanged minimum widths and complete
center/boundary support from both ground estimators after export/reload. Inspect
local and global route station spans separately; connector reachability does not
increase original source-corridor extent or prove permitted turns.
Connections may extend an audited connected or gap-patched draft. Existing
lanes and directed edges are inherited; supply only the new inspected pairs.
Old source samples must not gain failures in either estimator or saved format.
To extend a combined local patch's HD envelope independently,
use inspect_connection_region with explicit bounds_xy containing the frozen
point-update box and sides at most 20 m. Point records remain fixed. Only links
touching repair lanes are offered; inspect exact geometry and connect seen pairs.
Every new trace needs full support from both estimators in IR/reopened OSM.
For missing intervals, inspect_gaps to compare profiles and aligned raw-log
neighborhoods. Counts do not prove coherent ground or a thinning-related cause.
Explicit retry_pointcloud can reduce scan/map thinning once, keeping motion,
frame selection, dynamic-filter policy, thresholds and layout fixed. Continue the
returned child_run, inspect fresh candidates and draft using the transferred
shared HD budget. compare_retry shows gained AND lost source intervals, both
audits and route spans. More points do not establish improvement. Finish the
parent draft if the trial is worse, or finish_retry explicitly after comparing
and finishing the child. Earlier map artifacts are never replaced.
Alternatively inspect_unused_frames after inspect_gaps: read original non-keyframe
returns, interpolated odometry-correction poses and consistency checks against both
corrected bracket scans. retry_frames explicitly adopts inspected eligible IDs,
keeping thinning and all existing corrected poses fixed. ICP corrections are never
applied; these are consistency-tested pose hypotheses, not independent accuracy.
Density and unused-frame trials share one root retry and the same transferred budget.
For a local point update, inspect_local_points with seen root gap IDs and explicit
bounds_xy, then retry_local_frames using inspected eligible frames and that preview.
Alternatively inspect_local_density with seen gap IDs and bounds_xy, then
retry_local_density with that exact preview and explicit bounded thinning options.
This keeps the original retained frames and poses; it needs no unused-frame checks.
The full fusion candidate is still generated; only the previewed XY column at all
heights replaces baseline points. Outside PLY record bytes/attributes/order remain
fixed. Four audits must retain old HD support. Use a combined HD gap patch inside
that box before compare_retry and finish_retry, or finish the root baseline.
If an isolated local addition has height mismatches, inspect_heights returns affected
interior boundary vertices. Explicit edit_heights tries seen individual Z deltas
up to 0.1 m, retaining boundary XY/endpoints and semantics. Every addition trace
must pass all four audits. One shared attempt is spent, with another reserved for
the combined patch. No recursive height trial or automatic estimator preference.
The child retains separate fusion graph/trajectory artifacts plus the unchanged
reference graph/trajectory used for the original extent comparison.
To repair HD intervals locally, draft only observed missing ranges in the retry
child, inspect_patch with root gap IDs already inspected, then patch_gaps with
explicit inspected endpoint pairs ([] when isolated). One shared attempt
preserves the root's lane geometry, IDs, metadata and directed connections and
adds only lanes inside those gaps. All four audits must be complete; no previously
supported retained sample may fail, and new traces must be fully supported.
The point cloud remains the full fusion trial unless a local retry was chosen.
Existing HD source failures remain
explicit. Compare and finish explicitly; failed patches preserve the baseline.
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
        for key in ("gap_observation", "unused_frame_observation", "local_point_observation", "height_observation", "patch_observation", "retry_comparison"):
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
        "history": [{k: v for k, v in a.items() if k not in {"observation", "connection_observation", "gap_observation", "unused_frame_observation", "local_point_observation", "height_observation", "height_inputs", "patch_observation", "patch_inputs", "action"}} for a in run["history"][-8:]],
        "history_limited": len(run["history"]) > 8, "output": run["output"], "guidance": GUIDANCE,
        "action_contract": {"inspect": {"type": "inspect", "candidate_ids": "1..8 IDs from the frozen proposal"},
            "refine": {"type": "refine", "association": "trajectory_containing (one source extraction experiment; inspect new IDs afterward)"},
            "draft": {"type": "draft", "decisions": "complete include/defer choices with reasons and optional observed ranges"},
            "inspect_connections": {"type": "inspect_connections", "candidate_id": "own audited lane, connected or gap-patched draft ID", "offset": "nonnegative; pages of 8"},
            "inspect_connection_region": {"type": "inspect_connection_region", "candidate_id": "own combined local-retry patch", "bounds_xy": "explicit HD-only [xmin,ymin,xmax,ymax]; contains frozen point box; sides <=20 m", "offset": "nonnegative; pages of 8; only links touching repair lanes"},
            "connect": {"type": "connect", "candidate_id": "inspected parent lane draft ID", "pairs": "1..32 inspected {from,to,reason} pairs; one shared HD attempt"},
            "inspect_gaps": {"type": "inspect_gaps", "candidate_id": "own audited lane draft", "offset": "nonnegative; pages of 8"},
            "retry_pointcloud": {"type": "retry_pointcloud", "candidate_id": "inspected baseline", "gap_ids": "1..8 inspected IDs",
                "options": "explicit scan_voxel_m/map_voxel_m; bounded thinning reduction, one child with transferred HD budget"},
            "inspect_unused_frames": {"type": "inspect_unused_frames", "candidate_id": "own gap-inspected baseline", "offset": "nonnegative; pages of 8 pose/registration observations"},
            "retry_frames": {"type": "retry_frames", "candidate_id": "inspected baseline", "frame_ids": "1..64 inspected eligible frame IDs; same thinning, original poses fixed, shared child budget"},
            "inspect_local_points": {"type": "inspect_local_points", "candidate_id": "inspected baseline", "gap_ids": "1..8 root-inspected gap IDs", "bounds_xy": "explicit [xmin,ymin,xmax,ymax]; sides <=20 m after voxel alignment; all heights"},
            "retry_local_frames": {"type": "retry_local_frames", "candidate_id": "inspected baseline", "frame_ids": "1..64 seen eligible frames observing the chosen box", "preview_file": "file artifact from inspect_local_points; needs three shared HD attempts"},
            "inspect_local_density": {"type": "inspect_local_density", "candidate_id": "inspected baseline", "gap_ids": "1..8 root-inspected gap IDs", "bounds_xy": "explicit [xmin,ymin,xmax,ymax]; sides <=20 m after original voxel alignment; all heights"},
            "retry_local_density": {"type": "retry_local_density", "candidate_id": "inspected baseline", "options": "bounded scan_voxel_m/map_voxel_m reduction; original frames and poses fixed", "preview_file": "file artifact from inspect_local_density; needs three shared HD attempts"},
            "inspect_heights": {"type": "inspect_heights", "candidate_id": "own isolated local-retry addition draft", "offset": "nonnegative; pages of 8 affected interior vertices with observed height mismatches"},
            "edit_heights": {"type": "edit_heights", "candidate_id": "same inspected addition draft", "preview_file": "exact height observation artifact", "edits": "1..16 seen {boundary_id, vertex_index, delta_z_m, reason}; nonzero absolute delta <=0.1 m, one trial and two remaining shared attempts"},
            "inspect_patch": {"type": "inspect_patch", "candidate_id": "own unconnected retry-child draft containing gap additions only", "gap_ids": "1..32 root-inspected gap IDs", "offset": "nonnegative; pages of 8 exact geometric endpoint pairs"},
            "patch_gaps": {"type": "patch_gaps", "candidate_id": "inspected gap-only draft ID", "gap_ids": "same root-inspected IDs", "pairs": "all seen endpoint pairs {from: baseline:ID or addition:ID, to: same form, reason}; [] if none, one shared HD attempt"},
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
        return {"status": "failed", "stage": "connections", "error": attempt.get("error"),
                "connection_checks": attempt.get("connection_checks"), "connection_audits": attempt.get("connection_audits")}
    return {"status": "audited_draft", "diagnosis": _diagnosis(root, lid),
            "connection_checks": attempt.get("connection_checks")}


def _retry(root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    a = entry["action"]
    if a["type"] in {"retry_frames", "retry_local_frames"}:
        saved = frames.validate(entry["frame_evidence"], a["frame_ids"])
        gap_ids = sorted({g["gap_id"] for r in saved["frames"] if r["frame_id"] in a["frame_ids"] for g in r["gap_observations"]})
        local_file = a.get('preview_file')
        if local_file is not None:
            gap_ids = local_points.validate(local_file, entry['frame_evidence'], a['candidate_id'], a['frame_ids'])['request']['gap_ids']
        stage = retries.retry(root, a["candidate_id"], entry["frame_evidence"], gap_ids, saved["pointcloud_options"], entry["reason"], frame_ids=a["frame_ids"], local_evidence=local_file)
    else:
        local_file = a.get('preview_file')
        gap_ids = (local_points.validate_density(local_file, entry['gap_evidence'], a['candidate_id'], a['options'])['request']['gap_ids']
                   if local_file is not None else a['gap_ids'])
        stage = retries.retry(root, a["candidate_id"], entry["gap_evidence"], gap_ids, a["options"], entry["reason"], local_evidence=local_file)
    return {"status": stage["status"], "stage": stage,
            "child_run": inspect_mapping_run(stage["child_job_dir"]) if stage["status"] == "ready" else None}


def _patch(root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    job = jobs.inspect_mapping_job(str(root))
    lid = entry["lane_candidate_id"]
    attempt = next((a for a in job["attempts"] if a["id"] == lid), None)
    if attempt is None:
        if len(job["attempts"]) + 1 != lid:
            raise ValueError("mapping job changed outside this run")
        job = patches.patch(root, entry["action"]["candidate_id"], entry["action"]["gap_ids"], entry["action"]["pairs"], entry["patch_preview"], entry["reason"], {"inputs": entry["patch_inputs"]})
        attempt = job["attempts"][-1]
    if (attempt.get("kind") != "patched_corridor_lanes" or attempt["reason"] != entry["reason"]
        or attempt["parent_candidate_id"] != entry["action"]["candidate_id"]
        or attempt["gap_ids"] != entry["action"]["gap_ids"] or attempt["patch_inputs"] != entry["patch_inputs"]
        or attempt["patch_pairs"] != entry["action"]["pairs"] or attempt["patch_preview"] != entry["patch_preview"]):
        raise ValueError("planned patch attempt belongs to a different action")
    retries._verify_inputs(entry["patch_inputs"])
    jobs._verify(entry["patch_preview"])
    if attempt["status"] == "running":
        raise RuntimeError("patch processing did not finish; inspect its retained state before recovery")
    if attempt["status"] != "audited_draft":
        return {"status": "failed", "stage": "patch", "error": attempt.get("error"),
                "patch_checks": attempt.get("patch_checks"), "patch_audits": attempt.get("patch_audits")}
    return {"status": "audited_draft", "diagnosis": _diagnosis(root, lid), "patch_checks": attempt["patch_checks"]}


def _height(root: Path, entry: dict[str, Any]) -> dict[str, Any]:
    job = jobs.inspect_mapping_job(str(root))
    lid, action = entry["lane_candidate_id"], entry["action"]
    attempt = next((row for row in job["attempts"] if row["id"] == lid), None)
    if attempt is None:
        if len(job["attempts"]) + 1 != lid:
            raise ValueError("mapping job changed outside this height action")
        job = heights.edit(root, action["candidate_id"], action["edits"], action["preview_file"],
                           entry["reason"], {"inputs": entry["height_inputs"]})
        attempt = job["attempts"][-1]
    if (attempt.get("height_inputs") != entry["height_inputs"] or attempt.get("height_edits") != action["edits"]
        or attempt["parent_candidate_id"] != action["candidate_id"] or attempt["reason"] != entry["reason"]):
        raise ValueError("height attempt belongs to another action")
    retries._verify_inputs(entry["height_inputs"])
    artifacts = {key: attempt[key] for key in ("height_checks", "height_audits", "height_trial") if key in attempt}
    for artifact in artifacts.values():
        jobs._verify(artifact)
    if attempt["status"] == "running":
        raise RuntimeError("height processing did not finish; inspect retained state before recovery")
    if attempt["status"] != "audited_draft":
        return {"status": "failed", "error": attempt.get("error"), **artifacts}
    return {"status": "audited_draft", "diagnosis": _diagnosis(root, lid), **artifacts}


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
    inspect_unused_frames examines raw gaps and hashed original odometry, returns
    interpolated pose hypotheses and checks against both corrected bracket scans.
    retry_frames fuses explicitly inspected eligible IDs at unchanged thinning,
    with original poses/denominator fixed; it shares the single root retry allocation.
    inspect_patch previews a gap-only child draft against the retained root map.
    patch_gaps preserves original geometry and connections, adopts explicitly seen
    coincident endpoint pairs and adds lanes only inside selected missing intervals.
    Four complete audits require fully supported new traces and no new failure
    locations on retained traces; one transferred HD attempt is spent.
    """
    if not isinstance(reason, str) or not reason.strip() or type(expected_revision) is not int:
        raise ValueError("supply a reason and the inspected integer revision")
    if not isinstance(action, dict) or not isinstance(action.get("type"), str) or action["type"] not in {
        "inspect", "refine", "draft", "inspect_connections", "inspect_connection_region", "connect", "inspect_gaps", "retry_pointcloud", "inspect_unused_frames", "retry_frames", "inspect_local_points", "retry_local_frames", "inspect_local_density", "retry_local_density", "inspect_heights", "edit_heights", "inspect_patch", "patch_gaps", "compare_retry", "finish_retry", "finish", "resume"}:
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
                "inspect_connection_region": {"type", "candidate_id", "bounds_xy", "offset"},
                "inspect_gaps": {"type", "candidate_id", "offset"}, "retry_pointcloud": {"type", "candidate_id", "gap_ids", "options"},
                "inspect_unused_frames": {"type", "candidate_id", "offset"}, "retry_frames": {"type", "candidate_id", "frame_ids"},
                "inspect_local_points": {"type", "candidate_id", "gap_ids", "bounds_xy"}, "retry_local_frames": {"type", "candidate_id", "frame_ids", "preview_file"},
                "inspect_local_density": {"type", "candidate_id", "gap_ids", "bounds_xy"}, "retry_local_density": {"type", "candidate_id", "options", "preview_file"},
                "inspect_heights": {"type", "candidate_id", "offset"}, "edit_heights": {"type", "candidate_id", "preview_file", "edits"},
                "inspect_patch": {"type", "candidate_id", "gap_ids", "offset"}, "patch_gaps": {"type", "candidate_id", "gap_ids", "pairs"},
                "compare_retry": {"type", "candidate_id"}, "finish_retry": {"type", "candidate_id"},
                "finish": {"type", "candidate_id"}, "resume": {"type"}}
        if set(action) != keys[kind]:
            raise ValueError("supply only the required action fields from action_contract")
        if kind == "resume":
            if run["status"] != "interrupted" or not run["history"] or run["history"][-1]["action"]["type"] not in {
                "draft", "refine", "inspect_connections", "inspect_connection_region", "connect", "inspect_gaps", "retry_pointcloud", "inspect_unused_frames", "retry_frames", "inspect_local_points", "retry_local_frames", "inspect_local_density", "retry_local_density", "inspect_heights", "edit_heights", "inspect_patch", "patch_gaps", "compare_retry"}:
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
            elif kind in {"inspect_connections", "inspect_connection_region", "connect"}:
                cid = action["candidate_id"]
                if type(cid) is not int or cid not in [h.get("lane_candidate_id") for h in run["history"]]:
                    raise ValueError("use a lane draft generated by this run")
                connection_parent = connections._parent(job, cid)[0]
                if kind in {"inspect_connections", "inspect_connection_region"}:
                    if type(action["offset"]) is not int or action["offset"] < 0:
                        raise ValueError("connection offset must be a nonnegative integer")
                    if kind == "inspect_connection_region":
                        connections.connection_region(job, connection_parent, action["bounds_xy"])
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
            elif kind in {"inspect_gaps", "retry_pointcloud", "inspect_unused_frames", "retry_frames", "inspect_local_points", "retry_local_frames", "inspect_local_density", "retry_local_density"}:
                cid = action["candidate_id"]
                if type(cid) is not int or cid not in [h.get("lane_candidate_id") for h in run["history"]]:
                    raise ValueError("use an audited lane draft generated by this run")
                retries._parent(job, cid)
                if kind in {'inspect_local_points', 'inspect_local_density', 'retry_local_density'}:
                    gaps_seen = [h['gap_observation'] for h in run['history'] if h.get('gap_observation', {}).get('candidate_id') == cid]
                    frames_seen = [h['unused_frame_observation'] for h in run['history'] if h.get('unused_frame_observation', {}).get('candidate_id') == cid]
                    if not gaps_seen or (kind == 'inspect_local_points' and not frames_seen):
                        raise ValueError("inspect root gaps and, for frame updates, unused frames before a local preview")
                    gap_evidence = gaps_seen[-1]['file']
                    frame_evidence = frames_seen[-1]['file'] if kind == 'inspect_local_points' else None
                    seen_ids = {g['id'] for r in gaps_seen if r['file'] == gap_evidence for g in r['gaps']}
                    if kind == 'retry_local_density':
                        if not run.get('pointcloud_retry_allowed', True) or 'pointcloud_retry' in job:
                            raise ValueError("one point-cloud retry is allowed per root run; no recursive retries")
                        if job['remaining_attempts'] < 3:
                            raise ValueError("local density retry needs three shared HD attempts")
                        previews = [h['local_point_observation'] for h in run['history'] if h.get('local_point_observation', {}).get('candidate_id') == cid]
                        if not any(r['file'] == action['preview_file'] for r in previews):
                            raise ValueError("inspect the chosen local density preview through this run")
                        local_points.validate_density(action['preview_file'], gap_evidence, cid, action['options'])
                    else:
                        if not isinstance(action['gap_ids'], list) or any(type(i) is not int or i not in seen_ids for i in action['gap_ids']):
                            raise ValueError("inspect every chosen local gap through this run")
                        local_points.validate_region(gap_evidence, cid, action['gap_ids'], action['bounds_xy'])
                elif kind in {"inspect_gaps", "inspect_unused_frames"}:
                    if type(action["offset"]) is not int or action["offset"] < 0:
                        raise ValueError("gap offset must be a nonnegative integer")
                    if kind == "inspect_unused_frames":
                        receipts_gaps = [h["gap_observation"] for h in run["history"] if h.get("gap_observation", {}).get("candidate_id") == cid]
                        if not receipts_gaps:
                            raise ValueError("inspect gaps through this run before unused frames")
                        gap_evidence = receipts_gaps[-1]["file"]
                else:
                    if not run.get("pointcloud_retry_allowed", True) or "pointcloud_retry" in job:
                        raise ValueError("one point-cloud retry is allowed per root run; no recursive retries")
                    if job["remaining_attempts"] < (3 if kind == 'retry_local_frames' else 2):
                        raise ValueError("retry needs enough shared HD attempts (three for a local combined patch)")
                    if kind == "retry_pointcloud":
                        receipts_gaps = [h["gap_observation"] for h in run["history"] if h.get("gap_observation", {}).get("candidate_id") == cid]
                        if not receipts_gaps:
                            raise ValueError("inspect gaps through this run before retrying the point map")
                        gap_evidence = receipts_gaps[-1]["file"]
                        retries.validate(gap_evidence, action["gap_ids"], action["options"])
                        seen_gaps = {g["id"] for r in receipts_gaps if r["file"] == gap_evidence for g in r["gaps"]}
                        if not set(action["gap_ids"]) <= seen_gaps:
                            raise ValueError("inspect every chosen gap before the density trial")
                    else:
                        receipts_frames = [h["unused_frame_observation"] for h in run["history"] if h.get("unused_frame_observation", {}).get("candidate_id") == cid]
                        if not receipts_frames:
                            raise ValueError("inspect unused frames through this run before fusion")
                        frame_evidence = receipts_frames[-1]["file"]
                        frames.validate(frame_evidence, action["frame_ids"])
                        seen_frames = {r["frame_id"] for receipt in receipts_frames if receipt["file"] == frame_evidence for r in receipt["frames"]}
                        if not set(action["frame_ids"]) <= seen_frames:
                            raise ValueError("inspect every chosen frame before fusion")
                        if kind == 'retry_local_frames':
                            previews = [h['local_point_observation'] for h in run['history'] if h.get('local_point_observation', {}).get('candidate_id') == cid]
                            if not any(r['file'] == action['preview_file'] for r in previews):
                                raise ValueError("inspect the chosen local point preview through this run before retrying")
                            local_points.validate(action['preview_file'], frame_evidence, cid, action['frame_ids'])
            elif kind in {'inspect_heights', 'edit_heights'}:
                cid = action['candidate_id']
                if type(cid) is not int or cid not in [h.get('lane_candidate_id') for h in run['history']]:
                    raise ValueError('inspect heights on an audited addition generated by this run')
                heights._parent(root, cid)
                if kind == 'inspect_heights':
                    if type(action['offset']) is not int or action['offset'] < 0: raise ValueError('height offset must be nonnegative')
                else:
                    receipts = [h['height_observation'] for h in run['history'] if h.get('height_observation',{}).get('candidate_id') == cid]
                    if not any(r['file'] == action['preview_file'] for r in receipts): raise ValueError('inspect the chosen height preview through this run')
                    height_inputs = heights.validate(root, cid, action['preview_file'], action['edits'])['inputs']
                    seen = {(v['boundary_id'],v['vertex_index']) for r in receipts if r['file']==action['preview_file'] for v in r['vertices']}
                    if not {(e['boundary_id'],e['vertex_index']) for e in action['edits']} <= seen:
                        raise ValueError('inspect every chosen height vertex before adoption')
            elif kind in {"inspect_patch", "patch_gaps"}:
                cid = action["candidate_id"]
                if type(cid) is not int or cid not in [h.get("lane_candidate_id") for h in run["history"]]:
                    raise ValueError("patch a lane draft generated by this retry run")
                patch_inputs = patches.validate(root, cid, action["gap_ids"])["inputs"]
                if kind == "inspect_patch":
                    if type(action["offset"]) is not int or action["offset"] < 0:
                        raise ValueError("patch offset must be nonnegative")
                else:
                    receipts_patch = [h["patch_observation"] for h in run["history"] if h.get("patch_observation", {}).get("candidate_id") == cid and h["patch_observation"]["gap_ids"] == action["gap_ids"]]
                    if not receipts_patch:
                        raise ValueError("inspect the gap patch and endpoint pairs through this run first")
                    patch_preview = receipts_patch[-1]["file"]
                    patches.validate_pairs(patch_preview, action["pairs"])
                    seen_patch = {(p["from"], p["to"]) for receipt in receipts_patch if receipt["file"] == patch_preview for p in receipt["pairs"]}
                    if not {(p["from"], p["to"]) for p in action["pairs"]} <= seen_patch:
                        raise ValueError("inspect every patch endpoint pair before adoption")
            elif kind in {"compare_retry", "finish_retry"}:
                cid = action["candidate_id"]
                if type(cid) is not int or cid < 1 or job.get("pointcloud_retry", {}).get("status") != "ready":
                    raise ValueError("use an audited candidate in the ready retry child")
                child_root = Path(job["pointcloud_retry"]["child_job_dir"])
                child_run = _load(child_root)
                child_job = jobs.inspect_mapping_job(str(child_root))
                attempt = retries._parent(child_job, cid)
                if job['pointcloud_retry'].get('strategy') in {'local_unused_frames', 'local_density'} and 'patch_inputs' not in attempt:
                    raise ValueError("local point updates require a combined HD patch retaining the baseline map")
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
                    if attempt is None or attempt.get("kind") not in {"corridor_lanes", "connected_corridor_lanes", "patched_corridor_lanes"} or cid not in [h.get("lane_candidate_id") for h in run["history"]]:
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
            elif kind in {"retry_pointcloud", "inspect_local_density", "retry_local_density"}:
                entry["gap_evidence"] = gap_evidence
            elif kind == "inspect_unused_frames":
                entry["gap_evidence"] = gap_evidence
            elif kind == "retry_frames":
                entry["frame_evidence"] = frame_evidence
            elif kind in {'inspect_local_points', 'retry_local_frames'}:
                entry['frame_evidence'] = frame_evidence
                if kind == 'inspect_local_points': entry['gap_evidence'] = gap_evidence
            elif kind == 'edit_heights':
                entry['lane_candidate_id'] = len(job['attempts']) + 1
                entry['height_inputs'] = height_inputs
            elif kind == "patch_gaps":
                entry["lane_candidate_id"] = len(job["attempts"]) + 1
                entry["patch_inputs"] = patch_inputs
                entry["patch_preview"] = patch_preview
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
            elif entry["action"]["type"] in {"inspect_connections", "inspect_connection_region"}:
                observation_connections = connections.inspect_connections(root, entry["action"]["candidate_id"], entry["action"]["offset"],
                                                                         entry["action"].get("bounds_xy"))
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
            elif entry["action"]["type"] == "inspect_unused_frames":
                frame_observation = frames.inspect(root, entry["action"]["candidate_id"], entry["gap_evidence"], entry["action"]["offset"])
                entry["unused_frame_observation"] = frame_observation
                entry["status"] = "inspected"
            elif entry['action']['type'] in {'inspect_local_points', 'inspect_local_density'}:
                a = entry['action']
                local_observation = local_points.preview(root, a['candidate_id'], entry['gap_evidence'], entry.get('frame_evidence'), a['gap_ids'], a['bounds_xy'])
                entry['local_point_observation'] = local_observation
                entry['status'] = 'inspected'
            elif entry['action']['type'] == 'inspect_heights':
                a = entry['action']; height_observation = heights.preview(root,a['candidate_id'],a['offset'])
                entry['height_observation'] = height_observation; entry['status'] = 'inspected'
            elif entry['action']['type'] == 'edit_heights':
                outcome = _height(root,entry); entry['status'] = outcome['status']; entry['outcome'] = outcome
            elif entry["action"]["type"] in {"retry_pointcloud", "retry_frames", "retry_local_frames", "retry_local_density"}:
                outcome = _retry(root, entry)
                entry["status"] = outcome["status"]
                entry["outcome"] = {k: v for k, v in outcome.items() if k != "child_run"}
            elif entry["action"]["type"] == "inspect_patch":
                patch_observation = patches.preview(root, entry["action"]["candidate_id"], entry["action"]["gap_ids"], entry["action"]["offset"])
                entry["patch_observation"] = patch_observation
                entry["status"] = "inspected"
            elif entry["action"]["type"] == "patch_gaps":
                outcome = _patch(root, entry)
                entry["status"] = outcome["status"]
                entry["outcome"] = outcome
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
                    if "patch_checks" in attempt:
                        artifacts["hd_patch_checks"] = attempt["patch_checks"]
                    if "connection_checks" in attempt:
                        artifacts["hd_connection_checks"] = attempt["connection_checks"]
                    if "connection_proposal" in attempt:
                        artifacts["hd_connection_proposal"] = attempt["connection_proposal"]
                    for key in ('height_checks', 'height_audits', 'height_trial'):
                        if key in attempt: artifacts[f'hd_{key}'] = attempt[key]
                run["output"] = {"status": "draft_needs_review" if action["candidate_id"] is not None else "hd_unavailable",
                    "candidate_id": action["candidate_id"], "artifacts": artifacts,
                    "diagnosis": diagnosis if action["candidate_id"] is not None else None,
                    "decision_history": str(root / "run.json"), "layout_hypothesis": run["layout_file"],
                    "road_semantics_inferred": False, "deployment_ready": False}
                if job.get("pointcloud_retry"):
                    stage = job["pointcloud_retry"]
                    if 'local_update' in stage:
                        for key in ('report', 'checks', 'audits'):
                            run['output']['artifacts'][f'pointcloud_trial_local_{key}'] = stage['local_update'][key]
                        run['output']['artifacts']['pointcloud_local_preview'] = stage['local_evidence']
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
    elif entry["action"]["type"] in {"inspect_connections", "inspect_connection_region"}:
        answer["connection_observation"] = observation_connections
    elif entry["action"]["type"] == "connect":
        answer["connect_result"] = outcome
    elif entry["action"]["type"] == "inspect_gaps":
        answer["gap_observation"] = gap_observation
    elif entry["action"]["type"] == "inspect_unused_frames":
        answer["unused_frame_observation"] = frame_observation
    elif entry['action']['type'] in {'inspect_local_points', 'inspect_local_density'}:
        answer['local_point_observation'] = local_observation
    elif entry['action']['type'] == 'inspect_heights':
        answer['height_observation'] = height_observation
    elif entry['action']['type'] == 'edit_heights':
        answer['height_result'] = outcome
    elif entry["action"]["type"] in {"retry_pointcloud", "retry_frames", "retry_local_frames", "retry_local_density"}:
        answer["pointcloud_retry_result"] = outcome
    elif entry["action"]["type"] == "patch_gaps":
        answer["patch_result"] = outcome
    elif entry["action"]["type"] == "inspect_patch":
        answer["patch_observation"] = patch_observation
    elif entry["action"]["type"] == "compare_retry":
        answer["retry_comparison"] = comparison
    elif kind in {"draft", "resume"}:
        answer["draft_result"] = outcome
    return answer
