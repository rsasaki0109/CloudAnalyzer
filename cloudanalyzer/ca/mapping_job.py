"""Persistent mapping runs controlled by an agent through small, explicit tools.

The caller chooses hypotheses and reads evidence between attempts. This module
executes and records those choices; it does not contain a language model or infer
traffic rules from a source-support score.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ca._rust import core
from ca.posegraph_fix import fix_session, odometry
from ca.vector_map import build_vector_map, _publish

SCHEMA = "cloudanalyzer.mapping_job.v1"


def _artifact(path: str | Path) -> dict[str, Any]:
    file = Path(path).resolve()
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(file), "sha256": digest.hexdigest(), "bytes": file.stat().st_size}


def _verify(artifact: dict[str, Any]) -> None:
    if _artifact(artifact["path"]) != artifact:
        raise ValueError(f"recorded input or artifact changed: {artifact['path']}")


def _save(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def _locked(root: Path) -> Iterator[None]:
    lock = root / ".mapping-lock"
    try:
        stream = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise RuntimeError("mapping job is busy; inspect job.json before retrying") from error
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink()


def _native() -> dict[str, Any]:
    module = core()
    if module is None or not hasattr(module, "propose_road_corridors"):
        raise RuntimeError('mapping jobs need an updated Rust core: pip install "cloudanalyzer[fast]"')
    extension = getattr(module, "_core", module)
    file = extension.__file__
    if not file:
        raise RuntimeError("native core has no binary path for provenance")
    return {"version": getattr(module, "__version__", "unknown"),
            "extension": _artifact(file)}


def _load(root: Path) -> dict[str, Any]:
    job: dict[str, Any] = json.loads((root / "job.json").read_text(encoding="utf-8"))
    if job.get("schema") != SCHEMA:
        raise ValueError("unsupported mapping job schema")
    return job


def _native_panic(error: BaseException) -> bool:
    # PyO3 panics inherit BaseException, unlike ordinary native ValueErrors.
    return type(error).__module__ == "pyo3_runtime" and type(error).__name__ == "PanicException"


def _inputs(job: dict[str, Any]) -> None:
    if job["pointcloud"] is None:
        raise ValueError("generate the point-cloud map before an HD candidate")
    _verify(job["source"])
    for artifact in job.get("retry_inputs", {}).values():
        _verify(artifact)
    for artifact in job["pointcloud"].get("source_motion", {}).values():
        _verify(artifact)
    if _native() != job["runtime"]["native"]:
        raise ValueError("native core changed; use a new mapping job")
    for artifact in job["pointcloud"]["files"].values():
        _verify(artifact)


def _remaining(job: dict[str, Any]) -> int:
    """HD attempts transferred to a retry child cannot be spent by its parent."""
    return int(job["max_attempts"] - len(job["attempts"]) - job.get("pointcloud_retry", {}).get("allocated_attempts", 0))


def inspect_mapping_job(job_dir: str) -> dict[str, Any]:
    """Read a mapping job's artifact paths, attempt evidence and remaining budget.

    No point clouds are loaded. Running/failed stages remain visible. Read each
    report before choosing another candidate or selecting a generated draft.
    """
    job = _load(Path(job_dir).resolve())
    for attempt in job["attempts"]:
        if attempt["status"] == "audited_draft":
            attempt["extent"] = _extent(attempt["extraction"], job.get("minimum_retained_fraction", 0.9))
    remaining = _remaining(job)
    job["remaining_attempts"] = remaining
    ready = job.get("pointcloud") is not None and job["status"] not in {"pointcloud_failed", "pointcloud_running"}
    job["next_actions"] = (["propose_mapping_corridors"] if ready and "corridor_proposal" not in job else []) + (
        ["inspect_mapping_corridors"] if job.get("corridor_proposal", {}).get("status") == "ready" else []
    ) + (["generate_mapping_geometry"] if ready and remaining and job.get("corridor_proposal", {}).get("status") == "ready" else []) + (
        ["inspect_mapping_geometry"] if any(a["status"] == "geometry_draft" for a in job["attempts"]) else []
    ) + (["generate_mapping_corridor_lanes"] if ready and remaining and any(a["status"] == "geometry_draft" for a in job["attempts"]) else []
    ) + (["generate_mapping_candidate"] if ready and remaining else []) + (
        ["select_mapping_candidate"] if any(a["status"] == "audited_draft" and a["extent"]["passes_requested_extent"] for a in job["attempts"]) else []
    )
    return job


def start_mapping_job(
    source: str,
    out_dir: str,
    keyframe_spacing: float = 1.0,
    remove_dynamic: bool = True,
    max_attempts: int = 4,
    pointcloud_topic: str | None = None,
    imu_topic: str | None = None,
    minimum_retained_fraction: float = 0.9,
) -> dict[str, Any]:
    """Generate a point-cloud map and corrected trajectory from a raw recording.

    Accepts MCAP, ROS1 bag or rosbag2 SQLite recordings. Uses LiDAR odometry,
    loop closure, the recording's IMU gravity and optional dynamic removal.
    A NEW output directory holds the job and stage reports, including failures.
    Call propose_mapping_corridors to inspect source bands without lane assumptions,
    then generate_mapping_candidate with explicit road assumptions and a reason.
    """
    if type(max_attempts) is not int or not 1 <= max_attempts <= 8:
        raise ValueError("max_attempts must be an integer between 1 and 8")
    if not 0 <= keyframe_spacing < float("inf"):
        raise ValueError("keyframe_spacing must be finite and nonnegative")
    if not 0 <= minimum_retained_fraction <= 1:
        raise ValueError("minimum_retained_fraction must be between 0 and 1")
    source_path = Path(source).resolve()
    if source_path.suffix.lower() not in {".mcap", ".bag", ".db3"} or not source_path.is_file():
        raise ValueError("source must be an existing MCAP, ROS1 bag or rosbag2 SQLite file")
    native = _native()
    root = Path(out_dir).resolve()
    job: dict[str, Any] = {
        "schema": SCHEMA, "job_dir": str(root), "status": "pointcloud_running",
        "source": _artifact(source_path), "runtime": {"python": platform.python_version(), "native": native},
        "pointcloud_options": {"keyframe_spacing": keyframe_spacing, "remove_dynamic": remove_dynamic,
                               "pointcloud_topic": pointcloud_topic, "imu_topic": imu_topic, "scan_voxel_m": .4, "map_voxel_m": .2},
        "max_attempts": max_attempts, "minimum_retained_fraction": minimum_retained_fraction,
        "pointcloud": None, "attempts": [], "selected": None,
        "scope": "Generated point-cloud map and HD road drafts; source checks are not independent survey truth. Lane identity, traffic rules, equipment and georeferencing are unresolved unless supplied separately.",
    }
    root.mkdir(parents=True, exist_ok=False)
    with _locked(root):
        _save(root / "job.json", job)
        try:
            initial = odometry(str(source_path), str(root / "odometry"),
                               pointcloud_topic=pointcloud_topic, imu_topic=imu_topic)
            _save(root / "odometry-report.json", initial)
            source_motion = {"odometry_report": _artifact(root / "odometry-report.json"),
                             "trajectory": _artifact(initial["trajectory"])}
            corrected = fix_session(initial["scans"], str(root / "pointcloud"),
                                    poses=initial["trajectory"], gravity=initial["gravity"],
                                    keyframe_spacing=keyframe_spacing, remove_dynamic=remove_dynamic)
            _save(root / "pointcloud-report.json", corrected)
            _verify(job["source"])
            for artifact in source_motion.values():
                _verify(artifact)
            if corrected["map_points"] <= 0:
                raise ValueError("corrected point-cloud map is empty")
            files = corrected["outputs"]
            job["pointcloud"] = {
                "files": {"map": _artifact(files["map"]), "trajectory": _artifact(files["kitti"]),
                          "graph": _artifact(files["g2o"])},
                "frames": initial["frames"], "map_points": corrected["map_points"],
                "path_length_m": initial["path_length_m"],
                "reports": {"odometry": str(root / "odometry-report.json"),
                            "correction": str(root / "pointcloud-report.json")},
                "source_motion": source_motion,
                "quality_status": "generated_unverified", "coordinate_frame": "local_slam_metres",
            }
            job["status"] = "pointcloud_ready"
        except BaseException as error:
            job["status"] = "pointcloud_failed"
            job["error"] = str(error)
            job["error_type"] = type(error).__name__
            if not isinstance(error, Exception) and not _native_panic(error):
                _save(root / "job.json", job)
                raise
        _save(root / "job.json", job)
    return inspect_mapping_job(str(root))


def _curb_width_hints(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**band, "review_required": True} for profile in report.get("profiles", []) for band in profile["bands"]
            if band["path_level_supported"] and band["left_evidence"] == "curb_profile" and band["right_evidence"] == "curb_profile"]


def _corridor_summary(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema") != "cloudanalyzer.corridor_proposals.v1":
        raise ValueError("unsupported native corridor report schema")
    return {**{key: report[key] for key in (
        "coordinate_frame", "protocol", "trajectory_length_m", "with_candidate_station_length_m",
        "without_candidate_station_length_m", "trajectory_covered_station_length_m", "curb_bounded_station_length_m",
        "ambiguous_station_length_m", "sampled_sections", "evaluated_sections", "sections_with_bands",
        "multiple_band_sections", "profile_queried_points", "interval_support_samples", "unstable_heading_sections",
        "unanchored_sections", "level_mismatch_bands",
        "limited", "road_semantics_inferred", "deployment_ready", "warnings")},
        "candidate_count": len(report["candidates"]),
        "longest_candidate_station_length_m": max((c["to_m"] - c["from_m"] for c in report["candidates"]), default=0.),
        "off_trajectory_bands": report.get("off_trajectory_bands", 0),
        "curb_bounded_candidates": sum(c["curb_width_range_m"] is not None for c in report["candidates"]),
        "paired_curb_profile_bands": len(_curb_width_hints(report)),
    }


def propose_mapping_corridors(job_dir: str, search_radius_m: float = 8.0) -> dict[str, Any]:
    """Generate lane-free road/path surface proposals from frozen map and trajectory.

    Searches low, spatially supported cross-section bands symmetrically around
    the path. The effective reach rounds outward to 0.5 m bins. No lane count,
    width, direction or speed is required or inferred. Edges distinguish curb-like
    profiles, source gaps, height steps and search limits; support spans are not
    complete road widths without physical evidence. Branches and missing intervals
    stay unresolved. These proposals do not change an HD draft or select traffic
    semantics. Inspect the saved candidates before choosing road assumptions.

    Saves geometry, exact input-station coverage and query limits as a hashed
    report. This one-time stage does not spend the HD attempt budget; identical
    calls verify and reuse it. Different reach/failed stages require a new job.
    """
    if not math.isfinite(search_radius_m) or not 1 <= search_radius_m <= 20:
        raise ValueError("search_radius_m must be finite and within 1..20 metres")
    root = Path(job_dir).resolve()
    with _locked(root):
        job = _load(root)
        _inputs(job)
        options = {"search_radius_m": search_radius_m}
        if "corridor_proposal" in job:
            stage = job["corridor_proposal"]
            if stage["options"] != options:
                raise ValueError("corridor search options are frozen; use a new mapping job")
            if stage["status"] != "ready":
                raise RuntimeError("corridor proposal stage did not finish; inspect its retained error and use a new job")
            _verify(stage["file"])
            summary = _corridor_summary(json.loads(Path(stage["file"]["path"]).read_text(encoding="utf-8")))
            return {**stage, "summary": summary, "cached": True, "remaining_attempts": _remaining(job)}
        stage = {"status": "running", "options": options}
        job["corridor_proposal"] = stage
        _save(root / "job.json", job)
        try:
            module = core()
            assert module is not None
            files = job["pointcloud"]["files"]
            report = json.loads(module.propose_road_corridors(files["map"]["path"], files["trajectory"]["path"], json.dumps(options, allow_nan=False)))
            report["coordinate_frame"] = job["pointcloud"]["coordinate_frame"]
            report["input_artifacts"] = {"source": job["source"], "pointcloud": files, "native": job["runtime"]["native"]}
            summary = _corridor_summary(report)
            _save(root / "corridor-proposals.json", report)
            _inputs(job)
            stage.update({"status": "ready", "file": _artifact(root / "corridor-proposals.json"), "summary": summary})
        except BaseException as error:
            stage.update({"status": "failed", "error": str(error), "error_type": type(error).__name__})
            if not isinstance(error, Exception) and not _native_panic(error):
                _save(root / "job.json", job)
                raise
        _save(root / "job.json", job)
    return {**stage, "cached": False, "remaining_attempts": _remaining(job)}


def _refine_mapping_corridors(job_dir: str, reason: str) -> dict[str, Any]:
    """One explicit path-association experiment; retain every previous artifact."""
    root = Path(job_dir).resolve()
    with _locked(root):
        job = _load(root)
        _inputs(job)
        prior = job.get("corridor_proposal")
        if not prior or prior["status"] != "ready":
            raise ValueError("refinement requires a ready original corridor proposal")
        _verify(prior["file"])
        if "corridor_refinement" in job:
            stage = job["corridor_refinement"]
            _verify(stage["previous"]["file"])
            if stage["reason"] != reason:
                raise ValueError("path-association refinement was already attempted")
            if stage["status"] == "running":
                raise RuntimeError("refinement processing did not finish; inspect its retained state before recovery")
            if stage["status"] == "ready":
                _verify(stage["file"])
            return {**stage, "cached": True}
        if prior["options"].get("association", "all_supported_bands") != "all_supported_bands":
            raise ValueError("refinement requires the original all-supported-bands proposal")
        stage = {"status": "running", "reason": reason, "previous": prior,
            "options": {**prior["options"], "association": "trajectory_containing"}}
        job["corridor_refinement"] = stage
        _save(root / "job.json", job)
        try:
            module = core()
            assert module is not None
            files = job["pointcloud"]["files"]
            report = json.loads(module.propose_road_corridors(files["map"]["path"], files["trajectory"]["path"], json.dumps(stage["options"])))
            if report["protocol"]["options"]["association"] != "trajectory_containing":
                raise ValueError("native core did not apply the requested path association")
            report["coordinate_frame"] = job["pointcloud"]["coordinate_frame"]
            report["input_artifacts"] = {"source": job["source"], "pointcloud": files, "native": job["runtime"]["native"]}
            summary = _corridor_summary(report)
            _save(root / "corridor-proposals-trajectory.json", report)
            _inputs(job)
            _verify(prior["file"])
            stage.update({"status": "ready", "file": _artifact(root / "corridor-proposals-trajectory.json"), "summary": summary})
            job["corridor_proposal"] = {k: stage[k] for k in ("status", "options", "file", "summary")}
        except BaseException as error:
            stage.update({"status": "failed", "error": str(error), "error_type": type(error).__name__})
            if not isinstance(error, Exception) and not _native_panic(error):
                _save(root / "job.json", job)
                raise
        _save(root / "job.json", job)
    return {**stage, "cached": False}


def inspect_mapping_corridors(job_dir: str, candidate_id: int | None = None, offset: int = 0) -> dict[str, Any]:
    """Read frozen lane-free proposals and geometry without processing or HD attempts.

    Verifies input/report hashes, then returns a candidate index (16 per page;
    next_offset selects the next page). A candidate_id returns its original-frame
    cross sections, width evidence and station range. Geometry previews cap at
    128 sections with explicit total/limited fields; the hashed file retains all
    geometry and every deferred/ambiguous interval. No native core is needed for
    inspection. Source spans do not establish road width, lanes or legal use.
    """
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    if candidate_id is not None and (type(candidate_id) is not int or candidate_id < 1 or offset):
        raise ValueError("candidate_id must be positive; omit offset when selecting a candidate")
    job = _load(Path(job_dir).resolve())
    stage = job.get("corridor_proposal")
    if stage is None:
        raise ValueError("run propose_mapping_corridors before inspecting proposals")
    for artifact in [job["source"], *job["pointcloud"]["files"].values()]:
        _verify(artifact)
    if stage["status"] != "ready":
        return {**stage, "remaining_attempts": _remaining(job)}
    _verify(stage["file"])
    report = json.loads(Path(stage["file"]["path"]).read_text(encoding="utf-8"))
    candidates = report["candidates"]
    answer: dict[str, Any] = {"file": stage["file"], "summary": _corridor_summary(report),
        "remaining_attempts": _remaining(job)}
    if candidate_id is None:
        hints = _curb_width_hints(report)
        answer.update({"candidate_index": [{k: c[k] for k in ("id", "from_m", "to_m", "minimum_support_span_m",
            "maximum_support_span_m", "paired_curb_sections", "curb_width_range_m", "review_required")}
            for c in candidates[offset:offset + 16]],
            "next_offset": offset + 16 if offset + 16 < len(candidates) else None,
            "curb_width_profile_hints": hints[:16], "curb_width_profile_hints_limited": len(hints) > 16,
            "deferred_intervals": report["deferred_intervals"][:128], "deferred_intervals_limited": len(report["deferred_intervals"]) > 128,
            "ambiguous_intervals": report["ambiguous_intervals"][:128], "ambiguous_intervals_limited": len(report["ambiguous_intervals"]) > 128})
    else:
        candidate = next((c for c in candidates if c["id"] == candidate_id), None)
        if candidate is None:
            raise ValueError("unknown corridor candidate_id")
        answer["candidate"] = {**candidate, "sections": candidate["sections"][:128],
            "total_sections": len(candidate["sections"]), "section_preview_limited": len(candidate["sections"]) > 128}
    return answer


def generate_mapping_geometry(job_dir: str, decisions: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    """Adopt chosen corridor intervals as editable source geometry without inventing lanes.

    Each decision needs candidate_id, action (include/defer), reason and optional
    from_m/to_m at observed candidate stations. Include at most one band per input
    interval; overlapping alternatives must be resolved explicitly. Adjacent pieces
    are not joined, filled or smoothed. Preserve unreviewed, agent-deferred and
    source-deferred extent across the whole input path, including ambiguous areas.

    Uses one shared HD attempt, with retained native failures and frozen provenance.
    Writes vectormap-ir Other curves for centre/left/right, exact source sections,
    decisions and unresolved semantics. No lane/road count, speed, traffic direction
    or complete width is established. No OSM is published: standalone unknown ways
    are dropped on Lanelet2 reload. The IR can be opened in the editor with virtual
    lines enabled. Existing selected HD drafts remain unchanged. This geometry
    stage cannot be selected as an audited lane map or claimed as a source pass.
    """
    from ca.mapping_geometry import assemble_geometry

    root = Path(job_dir).resolve()
    with _locked(root):
        job = _load(root)
        _inputs(job)
        stage = job.get("corridor_proposal", {})
        if stage.get("status") != "ready":
            raise ValueError("generate ready corridor proposals before drafting geometry")
        _verify(stage["file"])
        proposals = json.loads(Path(stage["file"]["path"]).read_text(encoding="utf-8"))
        _corridor_summary(proposals)
        ir, evidence = assemble_geometry(proposals, decisions, reason)
        ir["metadata"]["attributes"].update({"cloudanalyzer:proposal_sha256": stage["file"]["sha256"],
                                              "cloudanalyzer:pointcloud_sha256": job["pointcloud"]["files"]["map"]["sha256"]})
        summary = evidence["summary"]
        summary["included_station_fraction"] = summary["included_station_length_m"] / summary["trajectory_length_m"]
        summary["minimum_retained_fraction"] = job.get("minimum_retained_fraction", 0.9)
        summary["meets_requested_station_extent"] = summary["included_station_fraction"] + 1e-9 >= summary["minimum_retained_fraction"]
        if _remaining(job) <= 0:
            raise ValueError("mapping attempt budget exhausted; inspect the retained candidates")
        evidence["input_artifacts"] = {"source": job["source"], "pointcloud": job["pointcloud"]["files"],
                                       "native": job["runtime"]["native"], "corridor_proposal": stage["file"]}
        attempt: dict[str, Any] = {"id": len(job["attempts"]) + 1, "kind": "surface_geometry", "status": "running",
                                  "reason": reason.strip(), "decisions": evidence["decisions"], "corridor_proposal": stage["file"]}
        job["attempts"].append(attempt)
        _save(root / "job.json", job)
        target = root / f"geometry-{attempt['id']:02d}"
        try:
            if target.exists():
                raise FileExistsError(f"output directory already exists: {target}")
            with tempfile.TemporaryDirectory(prefix=".mapping-geometry-", dir=root) as temporary:
                draft = Path(temporary)
                _save(draft / "vector_map.json", ir)
                module = core()
                assert module is not None
                # Existing read-only native operation validates/normalizes IR. Its
                # OSM is intentionally not used: unknown standalone ways are lost.
                payload = json.loads(module.edit_vector_map_relations(str(draft / "vector_map.json")))
                native = payload["report"]
                if native["validation"]["counts"]["errors"] or any(i["severity"] == "error" for i in native["import_issues"]):
                    raise ValueError("resolve geometry import/structural errors before publishing")
                normalized = json.loads(payload["map_json"])
                if normalized.get("lanes") or normalized.get("roads") or normalized.get("boundaries") != ir["boundaries"]:
                    raise ValueError("native IR reload did not preserve the source reference curves")
                _save(draft / "vector_map.json", normalized)
                evidence["native_validation"] = {k: native[k] for k in ("validation", "import_issues", "georeference")}
                evidence["editing"] = {"file": str(target / "vector_map.json"), "show_virtual_lines": True}
                _save(draft / "report.json", evidence)
                _inputs(job)
                _verify(stage["file"])
                os.rename(draft, target)
            attempt.update({"status": "geometry_draft", "summary": evidence["summary"],
                            "files": {"editable_map": _artifact(target / "vector_map.json"), "report": _artifact(target / "report.json")}})
            job["status"] = "geometry_drafts_ready" if job["selected"] is None else "selected_draft"
        except BaseException as error:
            attempt.update({"status": "failed", "error": str(error), "error_type": type(error).__name__})
            if not isinstance(error, Exception) and not _native_panic(error):
                _save(root / "job.json", job)
                raise
        _save(root / "job.json", job)
    return inspect_mapping_job(str(root))


def inspect_mapping_geometry(job_dir: str, candidate_id: int, offset: int = 0) -> dict[str, Any]:
    """Read a saved geometry draft and full-extent decisions without native processing.

    Verifies source, map inputs, proposal and output hashes. Pages 16 segments via
    next_offset; section previews cap at 128 and station dispositions at 128 with
    explicit limits. The hashed report contains all curves, decisions and intervals.
    Empty lane semantics remain unresolved, never a quality or deployment pass.
    """
    if type(candidate_id) is not int or candidate_id < 1 or type(offset) is not int or offset < 0:
        raise ValueError("candidate_id must be positive and offset nonnegative integers")
    job = _load(Path(job_dir).resolve())
    attempt = next((a for a in job["attempts"] if a["id"] == candidate_id and a.get("kind") == "surface_geometry"), None)
    if attempt is None:
        raise ValueError("inspect a retained surface geometry attempt")
    for artifact in [job["source"], *job["pointcloud"]["files"].values(), attempt["corridor_proposal"]]:
        _verify(artifact)
    remaining = _remaining(job)
    if attempt["status"] != "geometry_draft":
        return {**attempt, "remaining_attempts": remaining}
    for artifact in attempt["files"].values():
        _verify(artifact)
    report = json.loads(Path(attempt["files"]["report"]["path"]).read_text(encoding="utf-8"))
    segments = report["segments"]
    return {"candidate_id": candidate_id, "files": attempt["files"], "summary": report["summary"],
        "decisions": report["decisions"], "native_validation": report["native_validation"], "editing": report["editing"],
        "segments": [{**s, "sections": s["sections"][:128], "total_sections": len(s["sections"]),
                      "section_preview_limited": len(s["sections"]) > 128} for s in segments[offset:offset + 16]],
        "next_offset": offset + 16 if offset + 16 < len(segments) else None,
        "station_disposition": report["station_disposition"][:128], "total_station_intervals": len(report["station_disposition"]),
        "station_disposition_limited": len(report["station_disposition"]) > 128,
        "source_proposal_limited": report["source_proposal_limited"], "warnings": report["warnings"],
        "remaining_attempts": remaining}


def generate_mapping_corridor_lanes(
    job_dir: str, geometry_candidate_id: int, lane_specs: list[dict[str, Any]], boundary_policy: str, reason: str,
) -> dict[str, Any]:
    """Export and audit explicit lane hypotheses inside retained source geometry.

    Choose boundary_policy="source_span_hypothesis" explicitly: observation gaps
    are not complete road edges. Each specification needs a retained center_curve_id,
    reason, speed_limit_kmh, and lanes ordered left
    to right along the original input stations. Each lane requires direction
    (forward/backward), kind="driving", one_way, positive fraction and minimum_width_m.
    Other lane kinds are excluded because the source-audit protocol evaluates driving lanes only.
    Fractions sum to 1. Virtual dividers interpolate the source edges; no painted
    marking or traffic rule is detected. Too-narrow source spans retain a failed
    attempt instead of widening, narrowing or changing lane count automatically.

    Outer source curves are preserved; separate pieces stay disconnected. Unassigned
    geometry, all unresolved input intervals and semantic hypotheses remain visible.
    Uses one shared HD attempt, pins parent/proposal/input/native hashes and preserves
    existing selection. Saves IR, OSM, projector and four source audits after checking
    lane membership/orientation/semantics/geometry on OSM reload. Native processing is
    required; old jobs remain inspectable but cannot switch their pinned binary.
    Use diagnose_mapping_candidate afterwards. Source support does not confirm width,
    lane identity, legal direction/speed, clearance or deployment readiness.
    """
    from ca.mapping_geometry import corridor_lane_requests, verify_lane_roundtrip

    if type(geometry_candidate_id) is not int or geometry_candidate_id < 1 or not isinstance(reason, str) or not reason.strip():
        raise ValueError("supply a positive geometry_candidate_id and a reason")
    root = Path(job_dir).resolve()
    with _locked(root):
        job = _load(root)
        _inputs(job)
        parent = next((a for a in job["attempts"] if a["id"] == geometry_candidate_id and a["status"] == "geometry_draft"), None)
        if parent is None:
            raise ValueError("use a retained geometry draft in the same mapping job")
        for artifact in [*parent["files"].values(), parent["corridor_proposal"]]:
            _verify(artifact)
        geometry = json.loads(Path(parent["files"]["report"]["path"]).read_text(encoding="utf-8"))
        requests, chosen = corridor_lane_requests(geometry, lane_specs, boundary_policy)
        module = core()
        if module is None or not hasattr(module, "build_corridor_lanes"):
            raise RuntimeError("corridor lane drafting needs an updated native core and a new mapping job")
        if _remaining(job) <= 0:
            raise ValueError("mapping attempt budget exhausted; inspect the retained candidates")
        options = {"geometry_candidate_id": geometry_candidate_id, "boundary_policy": boundary_policy, "lane_specs": lane_specs}
        attempt: dict[str, Any] = {"id": len(job["attempts"]) + 1, "kind": "corridor_lanes", "status": "running",
            "reason": reason.strip(), "road_options": options, "geometry_inputs": parent["files"], "corridor_proposal": parent["corridor_proposal"]}
        job["attempts"].append(attempt)
        _save(root / "job.json", job)
        target = root / f"candidate-{attempt['id']:02d}"
        try:
            if target.exists():
                raise FileExistsError(f"output directory already exists: {target}")
            payload = json.loads(module.build_corridor_lanes(parent["files"]["editable_map"]["path"], json.dumps(requests, allow_nan=False)))
            evidence = payload["report"]
            evidence["options"] = options
            evidence["reason"] = reason.strip()
            evidence["geometry_inputs"] = parent["files"]
            evidence["input_artifacts"] = geometry["input_artifacts"]
            assigned = {s["curve_ids"]["center"] for s in chosen}
            evidence["station_disposition"] = [{**i, "status": "included_lane_hypothesis" if any(
                s["from_m"] <= i["from_m"] and i["to_m"] <= s["to_m"] for s in chosen)
                else "geometry_only" if i["status"] == "included_geometry" else i["status"]} for i in geometry["station_disposition"]]
            evidence["unassigned_geometry"] = [s for s in geometry["segments"] if s["curve_ids"]["center"] not in assigned]
            extraction = {"roads": len(chosen), "lanes": sum(len(r["lanes"]) for r in requests),
                "trajectory_length": geometry["summary"]["trajectory_length_m"], "generated_length": sum(s["to_m"] - s["from_m"] for s in chosen),
                "length_measurement": "original_input_xy_station_union", "width_prior_vertices": 0,
                "layout_prior_vertices": sum((len(r["lanes"]) - 1) * len(s["sections"]) for r, s in zip(requests, chosen))}
            evidence["extraction"] = extraction
            evidence["extent"] = _extent(extraction, job.get("minimum_retained_fraction", .9))
            with tempfile.TemporaryDirectory(prefix=".mapping-lanes-", dir=root) as temporary:
                draft = Path(temporary) / "result"
                published = _publish(payload, draft)
                files = published["files"]
                reopened = json.loads(module.edit_vector_map_relations(files["map"]))
                verify_lane_roundtrip(json.loads(payload["map_json"]), json.loads(reopened["map_json"]))
                evidence["lane_roundtrip_verified"] = True
                cloud = job["pointcloud"]["files"]["map"]["path"]
                audits = {"editable": json.loads(module.audit_vector_map_quality_details(cloud, files["editable_map"])),
                          "reopened_osm": json.loads(module.audit_vector_map_quality_details(cloud, files["map"]))}
                audits["ground_consensus"] = {key: json.loads(module.audit_vector_map_ground_consensus_details(cloud, files[fkey]))
                                             for key, fkey in (("editable", "editable_map"), ("reopened_osm", "map"))}
                _save(draft / "source-quality.json", audits)
                evidence["files"] = {k: str(target / Path(v).name) for k, v in files.items()}
                evidence["editing"] = {"command": "vectormap", "args": ["mcp", evidence["files"]["map"]]}
                _save(draft / "report.json", evidence)
                result = {"files": {k: {**_artifact(files[k]), "path": v} for k, v in evidence["files"].items()},
                    "quality_report": {**_artifact(draft / "source-quality.json"), "path": str(target / "source-quality.json")},
                    "quality": _quality_summary(audits["editable"]), "reopened_quality": _quality_summary(audits["reopened_osm"]),
                    "ground_consensus_quality": {k: _quality_summary(v) for k, v in audits["ground_consensus"].items()},
                    "extraction": extraction, "extent": evidence["extent"], "export_issues": evidence["autoware_issues"], "status": "audited_draft"}
                _inputs(job)
                for artifact in [*parent["files"].values(), parent["corridor_proposal"]]:
                    _verify(artifact)
                os.rename(draft, target)
            attempt.update(result)
            job["status"] = "candidates_ready" if job["selected"] is None else "selected_draft"
        except BaseException as error:
            attempt.update({"status": "failed", "error": str(error), "error_type": type(error).__name__})
            if not isinstance(error, Exception) and not _native_panic(error):
                _save(root / "job.json", job)
                raise
        _save(root / "job.json", job)
    return inspect_mapping_job(str(root))


def _quality_summary(audit: dict[str, Any]) -> dict[str, Any]:
    q = audit["quality"]
    return {"lanes_checked": len(q["lanes"]), "needs_review": q["low_support_lanes"],
            "ground_estimator": q.get("ground_estimator", {"model": "low_quantile", "quantile": 0.15, "minimum_returns": 3}),
            "omitted": q["omitted_lanes"], "malformed": q["malformed_lanes"],
            "limited": q["limited"], "sampled_points": q["sampled_points"],
            "supported_samples": sum(l[c]["supported"] for l in q["lanes"] for c in ("center", "left", "right")),
            "validation_errors": [i for i in [*audit["validation"]["issues"], *audit.get("import_issues", [])]
                                  if i["severity"] == "error"]}


def _extent(extraction: dict[str, Any], minimum: float) -> dict[str, Any]:
    total, generated = extraction["trajectory_length"], extraction["generated_length"]
    if not math.isfinite(total) or not math.isfinite(generated) or total <= 0 or generated < 0:
        raise ValueError("generation report needs finite, nonnegative extent and positive trajectory length")
    fraction = generated / total
    return {"trajectory_length_m": total, "generated_length_m": generated,
            "retained_fraction": fraction, "minimum_retained_fraction": minimum,
            "passes_requested_extent": fraction >= minimum}


def _diagnose_audit(audit: dict[str, Any]) -> dict[str, Any]:
    quality = audit["quality"]
    lanes = []
    totals = {key: 0 for key in ("samples", "supported", "height_mismatches", "insufficient_returns")}
    for lane in quality["lanes"]:
        traces = {}
        for name in ("center", "left", "right"):
            trace = lane[name]
            for key in totals:
                totals[key] += trace[key]
            holds = []
            if trace["fraction"] < quality["minimum_support_fraction"]:
                holds.append("below_support_threshold")
            if not trace["start_supported"]:
                holds.append("unsupported_start")
            if not trace["end_supported"]:
                holds.append("unsupported_end")
            traces[name] = {**trace, "holds": holds}
        lanes.append({"lane": lane["lane"], "needs_review": lane["needs_review"], "traces": traces})
    errors = [i for i in [*audit["validation"]["issues"], *audit.get("import_issues", [])]
              if i["severity"] == "error"]
    return {
        "protocol": {**{key: quality[key] for key in ("sampling_step_m", "ground_radius_m",
                     "ground_height_tolerance_m", "minimum_support_fraction", "sample_budget")},
                     "ground_estimator": quality.get("ground_estimator", {"model": "low_quantile", "quantile": 0.15, "minimum_returns": 3})},
        "complete": bool(lanes) and not (quality["limited"] or quality["omitted_lanes"]
                                         or quality["malformed_lanes"] or errors),
        "sample_totals": totals, "needs_review": quality["low_support_lanes"], "lanes": lanes,
        "omitted": quality["omitted_lanes"], "malformed": quality["malformed_lanes"],
        "limited": quality["limited"], "errors": errors, "warnings": quality["warnings"],
        "problems": quality.get("problems", []),
        "problems_available": "problems" in quality,
        "problems_limited": quality.get("problems_limited"),
    }


def diagnose_mapping_candidate(job_dir: str, candidate_id: int) -> dict[str, Any]:
    """Explain a candidate's source holds from its saved native IR and OSM audits.

    Verifies recorded source, point-map and candidate hashes, then reads small
    reports without rerunning native processing or spending attempts. Returns
    per-lane/trace height mismatches, insufficient returns, endpoint holds and
    retained extent. New jobs retain legacy quantile and spatial-layer evidence,
    bounded problem locations and local source heights. Missing or limited
    location previews are explicit. These are
    observed audit failures, not proven root causes:
    wrong XY, another level, sparse source and unverified lane priors can overlap.
    Use the evidence to choose a trial; do not erase lanes or shrink the map to pass.
    """
    root = Path(job_dir).resolve()
    job = _load(root)
    if type(candidate_id) is not int or candidate_id < 1:
        raise ValueError("candidate_id must be a positive integer")
    attempt = next((a for a in job["attempts"] if a["id"] == candidate_id), None)
    if attempt is None or attempt["status"] != "audited_draft":
        raise ValueError("diagnose an audited draft candidate")
    for artifact in [job["source"], *job["pointcloud"]["files"].values(), *attempt.get("geometry_inputs", {}).values(),
                     *([attempt["corridor_proposal"]] if "corridor_proposal" in attempt else []),
                     *attempt.get("connection_inputs", {}).values(),
                     *([attempt["connection_proposal"]] if "connection_proposal" in attempt else []),
                     *job.get("retry_inputs", {}).values(),
                     *attempt["files"].values(), attempt["quality_report"]]:
        _verify(artifact)
    saved = json.loads(Path(attempt["quality_report"]["path"]).read_text(encoding="utf-8"))
    editable = _diagnose_audit(saved["editable"])
    reopened = _diagnose_audit(saved["reopened_osm"])
    consensus = ({key: _diagnose_audit(saved["ground_consensus"][key]) for key in ("editable", "reopened_osm")}
                 if "ground_consensus" in saved else None)
    report = json.loads(Path(attempt["files"]["report"]["path"]).read_text(encoding="utf-8"))
    extent = _extent(attempt["extraction"], job.get("minimum_retained_fraction", 0.9))
    investigations = []
    totals = editable["sample_totals"]
    if totals["height_mismatches"]:
        investigations.append("Compare trace Z with nearby low returns and inspect XY/ground-level alignment; a height mismatch does not establish that a Z-only correction is appropriate.")
    if totals["insufficient_returns"]:
        investigations.append("Inspect the point footprint and trajectory/lane assumptions for the affected traces; sparse or occluded returns do not prove that a road is absent.")
    if attempt["extraction"].get("width_prior_vertices", 0):
        investigations.append("Inspect assumed-width boundaries and their anchors. Point-coverage edges may be scan gaps rather than physical road edges; compare fitting choices at unchanged lane count, width and extent.")
    if attempt.get("kind") in {"corridor_lanes", "connected_corridor_lanes"}:
        investigations.append("The observed support span was explicitly adopted as an unverified layout hypothesis. Inspect the parent edge evidence and assigned lane fractions/directions/minimum widths; source support does not confirm complete road width or legal traffic rules.")
    if attempt.get("kind") == "connected_corridor_lanes":
        investigations.append("Connector turn_direction tags classify geometric headings, not permitted manoeuvres. Review legal routing, full-width interior and clearance; graph station spans do not increase original source-corridor extent.")
    if not editable["complete"] or not reopened["complete"] or (consensus is not None and any(not a["complete"] for a in consensus.values())):
        investigations.append("Resolve incomplete or invalid audits before interpreting support or selecting a draft.")
    if editable != reopened:
        investigations.append("Inspect differences between editable and reopened OSM evidence before comparing candidates.")
    if not extent["passes_requested_extent"]:
        investigations.append("Inspect deferred road length; a supported fragment does not meet the requested extent.")
    if consensus is not None:
        if consensus["editable"]["lanes"] != editable["lanes"]:
            investigations.append("Ground estimators disagree. Inspect layered/density-dominated source columns and actual surface levels; do not select the estimator with the better score as proof of road accuracy.")
        if consensus["editable"] != consensus["reopened_osm"]:
            investigations.append("Inspect differences between editable and reopened OSM ground-consensus evidence.")
    return {
        "schema": "cloudanalyzer.mapping_diagnosis.v1", "candidate_id": candidate_id,
        "quality_report": attempt["quality_report"], "road_options": attempt["road_options"],
        "effective_options": report["options"], "extent": extent,
        "editable": editable, "reopened_osm": reopened,
        "editable_and_reopened_match": editable == reopened,
        "ground_consensus": consensus,
        "extraction": attempt["extraction"], "export_issues": attempt["export_issues"],
        "pointcloud_quality_status": job["pointcloud"]["quality_status"],
        "remaining_attempts": _remaining(job),
        "routes": attempt.get("routes"),
        "investigations": investigations, "deployment_ready": False,
        "counting_note": "Totals count oriented samples per lane/trace; a shared boundary can be checked for both lanes. They are not unique points or percentages of road length.",
    }


def generate_mapping_candidate(job_dir: str, road_options: dict[str, Any], reason: str) -> dict[str, Any]:
    """Generate and audit one HD road hypothesis against the job's frozen point cloud.

    Supply a decision reason and explicit forward_lanes, backward_lanes,
    left_hand_traffic, lane_width and speed_limit. Other build_vector_map fitting
    options are accepted; changing input maps or coordinate metadata is excluded.
    Each bounded attempt keeps its settings, native failures, output hashes and
    both source estimators for IR/OSM. Failed trials do not replace earlier drafts or their selection.
    """
    required = {"forward_lanes", "backward_lanes", "left_hand_traffic", "lane_width", "speed_limit"}
    excluded = {"cloud", "trajectory", "out_dir", "reference_map", "existing_map", "projection", "origin_lat", "origin_lon"}
    allowed = set(inspect.signature(build_vector_map).parameters) - excluded
    if not reason.strip() or not required <= road_options.keys() or road_options.keys() - allowed:
        raise ValueError("supply a reason, explicit lane/traffic/width/speed assumptions, and supported fitting options")
    json.dumps(road_options, allow_nan=False)
    root = Path(job_dir).resolve()
    with _locked(root):
        job = _load(root)
        if job["pointcloud"] is None:
            raise ValueError("generate the point-cloud map before an HD candidate")
        if _remaining(job) <= 0:
            raise ValueError("mapping attempt budget exhausted; inspect the retained candidates")
        _inputs(job)
        job.setdefault("minimum_retained_fraction", 0.9)
        attempt: dict[str, Any] = {"id": len(job["attempts"]) + 1, "status": "running",
                                   "reason": reason.strip(), "road_options": road_options}
        job["attempts"].append(attempt)
        _save(root / "job.json", job)
        target = root / f"candidate-{attempt['id']:02d}"
        try:
            files = job["pointcloud"]["files"]
            report = build_vector_map(files["map"]["path"], files["trajectory"]["path"], str(target), **road_options)
            attempt["files"] = {key: _artifact(path) for key, path in report["files"].items()}
            module = core()
            assert module is not None
            audit = json.loads(module.audit_vector_map_quality_details(files["map"]["path"], report["files"]["editable_map"]))
            reopened = json.loads(module.audit_vector_map_quality_details(files["map"]["path"], report["files"]["map"]))
            consensus = {key: json.loads(module.audit_vector_map_ground_consensus_details(files["map"]["path"], report["files"][file_key]))
                         for key, file_key in (("editable", "editable_map"), ("reopened_osm", "map"))}
            _save(root / f"candidate-{attempt['id']:02d}-quality.json", {"editable": audit, "reopened_osm": reopened, "ground_consensus": consensus})
            _inputs(job)
            attempt["quality_report"] = _artifact(root / f"candidate-{attempt['id']:02d}-quality.json")
            attempt["quality"] = _quality_summary(audit)
            attempt["reopened_quality"] = _quality_summary(reopened)
            attempt["ground_consensus_quality"] = {key: _quality_summary(value) for key, value in consensus.items()}
            attempt["extraction"] = report["extraction"]
            attempt["extent"] = _extent(report["extraction"], job["minimum_retained_fraction"])
            attempt["export_issues"] = report["autoware_issues"]
            attempt["status"] = "audited_draft"
            job["status"] = "candidates_ready" if job["selected"] is None else "selected_draft"
        except BaseException as error:
            attempt["status"] = "failed"
            attempt["error"] = str(error)
            attempt["error_type"] = type(error).__name__
            if not isinstance(error, Exception) and not _native_panic(error):
                _save(root / "job.json", job)
                raise
        _save(root / "job.json", job)
    return inspect_mapping_job(str(root))


def select_mapping_candidate(job_dir: str, candidate_id: int, reason: str) -> dict[str, Any]:
    """Select an audited HD draft explicitly and record the agent's justification.

    Requires the job's retained-extent goal, nonempty lanes, complete audits and zero structural/import/export errors,
    and unchanged source/artifacts. Low source support stays visible and does not
    become a pass. Selecting is a local draft decision, not deployment certification.
    Read all candidates' road assumptions and retained extent before choosing.
    """
    if not reason.strip():
        raise ValueError("a selection needs a reason")
    root = Path(job_dir).resolve()
    with _locked(root):
        job = _load(root)
        _inputs(job)
        attempt = next((a for a in job["attempts"] if a["id"] == candidate_id), None)
        if attempt is None or attempt["status"] != "audited_draft":
            raise ValueError("select an audited draft candidate")
        for artifact in [*attempt["files"].values(), attempt["quality_report"], *attempt.get("geometry_inputs", {}).values(),
                         *attempt.get("connection_inputs", {}).values(),
                         *([attempt["connection_proposal"]] if "connection_proposal" in attempt else []),
                         *([attempt["corridor_proposal"]] if "corridor_proposal" in attempt else [])]:
            _verify(artifact)
        extent = _extent(attempt["extraction"], job.get("minimum_retained_fraction", 0.9))
        if not extent["passes_requested_extent"]:
            raise ValueError("candidate does not meet the job's retained-extent goal; inspect deferred road length")
        # Interpret the verified saved audits with the current contract, rather
        # than trusting cached summaries from an earlier job implementation.
        saved = json.loads(Path(attempt["quality_report"]["path"]).read_text(encoding="utf-8"))
        qualities = [_quality_summary(saved[key]) for key in ("editable", "reopened_osm")]
        if "ground_consensus" in saved:
            qualities.extend(_quality_summary(saved["ground_consensus"][key]) for key in ("editable", "reopened_osm"))
        for q in qualities:
            if not q["lanes_checked"] or q["omitted"] or q["malformed"] or q["limited"] or q["validation_errors"]:
                raise ValueError("candidate needs complete source audits and nonempty, structurally valid roads")
        if any(i["severity"] == "error" for i in attempt["export_issues"]):
            raise ValueError("resolve export errors before selecting")
        job["selected"] = {"candidate_id": candidate_id, "reason": reason.strip(),
                           "source_quality_passed": all(not q["needs_review"] for q in qualities),
                           "deployment_ready": False, "files": attempt["files"],
                           "assumptions": attempt["road_options"], "extent": extent}
        job["status"] = "selected_draft"
        _save(root / "job.json", job)
    return inspect_mapping_job(str(root))
