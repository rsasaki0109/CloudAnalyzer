"""Fresh HD generation and reversible selection of immutable motion/map pairs."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from ca import mapping_job as jobs, mapping_run as runs, mapping_retry as retries
from ca.mapping_motion_trial import SCHEMA as TRIAL_SCHEMA
from ca.mapping_revision import _artifacts
from ca.mapping_trajectory_review import _artifact, _graph_ids, _read, compare_mapping_motion_trials
from ca.posegraph_fix import read_trajectory

SCHEMA = "cloudanalyzer.mapping_motion_maps.v1"
SELECTION_SCHEMA = "cloudanalyzer.mapping_motion_selection.v1"


def _reason(reason: str) -> str:
    if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 4096:
        raise ValueError("supply a bounded nonempty decision reason")
    return reason.strip()


def _target(out_dir: str) -> Path:
    root = Path(out_dir).resolve()
    if root.exists():
        raise FileExistsError(f"output already exists; inspect retained state: {root}")
    if any((p / "job.json").exists() for p in (root.parent, *root.parent.parents)):
        raise ValueError("save outputs outside existing mapping jobs")
    return root


def _finished(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    run = runs._load(root)
    if run["status"] != "finished" or not run.get("output") or run["output"]["candidate_id"] is None:
        raise ValueError("use a finished run delivering an audited point/HD pair")
    runs.inspect_mapping_run(str(root))
    job = jobs._load(root)
    jobs._inputs(job)
    if Path(run["output"].get("candidate_job_dir", str(root))).resolve() != root:
        raise ValueError("use the finished owner run of the delivered pair")
    candidate = retries._parent(job, run["output"]["candidate_id"])
    for key, value in job["pointcloud"]["files"].items():
        if run["output"]["artifacts"].get(key) != value:
            raise ValueError("delivered point map differs from its owner")
    for key, value in candidate["files"].items():
        if run["output"]["artifacts"].get(f"hd_{key}") != value:
            raise ValueError("delivered HD map differs from its owner")
    if run["output"]["artifacts"].get("hd_source_audits") != candidate["quality_report"]:
        raise ValueError("delivered pair is missing its exact HD audits")
    diagnosis = jobs.diagnose_mapping_candidate(str(root), candidate["id"])
    audits = [diagnosis["editable"], diagnosis["reopened_osm"], *(diagnosis["ground_consensus"] or {}).values()]
    if len(audits) != 4 or any(not a["complete"] or a["errors"] for a in audits) or any(i["severity"] == "error" for i in diagnosis["export_issues"]):
        raise ValueError("pair comparison needs four complete audits without structural/export errors")
    return run, job, candidate, diagnosis


def start_mapping_motion_run(trial_job_dir: str, finished_job_dir: str, out_dir: str, max_attempts: int, reason: str) -> dict[str, Any]:
    """Prepare fresh HD proposals from an immutable motion trial, keeping the old pair.

    Use the finished owner run used as the trial baseline and a NEW external output
    directory and an explicit NEW 2..8 shared HD attempt budget. Inherit its fixed
    lane hypothesis and source-extraction policy. Earlier budgets stay fixed. No odometry,
    fusion, HD geometry, connections or audits are reused from the old pair. The
    trial and its saved trajectory evaluation remain unchanged. Inspect/draft/finish
    with advance_mapping_run; compare_mapping_motion_maps then compares exact
    finished pairs on common original-motion stations. Startup failures are retained;
    existing outputs are never silently rerun. Keep all referenced directories.
    """
    reason = _reason(reason)
    if type(max_attempts) is not int or not 2 <= max_attempts <= 8:
        raise ValueError("supply an explicit new 2..8 shared HD attempt budget")
    target, source, baseline = _target(out_dir), Path(trial_job_dir).resolve(), Path(finished_job_dir).resolve()
    baseline_run, parent, _, _ = _finished(baseline)
    trial_job_artifact = jobs._artifact(source / "job.json")
    trial_job = jobs._load(source)
    jobs._inputs(trial_job)
    metadata = trial_job.get("motion_trial")
    if not metadata:
        raise ValueError("start from a retained motion trial")
    _artifact(metadata["report"])
    trial = json.loads(Path(metadata["report"]["path"]).read_text())
    if (trial.get("schema") != TRIAL_SCHEMA or trial["status"] != "ready_unverified"
            or trial_job["status"] != "pointcloud_ready" or trial_job["attempts"] or trial_job["selected"] is not None
            or trial["inputs"]["baseline_job"] != jobs._artifact(baseline / "job.json")
            or trial["outputs"] != trial_job["pointcloud"]["files"]):
        raise ValueError("use an untouched ready motion trial of this exact finished baseline")
    for key in ("source", "runtime", "pointcloud_options", "minimum_retained_fraction"):
        if trial_job[key] != parent[key]:
            raise ValueError("motion trial changed frozen source, runtime, policy or extent goal")
    if trial_job["pointcloud"]["source_motion"] != parent["pointcloud"]["source_motion"]:
        raise ValueError("motion trial changed original motion")
    inputs: dict[str, Any] = {}
    for value in (baseline_run, parent, trial, trial_job, trial_job_artifact):
        _artifacts(value, inputs)
    for path in (baseline / "job.json", baseline / "run.json"):
        _artifacts(jobs._artifact(path), inputs)
    retries._verify_inputs(inputs)
    proposal = json.loads(Path(parent["corridor_proposal"]["file"]["path"]).read_text())
    policy = proposal["protocol"]["options"]
    if set(policy) - {"search_radius_m", "association"} or policy.get("association", "all_supported_bands") not in {"all_supported_bands", "trajectory_containing"}:
        raise ValueError("unsupported frozen source-extraction policy")
    layout = json.loads(Path(baseline_run["layout_file"]["path"]).read_text())
    runs._layout(layout)
    target.mkdir(parents=True)
    manifest = {"schema": SCHEMA, "baseline_job_dir": str(baseline), "trial_job_dir": str(source),
                "trial_job": trial_job_artifact, "inputs": inputs, "reason": reason,
                "source_policy": policy, "max_attempts": max_attempts, "old_hd_reused": False, "independent_accuracy_established": False}
    jobs._save(target / "motion-run.json", manifest)
    child = deepcopy(trial_job)
    child.update(job_dir=str(target), max_attempts=max_attempts, motion_run={"file": jobs._artifact(target / "motion-run.json"),
                                              "baseline_job_dir": str(baseline), "trial_job": trial_job_artifact})
    child["retry_inputs"].update(inputs)
    child["retry_inputs"]["motion_run_manifest"] = child["motion_run"]["file"]
    jobs._save(target / "job.json", child)
    jobs._save(target / "layout-hypothesis.json", layout)
    run = {"schema": runs.SCHEMA, "revision": 0, "status": "preparing", "layout_file": jobs._artifact(target / "layout-hypothesis.json"),
           "reviewed_candidates": [], "history": [], "output": None, "maximum_actions": 128}
    jobs._save(target / "run.json", run)
    try:
        stage = jobs.propose_mapping_corridors(str(target), policy["search_radius_m"])
        if stage["status"] == "ready" and policy.get("association") == "trajectory_containing":
            stage = jobs._refine_mapping_corridors(str(target), reason)
        run["status"] = "needs_agent" if stage["status"] == "ready" else "processing_failed"
        if stage["status"] != "ready":
            run["error"] = stage.get("error")
        retries._verify_inputs(inputs)
    except BaseException as error:
        run.update(status="processing_failed", error=str(error), error_type=type(error).__name__)
        raise
    finally:
        jobs._save(target / "run.json", run)
    return runs.inspect_mapping_run(str(target))


def _stations(poses: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(poses[:, :2, 3], axis=0), axis=1)
    if not np.isfinite(steps).all() or np.any(steps <= 1e-9):
        raise ValueError("station correspondence requires nonzero finite XY travel between retained poses")
    return cast(np.ndarray, np.r_[0., np.cumsum(steps)])


def _intervals(report: dict[str, Any], stations: np.ndarray, common: np.ndarray, stamps: np.ndarray, ids: list[int]) -> list[dict[str, Any]]:
    if not np.isclose(report["extraction"]["trajectory_length"], stations[-1], atol=1e-8, rtol=1e-10):
        raise ValueError("HD source stations do not match the delivered motion")
    result = []
    last = 0.
    rows = report["station_disposition"]
    if not isinstance(rows, list) or len(rows) > 16384:
        raise ValueError("bounded HD station disposition is required")
    for row in rows:
        lo, hi = row["from_m"], row["to_m"]
        if (type(lo) not in (int, float) or type(hi) not in (int, float) or not np.isfinite([lo, hi]).all()
                or abs(lo - last) > 1e-8 or hi <= lo or hi > stations[-1] + 1e-8):
            raise ValueError("HD station disposition must partition its whole delivered motion")
        last = hi
        if row["status"] == "included_lane_hypothesis":
            result.append({"from_m": float(np.interp(lo, stations, common)), "to_m": float(np.interp(hi, stations, common)),
                "map_from_m": lo, "map_to_m": hi,
                "from_timestamp_s": float(np.interp(lo, stations, stamps)), "to_timestamp_s": float(np.interp(hi, stations, stamps)),
                "from_original_frame_coordinate": float(np.interp(lo, stations, ids)),
                "to_original_frame_coordinate": float(np.interp(hi, stations, ids))})
    if abs(last - stations[-1]) > 1e-8:
        raise ValueError("HD disposition omits source extent")
    return result


def _audit_summary(diagnosis: dict[str, Any]) -> dict[str, Any]:
    audits = {"editable": diagnosis["editable"], "reopened_osm": diagnosis["reopened_osm"],
              **{f"consensus_{k}": v for k, v in diagnosis["ground_consensus"].items()}}
    return {key: {k: row[k] for k in ("complete", "sample_totals", "needs_review", "protocol")} for key, row in audits.items()}


def compare_mapping_motion_maps(
    candidate_job_dir: str, baseline_report_file: dict[str, Any], candidate_report_file: dict[str, Any], out_dir: str, reason: str,
) -> dict[str, Any]:
    """Compare finished fresh HD and motion pairs; initialize reversible selection at baseline.

    Use a finished owner run from start_mapping_motion_run and the exact saved
    trajectory comparisons of its untouched baseline and motion trial. Same source,
    reference, retained frame IDs, fitted/evaluated coverage and quality protocols
    are mandatory. Source intervals are mapped piecewise through those original
    frame IDs onto one original-odometry XY station axis; moved poses' metre values
    are not compared directly. This temporal association is not physical road
    correspondence. All four audits, raw extents, route summaries and local motion
    regressions remain visible. Candidate HD geometry/connections are freshly
    generated; old IDs and edges are not carried over. A NEW external directory
    saves immutable comparison and selection.json at revision 0 with baseline
    chosen. No job, map or previous selection changes. Select explicitly afterward.
    """
    reason, target = _reason(reason), _target(out_dir)
    root = Path(candidate_job_dir).resolve()
    candidate_run, candidate_job, candidate, candidate_diagnosis = _finished(root)
    lineage = candidate_job.get("motion_run")
    if not lineage:
        raise ValueError("compare fresh pairs from start_mapping_motion_run")
    baseline = Path(lineage["baseline_job_dir"])
    baseline_run, baseline_job, parent, baseline_diagnosis = _finished(baseline)
    first = _read(baseline_report_file)[0]
    second = _read(candidate_report_file)[0]
    if (first["inputs"]["job"] != jobs._artifact(baseline / "job.json")
            or second["inputs"]["job"] != lineage["trial_job"]):
        raise ValueError("trajectory reports must describe the exact frozen baseline and motion trial")
    for job, report in ((baseline_job, first), (candidate_job, second)):
        if any(job["pointcloud"]["files"][key] != report["inputs"][f"pointcloud_{key}"] for key in ("map", "graph", "trajectory")):
            raise ValueError("trajectory report is not bound to the delivered point map")
    if baseline_run["layout_file"]["sha256"] != candidate_run["layout_file"]["sha256"] or baseline_job["minimum_retained_fraction"] != candidate_job["minimum_retained_fraction"]:
        raise ValueError("pair comparison changed the fixed layout or extent goal")
    pages = []
    offset = 0
    while True:
        page = compare_mapping_motion_trials(baseline_report_file, candidate_report_file, offset=offset)
        pages.append(page)
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    ids = _graph_ids(first)
    original, timestamps = read_trajectory(Path(first["inputs"]["source_motion_trajectory"]["path"]))
    assert timestamps is not None
    common = _stations(original[ids])
    intervals, summaries = [], []
    for job, attempt, diagnosis in ((baseline_job, parent, baseline_diagnosis), (candidate_job, candidate, candidate_diagnosis)):
        poses, _ = read_trajectory(Path(job["pointcloud"]["files"]["trajectory"]["path"]))
        evidence = json.loads(Path(attempt["files"]["report"]["path"]).read_text())
        intervals.append(_intervals(evidence, _stations(poses), common, timestamps[ids], ids))
        summaries.append({"raw_extent": diagnosis["extent"], "audits": _audit_summary(diagnosis),
                          "routes": retries._routes(attempt), "map_points": job["pointcloud"]["map_points"]})
    if any(summaries[0]["audits"][key]["protocol"] != summaries[1]["audits"][key]["protocol"] for key in summaries[0]["audits"]):
        raise ValueError("pair comparison changed source-audit protocols")
    proposals = [json.loads(Path(a["corridor_proposal"]["path"]).read_text()) for a in (parent, candidate)]
    if proposals[0]["protocol"] != proposals[1]["protocol"]:
        raise ValueError("pair comparison changed source-extraction protocol")
    cuts = sorted({0., float(common[-1]), *[v for rows in intervals for row in rows for v in (row["from_m"], row["to_m"])]})
    changes: dict[str, list[dict[str, float]]] = {"gained": [], "lost": []}
    for lo, hi in zip(cuts, cuts[1:]):
        before, after = [any(row["from_m"] <= (lo + hi) / 2 <= row["to_m"] for row in rows) for rows in intervals]
        if before != after:
            changes["gained" if after else "lost"].append({"from_m": lo, "to_m": hi})
    for summary, rows in zip(summaries, intervals):
        generated = sum(row["to_m"] - row["from_m"] for row in rows)
        fraction = float(generated / common[-1])
        summary["common_extent"] = {"trajectory_length_m": float(common[-1]), "generated_length_m": generated,
            "retained_fraction": fraction, "minimum_retained_fraction": baseline_job["minimum_retained_fraction"],
            "passes_requested_extent": fraction >= baseline_job["minimum_retained_fraction"]}
        summary["included_intervals"] = rows
    inputs: dict[str, Any] = {}
    for value in (baseline_run, baseline_job, candidate_run, candidate_job, baseline_report_file, candidate_report_file):
        _artifacts(value, inputs)
    for folder in (baseline, root):
        for filename in ("job.json", "run.json"):
            _artifacts(jobs._artifact(folder / filename), inputs)
    retries._verify_inputs(inputs)
    windows = [window for page in pages for window in page["windows"]]
    report = {"schema": SCHEMA, "reason": reason, "inputs": inputs,
        "baseline_job_dir": str(baseline), "candidate_job_dir": str(root),
        "baseline": summaries[0], "candidate": summaries[1], "source_changes": changes,
        "motion": {"global_results": pages[0]["global_results"], "windows": windows,
                   "worsened_windows": sum(w["ate_rmse_m_candidate_minus_baseline"] > 0 for w in windows),
                   "reference_provenance": pages[0]["reference_provenance"], "coverage": pages[0]["coverage"]},
        "station_protocol": {"model": "piecewise_retained_frame_association_to_original_odometry_xy_stations",
                             "original_frame_ids": ids, "original_timestamps_s": timestamps[ids].tolist(), "original_stations_m": common.tolist()},
        "pairs": {"baseline": baseline_run["output"], "candidate": candidate_run["output"]},
        "independent_accuracy_established": False, "deployment_ready": False,
        "holds": ["Reference-informed candidate exploration is not unseen-drive validation.",
                  "Source support and temporal interval correspondence do not establish physical road identity or HD accuracy.",
                  "New HD geometry has new lane IDs; baseline routes and explicit connections are not inherited.",
                  "Inspect gains AND losses, all four audits, local motion regressions and unknown traffic semantics before choosing."]}
    target.mkdir(parents=True)
    jobs._save(target / "comparison.json", report)
    jobs._save(target / "selection.json", {"schema": SELECTION_SCHEMA, "revision": 0, "choice": "baseline",
        "comparison": jobs._artifact(target / "comparison.json"), "history": [{"revision": 0, "choice": "baseline", "reason": reason}]})
    return inspect_mapping_motion_selection(str(target))


def _selection(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    saved = json.loads((root / "selection.json").read_text())
    if saved.get("schema") != SELECTION_SCHEMA or saved.get("choice") not in {"baseline", "candidate"}:
        raise ValueError("unsupported motion pair selection")
    _artifact(saved["comparison"])
    comparison = json.loads(Path(saved["comparison"]["path"]).read_text())
    if comparison.get("schema") != SCHEMA:
        raise ValueError("unsupported motion pair comparison")
    retries._verify_inputs(comparison["inputs"])
    for pair in comparison["pairs"].values():
        for artifact in pair["artifacts"].values():
            jobs._verify(artifact)
    return saved, comparison


def inspect_mapping_motion_selection(selection_dir: str) -> dict[str, Any]:
    """Read the exact selected point/HD pair without processing or changing history."""
    root = Path(selection_dir).resolve()
    saved, comparison = _selection(root)
    summaries = deepcopy({key: comparison[key] for key in ("baseline", "candidate", "source_changes", "holds")})
    for name in ("baseline", "candidate"):
        summary = summaries[name]
        summary["included_intervals_total"] = len(summary["included_intervals"])
        summary["included_intervals"] = summary["included_intervals"][:16]
        for audit in summary["audits"].values():
            audit["needs_review_total"] = len(audit["needs_review"])
            audit["needs_review"] = audit["needs_review"][:16]
    for key in ("gained", "lost"):
        changes = summaries["source_changes"]
        changes[f"{key}_total"] = len(changes[key])
        changes[f"{key}_length_m"] = sum(row["to_m"] - row["from_m"] for row in changes[key])
        changes[key] = changes[key][:16]
    return {**saved, "history": saved["history"][-8:], "history_total": len(saved["history"]),
            "selection_file": jobs._artifact(root / "selection.json"),
            "selection_dir": str(root), "output": comparison["pairs"][saved["choice"]],
            "comparison_summary": summaries,
            "motion_summary": {k: v for k, v in comparison["motion"].items() if k != "windows"},
            "motion_windows_file": saved["comparison"], "deployment_ready": False}


def choose_mapping_motion_pair(selection_dir: str, choice: str, reason: str, expected_revision: int) -> dict[str, Any]:
    """Choose or restore one exact draft pair after comparison, without modifying old runs.

    choice is baseline or candidate. Use the inspected revision and an explicit
    reason. One atomic journal update selects BOTH point and HD artifacts together;
    baseline restores the original bytes. Repeating the current choice spends no
    revision; stale revisions are rejected. Verify every frozen input and output
    before committing. Source/extent/semantic holds remain; this is a review draft
    selection, not a quality gate, job certification or deployment operation.
    """
    reason = _reason(reason)
    if choice not in {"baseline", "candidate"} or type(expected_revision) is not int:
        raise ValueError("choose baseline or candidate with an inspected integer revision")
    root = Path(selection_dir).resolve()
    with runs._locked(root):
        saved, comparison = _selection(root)
        if saved["revision"] != expected_revision:
            raise ValueError("stale selection revision; inspect before deciding again")
        if saved["choice"] != choice:
            saved.update(choice=choice, revision=saved["revision"] + 1)
            saved["history"].append({"revision": saved["revision"], "choice": choice, "reason": reason})
            retries._verify_inputs(comparison["inputs"])
            jobs._save(root / "selection.json", saved)
    return inspect_mapping_motion_selection(str(root))
