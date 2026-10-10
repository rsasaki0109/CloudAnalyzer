"""Execute an agent's explicit source-supported HD intervals through fixed gates."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from ca import mapping_job as jobs, mapping_run as runs, mapping_hd_plan as plans
from ca import mapping_retry as retries


@contextmanager
def _locked(root: Path) -> Iterator[None]:
    path = root / ".hd-plan-application-lock"
    try:
        stream = path.open("x")
    except FileExistsError as error:
        raise RuntimeError(
            "HD plan application is busy; inspect the saved actions"
        ) from error
    try:
        with stream:
            stream.write("Applying the explicit retained HD plan")
        yield
    finally:
        path.unlink()


def _selection(
    root: Path, plan_files: list[str], interval_ids: list[int]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = runs._load(root)
    if (
        not isinstance(plan_files, list)
        or not 1 <= len(plan_files) <= 8
        or any(not isinstance(p, str) for p in plan_files)
    ):
        raise ValueError("supply 1..8 inspected plan page paths")
    paths = sorted(str(Path(p).resolve()) for p in plan_files)
    if len(set(paths)) != len(paths):
        raise ValueError("plan pages must be distinct")
    observations: list[dict[str, Any]] = []
    rows: dict[int, dict[str, Any]] = {}
    for path in paths:
        found = [
            h["hd_plan_observation"]
            for h in run["history"]
            if h.get("hd_plan_observation", {}).get("file", {}).get("path") == path
        ]
        if not found:
            raise ValueError("use HD plan pages inspected through this root run")
        observation = found[-1]
        plans.verify_observation(observation)
        if observations and observation["index_file"] != observations[0]["index_file"]:
            raise ValueError("chosen pages must belong to the same frozen plan index")
        observations.append(observation)
        saved = json.loads(Path(path).read_text())
        if saved["schema"] != "cloudanalyzer.hd_gap_plan.v1":
            raise ValueError("unsupported HD plan")
        for row in saved["intervals"]:
            if row["plan_interval_id"] in rows and rows[row["plan_interval_id"]] != row:
                raise ValueError("plan pages contain conflicting interval identities")
            rows[row["plan_interval_id"]] = row
    if (
        not isinstance(interval_ids, list)
        or not 1 <= len(interval_ids) <= 8
        or any(type(i) is not int for i in interval_ids)
        or len(set(interval_ids)) != len(interval_ids)
    ):
        raise ValueError("choose 1..8 distinct inspected plan interval IDs")
    if not set(interval_ids) <= rows.keys():
        raise ValueError("every chosen interval must appear on the inspected plan page")
    selected = [rows[i] for i in interval_ids]
    if any(
        not row["reference_traces_fully_supported"] or row["source_ambiguous"]
        for row in selected
    ):
        raise ValueError(
            "every chosen interval needs complete support from both estimators and unambiguous source"
        )
    selected.sort(key=lambda r: (r["from_m"], r["candidate_id"]))
    if any(a["to_m"] > b["from_m"] for a, b in zip(selected, selected[1:])):
        raise ValueError("selected source intervals overlap")
    # Root and child share this exact proposal. Reject unavailable preview stations
    # before preparing a child or allocating any generation attempt.
    for cid in sorted({r["candidate_id"] for r in selected}):
        candidate = jobs.inspect_mapping_corridors(str(root), candidate_id=cid)[
            "candidate"
        ]
        stations = {s["station_m"] for s in candidate["sections"]}
        if any(
            r["from_m"] not in stations or r["to_m"] not in stations
            for r in selected
            if r["candidate_id"] == cid
        ):
            raise ValueError(
                "selected interval endpoints are outside the bounded candidate preview"
            )
    return {
        "files": [o["file"] for o in observations],
        "candidate_id": observations[0]["candidate_id"],
    }, selected


def apply_supported_hd_plan(
    job_dir: str,
    plan_files: list[str],
    interval_ids: list[int],
    connect_endpoints: bool,
    reason: str,
    expected_revision: int,
) -> dict[str, Any]:
    """Generate and audit only the agent's explicitly chosen preflight intervals.

    Supply 1..8 pages from the same inspect_hd_plan index, 1..8 supported interval
    IDs, a reason, an explicit endpoint-link policy and the current root revision.
    This prepares the unchanged-point HD child, inspects its frozen proposal,
    drafts just those intervals, previews exact geometric joins, creates the
    retention-checked combined patch, finishes the child and compares it to the
    baseline. At most three shared HD attempts are spent. It does NOT finish or
    adopt the parent: read comparison/holds, then use finish_retry or retain the
    baseline. Traffic semantics and independent accuracy remain unverified.

    No candidate ranking or LLM is embedded. False connect_endpoints requires
    an isolated patch; implicit geometric joins are held rather than concealed.
    Interrupted ordinary actions use their existing resume path. Retry this same
    choice with a freshly inspected root revision; completed stages are verified
    and reused, and failed native attempts are never silently rerun. A child with
    unrelated manual decisions requires manual continuation. Keep prior files.
    """
    root = Path(job_dir).resolve()
    if type(connect_endpoints) is not bool:
        raise ValueError("supply an explicit boolean endpoint-link policy")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("supply an application reason")
    with _locked(root):
        with runs._locked(root):
            run = runs._load(root)
            if (
                type(expected_revision) is not int
                or expected_revision != run["revision"]
            ):
                raise ValueError(
                    "stale root revision; inspect before applying or resuming a plan"
                )
            if run["status"] not in {"needs_agent", "interrupted"}:
                raise ValueError("apply a plan on an unfinished root run")
            runs.inspect_mapping_run(str(root))
            job = jobs._load(root)
            jobs._inputs(job)
            if "retry_inputs" in job:
                raise ValueError("HD plan application requires a root run")
            observation, selected = _selection(root, plan_files, interval_ids)
            policy = {
                "schema": "cloudanalyzer.hd_plan_application.v1",
                "plan": observation["files"],
                "interval_ids": sorted(interval_ids),
                "connect_endpoints": connect_endpoints,
                "reason": reason.strip(),
                "candidate_id": observation["candidate_id"],
                "selected_intervals": selected,
            }
            digest = hashlib.sha256(
                json.dumps(policy, sort_keys=True).encode()
            ).hexdigest()
            prefix = f"HD plan {digest}: "
            own = any(h["reason"] == prefix + "prepare child" for h in run["history"])
            if "pointcloud_retry" in job and not own:
                raise ValueError(
                    "the root repair allocation belongs to another decision"
                )
            if not own and (
                jobs._remaining(job) < 3
                or len(run["history"]) + 2 > run["maximum_actions"]
            ):
                raise ValueError(
                    "application needs three remaining HD attempts and two root actions"
                )
            path = root / "hd-plan-application.json"
            if path.exists():
                if json.loads(path.read_text()) != policy:
                    raise ValueError(
                        "this root already has a different explicit HD application"
                    )
            else:
                jobs._save(path, policy)
            policy_file = jobs._artifact(path)
        revisions = {root: expected_revision}

        def step(owner: Path, action: dict[str, Any], label: str) -> dict[str, Any]:
            state = runs._load(owner)
            if owner not in revisions:
                if any(not h["reason"].startswith(prefix) for h in state["history"]):
                    raise ValueError(
                        "child contains unrelated manual decisions; continue through advance_mapping_run"
                    )
                revisions[owner] = state["revision"]
            if state["revision"] != revisions[owner]:
                raise ValueError(
                    "run changed during application; inspect before resuming"
                )
            found: list[dict[str, Any]] = [
                h for h in state["history"] if h["reason"] == prefix + label
            ]
            if found:
                if len(found) != 1 or found[0]["action"] != action:
                    raise ValueError(
                        "saved application action differs from the explicit policy"
                    )
                if found[0]["status"] != "running":
                    jobs._inputs(jobs._load(owner))
                    return found[0]
                if state["status"] != "interrupted":
                    raise RuntimeError(
                        "application action is still running; inspect retained state"
                    )
                next_action = {"type": "resume"}
            else:
                next_action = action
            jobs._verify(policy_file)
            answer = runs.advance_mapping_run(
                str(owner), next_action, prefix + label, revisions[owner]
            )
            revisions[owner] = answer["revision"]
            completed: list[dict[str, Any]] = [
                h for h in runs._load(owner)["history"] if h["reason"] == prefix + label
            ]
            if len(completed) != 1 or completed[0]["action"] != action:
                raise ValueError(
                    "completed application action differs from the explicit policy"
                )
            return completed[0]

        def held(label: str, entry: dict[str, Any]) -> dict[str, Any]:
            return {
                "status": "held",
                "stage": label,
                "outcome": entry.get("outcome"),
                "policy_file": policy_file,
                "root": runs.inspect_mapping_run(str(root)),
                "automatic_adoption": False,
                "deployment_ready": False,
            }

        gap_ids = sorted({r["gap_id"] for r in selected})
        prepared = step(
            root,
            {
                "type": "repair_hd",
                "candidate_id": observation["candidate_id"],
                "gap_ids": gap_ids,
            },
            "prepare child",
        )
        if prepared["status"] != "ready":
            return held("prepare child", prepared)
        child = Path(prepared["outcome"]["stage"]["child_job_dir"])
        inspected = step(
            child,
            {
                "type": "inspect",
                "candidate_ids": sorted({r["candidate_id"] for r in selected}),
            },
            "inspect selected source",
        )
        stations = {
            c["id"]: {s["station_m"] for s in c["sections"]}
            for c in inspected["observation"]
        }
        if any(
            r["from_m"] not in stations[r["candidate_id"]]
            or r["to_m"] not in stations[r["candidate_id"]]
            for r in selected
        ):
            raise ValueError(
                "child preview differs from the inspected source intervals"
            )
        decisions = [
            {
                "candidate_id": r["candidate_id"],
                "from_m": r["from_m"],
                "to_m": r["to_m"],
                "action": "include",
                "reason": reason.strip(),
            }
            for r in selected
        ]
        drafted = step(
            child, {"type": "draft", "decisions": decisions}, "draft selected intervals"
        )
        if drafted["status"] != "audited_draft":
            return held("draft selected intervals", drafted)
        cid = drafted["lane_candidate_id"]
        pairs: list[dict[str, Any]] = []
        offset = 0
        while True:
            inspected_patch = step(
                child,
                {
                    "type": "inspect_patch",
                    "candidate_id": cid,
                    "gap_ids": gap_ids,
                    "offset": offset,
                },
                f"inspect patch {offset}",
            )
            preview = inspected_patch["patch_observation"]
            if preview["holds"] or (preview["pairs_total"] and not connect_endpoints):
                return {
                    **held("inspect patch", inspected_patch),
                    "patch_preview": preview,
                    "holds": preview["holds"]
                    or ["endpoint_links_require_explicit_policy"],
                }
            pairs.extend(
                {"from": p["from"], "to": p["to"], "reason": reason.strip()}
                for p in preview["pairs"]
            )
            if preview["next_offset"] is None:
                break
            offset = preview["next_offset"]
            if len(pairs) > 64:
                raise ValueError("patch exceeds the explicit endpoint-pair budget")
        patched = step(
            child,
            {
                "type": "patch_gaps",
                "candidate_id": cid,
                "gap_ids": gap_ids,
                "pairs": pairs,
            },
            "patch retained map",
        )
        if patched["status"] != "audited_draft":
            return held("patch retained map", patched)
        checks_file = patched["outcome"]["patch_checks"]
        jobs._verify(checks_file)
        checks = json.loads(Path(checks_file["path"]).read_text())
        if not checks["passes"]:
            return held("patch retained map", patched)
        final_id = patched["lane_candidate_id"]
        step(
            child, {"type": "finish", "candidate_id": final_id}, "finish audited child"
        )
        compared = step(
            root,
            {"type": "compare_retry", "candidate_id": final_id},
            "compare retained baseline",
        )
        comparison = compared["retry_comparison"]
        if comparison["lost_source_length_m"] != 0:
            return {
                **held("compare retained baseline", compared),
                "comparison": comparison,
                "holds": ["source_extent_regressed"],
            }
        jobs._verify(comparison["file"])
        retries._verify_inputs(
            json.loads(Path(comparison["file"]["path"]).read_text())["inputs"]
        )
        return {
            "status": "ready_for_agent_decision",
            "policy_file": policy_file,
            "candidate_job_dir": str(child),
            "candidate_id": final_id,
            "comparison": comparison,
            "patch_checks": checks,
            "patch_checks_file": checks_file,
            "root": runs.inspect_mapping_run(str(root)),
            "child_output": runs.inspect_mapping_run(str(child))["output"],
            "automatic_adoption": False,
            "independent_accuracy_established": False,
            "deployment_ready": False,
        }
