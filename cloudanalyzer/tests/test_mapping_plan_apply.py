"""The high-level application keeps caller choices, budgets and frozen gates."""

import json
from pathlib import Path

import pytest
from ca import mapping_job as jobs, mapping_run as runs
from ca.mapping_plan_apply import apply_supported_hd_plan
from tests.test_mapping_hd_plan import plan
from tests.test_mapping_hd_repair import hd_only_run
from tests.test_mapping_job import (
    job_backend,
    unused_frame_run,
    _attach_lane_native,
    _advance,
    _run_layout,
)


def apply(root, page, ids=None, connect=True, revision=None):
    return apply_supported_hd_plan(
        str(root),
        [page["file"]["path"]],
        ids if ids is not None else [1, 2],
        connect,
        "Apply only chosen supported observed intervals",
        (
            runs.inspect_mapping_run(str(root))["revision"]
            if revision is None
            else revision
        ),
    )


def test_complete_patch_retains_points_and_baseline_and_reuses_all_stages(hd_only_run):
    root = hd_only_run
    page = plan(root)
    _attach_lane_native()
    before = jobs._load(root)
    result = apply(root, page)
    assert result["status"] == "ready_for_agent_decision", result
    assert result["patch_checks"]["passes"] and result["patch_checks"]["holds"] == []
    assert result["comparison"]["gained_source_length_m"] == 4
    assert result["comparison"]["lost_source_length_m"] == 0
    assert not result["automatic_adoption"] and not result["deployment_ready"]
    assert runs._load(root)["status"] == "needs_agent"
    child = Path(result["candidate_job_dir"])
    after = jobs._load(child)
    assert after["pointcloud"] == before["pointcloud"]
    assert len(after["attempts"]) == 3
    assert runs._load(child)["status"] == "finished"
    root_state, child_state = (root / "run.json").read_bytes(), (
        child / "run.json"
    ).read_bytes()
    repeated = apply(root, page)
    assert repeated["candidate_id"] == result["candidate_id"]
    assert (root / "run.json").read_bytes() == root_state
    assert (child / "run.json").read_bytes() == child_state
    _advance(root, {"type": "finish_retry", "candidate_id": result["candidate_id"]})
    assert (
        runs._load(root)["output"]["artifacts"]["map"]
        == before["pointcloud"]["files"]["map"]
    )
    assert "hd_application_policy" in runs._load(root)["output"]["artifacts"]


@pytest.mark.parametrize("fault", ["unseen", "duplicate", "stale", "different_page"])
def test_invalid_policy_spends_no_attempt_and_creates_no_child(
    hd_only_run, fault, tmp_path
):
    root = hd_only_run
    page = plan(root)
    before = (root / "job.json").read_bytes()
    kwargs = {}
    if fault == "unseen":
        kwargs["ids"] = [999]
    elif fault == "duplicate":
        kwargs["ids"] = [1, 1]
    elif fault == "stale":
        kwargs["revision"] = 0
    else:
        copied = tmp_path / "unauthorized.json"
        copied.write_bytes(Path(page["file"]["path"]).read_bytes())
        page = {**page, "file": {"path": str(copied)}}
    with pytest.raises(ValueError):
        apply(root, page, **kwargs)
    assert (root / "job.json").read_bytes() == before
    assert not (root / "hd-repair").exists()
    assert not (root / "hd-plan-application.json").exists()


def test_explicit_isolated_policy_holds_geometric_links_without_patching(hd_only_run):
    root = hd_only_run
    page = plan(root)
    _attach_lane_native()
    result = apply(root, page, connect=False)
    assert result["status"] == "held"
    assert result["holds"] == ["endpoint_links_require_explicit_policy"]
    child = root / "hd-repair"
    assert len(jobs._load(child)["attempts"]) == 2
    assert runs._load(root)["output"] is None
    repeated = apply(root, page, connect=False)
    assert repeated["status"] == "held"
    assert len(jobs._load(child)["attempts"]) == 2


def test_interrupted_lane_stage_resumes_completed_geometry_once(
    hd_only_run, monkeypatch
):
    root = hd_only_run
    page = plan(root)
    _attach_lane_native()
    original = jobs.generate_mapping_corridor_lanes

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted before lane allocation")

    monkeypatch.setattr(jobs, "generate_mapping_corridor_lanes", interrupted)
    with pytest.raises(RuntimeError, match="interrupted before"):
        apply(root, page)
    child = root / "hd-repair"
    assert len(jobs._load(child)["attempts"]) == 1
    monkeypatch.setattr(jobs, "generate_mapping_corridor_lanes", original)
    result = apply(root, page)
    assert result["status"] == "ready_for_agent_decision", result
    assert len(jobs._load(child)["attempts"]) == 3


def test_real_unsupported_heights_are_rejected_before_preparing_a_child(hd_only_run):
    original = jobs._load(hd_only_run)
    proposal = json.loads(
        Path(original["corridor_proposal"]["file"]["path"]).read_text()
    )
    for side in ("center", "left", "right"):
        proposal["candidates"][0]["sections"][3][side][2] += 0.5
    jobs.core().propose_road_corridors = lambda *args: json.dumps(proposal)
    root = hd_only_run.parent / "unsupported-application"
    runs.start_mapping_run(original["source"]["path"], str(root), _run_layout())
    _advance(root, {"type": "inspect", "candidate_ids": [1]})
    _advance(
        root,
        {
            "type": "draft",
            "decisions": [
                {
                    "candidate_id": 1,
                    "action": "include",
                    "from_m": a,
                    "to_m": b,
                    "reason": "Retain supported outer ranges",
                }
                for a, b in [(0.0, 4.0), (8.0, 10.0)]
            ],
        },
    )
    _advance(root, {"type": "inspect_gaps", "candidate_id": 2, "offset": 0})
    page = plan(root)
    assert not any(r["reference_traces_fully_supported"] for r in page["intervals"])
    before = (root / "job.json").read_bytes()
    with pytest.raises(ValueError, match="complete support"):
        apply(root, page)
    assert (root / "job.json").read_bytes() == before
    assert not (root / "hd-repair").exists()


def test_failed_native_lane_attempt_is_retained_without_automatic_rerun(
    hd_only_run, monkeypatch
):
    root = hd_only_run
    page = plan(root)
    _attach_lane_native()

    def failed(*args, **kwargs):
        raise RuntimeError("native lane construction failed")

    monkeypatch.setattr(jobs.core(), "build_corridor_lanes", failed)
    result = apply(root, page)
    assert result["status"] == "held" and result["stage"] == "draft selected intervals"
    child = root / "hd-repair"
    before = (child / "job.json").read_bytes()
    assert len(jobs._load(child)["attempts"]) == 2
    repeated = apply(root, page)
    assert repeated["status"] == "held"
    assert (child / "job.json").read_bytes() == before
    assert runs._load(root)["output"] is None


def test_pages_from_one_index_combine_without_changing_choices_on_reordered_retry(
    hd_only_run,
):
    root = hd_only_run
    first, second = plan(root), plan(root, offset=1)
    _attach_lane_native()
    files = [first["file"]["path"], second["file"]["path"]]

    def invoke(paths):
        return apply_supported_hd_plan(
            str(root),
            paths,
            [1, 2],
            True,
            "Combine inspected pages",
            runs.inspect_mapping_run(str(root))["revision"],
        )

    result = invoke(files)
    assert result["status"] == "ready_for_agent_decision", result
    child = Path(result["candidate_job_dir"])
    before = (child / "job.json").read_bytes()
    assert invoke(list(reversed(files)))["candidate_id"] == result["candidate_id"]
    assert (child / "job.json").read_bytes() == before
