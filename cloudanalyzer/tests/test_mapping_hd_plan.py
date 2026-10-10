"""Preflight source evidence must not generate or silently adopt an HD map."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from ca import mapping_job as jobs, mapping_run as runs
from tests.test_mapping_job import (
    job_backend,
    unused_frame_run,
    _advance,
    _patch_pairs,
    _attach_lane_native,
    _run_layout,
)
from tests.test_mapping_hd_repair import hd_only_run, child_run


def plan(root, offset=0):
    return _advance(
        root,
        {
            "type": "inspect_hd_plan",
            "candidate_id": 2,
            "gap_ids": [1],
            "offset": offset,
        },
    )["hd_plan_observation"]


def test_native_reference_checks_match_generated_traces_without_generation(
    hd_only_run, monkeypatch
):
    root = hd_only_run
    before = (root / "job.json").read_bytes()
    builder = jobs.core().build_corridor_lanes

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight generated or exported a lane")

    monkeypatch.setattr(jobs.core(), "build_corridor_lanes", forbidden)
    monkeypatch.setattr(jobs.core(), "edit_vector_map_relations", forbidden)
    observed = plan(root)
    assert (root / "job.json").read_bytes() == before
    assert [(r["from_m"], r["to_m"]) for r in observed["intervals"]] == [
        (4.0, 6.0),
        (6.0, 8.0),
    ]
    assert all(r["reference_traces_fully_supported"] for r in observed["intervals"])
    assert all(r["retained_endpoint_links"] for r in observed["intervals"])
    assert (
        observed["protocol"]["hd_generation_attempts_spent"] == 0
        and not observed["protocol"]["lane_export_performed"]
    )
    assert jobs.inspect_mapping_job(str(root))["remaining_attempts"] == 4
    monkeypatch.setattr(jobs.core(), "build_corridor_lanes", builder)
    jobs.core().edit_vector_map_relations = (
        _attach_lane_native().edit_vector_map_relations
    )
    child = child_run(root)
    _advance(child, {"type": "inspect", "candidate_ids": [1]})
    decisions = [
        {
            **{k: r[k] for k in ("candidate_id", "from_m", "to_m")},
            "action": "include",
            "reason": "Use inspected observed range",
        }
        for r in observed["intervals"]
    ]
    _advance(child, {"type": "draft", "decisions": decisions})
    candidate = jobs._load(child)["attempts"][1]
    after = json.loads(Path(candidate["quality_report"]["path"]).read_text())
    for index, r in enumerate(observed["intervals"]):
        source = json.loads(Path(r["source_audits_file"]["path"]).read_text())["audits"]
        for estimator, audit in [
            ("quantile", after["editable"]),
            ("consensus", after["ground_consensus"]["editable"]),
        ]:
            for role in ("center", "left", "right"):
                assert (
                    source[estimator]["quality"]["lanes"][0][role]
                    == audit["quality"]["lanes"][index][role]
                )
    _advance(
        child, {"type": "inspect_patch", "candidate_id": 2, "gap_ids": [1], "offset": 0}
    )
    result = _advance(
        child,
        {
            "type": "patch_gaps",
            "candidate_id": 2,
            "gap_ids": [1],
            "pairs": _patch_pairs(child),
        },
    )
    assert result["patch_result"]["status"] == "audited_draft", result
    compared = _advance(root, {"type": "compare_retry", "candidate_id": 3})[
        "retry_comparison"
    ]
    assert (
        compared["gained_source_length_m"] == 4
        and compared["lost_source_length_m"] == 0
    )


def test_completed_plan_and_empty_page_do_not_repeat_source_queries(
    hd_only_run, monkeypatch
):
    observed = plan(hd_only_run)

    def forbidden(*args):
        pytest.fail("retained source inspection repeated")

    monkeypatch.setattr(jobs.core(), "audit_vector_map_quality_details", forbidden)
    monkeypatch.setattr(
        jobs.core(), "audit_vector_map_ground_consensus_details", forbidden
    )
    assert plan(hd_only_run) == observed
    empty = plan(hd_only_run, 99)
    assert (
        empty["intervals"] == []
        and empty["intervals_total"] == 2
        and empty["next_offset"] is None
    )
    assert jobs.inspect_mapping_job(str(hd_only_run))["remaining_attempts"] == 4


def test_real_height_failure_holds_reference_ranges_without_modifying_baseline(
    hd_only_run,
):
    original = jobs._load(hd_only_run)
    proposal = json.loads(
        Path(original["corridor_proposal"]["file"]["path"]).read_text()
    )
    for side in ("center", "left", "right"):
        proposal["candidates"][0]["sections"][3][side][2] += 0.5
    jobs.core().propose_road_corridors = lambda *args: json.dumps(proposal)
    root = hd_only_run.parent / "height-held-root"
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
                    "reason": "Observed outer baseline",
                }
                for a, b in [(0.0, 4.0), (8.0, 10.0)]
            ],
        },
    )
    _advance(root, {"type": "inspect_gaps", "candidate_id": 2, "offset": 0})
    before = (root / "job.json").read_bytes()
    result = plan(root)
    assert all(not r["reference_traces_fully_supported"] for r in result["intervals"])
    assert all(
        r["reference_audits"]["quantile"]["sample_totals"]["height_mismatches"] > 0
        for r in result["intervals"]
    )
    assert (root / "job.json").read_bytes() == before


@pytest.mark.parametrize("fault", ["source_audits", "index", "page"])
def test_saved_plan_provenance_changes_are_detected(hd_only_run, fault):
    result = plan(hd_only_run)
    page = Path(result["file"]["path"])
    saved = json.loads(page.read_text())
    artifact = (
        result["intervals"][0]["source_audits_file"]
        if fault == "source_audits"
        else (saved["index"] if fault == "index" else result["file"])
    )
    Path(artifact["path"]).write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        runs.inspect_mapping_run(str(hd_only_run))


def test_interrupted_second_range_resumes_frozen_first_range(hd_only_run, monkeypatch):
    core = jobs.core()
    original = core.audit_vector_map_ground_consensus_details
    calls = 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return original(*args)

    monkeypatch.setattr(core, "audit_vector_map_ground_consensus_details", interrupted)
    with pytest.raises(KeyboardInterrupt):
        plan(hd_only_run)
    retained = list(hd_only_run.glob("hd-plan-*/source-0001.json"))
    assert len(retained) == 1
    before = retained[0].read_bytes()
    monkeypatch.setattr(core, "audit_vector_map_ground_consensus_details", original)
    result = _advance(hd_only_run, {"type": "resume"})["hd_plan_observation"]
    assert all(r["reference_traces_fully_supported"] for r in result["intervals"])
    assert (
        retained[0].read_bytes() == before
        and jobs.inspect_mapping_job(str(hd_only_run))["remaining_attempts"] == 4
    )


def test_width_and_occupied_connector_ranges_are_not_offered(hd_only_run):
    from ca.mapping_hd_plan import _index
    from ca.mapping_patch import _intervals

    root = hd_only_run
    j = jobs._load(root)
    parent = j["attempts"][1]
    proposal = json.loads(Path(parent["corridor_proposal"]["path"]).read_text())
    gap = {"id": 1, "from_m": 4.0, "to_m": 8.0}
    narrow = deepcopy(proposal)
    for side in ("left", "right"):
        narrow["candidates"][0]["sections"][3][side][1] *= 0.5
    assert _index(parent, narrow, [gap], _run_layout()) == []
    jobs.core().connect_vector_map_junctions = (
        _attach_lane_native().connect_vector_map_junctions
    )
    _advance(root, {"type": "inspect_connections", "candidate_id": 2, "offset": 0})
    _advance(
        root,
        {
            "type": "connect",
            "candidate_id": 2,
            "pairs": [{"from": 3, "to": 6, "reason": "Supported retained connector"}],
        },
    )
    connected = jobs._load(root)["attempts"][2]
    assert any(a == 4.0 and b == 8.0 for a, b in _intervals(connected).values())
    assert _index(connected, proposal, [gap], _run_layout()) == []


def test_invalid_choices_and_large_source_query_are_rejected_before_processing(
    hd_only_run, monkeypatch
):
    for ids, offset in [
        ([], 0),
        ([True], 0),
        ([1, 1], 0),
        ([2], 0),
        ([1], -1),
        ([1], True),
    ]:
        with pytest.raises(ValueError):
            _advance(
                hd_only_run,
                {
                    "type": "inspect_hd_plan",
                    "candidate_id": 2,
                    "gap_ids": ids,
                    "offset": offset,
                },
            )
    from ca import mapping_hd_plan as planner

    original = planner._index

    def over_budget(*args):
        rows = original(*args)
        for r in rows:
            r["source_sample_upper_bound"] = 100001
        return rows

    monkeypatch.setattr(planner, "_index", over_budget)
    monkeypatch.setattr(
        jobs.core(),
        "audit_vector_map_quality_details",
        lambda *args: pytest.fail("budget exceeded before source query"),
    )
    with pytest.raises(ValueError, match="sampling budget"):
        plan(hd_only_run)
    assert jobs.inspect_mapping_job(str(hd_only_run))["remaining_attempts"] == 4


def test_continuation_pins_the_preflight_index_and_source_reports(hd_only_run):
    from ca.mapping_revision import continue_mapping_run

    root = hd_only_run
    observed = plan(root)
    child = child_run(root)
    _advance(child, {"type": "inspect", "candidate_ids": [1]})
    _advance(
        child,
        {
            "type": "draft",
            "decisions": [
                {
                    "candidate_id": 1,
                    "action": "include",
                    "from_m": r["from_m"],
                    "to_m": r["to_m"],
                    "reason": "Observed supported range",
                }
                for r in observed["intervals"]
            ],
        },
    )
    _advance(
        child, {"type": "inspect_patch", "candidate_id": 2, "gap_ids": [1], "offset": 0}
    )
    _advance(
        child,
        {
            "type": "patch_gaps",
            "candidate_id": 2,
            "gap_ids": [1],
            "pairs": _patch_pairs(child),
        },
    )
    _advance(root, {"type": "compare_retry", "candidate_id": 3})
    _advance(child, {"type": "finish", "candidate_id": 3})
    _advance(root, {"type": "finish_retry", "candidate_id": 3})
    target = root.parent / "continued-from-plan"
    continue_mapping_run(str(root), str(target), 3, "Keep all decision evidence")
    pinned = jobs._load(target)["continuation_inputs"].values()
    assert observed["index_file"] in pinned
    assert all(r["source_audits_file"] in pinned for r in observed["intervals"])
    Path(observed["index_file"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        runs.inspect_mapping_run(str(target))
