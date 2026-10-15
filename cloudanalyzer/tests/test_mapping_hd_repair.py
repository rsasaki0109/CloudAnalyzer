"""HD-only repairs use frozen evidence and retain the exact accepted point map."""

import json
from pathlib import Path
from copy import deepcopy

import pytest
from ca import mapping_job as jobs, mapping_run as runs
from tests.test_mapping_job import (
    job_backend,
    unused_frame_run,
    _corridor_report,
    _run_layout,
    _advance,
    _patch_pairs,
)


@pytest.fixture
def hd_only_run(unused_frame_run, request):
    original = jobs._load(unused_frame_run)
    report = json.loads(Path(original["corridor_proposal"]["file"]["path"]).read_text())
    report.update(
        candidates=_corridor_report()["candidates"],
        deferred_intervals=[],
        with_candidate_station_length_m=10.0,
        without_candidate_station_length_m=0.0,
        trajectory_covered_station_length_m=10.0,
    )
    jobs.core().propose_road_corridors = lambda *args: json.dumps(deepcopy(report))
    root = unused_frame_run.parent / "hd-only-base"
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
                    "reason": "Keep observed outer pieces",
                }
                for a, b in getattr(request, "param", [(0.0, 4.0), (8.0, 10.0)])
            ],
        },
    )
    _advance(root, {"type": "inspect_gaps", "candidate_id": 2, "offset": 0})
    return root


def child_run(root):
    result = _advance(root, {"type": "repair_hd", "candidate_id": 2, "gap_ids": [1]})
    assert result["hd_repair_result"]["status"] == "ready", result
    return root / "hd-repair"


def draft(child):
    _advance(child, {"type": "inspect", "candidate_ids": [1]})
    return _advance(
        child,
        {
            "type": "draft",
            "decisions": [
                {
                    "candidate_id": 1,
                    "action": "include",
                    "from_m": 4.0,
                    "to_m": 8.0,
                    "reason": "Fill observed deferred interval without point processing",
                }
            ],
        },
    )


def test_adopts_native_patch_with_exact_points_and_shared_budget(
    hd_only_run, monkeypatch
):
    from ca import mapping_retry as retry

    root = hd_only_run
    before = jobs._load(root)

    def forbidden(*args, **kwargs):
        raise AssertionError("HD-only repair regenerated evidence")

    for module, name in [
        (retry, "fix_session"),
        (jobs, "fix_session"),
        (jobs, "odometry"),
        (jobs.core(), "propose_road_corridors"),
    ]:
        monkeypatch.setattr(module, name, forbidden)
    child = child_run(root)
    job = jobs._load(child)
    assert job["pointcloud"] == before["pointcloud"] and job["attempts"] == []
    assert (
        job["corridor_proposal"]["file"] == before["attempts"][1]["corridor_proposal"]
    )
    assert (
        jobs.inspect_mapping_job(str(root))["remaining_attempts"] == 0
        and job["max_attempts"] == 4
    )
    draft(child)
    with pytest.raises(ValueError, match="combined HD patch"):
        _advance(root, {"type": "compare_retry", "candidate_id": 2})
    with pytest.raises(ValueError, match="frozen source"):
        _advance(child, {"type": "refine", "association": "trajectory_containing"})
    with pytest.raises(ValueError, match="frozen source"):
        jobs._refine_mapping_corridors(str(child), "Cannot regenerate source")
    _advance(
        child, {"type": "inspect_patch", "candidate_id": 2, "gap_ids": [1], "offset": 0}
    )
    patched = _advance(
        child,
        {
            "type": "patch_gaps",
            "candidate_id": 2,
            "gap_ids": [1],
            "pairs": _patch_pairs(child),
        },
    )
    assert patched["patch_result"]["status"] == "audited_draft", patched
    compared = _advance(root, {"type": "compare_retry", "candidate_id": 3})[
        "retry_comparison"
    ]
    assert (
        compared["pointcloud_artifacts_identical"]
        and compared["source_proposal_identical"]
    )
    assert (
        not compared["pointcloud_regenerated"]
        and compared["retry_strategy"] == "hd_only"
    )
    assert (
        compared["gained_source_length_m"] == 4
        and compared["lost_source_length_m"] == 0
    )
    with pytest.raises(ValueError, match="finish the child"):
        _advance(root, {"type": "finish_retry", "candidate_id": 3})
    _advance(child, {"type": "finish", "candidate_id": 3})
    final = _advance(root, {"type": "finish_retry", "candidate_id": 3})["output"]
    assert (
        final["hd_repair_decision"]["adopted"]
        and not final["hd_repair_decision"]["pointcloud_regenerated"]
    )
    assert "pointcloud_retry_decision" not in final
    assert final["artifacts"]["map"] == before["pointcloud"]["files"]["map"]
    assert len(jobs._load(child)["attempts"]) == 3


@pytest.mark.parametrize(
    "target", ["pointcloud", "proposal", "options", "manifest", "budget"]
)
def test_pins_artifact_identity_and_processing_choices(hd_only_run, target):
    child = child_run(hd_only_run)
    job = jobs._load(child)
    if target == "pointcloud":
        path = child / "redirected.ply"
        path.write_bytes(Path(job["pointcloud"]["files"]["map"]["path"]).read_bytes())
        job["pointcloud"]["files"]["map"] = jobs._artifact(path)
    elif target == "proposal":
        path = child / "redirected.json"
        path.write_bytes(Path(job["corridor_proposal"]["file"]["path"]).read_bytes())
        job["corridor_proposal"]["file"] = jobs._artifact(path)
    elif target == "budget":
        job["max_attempts"] = 8
    elif target == "options":
        job["pointcloud_options"]["map_voxel_m"] = 0.05
    else:
        Path(job["retry_inputs"]["hd_repair_manifest"]["path"]).write_text("{}")
    jobs._save(child / "job.json", job)
    with pytest.raises(ValueError, match="changed"):
        runs.inspect_mapping_run(str(child))


def test_needs_seen_gaps_three_attempts_and_exclusive_allocation(hd_only_run):
    root = hd_only_run
    for ids in ([], [True], [1, 1], [2]):
        with pytest.raises(ValueError):
            _advance(root, {"type": "repair_hd", "candidate_id": 2, "gap_ids": ids})
    assert "pointcloud_retry" not in jobs._load(root)
    job = jobs._load(root)
    job["max_attempts"] = 4
    jobs._save(root / "job.json", job)
    with pytest.raises(ValueError, match="shared HD attempts"):
        child_run(root)
    job["max_attempts"] = 6
    jobs._save(root / "job.json", job)
    child = child_run(root)
    with pytest.raises(ValueError, match="recursive"):
        _advance(
            root,
            {
                "type": "retry_pointcloud",
                "candidate_id": 2,
                "gap_ids": [1],
                "options": {"scan_voxel_m": 0.2, "map_voxel_m": 0.1},
            },
        )
    draft(child)
    _advance(child, {"type": "inspect_gaps", "candidate_id": 2, "offset": 0})
    with pytest.raises(ValueError, match="recursive"):
        _advance(child, {"type": "repair_hd", "candidate_id": 2, "gap_ids": [1]})


def test_resume_reuses_ready_child_without_double_allocation(hd_only_run, monkeypatch):
    original = runs._retry

    def interrupted(*args):
        original(*args)
        raise KeyboardInterrupt()

    monkeypatch.setattr(runs, "_retry", interrupted)
    with pytest.raises(KeyboardInterrupt):
        child_run(hd_only_run)
    child = hd_only_run / "hd-repair"
    before = (child / "job.json").read_bytes()
    monkeypatch.setattr(runs, "_retry", original)
    result = _advance(hd_only_run, {"type": "resume"})
    assert (
        result["hd_repair_result"]["status"] == "ready"
        and result["remaining_attempts"] == 0
    )
    assert (child / "job.json").read_bytes() == before


def test_failed_patch_keeps_baseline(hd_only_run, monkeypatch):
    root = hd_only_run
    child = child_run(root)
    draft(child)
    _advance(
        child, {"type": "inspect_patch", "candidate_id": 2, "gap_ids": [1], "offset": 0}
    )
    original = jobs.core().audit_vector_map_quality_details

    def bad_support(*args):
        audit = json.loads(original(*args))
        audit["quality"]["lanes"][-1]["left"].update(
            supported=0, fraction=0.0, start_supported=False, end_supported=False
        )
        return json.dumps(audit)

    monkeypatch.setattr(jobs.core(), "audit_vector_map_quality_details", bad_support)
    failed = _advance(
        child,
        {
            "type": "patch_gaps",
            "candidate_id": 2,
            "gap_ids": [1],
            "pairs": _patch_pairs(child),
        },
    )
    assert failed["patch_result"]["status"] == "failed"
    with pytest.raises(ValueError):
        _advance(root, {"type": "compare_retry", "candidate_id": 3})
    final = _advance(root, {"type": "finish", "candidate_id": 2})["output"]
    assert not final["hd_repair_decision"]["adopted"]
    assert final["artifacts"]["map"] == jobs._load(root)["pointcloud"]["files"]["map"]


def test_preparation_error_preserves_retained_pair(hd_only_run, monkeypatch):
    from ca import mapping_hd_repair as repair

    root = hd_only_run
    before = jobs._load(root)

    def fail(*args):
        raise OSError("simulated unavailable child layout copy")

    monkeypatch.setattr(repair.shutil, "copyfile", fail)
    with pytest.raises(OSError):
        child_run(root)
    assert jobs._load(root)["pointcloud_retry"]["status"] == "failed"
    final = _advance(root, {"type": "finish", "candidate_id": 2})["output"]
    assert final["artifacts"]["map"] == before["pointcloud"]["files"]["map"]
    assert final["artifacts"]["hd_map"] == before["attempts"][1]["files"]["map"]


@pytest.mark.parametrize(
    "hd_only_run", [[(0.0, 2.0), (4.0, 6.0), (8.0, 10.0)]], indirect=True
)
def test_patch_rejects_an_inspected_gap_outside_allocated_choice(hd_only_run):
    child = child_run(hd_only_run)
    _advance(child, {"type": "inspect", "candidate_ids": [1]})
    _advance(
        child,
        {
            "type": "draft",
            "decisions": [
                {
                    "candidate_id": 1,
                    "action": "include",
                    "from_m": 6.0,
                    "to_m": 8.0,
                    "reason": "Observed but outside the explicit allocation",
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="allocated root gaps"):
        _advance(
            child,
            {"type": "inspect_patch", "candidate_id": 2, "gap_ids": [2], "offset": 0},
        )
    assert len(jobs._load(child)["attempts"]) == 2


def test_inspected_gap_receipts_are_required_before_allocation(hd_only_run):
    root = hd_only_run
    run = json.loads((root / "run.json").read_text())
    run["history"] = [h for h in run["history"] if "gap_observation" not in h]
    jobs._save(root / "run.json", run)
    with pytest.raises(ValueError, match="inspect gaps"):
        child_run(root)
    assert "pointcloud_retry" not in jobs._load(root)


def test_source_gap_occupied_by_connector_is_visible_and_not_reallocated(hd_only_run):
    from tests.test_mapping_job import _attach_lane_native
    jobs.core().connect_vector_map_junctions = _attach_lane_native().connect_vector_map_junctions
    root = hd_only_run
    observed = _advance(
        root, {"type": "inspect_connections", "candidate_id": 2, "offset": 0}
    )["connection_observation"]
    assert any(c["from"] == 3 and c["to"] == 6 for c in observed["candidates"])
    _advance(
        root,
        {
            "type": "connect",
            "candidate_id": 2,
            "pairs": [{"from": 3, "to": 6, "reason": "Explicit supported path bridge"}],
        },
    )
    gap = _advance(root, {"type": "inspect_gaps", "candidate_id": 3, "offset": 0})[
        "gap_observation"
    ]["gaps"][0]
    assert gap["retained_hd_occupancy"]["available"]
    assert gap["retained_hd_occupancy"]["unoccupied_intervals_total"] == 0
    assert gap["retained_hd_occupancy"]["overlaps_total"] == 1
    before = (root / "run.json").read_bytes()
    with pytest.raises(ValueError, match="unoccupied intervals"):
        _advance(root, {"type": "repair_hd", "candidate_id": 3, "gap_ids": [1]})
    assert (
        root / "run.json"
    ).read_bytes() == before and "pointcloud_retry" not in jobs._load(root)
