"""Agent choices retain artifacts and failures without laundering quality holds."""

import json
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ca import mapping_job as jobs
from cloudanalyzer_cli.main import app

OPTIONS = {"forward_lanes": 1, "backward_lanes": 0, "left_hand_traffic": False,
           "lane_width": 3.5, "speed_limit": 25}


def _lane_specs(center=1, width=1.5):
    return [{"center_curve_id": center, "speed_limit_kmh": 40., "reason": "Test an explicitly unverified lane layout",
             "lanes": [{"direction": "forward", "kind": "driving", "one_way": True,
                        "fraction": 1., "minimum_width_m": width}]}]


def _attach_lane_native():
    native = pytest.importorskip("cloudanalyzer_core")
    if not hasattr(native, "build_corridor_lanes"):
        pytest.skip("installed core predates corridor lane drafting")
    jobs.core().build_corridor_lanes = native.build_corridor_lanes
    jobs.core().edit_vector_map_relations = native.edit_vector_map_relations
    return native


def _run_layout(width=1.5):
    spec = _lane_specs(width=width)[0]
    return {"boundary_policy": "source_span_hypothesis", **{k: spec[k] for k in ("reason", "lanes", "speed_limit_kmh")}}


@pytest.mark.parametrize("proposal_id", [1, 37])
def test_agent_run_binds_one_layout_rejects_stale_replay_and_returns_both_maps(job_backend, tmp_path, proposal_id):
    from ca import mapping_run as runs
    source, _ = job_backend
    _attach_lane_native()
    report = _corridor_report()
    report["candidates"][0]["id"] = proposal_id
    jobs.core().propose_road_corridors = lambda *args: json.dumps(report)
    root = tmp_path / "run"
    layout = tmp_path / "layout.json"
    layout.write_text(json.dumps(_run_layout()))
    started = CliRunner().invoke(app, ["mapping-run", str(source), "--out", str(root), "--layout", str(layout)])
    assert started.exit_code == 0, started.output
    run = json.loads(started.stdout)
    assert run["status"] == "needs_agent" and run["reviewed_candidates"] == [] and run["revision"] == 0
    observed_id = run["candidate_index"]["candidate_index"][0]["id"]
    choices = [{"candidate_id": observed_id, "action": "include", "reason": "Retain observed source span"}]
    with pytest.raises(ValueError, match="inspect candidates"):
        runs.advance_mapping_run(str(root), {"type": "draft", "decisions": choices}, "Read evidence first", 0)
    run = runs.advance_mapping_run(str(root), {"type": "inspect", "candidate_ids": [observed_id]}, "Read original cross-sections", 0)
    assert run["observation"][0]["sections"][0]["station_m"] == 0.
    run = runs.advance_mapping_run(str(root), {"type": "draft", "decisions": choices}, "Explicit source-span hypothesis", run["revision"])
    assert run["draft_result"]["status"] == "audited_draft" and run["remaining_attempts"] == 4
    assert run["draft_result"]["diagnosis"]["extent"]["passes_requested_extent"] is True
    before = (root / "run.json").read_bytes()
    with pytest.raises(ValueError, match="stale"):
        runs.advance_mapping_run(str(root), {"type": "draft", "decisions": choices}, "Do not replay a completed stage", 1)
    assert (root / "run.json").read_bytes() == before
    action = tmp_path / "finish.json"
    action.write_text(json.dumps({"type": "finish", "candidate_id": 2}))
    finished = CliRunner().invoke(app, ["mapping-run-advance", str(root), "--action", str(action), "--revision", str(run["revision"]),
                                     "--reason", "Deliver both draft maps with source and semantic holds"])
    assert finished.exit_code == 0, finished.output
    final = json.loads(finished.stdout)
    assert final["status"] == "finished" and final["output"]["status"] == "draft_needs_review"
    assert {"map", "trajectory", "hd_map", "hd_editable_map", "hd_source_audits"} <= final["output"]["artifacts"].keys()
    assert final["output"]["road_semantics_inferred"] is False and jobs.inspect_mapping_job(str(root))["selected"] is None
    inspect = CliRunner().invoke(app, ["mapping-run-inspect", str(root)])
    assert inspect.exit_code == 0 and json.loads(inspect.stdout)["output"] == final["output"]
    (root / "layout-hypothesis.json").write_text(json.dumps(_run_layout(width=1.)))
    with pytest.raises(ValueError, match="changed"):
        runs.inspect_mapping_run(str(root))


def test_agent_run_retains_failed_layout_without_changing_policy_or_selecting(job_backend, tmp_path):
    from ca import mapping_run as runs
    source, _ = job_backend
    _attach_lane_native()
    root = tmp_path / "run"
    runs.start_mapping_run(str(source), str(root), _run_layout(width=3.), max_attempts=2)
    runs.advance_mapping_run(str(root), {"type": "inspect", "candidate_ids": [1]}, "Read narrow source span", 0)
    run = runs.advance_mapping_run(str(root), {"type": "draft", "decisions": [{"candidate_id": 1, "action": "include", "reason": "Test minimum width"}]},
                                    "Record width failure", 1)
    assert run["draft_result"]["status"] == "failed" and "minimum" in run["draft_result"]["error"]
    assert run["remaining_attempts"] == 0 and run["layout_hypothesis"] == _run_layout(width=3.)
    assert not (root / "candidate-02").exists()
    final = runs.advance_mapping_run(str(root), {"type": "finish", "candidate_id": None}, "No lane fits this fixed layout", 2)
    assert final["output"]["status"] == "hd_unavailable" and "hd_map" not in final["output"]["artifacts"]


def test_agent_run_retries_another_observed_band_with_unchanged_layout(job_backend, tmp_path):
    from copy import deepcopy
    from ca import mapping_run as runs
    source, _ = job_backend
    _attach_lane_native()
    report = _corridor_report()
    alternative = deepcopy(report["candidates"][0])
    alternative.update({"id": 9, "minimum_support_span_m": 4., "maximum_support_span_m": 4.})
    for section in alternative["sections"]:
        section["left"][1], section["right"][1] = 2., -2.
        section["support_span_m"] = 4.
    report["candidates"].append(alternative)
    jobs.core().propose_road_corridors = lambda *args: json.dumps(report)
    root = tmp_path / "run"
    runs.start_mapping_run(str(source), str(root), _run_layout(width=3.), max_attempts=4)
    runs.advance_mapping_run(str(root), {"type": "inspect", "candidate_ids": [1, 9]}, "Read competing source bands", 0)
    first = runs.advance_mapping_run(str(root), {"type": "draft", "decisions": [{"candidate_id": 1, "action": "include", "reason": "Test narrow band"}]},
                                     "Retain failed hypothesis", 1)
    assert first["draft_result"]["status"] == "failed"
    second = runs.advance_mapping_run(str(root), {"type": "draft", "decisions": [{"candidate_id": 9, "action": "include", "reason": "Try inspected alternate source geometry"}]},
                                      "Another band, identical lane assumptions", 2)
    assert second["draft_result"]["status"] == "audited_draft" and second["remaining_attempts"] == 0
    assert second["layout_hypothesis"] == _run_layout(width=3.)
    assert not (root / "candidate-02").exists() and (root / "candidate-04").exists()
    final = runs.advance_mapping_run(str(root), {"type": "finish", "candidate_id": 4}, "Deliver retained alternate draft", 3)
    assert final["output"]["candidate_id"] == 4 and jobs.inspect_mapping_job(str(root))["selected"] is None


@pytest.mark.parametrize("after_lane", [False, True])
def test_agent_run_resumes_interrupted_stage_without_spending_attempts_twice(job_backend, tmp_path, monkeypatch, after_lane):
    from ca import mapping_run as runs
    source, _ = job_backend
    _attach_lane_native()
    root = tmp_path / "run"
    runs.start_mapping_run(str(source), str(root), _run_layout(), max_attempts=2)
    runs.advance_mapping_run(str(root), {"type": "inspect", "candidate_ids": [1]}, "Read candidate evidence", 0)
    original = jobs.generate_mapping_corridor_lanes
    def interrupted(*args, **kwargs):
        if after_lane:
            original(*args, **kwargs)
        raise KeyboardInterrupt()
    monkeypatch.setattr(jobs, "generate_mapping_corridor_lanes", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runs.advance_mapping_run(str(root), {"type": "draft", "decisions": [{"candidate_id": 1, "action": "include", "reason": "Keep source"}]},
                                 "Draft and continue after interruption", 1)
    assert runs.inspect_mapping_run(str(root))["status"] == "interrupted"
    assert not (root / ".mapping-run-lock").exists()
    monkeypatch.setattr(jobs, "generate_mapping_corridor_lanes", original)
    run = runs.advance_mapping_run(str(root), {"type": "resume"}, "Continue retained stages", 2)
    assert run["draft_result"]["status"] == "audited_draft" and run["remaining_attempts"] == 0
    assert len(jobs.inspect_mapping_job(str(root))["attempts"]) == 2
    assert [a["id"] for a in jobs.inspect_mapping_job(str(root))["attempts"]] == [1, 2]


def test_agent_run_validates_layout_before_raw_processing(job_backend, tmp_path):
    from ca import mapping_run as runs
    source, _ = job_backend
    bad = _run_layout()
    bad["lanes"][0]["direction"] = "infer"
    with pytest.raises(ValueError):
        runs.start_mapping_run(str(source), str(tmp_path / "run"), bad)
    assert not (tmp_path / "run").exists()


def test_agent_refinement_reinspects_reused_ids_and_preserves_prior_drafts(job_backend, tmp_path):
    from ca import mapping_run as runs
    source, _ = job_backend
    _attach_lane_native()
    root = tmp_path / "run"
    runs.start_mapping_run(str(source), str(root), _run_layout())
    runs.advance_mapping_run(str(root), {"type": "inspect", "candidate_ids": [1]}, "Read original proposal", 0)
    runs.advance_mapping_run(str(root), {"type": "draft", "decisions": [{"candidate_id": 1, "action": "include", "reason": "Original draft"}]}, "Keep original draft", 1)
    original = jobs.inspect_mapping_job(str(root))
    files = [original["corridor_proposal"]["file"], *original["attempts"][0]["files"].values(), *original["attempts"][1]["files"].values()]
    snapshots = {f["path"]: Path(f["path"]).read_bytes() for f in files}
    def refined(cloud, trajectory, options):
        report = _corridor_report()
        report["protocol"] = {"options": json.loads(options)}
        candidate = report["candidates"][0]
        candidate.update(from_m=2., to_m=8.)
        candidate["sections"] = candidate["sections"][1:-1]
        return json.dumps(report)
    jobs.core().propose_road_corridors = refined
    run = runs.advance_mapping_run(str(root), {"type": "refine", "association": "trajectory_containing"}, "Resolve off-path band overlap", 2)
    assert run["refine_result"]["status"] == "ready" and run["remaining_attempts"] == 4
    assert run["reviewed_candidates"] == [] and run["layout_hypothesis"] == _run_layout()
    choices = [{"candidate_id": 1, "action": "include", "reason": "Reused ID, different original stations"}]
    with pytest.raises(ValueError, match="inspect candidates"):
        runs.advance_mapping_run(str(root), {"type": "draft", "decisions": choices}, "Old receipt cannot authorize new geometry", 3)
    with pytest.raises(ValueError, match="already attempted"):
        runs.advance_mapping_run(str(root), {"type": "refine", "association": "trajectory_containing"}, "Do not repeat refinement", 3)
    run = runs.advance_mapping_run(str(root), {"type": "inspect", "candidate_ids": [1]}, "Read refined geometry", 3)
    assert run["observation"][0]["from_m"] == 2.
    run = runs.advance_mapping_run(str(root), {"type": "draft", "decisions": choices}, "Generate refined replacement", 4)
    assert run["draft_result"]["status"] == "audited_draft" and run["remaining_attempts"] == 2
    assert all(Path(path).read_bytes() == content for path, content in snapshots.items())
    assert jobs.inspect_mapping_geometry(str(root), 1)["segments"][0]["from_m"] == 0.
    final = runs.advance_mapping_run(str(root), {"type": "finish", "candidate_id": 2}, "Retain earlier audited draft if preferred", 5)
    assert final["output"]["candidate_id"] == 2
    assert jobs.inspect_mapping_job(str(root))["selected"] is None


@pytest.mark.parametrize("after_processing", [False, True])
def test_agent_refinement_failure_or_interruption_keeps_original_source(job_backend, tmp_path, monkeypatch, after_processing):
    from ca import mapping_run as runs
    source, _ = job_backend
    root = tmp_path / "run"
    runs.start_mapping_run(str(source), str(root), _run_layout())
    original = jobs.inspect_mapping_job(str(root))["corridor_proposal"]
    if after_processing:
        def refined(cloud, trajectory, options):
            report = _corridor_report()
            report["protocol"] = {"options": json.loads(options)}
            return json.dumps(report)
        jobs.core().propose_road_corridors = refined
        process = jobs._refine_mapping_corridors
        def interrupted(*args):
            process(*args)
            raise KeyboardInterrupt()
        monkeypatch.setattr(jobs, "_refine_mapping_corridors", interrupted)
        with pytest.raises(KeyboardInterrupt):
            runs.advance_mapping_run(str(root), {"type": "refine", "association": "trajectory_containing"}, "Retain completed extraction", 0)
        monkeypatch.setattr(jobs, "_refine_mapping_corridors", process)
        jobs.core().propose_road_corridors = lambda *args: pytest.fail("Completed extraction must not run twice")
        run = runs.advance_mapping_run(str(root), {"type": "resume"}, "Resume saved refinement", 1)
        assert run["refine_result"]["cached"] is True and run["reviewed_candidates"] == []
    else:
        def fail(*args):
            raise ValueError("Source extraction failed")
        jobs.core().propose_road_corridors = fail
        action = tmp_path / "refine.json"
        action.write_text(json.dumps({"type": "refine", "association": "trajectory_containing"}))
        result = CliRunner().invoke(app, ["mapping-run-advance", str(root), "--action", str(action), "--revision", "0", "--reason", "Retain failed extraction"])
        assert result.exit_code == 1
        run = runs.inspect_mapping_run(str(root))
        assert jobs.inspect_mapping_job(str(root))["corridor_proposal"] == original
        assert run["corridor_refinement"]["status"] == "failed"
    assert run["status"] == "needs_agent" and run["remaining_attempts"] == 6
    jobs._verify(original["file"])
    assert not (root / ".mapping-run-lock").exists()


def _corridor_report():
    return {"schema": "cloudanalyzer.corridor_proposals.v1", "coordinate_frame": "input_metres", "protocol": {},
        "trajectory_length_m": 10., "with_candidate_station_length_m": 10., "without_candidate_station_length_m": 0.,
        "trajectory_covered_station_length_m": 10., "curb_bounded_station_length_m": 0., "ambiguous_station_length_m": 0.,
        "sampled_sections": 6, "evaluated_sections": 6, "sections_with_bands": 6, "multiple_band_sections": 0,
        "profile_queried_points": 50, "interval_support_samples": 15, "unstable_heading_sections": 0,
        "unanchored_sections": 0, "level_mismatch_bands": 0, "limited": False,
        "road_semantics_inferred": False, "deployment_ready": False, "warnings": [], "deferred_intervals": [], "ambiguous_intervals": [],
        "candidates": [{"id": 1, "from_m": 0., "to_m": 10., "minimum_support_span_m": 2., "maximum_support_span_m": 2.,
            "paired_curb_sections": 0, "curb_width_range_m": None, "review_required": True,
            "sections": [{"station_m": k * 2., "center": [k * 2., 0., 1.], "left": [k * 2., 1., 1.],
                "right": [k * 2., -1., 1.], "left_evidence": "support_gap", "right_evidence": "support_gap"} for k in range(6)]}]}


@pytest.fixture
def job_backend(tmp_path, monkeypatch):
    extension = tmp_path / "native.bin"
    extension.write_bytes(b"fixed native implementation")
    support = {"samples": 3, "supported": 2, "height_mismatches": 1, "insufficient_returns": 0,
               "fraction": 2 / 3, "start_supported": True, "end_supported": False}
    audit = {"quality": {"lanes": [{"lane": 3, "needs_review": True, "center": support, "left": support, "right": support}],
                         "low_support_lanes": [3], "omitted_lanes": [], "malformed_lanes": [],
                         "limited": False, "sampled_points": 9, "warnings": [],
                         "sampling_step_m": 0.5, "ground_radius_m": 0.75,
                         "ground_height_tolerance_m": 0.3, "minimum_support_fraction": 0.9,
                         "sample_budget": 100000}, "validation": {"issues": [], "counts": {"errors": 0}}}
    module = SimpleNamespace(__file__=str(extension), __version__="test",
                             audit_vector_map_quality_details=lambda *args: json.dumps(audit),
                             audit_vector_map_ground_consensus_details=lambda *args: json.dumps(audit),
                             propose_road_corridors=lambda *args: json.dumps(_corridor_report()))
    monkeypatch.setattr(jobs, "core", lambda: module)

    def normalize_geometry(path):
        return json.dumps({"map_json": Path(path).read_text(), "report": {
            "validation": {"counts": {"errors": 0}, "issues": []}, "import_issues": [], "georeference": None}})
    module.edit_vector_map_relations = normalize_geometry

    def odometry(source, out, **kwargs):
        root = Path(out); root.mkdir()
        trajectory = root / 'trajectory.tum'
        trajectory.write_text('0 0 0 1 0 0 0 1\n1 10 0 1 0 0 0 1\n')
        return {"scans": out, "trajectory": str(trajectory), "gravity": None, "frames": 4, "path_length_m": 10}

    def fix(folder, out, **kwargs):
        root = Path(out)
        root.mkdir()
        files = {}
        for key, name in [("map", "map.ply"), ("kitti", "poses.txt"), ("g2o", "map.g2o")]:
            (root / name).write_text(key)
            files[key] = str(root / name)
        return {"map_points": 100, "outputs": files}

    @wraps(jobs.build_vector_map)
    def build(cloud, trajectory, out_dir, **kwargs):
        if kwargs.get("forward_lanes") == 8:
            raise ValueError("insufficient ground for this road hypothesis")
        root = Path(out_dir)
        root.mkdir()
        files = {}
        for key, name in [("editable_map", "vector_map.json"), ("map", "map.osm"),
                          ("projector", "projector.yaml"), ("report", "report.json")]:
            (root / name).write_text(json.dumps({"options": kwargs}) if key == "report" else key)
            files[key] = str(root / name)
        return {"files": files, "extraction": {"lanes": 1, "trajectory_length": 10, "generated_length": 10}, "autoware_issues": []}

    monkeypatch.setattr(jobs, "odometry", odometry)
    monkeypatch.setattr(jobs, "fix_session", fix)
    monkeypatch.setattr(jobs, "build_vector_map", build)
    source = tmp_path / "drive.mcap"
    source.write_bytes(b"raw immutable recording")
    return source, audit


def test_agent_choices_persist_quality_holds_failed_trials_and_budget(job_backend, tmp_path):
    source, _ = job_backend
    root = tmp_path / "job"
    started = jobs.start_mapping_job(str(source), str(root), max_attempts=2)
    assert started["status"] == "pointcloud_ready"
    assert started["next_actions"] == ["propose_mapping_corridors", "generate_mapping_candidate"]
    first = jobs.generate_mapping_candidate(str(root), OPTIONS, "Test one explicit lane hypothesis")
    assert first["attempts"][0]["quality"]["needs_review"] == [3]
    selected = jobs.select_mapping_candidate(str(root), 1, "Retain the draft while source issues remain unresolved")
    assert selected["selected"]["source_quality_passed"] is False
    assert selected["selected"]["deployment_ready"] is False
    original = Path(selected["selected"]["files"]["map"]["path"]).read_bytes()
    failed = jobs.generate_mapping_candidate(str(root), {**OPTIONS, "forward_lanes": 8}, "Test whether wider extent has supporting ground")
    assert failed["attempts"][1]["status"] == "failed"
    assert "insufficient ground" in failed["attempts"][1]["error"]
    assert failed["selected"] == selected["selected"]
    assert Path(selected["selected"]["files"]["map"]["path"]).read_bytes() == original
    assert not (root / "candidate-02").exists()
    assert failed["remaining_attempts"] == 0
    with pytest.raises(ValueError, match="budget"):
        jobs.generate_mapping_candidate(str(root), OPTIONS, "Third attempt")
    with pytest.raises(ValueError, match="audited"):
        jobs.select_mapping_candidate(str(root), 2, "Cannot select a failed attempt")
    assert jobs.inspect_mapping_job(str(root))["selected"] == selected["selected"]


def test_changed_inputs_or_candidates_cannot_be_selected_or_reused(job_backend, tmp_path):
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    generated = jobs.generate_mapping_candidate(str(root), OPTIONS, "Initial hypothesis")
    artifact = Path(generated["attempts"][0]["files"]["map"]["path"])
    artifact.write_text("edited outside this job")
    with pytest.raises(ValueError, match="changed"):
        jobs.select_mapping_candidate(str(root), 1, "Stale artifact")
    source.write_bytes(b"different recording")
    with pytest.raises(ValueError, match="changed"):
        jobs.generate_mapping_candidate(str(root), OPTIONS, "Stale input")
    assert len(jobs.inspect_mapping_job(str(root))["attempts"]) == 1


def test_ground_estimator_disagreement_cannot_be_selected_as_source_pass(job_backend, tmp_path):
    source, audit = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    alternate = json.loads(json.dumps(audit))
    # Legacy source passes while the alternate layer has endpoint/support holds.
    audit["quality"]["low_support_lanes"] = []
    for trace in ("center", "left", "right"):
        audit["quality"]["lanes"][0][trace] = {"samples": 3, "supported": 3,
            "height_mismatches": 0, "insufficient_returns": 0, "fraction": 1,
            "start_supported": True, "end_supported": True}
    alternate["quality"]["ground_estimator"] = {"model": "lowest_supported_layer"}
    jobs.core().audit_vector_map_ground_consensus_details = lambda *args: json.dumps(alternate)
    generated = jobs.generate_mapping_candidate(str(root), OPTIONS, "Compare source layers")
    assert generated["attempts"][0]["quality"]["needs_review"] == []
    diagnosis = jobs.diagnose_mapping_candidate(str(root), 1)
    assert diagnosis["ground_consensus"]["editable"]["protocol"]["ground_estimator"]["model"] == "lowest_supported_layer"
    assert any("estimators disagree" in reason for reason in diagnosis["investigations"])
    selected = jobs.select_mapping_candidate(str(root), 1, "Keep geometry as a held review draft")
    assert selected["selected"]["source_quality_passed"] is False


def test_incomplete_ground_consensus_cannot_be_selected(job_backend, tmp_path):
    source, audit = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    alternate = json.loads(json.dumps(audit))
    alternate["quality"]["limited"] = True
    jobs.core().audit_vector_map_ground_consensus_details = lambda *args: json.dumps(alternate)
    jobs.generate_mapping_candidate(str(root), OPTIONS, "Partial alternate evidence")
    with pytest.raises(ValueError, match="complete"):
        jobs.select_mapping_candidate(str(root), 1, "Partial evidence must remain held")


def test_corridor_proposals_cache_inputs_and_preserve_hd_budget_and_selection(job_backend, tmp_path):
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root), max_attempts=2)
    jobs.generate_mapping_candidate(str(root), OPTIONS, "Keep an explicit draft")
    selected = jobs.select_mapping_candidate(str(root), 1, "Unconfirmed source and traffic semantics")
    module = jobs.core()
    calls = []
    def propose(cloud, trajectory, options):
        calls.append((cloud, trajectory, json.loads(options)))
        return json.dumps(_corridor_report())
    module.propose_road_corridors = propose
    proposed = jobs.propose_mapping_corridors(str(root))
    assert proposed["status"] == "ready" and proposed["remaining_attempts"] == 1
    assert proposed["summary"]["curb_bounded_candidates"] == 0
    assert proposed["summary"]["coordinate_frame"] == "local_slam_metres"
    cached = jobs.propose_mapping_corridors(str(root))
    assert cached["cached"] is True and len(calls) == 1
    assert calls[0][2] == {"search_radius_m": 8.0}
    after = jobs.inspect_mapping_job(str(root))
    assert after["selected"] == selected["selected"] and after["attempts"] == selected["attempts"]
    assert "inspect_mapping_corridors" in after["next_actions"]
    with pytest.raises(ValueError, match="frozen"):
        jobs.propose_mapping_corridors(str(root), 9)
    artifact = Path(proposed["file"]["path"])
    artifact.write_text("changed proposal geometry")
    with pytest.raises(ValueError, match="changed"):
        jobs.propose_mapping_corridors(str(root))
    with pytest.raises(ValueError, match="changed"):
        jobs.inspect_mapping_corridors(str(root))


def test_corridor_inspection_pages_and_bounds_geometry_without_a_native_core(job_backend, tmp_path, monkeypatch):
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    report = _corridor_report()
    sample = report["candidates"][0]
    report["candidates"] = [{**sample, "id": i + 1, "sections": sample["sections"] * 30} for i in range(20)]
    jobs.core().propose_road_corridors = lambda *args: json.dumps(report)
    jobs.propose_mapping_corridors(str(root))
    original = (root / "job.json").read_bytes()
    monkeypatch.setattr(jobs, "core", lambda: None)
    index = jobs.inspect_mapping_corridors(str(root))
    assert len(index["candidate_index"]) == 16 and index["next_offset"] == 16
    page = jobs.inspect_mapping_corridors(str(root), offset=16)
    assert len(page["candidate_index"]) == 4 and page["next_offset"] is None
    geometry = jobs.inspect_mapping_corridors(str(root), candidate_id=17)["candidate"]
    assert geometry["id"] == 17 and geometry["section_preview_limited"] is True
    assert geometry["total_sections"] == 180 and len(geometry["sections"]) == 128
    assert (root / "job.json").read_bytes() == original
    with pytest.raises(ValueError, match="unknown"):
        jobs.inspect_mapping_corridors(str(root), candidate_id=21)
    source.write_bytes(b"changed raw recording")
    with pytest.raises(ValueError, match="changed"):
        jobs.inspect_mapping_corridors(str(root))


def test_failed_corridor_stage_retains_error_without_spending_attempts(job_backend, tmp_path):
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    def failed(*args):
        raise ValueError("source frame has invalid coordinates")
    jobs.core().propose_road_corridors = failed
    result = jobs.propose_mapping_corridors(str(root))
    assert result["status"] == "failed" and result["remaining_attempts"] == 4
    assert result["error"] == "source frame has invalid coordinates"
    assert jobs.inspect_mapping_corridors(str(root))["status"] == "failed"
    assert jobs.inspect_mapping_job(str(root))["status"] == "pointcloud_ready"
    assert not (root / ".mapping-lock").exists()
    with pytest.raises(RuntimeError, match="did not finish"):
        jobs.propose_mapping_corridors(str(root))


def test_corridor_stage_rechecks_inputs_after_native_processing(job_backend, tmp_path):
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    def changed(*args):
        source.write_bytes(b"recording changed during processing")
        return json.dumps(_corridor_report())
    jobs.core().propose_road_corridors = changed
    result = jobs.propose_mapping_corridors(str(root))
    assert result["status"] == "failed" and "changed" in result["error"]
    assert "file" not in result and result["remaining_attempts"] == 4


def test_corridor_cli_and_validation_need_no_lane_assumptions(job_backend, tmp_path):
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    runner = CliRunner()
    proposed = runner.invoke(app, ["mapping-corridors", str(root)])
    assert proposed.exit_code == 0, proposed.output
    assert json.loads(proposed.stdout)["summary"]["road_semantics_inferred"] is False
    inspected = runner.invoke(app, ["mapping-corridors-inspect", str(root), "--candidate", "1"])
    assert inspected.exit_code == 0 and json.loads(inspected.stdout)["candidate"]["curb_width_range_m"] is None
    for reach in (0, 21, float("nan")):
        with pytest.raises(ValueError, match="search_radius"):
            jobs.propose_mapping_corridors(str(root), reach)
    for kwargs in ({"offset": -1}, {"candidate_id": True}, {"candidate_id": 1, "offset": 16}):
        with pytest.raises(ValueError):
            jobs.inspect_mapping_corridors(str(root), **kwargs)


def test_single_profile_curb_width_hints_do_not_complete_a_corridor_width(job_backend, tmp_path):
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    report = _corridor_report()
    band = {**report["candidates"][0]["sections"][2], "support_span_m": 2., "path_level_supported": True,
            "left_evidence": "curb_profile", "right_evidence": "curb_profile"}
    report["profiles"] = [{"station_m": 4., "bands": [band]}]
    jobs.core().propose_road_corridors = lambda *args: json.dumps(report)
    result = jobs.propose_mapping_corridors(str(root))
    assert result["summary"]["paired_curb_profile_bands"] == 1
    assert result["summary"]["curb_bounded_candidates"] == 0
    index = jobs.inspect_mapping_corridors(str(root))
    assert index["curb_width_profile_hints"][0]["support_span_m"] == 2.
    assert index["curb_width_profile_hints"][0]["review_required"] is True
    assert index["candidate_index"][0]["curb_width_range_m"] is None
    assert jobs.inspect_mapping_job(str(root))["selected"] is None


def test_geometry_draft_preserves_chosen_source_curves_full_extent_and_selected_lane_map(job_backend, tmp_path, monkeypatch):
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root), max_attempts=2)
    jobs.generate_mapping_candidate(str(root), OPTIONS, "Retain the earlier lane hypothesis")
    prior = jobs.select_mapping_candidate(str(root), 1, "Earlier review draft")["selected"]
    jobs.propose_mapping_corridors(str(root))
    choices = [{"candidate_id": 1, "action": "include", "reason": "Inspect source curves", "from_m": 0., "to_m": 4.},
               {"candidate_id": 1, "action": "defer", "reason": "Width unresolved", "from_m": 4., "to_m": 6.},
               {"candidate_id": 1, "action": "include", "reason": "Separate later source fragment", "from_m": 8., "to_m": 10.}]
    result = jobs.generate_mapping_geometry(str(root), choices, "Use actual candidate geometry with unresolved lanes")
    assert result["selected"] == prior and result["remaining_attempts"] == 0
    attempt = result["attempts"][1]
    assert attempt["status"] == "geometry_draft", attempt
    ir = json.loads(Path(attempt["files"]["editable_map"]["path"]).read_text())
    assert ir["lanes"] == [] and ir["roads"] == [] and len(ir["boundaries"]) == 6
    assert ir["boundaries"][0]["geometry"] == [[0., 0., 1.], [2., 0., 1.], [4., 0., 1.]]
    assert all(b["kind"] == {"type": "other"} for b in ir["boundaries"])
    assert ir["metadata"]["attributes"]["cloudanalyzer:proposal_sha256"] == result["corridor_proposal"]["file"]["sha256"]
    assert ir["metadata"]["attributes"]["cloudanalyzer:pointcloud_sha256"] == result["pointcloud"]["files"]["map"]["sha256"]
    assert not list((root / "geometry-02").glob("*.osm"))
    monkeypatch.setattr(jobs, "core", lambda: None)
    inspected = jobs.inspect_mapping_geometry(str(root), 2)
    assert inspected["summary"]["included_station_length_m"] == 6.
    assert inspected["summary"]["unresolved_station_length_m"] == 4.
    assert inspected["summary"]["included_station_fraction"] == .6
    assert inspected["summary"]["meets_requested_station_extent"] is False
    assert inspected["summary"]["lane_count"] is None and inspected["summary"]["source_quality_passed"] is False
    intervals = inspected["station_disposition"]
    assert [i["status"] for i in intervals] == ["included_geometry", "agent_deferred", "not_reviewed", "included_geometry"]
    assert sum(i["to_m"] - i["from_m"] for i in intervals) == 10.
    assert inspected["editing"]["file"].endswith("vector_map.json")
    report = Path(attempt["files"]["report"]["path"])
    report.write_text(report.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        jobs.inspect_mapping_geometry(str(root), 2)


def test_geometry_decisions_reject_overlapping_bands_extrapolation_and_implicit_lanes(job_backend, tmp_path):
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    report = _corridor_report()
    report["candidates"].append({**report["candidates"][0], "id": 2})
    jobs.core().propose_road_corridors = lambda *args: json.dumps(report)
    jobs.propose_mapping_corridors(str(root))
    included = {"candidate_id": 1, "action": "include", "reason": "Inspect the observed band"}
    invalid = [[], [{**included, "candidate_id": True}], [{**included, "from_m": 1.}],
        [{**included, "to_m": 12.}], [{**included, "lane_width": 3.5}], [{**included, "reason": " "}],
        [included, {**included, "candidate_id": 2}], [included, {**included, "action": "defer"}]]
    for choices in invalid:
        with pytest.raises(ValueError):
            jobs.generate_mapping_geometry(str(root), choices, "No unobserved geometry")
    assert jobs.inspect_mapping_job(str(root))["attempts"] == []
    result = jobs.generate_mapping_geometry(str(root), [included], "Geometry alone does not establish lanes")
    assert result["attempts"][0]["status"] == "geometry_draft"
    with pytest.raises(ValueError, match="audited draft"):
        jobs.select_mapping_candidate(str(root), 1, "Empty lanes must not pass")
    with pytest.raises(ValueError, match="audited draft"):
        jobs.diagnose_mapping_candidate(str(root), 1)


def test_geometry_failures_are_retained_and_atomic_without_spending_extra_attempts(job_backend, tmp_path):
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root), max_attempts=1)
    jobs.propose_mapping_corridors(str(root))
    def failed(path):
        raise ValueError("native map import failed")
    jobs.core().edit_vector_map_relations = failed
    choices = [{"candidate_id": 1, "action": "include", "reason": "Keep source geometry"}]
    result = jobs.generate_mapping_geometry(str(root), choices, "Try the saved source band")
    assert result["attempts"][0]["status"] == "failed"
    assert result["attempts"][0]["error"] == "native map import failed"
    assert result["remaining_attempts"] == 0 and result["selected"] is None
    assert jobs.inspect_mapping_geometry(str(root), 1)["status"] == "failed"
    assert not (root / "geometry-01").exists() and not (root / ".mapping-lock").exists()
    with pytest.raises(ValueError, match="budget exhausted"):
        jobs.generate_mapping_geometry(str(root), choices, "No automatic retry")


@pytest.mark.parametrize("mutate_proposal", [False, True])
def test_geometry_rechecks_proposal_and_inputs_before_publishing(job_backend, tmp_path, mutate_proposal):
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    jobs.propose_mapping_corridors(str(root))
    normalizer = jobs.core().edit_vector_map_relations
    def changed(path):
        answer = normalizer(path)
        if mutate_proposal:
            proposal = root / "corridor-proposals.json"
            proposal.write_bytes(proposal.read_bytes() + b" ")
        else:
            source.write_bytes(b"changed while normalizing")
        return answer
    jobs.core().edit_vector_map_relations = changed
    result = jobs.generate_mapping_geometry(str(root), [{"candidate_id": 1, "action": "include", "reason": "Keep the band"}], "Reject stale processing")
    assert result["attempts"][0]["status"] == "failed" and "changed" in result["attempts"][0]["error"]
    assert not (root / "geometry-01").exists()


def test_geometry_preserves_source_deferred_and_ambiguous_intervals(job_backend, tmp_path):
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    report = _corridor_report()
    report["candidates"][0].update({"from_m": 2., "to_m": 8., "sections": report["candidates"][0]["sections"][1:5]})
    report["deferred_intervals"] = [{"from_m": 0., "to_m": 2., "reason": "missing_coherent_surface"},
                                    {"from_m": 8., "to_m": 10., "reason": "source_gap"}]
    report["ambiguous_intervals"] = [{"from_m": 4., "to_m": 6., "reason": "branching_bands"}]
    jobs.core().propose_road_corridors = lambda *args: json.dumps(report)
    jobs.propose_mapping_corridors(str(root))
    jobs.generate_mapping_geometry(str(root), [{"candidate_id": 1, "action": "include", "reason": "Separate retained band still needs review"}], "Preserve unresolved context")
    intervals = jobs.inspect_mapping_geometry(str(root), 1)["station_disposition"]
    assert intervals[0]["status"] == "source_deferred" and intervals[0]["source_reasons"] == ["missing_coherent_surface"]
    assert intervals[-1]["status"] == "source_deferred" and intervals[-1]["source_reasons"] == ["source_gap"]
    assert any(i["status"] == "included_geometry" and i["source_ambiguous"] for i in intervals)
    assert sum(i["to_m"] - i["from_m"] for i in intervals) == 10.


def test_geometry_cli_preserves_source_and_native_ir_reload(job_backend, tmp_path, monkeypatch):
    native = pytest.importorskip("cloudanalyzer_core")
    if not hasattr(native, "edit_vector_map_relations"):
        pytest.skip("installed native core predates IR normalization")
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    jobs.propose_mapping_corridors(str(root))
    jobs.core().edit_vector_map_relations = native.edit_vector_map_relations
    choices = tmp_path / "decisions.json"
    choices.write_text(json.dumps([{"candidate_id": 1, "action": "include", "reason": "Use source curves only"}]))
    result = CliRunner().invoke(app, ["mapping-geometry", str(root), "--decisions", str(choices), "--reason", "No lane assumptions"])
    assert result.exit_code == 0, result.output
    attempt = json.loads(result.stdout)["attempts"][0]
    assert attempt["status"] == "geometry_draft", attempt
    normalized = json.loads(native.edit_vector_map_relations(attempt["files"]["editable_map"]["path"]))
    ir = json.loads(normalized["map_json"])
    assert len(ir["boundaries"]) == 3 and not ir.get("lanes")
    assert ir["boundaries"][0]["attributes"]["cloudanalyzer:role"] == "source_center"
    assert normalized["report"]["validation"]["counts"]["errors"] == 0
    monkeypatch.setattr(jobs, "core", lambda: None)
    inspected = CliRunner().invoke(app, ["mapping-geometry-inspect", str(root), "--candidate", "1"])
    assert inspected.exit_code == 0 and json.loads(inspected.stdout)["summary"]["deployment_ready"] is False


def test_lane_layout_validation_never_spends_attempts_or_infers_semantics(job_backend, tmp_path):
    from copy import deepcopy
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    jobs.propose_mapping_corridors(str(root))
    jobs.generate_mapping_geometry(str(root), [{"candidate_id": 1, "action": "include", "reason": "Retain source"}], "Unknown lanes")
    good = _lane_specs()
    invalid = [[], [{**good[0], "center_curve_id": True}], [{**good[0], "center_curve_id": 2}], good + good,
               [{**good[0], "speed_limit_kmh": None}], [{**good[0], "reason": " "}]]
    for field, value in [("kind", "walkway"), ("direction", "both"), ("one_way", 1), ("fraction", .5), ("minimum_width_m", float("nan"))]:
        bad = deepcopy(good)
        bad[0]["lanes"][0][field] = value
        invalid.append(bad)
    for specs in invalid:
        with pytest.raises(ValueError):
            jobs.generate_mapping_corridor_lanes(str(root), 1, specs, "source_span_hypothesis", "No hidden assumptions")
    with pytest.raises(ValueError, match="boundary_policy"):
        jobs.generate_mapping_corridor_lanes(str(root), 1, good, "observed_road_width", "Not confirmed road edges")
    assert len(jobs.inspect_mapping_job(str(root))["attempts"]) == 1


def test_lane_width_failure_and_cli_export_preserve_parent_selection_and_shared_budget(job_backend, tmp_path):
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root), max_attempts=4)
    jobs.generate_mapping_candidate(str(root), OPTIONS, "Earlier lane hypothesis")
    selected = jobs.select_mapping_candidate(str(root), 1, "Earlier source-review holds")["selected"]
    jobs.propose_mapping_corridors(str(root))
    jobs.generate_mapping_geometry(str(root), [{"candidate_id": 1, "action": "include", "reason": "Keep reference curves"}], "Preserve observed geometry")
    _attach_lane_native()
    original = (root / "geometry-02" / "vector_map.json").read_bytes()
    failed = jobs.generate_mapping_corridor_lanes(str(root), 2, _lane_specs(width=3.), "source_span_hypothesis", "Reject insufficient span")
    assert failed["attempts"][-1]["status"] == "failed" and "minimum lane width" in failed["attempts"][-1]["error"]
    assert failed["remaining_attempts"] == 1 and failed["selected"] == selected
    assert not (root / "candidate-03").exists()
    specs = tmp_path / "lanes.json"
    specs.write_text(json.dumps(_lane_specs()))
    result = CliRunner().invoke(app, ["mapping-lanes", str(root), "--geometry", "2", "--specs", str(specs),
                                    "--boundary-policy", "source_span_hypothesis", "--reason", "Test source geometry, not certified traffic"])
    assert result.exit_code == 0, result.output
    job = json.loads(result.stdout)
    attempt = job["attempts"][-1]
    assert attempt["status"] == "audited_draft", attempt
    assert job["remaining_attempts"] == 0 and job["selected"] == selected
    assert (root / "geometry-02" / "vector_map.json").read_bytes() == original
    report = json.loads(Path(attempt["files"]["report"]["path"]).read_text())
    assert report["lane_roundtrip_verified"] is True and report["complete_width_resolved"] is False
    assert report["extraction"]["length_measurement"] == "original_input_xy_station_union"
    assert sum(i["to_m"]-i["from_m"] for i in report["station_disposition"]) == 10.
    diagnosis = jobs.diagnose_mapping_candidate(str(root), 4)
    assert any("unverified layout" in i for i in diagnosis["investigations"])
    assert "source_span_hypothesis" in Path(attempt["files"]["map"]["path"]).read_text()
    parent_report = root / "geometry-02" / "report.json"
    parent_report.write_bytes(parent_report.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        jobs.diagnose_mapping_candidate(str(root), 4)
    with pytest.raises(ValueError, match="changed"):
        jobs.select_mapping_candidate(str(root), 4, "Reject stale parent")


@pytest.mark.parametrize("fault", ["roundtrip", "parent", "audit"])
def test_lane_export_rejects_roundtrip_loss_stale_parent_or_invalid_audit_before_publishing(job_backend, tmp_path, fault):
    source, audit = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    jobs.propose_mapping_corridors(str(root))
    jobs.generate_mapping_geometry(str(root), [{"candidate_id": 1, "action": "include", "reason": "Keep source"}], "Unresolved semantics")
    _attach_lane_native()
    real = jobs.core().edit_vector_map_relations
    def changed(path):
        result = json.loads(real(path))
        if fault == "parent":
            parent = root / "geometry-01" / "report.json"
            parent.write_bytes(parent.read_bytes() + b" ")
        elif fault == "roundtrip":
            ir = json.loads(result["map_json"])
            ir["lanes"] = []
            result["map_json"] = json.dumps(ir)
        return json.dumps(result)
    jobs.core().edit_vector_map_relations = changed
    if fault == "audit":
        audit["quality"].pop("low_support_lanes")
    failed = jobs.generate_mapping_corridor_lanes(str(root), 1, _lane_specs(), "source_span_hypothesis", "Verify reload and hashes")
    assert failed["attempts"][-1]["status"] == "failed"
    assert {"parent": "changed", "roundtrip": "retain every", "audit": "low_support_lanes"}[fault] in failed["attempts"][-1]["error"]
    assert not (root / "candidate-02").exists() and not (root / ".mapping-lock").exists()


def test_failed_pointcloud_stage_is_recorded_and_existing_jobs_are_preserved(job_backend, tmp_path, monkeypatch):
    source, _ = job_backend
    def failed(*args, **kwargs):
        raise ValueError("recording contains no usable scans")
    monkeypatch.setattr(jobs, "odometry", failed)
    root = tmp_path / "job"
    report = jobs.start_mapping_job(str(source), str(root))
    assert report["status"] == "pointcloud_failed" and report["next_actions"] == []
    assert "no usable scans" in report["error"]
    with pytest.raises(FileExistsError):
        jobs.start_mapping_job(str(source), str(root))
    with pytest.raises(ValueError, match="point-cloud"):
        jobs.generate_mapping_candidate(str(root), OPTIONS, "Must not skip failed SLAM")
    assert jobs.inspect_mapping_job(str(root))["error"] == report["error"]


@pytest.mark.parametrize("field,value", [("limited", True), ("omitted_lanes", [3]), ("malformed_lanes", [3]), ("lanes", [])])
def test_partial_or_empty_audits_do_not_become_selected_drafts(job_backend, tmp_path, field, value):
    source, audit = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    audit["quality"][field] = value
    jobs.generate_mapping_candidate(str(root), OPTIONS, "Try a draft")
    diagnosis = jobs.diagnose_mapping_candidate(str(root), 1)
    assert diagnosis["editable"]["complete"] is False
    with pytest.raises(ValueError, match="complete"):
        jobs.select_mapping_candidate(str(root), 1, "Incomplete coverage is not a pass")
    assert jobs.inspect_mapping_job(str(root))["selected"] is None


def test_diagnosis_distinguishes_trace_failures_without_processing_or_mutating(job_backend, tmp_path, monkeypatch):
    source, audit = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    audit["quality"]["lanes"][0].update({
        "left": {"samples": 3, "supported": 1, "height_mismatches": 0, "insufficient_returns": 2,
                 "fraction": 1 / 3, "start_supported": False, "end_supported": True},
        # Endpoint holds still matter independently of aggregate fraction.
        "right": {"samples": 10, "supported": 9, "height_mismatches": 0, "insufficient_returns": 1,
                  "fraction": 0.9, "start_supported": False, "end_supported": True},
    })
    audit["validation"]["issues"] = [{"severity": "error", "code": "invalid_geometry"}]
    jobs.generate_mapping_candidate(str(root), OPTIONS, "Diagnose mixed evidence")
    original = (root / "job.json").read_bytes()
    def forbidden():
        raise AssertionError("diagnosis must use saved evidence, not native processing")
    monkeypatch.setattr(jobs, "core", forbidden)
    diagnosis = jobs.diagnose_mapping_candidate(str(root), 1)
    assert diagnosis["editable"]["sample_totals"] == {
        "samples": 16, "supported": 12, "height_mismatches": 1, "insufficient_returns": 3}
    traces = diagnosis["editable"]["lanes"][0]["traces"]
    assert traces["center"]["holds"] == ["below_support_threshold", "unsupported_end"]
    assert traces["left"]["holds"] == ["below_support_threshold", "unsupported_start"]
    assert traces["right"]["holds"] == ["unsupported_start"]
    assert diagnosis["editable"]["complete"] is False
    assert diagnosis["editable_and_reopened_match"] is True
    assert diagnosis["editable"]["problems_available"] is False
    assert diagnosis["editable"]["problems_limited"] is None
    assert diagnosis["deployment_ready"] is False
    assert diagnosis["remaining_attempts"] == 3
    result = CliRunner().invoke(app, ["mapping-diagnose", str(root), "--candidate", "1"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == diagnosis
    assert (root / "job.json").read_bytes() == original


@pytest.mark.parametrize("changed", ["pointcloud", "quality", "map", "report"])
def test_diagnosis_rejects_changed_evidence(job_backend, tmp_path, changed):
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    generated = jobs.generate_mapping_candidate(str(root), OPTIONS, "Freeze the evidence")
    candidate = generated["attempts"][0]
    artifact = (generated["pointcloud"]["files"]["map"] if changed == "pointcloud" else
                candidate["quality_report"] if changed == "quality" else candidate["files"][changed])
    Path(artifact["path"]).write_text("changed after the recorded audit")
    with pytest.raises(ValueError, match="changed"):
        jobs.diagnose_mapping_candidate(str(root), 1)


def test_diagnosis_keeps_reopened_discrepancies_and_import_errors_visible(job_backend, tmp_path):
    source, audit = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    module = jobs.core()
    def audit_each(cloud, path):
        result = json.loads(json.dumps(audit))
        if Path(path).suffix == ".osm":
            result["quality"]["lanes"][0]["right"]["end_supported"] = True
            result["import_issues"] = [{"severity": "error", "code": "invalid_osm"}]
        return json.dumps(result)
    module.audit_vector_map_quality_details = audit_each
    generated = jobs.generate_mapping_candidate(str(root), OPTIONS, "Inspect a reopened discrepancy")
    diagnosis = jobs.diagnose_mapping_candidate(str(root), 1)
    assert diagnosis["editable_and_reopened_match"] is False
    assert diagnosis["editable"]["complete"] is True
    assert diagnosis["reopened_osm"]["complete"] is False
    assert diagnosis["reopened_osm"]["errors"][0]["code"] == "invalid_osm"
    assert any("differences" in text for text in diagnosis["investigations"])
    # Earlier job summaries omitted import errors. Selection must read the
    # verified saved audit, including when opening such a persisted job.
    generated["attempts"][0]["reopened_quality"]["validation_errors"] = []
    (root / "job.json").write_text(json.dumps(generated))
    with pytest.raises(ValueError, match="complete"):
        jobs.select_mapping_candidate(str(root), 1, "Import failure cannot become a selected draft")


def test_cli_records_choices_and_reports_failed_attempts_with_nonzero_exit(job_backend, tmp_path):
    source, _ = job_backend
    root, options = tmp_path / "job", tmp_path / "options.json"
    runner = CliRunner()
    started = runner.invoke(app, ["mapping-start", str(source), "--out", str(root)])
    assert started.exit_code == 0, started.output
    options.write_text(json.dumps({**OPTIONS, "forward_lanes": 8}))
    failed = runner.invoke(app, ["mapping-candidate", str(root), "--options", str(options), "--reason", "Test a road hypothesis"])
    assert failed.exit_code == 1
    assert json.loads(failed.stdout)["attempts"][0]["status"] == "failed"
    options.write_text(json.dumps(OPTIONS))
    draft = runner.invoke(app, ["mapping-candidate", str(root), "--options", str(options), "--reason", "Retain a narrower explicit hypothesis"])
    assert draft.exit_code == 0, draft.output
    chosen = runner.invoke(app, ["mapping-select", str(root), "--candidate", "2", "--reason", "Keep this draft and its unresolved issues"])
    assert chosen.exit_code == 0, chosen.output
    read = runner.invoke(app, ["mapping-status", str(root)])
    assert json.loads(read.stdout)["selected"]["candidate_id"] == 2


def test_missing_assumptions_and_busy_jobs_do_not_spend_attempts(job_backend, tmp_path):
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    with pytest.raises(ValueError, match="explicit"):
        jobs.generate_mapping_candidate(str(root), {"forward_lanes": 1}, "Missing traffic assumptions")
    with pytest.raises(ValueError, match="supported"):
        jobs.generate_mapping_candidate(str(root), {**OPTIONS, "existing_map": "other.json"}, "Different map")
    with pytest.raises(ValueError):
        jobs.generate_mapping_candidate(str(root), {**OPTIONS, "lane_width": float("nan")}, "Invalid width")
    (root / ".mapping-lock").write_text("another writer")
    with pytest.raises(RuntimeError, match="busy"):
        jobs.generate_mapping_candidate(str(root), OPTIONS, "Concurrent attempt")
    assert jobs.inspect_mapping_job(str(root))["attempts"] == []


def test_native_candidate_contract_audits_saved_osm_and_preserves_failed_trials(job_backend, tmp_path, monkeypatch):
    native = pytest.importorskip("cloudanalyzer_core")
    if not hasattr(native, "propose_road_corridors"):
        pytest.skip("installed core predates detailed source audit")
    from ca.vector_map import build_vector_map
    source, _ = job_backend
    monkeypatch.setattr(jobs, "core", lambda: native)
    monkeypatch.setattr(jobs, "build_vector_map", build_vector_map)

    def fixed(folder, out, **kwargs):
        root = Path(out)
        root.mkdir()
        cloud = root / "map.xyz"
        cloud.write_text("".join(f"{x/4} {y/4} 2\n" for x in range(-4, 85) for y in range(-20, 21)))
        trajectory, graph = root / "poses.txt", root / "map.g2o"
        trajectory.write_text("1 0 0 0 0 1 0 0 0 0 1 20\n1 0 0 20 0 1 0 0 0 0 1 20\n")
        graph.write_text("not used by the native HD builder")
        return {"map_points": 3649, "outputs": {"map": str(cloud), "kitti": str(trajectory), "g2o": str(graph)}}
    monkeypatch.setattr(jobs, "fix_session", fixed)
    root = tmp_path / "native-job"
    jobs.start_mapping_job(str(source), str(root))
    proposals = jobs.propose_mapping_corridors(str(root))
    assert proposals["status"] == "ready", proposals
    assert proposals["summary"]["trajectory_covered_station_length_m"] == 20.
    assert proposals["summary"]["curb_bounded_candidates"] == 0
    assert proposals["remaining_attempts"] == 4
    assert jobs.inspect_mapping_corridors(str(root), candidate_id=1)["candidate"]["curb_width_range_m"] is None
    generated = jobs.generate_mapping_candidate(str(root), OPTIONS, "A straight, source-supported single-lane hypothesis")
    candidate = generated["attempts"][0]
    assert candidate["status"] == "audited_draft", candidate
    assert candidate["quality"]["needs_review"] == []
    assert candidate["reopened_quality"] == candidate["quality"]
    diagnosis = jobs.diagnose_mapping_candidate(str(root), 1)
    assert diagnosis["editable_and_reopened_match"] is True
    assert diagnosis["editable"]["complete"] is True
    assert diagnosis["editable"]["problems_available"] is True
    assert diagnosis["editable"]["problems"] == []
    assert diagnosis["editable"]["problems_limited"] is False
    assert diagnosis["editable"]["sample_totals"]["height_mismatches"] == 0
    assert diagnosis["editable"]["sample_totals"]["insufficient_returns"] == 0
    selected = jobs.select_mapping_candidate(str(root), 1, "Complete structural/source checks; road semantics still unconfirmed")
    assert selected["selected"]["source_quality_passed"] is True
    assert selected["selected"]["deployment_ready"] is False
    failed = jobs.generate_mapping_candidate(str(root), {**OPTIONS, "lane_width": 1.0}, "Check a width incompatible with the boundary search margin")
    assert failed["attempts"][-1]["status"] == "failed"
    assert failed["selected"] == selected["selected"]
    assert not (root / "candidate-02").exists()

    if hasattr(native, "build_corridor_lanes"):
        jobs.propose_mapping_corridors(str(root))
        jobs.generate_mapping_geometry(str(root), [{"candidate_id": 1, "action": "include", "reason": "Keep actual source edges"}], "Explicit geometry before lane hypotheses")
        specs = _lane_specs(width=1.)
        specs[0]["lanes"] = [
            {"direction": "backward", "kind": "driving", "one_way": True, "fraction": .5, "minimum_width_m": 1.},
            {"direction": "forward", "kind": "driving", "one_way": True, "fraction": .5, "minimum_width_m": 1.}]
        lanes = jobs.generate_mapping_corridor_lanes(str(root), 3, specs, "source_span_hypothesis", "Test a two-direction hypothesis without changing outer curves")
        assert lanes["attempts"][-1]["status"] == "audited_draft", lanes["attempts"][-1]
        assert lanes["attempts"][-1]["quality"]["lanes_checked"] == 2
        assert lanes["selected"] == selected["selected"] and lanes["remaining_attempts"] == 0
        diagnosis = jobs.diagnose_mapping_candidate(str(root), 4)
        assert diagnosis["editable_and_reopened_match"] is True
        assert diagnosis["editable"]["complete"] and diagnosis["ground_consensus"]["editable"]["complete"]
        assert diagnosis["extent"]["passes_requested_extent"] is True


def test_native_report_contract_errors_are_persisted(job_backend, tmp_path):
    source, audit = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    audit["validation"] = {"unexpected": "incompatible core output"}
    failed = jobs.generate_mapping_candidate(str(root), OPTIONS, "Record an incompatible native report")
    assert failed["attempts"][-1]["status"] == "failed"
    assert failed["attempts"][-1]["error_type"] == "KeyError"
    assert failed["selected"] is None


def test_high_support_on_a_short_fragment_cannot_replace_the_requested_extent(job_backend, tmp_path, monkeypatch):
    source, audit = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    original = jobs.build_vector_map
    @wraps(original)
    def truncated(*args, **kwargs):
        result = original(*args, **kwargs)
        result["extraction"]["generated_length"] = 1
        return result
    monkeypatch.setattr(jobs, "build_vector_map", truncated)
    audit["quality"]["low_support_lanes"] = []
    generated = jobs.generate_mapping_candidate(str(root), OPTIONS, "Supported fragment only")
    assert generated["attempts"][-1]["quality"]["needs_review"] == []
    assert generated["attempts"][-1]["extent"]["passes_requested_extent"] is False
    diagnosis = jobs.diagnose_mapping_candidate(str(root), 1)
    assert diagnosis["extent"]["passes_requested_extent"] is False
    with pytest.raises(ValueError, match="retained-extent"):
        jobs.select_mapping_candidate(str(root), 1, "A perfect local score cannot recover missing road")
    assert jobs.inspect_mapping_job(str(root))["selected"] is None


@pytest.mark.parametrize("interrupt", [False, True])
def test_native_panics_are_recorded_but_user_interrupts_still_propagate(job_backend, tmp_path, monkeypatch, interrupt):
    source, _ = job_backend
    root = tmp_path / "job"
    jobs.start_mapping_job(str(source), str(root))
    panic = type("PanicException", (BaseException,), {"__module__": "pyo3_runtime"})
    @wraps(jobs.build_vector_map)
    def fail(*args, **kwargs):
        raise KeyboardInterrupt() if interrupt else panic("native panic")
    monkeypatch.setattr(jobs, "build_vector_map", fail)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            jobs.generate_mapping_candidate(str(root), OPTIONS, "Interrupted trial")
    else:
        result = jobs.generate_mapping_candidate(str(root), OPTIONS, "Recoverable native panic")
        assert result["attempts"][-1]["error_type"] == "PanicException"
    assert jobs.inspect_mapping_job(str(root))["attempts"][-1]["status"] == "failed"
    assert not (root / ".mapping-lock").exists()


@pytest.fixture
def connection_run(job_backend, tmp_path, monkeypatch):
    """A surveyed flat source with two lane pieces, a four-metre gap and real OSM."""
    from copy import deepcopy
    from ca import mapping_run as runs
    source, _ = job_backend
    native = _attach_lane_native()
    if not hasattr(native, "connect_vector_map_junctions"):
        pytest.skip("installed core predates connection support")
    module = jobs.core()
    for name in ("connect_vector_map_junctions", "audit_vector_map_quality_details", "audit_vector_map_ground_consensus_details"):
        setattr(module, name, getattr(native, name))
    report = _corridor_report()
    first, second = deepcopy(report["candidates"][0]), deepcopy(report["candidates"][0])
    first.update(to_m=4., sections=first["sections"][:3])
    second.update(id=17, from_m=8., sections=second["sections"][4:])
    report.update(candidates=[first, second], with_candidate_station_length_m=6., without_candidate_station_length_m=4.,
                  trajectory_covered_station_length_m=6., deferred_intervals=[{"from_m": 4., "to_m": 8., "reason": "unresolved_source_band"}])
    module.propose_road_corridors = lambda *args: json.dumps(report)

    def fixed(folder, out, **kwargs):
        root = Path(out)
        root.mkdir()
        cloud, trajectory, graph = root / "map.xyz", root / "poses.txt", root / "map.g2o"
        cloud.write_text("".join(f"{x/5} {y/5} 1\n" for x in range(-5, 56) for y in range(-15, 16)))
        trajectory.write_text("1 0 0 0 0 1 0 0 0 0 1 1\n1 0 0 10 0 1 0 0 0 0 1 1\n")
        graph.write_text("unused graph")
        return {"map_points": 1891, "outputs": {"map": str(cloud), "kitti": str(trajectory), "g2o": str(graph)}}
    monkeypatch.setattr(jobs, "fix_session", fixed)
    root = tmp_path / "connections-run"
    runs.start_mapping_run(str(source), str(root), _run_layout())
    runs.advance_mapping_run(str(root), {"type": "inspect", "candidate_ids": [1, 17]}, "Inspect separated source pieces", 0)
    result = runs.advance_mapping_run(str(root), {"type": "draft", "decisions": [
        {"candidate_id": cid, "action": "include", "reason": "Retain observed source geometry"} for cid in (1, 17)]}, "Draft original disconnected pieces", 1)
    assert result["draft_result"]["status"] == "audited_draft", result
    return root, native


def _inspect_connection(root):
    from ca import mapping_run as runs
    return runs.advance_mapping_run(str(root), {"type": "inspect_connections", "candidate_id": 2, "offset": 0},
                                    "Read source, original drive ordering and exact connection geometry", 2)


def _connection_action(observation):
    return {"type": "connect", "candidate_id": 2, "pairs": [
        {"from": c["from"], "to": c["to"], "reason": "Inspected short consecutive gap with source support and recorded path containment"}
        for c in observation["connection_observation"]["candidates"]]}


@pytest.fixture
def density_run(job_backend, tmp_path, monkeypatch):
    """Real native fusion/HD export with synthetic raw returns and frozen motion."""
    from copy import deepcopy
    import numpy as np
    from ca import mapping_run as runs, mapping_retry as retry
    source, _ = job_backend
    native = _attach_lane_native()
    module = jobs.core()
    for name in ("PoseGraph", "read", "connect_vector_map_junctions", "audit_vector_map_quality_details", "audit_vector_map_ground_consensus_details"):
        setattr(module, name, getattr(native, name))
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[:, 2, 3] = 1.
    poses[1, 0, 3] = 10.
    graph = native.PoseGraph.from_poses(poses)
    xyz = np.array([[x/10, y/10, 0., 1.] for x in range(-10, 111) for y in range(-30, 31)], dtype=np.float32)
    def decode(source, out, **kwargs):
        out = Path(out); out.mkdir(parents=True)
        paths = []
        for i in range(2):
            path = out / f'frame_{i:06d}.bin'
            (xyz - np.array([i*10, 0, 0, 0], dtype=np.float32)).tofile(path)
            paths.append(path)
        return paths, [0., 1.]
    monkeypatch.setattr(retry, "materialize_pointcloud_bag", decode)
    def fixed(folder, out, **kwargs):
        root = Path(out); root.mkdir()
        cloud, trajectory, g2o = root/'map.xyz', root/'poses.txt', root/'map.g2o'
        np.savetxt(cloud, xyz[::4, :3] + [0, 0, 1])
        np.savetxt(trajectory, poses[:, :3, :].reshape(2, 12))
        g2o.write_text(graph.to_g2o())
        return {"map_points": len(xyz[::4]), "nodes": 2, "scans": 2, "unmatched_scans": 0,
                "outputs": {"map": str(cloud), "kitti": str(trajectory), "g2o": str(g2o)}}
    monkeypatch.setattr(jobs, "fix_session", fixed)
    report = _corridor_report()
    original = deepcopy(report["candidates"][0])
    first, second = deepcopy(original), deepcopy(original)
    first.update(to_m=4., sections=first["sections"][:3])
    second.update(id=17, from_m=8., sections=second["sections"][4:])
    report.update(candidates=[first, second], with_candidate_station_length_m=6., without_candidate_station_length_m=4.,
                  trajectory_covered_station_length_m=6., deferred_intervals=[{"from_m": 4., "to_m": 8., "reason": "unresolved_source_band"}])
    report["profiles"] = [{"station_m": float(k), "trajectory": [float(k), 0., 1.], "heading_usable": True,
                           "reference_ground_height_m": 1., "bands": []} for k in range(0, 11, 2)]
    def propose(cloud, trajectory, options):
        r = deepcopy(report)
        r["protocol"] = {"options": {"association": "all_supported_bands", **json.loads(options)}}
        if 'pointcloud-retry' in cloud:
            r.update(candidates=[original], with_candidate_station_length_m=10., without_candidate_station_length_m=0.,
                     trajectory_covered_station_length_m=10., deferred_intervals=[])
        return json.dumps(r)
    module.propose_road_corridors = propose
    root = tmp_path / 'density-run'
    runs.start_mapping_run(str(source), str(root), _run_layout())
    runs.advance_mapping_run(str(root), {"type": "inspect", "candidate_ids": [1, 17]}, "Read separated pieces", 0)
    result = runs.advance_mapping_run(str(root), {"type": "draft", "decisions": [
        {"candidate_id": cid, "action": "include", "reason": "Keep observed source"} for cid in (1, 17)]}, "Baseline", 1)
    assert result["draft_result"]["status"] == "audited_draft"
    return root


def _advance(root, action, reason='Test explicit inspected decision'):
    from ca import mapping_run as runs
    return runs.advance_mapping_run(str(root), action, reason, runs.inspect_mapping_run(str(root))["revision"])


def _density_trial(root):
    evidence = _advance(root, {"type": "inspect_gaps", "candidate_id": 2, "offset": 0})
    ids = [g["id"] for g in evidence["gap_observation"]["gaps"]]
    result = _advance(root, {"type": "retry_pointcloud", "candidate_id": 2, "gap_ids": ids,
                    "options": {"scan_voxel_m": .2, "map_voxel_m": .1}})
    return result


def test_density_retry_freezes_motion_shares_budget_and_requires_comparison(density_run):
    from ca import mapping_run as runs, mapping_connections as connections
    root = density_run
    parent = jobs.inspect_mapping_job(str(root))
    originals = {v['path']: Path(v['path']).read_bytes() for v in [*parent['pointcloud']['files'].values(), *parent['attempts'][1]['files'].values()]}
    proposal = connections.inspect_connections(root, 2, 0)
    assert proposal['candidates']
    result = _density_trial(root)
    assert result['pointcloud_retry_result']['status'] == 'ready', result
    assert result['remaining_attempts'] == 0
    child = Path(result['pointcloud_retry_result']['stage']['child_job_dir'])
    child_job = jobs.inspect_mapping_job(str(child))
    assert child_job['max_attempts'] == 4
    for key in ('trajectory', 'graph'):
        assert parent['pointcloud']['files'][key]['sha256'] == child_job['pointcloud']['files'][key]['sha256']
    for generate in (lambda: jobs.generate_mapping_candidate(str(root), OPTIONS, 'Budget'),
                     lambda: jobs.generate_mapping_geometry(str(root), [{"candidate_id": 1, "action": "include", "reason": "Budget"}], 'Budget'),
                     lambda: jobs.generate_mapping_corridor_lanes(str(root), 1, parent['attempts'][1]['road_options']['lane_specs'], 'source_span_hypothesis', 'Budget'),
                     lambda: connections.connect(root, 2, [{'from': c['from'], 'to': c['to'], 'reason': 'Budget'} for c in proposal['candidates']], proposal['proposal_file'], 'Budget')):
        with pytest.raises(ValueError, match='budget'): generate()
    _advance(child, {'type': 'inspect', 'candidate_ids': [1]})
    drafted = _advance(child, {'type': 'draft', 'decisions': [{'candidate_id': 1, 'action': 'include', 'reason': 'Newly inspected continuous source'}]})
    assert drafted['draft_result']['status'] == 'audited_draft', drafted
    with pytest.raises(ValueError, match='finish the child'):
        _advance(root, {'type': 'finish_retry', 'candidate_id': 2})
    _advance(child, {'type': 'finish', 'candidate_id': 2})
    with pytest.raises(ValueError, match='compare_retry'):
        _advance(root, {'type': 'finish_retry', 'candidate_id': 2})
    compared = _advance(root, {'type': 'compare_retry', 'candidate_id': 2})['retry_comparison']
    assert compared['gained_source_length_m'] == 4. and compared['lost_source_length_m'] == 0.
    assert compared['before']['routes']['longest_route_station_span_m'] == 4.
    assert compared['after']['routes']['longest_route_station_span_m'] == 10.
    assert set(compared['after']['audits']) == {'editable', 'reopened_osm', 'consensus_editable', 'consensus_reopened_osm'}
    assert not compared['automatic_adoption'] and runs.inspect_mapping_run(str(root))['output'] is None
    final = _advance(root, {'type': 'finish_retry', 'candidate_id': 2})
    assert final['output']['candidate_job_dir'] == str(child)
    assert final['output']['artifacts']['map'] == child_job['pointcloud']['files']['map']
    assert all(Path(p).read_bytes() == content for p, content in originals.items())
    assert jobs.inspect_mapping_job(str(root))['selected'] is None


@pytest.mark.parametrize('options', [
    {'scan_voxel_m': .4, 'map_voxel_m': .2}, {'scan_voxel_m': .5, 'map_voxel_m': .1},
    {'scan_voxel_m': True, 'map_voxel_m': .1}, {'scan_voxel_m': float('nan'), 'map_voxel_m': .1},
    {'scan_voxel_m': .05, 'map_voxel_m': .1}, {'map_voxel_m': .1},
])
def test_density_retry_rejects_unbounded_or_unchanged_options_before_spending(density_run, options):
    root = density_run
    _advance(root, {'type': 'inspect_gaps', 'candidate_id': 2, 'offset': 0})
    snapshot = (root / 'job.json').read_bytes()
    with pytest.raises(ValueError):
        _advance(root, {'type': 'retry_pointcloud', 'candidate_id': 2, 'gap_ids': [1], 'options': options})
    assert (root / 'job.json').read_bytes() == snapshot


@pytest.mark.parametrize('fault', ['raw', 'extra_frame', 'motion', 'proposal'])
def test_density_retry_failure_preserves_baseline_and_reservation(density_run, monkeypatch, fault):
    import numpy as np
    from ca import mapping_retry as retry
    root = density_run
    _advance(root, {'type': 'inspect_gaps', 'candidate_id': 2, 'offset': 0})
    original = retry.fix_session
    def fail(*args, **kwargs):
        result = original(*args, **kwargs)
        if fault == 'raw':
            next((root / 'gap-source/scans').glob('*.bin')).write_bytes(b'changed')
        elif fault == 'extra_frame':
            (root / 'gap-source/scans/frame_999999.bin').write_bytes(b'changed')
        elif fault == 'motion':
            motion = np.loadtxt(result['outputs']['kitti']); motion[0, 3] += 1.
            np.savetxt(result['outputs']['kitti'], motion)
        else:
            jobs.core().propose_road_corridors = lambda *args: '{}'
        return result
    monkeypatch.setattr(retry, 'fix_session', fail)
    result = _advance(root, {'type': 'retry_pointcloud', 'candidate_id': 2, 'gap_ids': [1],
                           'options': {'scan_voxel_m': .2, 'map_voxel_m': .1}})
    assert result['pointcloud_retry_result']['status'] == 'failed'
    assert result['remaining_attempts'] == 0 and jobs.inspect_mapping_job(str(root))['pointcloud_retry']['allocated_attempts'] == 4
    final = _advance(root, {'type': 'finish', 'candidate_id': 2})
    assert final['output']['candidate_id'] == 2 and 'candidate_job_dir' not in final['output']


def test_density_retry_completed_stage_resumes_without_duplicate_allocation(density_run, monkeypatch):
    from ca import mapping_run as runs, mapping_retry as retry
    root = density_run
    _advance(root, {'type': 'inspect_gaps', 'candidate_id': 2, 'offset': 0})
    original = retry.retry
    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt()
    monkeypatch.setattr(retry, 'retry', interrupted)
    with pytest.raises(KeyboardInterrupt):
        _advance(root, {'type': 'retry_pointcloud', 'candidate_id': 2, 'gap_ids': [1],
                       'options': {'scan_voxel_m': .2, 'map_voxel_m': .1}})
    monkeypatch.setattr(retry, 'retry', original)
    result = _advance(root, {'type': 'resume'})
    assert result['pointcloud_retry_result']['status'] == 'ready'
    assert result['remaining_attempts'] == 0
    assert jobs.inspect_mapping_job(str(root / 'pointcloud-retry'))['max_attempts'] == 4
    assert len([h for h in json.loads((root / 'run.json').read_text())['history'] if h['action']['type'] == 'retry_pointcloud']) == 1


def test_density_retry_requires_seen_gaps_and_preserves_lost_intervals(density_run):
    from ca import mapping_run as runs
    root = density_run
    action = {'type': 'retry_pointcloud', 'candidate_id': 2, 'gap_ids': [1],
              'options': {'scan_voxel_m': .2, 'map_voxel_m': .1}}
    with pytest.raises(ValueError, match='inspect gaps'):
        _advance(root, action)
    # Inspect an empty page: a valid ID existing in the full report is still unseen.
    _advance(root, {'type': 'inspect_gaps', 'candidate_id': 2, 'offset': 99})
    with pytest.raises(ValueError, match='inspect every chosen gap'):
        _advance(root, action)
    result = _density_trial(root)
    child = Path(result['pointcloud_retry_result']['stage']['child_job_dir'])
    _advance(child, {'type': 'inspect', 'candidate_ids': [1]})
    _advance(child, {'type': 'draft', 'decisions': [{'candidate_id': 1, 'action': 'include', 'from_m': 2., 'to_m': 6.,
                                                'reason': 'Test a replacement with both recovery and loss'}]})
    with pytest.raises(ValueError, match='no recursive'):
        _advance(child, {'type': 'retry_pointcloud', 'candidate_id': 2, 'gap_ids': [1],
                         'options': {'scan_voxel_m': .1, 'map_voxel_m': .05}})
    with pytest.raises(ValueError, match='one point-cloud retry'):
        _advance(root, action)
    compared = _advance(root, {'type': 'compare_retry', 'candidate_id': 2})['retry_comparison']
    assert compared['gained_source_length_m'] == 2. and compared['lost_source_length_m'] == 4.
    assert compared['gained_source_intervals'] == [{'from_m': 4., 'to_m': 6.}]
    assert compared['lost_source_intervals'] == [{'from_m': 0., 'to_m': 2.}, {'from_m': 8., 'to_m': 10.}]
    assert runs.inspect_mapping_run(str(root))['output'] is None
    final = _advance(root, {'type': 'finish', 'candidate_id': 2})
    assert final['output']['artifacts']['map'] == jobs.inspect_mapping_job(str(root))['pointcloud']['files']['map']
    assert final['output']['pointcloud_retry_decision']['adopted'] is False
    assert final['output']['artifacts']['retry_comparison_2'] == compared['file']


def test_density_retry_rejects_changed_gap_evidence_before_allocation(density_run):
    root = density_run
    evidence = _advance(root, {'type': 'inspect_gaps', 'candidate_id': 2, 'offset': 0})['gap_observation']['file']
    Path(evidence['path']).write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        _advance(root, {'type': 'retry_pointcloud', 'candidate_id': 2, 'gap_ids': [1],
                        'options': {'scan_voxel_m': .2, 'map_voxel_m': .1}})
    assert 'pointcloud_retry' not in jobs.inspect_mapping_job(str(root))


def test_failed_raw_gap_inspection_can_finish_with_the_retained_baseline(density_run, monkeypatch):
    from ca import mapping_retry as retry, mapping_run as runs
    root = density_run
    def interrupted_decode(*args, **kwargs):
        raise ValueError('recording reader cannot decode this source')
    monkeypatch.setattr(retry, 'materialize_pointcloud_bag', interrupted_decode)
    with pytest.raises(ValueError, match='reader cannot decode'):
        _advance(root, {'type': 'inspect_gaps', 'candidate_id': 2, 'offset': 0})
    assert runs.inspect_mapping_run(str(root))['status'] == 'interrupted'
    final = _advance(root, {'type': 'finish', 'candidate_id': 2}, 'Retain baseline after failed raw inspection')
    assert final['status'] == 'finished' and final['output']['candidate_id'] == 2
    assert final['remaining_attempts'] == 4


@pytest.fixture
def unused_frame_run(job_backend, tmp_path, monkeypatch, request):
    from copy import deepcopy
    import numpy as np
    from ca import mapping_run as runs, mapping_retry as retry
    source, _ = job_backend
    native = _attach_lane_native()
    module = jobs.core()
    for name in ('PoseGraph', 'read', 'icp', 'audit_vector_map_quality_details', 'audit_vector_map_ground_consensus_details'):
        setattr(module, name, getattr(native, name))
    poses = np.repeat(np.eye(4)[None], 12, axis=0); poses[:, 0, 3] = np.arange(12); poses[:, 2, 3] = 1.
    ids = getattr(request, 'param', list(range(0, 11, 2)))
    xyz = np.array([[x/10, y/10, 0., 1.] for x in range(-10, 111) for y in range(-15, 16)], dtype=np.float32)
    def decode(source, out, **kwargs):
        out = Path(out); out.mkdir(parents=True); paths = []
        for i in range(12):
            path = out / f'frame_{i:06d}.bin'
            (xyz - np.array([i, 0, 0, 0], dtype=np.float32)).tofile(path); paths.append(path)
        return paths, (np.arange(12)*.1).tolist()
    monkeypatch.setattr(retry, 'materialize_pointcloud_bag', decode)
    def odometry(source, out, **kwargs):
        out = Path(out); out.mkdir()
        path = out/'trajectory.tum'; path.write_text(native.PoseGraph.from_poses(poses).to_tum((np.arange(12)*.1).tolist()))
        return {'scans': str(out), 'trajectory': str(path), 'gravity': None, 'frames': 12, 'path_length_m': 11.}
    monkeypatch.setattr(jobs, 'odometry', odometry)
    def fixed(folder, out, **kwargs):
        out = Path(out); out.mkdir()
        from ca.posegraph_fix import write_ply
        cloud, trajectory, graph = out/'map.ply', out/'poses.txt', out/'map.g2o'
        points = xyz[::2, :3]+[0, 0, 1]
        write_ply(cloud, points, {'intensity': np.ones(len(points)), 'correction': np.zeros(len(points))})
        np.savetxt(trajectory, poses[ids, :3].reshape(len(ids), 12))
        graph.write_text(native.PoseGraph.from_poses(poses[ids], ids=ids).to_g2o())
        return {'map_points': len(xyz[::2]), 'nodes': len(ids), 'scans': len(ids), 'unmatched_scans': 6,
                'outputs': {'map': str(cloud), 'kitti': str(trajectory), 'g2o': str(graph)}}
    monkeypatch.setattr(jobs, 'fix_session', fixed)
    report = _corridor_report(); original = deepcopy(report['candidates'][0])
    first, second = deepcopy(original), deepcopy(original)
    first.update(to_m=4., sections=first['sections'][:3]); second.update(id=17, from_m=8., sections=second['sections'][4:])
    report.update(candidates=[first, second], with_candidate_station_length_m=6., without_candidate_station_length_m=4.,
                  trajectory_covered_station_length_m=6., deferred_intervals=[{'from_m': 4., 'to_m': 8., 'reason': 'source_gap'}])
    report['profiles'] = [{'station_m': float(k), 'trajectory': [float(k),0.,1.], 'heading_usable': True,
                           'reference_ground_height_m': 1., 'bands': []} for k in range(0, 11, 2)]
    def propose(cloud, trajectory, options):
        r = deepcopy(report); r['protocol'] = {'options': {'association': 'all_supported_bands', **json.loads(options)}}
        if 'pointcloud-retry' in cloud:
            r.update(candidates=[original], with_candidate_station_length_m=10., without_candidate_station_length_m=0.,
                     trajectory_covered_station_length_m=10., deferred_intervals=[])
        return json.dumps(r)
    module.propose_road_corridors = propose
    root = tmp_path/'unused-frame-run'
    runs.start_mapping_run(str(source), str(root), _run_layout())
    _advance(root, {'type': 'inspect', 'candidate_ids': [1,17]})
    _advance(root, {'type': 'draft', 'decisions': [{'candidate_id': i, 'action': 'include', 'reason': 'Baseline source'} for i in (1,17)]})
    _advance(root, {'type': 'inspect_gaps', 'candidate_id': 2, 'offset': 0})
    return root


def _inspect_unused(root, offset=0):
    return _advance(root, {'type': 'inspect_unused_frames', 'candidate_id': 2, 'offset': offset})['unused_frame_observation']


def test_unused_frames_two_sided_checks_fuse_only_explicit_ids_and_keep_reference(unused_frame_run):
    from ca import mapping_run as runs
    root = unused_frame_run
    original = jobs.inspect_mapping_job(str(root))
    observed = _inspect_unused(root)
    assert observed['frames_total'] == 6 and observed['eligible_total'] == 5
    tail = next(r for r in observed['frames'] if r['frame_id']==11)
    assert not tail['eligible'] and tail['holds'] == ['no_two_sided_corrected_bracket']
    for r in observed['frames'][:-1]:
        assert r['eligible'] and set(r['registration']) == {'before','after'}
        assert all(v['passes'] for v in r['registration'].values())
    with pytest.raises(ValueError, match='eligible'):
        _advance(root, {'type': 'retry_frames', 'candidate_id': 2, 'frame_ids': [11]})
    result = _advance(root, {'type': 'retry_frames', 'candidate_id': 2, 'frame_ids': [5]})
    assert result['pointcloud_retry_result']['status'] == 'ready', result
    child = root/'pointcloud-retry'; job = jobs.inspect_mapping_job(str(child))
    assert result['remaining_attempts'] == 0 and job['max_attempts'] == 4
    assert job['pointcloud']['additional_frame_ids'] == [5]
    assert {'fusion_graph','fusion_trajectory'} <= job['pointcloud']['files'].keys()
    for key in ('trajectory','graph'):
        assert original['pointcloud']['files'][key]['sha256'] == job['pointcloud']['files'][key]['sha256']
    native = jobs.core()
    fusion = native.PoseGraph.from_g2o(Path(job['pointcloud']['files']['fusion_graph']['path']).read_text())
    assert list(fusion.node_ids) == [0,2,4,5,6,8,10]
    _advance(child, {'type': 'inspect', 'candidate_ids': [1]})
    _advance(child, {'type': 'draft', 'decisions': [{'candidate_id': 1, 'action': 'include', 'reason': 'Observed replacement'}]})
    compared = _advance(root, {'type': 'compare_retry', 'candidate_id': 2})['retry_comparison']
    assert compared['retry_strategy'] == 'unused_frames' and compared['added_frame_ids'] == [5]
    assert compared['gained_source_length_m'] == 4. and not compared['lost_source_length_m']
    assert runs.inspect_mapping_run(str(root))['output'] is None
    _advance(child, {'type': 'finish', 'candidate_id': 2})
    final = _advance(root, {'type': 'finish_retry', 'candidate_id': 2})
    assert final['output']['pointcloud_retry_decision']['adopted'] and not final['deployment_ready']


def test_unused_frames_require_seen_receipts_and_reuse_saved_checks(unused_frame_run, monkeypatch):
    root = unused_frame_run
    with pytest.raises(ValueError, match='inspect unused'):
        _advance(root, {'type': 'retry_frames', 'candidate_id': 2, 'frame_ids': [5]})
    _inspect_unused(root, 99)
    with pytest.raises(ValueError, match='inspect every chosen frame'):
        _advance(root, {'type': 'retry_frames', 'candidate_id': 2, 'frame_ids': [5]})
    def do_not_repeat(*args, **kwargs):raise AssertionError('completed registration reran')
    monkeypatch.setattr(jobs.core(), 'icp', do_not_repeat)
    observed = _inspect_unused(root)
    assert observed['eligible_total'] == 5
    for ids in ([True], [5,5], [0], []):
        with pytest.raises(ValueError):_advance(root, {'type': 'retry_frames', 'candidate_id': 2, 'frame_ids': ids})
    assert jobs.inspect_mapping_job(str(root))['remaining_attempts'] == 4


def test_unused_frames_withhold_scan_pose_disagreement(unused_frame_run, monkeypatch):
    import numpy as np
    root = unused_frame_run
    original = jobs.core().icp
    def disagree(*args, **kwargs):
        r = original(*args, **kwargs); m = np.array(r['transformation']); m[0,3] += 1.;r['transformation'] = m
        return r
    monkeypatch.setattr(jobs.core(), 'icp', disagree)
    observed = _inspect_unused(root)
    assert observed['eligible_total'] == 0
    assert any('disagrees' in h for r in observed['frames'] for h in r['holds'])
    with pytest.raises(ValueError, match='eligible'):_advance(root, {'type': 'retry_frames', 'candidate_id': 2, 'frame_ids': [5]})
    assert 'pointcloud_retry' not in jobs.inspect_mapping_job(str(root))


@pytest.mark.parametrize('target', ['raw', 'original_motion', 'frame_evidence'])
def test_unused_frames_pin_original_motion_and_raw_frames_before_allocation(unused_frame_run, target):
    root = unused_frame_run
    observed = _inspect_unused(root)
    frame = next(r for r in observed['frames'] if r['frame_id']==5)
    artifact = frame['raw_frame'] if target == 'raw' else (observed['file'] if target == 'frame_evidence' else jobs.inspect_mapping_job(str(root))['pointcloud']['source_motion']['trajectory'])
    Path(artifact['path']).write_bytes(b'changed source evidence')
    with pytest.raises(ValueError, match='changed'):_advance(root, {'type': 'retry_frames', 'candidate_id': 2, 'frame_ids': [5]})
    assert 'pointcloud_retry' not in jobs.inspect_mapping_job(str(root))


def test_unused_frames_failed_fusion_keeps_baseline_and_allocation(unused_frame_run, monkeypatch):
    import numpy as np
    from ca import mapping_frames as frames
    root = unused_frame_run; _inspect_unused(root)
    original = frames.fix_session
    def altered(*args, **kwargs):
        r = original(*args, **kwargs); p = np.loadtxt(r['outputs']['kitti']); p[0,3] += .1
        np.savetxt(r['outputs']['kitti'],p);return r
    monkeypatch.setattr(frames, 'fix_session', altered)
    result = _advance(root, {'type': 'retry_frames', 'candidate_id': 2, 'frame_ids': [5]})
    assert result['pointcloud_retry_result']['status'] == 'failed' and result['remaining_attempts'] == 0
    final = _advance(root, {'type': 'finish', 'candidate_id': 2})
    assert not final['output']['pointcloud_retry_decision']['adopted']


def test_unused_frames_completed_retry_resumes_once(unused_frame_run, monkeypatch):
    from ca import mapping_retry as retry
    root = unused_frame_run; _inspect_unused(root)
    original = retry.retry
    def interrupted(*args, **kwargs):original(*args, **kwargs);raise KeyboardInterrupt()
    monkeypatch.setattr(retry, 'retry', interrupted)
    with pytest.raises(KeyboardInterrupt):_advance(root, {'type': 'retry_frames', 'candidate_id': 2, 'frame_ids': [5]})
    monkeypatch.setattr(retry, 'retry', original)
    result = _advance(root, {'type': 'resume'})
    assert result['pointcloud_retry_result']['status'] == 'ready' and result['remaining_attempts'] == 0
    assert jobs.inspect_mapping_job(str(root/'pointcloud-retry'))['pointcloud']['additional_frame_ids'] == [5]


@pytest.mark.parametrize('unused_frame_run', [[0,3,6,9]], indirect=True)
def test_unused_frames_sparse_keyframes_keep_original_ids_without_order_fallback(unused_frame_run):
    root = unused_frame_run
    observed = _inspect_unused(root)
    assert next(r for r in observed['frames'] if r['frame_id']==5)['eligible']
    result = _advance(root, {'type': 'retry_frames', 'candidate_id': 2, 'frame_ids': [5]})
    assert result['pointcloud_retry_result']['status'] == 'ready', result
    job = jobs.inspect_mapping_job(str(root/'pointcloud-retry'))
    graph = jobs.core().PoseGraph.from_g2o(Path(job['pointcloud']['files']['fusion_graph']['path']).read_text())
    assert list(graph.node_ids) == [0,3,5,6,9]


def test_connections_preserve_source_extent_and_verify_real_reopened_route(connection_run):
    from ca import mapping_run as runs
    from ca.mapping_connections import edges
    root, _ = connection_run
    original = jobs.inspect_mapping_job(str(root))["attempts"][1]
    original_bytes = {k: Path(v["path"]).read_bytes() for k, v in original["files"].items()}
    before = (root / "run.json").read_bytes()
    with pytest.raises(ValueError, match="inspect connections"):
        runs.advance_mapping_run(str(root), {"type": "connect", "candidate_id": 2, "pairs": [{"from": 3, "to": 6, "reason": "Uninspected"}]}, "Need receipt", 2)
    assert (root / "run.json").read_bytes() == before
    preview = _inspect_connection(root)
    assert preview["remaining_attempts"] == 4
    c = preview["connection_observation"]["candidates"][0]
    assert (c["from"], c["to"], c["station_gap_m"]) == (3, 6, 4.)
    for pairs in ([], [{"from": 6, "to": 3, "reason": "Reverse"}], _connection_action(preview)["pairs"] * 2):
        with pytest.raises(ValueError):
            runs.advance_mapping_run(str(root), {"type": "connect", "candidate_id": 2, "pairs": pairs}, "Reject invalid decision", 3)
    result = runs.advance_mapping_run(str(root), _connection_action(preview), "Join inspected gap without changing prior geometry", 3)
    assert result["connect_result"]["status"] == "audited_draft", result
    diagnosis = result["connect_result"]["diagnosis"]
    assert diagnosis["extent"] == original["extent"] and not diagnosis["extent"]["passes_requested_extent"]
    assert diagnosis["routes"]["before"]["longest_route_station_span_m"] == 4.
    assert diagnosis["routes"]["after"]["longest_route_station_span_m"] == 10.
    assert diagnosis["routes"]["after"]["connected_components"] == 1
    assert result["remaining_attempts"] == 3 and diagnosis["deployment_ready"] is False
    attempt = jobs.inspect_mapping_job(str(root))["attempts"][-1]
    ir = json.loads(Path(attempt["files"]["editable_map"]["path"]).read_text())
    assert edges(ir) == {(3, 9), (9, 6)}
    assert all(Path(original["files"][k]["path"]).read_bytes() == v for k, v in original_bytes.items())
    before = (root / "job.json").read_bytes()
    with pytest.raises(ValueError, match="stale"):
        runs.advance_mapping_run(str(root), _connection_action(preview), "No replay", 3)
    assert (root / "job.json").read_bytes() == before
    with pytest.raises(ValueError, match="retained-extent"):
        jobs.select_mapping_candidate(str(root), 3, "Connection must not inflate source-corridor extent")
    final = runs.advance_mapping_run(str(root), {"type": "finish", "candidate_id": 3}, "Deliver graph with unresolved legal routing", 4)
    assert final["output"]["candidate_id"] == 3 and jobs.inspect_mapping_job(str(root))["selected"] is None


@pytest.mark.parametrize("fault", ["legacy", "consensus", "limited", "topology", "turn_label"])
def test_connection_audit_or_reload_failures_do_not_publish_or_replace_prior_draft(connection_run, fault, tmp_path):
    from ca import mapping_run as runs
    root, native = connection_run
    preview = _inspect_connection(root)
    module = jobs.core()
    if fault in {"topology", "turn_label"}:
        def reopen(path):
            result = json.loads(native.edit_vector_map_relations(path))
            ir = json.loads(result["map_json"])
            if fault == "topology":
                ir["topology"] = []
            else:
                ir["lanes"][-1]["turn_direction"] = "right"
            result["map_json"] = json.dumps(ir)
            return json.dumps(result)
        module.edit_vector_map_relations = reopen
    else:
        name = "audit_vector_map_quality_details" if fault == "legacy" else "audit_vector_map_ground_consensus_details"
        def bad_audit(*args):
            result = json.loads(getattr(native, name)(*args))
            if fault == "limited":
                result["quality"]["limited"] = True
            else:
                connector = next(l for l in result["quality"]["lanes"] if l["lane"] == 9)
                connector["left"].update(fraction=.99, end_supported=False)
            return json.dumps(result)
        setattr(module, name, bad_audit)
    action = tmp_path / "connect.json"
    action.write_text(json.dumps(_connection_action(preview)))
    result = CliRunner().invoke(app, ["mapping-run-advance", str(root), "--action", str(action), "--revision", "3", "--reason", "Retain failed connection evidence"])
    assert result.exit_code == 1, result.output
    state = json.loads(result.stdout)
    assert state["connect_result"]["status"] == "failed" and state["remaining_attempts"] == 3
    assert not (root / "candidate-03").exists()
    attempt = jobs.inspect_mapping_job(str(root))["attempts"][-1]
    if fault not in {"topology", "turn_label"}:
        jobs._verify(attempt["connection_audits"])
    final = runs.advance_mapping_run(str(root), {"type": "finish", "candidate_id": 2}, "Keep earlier audited draft", 4)
    assert final["output"]["candidate_id"] == 2


def test_completed_connection_resumes_without_another_attempt(connection_run, monkeypatch):
    from ca import mapping_run as runs
    root, _ = connection_run
    preview = _inspect_connection(root)
    original = runs.connections.connect
    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt("caller stopped after native completion")
    monkeypatch.setattr(runs.connections, "connect", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runs.advance_mapping_run(str(root), _connection_action(preview), "Explicit inspected connections", 3)
    assert runs.inspect_mapping_run(str(root))["status"] == "interrupted"
    monkeypatch.setattr(runs.connections, "connect", original)
    result = runs.advance_mapping_run(str(root), {"type": "resume"}, "Reuse completed connection", 4)
    assert result["connect_result"]["status"] == "audited_draft" and result["remaining_attempts"] == 3


def test_connection_receipts_bind_source_artifacts_and_all_inspected_pairs(connection_run):
    from ca import mapping_run as runs
    root, _ = connection_run
    preview = _inspect_connection(root)
    # An unobserved page cannot authorize a pair, even if a cached proposal offers it.
    run = json.loads((root / "run.json").read_text())
    run["history"][-1]["connection_observation"]["candidates"] = []
    jobs._save(root / "run.json", run)
    with pytest.raises(ValueError, match="inspect every"):
        runs.advance_mapping_run(str(root), _connection_action(preview), "No unseen adoption", 3)
    run["history"][-1]["connection_observation"]["candidates"] = preview["connection_observation"]["candidates"]
    jobs._save(root / "run.json", run)
    Path(preview["connection_observation"]["proposal_file"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        runs.advance_mapping_run(str(root), _connection_action(preview), "Reject stale geometry", 3)
    assert len(jobs.inspect_mapping_job(str(root))["attempts"]) == 2


def test_recorded_path_containment_includes_edges_but_rejects_shortcuts():
    import numpy as np
    from ca.mapping_connections import _inside, route_metrics
    polygon = np.array([[0, 1], [4, 1], [4, -1], [0, -1]], dtype=float)
    assert _inside(np.array([[0, 0], [2, 0], [4, 0]], dtype=float), polygon)
    assert not _inside(np.array([[0, 0], [2, 2], [4, 0]], dtype=float), polygon)
    with pytest.raises(ValueError, match="station-contiguous"):
        route_metrics({"lanes": [{"id": 1}, {"id": 2}], "topology": [{"lane": 1, "successors": [2]}]}, {1: (0., 4.), 2: (8., 10.)})


@pytest.mark.parametrize("hold", ["recorded_path_outside_connection", "below_fixed_minimum_width", "ambiguous_native_branch"])
def test_connection_preview_withholds_geometric_shortcuts_narrowing_and_ambiguity(connection_run, hold):
    from ca import mapping_run as runs
    root, native = connection_run
    module = jobs.core()
    def changed_preview(*args):
        result = json.loads(native.connect_vector_map_junctions(*args))
        c = result["report"]["junctions"]["candidates"][0]
        if hold == "ambiguous_native_branch":
            c["ambiguous"] = True
        elif hold == "below_fixed_minimum_width":
            for p in c["left"]:
                p[1] = .5
            for p in c["right"]:
                p[1] = -.5
        else:
            for side in ("left", "right"):
                for p in c[side]:
                    p[1] += 3
        return json.dumps(result)
    module.connect_vector_map_junctions = changed_preview
    result = _inspect_connection(root)
    observation = result["connection_observation"]
    assert observation["candidates"] == [] and hold in observation["rejected"][0]["holds"]
    assert result["remaining_attempts"] == 4
    # The retained proposal is reused, rather than silently retrying native extraction.
    module.connect_vector_map_junctions = lambda *args: pytest.fail("cached preview must not rerun")
    result = runs.advance_mapping_run(str(root), {"type": "inspect_connections", "candidate_id": 2, "offset": 0}, "Read retained holds", 3)
    assert result["connection_observation"] == observation


def test_completed_connection_inspection_resumes_from_its_saved_proposal(connection_run, monkeypatch):
    from ca import mapping_run as runs
    root, _ = connection_run
    original = runs.connections.inspect_connections
    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt("caller stopped after saved preview")
    monkeypatch.setattr(runs.connections, "inspect_connections", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _inspect_connection(root)
    monkeypatch.setattr(runs.connections, "inspect_connections", original)
    jobs.core().connect_vector_map_junctions = lambda *args: pytest.fail("completed proposal must be reused")
    result = runs.advance_mapping_run(str(root), {"type": "resume"}, "Reuse saved preview", 3)
    assert len(result["connection_observation"]["candidates"]) == 1 and result["remaining_attempts"] == 4


def _gap_patch_draft(root, start=4., stop=8., preview=True):
    _inspect_unused(root)
    _advance(root, {'type': 'retry_frames', 'candidate_id': 2, 'frame_ids': [5]})
    child = root/'pointcloud-retry'
    _advance(child, {'type': 'inspect', 'candidate_ids': [1]})
    _advance(child, {'type': 'draft', 'decisions': [{'candidate_id': 1, 'action': 'include',
              'from_m': start, 'to_m': stop, 'reason': 'Only the observed missing source interval'}]})
    if preview and start == 4. and stop == 8.:
        _advance(child, {'type':'inspect_patch','candidate_id':2,'gap_ids':[1],'offset':0})
    return child


def _patch_pairs(child):
    run=json.loads((child/'run.json').read_text())
    receipt=next(h['patch_observation'] for h in reversed(run['history']) if 'patch_observation' in h)
    return [{'from':p['from'],'to':p['to'],'reason':'Explicitly join inspected coincident endpoints'} for p in receipt['pairs']]


def test_gap_patch_preserves_original_ids_geometry_and_routes_and_delivers_pair(unused_frame_run):
    from ca import mapping_run as runs, mapping_connections as connections
    root = unused_frame_run
    original_job = jobs.inspect_mapping_job(str(root)); baseline = original_job['attempts'][1]
    original = json.loads(Path(baseline['files']['editable_map']['path']).read_text())
    child = _gap_patch_draft(root)
    result = _advance(child, {'type': 'patch_gaps', 'candidate_id': 2, 'gap_ids': [1], 'pairs':_patch_pairs(child)})
    assert result['patch_result']['status'] == 'audited_draft', result
    job = jobs.inspect_mapping_job(str(child)); patch = job['attempts'][2]
    combined = json.loads(Path(patch['files']['editable_map']['path']).read_text())
    for key in ('lanes','boundaries'):
        current = {r['id']: r for r in combined[key]}
        assert all(current[r['id']] == r for r in original[key])
    assert combined['metadata'] == original['metadata']
    assert len(combined['lanes']) == 3 and connections.edges(original) <= connections.edges(combined)
    assert patch['routes']['after']['longest_route_station_span_m'] >= patch['routes']['before']['longest_route_station_span_m']
    assert patch['extent']['generated_length_m'] == 10.
    checks = json.loads(Path(patch['patch_checks']['path']).read_text()); assert checks['passes'] and not checks['holds']
    assert result['remaining_attempts'] == 1 and original_job['pointcloud']['files']['map'] != job['pointcloud']['files']['map']
    compared = _advance(root, {'type': 'compare_retry', 'candidate_id': 3})['retry_comparison']
    assert compared['gained_source_length_m'] == 4. and compared['lost_source_length_m'] == 0.
    _advance(child, {'type': 'finish', 'candidate_id': 3})
    final = _advance(root, {'type': 'finish_retry', 'candidate_id': 3})
    assert final['output']['artifacts']['hd_patch_checks'] == patch['patch_checks']
    assert final['output']['artifacts']['map'] == job['pointcloud']['files']['map']
    assert runs.inspect_mapping_run(str(root))['output']['pointcloud_retry_decision']['adopted']
    with pytest.raises(ValueError, match='awaiting'):_advance(child, {'type':'patch_gaps','candidate_id':2,'gap_ids':[1],'pairs':[]})


@pytest.mark.parametrize('invalid_ids', [[], [True], [1,1], [99]])
def test_gap_patch_rejects_invalid_or_unseen_gaps_before_budget(unused_frame_run, invalid_ids):
    child = _gap_patch_draft(unused_frame_run)
    with pytest.raises(ValueError, match='gap IDs'):_advance(child, {'type':'patch_gaps','candidate_id':2,'gap_ids':invalid_ids,'pairs':[]})
    assert len(jobs.inspect_mapping_job(str(child))['attempts']) == 2
    assert 'gap_patch' not in jobs.inspect_mapping_job(str(child))


def test_gap_patch_rejects_whole_drive_replacement_and_tampered_baseline(unused_frame_run):
    root = unused_frame_run; child = _gap_patch_draft(root,0.,10.)
    with pytest.raises(ValueError, match='entirely inside'):_advance(child, {'type':'patch_gaps','candidate_id':2,'gap_ids':[1],'pairs':[]})
    baseline = jobs.inspect_mapping_job(str(root))['attempts'][1]
    Path(baseline['files']['editable_map']['path']).write_text('{}')
    with pytest.raises(ValueError, match='changed'):_advance(child, {'type':'patch_gaps','candidate_id':2,'gap_ids':[1],'pairs':[]})
    assert 'gap_patch' not in jobs.inspect_mapping_job(str(child))


def test_gap_patch_cannot_fill_over_an_existing_connector(unused_frame_run):
    from ca import mapping_retry as retry
    root = unused_frame_run
    native = _attach_lane_native(); jobs.core().connect_vector_map_junctions = native.connect_vector_map_junctions
    observed = _advance(root, {'type':'inspect_connections','candidate_id':2,'offset':0})['connection_observation']
    pair = observed['candidates'][0]
    _advance(root, {'type':'connect','candidate_id':2,'pairs':[{'from':pair['from'],'to':pair['to'],'reason':'Preserve checked original route'}]})
    _advance(root, {'type':'inspect_gaps','candidate_id':3,'offset':0})
    _advance(root, {'type':'inspect_unused_frames','candidate_id':3,'offset':0})
    _advance(root, {'type':'retry_frames','candidate_id':3,'frame_ids':[5]})
    child = root/'pointcloud-retry'
    _advance(child, {'type':'inspect','candidate_ids':[1]})
    _advance(child, {'type':'draft','decisions':[{'candidate_id':1,'action':'include','from_m':4.,'to_m':8.,'reason':'Investigate source gap'}]})
    with pytest.raises(ValueError, match='retained lane or connector'):_advance(child, {'type':'patch_gaps','candidate_id':2,'gap_ids':[1],'pairs':[]})
    assert retry._routes(jobs.inspect_mapping_job(str(root))['attempts'][2])['longest_route_station_span_m'] == 10.
    assert 'gap_patch' not in jobs.inspect_mapping_job(str(child))


def test_gap_patch_rejects_new_failure_location_even_when_totals_unchanged():
    from copy import deepcopy
    from ca import mapping_patch as patch
    trace = {'samples':10,'supported':9,'insufficient_returns':0,'height_mismatches':1,'fraction':.9,'start_supported':True,'end_supported':True}
    audit = {'validation':{'issues':[]},'import_issues':[], 'quality':{
        'lanes':[{'lane':3,'needs_review':True,**{k:deepcopy(trace) for k in ('center','left','right')}}],
        'minimum_support_fraction':.9,'limited':False,'omitted_lanes':[],'malformed_lanes':[],'low_support_lanes':[3],
        'sampling_step_m':.5,'ground_radius_m':.75,'ground_height_tolerance_m':.25,'sample_budget':100000,'warnings':[],
        'problems_limited':False,'problems':[{'lane':3,'curve':k,'reason':'height_mismatch','points':[[1.,0.,0.]]} for k in ('center','left','right')]}}
    before={'editable':audit,'reopened_osm':deepcopy(audit),'ground_consensus':{'editable':deepcopy(audit),'reopened_osm':deepcopy(audit)}}
    after=deepcopy(before)
    after['editable']['quality']['problems'][0]['points']=[[2.,0.,0.]]
    result=patch._checks(before,after,{3},set())
    assert not result['passes'] and result['holds']==['editable:new_retained_failure_location:3:center']
    after=deepcopy(before);after['editable']['quality']['problems_limited']=True
    assert not patch._checks(before,after,{3},set())['passes']


def test_gap_patch_native_change_is_failed_without_replacing_baseline(unused_frame_run,monkeypatch):
    root=unused_frame_run;child=_gap_patch_draft(root)
    original = jobs.core().edit_vector_map_relations
    def changed(path,*args):
        payload=json.loads(original(path,*args));ir=json.loads(payload['map_json'])
        ir['boundaries'][0]['geometry'][0][1] += .1
        payload['map_json']=json.dumps(ir);return json.dumps(payload)
    monkeypatch.setattr(jobs.core(),'edit_vector_map_relations',changed)
    result=_advance(child, {'type':'patch_gaps','candidate_id':2,'gap_ids':[1],'pairs':_patch_pairs(child)})
    assert result['patch_result']['status']=='failed' and 'retained' in result['patch_result']['error']
    assert len(jobs.inspect_mapping_job(str(child))['attempts'])==3
    assert not (child/'candidate-03').exists()
    final=_advance(root, {'type':'finish','candidate_id':2}, 'Keep original pair after failed local repair')
    assert final['output']['pointcloud_retry_decision']['adopted'] is False


def test_gap_patch_completed_stage_resumes_without_spending_twice(unused_frame_run,monkeypatch):
    from ca import mapping_run as runs, mapping_patch as patch
    root=unused_frame_run;child=_gap_patch_draft(root)
    diagnosis=runs._diagnosis
    monkeypatch.setattr(runs,'_diagnosis',lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):_advance(child, {'type':'patch_gaps','candidate_id':2,'gap_ids':[1],'pairs':_patch_pairs(child)})
    assert len(jobs.inspect_mapping_job(str(child))['attempts'])==3
    monkeypatch.setattr(runs,'_diagnosis',diagnosis)
    monkeypatch.setattr(patch,'patch',lambda *a: pytest.fail('completed patch was repeated'))
    result=_advance(child, {'type':'resume'})
    assert result['patch_result']['status']=='audited_draft' and len(jobs.inspect_mapping_job(str(child))['attempts'])==3


def test_gap_patch_requires_explicit_seen_endpoint_pairs_before_spending(unused_frame_run):
    from ca import mapping_patch as patch
    child=_gap_patch_draft(unused_frame_run,preview=False)
    observed=patch.preview(child,2,[1],0)
    pairs=[{'from':p['from'],'to':p['to'],'reason':'Join inspected endpoints'} for p in observed['pairs']]
    with pytest.raises(ValueError,match='inspect the gap patch'):
        _advance(child,{'type':'patch_gaps','candidate_id':2,'gap_ids':[1],'pairs':pairs})
    _advance(child,{'type':'inspect_patch','candidate_id':2,'gap_ids':[1],'offset':99})
    with pytest.raises(ValueError,match='every patch endpoint pair'):
        _advance(child,{'type':'patch_gaps','candidate_id':2,'gap_ids':[1],'pairs':pairs})
    _advance(child,{'type':'inspect_patch','candidate_id':2,'gap_ids':[1],'offset':0})
    with pytest.raises(ValueError,match='every implicit endpoint pair'):
        _advance(child,{'type':'patch_gaps','candidate_id':2,'gap_ids':[1],'pairs':[]})
    assert len(jobs.inspect_mapping_job(str(child))['attempts'])==2


def test_gap_patch_failed_source_check_retains_full_audits_and_baseline(unused_frame_run,monkeypatch):
    root=unused_frame_run;child=_gap_patch_draft(root)
    original=jobs.core().audit_vector_map_quality_details
    def unsupported(*args):
        result=json.loads(original(*args))
        for lane in result['quality']['lanes']:
            if lane['lane'] <= 6:continue
            lane['left']['supported'] -= 1;lane['left']['height_mismatches'] += 1
            lane['left']['fraction']=lane['left']['supported']/lane['left']['samples']
        return json.dumps(result)
    monkeypatch.setattr(jobs.core(),'audit_vector_map_quality_details',unsupported)
    result=_advance(child,{'type':'patch_gaps','candidate_id':2,'gap_ids':[1],'pairs':_patch_pairs(child)})
    assert result['patch_result']['status']=='failed'
    checks=json.loads(Path(result['patch_result']['patch_checks']['path']).read_text())
    assert not checks['passes'] and any('new_lane_not_fully_supported' in h for h in checks['holds'])
    assert Path(result['patch_result']['patch_audits']['path']).is_file() and not (child/'candidate-03').exists()
    with pytest.raises(ValueError):_advance(root,{'type':'compare_retry','candidate_id':3})
    final=_advance(root,{'type':'finish','candidate_id':2},'Keep the earlier pair after held source evidence')
    assert final['output']['pointcloud_retry_decision']['adopted'] is False


@pytest.fixture
def patched_connection_run(unused_frame_run):
    """A retained endpoint link plus a partial repair, leaving a supported 2 m gap."""
    root = unused_frame_run
    native = _attach_lane_native()
    jobs.core().connect_vector_map_junctions = native.connect_vector_map_junctions
    child = _gap_patch_draft(root, 4., 6.)
    _advance(child, {'type': 'inspect_patch', 'candidate_id': 2, 'gap_ids': [1], 'offset': 0})
    result = _advance(child, {'type': 'patch_gaps', 'candidate_id': 2, 'gap_ids': [1], 'pairs': _patch_pairs(child)})
    assert result['patch_result']['status'] == 'audited_draft', result
    parent = jobs.inspect_mapping_job(str(child))['attempts'][2]
    return root, child, parent, native


def _patch_connection_action(child):
    observed = _advance(child, {'type': 'inspect_connections', 'candidate_id': 3, 'offset': 0})['connection_observation']
    assert [(c['from'], c['to'], c['station_gap_m']) for c in observed['candidates']] == [(9, 6, 2.)]
    return {'type': 'connect', 'candidate_id': 3, 'pairs': [{'from': 9, 'to': 6, 'reason': 'Join the inspected open gap after partial repair'}]}


def test_connections_extend_partial_repairs_preserving_routes_and_deliver_checks(patched_connection_run):
    from ca import mapping_connections as connections
    root, child, parent, _ = patched_connection_run
    original = json.loads(Path(parent['files']['editable_map']['path']).read_text())
    assert connections.edges(original) == {(3, 9)}
    with pytest.raises(ValueError, match='inspect connections'):
        _advance(child, {'type': 'connect', 'candidate_id': 3, 'pairs': [{'from': 9, 'to': 6, 'reason': 'Unseen'}]})
    result = _advance(child, _patch_connection_action(child))
    assert result['connect_result']['status'] == 'audited_draft', result
    current = jobs.inspect_mapping_job(str(child))['attempts'][3]
    ir = json.loads(Path(current['files']['editable_map']['path']).read_text())
    connections_set = {(3, 9), (9, 12), (12, 6)}
    assert connections.edges(ir) == connections_set
    for key in ('lanes', 'boundaries'):
        assert all(item in ir[key] for item in original[key])
    assert current['routes']['before']['connected_components'] == 2
    assert current['routes']['after']['routes'][0]['lane_ids'] == [3, 9, 12, 6]
    assert current['routes']['after']['longest_route_station_span_m'] == 10.
    assert current['extent'] == parent['extent'] and current['extent']['generated_length_m'] == 8.
    assert result['remaining_attempts'] == 0
    checks = json.loads(Path(current['connection_checks']['path']).read_text())
    assert checks['passes'] and not checks['holds']
    assert connections.inspect_connections(child, 4, 0)['candidates_total'] == 0
    _advance(root, {'type': 'compare_retry', 'candidate_id': 4})
    _advance(child, {'type': 'finish', 'candidate_id': 4})
    finished = _advance(root, {'type': 'finish_retry', 'candidate_id': 4})
    assert finished['output']['artifacts']['hd_connection_checks'] == current['connection_checks']
    assert finished['output']['artifacts']['hd_patch_checks'] == parent['patch_checks']


@pytest.mark.parametrize('fault', ['remove_old_edge', 'old_source_failure'])
def test_partial_repair_connections_reject_regressions_and_keep_parent(patched_connection_run, fault, monkeypatch):
    root, child, parent, native = patched_connection_run
    action = _patch_connection_action(child)
    if fault == 'remove_old_edge':
        def changed(*args):
            payload = json.loads(native.connect_vector_map_junctions(*args))
            ir = json.loads(payload['map_json'])
            for row in ir['topology']:
                if row['lane'] == 3: row['successors'] = []
                if row['lane'] == 9: row['predecessors'] = []
            payload['map_json'] = json.dumps(ir)
            return json.dumps(payload)
        monkeypatch.setattr(jobs.core(), 'connect_vector_map_junctions', changed)
    else:
        def changed(*args):
            audit = json.loads(native.audit_vector_map_quality_details(*args))
            old = next(l for l in audit['quality']['lanes'] if l['lane'] == 3)['left']
            old.update(supported=old['supported']-1, height_mismatches=1, fraction=.9, start_supported=False)
            return json.dumps(audit)
        monkeypatch.setattr(jobs.core(), 'audit_vector_map_quality_details', changed)
    result = _advance(child, action)
    assert result['connect_result']['status'] == 'failed', result
    assert not (child/'candidate-04').exists()
    assert jobs.inspect_mapping_job(str(child))['selected'] is None
    if fault == 'old_source_failure':
        failed = jobs.inspect_mapping_job(str(child))['attempts'][3]
        assert not json.loads(Path(failed['connection_checks']['path']).read_text())['passes']
        jobs._verify(failed['connection_audits'])
    _advance(root, {'type': 'compare_retry', 'candidate_id': 3})
    _advance(child, {'type': 'finish', 'candidate_id': 3})
    final = _advance(root, {'type': 'finish_retry', 'candidate_id': 3})
    assert final['output']['artifacts']['hd_patch_checks'] == parent['patch_checks']


def test_partial_repair_connection_binds_inherited_checks_before_spending(patched_connection_run):
    _, child, parent, _ = patched_connection_run
    action = _patch_connection_action(child)
    Path(parent['patch_checks']['path']).write_text('tampered inherited support checks')
    with pytest.raises(ValueError, match='changed'):
        _advance(child, action)
    assert len(jobs._load(child)['attempts']) == 3


def test_partial_repair_connection_resume_reuses_the_shared_attempt(patched_connection_run, monkeypatch):
    from ca import mapping_run as runs
    _, child, _, _ = patched_connection_run
    action = _patch_connection_action(child)
    diagnosis = runs._diagnosis
    monkeypatch.setattr(runs, '_diagnosis', lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt): _advance(child, action)
    monkeypatch.setattr(runs, '_diagnosis', diagnosis)
    monkeypatch.setattr(runs.connections, 'connect', lambda *args: pytest.fail('completed connection must not repeat'))
    result = _advance(child, {'type': 'resume'})
    assert result['connect_result']['status'] == 'audited_draft' and result['remaining_attempts'] == 0
    assert len(jobs.inspect_mapping_job(str(child))['attempts']) == 4


def _local_preview(root, box=None):
    _inspect_unused(root)
    return _advance(root, {'type':'inspect_local_points','candidate_id':2,'gap_ids':[1],
                          'bounds_xy':box or [2.,-1.6,8.,1.6]})['local_point_observation']


def _local_action(preview):
    return {'type':'retry_local_frames','candidate_id':2,'frame_ids':[5],'preview_file':preview['file']}


def test_local_point_update_keeps_outside_records_attributes_motion_and_hd(unused_frame_run):
    from ca import mapping_local_points as local
    root=unused_frame_run;original=jobs.inspect_mapping_job(str(root))
    base_file=original['pointcloud']['files']['map'];base_bytes=Path(base_file['path']).read_bytes()
    preview=_local_preview(root)
    assert 5 in preview['eligible_frame_ids'] and preview['inside_points']>0 and preview['outside_points']>0
    result=_advance(root,_local_action(preview))
    assert result['pointcloud_retry_result']['status']=='ready',result
    child=root/'pointcloud-retry';cj=jobs.inspect_mapping_job(str(child))
    _,base=local.records(Path(base_file['path']));_,updated=local.records(Path(cj['pointcloud']['files']['map']['path']))
    box=preview['effective_bounds_xy']
    assert updated[~local.mask(updated,box)].tobytes()==base[~local.mask(base,box)].tobytes()
    assert updated.dtype.names==('x','y','z','intensity','correction')
    assert any(base['x']==box[2]) and all(p['x']<box[2] for p in updated[local.mask(updated,box)])
    assert Path(base_file['path']).read_bytes()==base_bytes
    assert all(cj['pointcloud']['files'][k]['sha256']==original['pointcloud']['files'][k]['sha256'] for k in ('graph','trajectory'))
    _advance(child,{'type':'inspect','candidate_ids':[1]})
    _advance(child,{'type':'draft','decisions':[{'candidate_id':1,'action':'include','from_m':4.,'to_m':6.,'reason':'Only observed local source'}]})
    with pytest.raises(ValueError,match='combined HD patch'):_advance(root,{'type':'compare_retry','candidate_id':2})
    _advance(child,{'type':'inspect_patch','candidate_id':2,'gap_ids':[1],'offset':0})
    patched=_advance(child,{'type':'patch_gaps','candidate_id':2,'gap_ids':[1],'pairs':_patch_pairs(child)})
    assert patched['patch_result']['status']=='audited_draft',patched
    compared=_advance(root,{'type':'compare_retry','candidate_id':3})['retry_comparison']
    assert compared['local_point_update']['outside_records_bit_identical'] and compared['lost_source_length_m']==0
    _advance(child,{'type':'finish','candidate_id':3})
    final=_advance(root,{'type':'finish_retry','candidate_id':3})
    assert final['output']['artifacts']['local_update_checks']==cj['pointcloud']['files']['local_update_checks']
    assert final['output']['artifacts']['map']==cj['pointcloud']['files']['map']


@pytest.mark.parametrize('box',[[True,0,1,1],[0,0,float('nan'),1],[0,0,50,1],[3,0,2,1],[40,40,42,42]])
def test_local_point_preview_rejects_invalid_unobserved_boxes_without_allocation(unused_frame_run,box):
    root=unused_frame_run;_inspect_unused(root)
    with pytest.raises(ValueError):_advance(root,{'type':'inspect_local_points','candidate_id':2,'gap_ids':[1],'bounds_xy':box})
    assert 'pointcloud_retry' not in jobs._load(root)


@pytest.mark.parametrize('fault',['unseen_preview','tampered_preview','unseen_frame'])
def test_local_point_adoption_requires_seen_frozen_region_and_frames(unused_frame_run,fault):
    root=unused_frame_run;preview=_local_preview(root);action=_local_action(preview)
    if fault=='unseen_preview':
        run=json.loads((root/'run.json').read_text());run['history'][-1].pop('local_point_observation');jobs._save(root/'run.json',run)
    elif fault=='tampered_preview':Path(preview['file']['path']).write_text('changed local box')
    else:
        run=json.loads((root/'run.json').read_text())
        for h in run['history']:
            if 'unused_frame_observation' in h:h['unused_frame_observation']['frames']=[]
        jobs._save(root/'run.json',run)
    with pytest.raises(ValueError):_advance(root,action)
    assert 'pointcloud_retry' not in jobs._load(root)


def test_local_point_hd_patch_rejects_changes_outside_the_preview_box(unused_frame_run):
    root=unused_frame_run;preview=_local_preview(root,[4.,-.8,6.,.8])
    result=_advance(root,_local_action(preview));assert result['pointcloud_retry_result']['status']=='ready',result
    child=root/'pointcloud-retry';_advance(child,{'type':'inspect','candidate_ids':[1]})
    _advance(child,{'type':'draft','decisions':[{'candidate_id':1,'action':'include','from_m':4.,'to_m':6.,'reason':'Inspected local station range'}]})
    with pytest.raises(ValueError,match='inside the local point update box'):
        _advance(child,{'type':'inspect_patch','candidate_id':2,'gap_ids':[1],'offset':0})
    assert 'gap_patch' not in jobs._load(child)


def test_local_point_failed_source_gate_retains_checks_and_original_pair(unused_frame_run,monkeypatch):
    root=unused_frame_run;preview=_local_preview(root);original=jobs.core().audit_vector_map_quality_details
    def unsupported(*args):
        audit=json.loads(original(*args));audit['quality']['lanes'][0]['left']['start_supported']=False
        return json.dumps(audit)
    monkeypatch.setattr(jobs.core(),'audit_vector_map_quality_details',unsupported)
    result=_advance(root,_local_action(preview));stage=result['pointcloud_retry_result']['stage']
    assert stage['status']=='failed' and not stage['local_update']['passes']
    assert Path(stage['local_update']['checks']['path']).is_file() and Path(stage['local_update']['audits']['path']).is_file()
    final=_advance(root,{'type':'finish','candidate_id':2})
    assert not final['output']['pointcloud_retry_decision']['adopted']
    assert final['output']['artifacts']['map']==jobs._load(root)['pointcloud']['files']['map']
    for key in ('report','checks','audits'):
        assert final['output']['artifacts'][f'pointcloud_trial_local_{key}']==stage['local_update'][key]


def test_local_point_completed_retry_resumes_without_second_allocation(unused_frame_run,monkeypatch):
    from ca import mapping_retry as retry
    root=unused_frame_run;preview=_local_preview(root);original=retry.retry
    def interrupted(*args,**kwargs):original(*args,**kwargs);raise KeyboardInterrupt()
    monkeypatch.setattr(retry,'retry',interrupted)
    with pytest.raises(KeyboardInterrupt):_advance(root,_local_action(preview))
    monkeypatch.setattr(retry,'retry',original)
    result=_advance(root,{'type':'resume'})
    assert result['pointcloud_retry_result']['status']=='ready' and result['remaining_attempts']==0
    assert jobs._load(root)['pointcloud_retry']['strategy']=='local_unused_frames'


def test_local_point_inverted_subvoxel_box_can_be_corrected_without_interrupt(unused_frame_run):
    root=unused_frame_run;_inspect_unused(root)
    with pytest.raises(ValueError,match='positive requested sides'):
        _advance(root,{'type':'inspect_local_points','candidate_id':2,'gap_ids':[1],'bounds_xy':[4.09,-1.,4.01,1.]})
    assert json.loads((root/'run.json').read_text())['status']=='needs_agent'
    assert _local_preview(root)['outside_points']>0


def test_local_point_export_attribute_corruption_is_not_adopted(unused_frame_run,monkeypatch):
    from ca import mapping_local_points as local
    root=unused_frame_run;preview=_local_preview(root);original=local.records
    def corrupt(path):
        header,rows=original(path)
        if path.name=='local_map.ply':rows[0]['intensity']+=1.
        return header,rows
    monkeypatch.setattr(local,'records',corrupt)
    result=_advance(root,_local_action(preview))
    assert result['pointcloud_retry_result']['status']=='failed'
    assert 'outside coordinates, attributes or record order' in result['pointcloud_retry_result']['stage']['error']
    final=_advance(root,{'type':'finish','candidate_id':2})
    assert final['output']['pointcloud_retry_decision']['adopted'] is False


def test_local_point_connection_preview_holds_connectors_leaving_the_update_box(unused_frame_run):
    root=unused_frame_run;preview=_local_preview(root,[2.,-1.6,6.,1.6])
    jobs.core().connect_vector_map_junctions=_attach_lane_native().connect_vector_map_junctions
    result=_advance(root,_local_action(preview));assert result['pointcloud_retry_result']['status']=='ready',result
    child=root/'pointcloud-retry';_advance(child,{'type':'inspect','candidate_ids':[1]})
    _advance(child,{'type':'draft','decisions':[{'candidate_id':1,'action':'include','from_m':4.,'to_m':6.,'reason':'Local repair within the selected box'}]})
    _advance(child,{'type':'inspect_patch','candidate_id':2,'gap_ids':[1],'offset':0})
    patched=_advance(child,{'type':'patch_gaps','candidate_id':2,'gap_ids':[1],'pairs':_patch_pairs(child)})
    assert patched['patch_result']['status']=='audited_draft',patched
    observed=_advance(child,{'type':'inspect_connections','candidate_id':3,'offset':0})['connection_observation']
    assert observed['candidates_total']==0
    assert any(r['from']==9 and r['to']==6 and 'outside_local_point_update_bounds' in r['holds'] for r in observed['rejected'])


def _local_density_preview(root):
    return _advance(root, {'type':'inspect_local_density','candidate_id':2,'gap_ids':[1],
                          'bounds_xy':[2.,-1.6,8.,1.6]})['local_point_observation']


def _local_density_action(preview, options=None):
    return {'type':'retry_local_density','candidate_id':2,'preview_file':preview['file'],
            'options':options or {'scan_voxel_m':.2,'map_voxel_m':.1}}


def test_local_density_updates_inside_records_without_unused_frames_and_keeps_hd(unused_frame_run):
    from ca import mapping_local_points as local
    root=unused_frame_run;before=jobs._load(root)
    base_file=before['pointcloud']['files']['map'];base_bytes=Path(base_file['path']).read_bytes()
    preview=_local_density_preview(root)
    assert preview['eligible_frame_ids']==[] and preview['holds']==[]
    assert not list(root.glob('unused-frames-*.json'))
    result=_advance(root,_local_density_action(preview));stage=result['pointcloud_retry_result']['stage']
    assert stage['status']=='ready' and stage['strategy']=='local_density',result
    child=root/'pointcloud-retry';cj=jobs.inspect_mapping_job(str(child))
    _,base=local.records(Path(base_file['path']));_,updated=local.records(Path(cj['pointcloud']['files']['map']['path']))
    box=preview['effective_bounds_xy']
    assert updated[~local.mask(updated,box)].tobytes()==base[~local.mask(base,box)].tobytes()
    assert Path(base_file['path']).read_bytes()==base_bytes
    report=json.loads(Path(cj['pointcloud']['reports']['correction']).read_text())
    original_report=json.loads(Path(before['pointcloud']['reports']['correction']).read_text())
    assert report['nodes']==original_report['nodes'] and report['unmatched_scans']==6
    assert 'additional_frame_ids' not in cj['pointcloud'] and not list(child.glob('fusion-scans*'))
    assert cj['pointcloud_options']['scan_voxel_m']==.2 and cj['pointcloud_options']['map_voxel_m']==.1
    for key in ('graph','trajectory'):
        assert cj['pointcloud']['files'][key]['sha256']==before['pointcloud']['files'][key]['sha256']
    _advance(child,{'type':'inspect','candidate_ids':[1]})
    _advance(child,{'type':'draft','decisions':[{'candidate_id':1,'action':'include','from_m':4.,'to_m':6.,'reason':'Only source-supported local addition'}]})
    with pytest.raises(ValueError,match='combined HD patch'):_advance(root,{'type':'compare_retry','candidate_id':2})
    _advance(child,{'type':'inspect_patch','candidate_id':2,'gap_ids':[1],'offset':0})
    patched=_advance(child,{'type':'patch_gaps','candidate_id':2,'gap_ids':[1],'pairs':_patch_pairs(child)})
    assert patched['patch_result']['status']=='audited_draft',patched
    comparison=_advance(root,{'type':'compare_retry','candidate_id':3})['retry_comparison']
    assert comparison['local_point_update']['outside_records_bit_identical'] and comparison['lost_source_length_m']==0
    _advance(child,{'type':'finish','candidate_id':3})
    final=_advance(root,{'type':'finish_retry','candidate_id':3})
    assert final['output']['artifacts']['map']==cj['pointcloud']['files']['map']


@pytest.mark.parametrize('options',[{'scan_voxel_m':.4,'map_voxel_m':.2},
    {'scan_voxel_m':.05,'map_voxel_m':.1},{'scan_voxel_m':True,'map_voxel_m':.1},
    {'scan_voxel_m':.2,'map_voxel_m':.1,'remove_dynamic':False}])
def test_local_density_invalid_options_do_not_consume_a_retry(unused_frame_run,options):
    root=unused_frame_run;preview=_local_density_preview(root)
    before=(root/'run.json').read_bytes()
    with pytest.raises(ValueError):_advance(root,_local_density_action(preview,options))
    assert before==(root/'run.json').read_bytes() and 'pointcloud_retry' not in jobs._load(root)
    assert not (root/'pointcloud-retry').exists()


@pytest.mark.parametrize('fault',['unseen','tampered','frame_strategy'])
def test_local_density_requires_seen_preview_for_the_density_strategy(unused_frame_run,fault):
    root=unused_frame_run;preview=_local_density_preview(root)
    if fault=='unseen':
        run=json.loads((root/'run.json').read_text());run['history'][-1].pop('local_point_observation');jobs._save(root/'run.json',run)
    elif fault=='tampered':
        Path(preview['file']['path']).write_text('{}')
    else:
        preview=_local_preview(root)
    with pytest.raises(ValueError):_advance(root,_local_density_action(preview))
    assert 'pointcloud_retry' not in jobs._load(root) and not (root/'pointcloud-retry').exists()


def test_frame_retry_cannot_use_a_density_preview(unused_frame_run):
    root=unused_frame_run;preview=_local_density_preview(root);_inspect_unused(root)
    with pytest.raises(ValueError,match='another decision or protocol'):_advance(root,_local_action(preview))
    assert 'pointcloud_retry' not in jobs._load(root)


def test_local_density_completed_retry_resumes_without_recomputing(unused_frame_run,monkeypatch):
    from ca import mapping_retry as retry
    root=unused_frame_run;preview=_local_density_preview(root);original=retry.retry
    def interrupted(*args,**kwargs):original(*args,**kwargs);raise KeyboardInterrupt()
    monkeypatch.setattr(retry,'retry',interrupted)
    with pytest.raises(KeyboardInterrupt):_advance(root,_local_density_action(preview))
    monkeypatch.setattr(retry,'retry',original)
    monkeypatch.setattr(retry,'fix_session',lambda *args,**kwargs:pytest.fail('completed fusion must not repeat'))
    resumed=_advance(root,{'type':'resume'})
    assert resumed['pointcloud_retry_result']['status']=='ready' and resumed['remaining_attempts']==0
    assert jobs._load(root)['pointcloud_retry']['allocated_attempts']==4
    with pytest.raises(ValueError,match='one point-cloud retry'):_advance(root,_local_density_action(preview))


def test_local_density_retained_support_failure_returns_original_pair(unused_frame_run,monkeypatch):
    root=unused_frame_run;preview=_local_density_preview(root);original=jobs.core().audit_vector_map_quality_details
    def regressed(*args):
        report=json.loads(original(*args));report['quality']['lanes'][0]['left']['start_supported']=False
        return json.dumps(report)
    monkeypatch.setattr(jobs.core(),'audit_vector_map_quality_details',regressed)
    stage=_advance(root,_local_density_action(preview))['pointcloud_retry_result']['stage']
    assert stage['status']=='failed' and not stage['local_update']['passes']
    child=jobs._load(root/'pointcloud-retry');assert child['pointcloud'] is None and child['attempts']==[]
    final=_advance(root,{'type':'finish','candidate_id':2})['output']
    assert not final['pointcloud_retry_decision']['adopted']
    assert final['artifacts']['map']==jobs._load(root)['pointcloud']['files']['map']
    assert final['artifacts']['hd_map']==jobs._load(root)['attempts'][1]['files']['map']
    assert final['artifacts']['pointcloud_trial_local_checks']==stage['local_update']['checks']


def test_local_density_rejects_changed_fusion_frame_ids_before_map_publication(unused_frame_run,monkeypatch):
    from ca import mapping_retry as retry
    root=unused_frame_run;preview=_local_density_preview(root);original=retry.fix_session
    def altered(*args,**kwargs):
        result=original(*args,**kwargs)
        path=Path(result['outputs']['g2o']);graph=jobs.core().PoseGraph.from_g2o(path.read_text())
        path.write_text(jobs.core().PoseGraph.from_poses(graph.poses(),ids=[i+100 for i in graph.node_ids]).to_g2o())
        return result
    monkeypatch.setattr(retry,'fix_session',altered)
    stage=_advance(root,_local_density_action(preview))['pointcloud_retry_result']['stage']
    assert stage['status']=='failed' and 'retained frames or frozen poses' in stage['error']
    assert 'local_update' not in stage and jobs._load(root/'pointcloud-retry')['pointcloud'] is None
    final=_advance(root,{'type':'finish','candidate_id':2})['output']
    assert final['artifacts']['map']==jobs._load(root)['pointcloud']['files']['map']


def _height_child(root,monkeypatch,with_failure=True):
    module=jobs.core();original=module.propose_road_corridors
    def propose(*args):
        p=json.loads(original(*args))
        if with_failure:
            for c in p['candidates']:
                for section in c['sections']:
                    if section['station_m']==6.:section['right'][2]-=.35
        return json.dumps(p)
    monkeypatch.setattr(module,'propose_road_corridors',propose)
    preview=_advance(root,{'type':'inspect_local_density','candidate_id':2,'gap_ids':[1],'bounds_xy':[2.,-1.6,8.4,1.6]})['local_point_observation']
    result=_advance(root,_local_density_action(preview));assert result['pointcloud_retry_result']['status']=='ready',result
    child=root/'pointcloud-retry';_advance(child,{'type':'inspect','candidate_ids':[1]})
    drafted=_advance(child,{'type':'draft','decisions':[{'candidate_id':1,'action':'include','from_m':4.,'to_m':8.,'reason':'Only the local missing interval'}]})
    assert drafted['draft_result']['status']=='audited_draft',drafted
    return child


def _height_preview(child,offset=0):
    return _advance(child,{'type':'inspect_heights','candidate_id':2,'offset':offset})['height_observation']


def _height_action(preview,dz=.075):
    return {'type':'edit_heights','candidate_id':2,'preview_file':preview['file'],
            'edits':[{'boundary_id':2,'vertex_index':1,'delta_z_m':dz,'reason':'Bounded Z hypothesis at the observed height mismatch; verify both estimators'}]}


def test_height_edit_keeps_xy_endpoints_metadata_and_supports_combined_patch(unused_frame_run,monkeypatch):
    from ca import mapping_heights as height
    root=unused_frame_run;child=_height_child(root,monkeypatch);before=jobs._load(child)['attempts'][1]
    original=json.loads(Path(before['files']['editable_map']['path']).read_text());before_bytes=Path(before['files']['editable_map']['path']).read_bytes()
    preview=_height_preview(child)
    assert [(r['boundary_id'],r['vertex_index']) for r in preview['vertices']]==[(2,1)]
    result=_advance(child,_height_action(preview));assert result['height_result']['status']=='audited_draft',result
    after=jobs.inspect_mapping_job(str(child))['attempts'][2]
    height._preserved(original,json.loads(Path(after['files']['editable_map']['path']).read_text()),_height_action(preview)['edits'])
    assert Path(before['files']['editable_map']['path']).read_bytes()==before_bytes
    assert result['remaining_attempts']==1 and json.loads(Path(after['height_checks']['path']).read_text())['passes']
    _advance(child,{'type':'inspect_patch','candidate_id':3,'gap_ids':[1],'offset':0})
    patched=_advance(child,{'type':'patch_gaps','candidate_id':3,'gap_ids':[1],'pairs':_patch_pairs(child)})
    assert patched['patch_result']['status']=='audited_draft' and patched['remaining_attempts']==0,patched
    comparison=_advance(root,{'type':'compare_retry','candidate_id':4})['retry_comparison']
    assert comparison['gained_source_length_m']==4 and comparison['lost_source_length_m']==0
    _advance(child,{'type':'finish','candidate_id':4})
    final=_advance(root,{'type':'finish_retry','candidate_id':4})['output']
    assert final['artifacts']['hd_height_checks']==after['height_checks']
    assert any('addition_height' in k for k in jobs._load(child)['attempts'][3]['patch_inputs'])


@pytest.mark.parametrize('fault',['large_delta','zero','nan','boolean','endpoint','xy','duplicate'])
def test_height_invalid_edits_do_not_spend_attempts(unused_frame_run,monkeypatch,fault):
    root=unused_frame_run;child=_height_child(root,monkeypatch);preview=_height_preview(child);action=_height_action(preview)
    row=action['edits'][0]
    if fault in ('large_delta','zero','nan','boolean'):row['delta_z_m']={'large_delta':.101,'zero':0.,'nan':float('nan'),'boolean':True}[fault]
    elif fault=='endpoint':row['vertex_index']=0
    elif fault=='xy':row['delta_x_m']=.01
    else:action['edits'].append(dict(row))
    before=(child/'run.json').read_bytes()
    with pytest.raises(ValueError):_advance(child,action)
    assert (child/'run.json').read_bytes()==before and len(jobs._load(child)['attempts'])==2


@pytest.mark.parametrize('fault',['unseen_page','tamper','unseen_preview'])
def test_height_requires_frozen_seen_preview_and_every_vertex(unused_frame_run,monkeypatch,fault):
    child=_height_child(unused_frame_run,monkeypatch);preview=_height_preview(child,99 if fault=='unseen_page' else 0)
    if fault=='tamper':Path(preview['file']['path']).write_text('{}')
    elif fault=='unseen_preview':
        run=json.loads((child/'run.json').read_text());run['history'][-1].pop('height_observation');jobs._save(child/'run.json',run)
    with pytest.raises(ValueError):_advance(child,_height_action(preview))
    assert len(jobs._load(child)['attempts'])==2


def test_height_failed_hypothesis_keeps_audits_and_root_baseline(unused_frame_run,monkeypatch):
    root=unused_frame_run;child=_height_child(root,monkeypatch);preview=_height_preview(child)
    result=_advance(child,_height_action(preview,-.075));assert result['height_result']['status']=='failed'
    attempt=jobs._load(child)['attempts'][2]
    assert not json.loads(Path(attempt['height_checks']['path']).read_text())['passes']
    assert Path(attempt['height_audits']['path']).is_file() and Path(attempt['height_trial']['path']).is_file()
    assert 'files' not in attempt and result['remaining_attempts']==1
    with pytest.raises(ValueError,match='one height trial'):_advance(child,_height_action(preview))
    final=_advance(root,{'type':'finish','candidate_id':2})['output']
    assert not final['pointcloud_retry_decision']['adopted'] and final['artifacts']['map']==jobs._load(root)['pointcloud']['files']['map']


def test_height_completed_trial_resumes_without_reexport(unused_frame_run,monkeypatch):
    from ca import mapping_run as runs
    child=_height_child(unused_frame_run,monkeypatch);preview=_height_preview(child);original=runs._diagnosis
    monkeypatch.setattr(runs,'_diagnosis',lambda *args:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):_advance(child,_height_action(preview))
    monkeypatch.setattr(runs,'_diagnosis',original)
    monkeypatch.setattr(runs.heights,'edit',lambda *args:pytest.fail('completed height export must not repeat'))
    resumed=_advance(child,{'type':'resume'})
    assert resumed['height_result']['status']=='audited_draft' and resumed['remaining_attempts']==1
    assert len(jobs._load(child)['attempts'])==3


def test_height_preview_has_no_edits_for_fully_supported_addition(unused_frame_run,monkeypatch):
    child=_height_child(unused_frame_run,monkeypatch,with_failure=False);preview=_height_preview(child)
    assert preview['vertices_total']==0 and preview['vertices']==[]
    with pytest.raises(ValueError):_advance(child,_height_action(preview))
    assert len(jobs._load(child)['attempts'])==2


def test_height_export_rejects_native_xy_mutation_and_keeps_point_map(unused_frame_run,monkeypatch):
    child=_height_child(unused_frame_run,monkeypatch);preview=_height_preview(child)
    point=jobs._load(child)['pointcloud']['files']['map'];point_bytes=Path(point['path']).read_bytes()
    native=jobs.core();original=native.edit_vector_map_relations
    def changed(*args):
        payload=json.loads(original(*args));ir=json.loads(payload['map_json'])
        ir['boundaries'][0]['geometry'][1][0]+=.01
        payload['map_json']=json.dumps(ir);return json.dumps(payload)
    monkeypatch.setattr(native,'edit_vector_map_relations',changed)
    result=_advance(child,_height_action(preview))['height_result']
    assert result['status']=='failed' and 'boundary XY/endpoints' in result['error']
    attempt=jobs._load(child)['attempts'][2]
    assert 'files' not in attempt and Path(attempt['height_trial']['path']).is_file()
    assert Path(point['path']).read_bytes()==point_bytes
    assert not (child/'candidate-03').exists()


@pytest.mark.parametrize('fault',['changed_samples','incomplete','consensus_failure'])
def test_height_gate_rejects_changed_or_incomplete_audits(unused_frame_run,monkeypatch,fault):
    child=_height_child(unused_frame_run,monkeypatch);preview=_height_preview(child)
    native=jobs.core();method='audit_vector_map_ground_consensus_details' if fault=='consensus_failure' else 'audit_vector_map_quality_details'
    original=getattr(native,method)
    def changed(*args):
        report=json.loads(original(*args));quality=report['quality']
        if fault=='changed_samples':quality['lanes'][0]['left']['samples']+=1
        elif fault=='incomplete':quality['problems_limited']=True
        else:quality['lanes'][0]['left']['end_supported']=False
        return json.dumps(report)
    monkeypatch.setattr(native,method,changed)
    result=_advance(child,_height_action(preview))
    assert result['height_result']['status']=='failed' and result['remaining_attempts']==1
    attempt=jobs._load(child)['attempts'][2];checks=json.loads(Path(attempt['height_checks']['path']).read_text())
    assert not checks['passes'] and checks['holds'] and 'files' not in attempt


def test_height_lineage_tamper_blocks_combined_patch_without_spending(unused_frame_run,monkeypatch):
    child=_height_child(unused_frame_run,monkeypatch);preview=_height_preview(child)
    _advance(child,_height_action(preview))
    _advance(child,{'type':'inspect_patch','candidate_id':3,'gap_ids':[1],'offset':0})
    attempt=jobs._load(child)['attempts'][2]
    Path(attempt['height_checks']['path']).write_text('{}')
    before=(child/'run.json').read_bytes()
    with pytest.raises(ValueError,match='changed'):
        _advance(child,{'type':'patch_gaps','candidate_id':3,'gap_ids':[1],'pairs':[]})
    assert (child/'run.json').read_bytes()==before and len(jobs._load(child)['attempts'])==3


def test_height_budget_reserves_the_combined_patch(unused_frame_run,monkeypatch):
    child=_height_child(unused_frame_run,monkeypatch);preview=_height_preview(child)
    job=jobs._load(child);job['max_attempts']=3;jobs._save(child/'job.json',job)
    before=(child/'run.json').read_bytes()
    with pytest.raises(ValueError,match='two remaining'):_advance(child,_height_action(preview))
    assert (child/'run.json').read_bytes()==before and len(jobs._load(child)['attempts'])==2
