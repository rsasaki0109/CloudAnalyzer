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
        return {"scans": out, "trajectory": "unused", "gravity": None, "frames": 4, "path_length_m": 10}

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
