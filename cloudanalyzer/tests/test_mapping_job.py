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


@pytest.fixture
def job_backend(tmp_path, monkeypatch):
    extension = tmp_path / "native.bin"
    extension.write_bytes(b"fixed native implementation")
    support = {"supported": 2}
    audit = {"quality": {"lanes": [{"center": support, "left": support, "right": support}],
                         "low_support_lanes": [3], "omitted_lanes": [], "malformed_lanes": [],
                         "limited": False, "sampled_points": 9}, "validation": {"issues": [], "counts": {"errors": 0}}}
    module = SimpleNamespace(__file__=str(extension), __version__="test",
                             audit_vector_map_quality=lambda *args: json.dumps(audit))
    monkeypatch.setattr(jobs, "core", lambda: module)

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
            (root / name).write_text(key)
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
    assert started["next_actions"] == ["generate_mapping_candidate"]
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
    with pytest.raises(ValueError, match="complete"):
        jobs.select_mapping_candidate(str(root), 1, "Incomplete coverage is not a pass")
    assert jobs.inspect_mapping_job(str(root))["selected"] is None


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
    if not hasattr(native, "audit_vector_map_quality"):
        pytest.skip("installed core predates source audit")
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
    generated = jobs.generate_mapping_candidate(str(root), OPTIONS, "A straight, source-supported single-lane hypothesis")
    candidate = generated["attempts"][0]
    assert candidate["status"] == "audited_draft", candidate
    assert candidate["quality"]["needs_review"] == []
    assert candidate["reopened_quality"] == candidate["quality"]
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
