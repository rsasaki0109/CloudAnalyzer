"""Audited additions retain the original map, source failures and shared budget."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ca import mapping_append as appends, mapping_connections as connections, mapping_job as jobs, mapping_run as runs
from ca.mapping_bundle import export_mapping_run, inspect_mapping_bundle
from ca.mapping_patch import _preserved
from tests.test_mapping_job import job_backend, _attach_lane_native, _corridor_report, _run_layout
from tests.test_mapping_import import external_pair


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _step(root, action):
    return runs.advance_mapping_run(str(root), action, "Explicit fixture decision", runs.inspect_mapping_run(str(root))["revision"])


@pytest.fixture
def prepared_run(external_pair, tmp_path):
    native = _attach_lane_native()
    module = jobs.core()
    module.audit_vector_map_quality_details = native.audit_vector_map_quality_details
    module.audit_vector_map_ground_consensus_details = native.audit_vector_map_ground_consensus_details
    cloud, trajectory, producer = external_pair
    points = [f"{x / 4} {y / 4} 1" for x in range(41) for y in range(-8, 9)]
    cloud.write_text("ply\nformat ascii 1.0\nelement vertex " + str(len(points))
        + "\nproperty double x\nproperty double y\nproperty double z\nend_header\n" + "\n".join(points) + "\n")

    def proposal(cloud, trajectory, options):
        report = _corridor_report()
        report["protocol"] = {"options": json.loads(options)}
        if json.loads(options).get("association", "all_supported_bands") == "all_supported_bands":
            report["ambiguous_intervals"] = [{"from_m": 4., "to_m": 8., "reason": "competing_bands"}]
            report["ambiguous_station_length_m"] = 4.
        report["profiles"] = [{"station_m": s["station_m"], "trajectory": s["center"], "heading_usable": True,
            "reference_ground_height_m": 1., "bands": [{**s, "support_span_m": 2., "path_level_supported": True}]}
            for s in report["candidates"][0]["sections"]]
        return json.dumps(report)
    module.propose_road_corridors = proposal

    def prepare(interval=(4., 6.), seen=True, attempts=6, preview=True):
        root = tmp_path / "run"
        runs.start_mapping_run_from_slam(str(cloud), str(trajectory), str(root), producer, _run_layout(), max_attempts=attempts)
        _step(root, {"type": "inspect", "candidate_ids": [1]})
        baseline = _step(root, {"type": "draft", "decisions": [
            {"candidate_id": 1, "action": "include", "from_m": lo, "to_m": hi, "reason": "Retain original piece"}
            for lo, hi in ((0., 4.), (8., 10.))]})
        assert baseline["draft_result"]["status"] == "audited_draft"
        if seen:
            _step(root, {"type": "inspect_gaps", "candidate_id": 2, "offset": 0})
        _step(root, {"type": "refine", "association": "trajectory_containing"})
        _step(root, {"type": "inspect", "candidate_ids": [1]})
        addition = _step(root, {"type": "draft", "decisions": [{"candidate_id": 1, "action": "include",
            "from_m": interval[0], "to_m": interval[1], "reason": "Audited gap-only addition"}]})
        assert addition["draft_result"]["status"] == "audited_draft"
        if seen and preview:
            _step(root, {"type": "inspect_append", "baseline_candidate_id": 2, "addition_candidate_id": 4, "gap_ids": [1]})
        return root
    return prepare, module, cloud, trajectory


ACTION = {"type": "append_draft", "baseline_candidate_id": 2, "addition_candidate_id": 4, "gap_ids": [1],
          "pairs": [{"from": 3, "to": 9, "reason": "Explicit exact adjacent endpoints"}]}


def _frozen(root):
    job = jobs.inspect_mapping_job(str(root))
    artifacts = [*job["pointcloud"]["files"].values(),
                 *[v for a in job["attempts"] for v in a["files"].values()]]
    return {a["path"]: Path(a["path"]).read_bytes() for a in artifacts}


def _unchanged(snapshot):
    assert all(Path(path).read_bytes() == data for path, data in snapshot.items())


def test_append_preserves_inputs_ids_curves_routes_and_portable_sources(prepared_run, tmp_path, monkeypatch):
    prepare, module, _, _ = prepared_run
    root = prepare()
    before = _frozen(root)
    baseline = _read(root / "candidate-02/vector_map.json")
    result = _step(root, ACTION)
    assert result["append_result"]["status"] == "audited_draft", result["append_result"]
    assert result["remaining_attempts"] == 1
    appended = _read(root / "candidate-05/vector_map.json")
    _preserved(baseline, appended, {(3, 9)})
    assert len(appended["lanes"]) == 3
    report = _read(root / "candidate-05/report.json")
    assert report["extraction"]["generated_length"] == 8.
    assert report["extent"]["trajectory_length_m"] == 10.
    assert report["extent"]["minimum_retained_fraction"] == .9
    assert report["append_checks"]["passes"] and report["retained_geometry_and_connections"]
    assert report["source_proposals"]["baseline"] != report["source_proposals"]["addition"]
    assert all(i["source_ambiguous"] for i in report["station_disposition"] if i["origin"] == "addition")
    assert sum(i["to_m"] - i["from_m"] for i in report["station_disposition"]) == 10.
    assert report["routes"]["after"]["connected_components"] == 2
    _unchanged(before)

    def forbidden(*args, **kwargs):
        raise AssertionError("saved append inspection must not invoke native processing")
    for name in ("read", "propose_road_corridors", "edit_vector_map_relations",
                 "audit_vector_map_quality_details", "audit_vector_map_ground_consensus_details"):
        monkeypatch.setattr(module, name, forbidden, raising=False)
    observation = _step(root, {"type": "inspect_gaps", "candidate_id": 5, "offset": 0})["gap_observation"]
    assert [(g["from_m"], g["to_m"]) for g in observation["gaps"]] == [(6., 8.)]
    assert all(g["origin"] == "baseline" for g in observation["gaps"])
    _step(root, {"type": "finish", "candidate_id": 5})
    bundle = tmp_path / "review.zip"
    export_mapping_run(str(root), str(bundle), "Unit fixture; no real recording")
    inspected = inspect_mapping_bundle(str(bundle))
    assert inspected["integrity_verified"] and not inspected["deployment_ready"]
    import zipfile
    with zipfile.ZipFile(bundle) as archive:
        data = json.loads(archive.read("manifest.json"))
    assert "append_input_addition_proposal" in data["roles"]
    assert "append_input_baseline_geometry_report" in data["roles"]
    assert "append_checks" in data["roles"] and "append_audits" in data["roles"]
    assert "append_input_native" not in data["roles"]
    _unchanged(before)


@pytest.mark.parametrize("changes,pattern", [
    ({"baseline_candidate_id": True}, "two audited"), ({"addition_candidate_id": 2}, "distinct"),
    ({"gap_ids": []}, "1..8"), ({"gap_ids": [True]}, "1..8"),
    ({"gap_ids": [1, 1]}, "distinct"), ({"gap_ids": [2]}, "inspect every"),
])
def test_invalid_append_decisions_change_no_state_or_budget(prepared_run, changes, pattern):
    root = prepared_run[0]()
    before = [(root / name).read_bytes() for name in ("job.json", "run.json")]
    with pytest.raises(ValueError, match=pattern):
        _step(root, {**ACTION, **changes})
    assert before == [(root / name).read_bytes() for name in ("job.json", "run.json")]


@pytest.mark.parametrize("condition,pattern", [("overlap", "overlaps"), ("unseen", "inspect baseline gaps"), ("budget", "remaining")])
def test_append_requires_gap_only_seen_additions_and_shared_budget(prepared_run, condition, pattern):
    root = prepared_run[0](interval=(2., 6.) if condition == "overlap" else (4., 6.), seen=condition != "unseen",
                           attempts=4 if condition == "budget" else 6, preview=condition != "overlap")
    before = [(root / name).read_bytes() for name in ("job.json", "run.json")]
    with pytest.raises(ValueError, match=pattern):
        _step(root, ACTION)
    assert before == [(root / name).read_bytes() for name in ("job.json", "run.json")]


def test_storage_guard_precedes_append_journal_and_attempt(prepared_run, monkeypatch):
    root = prepared_run[0]()
    before = [(root / name).read_bytes() for name in ("job.json", "run.json")]
    monkeypatch.setattr(appends.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    with pytest.raises(ValueError, match="disk space"):
        _step(root, ACTION)
    assert before == [(root / name).read_bytes() for name in ("job.json", "run.json")]


@pytest.mark.parametrize("regression", ["retained_endpoint", "added_endpoint"])
def test_failed_source_checks_retain_evidence_and_original_draft(prepared_run, monkeypatch, regression):
    prepare, module, _, _ = prepared_run
    root = prepare()
    before = _frozen(root)
    for name in ("audit_vector_map_quality_details", "audit_vector_map_ground_consensus_details"):
        original = getattr(module, name)
        def changed(*args, _original=original):
            audit = json.loads(_original(*args))
            target = min(l["lane"] for l in audit["quality"]["lanes"]) if regression == "retained_endpoint" else max(l["lane"] for l in audit["quality"]["lanes"])
            lane = next(l for l in audit["quality"]["lanes"] if l["lane"] == target)
            for trace in ("center", "left", "right"):
                lane[trace]["end_supported"] = False
            return json.dumps(audit)
        monkeypatch.setattr(module, name, changed)
    result = _step(root, ACTION)
    assert result["append_result"]["status"] == "failed" and result["remaining_attempts"] == 1
    assert not (root / "candidate-05").exists()
    checks = _read(root / "candidate-05-append-checks.json")
    assert not checks["passes"] and checks["holds"]
    assert (root / "candidate-05-append-audits.json").exists()
    _unchanged(before)
    assert _step(root, {"type": "finish", "candidate_id": 2})["output"]["candidate_id"] == 2


def test_resume_after_completed_append_spends_no_extra_attempt(prepared_run, monkeypatch):
    root = prepared_run[0]()
    real = runs._append
    def interrupted(*args):
        real(*args)
        raise KeyboardInterrupt("after atomic append completion")
    monkeypatch.setattr(runs, "_append", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _step(root, ACTION)
    assert jobs.inspect_mapping_job(str(root))["remaining_attempts"] == 1
    monkeypatch.setattr(runs, "_append", real)
    result = _step(root, {"type": "resume"})
    assert result["append_result"]["status"] == "audited_draft" and result["remaining_attempts"] == 1
    assert len(jobs.inspect_mapping_job(str(root))["attempts"]) == 5


@pytest.mark.parametrize("artifact", ["candidate-05-append-checks.json", "geometry-01/report.json", "corridor-proposals-trajectory.json", "append-preview-*.json"])
def test_tampered_append_sources_and_checks_are_rejected_on_inspection(prepared_run, artifact):
    root = prepared_run[0]()
    _step(root, ACTION)
    path = next(root.glob(artifact))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        runs.inspect_mapping_run(str(root))
    with pytest.raises(ValueError, match="changed"):
        jobs.select_mapping_candidate(str(root), 5, "Reject changed append evidence")


def test_append_refuses_recursive_edits(prepared_run):
    root = prepared_run[0]()
    _step(root, ACTION)
    before = [(root / name).read_bytes() for name in ("job.json", "run.json")]
    with pytest.raises(ValueError, match="one draft append"):
        _step(root, ACTION)
    assert before == [(root / name).read_bytes() for name in ("job.json", "run.json")]



def _appended_connection_parent(prepared_run, attempts=6):
    prepare, module, _, _ = prepared_run
    root = prepare(attempts=attempts)
    module.connect_vector_map_junctions = _attach_lane_native().connect_vector_map_junctions
    result = _step(root, ACTION)
    assert result["append_result"]["status"] == "audited_draft", result
    return root, module


def _connection_action():
    return {"type": "connect", "candidate_id": 5,
            "pairs": [{"from": 9, "to": 6, "reason": "Inspected the open gap after the audited addition"}]}


def test_appended_connections_keep_source_extent_lineage_and_portable_review(prepared_run, tmp_path, monkeypatch):
    root, module = _appended_connection_parent(prepared_run)
    before = _frozen(root)
    original = _read(root / "candidate-05/vector_map.json")
    original_report = _read(root / "candidate-05/report.json")
    original_job = (root / "job.json").read_bytes()
    observation = _step(root, {"type": "inspect_connections", "candidate_id": 5, "offset": 0})["connection_observation"]
    assert [(c["from"], c["to"]) for c in observation["candidates"]] == [(9, 6)]
    assert (root / "job.json").read_bytes() == original_job
    assert observation["baseline_routes"]["connected_components"] == 2
    def forbidden(*args, **kwargs):
        raise AssertionError("cached connection inspection repeated native discovery")
    with monkeypatch.context() as cached:
        cached.setattr(module, "connect_vector_map_junctions", forbidden)
        repeated = _step(root, {"type": "inspect_connections", "candidate_id": 5, "offset": 0})["connection_observation"]
    assert repeated == observation and (root / "job.json").read_bytes() == original_job
    connected = _step(root, _connection_action())
    assert connected["connect_result"]["status"] == "audited_draft", connected
    assert connected["remaining_attempts"] == 0
    current = jobs.inspect_mapping_job(str(root))["attempts"][-1]
    report = _read(current["files"]["report"]["path"])
    ir = _read(current["files"]["editable_map"]["path"])
    added = report["routes"]["added"]
    assert len(added) == 1 and (added[0]["from"], added[0]["to"]) == (9, 6)
    extra_edges = {(9, added[0]["lane"]), (added[0]["lane"], 6)}
    _preserved(original, ir, extra_edges)
    assert connections.edges(ir) == connections.edges(original) | extra_edges
    assert report["routes"]["after"]["connected_components"] == 1
    assert report["routes"]["after"]["longest_route_station_span_m"] == 10.
    assert report["extraction"]["generated_length"] == 8.
    assert current["extent"] == jobs.inspect_mapping_job(str(root))["attempts"][4]["extent"]
    assert not current["extent"]["passes_requested_extent"]
    assert _read(current["connection_checks"]["path"])["passes"]
    for key in ("source_proposals", "append_inputs", "append_checks", "baseline_candidate_id", "addition_candidate_id",
                "selected_gap_ids", "explicit_endpoint_pairs", "added_lane_id_map", "station_disposition", "options"):
        assert report[key] == original_report[key], key
    assert set(report["lane_minimum_width_m"]) == {str(l["id"]) for l in ir["lanes"]}
    assert set(report["lane_minimum_width_m"].values()) == {1.5}
    assert any("both proposal sources" in note for note in connected["connect_result"]["diagnosis"]["investigations"])
    after = _step(root, {"type": "inspect_connections", "candidate_id": 6, "offset": 0})["connection_observation"]
    assert after["candidates_total"] == 0
    gap = _step(root, {"type": "inspect_gaps", "candidate_id": 6, "offset": 0})["gap_observation"]
    assert [(g["from_m"], g["to_m"]) for g in gap["gaps"]] == [(6., 8.)]
    _step(root, {"type": "finish", "candidate_id": 6})
    bundle = tmp_path / "connected-review.zip"
    export_mapping_run(str(root), str(bundle), "Unit fixture; no real recording")
    inspected = inspect_mapping_bundle(str(bundle))
    assert inspected["integrity_verified"] and not inspected["deployment_ready"]
    import zipfile
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        delivered_audits = json.loads(archive.read(manifest["roles"]["hd_source_audits"]))
    assert delivered_audits == _read(current["connection_audits"]["path"])
    assert {"append_input_addition_proposal", "append_input_baseline_geometry_report", "append_checks", "append_audits",
            "hd_connection_checks", "hd_source_audits", "hd_connection_proposal"} <= manifest["roles"].keys()
    native_sha = jobs._load(root)["runtime"]["native"]["extension"]["sha256"]
    assert all(f["sha256"] != native_sha for f in manifest["files"])
    _unchanged(before)


@pytest.mark.parametrize("condition", ["unseen", "unseen_page", "unoffered", "budget"])
def test_appended_connections_require_seen_pairs_and_shared_budget(prepared_run, condition):
    root, _ = _appended_connection_parent(prepared_run, attempts=5 if condition == "budget" else 6)
    if condition != "unseen":
        _step(root, {"type": "inspect_connections", "candidate_id": 5, "offset": 8 if condition == "unseen_page" else 0})
    action = _connection_action()
    if condition == "unoffered":
        action["pairs"][0]["from"] = 3
    before = [(root / name).read_bytes() for name in ("job.json", "run.json")]
    with pytest.raises(ValueError, match="inspect|offered|remaining"):
        _step(root, action)
    assert before == [(root / name).read_bytes() for name in ("job.json", "run.json")]


@pytest.mark.parametrize("artifact", ["candidate-05-append-checks.json", "candidate-05-append-audits.json",
    "geometry-01/report.json", "corridor-proposals-trajectory.json", "append-preview-*.json", "connections-05.json"])
def test_appended_connections_reject_changed_ancestry_before_spending(prepared_run, artifact):
    root, _ = _appended_connection_parent(prepared_run)
    _step(root, {"type": "inspect_connections", "candidate_id": 5, "offset": 0})
    before = [(root / name).read_bytes() for name in ("job.json", "run.json")]
    path = next(root.glob(artifact))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        _step(root, _connection_action())
    assert before == [(root / name).read_bytes() for name in ("job.json", "run.json")]


@pytest.mark.parametrize("regression", ["retained_endpoint", "connector_endpoint"])
def test_appended_connection_failure_retains_inputs_and_audit_evidence(prepared_run, monkeypatch, regression):
    root, module = _appended_connection_parent(prepared_run)
    _step(root, {"type": "inspect_connections", "candidate_id": 5, "offset": 0})
    before = _frozen(root)
    for name in ("audit_vector_map_quality_details", "audit_vector_map_ground_consensus_details"):
        original = getattr(module, name)
        def changed(*args, _original=original):
            audit = json.loads(_original(*args))
            ids = [l["lane"] for l in audit["quality"]["lanes"]]
            target = min(ids) if regression == "retained_endpoint" else max(ids)
            lane = next(l for l in audit["quality"]["lanes"] if l["lane"] == target)
            for trace in ("center", "left", "right"):
                lane[trace]["end_supported"] = False
            return json.dumps(audit)
        monkeypatch.setattr(module, name, changed)
    result = _step(root, _connection_action())
    assert result["connect_result"]["status"] == "failed" and result["remaining_attempts"] == 0
    assert not (root / "candidate-06").exists()
    assert not _read(root / "candidate-06-connection-checks.json")["passes"]
    assert (root / "candidate-06-connection-audits.json").exists()
    _unchanged(before)
    assert _step(root, {"type": "finish", "candidate_id": 5})["output"]["candidate_id"] == 5


def test_appended_connection_resume_reuses_completed_attempt(prepared_run, monkeypatch):
    root, module = _appended_connection_parent(prepared_run)
    _step(root, {"type": "inspect_connections", "candidate_id": 5, "offset": 0})
    real = runs._connect
    def interrupted(*args):
        real(*args)
        raise KeyboardInterrupt("after completed connection")
    monkeypatch.setattr(runs, "_connect", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _step(root, _connection_action())
    assert jobs.inspect_mapping_job(str(root))["remaining_attempts"] == 0
    monkeypatch.setattr(runs, "_connect", real)
    def forbidden(*args, **kwargs):
        raise AssertionError("completed connection invoked native processing again")
    monkeypatch.setattr(module, "connect_vector_map_junctions", forbidden)
    result = _step(root, {"type": "resume"})
    assert result["connect_result"]["status"] == "audited_draft" and result["remaining_attempts"] == 0
    assert len(jobs.inspect_mapping_job(str(root))["attempts"]) == 6



def test_raw_runs_do_not_advertise_or_authorize_external_draft_append(job_backend, tmp_path):
    source, _ = job_backend
    root = tmp_path / "raw"
    run = runs.start_mapping_run(str(source), str(root), _run_layout())
    assert "append_draft" not in run["action_contract"]
    before = [(root / name).read_bytes() for name in ("job.json", "run.json")]
    with pytest.raises(ValueError, match="imported external"):
        _step(root, ACTION)
    assert before == [(root / name).read_bytes() for name in ("job.json", "run.json")]


def test_append_preview_uses_saved_inputs_without_native_calls_or_attempts(prepared_run, monkeypatch):
    prepare, module, _, _ = prepared_run
    root = prepare(preview=False)
    before = (root / "job.json").read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("append preview invoked native processing")
    for name in ("read", "propose_road_corridors", "edit_vector_map_relations",
                 "audit_vector_map_quality_details", "audit_vector_map_ground_consensus_details"):
        monkeypatch.setattr(module, name, forbidden, raising=False)
    action = {k: v for k, v in ACTION.items() if k != "pairs"}
    action["type"] = "inspect_append"
    first = _step(root, action)["append_observation"]
    second = _step(root, action)["append_observation"]
    assert first == second and before == (root / "job.json").read_bytes()
    assert [(p["from"], p["to"]) for p in first["candidates"]] == [(3, 9)]
    assert not first["protocol"]["native_processing_performed"]
    assert first["protocol"]["shared_hd_attempts_spent"] == 0


@pytest.mark.parametrize("condition", ["unseen", "omitted", "unoffered"])
def test_append_rejects_unseen_omitted_or_unoffered_endpoint_pairs(prepared_run, condition):
    root = prepared_run[0](preview=condition != "unseen")
    action = {**ACTION}
    if condition == "omitted":
        action["pairs"] = []
    elif condition == "unoffered":
        action["pairs"] = [{"from": 6, "to": 9, "reason": "Not offered by inspected endpoints"}]
    before = [(root / name).read_bytes() for name in ("job.json", "run.json")]
    with pytest.raises(ValueError, match="inspect append endpoints|every exact inspected"):
        _step(root, action)
    assert before == [(root / name).read_bytes() for name in ("job.json", "run.json")]


def test_only_explicitly_inspected_pairs_can_extend_retained_routes(prepared_run):
    root = prepared_run[0](interval=(4., 8.))
    before = [(root / name).read_bytes() for name in ("job.json", "run.json")]
    with pytest.raises(ValueError, match="every exact inspected"):
        _step(root, ACTION)
    assert before == [(root / name).read_bytes() for name in ("job.json", "run.json")]
    action = {**ACTION, "pairs": [*ACTION["pairs"], {"from": 9, "to": 6, "reason": "Explicit observed tail endpoint"}]}
    result = _step(root, action)
    assert result["append_result"]["status"] == "audited_draft", result["append_result"]
    report = _read(root / "candidate-05/report.json")
    assert report["extraction"]["generated_length"] == 10.
    assert report["routes"]["after"]["connected_components"] == 1
    assert report["routes"]["after"]["longest_route_station_span_m"] == 10.
    assert report["explicit_endpoint_pairs"] == action["pairs"]


def test_append_preview_size_guard_counts_the_saved_indented_page(prepared_run, monkeypatch):
    root = prepared_run[0](preview=False)
    before = (root / "job.json").read_bytes()
    real = appends._joins
    def limited(prepared):
        rows = real(prepared)
        expected = {"schema": "cloudanalyzer.mapping_append_preview.v1", "baseline_candidate_id": 2,
            "addition_candidate_id": 4, "gap_ids": [1], "inputs": prepared["inputs"], "protocol": appends.PREVIEW_PROTOCOL,
            "candidates": rows, "note": "Exact endpoint observations only. Select every listed pair explicitly or retain the input drafts; source audits are checked during append."}
        # Keep the compact representation below the limit but the actual page
        # above it, so omitting indentation from the size check cannot pass.
        compact = len(json.dumps(expected).encode())
        monkeypatch.setitem(appends.PREVIEW_PROTOCOL, "maximum_bytes", compact + 100)
        return rows
    monkeypatch.setattr(appends, "_joins", limited)
    with pytest.raises(ValueError, match="2 MiB budget"):
        _step(root, {"type": "inspect_append", "baseline_candidate_id": 2, "addition_candidate_id": 4, "gap_ids": [1]})
    assert before == (root / "job.json").read_bytes()
    assert not list(root.glob("append-preview-*.json"))
