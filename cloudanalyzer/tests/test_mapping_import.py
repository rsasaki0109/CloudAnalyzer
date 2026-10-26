"""External outputs enter the existing drafting loop without rerunning SLAM."""
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
from typer.testing import CliRunner

from ca import mapping_import as imports, mapping_job as jobs, mapping_run as runs
from cloudanalyzer_cli.main import app
from tests.test_mapping_job import job_backend, _attach_lane_native, _run_layout


@pytest.fixture
def external_pair(tmp_path, monkeypatch, job_backend):
    cloud = tmp_path / "external.ply"
    cloud.write_text("ply\nformat ascii 1.0\nelement vertex 4\nproperty double x\n"
                     "property double y\nproperty double z\nend_header\n"
                     "0 -1 1\n0 1 1\n10 -1 1\n10 1 1\n")
    trajectory = tmp_path / "external.tum"
    trajectory.write_text("100 0 0 1 0 0 0 1\n101 10 0 1 0 0 0 1\n")
    provenance = {"backend": "fixture-export", "version": "fixed-fixture",
                  "source": "test fixture", "license": "test fixture",
                  "coordinate_frame": "fixture_map", "length_unit": "metres", "up_axis": "z",
                  "pose_convention": "world_from_sensor", "time_unit": "seconds",
                  "time_basis": "fixture seconds", "configuration": {"deskew": False}}
    monkeypatch.setattr(imports.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    return cloud, trajectory, provenance


def _start(pair, root):
    cloud, trajectory, provenance = pair
    return runs.start_mapping_run_from_slam(str(cloud), str(trajectory), str(root),
                                           provenance, _run_layout())


def test_import_preserves_map_times_frame_and_provenance_without_running_slam(external_pair, tmp_path, monkeypatch):
    cloud, trajectory, provenance = external_pair
    before = cloud.read_bytes(), trajectory.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("external imports must not run odometry or fusion")

    monkeypatch.setattr(jobs, "odometry", forbidden)
    monkeypatch.setattr(jobs, "fix_session", forbidden)
    root = tmp_path / "import"
    run = _start(external_pair, root)
    assert run["status"] == "needs_agent" and run["revision"] == 0
    assert run["pointcloud"]["quality_status"] == "imported_unverified"
    assert run["pointcloud"]["coordinate_frame"] == "fixture_map"
    assert run["pointcloud"]["files"]["map"] == jobs._artifact(cloud)
    assert run["pointcloud"]["files"]["source_trajectory"] == jobs._artifact(trajectory)
    assert not (root / "map.ply").exists() and not (root / "odometry").exists()
    np.testing.assert_array_equal(np.loadtxt(root / "trajectory.txt")[:, [3, 7, 11]], [[0, 0, 1], [10, 0, 1]])
    declaration = json.loads((root / "slam-import.json").read_text())
    assert declaration["producer_declaration"] == provenance
    assert not declaration["alignment_applied"] and not declaration["timestamps_rebased"]
    assert (cloud.read_bytes(), trajectory.read_bytes()) == before
    assert run["pointcloud_retry_allowed"] is False
    assert set(run["action_contract"]) == runs.FROZEN_MAP_ACTIONS
    assert jobs.inspect_mapping_job(str(root))["storage"]["point_map_copied"] is False
    assert run["pointcloud"]["input_recording_coverage"] == "not_verified"


@pytest.mark.parametrize("rows", [
    "0 0 0 0 0 0 0 1\n0 1 0 0 0 0 0 1\n",
    "1 0 0 0 0 0 0 1\n0 1 0 0 0 0 0 1\n",
    "0 0 0 0 0 0 0 0\n1 1 0 0 0 0 0 1\n",
    "0 nan 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n",
    "0 0 0 0 0 0 0 1\n",
    "0 0 0 0 0 0 1\n1 1 0 0 0 0 1\n",
    "0 -1e308 0 0 0 0 0 1\n1 1e308 0 0 0 0 0 1\n",
])
def test_invalid_poses_are_rejected_without_creating_an_output(external_pair, tmp_path, rows):
    external_pair[1].write_text(rows)
    root = tmp_path / "import"
    with pytest.raises(ValueError, match="TUM"):
        _start(external_pair, root)
    assert not root.exists()


def test_frame_units_are_explicit_and_not_silently_converted(external_pair, tmp_path):
    external_pair[2]["length_unit"] = "centimetres"
    with pytest.raises(ValueError, match="metre-frame"):
        _start(external_pair, tmp_path / "import")
    assert not (tmp_path / "import").exists()


def test_disk_guard_rejects_before_loading_or_creating_outputs(external_pair, tmp_path, monkeypatch):
    monkeypatch.setattr(imports.shutil, "disk_usage", lambda _: SimpleNamespace(free=imports.MIN_FREE_BYTES))
    monkeypatch.setattr(imports, "load_frame", lambda _: pytest.fail("low-space import loaded the map"))
    with pytest.raises(ValueError, match="disk space"):
        _start(external_pair, tmp_path / "import")
    assert not (tmp_path / "import").exists()


def test_input_size_guard_does_not_create_or_copy_large_files(external_pair, tmp_path, monkeypatch):
    monkeypatch.setattr(imports, "MAX_MAP_BYTES", 8)
    with pytest.raises(ValueError, match="size limit"):
        _start(external_pair, tmp_path / "import")
    assert not (tmp_path / "import").exists()


def test_changed_external_input_and_recording_actions_are_rejected(external_pair, tmp_path):
    root = tmp_path / "import"
    _start(external_pair, root)
    before = (root / "run.json").read_bytes(), (root / "job.json").read_bytes()
    with pytest.raises(ValueError, match="frozen external"):
        runs.advance_mapping_run(str(root), {"type": "retry_pointcloud", "candidate_id": 2, "gap_ids": [1],
            "options": {"scan_voxel_m": .2, "map_voxel_m": .1}}, "Unavailable raw log", 0)
    assert ((root / "run.json").read_bytes(), (root / "job.json").read_bytes()) == before
    from ca import mapping_retry
    with pytest.raises(ValueError, match="raw-recording evidence"):
        mapping_retry._raw_source(root, jobs._load(root))
    assert not (root / "gap-source").exists()
    external_pair[1].write_text("100 0 0 0 0 0 0 1\n101 20 0 0 0 0 0 1\n")
    with pytest.raises(ValueError, match="changed"):
        runs.inspect_mapping_run(str(root))


def test_external_pair_drafts_finishes_and_exports_with_existing_quality_holds(external_pair, tmp_path):
    _attach_lane_native()
    from ca import mapping_bundle
    root = tmp_path / "import"
    run = _start(external_pair, root)
    run = runs.advance_mapping_run(str(root), {"type": "inspect", "candidate_ids": [1]}, "Read source sections", run["revision"])
    run = runs.advance_mapping_run(str(root), {"type": "draft", "decisions": [
        {"candidate_id": 1, "action": "include", "reason": "Retain inspected source extent"}]}, "Fixed layout", run["revision"])
    assert run["draft_result"]["status"] == "audited_draft"
    run = runs.advance_mapping_run(str(root), {"type": "finish", "candidate_id": 2}, "Deliver with source holds", run["revision"])
    assert run["output"]["status"] == "draft_needs_review" and not run["output"]["deployment_ready"]
    assert run["output"]["diagnosis"]["review_holds"]["traces_total"] > 0
    assert {"map", "trajectory", "graph", "source_trajectory", "slam_import", "hd_map"} <= run["output"]["artifacts"].keys()
    native = pytest.importorskip("cloudanalyzer_core")
    graph = native.PoseGraph.from_g2o((root / "poses.g2o").read_text())
    np.testing.assert_allclose(np.asarray(graph.poses())[:, :3], np.loadtxt(root / "trajectory.txt").reshape(-1, 3, 4), atol=1e-10, rtol=0)
    target = tmp_path / "external-review.zip"
    mapping_bundle.export_mapping_run(str(root), str(target), "Test fixture")
    inspected = mapping_bundle.inspect_mapping_bundle(str(target))
    assert not inspected["deployment_ready"]
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        imported = json.loads(archive.read(manifest["roles"]["slam_import"]))
    assert imported["producer_declaration"] == external_pair[2]


def test_cli_and_mcp_expose_external_import(external_pair, tmp_path):
    cloud, trajectory, provenance = external_pair
    declaration, layout = tmp_path / "provenance.json", tmp_path / "layout.json"
    declaration.write_text(json.dumps(provenance))
    layout.write_text(json.dumps(_run_layout()))
    result = CliRunner().invoke(app, ["mapping-import", str(cloud), "--trajectory", str(trajectory),
        "--out", str(tmp_path / "cli-import"), "--provenance", str(declaration), "--layout", str(layout)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "needs_agent"
    from ca.mcp_server import TOOLS
    assert runs.start_mapping_run_from_slam in TOOLS


def _partial_external_draft(external_pair, root, monkeypatch, pieces=1):
    from copy import deepcopy
    from tests.test_mapping_job import _corridor_report

    _attach_lane_native()
    report = _corridor_report()
    sections = deepcopy(report["candidates"][0]["sections"])
    if pieces > 1:
        sections = [{**deepcopy(sections[0]), "station_m": float(x),
                     "center": [float(x), 0., 1.], "left": [float(x), 1., 1.],
                     "right": [float(x), -1., 1.]} for x in range(0, 39, 2)]
    end = sections[-1]["station_m"]
    report["trajectory_length_m"] = end
    report["candidates"][0].update(sections=sections, from_m=0., to_m=end)
    report["profiles"] = [{"station_m": s["station_m"], "trajectory": s["center"],
        "heading_usable": True, "reference_ground_height_m": 1.,
        "bands": [{**deepcopy(s), "support_span_m": 2., "path_level_supported": True}]} for s in sections]
    monkeypatch.setattr(jobs.core(), "propose_road_corridors", lambda *args: json.dumps(report))
    run = _start(external_pair, root)
    run = runs.advance_mapping_run(str(root), {"type": "inspect", "candidate_ids": [1]}, "Read original sections", run["revision"])
    decisions = [{"candidate_id": 1, "action": "include", "from_m": i * 4.,
                  "to_m": i * 4. + (4. if pieces == 1 else 2.), "reason": "Retain observed partial ranges"}
                 for i in range(pieces)]
    run = runs.advance_mapping_run(str(root), {"type": "draft", "decisions": decisions}, "Partial draft with visible gaps", run["revision"])
    assert run["draft_result"]["status"] == "audited_draft", run["draft_result"]
    return run


def test_external_gap_inspection_uses_saved_reports_without_loading_points_or_processing(external_pair, tmp_path, monkeypatch):
    from ca import mapping_retry as retries, mapping_hd_repair

    root = tmp_path / "import"
    run = _partial_external_draft(external_pair, root, monkeypatch)
    before = (root / "job.json").read_bytes(), external_pair[0].read_bytes(), external_pair[1].read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("saved-map gap inspection must not read points, process geometry/audits, or extract raw frames")

    for name in ("propose_road_corridors", "audit_vector_map_quality_details", "audit_vector_map_ground_consensus_details"):
        monkeypatch.setattr(jobs.core(), name, forbidden)
    monkeypatch.setattr(jobs.core(), "read", forbidden, raising=False)
    monkeypatch.setattr(retries, "_raw_source", forbidden)
    monkeypatch.setattr(retries, "materialize_pointcloud_bag", forbidden)
    run = runs.advance_mapping_run(str(root), {"type": "inspect_gaps", "candidate_id": 2, "offset": 0},
        "Inspect frozen-map gaps without raw evidence", run["revision"])
    observation = run["gap_observation"]
    assert observation["schema"] == "cloudanalyzer.mapping_map_gaps.v1"
    assert observation["gaps_total"] == 1 and observation["next_offset"] is None
    gap, = observation["gaps"]
    assert (gap["from_m"], gap["to_m"]) == (4., 10.)
    assert gap["profiles"] and gap["width_observations"][0]["layout_width_preview"]["ranges"]
    assert not observation["raw_evidence_available"] and not observation["fresh_source_audit_performed"]
    assert "raw_manifest" not in json.loads(Path(observation["file"]["path"]).read_text())
    assert run["remaining_attempts"] == 4
    assert before == ((root / "job.json").read_bytes(), external_pair[0].read_bytes(), external_pair[1].read_bytes())
    assert not (root / "gap-source").exists()
    again = runs.advance_mapping_run(str(root), {"type": "inspect_gaps", "candidate_id": 2, "offset": 0},
        "Reuse the same frozen evidence page", run["revision"])
    assert again["gap_observation"] == observation
    assert len(list(root.glob("map-gaps-*.json"))) == 1
    # A copied map-only observation must still be refused if a raw-manifest field
    # is added; its explicit schema cannot become evidence for a repair trial.
    decorated = json.loads(Path(observation["file"]["path"]).read_text())
    decorated.update(raw_manifest=jobs._artifact(external_pair[0]),
                     pointcloud_options={"scan_voxel_m": .4, "map_voxel_m": .2})
    decorated_path = tmp_path / "decorated-map-observation.json"
    decorated_path.write_text(json.dumps(decorated))
    for evidence in (observation["file"], jobs._artifact(decorated_path)):
        for validator in (lambda: retries.validate(evidence, [gap["id"]], {"scan_voxel_m": .2, "map_voxel_m": .1}),
                          lambda: mapping_hd_repair.validate(evidence, 2, [gap["id"]])):
            with pytest.raises(ValueError, match="raw-recording gap evidence"):
                validator()
    stored = (root / "run.json").read_bytes(), (root / "job.json").read_bytes()
    for action in ({"type": "retry_pointcloud", "candidate_id": 2, "gap_ids": [1], "options": {"scan_voxel_m": .2, "map_voxel_m": .1}},
                   {"type": "repair_hd", "candidate_id": 2, "gap_ids": [1]},
                   {"type": "inspect_hd_plan", "candidate_id": 2, "gap_ids": [1], "offset": 0}):
        with pytest.raises(ValueError, match="frozen external"):
            runs.advance_mapping_run(str(root), action, "Saved evidence cannot authorize unsupported trials", again["revision"])
    assert stored == ((root / "run.json").read_bytes(), (root / "job.json").read_bytes())
    from ca.mapping_bundle import export_mapping_run, inspect_mapping_bundle

    finished = runs.advance_mapping_run(str(root), {"type": "finish", "candidate_id": 2},
        "Deliver with saved gap observations and all original review holds", again["revision"])
    assert finished["output"]["status"] == "draft_needs_review"
    bundle = tmp_path / "review-with-map-gaps.zip"
    export_mapping_run(str(root), str(bundle), "External fixture; saved gap observations only")
    assert inspect_mapping_bundle(str(bundle))["integrity_verified"]


def test_external_gap_pages_keep_global_ids_and_report_empty_tail(external_pair, tmp_path, monkeypatch):
    root = tmp_path / "import"
    run = _partial_external_draft(external_pair, root, monkeypatch, pieces=9)
    run = runs.advance_mapping_run(str(root), {"type": "inspect_gaps", "candidate_id": 2, "offset": 0}, "Read first gap page", run["revision"])
    page = run["gap_observation"]
    assert page["gaps_total"] == 9 and len(page["gaps"]) == 8 and page["next_offset"] == 8
    assert [g["id"] for g in page["gaps"]] == list(range(1, 9))
    run = runs.advance_mapping_run(str(root), {"type": "inspect_gaps", "candidate_id": 2, "offset": 8}, "Read last gap page", run["revision"])
    assert [g["id"] for g in run["gap_observation"]["gaps"]] == [9]
    assert run["gap_observation"]["next_offset"] is None
    run = runs.advance_mapping_run(str(root), {"type": "inspect_gaps", "candidate_id": 2, "offset": 99}, "Explicit empty page", run["revision"])
    assert run["gap_observation"]["gaps"] == [] and run["gap_observation"]["gaps_total"] == 9
    assert run["remaining_attempts"] == 4


@pytest.mark.parametrize("guard", ["size", "disk"])
def test_external_gap_storage_guards_preserve_job_and_create_no_evidence(external_pair, tmp_path, monkeypatch, guard):
    from ca import mapping_map_gaps

    root = tmp_path / "import"
    _partial_external_draft(external_pair, root, monkeypatch)
    before = (root / "job.json").read_bytes(), (root / "run.json").read_bytes()
    if guard == "size":
        monkeypatch.setattr(mapping_map_gaps, "MAX_REPORT_BYTES", 8)
    else:
        monkeypatch.setattr(mapping_map_gaps.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    with pytest.raises(ValueError, match="page budget|disk space"):
        mapping_map_gaps.inspect(root, 2, 0, runs._load(root)["layout_file"])
    assert not list(root.glob("map-gaps-*"))
    assert before == ((root / "job.json").read_bytes(), (root / "run.json").read_bytes())


def test_external_gap_observation_tampering_is_rejected_on_resume(external_pair, tmp_path, monkeypatch):
    from ca import mapping_map_gaps
    root = tmp_path / "import"
    run = _partial_external_draft(external_pair, root, monkeypatch)
    run = runs.advance_mapping_run(str(root), {"type": "inspect_gaps", "candidate_id": 2, "offset": 0}, "Read saved gaps", run["revision"])
    path = Path(run["gap_observation"]["file"]["path"])
    value = json.loads(path.read_text())
    value["raw_evidence_available"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed"):
        runs.inspect_mapping_run(str(root))
    with pytest.raises(ValueError, match="inputs changed"):
        mapping_map_gaps.inspect(root, 2, 0, runs._load(root)["layout_file"])


def test_external_gap_context_and_width_previews_report_every_truncation(external_pair, tmp_path, monkeypatch):
    from copy import deepcopy
    from ca import mapping_map_gaps

    root = tmp_path / "import"
    _partial_external_draft(external_pair, root, monkeypatch)
    parent = jobs._load(root)["attempts"][-1]
    proposal = json.loads(Path(parent["corridor_proposal"]["path"]).read_text())
    section = proposal["candidates"][0]["sections"][0]
    prototype = proposal["profiles"][0]
    proposal["profiles"] = [{**deepcopy(prototype), "station_m": float(i * 2),
                             "bands": deepcopy(prototype["bands"]) * 32} for i in range(181)]
    proposal["candidates"] = [{**deepcopy(proposal["candidates"][0]), "id": i + 1, "from_m": 0., "to_m": 358.,
        "sections": [{**deepcopy(section), "station_m": float(k * 2)} for k in range(180)]} for i in range(10)]
    gap = mapping_map_gaps._gap({"from_m": 0., "to_m": 360., "status": "not_reviewed"}, 1, proposal, _run_layout(), parent)
    assert len(gap["profiles"]) == 8 and gap["profiles_total"] == 181 and gap["profiles_limited"]
    assert all(len(p["bands"]) == 16 and p["bands_total"] == 32 and p["bands_limited"] for p in gap["profiles"])
    assert gap["width_candidates_total"] == 10 and len(gap["width_observations"]) == 8 and gap["width_candidates_limited"]
    assert all(c["layout_width_preview"]["checked_section_pairs"] == 127
               and c["layout_width_preview"]["uninspected_section_pairs"] == 52 for c in gap["width_observations"])


@pytest.mark.parametrize("artifact", ["proposal", "audits"])
def test_external_gap_inspection_rejects_changed_source_reports(external_pair, tmp_path, monkeypatch, artifact):
    root = tmp_path / "import"
    run = _partial_external_draft(external_pair, root, monkeypatch)
    parent = jobs._load(root)["attempts"][-1]
    path = Path(parent["corridor_proposal" if artifact == "proposal" else "quality_report"]["path"])
    path.write_text(path.read_text() + "\n")
    before = (root / "job.json").read_bytes(), (root / "run.json").read_bytes()
    with pytest.raises(ValueError, match="changed"):
        runs.advance_mapping_run(str(root), {"type": "inspect_gaps", "candidate_id": 2, "offset": 0},
            "Reject stale source reports before recording an observation", run["revision"])
    assert not list(root.glob("map-gaps-*"))
    assert before == ((root / "job.json").read_bytes(), (root / "run.json").read_bytes())


def test_read_only_preflight_reports_geometry_without_claiming_accuracy(external_pair, tmp_path):
    cloud, trajectory, provenance = external_pair
    before = set(tmp_path.iterdir())
    report = imports.inspect_slam_inputs(str(cloud), str(trajectory), provenance)
    assert report['map_points'] == 4
    assert report['pose_rows'] == 2
    assert report['path_length_m'] == pytest.approx(10)
    assert report['map_bounds']['minimum'] == [0, -1, 1]
    assert report['frame_alignment_verified'] is False
    assert report['accuracy_verified'] is False
    assert set(tmp_path.iterdir()) == before


def test_preflight_rejects_nonmonotonic_external_clock(external_pair):
    cloud, trajectory, provenance = external_pair
    trajectory.write_text('100 0 0 1 0 0 0 1\n100 10 0 1 0 0 0 1\n')
    with pytest.raises(ValueError, match='increase strictly'):
        imports.inspect_slam_inputs(str(cloud), str(trajectory), provenance)
