"""Endpoint revisions keep changed geometry explicit and preserve all other evidence."""
import copy
import json
from pathlib import Path

import pytest

from ca import mapping_connections as links, mapping_endpoints as endpoints
from ca import mapping_job as jobs, mapping_run as runs
from tests.test_mapping_job import _attach_lane_native, _corridor_report, _run_layout, job_backend


def _advance(root, action):
    return runs.advance_mapping_run(str(root), action, "Explicitly inspect the fixed anchor and bounded endpoint revision",
                                    runs.inspect_mapping_run(str(root))["revision"])


@pytest.fixture
def endpoint_run(job_backend, tmp_path, monkeypatch, request):
    source, _ = job_backend
    native = _attach_lane_native()
    if not hasattr(native, "inspect_corridor_lane_directions"):
        pytest.skip("native endpoint direction preflight unavailable")
    module = jobs.core()
    for name in ("connect_vector_map_junctions", "inspect_corridor_lane_directions",
                 "audit_vector_map_quality_details", "audit_vector_map_ground_consensus_details"):
        setattr(module, name, getattr(native, name))
    proposal = _corridor_report()
    template = proposal["candidates"][0]
    candidates = []
    for cid, stations in ((1, [0., 2., 4.]), (17, [4., 6., 8.]), (29, [9., 10.])):
        c = copy.deepcopy(template)
        c.update(id=cid, from_m=stations[0], to_m=stations[-1])
        c["sections"] = []
        for station in stations:
            x = station + (getattr(request, "param", .2) if cid == 17 and station == 4. else 0.)
            c["sections"].append({"station_m": station, "center": [x, 0., 1.], "left": [x, 1., 1.],
                "right": [x, -1., 1.], "left_evidence": "support_gap", "right_evidence": "support_gap",
                "intersects_trajectory": True})
        candidates.append(c)
    proposal.update(candidates=candidates, with_candidate_station_length_m=9., trajectory_covered_station_length_m=9.,
        without_candidate_station_length_m=1., deferred_intervals=[{"from_m": 8., "to_m": 9., "reason": "unresolved_source_band"}])
    module.propose_road_corridors = lambda *args: json.dumps(proposal)

    def fixed(folder, out, **kwargs):
        root = Path(out)
        root.mkdir()
        cloud, trajectory, graph = root / "map.xyz", root / "poses.txt", root / "map.g2o"
        cloud.write_text("".join(f"{x/5} {y/5} 1\n" for x in range(-5, 56) for y in range(-15, 16)))
        trajectory.write_text("1 0 0 0 0 1 0 0 0 0 1 1\n1 0 0 10 0 1 0 0 0 0 1 1\n")
        graph.write_text("unused graph")
        return {"map_points": 1891, "outputs": {"map": str(cloud), "kitti": str(trajectory), "g2o": str(graph)}}

    monkeypatch.setattr(jobs, "fix_session", fixed)
    root = tmp_path / "endpoint-run"
    runs.start_mapping_run(str(source), str(root), _run_layout())
    _advance(root, {"type": "inspect", "candidate_ids": [1, 17, 29]})
    result = _advance(root, {"type": "draft", "decisions": [
        {"candidate_id": cid, "action": "include", "reason": "Retain the observed source section vertices"} for cid in (1, 17, 29)]})
    assert result["draft_result"]["status"] == "audited_draft", result
    parent = jobs._load(root)["attempts"][-1]
    ir = json.loads(Path(parent["files"]["editable_map"]["path"]).read_text())
    ids = [p["lane"] for p in links._parent(jobs._load(root), parent["id"])[3]]
    return root, parent, ir, ids, native


def _preview(fixture):
    root, parent, _, ids, _ = fixture
    return _advance(root, {"type": "inspect_endpoint_alignment", "candidate_id": parent["id"],
                          "from": ids[0], "to": ids[1], "revise_lane": ids[1]})["endpoint_observation"]


def _align(fixture, observation):
    root, parent, _, _, _ = fixture
    return _advance(root, {"type": "align_endpoints", "candidate_id": parent["id"], "preview_file": observation["file"]})


def test_native_alignment_preserves_other_vertices_and_can_connect_finish_and_bundle(endpoint_run, tmp_path, monkeypatch):
    from ca.mapping_bundle import export_mapping_run, inspect_mapping_bundle
    root, parent, original, ids, _ = endpoint_run
    original_bytes = {p: p.read_bytes() for p in (root / "candidate-02").iterdir() if p.is_file()}
    observation = _preview(endpoint_run)
    assert observation["eligible"], observation
    assert observation["anchor_check"]["passes"] and observation["width_check"]["passes"]
    assert observation["boundary_displacements_xyz_m"] == pytest.approx({"left": .2, "right": .2})
    assert runs.inspect_mapping_run(str(root))["remaining_attempts"] == 4
    with monkeypatch.context() as cached:
        cached.setattr(jobs.core(), "inspect_corridor_lane_directions", lambda *a: pytest.fail("cached preview must not rerun native planning"))
        assert _preview(endpoint_run) == observation
    result = _align(endpoint_run, observation)
    assert result["endpoint_result"]["status"] == "audited_draft", result
    revised = jobs._load(root)["attempts"][-1]
    actual = json.loads(Path(revised["files"]["editable_map"]["path"]).read_text())
    assert len(actual["lanes"]) == len(original["lanes"])
    for side in ("left", "right"):
        old, points, _ = endpoints._side(original, ids[1], side)
        boundary, new, _ = endpoints._side(actual, ids[1], side)
        assert new[0].tolist() == endpoints._side(original, ids[0], side)[1][-1].tolist()
        assert new[1:].tolist() == points[1:].tolist()
        assert boundary["id"] == old["id"]
    assert revised["extent"] == parent["extent"]
    assert revised["routes"]["after"]["connected_components"] == 2
    assert {p: p.read_bytes() for p in original_bytes} == original_bytes
    report = json.loads(Path(revised["files"]["report"]["path"]).read_text())
    assert report["endpoint_alignment"]["four_fresh_audits_complete"]
    assert json.loads(Path(revised["endpoint_checks"]["path"]).read_text())["passes"]
    connections = _advance(root, {"type": "inspect_connections", "candidate_id": revised["id"], "offset": 0})["connection_observation"]
    c = next(c for c in connections["candidates"] if (c["from"], c["to"]) == (ids[1], ids[2]))
    connected = _advance(root, {"type": "connect", "candidate_id": revised["id"], "pairs": [
        {"from": c["from"], "to": c["to"], "reason": "Connect the separately inspected remaining station gap"}]})
    assert connected["connect_result"]["status"] == "audited_draft", connected
    final = jobs._load(root)["attempts"][-1]
    assert final["routes"]["after"]["longest_route_station_span_m"] == 10.
    assert final["endpoint_preview"] == revised["endpoint_preview"]
    _advance(root, {"type": "finish", "candidate_id": final["id"]})
    bundle = tmp_path / "review.zip"
    export_mapping_run(str(root), str(bundle), "Generated native integration test fixture; no accuracy claim")
    inspection = inspect_mapping_bundle(str(bundle))
    assert inspection["verified_files"] > 0


@pytest.mark.parametrize("fault,hold", [
    ("anchor", "anchor_endpoint_not_supported_in_all_four_audits"),
    ("direction", "whole_revised_lane_requires_deferral_or_cuts"),
    ("limited_direction", "whole_revised_lane_requires_deferral_or_cuts"),
    ("width", "below_fixed_minimum_width"),
    ("path", "recorded_path_outside_revised_lane"),
])
def test_endpoint_preview_holds_do_not_spend_shared_attempts(endpoint_run, monkeypatch, fault, hold):
    root, _, _, _, _ = endpoint_run
    if fault == "anchor":
        monkeypatch.setattr(endpoints, "_anchor_check", lambda *a: {"passes": False})
    elif fault in {"direction", "limited_direction"}:
        method = jobs.core().inspect_corridor_lane_directions
        def held(*args):
            response = json.loads(method(*args))
            if fault == "direction":
                response["segments"][0]["planned_pieces"] = [[0, 1], [1, 2]]
            else:
                response["limited"] = True
            return json.dumps(response)
        monkeypatch.setattr(jobs.core(), "inspect_corridor_lane_directions", held)
    elif fault == "width":
        original = links._width_check
        monkeypatch.setattr(links, "_width_check", lambda *a: {**original(*a), "passes": False})
    else:
        monkeypatch.setattr(links, "_outside_by", lambda *a: .26)
    observation = _preview(endpoint_run)
    assert not observation["eligible"] and hold in observation["holds"]
    before = runs.inspect_mapping_run(str(root))
    with pytest.raises(ValueError, match="eligible"):
        _align(endpoint_run, observation)
    assert runs.inspect_mapping_run(str(root))["revision"] == before["revision"]
    assert before["remaining_attempts"] == 4


def test_alignment_rejects_unseen_and_changed_previews(endpoint_run):
    root, parent, _, ids, _ = endpoint_run
    unseen = endpoints.preview(root, parent["id"], ids[0], ids[1], ids[1])
    with pytest.raises(ValueError, match="through this run"):
        _align(endpoint_run, unseen)
    observation = _preview(endpoint_run)
    Path(observation["file"]["path"]).write_text("changed inspected evidence")
    with pytest.raises(ValueError, match="changed"):
        runs.inspect_mapping_run(str(root))
    assert len(jobs._load(root)["attempts"]) == 2


@pytest.mark.parametrize("endpoint_run", [.8], indirect=True)
def test_endpoint_displacement_cannot_exceed_the_unchanged_ground_radius(endpoint_run):
    observation = _preview(endpoint_run)
    assert not observation["eligible"]
    assert "endpoint_displacement_exceeds_fixed_limit" in observation["holds"]
    assert observation["effective_displacement_limit_m"] == .75


def test_endpoint_trial_cannot_expand_the_shared_attempt_budget(endpoint_run):
    root, _, _, _, _ = endpoint_run
    observation = _preview(endpoint_run)
    job = jobs._load(root)
    job["max_attempts"] = len(job["attempts"])
    jobs._save(root / "job.json", job)
    with pytest.raises(ValueError, match="remaining shared HD attempt"):
        _align(endpoint_run, observation)
    assert len(jobs._load(root)["attempts"]) == 2


def test_alignment_resume_reuses_completed_attempt(endpoint_run, monkeypatch):
    root, _, _, _, _ = endpoint_run
    observation = _preview(endpoint_run)
    original = runs._align_endpoints
    with monkeypatch.context() as interrupted:
        def stop(*args):
            original(*args)
            raise KeyboardInterrupt("interrupt after the audited endpoint trial was saved")
        interrupted.setattr(runs, "_align_endpoints", stop)
        with pytest.raises(KeyboardInterrupt):
            _align(endpoint_run, observation)
    before = jobs._load(root)
    result = _advance(root, {"type": "resume"})
    assert result["endpoint_result"]["status"] == "audited_draft"
    assert jobs._load(root)["attempts"] == before["attempts"]
    assert result["remaining_attempts"] == 3


def test_new_source_failure_keeps_failed_evidence_without_publishing(endpoint_run, monkeypatch):
    root, _, _, ids, native = endpoint_run
    observation = _preview(endpoint_run)
    def damaged(cloud, path):
        audit = json.loads(native.audit_vector_map_ground_consensus_details(cloud, path))
        row = next(l for l in audit["quality"]["lanes"] if l["lane"] == ids[1])["left"]
        row.update(supported=row["samples"] - 1, start_supported=False, height_mismatches=1)
        return json.dumps(audit)
    monkeypatch.setattr(jobs.core(), "audit_vector_map_ground_consensus_details", damaged)
    result = _align(endpoint_run, observation)
    assert result["endpoint_result"]["status"] == "failed"
    failed = jobs._load(root)["attempts"][-1]
    assert failed["status"] == "failed" and "endpoint_audits" in failed and "endpoint_checks" in failed
    assert not (root / "candidate-03").exists()
    assert result["remaining_attempts"] == 3


def test_revision_guards_cover_reversed_storage_private_ownership_and_original_station_gaps(endpoint_run):
    _, _, original, ids, _ = endpoint_run
    pieces = [{"lane": lid, "from_m": a, "to_m": b} for lid, a, b in zip(ids, [0., 4., 9.], [4., 8., 10.])]
    with pytest.raises(ValueError, match="zero original-station gap"):
        endpoints._case(original, pieces, ids[1], ids[2], ids[2])
    reversed_ir = copy.deepcopy(original)
    lane = next(l for l in reversed_ir["lanes"] if l["id"] == ids[1])
    for side in ("left", "right"):
        boundary = endpoints._side(reversed_ir, ids[1], side)[0]
        boundary["geometry"].reverse()
        lane[side] = {"boundary": boundary["id"], "reversed": True}
    case = endpoints._case(reversed_ir, pieces, ids[0], ids[1], ids[1])
    revised = endpoints._revise(reversed_ir, case)
    assert endpoints._side(revised, ids[1], "left")[1][0].tolist() == endpoints._side(original, ids[0], "left")[1][-1].tolist()
    revised["topology"] = copy.deepcopy(original.get("topology", []))
    with pytest.raises(ValueError, match="accumulate"):
        endpoints._case(revised, pieces, ids[0], ids[1], ids[1])
    lane["left"] = next(l for l in reversed_ir["lanes"] if l["id"] == ids[0])["left"]
    with pytest.raises(ValueError, match="private"):
        endpoints._case(reversed_ir, pieces, ids[0], ids[1], ids[1])


def test_unchanged_failure_locations_are_exact_even_when_totals_do_not_change(endpoint_run):
    _, parent, _, ids, _ = endpoint_run
    before = json.loads(Path(parent["quality_report"]["path"]).read_text())
    after = copy.deepcopy(before)
    for saved in (before, after):
        saved["editable"]["quality"]["problems"] = [{"lane": ids[0], "curve": "left", "points": [[1., 1., 1.]]}]
    after["editable"]["quality"]["problems"][0]["points"][0][0] = 2.
    checks = endpoints._quality_checks(before, after, set(ids), ids[1])
    assert not checks["passes"] and "editable:unchanged_failure_locations_changed" in checks["holds"]
