"""Resolve recorded-drive shortcuts while retaining native and audit evidence."""
import copy
import json
from pathlib import Path

import pytest

from ca import mapping_connections as connections, mapping_job as jobs, mapping_run as runs
from tests.test_mapping_job import _attach_lane_native, _corridor_report, _run_layout, job_backend


def _pieces(count=3):
    return [{"lane": i, "from_m": float(i * 4), "to_m": float(i * 4 + 2)} for i in range(1, count + 1)]


def test_recorded_order_resolves_only_known_nonconsecutive_native_choices():
    candidate = {"from": 1, "to": 2, "ambiguous": True}
    offered = {(1, 2): candidate, (1, 3): {}}
    before = copy.deepcopy((candidate, offered))
    proof = connections._branch_check(candidate, offered, _pieces())
    assert proof["passes"] and proof["native_ambiguous"]
    assert proof["resolution"] == "only_nonconsecutive_alternatives"
    assert proof["alternatives"] == [{"from": 1, "to": 3, "from_interval_m": [4., 6.],
        "to_interval_m": [12., 14.], "original_station_adjacent": False}]
    assert (candidate, offered) == before


@pytest.mark.parametrize("offered,pair", [
    ({(1, 2): {}}, (1, 2)),  # Native ambiguity without the alternative evidence.
    ({(1, 2): {}, (1, 99): {}}, (1, 2)),  # No original station identity.
    ({(1, 3): {}, (1, 2): {}}, (1, 3)),  # The selected shortcut itself skips a piece.
])
def test_incomplete_or_nonconsecutive_branch_choices_stay_held(offered, pair):
    proof = connections._branch_check({"from": pair[0], "to": pair[1], "ambiguous": True}, offered, _pieces())
    assert not proof["passes"] and proof["resolution"] == "unresolved"


def test_bounded_branch_evidence_does_not_resolve_uninspected_alternatives():
    count = connections.BRANCH_ALTERNATIVE_LIMIT + 3
    offered = {(1, i): {} for i in range(2, count + 1)}
    proof = connections._branch_check({"from": 1, "to": 2, "ambiguous": True}, offered, _pieces(count))
    assert not proof["passes"] and proof["alternatives_limited_in_response"]
    assert proof["alternatives_total"] == count - 2
    assert len(proof["alternatives"]) == connections.BRANCH_ALTERNATIVE_LIMIT


@pytest.fixture
def branch_run(job_backend, tmp_path, monkeypatch):
    """Three real native lane pieces: discovery offers a shortcut over the middle."""
    source, _ = job_backend
    native = _attach_lane_native()
    module = jobs.core()
    for name in ("connect_vector_map_junctions", "audit_vector_map_quality_details", "audit_vector_map_ground_consensus_details"):
        setattr(module, name, getattr(native, name))
    report = _corridor_report()
    original = report["candidates"][0]
    candidates = []
    for cid, start in ((1, 0.), (17, 4.), (29, 8.)):
        candidate = copy.deepcopy(original)
        candidate.update(id=cid, from_m=start, to_m=start + 2.,
                         sections=[s for s in candidate["sections"] if start <= s["station_m"] <= start + 2.])
        candidates.append(candidate)
    report.update(candidates=candidates, with_candidate_station_length_m=6., without_candidate_station_length_m=4.,
                  trajectory_covered_station_length_m=6., deferred_intervals=[
                      {"from_m": a, "to_m": a + 2., "reason": "unresolved_source_band"} for a in (2., 6.)])
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
    root = tmp_path / "branch-run"
    runs.start_mapping_run(str(source), str(root), _run_layout())
    _advance(root, {"type": "inspect", "candidate_ids": [1, 17, 29]})
    result = _advance(root, {"type": "draft", "decisions": [
        {"candidate_id": cid, "action": "include", "reason": "Retain observed pieces in recorded order"} for cid in (1, 17, 29)]})
    assert result["draft_result"]["status"] == "audited_draft", result
    return root, native


def _advance(root, action):
    return runs.advance_mapping_run(str(root), action, "Inspect the original station order and source evidence",
                                    runs.inspect_mapping_run(str(root))["revision"])


def test_native_shortcut_resolution_connects_only_consecutive_pieces_and_roundtrips(branch_run, monkeypatch):
    from ca.mapping_patch import _preserved
    root, _ = branch_run
    parent = jobs.inspect_mapping_job(str(root))["attempts"][-1]
    original = json.loads(Path(parent["files"]["editable_map"]["path"]).read_text())
    original_bytes = {p: p.read_bytes() for p in (root / "candidate-02").iterdir() if p.is_file()}
    observation = _advance(root, {"type": "inspect_connections", "candidate_id": 2, "offset": 0})["connection_observation"]
    assert len(observation["candidates"]) == 2
    for candidate in observation["candidates"]:
        assert candidate["ambiguous"] and candidate["branch_check"]["passes"]
        assert candidate["branch_check"]["resolution"] == "only_nonconsecutive_alternatives"
        assert all(a["original_station_adjacent"] is False for a in candidate["branch_check"]["alternatives"])
        assert candidate["width_check"]["passes"] and candidate["recorded_path_outside_m"] == 0.
    with monkeypatch.context() as cached:
        cached.setattr(jobs.core(), "connect_vector_map_junctions", lambda *args: pytest.fail("retained preview must not rediscover branches"))
        assert _advance(root, {"type": "inspect_connections", "candidate_id": 2, "offset": 0})["connection_observation"] == observation
    pairs = [{"from": c["from"], "to": c["to"], "reason": "Only consecutive recorded pieces; inspected competing shortcuts skip the middle piece"}
             for c in observation["candidates"]]
    result = _advance(root, {"type": "connect", "candidate_id": 2, "pairs": pairs})
    assert result["connect_result"]["status"] == "audited_draft", result
    current = jobs.inspect_mapping_job(str(root))["attempts"][-1]
    ir = json.loads(Path(current["files"]["editable_map"]["path"]).read_text())
    _preserved(original, ir, connections.edges(ir) - connections.edges(original))
    assert current["routes"]["after"]["connected_components"] == 1
    assert current["routes"]["after"]["longest_route_station_span_m"] == 10.
    assert current["extent"] == parent["extent"] and not current["extent"]["passes_requested_extent"]
    assert json.loads(Path(current["connection_checks"]["path"]).read_text())["passes"]
    assert {p: p.read_bytes() for p in original_bytes} == original_bytes
    report = json.loads(Path(current["files"]["report"]["path"]).read_text())
    assert report["lane_roundtrip_verified"] and report["topology_roundtrip_verified"]
    assert report["both_connector_estimators_verified"]


@pytest.mark.parametrize("fault", ["below_fixed_minimum_width", "recorded_path_outside_connection"])
def test_resolved_branch_still_requires_unchanged_width_and_recorded_path(branch_run, fault):
    root, native = branch_run
    def damaged(*args):
        payload = json.loads(native.connect_vector_map_junctions(*args))
        candidate = payload["report"]["junctions"]["candidates"][0]
        for p in candidate["left"]:
            p[1] = .5 if fault == "below_fixed_minimum_width" else p[1] + 3.
        for p in candidate["right"]:
            p[1] = -.5 if fault == "below_fixed_minimum_width" else p[1] + 3.
        return json.dumps(payload)
    jobs.core().connect_vector_map_junctions = damaged
    result = _advance(root, {"type": "inspect_connections", "candidate_id": 2, "offset": 0})
    held = next(c for c in result["connection_observation"]["rejected"] if fault in c["holds"])
    assert held["branch_check"]["passes"] and "ambiguous_native_branch" not in held["holds"]
    assert result["remaining_attempts"] == 4
