"""Guard target masking and metrics against answer leakage and inflated success."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from vector_map_cross_scene_evaluate import mask_target, summarize_relations


def test_all_deferred_generation_is_frozen_before_reference_and_invents_no_roads(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import vector_map_cross_scene_evaluate as audit
    cloud=tmp_path/"cloud"
    cloud.write_bytes(b"source")
    reference=tmp_path/"reference"
    reference.write_bytes(b"reference")
    inputs=tmp_path/"inputs.json"
    inputs.write_text(json.dumps({"reference_inputs":[],"operator_paths":[[[0,0,0],[2,0,0]]],"build_options":{}}))
    out=tmp_path/"proof"
    def roads(source,cases,destination,commit):
        destination.mkdir()
        return {"after":{"final_map":None},"before":{"final_map":None}}
    def load_reference(path):
        freeze=json.loads((out/"generation-freeze.json").read_text())
        assert freeze["reference_inputs"]==[] and "discovery.json" in freeze["artifact_sha256"]
        return json.dumps({"map_json":json.dumps({"regulatory_elements":[],"crosswalks":[],"stop_lines":[],"traffic_signals":[]}),"report":{"import_issues":[]}})
    discovery={"candidates":[],"source_points":1,"windows":1,"detected_candidates":0,"limited":False,"unsupported_windows":1}
    native=SimpleNamespace(_core=SimpleNamespace(__file__=str(cloud)),edit_vector_map_relations=load_reference,
        discover_vector_map_features=lambda *args:json.dumps({"report":{"discovery":discovery}}))
    monkeypatch.setattr(audit,"evaluate_roads",roads)
    monkeypatch.setitem(sys.modules,"cloudanalyzer_core",native)
    report=audit.run(cloud,inputs,reference,out,"test")
    assert report["boundary_distance"] is None
    assert report["relations"]["existing"]["rules"]==0
    assert json.loads((out/"junction-preview.json").read_text())["status"]=="no_source_supported_roads"
    assert not (out/"component-input.json").exists()


def test_mask_removes_only_query_answer_without_inventing_marking_context():
    original = {"stop_lines": [{"id": 10}], "lanes": [{"id": 3}],
        "regulatory_elements": [
            {"id": 1, "rule": {"type": "traffic_light", "signals": [5], "stop_line": 10},
             "lanes": [3], "controlled_crosswalks": [7]},
            {"id": 2, "rule": {"type": "stop_line", "stop_line": 11}, "lanes": [4]}]}
    before = json.dumps(original, sort_keys=True)
    masked = mask_target(original, 1)
    assert json.dumps(original, sort_keys=True) == before
    assert masked["regulatory_elements"][0] == {"id":1,"rule":{"type":"traffic_light","signals":[5]},"lanes":[3]}
    assert masked["regulatory_elements"][1:] == original["regulatory_elements"][1:]
    assert masked["lanes"] == original["lanes"] and masked["stop_lines"] == original["stop_lines"]


def test_metrics_distinguish_wrong_targets_dropped_lanes_and_unresolved_rules():
    def candidate(target, lanes, eligible=True):
        return {"target_kind":"stop_line", "target_id":target,"lanes":lanes,"eligible":eligible}
    rows = [{"reference_stop":10,"reviewed_lanes":[3,4],"proposal":{
        "kind":"vehicle","limited":False,"eligible_count":3,
        "candidates":[candidate(10,[3]),candidate(11,[3,4]),candidate(10,[4,3]),candidate(12,[3,4],False)]}},
        {"reference_stop":13,"reviewed_lanes":[5],"proposal":{"kind":"mixed","limited":False,"eligible_count":0,"candidates":[]}}]
    r = summarize_relations(rows)
    assert r["matching_target_and_movement_pairs"] == r["rules_with_expected_candidate"] == 1
    assert r["different_reference_target_pairs"] == r["changed_movement_pairs"] == 1
    assert r["rules_without_eligible_candidate"] == r["mixed_kind_rules_held"] == 1


def test_native_connected_stop_cannot_replace_or_drop_reviewed_lanes(tmp_path):
    core = pytest.importorskip("cloudanalyzer_core")
    m = {"format":"vectormap-ir","version":1,
        "boundaries":[{"id":1,"kind":{"type":"virtual"},"geometry":[[0,2,0],[20,2,0]]},
          {"id":2,"kind":{"type":"virtual"},"geometry":[[0,-2,0],[20,-2,0]]},
          {"id":40,"kind":{"type":"virtual"},"geometry":[[20,-2,0],[0,-2,0]]},
          {"id":41,"kind":{"type":"virtual"},"geometry":[[20,2,0],[0,2,0]]}],
        "lanes":[{"id":3,"kind":"driving","left":1,"right":2},{"id":43,"kind":"driving","left":40,"right":41}],
        "topology":[{"lane":3,"successors":[43]},{"lane":43,"predecessors":[3]}],
        "stop_lines":[{"id":15,"geometry":[[6,-2,0],[6,2,0]]},{"id":16,"geometry":[[7,-2,0],[7,2,0]]}],
        "traffic_signals":[{"id":22,"kind":"vehicle","geometry":[[6,-.5,5],[6,.5,5]],"height":.5}],
        "regulatory_elements":[{"id":23,"rule":{"type":"traffic_light","signals":[22]},"lanes":[3]},
          {"id":32,"rule":{"type":"stop_line","stop_line":15},"lanes":[3]},
          {"id":33,"rule":{"type":"stop_line","stop_line":16},"lanes":[43]}]}
    path = tmp_path / "movements.json"
    path.write_text(json.dumps(m),encoding="utf-8")
    before = path.read_bytes()
    payload = json.loads(core.propose_vector_map_relations(str(path),23))
    p = payload["report"]["proposal"]
    wrong = next(c for c in p["candidates"] if c["target_id"] == 16)
    assert wrong["road_context"] == "connected_lanes" and not wrong["eligible"] and wrong["lanes"] == [3]
    adoption = {"rule_id":23,"map_snapshot":p["map_snapshot"],"candidate_key":"stop_line:16"}
    with pytest.raises(ValueError,match="rejected"):
        core.propose_vector_map_relations(str(path),23,json.dumps(adoption))
    assert path.read_bytes() == before
    adoption["candidate_key"] = "stop_line:15"
    saved = json.loads(json.loads(core.propose_vector_map_relations(str(path),23,json.dumps(adoption)))["map_json"])
    assert next(r for r in saved["regulatory_elements"] if r["id"]==23)["lanes"] == [3]
    m["regulatory_elements"][0]["lanes"] = [3,43]
    path.write_text(json.dumps(m),encoding="utf-8")
    p = json.loads(core.propose_vector_map_relations(str(path),23))["report"]["proposal"]
    assert p["eligible_count"] == 0
    assert all(c["lanes"] == [3,43] for c in p["candidates"])
