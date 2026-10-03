"""Guard common-source comparisons against omission and target switching."""
import copy
import sys
from pathlib import Path
import pytest
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from vector_map_anchor_evaluate import paired, assert_profile_geometry


def profile(y=0., start=0., end=1.):
    return {"roads": [{"reference": [[start, 0, 2], [end, 0, 2]],
                       "boundaries": [[[start, y, 2], [end, y, 2]]]}]}


def test_after_points_keep_the_before_survey_target():
    target = cKDTree([[0, 0], [.5, 0], [1, 0], [0, 2], [.5, 2], [1, 2]])
    r = paired(profile(.4), profile(1.9), target)
    assert r["samples"] == 3 and r["common_source_path_m"] == 1
    assert r["before"]["mean_xy_m"] == pytest.approx(.4)
    assert r["after"]["mean_xy_m"] == pytest.approx(1.9)  # Retargeting would misleadingly give .1.


def test_deferred_intervals_are_separate_from_the_paired_accuracy():
    before = profile(.4)
    before["roads"] += profile(5., 1., 3.)["roads"]
    r = paired(before, profile(.3), cKDTree([[0, 0], [.5, 0], [1, 0]]))
    assert r["common_source_path_m"] == 1 and r["before_only_source_path_m"] == 2
    assert r["before"]["mean_xy_m"] == pytest.approx(.4)
    r = paired(profile(), profile(start=2., end=3.), cKDTree([[0, 0]]))
    assert r["samples"] == 0 and r["before"] is r["after"] is None


def test_ambiguous_intervals_changed_slots_and_nonbuilt_profiles_are_rejected():
    p = profile()
    repeated = copy.deepcopy(p)
    repeated["roads"] *= 2
    with pytest.raises(ValueError, match="ambiguous"):
        paired(repeated, p, cKDTree([[0, 0]]))
    extra = copy.deepcopy(p)
    extra["roads"][0]["boundaries"] *= 2
    with pytest.raises(ValueError, match="slots"):
        paired(p, extra, cKDTree([[0, 0]]))
    doc = {"boundaries": [{"geometry": p["roads"][0]["boundaries"][0][::-1]}]}
    assert_profile_geometry(p, doc)
    with pytest.raises(ValueError, match="absent"):
        assert_profile_geometry(profile(1.), doc)


def test_all_source_outputs_frozen_before_reference_is_opened(tmp_path, monkeypatch):
    import json
    import vector_map_anchor_evaluate as audit
    source, config, reference, exe = [tmp_path / s for s in ("cloud", "config.json", "survey", "exe")]
    source.write_bytes(b"source")
    reference.write_bytes(b"reference")
    exe.write_bytes(b"binary")
    config.write_text(json.dumps({"reference_inputs": [], "cases": [{"name": "test"}]}))
    out = tmp_path / "proof"
    doc = {"boundaries": [{"id": 1, "geometry": [[0, 0, 2], [1, 0, 2]]}],
           "lanes": [{"kind": "driving", "left": 1, "right": 1}]}
    def generate(*args, **kwargs):
        out.mkdir()
        for mode in ("before", "after"):
            audit.save(out / f"{mode}-0.json", doc)
            audit.save(out / f"{mode}-0-profiles.json", profile())
            audit.save(out / f"{mode}-0-audit.json", {"extraction": {"generated_length": 1., "surface_fit": {"deferred_length_m": 0.},
                "coverage_edge_anchor_candidates_ignored": 0}, "quality": {"low_support_lanes": [], "sampled_points": 3, "limited": False}})
    def open_reference(*args):
        frozen = json.loads((out / "generation-freeze.json").read_text())
        assert frozen["reference_inputs"] == [] and len(frozen["artifact_sha256"]) == 6
        assert all(audit.digest(out / name) == sha for name, sha in frozen["artifact_sha256"].items())
        return doc
    monkeypatch.setattr(audit.subprocess, "run", generate)
    monkeypatch.setattr(audit, "reference_document", open_reference)
    r = audit.run(source, config, reference, out, exe, "test", None)
    assert r["generation_reference_inputs"] == [] and r["profile_geometry_verified_in_built_maps"]
    assert r["paired_source_intervals"][0]["after"]["mean_xy_m"] == 0


def test_operator_coordinates_must_invert_the_reported_source_translation():
    p = profile()
    p["roads"][0]["operator_reference"] = [[0,-2,2],[1,-2,2]]
    p["extraction"] = {"trace_alignment": {"shift_xy": [0,2]}}
    from vector_map_anchor_evaluate import intervals
    assert list(intervals(p))[0] == (0.,-2.,1.,-2.)
    p["extraction"]["trace_alignment"]["shift_xy"] = [0,1]
    with pytest.raises(ValueError,match="invert"):
        intervals(p)


def test_corridor_correspondence_keeps_slots_targets_and_reports_missing_reference():
    from vector_map_corridor_evaluate import corridor_comparison
    p = {"roads": [{"reference": [[0,0,2],[2,0,2]],
                    "boundaries": [[[0,y,2],[2,y,2]] for y in (1.75,-1.75,-5.25)]}]}
    q = copy.deepcopy(p)
    for line in q["roads"][0]["boundaries"]:
        for point in line:
            point[1] += 3.5
    survey = {"lanes": [{"id":1,"kind":"driving","left":1,"right":2},
                         {"id":2,"kind":"driving","left":{"boundary":3,"reversed":True},"right":{"boundary":2,"reversed":True}}],
              "boundaries": [{"id":i,"geometry":[[0,y,2],[2,y,2]]} for i,y in enumerate((5.25,1.75,-1.75),1)]}
    result = corridor_comparison(p,q,survey,{})
    assert result["evaluated_path_m"] == 2 and result["held_path_m"] == 0
    assert result["assignments"][0]["survey_boundaries_left_to_right"] == [1,2,3]
    assert result["before"]["mean_xy_m"] == 3.5 and result["after"]["mean_xy_m"] == 0
    bad = copy.deepcopy(survey)
    for boundary in bad["boundaries"]:
        boundary["geometry"][0][0] = .2
    held = corridor_comparison(p,q,bad,{})
    assert held["held_path_m"] == 2 and held["before"] is None
    assert corridor_comparison(p,q,survey,{"forward_lanes":3})["before"] is None
    assert corridor_comparison(p,q,survey,{"left_hand_traffic":False})["before"] is None
