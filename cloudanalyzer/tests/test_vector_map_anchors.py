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
