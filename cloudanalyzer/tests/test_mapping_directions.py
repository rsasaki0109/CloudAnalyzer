"""Direction planning diagnoses NCLT source geometry without generating lanes.

The real turn coordinates derive from the bundled University of Michigan NCLT
2012-04-29 excerpt (ODbL / Database Contents License).
"""
import json

import numpy as np
import pytest

from ca.mapping_geometry import preview_lane_directions, preview_layout_width, verify_lane_roundtrip
from ca import mapping_job as jobs, mapping_run as runs
from tests.test_mapping_centerlines import LEFT, RIGHT
from tests.test_mapping_job import job_backend, _run_layout
from tests.test_mapping_layout_width import _candidate, _proposal

native = pytest.importorskip("cloudanalyzer_core")

TURN_LEFT = [[19.848658029149806, 12.382499940138601, -1.4648010375400626],
             [25.304222831515215, 10.904133513770486, -1.7861402294883173]]
TURN_RIGHT = [[14.625794270589767, 5.052991487348599, -1.640423011043826],
              [16.003647351096976, 7.229994739439415, -1.6394225194474425]]
SUPPORTED_LEFT = [[20.878816694285465, 6.463576855086576, -1.7943594166119134],
                  [18.447440522219694, 5.663455029561229, -1.7108932205836336],
                  [24.67820274707269, 2.5218326565748175, -2.0049916803554497]]
SUPPORTED_RIGHT = [[15.953512814117511, 7.3246120971077655, -1.516217552692922],
                   [16.15799493185131, 6.6676656094411495, -1.5622882438108898],
                   [15.711803536048233, 5.6608906663191405, -1.6605248898779417]]


def _input(groups, directions=("forward",)):
    geometry = {"format": "vectormap-ir", "version": 1,
        "metadata": {"attributes": {"cloudanalyzer:draft": "surface_geometry"}},
        "boundaries": [], "lanes": [], "roads": [], "topology": []}
    specs = []
    for left, right in groups:
        cid = len(geometry["boundaries"]) + 1
        center = ((np.asarray(left) + np.asarray(right)) / 2).tolist()
        for line, role in ((center, "source_center"), (left, "source_left"), (right, "source_right")):
            geometry["boundaries"].append({"id": len(geometry["boundaries"]) + 1,
                "kind": {"type": "other"}, "geometry": line, "attributes": {"cloudanalyzer:role": role}})
        fractions = [1.] if len(directions) == 1 else [.25, .75]
        specs.append({"center_curve_id": cid, "left_curve_id": cid + 1, "right_curve_id": cid + 2,
            "lanes": [{"direction": direction, "kind": "driving", "one_way": True,
                       "fraction": fraction, "minimum_width_m": .5}
                      for direction, fraction in zip(directions, fractions)], "speed_limit_kmh": 40.})
    return geometry, specs


def test_preflight_identifies_the_real_export_refusal_without_files(tmp_path):
    geometry, specs = _input([(TURN_LEFT, TURN_RIGHT), (SUPPORTED_LEFT, SUPPORTED_RIGHT)])
    original = json.dumps(geometry)
    preview = json.loads(native.inspect_corridor_lane_directions(original, json.dumps(specs)))
    assert not preview["direction_checks_passed"] and preview["unresolved_intervals"] == 1
    held, supported = preview["segments"]
    assert held["issues"][0]["section_range"] == [0, 1]
    assert not held["issues"][0]["boundary_endpoint_alignment"]
    assert supported["direction_checks_passed"] and supported["planned_pieces"] == [[0, 2]]
    assert not preview["limited"] and not preview["geometry_generated"]
    assert not preview["source_support_checked"] and not preview["roundtrip_checked"] and not preview["adopted"]
    assert list(tmp_path.iterdir()) == [] and json.dumps(geometry) == original
    source = tmp_path / "source.json"
    source.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="centre curve 1 between sections 0 and 1"):
        native.build_corridor_lanes(str(source), json.dumps(specs))


@pytest.mark.parametrize("directions", [("forward",), ("backward",),
    ("forward", "backward"), ("backward", "forward")])
def test_preflight_plans_match_actual_direction_cuts_and_roundtrip(tmp_path, directions):
    geometry, specs = _input([(LEFT, RIGHT)], directions)
    preview = json.loads(native.inspect_corridor_lane_directions(json.dumps(geometry), json.dumps(specs)))
    assert preview["direction_checks_passed"]
    source = tmp_path / "source.json"
    source.write_text(json.dumps(geometry), encoding="utf-8")
    payload = json.loads(native.build_corridor_lanes(str(source), json.dumps(specs)))
    pieces = payload["report"]["built_segments"][0]["pieces"]
    assert preview["segments"][0]["planned_pieces"] == [p["section_range"] for p in pieces]
    osm = tmp_path / "map.osm"
    osm.write_text(payload["osm"], encoding="utf-8")
    reopened = json.loads(json.loads(native.edit_vector_map_relations(str(osm)))["map_json"])
    verify_lane_roundtrip(json.loads(payload["map_json"]), reopened)


def test_whole_range_and_adjacent_pair_have_independent_direction_results():
    layout = {"lanes": [{"direction": "forward", "kind": "driving", "one_way": True,
                         "fraction": 1., "minimum_width_m": 2.5}], "speed_limit_kmh": 40.}
    candidate = {"sections": [{"station_m": 23. + i, "left": left, "right": right,
        "center": ((np.array(left) + np.array(right)) / 2).tolist(), "intersects_trajectory": True}
        for i, (left, right) in enumerate(zip(SUPPORTED_LEFT, SUPPORTED_RIGHT))]}
    candidate["layout_width_preview"] = preview_layout_width(candidate, layout)
    original = json.dumps(candidate)
    result = preview_lane_directions(candidate, layout, native.inspect_corridor_lane_directions)
    assert result["native_check_completed"] and not result["limited"]
    assert result["ranges"][0]["direction_checks_passed"]
    assert result["ranges"][0]["planned_station_pieces"] == [[23., 25.]]
    pair = result["adjacent_interval_checks"][0]
    assert not pair["direction_checks_passed"]
    assert pair["issues"][0]["station_range_m"] == [23., 24.]
    assert not pair["issues"][0]["explicit_centerline_direction"]
    assert json.dumps(candidate) == original


def test_preflight_reports_all_segments_with_a_global_detail_budget():
    pair = ([[0., 2., 0.], [1., 2., 0.]], [[0., -2., 0.], [1., -2., 0.]])
    geometry, specs = _input([pair] * 129)
    result = json.loads(native.inspect_corridor_lane_directions(json.dumps(geometry), json.dumps(specs)))
    assert result["limited"] and result["direction_checks_passed"]
    assert len(result["segments"]) == 129
    assert sum(len(s["planned_pieces"]) for s in result["segments"]) == 128
    assert result["segments"][-1]["planned_pieces_limited"]


def test_run_records_native_turn_diagnostics_without_hd_attempts(job_backend, tmp_path):
    source, _ = job_backend
    candidate = _candidate([9., 10.])
    for i, (section, left, right) in enumerate(zip(candidate["sections"], TURN_LEFT, TURN_RIGHT)):
        section.update(station_m=21. + i, left=left, right=right,
            center=((np.array(left) + np.array(right)) / 2).tolist())
    report = _proposal(candidate)
    report["candidates"][0]["from_m"] = 21.
    jobs.core().propose_road_corridors = lambda *args: json.dumps(report)
    jobs.core().inspect_corridor_lane_directions = native.inspect_corridor_lane_directions
    root = tmp_path / "read-only-turn"
    runs.start_mapping_run(str(source), str(root), _run_layout(width=2.5))
    unchanged = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file() and p.name != "run.json"}
    observed = runs.advance_mapping_run(str(root), {"type": "inspect", "candidate_ids": [37]}, "Read turn directions", 0)
    preview = observed["observation"][0]["layout_direction_preview"]
    assert preview["native_check_completed"] and not preview["geometry_generated"]
    assert preview["native"] == jobs._native()
    issue = preview["adjacent_interval_checks"][0]["issues"][0]
    assert issue["station_range_m"] == [21., 22.] and not issue["boundary_endpoint_alignment"]
    assert observed["remaining_attempts"] == 6 and jobs.inspect_mapping_job(str(root))["attempts"] == []
    assert unchanged == {p.name: p.read_bytes() for p in root.iterdir() if p.is_file() and p.name != "run.json"}
    saved = json.loads((root / "run.json").read_text())
    assert saved["history"][-1]["observation"][0]["layout_direction_preview"] == preview


def test_preflight_shares_the_build_vertex_budget(tmp_path):
    geometry, specs = _input([([[float(i), 16., 0.] for i in range(3031)],
                               [[float(i), -16., 0.] for i in range(3031)])])
    specs[0]["lanes"] *= 16
    for lane in specs[0]["lanes"]:
        lane["fraction"] = 1. / 16
    source = tmp_path / "source.json"
    source.write_text(json.dumps(geometry), encoding="utf-8")
    with pytest.raises(ValueError, match="100000-vertex"):
        native.inspect_corridor_lane_directions(json.dumps(geometry), json.dumps(specs))
    with pytest.raises(ValueError, match="100000-vertex"):
        native.build_corridor_lanes(str(source), json.dumps(specs))
