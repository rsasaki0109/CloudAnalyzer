"""Source cross-section correspondence survives lane cuts and Lanelet2 reload."""
from copy import deepcopy
import json

import numpy as np
import pytest

from ca.mapping_geometry import verify_lane_roundtrip


native = pytest.importorskip("cloudanalyzer_core")
LEFT = [[0., 3., 0.], [1., 7., 1.], [4., 3., 0.], [8., 6., -1.], [10., 3., 0.]]
RIGHT = [[0., -3., 0.], [1., -3., -1.], [4., -2., 0.], [8., -3., 1.], [10., -3., 0.]]


def _build(tmp_path, name, directions=("forward",), start=0, stop=5, left=LEFT, right=RIGHT):
    left, right = np.array(left[start:stop]), np.array(right[start:stop])
    center = (left + right) / 2
    geometry = {"format": "vectormap-ir", "version": 1,
        "metadata": {"attributes": {"cloudanalyzer:draft": "surface_geometry"}},
        "boundaries": [{"id": i + 1, "kind": {"type": "other"}, "geometry": line.tolist(),
            "attributes": {"cloudanalyzer:role": role}}
            for i, (line, role) in enumerate(((center, "source_center"), (left, "source_left"), (right, "source_right")))],
        "lanes": [], "roads": [], "topology": []}
    source = tmp_path / f"{name}.json"
    source.write_text(json.dumps(geometry), encoding="utf-8")
    fractions = [1.] if len(directions) == 1 else [.25, .75]
    specs = [{"center_curve_id": 1, "left_curve_id": 2, "right_curve_id": 3,
        "lanes": [{"direction": direction, "kind": "driving", "one_way": True,
                   "fraction": fraction, "minimum_width_m": .5}
                  for direction, fraction in zip(directions, fractions)], "speed_limit_kmh": 40.}]
    payload = json.loads(native.build_corridor_lanes(str(source), json.dumps(specs)))
    ir = json.loads(payload["map_json"])
    osm = tmp_path / f"{name}.osm"
    osm.write_text(payload["osm"], encoding="utf-8")
    reopened = json.loads(json.loads(native.edit_vector_map_relations(str(osm)))["map_json"])
    verify_lane_roundtrip(ir, reopened)
    return payload, ir, reopened, fractions


@pytest.mark.parametrize("directions", [("forward",), ("backward",),
    ("forward", "backward"), ("backward", "forward")])
def test_explicit_centers_follow_paired_sections_fractions_and_travel_direction(tmp_path, directions):
    payload, ir, reopened, fractions = _build(tmp_path, "full", directions)
    assert payload["report"]["centerline_policy"] == "paired_source_section_midpoints"
    ids = payload["report"]["built_segments"][0]["lane_ids"]
    left, right = np.array(LEFT), np.array(RIGHT)
    offset = 0.
    for lid, direction, fraction in zip(ids, directions, fractions):
        expected = left + (right - left) * (offset + fraction / 2)
        offset += fraction
        if direction == "backward":
            expected = expected[::-1]
        for document in (ir, reopened):
            lane = next(l for l in document["lanes"] if l["id"] == lid)
            np.testing.assert_allclose(lane["centerline"], expected, rtol=0, atol=1e-9)
    # The two outer source curves keep every original XYZ vertex.
    curves = [np.array(b["geometry"]) for b in ir["boundaries"]]
    for expected in (left, right):
        assert any(np.array_equal(line, expected) or np.array_equal(line[::-1], expected) for line in curves)


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_clipping_unequal_length_bounds_keeps_the_same_section_centers(tmp_path, direction):
    _, full, _, _ = _build(tmp_path, "full", (direction,))
    _, clipped, reopened, _ = _build(tmp_path, "clipped", (direction,), 1, 4)
    expected = full["lanes"][0]["centerline"][1:4]
    for document in (clipped, reopened):
        np.testing.assert_allclose(document["lanes"][0]["centerline"], expected, rtol=0, atol=1e-9)


def test_adjacent_sections_whose_center_reverses_are_not_exported(tmp_path):
    # Two unchanged sections from the existing real-data folded-turn fixture.
    left = [[11.935, 3.806, -1.648], [16.075, 7.386, -1.713]]
    right = [[25.049, 9.991, -1.895], [21.659, 14.444, -1.592]]
    with pytest.raises(ValueError, match="between sections 0 and 1.*explicit centerline direction reverses"):
        _build(tmp_path, "ambiguous", left=left, right=right)


@pytest.mark.parametrize("change", ["missing", "shifted", "reversed", "truncated", "missing_z"])
def test_roundtrip_rejects_lost_or_changed_explicit_centers(tmp_path, change):
    _, ir, reopened, _ = _build(tmp_path, "full")
    altered = deepcopy(reopened)
    lane = altered["lanes"][0]
    if change == "missing":
        lane.pop("centerline")
    elif change == "shifted":
        lane["centerline"][2][2] += .01
    elif change == "reversed":
        lane["centerline"].reverse()
    elif change == "truncated":
        lane["centerline"].pop()
    else:
        lane["centerline"][2].pop()
    with pytest.raises(ValueError, match="explicit source-section centerline"):
        verify_lane_roundtrip(ir, altered)


def test_explicit_centers_share_the_existing_total_vertex_budget(tmp_path):
    sections = 34_000
    geometry = {"format": "vectormap-ir", "version": 1,
        "metadata": {"attributes": {"cloudanalyzer:draft": "surface_geometry"}},
        "boundaries": [{"id": i + 1, "kind": {"type": "other"},
            "geometry": [[float(x), y, 0.] for x in range(sections)],
            "attributes": {"cloudanalyzer:role": role}}
            for i, (y, role) in enumerate(((0., "source_center"), (2., "source_left"), (-2., "source_right")))],
        "lanes": []}
    path = tmp_path / "bounded.json"
    path.write_text(json.dumps(geometry), encoding="utf-8")
    spec = [{"center_curve_id": 1, "left_curve_id": 2, "right_curve_id": 3,
        "lanes": [{"direction": "forward", "kind": "driving", "one_way": True,
                   "fraction": 1., "minimum_width_m": 2.5}], "speed_limit_kmh": 40.}]
    with pytest.raises(ValueError, match="100000-vertex"):
        native.build_corridor_lanes(str(path), json.dumps(spec))
