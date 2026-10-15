"""Bridge plans with numbered elements on synthetic bridges of known layout."""

import csv
import math
from pathlib import Path

import laspy
import numpy as np
import pytest

from ca.bridge_plan import bridge_plan

HALF_WIDTH = 3.5
SLAB = 0.3
GIRDERS = (2.0, 0.0, -2.0)        # lateral offsets, left positive
CROSS = (-4.0, 4.0)               # skewed along-axis stations at mid width
SKEW = math.radians(15.0)
HALF_LENGTH = 8.0
STEP = 0.05


def _bridge(path, *, girders=True, soffit=True, pier=False, columns=False, abutments=False):
    """A skewed deck (class 2) scanned from above and, optionally, from below: the soffit
    and 0.3 m wide girders reaching 1.2 m below the deck top, with cross beams between
    them parallel to the deck ends (class 1)."""
    rng = np.random.default_rng(7)
    tan = math.tan(SKEW)
    pts, cls = [], []
    us = np.arange(-HALF_WIDTH, HALF_WIDTH + 1e-9, STEP)
    for s in np.arange(-HALF_LENGTH - 2, HALF_LENGTH + 2, STEP):
        for u in us:
            if abs(s - u * tan) > HALF_LENGTH:
                continue
            pts.append((s, u, 0.0)); cls.append(2)
            if soffit:
                pts.append((s, u, -SLAB)); cls.append(2)
        if not girders:
            continue
        for g in GIRDERS:
            if abs(s - g * tan) > HALF_LENGTH:
                continue
            for z in np.arange(-1.2, -SLAB, STEP):
                for face in (g - 0.15, g + 0.15):
                    pts.append((s, face, z)); cls.append(1)
            for u in np.arange(g - 0.15, g + 0.15 + 1e-9, STEP):
                pts.append((s, u, -1.2)); cls.append(1)
    if girders:
        for c in CROSS:
            for u in us:
                if not (-2.0 < u < 2.0) or any(abs(u - g) <= 0.15 for g in GIRDERS):
                    continue
                s0 = c + u * tan
                for z in np.arange(-1.0, -SLAB, STEP):
                    for face in (s0 - 0.1, s0 + 0.1):
                        pts.append((face, u, z)); cls.append(1)
    if pier:
        # A 0.6 m pier wall under mid deck, parallel to the skewed ends, down to the ground.
        for u in us:
            for z in np.arange(-1.3, -4.5, -STEP):
                for face in (-0.3, 0.3):
                    pts.append((face + u * tan, u, z)); cls.append(0)
    if columns:
        # Two 0.5 m square columns under mid deck, near the deck edges: a pier of columns.
        for c in (-2.5, 2.5):
            for z in np.arange(-1.3, -4.5, -STEP):
                for v in np.arange(-0.25, 0.25 + 1e-9, STEP):
                    for s0, u0 in ((-0.25, v), (0.25, v), (v, -0.25), (v, 0.25)):
                        pts.append((s0 + (c + u0) * tan, c + u0, z)); cls.append(0)
    if abutments:
        # Sloped abutment faces from beneath each deck end down toward the span.
        for sign in (-1, 1):
            for u in us:
                for t in np.arange(0, 1, 0.01):
                    pts.append((sign * (HALF_LENGTH - 0.5 - 2.5 * t) + u * tan, u, -1.3 - 3.2 * t)); cls.append(0)
    pts = np.array(pts) + rng.normal(0, 0.003, (len(pts), 3))
    h = math.radians(25.0)
    rot = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
    xy = pts[:, :2] @ rot.T + [50.0, 80.0]
    las = laspy.LasData(laspy.LasHeader(point_format=2, version="1.2"))
    las.header.scales = [0.001] * 3
    las.header.offsets = [50.0, 80.0, 0.0]
    las.x, las.y, las.z = xy[:, 0], xy[:, 1], pts[:, 2]
    las.classification = np.array(cls, dtype=np.uint8)
    las.write(str(path))
    return path


def test_girders_cross_beams_and_numbered_elements(tmp_path):
    path = _bridge(tmp_path / "girder.las")
    r = bridge_plan(str(path), str(tmp_path / "out"), axis_classes=[2], member_classes=[1, 2])
    assert r["underside"] == "girders_observed"
    assert abs(abs(r["deck"]["skew_deg"]) - 15.0) < 1.0
    assert r["deck"]["width_m"] == pytest.approx(2 * HALF_WIDTH, abs=0.1)
    # The axis may point either way along the deck: compare lateral offsets up to sign.
    lateral = [g["lateral_m"] for g in r["girders"]]
    assert len(lateral) == 3 and lateral == sorted(lateral, reverse=True)
    assert sorted(abs(v) for v in lateral) == pytest.approx([0.0, 2.0, 2.0], abs=0.1)
    assert [g["number"] for g in r["girders"]] == [1, 2, 3]
    assert all(g["depth_below_deck_top_m"] == pytest.approx(1.2, abs=0.05) for g in r["girders"])
    assert len(r["cross_beams"]) == 2
    assert sorted(abs(c["along_m"] - 0.5 * (r["deck"]["start_m"] + r["deck"]["end_m"])) for c in r["cross_beams"]) \
        == pytest.approx([4.0, 4.0], abs=0.15)
    ids = {e["id"] for e in r["elements"]}
    assert {f"Ds{i:02d}{j:02d}" for i in range(1, 4) for j in range(1, 5)} <= ids
    assert {f"Mg{i:02d}{g:02d}" for i in range(1, 4) for g in range(1, 4)} <= ids
    assert {f"Cr{c:02d}{b:02d}" for c in (1, 2) for b in (1, 2)} <= ids
    assert len(ids) == 12 + 9 + 4 and not r["warnings"]
    with open(r["outputs"]["csv"], encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 25
    svg = Path(r["outputs"]["svg"]).read_text(encoding="utf-8")
    assert svg.startswith("<svg") and "Ds0101" in svg and "G3" in svg


def test_a_scanned_soffit_without_members_is_one_slab_element(tmp_path):
    r = bridge_plan(str(_bridge(tmp_path / "slab.las", girders=False)), axis_classes=[2])
    assert r["underside"] == "no_girders_observed"
    assert not r["girders"] and not r["cross_beams"]
    assert [e["id"] for e in r["elements"]] == ["Ds0101"]


def test_a_deck_scanned_only_from_above_draws_nothing_beneath(tmp_path):
    r = bridge_plan(str(_bridge(tmp_path / "top.las", girders=False, soffit=False)), str(tmp_path / "o"),
                    axis_classes=[2])
    assert r["underside"] == "unobserved" and r["deck"]["slab_thickness_m"] is None
    assert [e["id"] for e in r["elements"]] == ["Ds0101"]
    assert "下面は未観測" in Path(r["outputs"]["svg"]).read_text(encoding="utf-8")


def test_input_validation(tmp_path):
    path = _bridge(tmp_path / "v.las", girders=False, soffit=False)
    with pytest.raises(ValueError):
        bridge_plan(str(path), spacing=0)


def test_the_command_writes_the_plan(tmp_path):
    from typer.testing import CliRunner

    from cloudanalyzer_cli.main import app

    path = _bridge(tmp_path / "cli.las", girders=False, soffit=False)
    result = CliRunner().invoke(app, ["bridge-plan", str(path), "-o", str(tmp_path / "out"), "--axis-classes", "2"])
    assert result.exit_code == 0, result.output
    assert "Underside: unobserved" in result.output
    assert (tmp_path / "out" / "plan.svg").exists() and (tmp_path / "out" / "members.csv").exists()


def test_a_pier_splits_spans_and_sloped_abutments_do_not(tmp_path):
    path = _bridge(tmp_path / "two.las", pier=True, abutments=True)
    r = bridge_plan(str(path), axis_classes=[2], member_classes=[1, 2])
    (pier,) = r["supports"]
    assert pier["label"] == "P1" and pier["source"] == "detected"
    assert pier["from_start_m"] == pytest.approx(HALF_LENGTH, abs=0.2)
    assert [round(sp["length_m"]) for sp in r["spans"]] == [8, 8]
    deck = {e["id"]: e["span"] for e in r["elements"] if e["kind"] == "deck_slab"}
    # Cross beams at +/-4 m and the pier at mid length make four rows of four columns.
    assert len(deck) == 16 and {deck[f"Ds{i:02d}01"] for i in (1, 2)} == {1} and {deck[f"Ds{i:02d}01"] for i in (3, 4)} == {2}
    with open(Path(bridge_plan(str(path), str(tmp_path / "o"), axis_classes=[2], member_classes=[1, 2])
                   ["outputs"]["svg"])) as f:
        svg = f.read()
    assert "P1" in svg and "A1" in svg and "第2径間" in svg


def test_a_row_of_columns_is_a_pier(tmp_path):
    # The columns fill little of the width but span most of it end to end.
    path = _bridge(tmp_path / "cols.las", columns=True, abutments=True)
    r = bridge_plan(str(path), axis_classes=[2], member_classes=[1, 2])
    (pier,) = r["supports"]
    assert pier["from_start_m"] == pytest.approx(HALF_LENGTH, abs=0.2)
    assert pier["width_share"] > 0.7


def test_explicit_supports_from_a_register(tmp_path):
    path = _bridge(tmp_path / "reg.las", girders=False, soffit=False)
    r = bridge_plan(str(path), axis_classes=[2], supports=[5.0, 11.0])
    assert [p["source"] for p in r["supports"]] == ["explicit", "explicit"]
    assert [(e["id"], e["span"]) for e in r["elements"]] == [("Ds0101", 1), ("Ds0201", 2), ("Ds0301", 3)]
    with pytest.raises(ValueError):
        bridge_plan(str(path), axis_classes=[2], supports=[-1.0])
