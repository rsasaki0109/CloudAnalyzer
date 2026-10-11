"""Bridge cross sections and dimension table on synthetic bridges of known size."""

import csv
import json
import math

import laspy
import numpy as np
import pytest

from ca.bridge import bridge_sections, measure_section

EFFECTIVE = 6.0
CURB_W, CURB_H = 0.5, 0.2
PARAPET = 0.8
FACE = 0.3
SLAB = 0.25
SLOPE = 0.02


def section_points(rng, soffit=False, noise=0.004):
    """One cross section: u lateral (left positive), z height."""
    u, z = [], []
    half = EFFECTIVE / 2
    deck = lambda x: SLOPE * x  # noqa: E731
    for x in np.arange(-half, half, 0.01):
        u.append(x); z.append(deck(x))
    for sign in (+1, -1):
        edge = sign * half
        top = deck(edge) + CURB_H
        for h in np.arange(0, CURB_H, 0.01):             # curb inner face
            u.append(edge); z.append(deck(edge) + h)
        for x in np.arange(0, CURB_W, 0.01):             # curb top
            u.append(edge + sign * x); z.append(top)
        outer = edge + sign * CURB_W
        for h in np.arange(0, CURB_H + FACE, 0.01):      # outer face below the curb top
            u.append(outer); z.append(top - h)
        for h in np.arange(0.1, PARAPET, 0.01):          # parapet post over the curb
            u.append(edge + sign * CURB_W * 0.5); z.append(top + h)
    if soffit:
        for x in np.arange(-half + 0.2, half - 0.2, 0.01):
            u.append(x); z.append(deck(x) - SLAB)
    u, z = np.array(u), np.array(z)
    return u + rng.normal(0, noise, len(u)), z + rng.normal(0, noise, len(z))


def test_section_dimensions_and_unobserved_soffit():
    u, z = section_points(np.random.default_rng(0))
    m = measure_section(u, z)
    d = m["dims"]
    assert m["status"] == "measured"
    assert d["effective_width_m"] == pytest.approx(EFFECTIVE, abs=0.03)
    assert d["total_width_m"] == pytest.approx(EFFECTIVE + 2 * CURB_W, abs=0.03)
    assert d["cross_slope_percent"] == pytest.approx(SLOPE * 100, abs=0.2)
    for side in ("left", "right"):
        assert d[f"{side}_curb_width_m"] == pytest.approx(CURB_W, abs=0.03)
        assert d[f"{side}_curb_height_m"] == pytest.approx(CURB_H, abs=0.02)
        assert d[f"{side}_parapet_height_m"] == pytest.approx(PARAPET, abs=0.03)
        assert d[f"{side}_outer_face_depth_m"] == pytest.approx(FACE, abs=0.02)
    # No returns beneath the deck: the slab thickness is not reported at all.
    assert "slab_thickness_m" not in d


def test_slab_thickness_only_from_soffit_returns():
    u, z = section_points(np.random.default_rng(1), soffit=True)
    d = measure_section(u, z)["dims"]
    assert d["slab_thickness_m"] == pytest.approx(SLAB, abs=0.02)


def test_no_deck_surface_is_reported_not_guessed():
    rng = np.random.default_rng(2)
    m = measure_section(rng.uniform(-3, 3, 500), rng.uniform(-3, 3, 500))
    assert m["status"] == "no_deck_surface"
    assert "dims" not in m


def write_bridge(path, rng, length=12.0, skew_deg=20.0, heading_deg=30.0, step=0.2):
    """A skewed deck with curbs, parapets and an outer face, labelled like the
    public dataset (2 deck, 1 girder/curb, 3 parapet) plus unlabelled trees."""
    tan = math.tan(math.radians(skew_deg))
    pts, cls = [], []
    half = EFFECTIVE / 2
    for s in np.arange(-length / 2 - 2, length / 2 + 2, step / 2):
        for u, z in zip(*section_points(rng, noise=0.003)):
            # Skewed ends: the deck spans start(u) .. end(u) along the axis.
            if abs(s - u * tan) > length / 2:
                continue
            if abs(u) <= half + 0.005:
                c = 2
            elif z > SLOPE * math.copysign(half, u) + CURB_H + 0.05:
                c = 3
            else:
                c = 1
            pts.append((s, u, z)); cls.append(c)
    pts = np.array(pts)
    keep = rng.random(len(pts)) < 0.8
    pts, cls = pts[keep], np.array(cls)[keep]
    trees = np.column_stack([rng.uniform(-8, 8, 4000), rng.uniform(6, 9, 4000), rng.uniform(0, 8, 4000)])
    pts = np.vstack([pts, trees]); cls = np.concatenate([cls, np.full(len(trees), 4)])
    h = math.radians(heading_deg)
    rot = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
    xy = pts[:, :2] @ rot.T + [100.0, 200.0]
    las = laspy.LasData(laspy.LasHeader(point_format=2, version="1.2"))
    las.header.scales = [0.001] * 3
    las.header.offsets = [100.0, 200.0, 0.0]
    las.x, las.y, las.z = xy[:, 0], xy[:, 1], pts[:, 2]
    las.classification = cls.astype(np.uint8)
    las.write(str(path))


@pytest.fixture(scope="module")
def bridge_las(tmp_path_factory):
    path = tmp_path_factory.mktemp("bridge") / "bridge.las"
    write_bridge(path, np.random.default_rng(3))
    return path


def test_bridge_table_on_a_skewed_rotated_bridge(bridge_las, tmp_path):
    r = bridge_sections(str(bridge_las), str(tmp_path / "out"), classes=[1, 2, 3], axis_classes=[2])
    rows = {row["item"]: row for row in r["table"]}
    assert abs(abs(rows["skew_deg"]["value"]) - 20.0) < 1.0
    assert rows["deck_length_m"]["value"] == pytest.approx(12.0, abs=0.15)
    assert rows["effective_width_m"]["value"] == pytest.approx(EFFECTIVE, abs=0.05)
    assert rows["total_width_m"]["value"] == pytest.approx(EFFECTIVE + 2 * CURB_W, abs=0.05)
    assert rows["left_parapet_height_m"]["value"] == pytest.approx(PARAPET, abs=0.05)
    assert rows["left_outer_face_depth_m"]["status"] == "lower_bound"
    assert rows["slab_thickness_m"]["status"] == "unobserved"
    assert r["unobserved"] == ["slab_thickness_m"]
    # Every section lies where the full width is on the deck.
    assert all(s["status"] == "measured" for s in r["sections"])
    assert len(r["sections"]) >= 5
    out = r["outputs"]
    saved = json.loads(open(out["json"], encoding="utf-8").read())
    assert saved["schema"] == "cloudanalyzer.bridge_sections.v1"
    with open(out["csv"], encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    slab = next(t for t in table if t["item"] == "slab_thickness_m")
    assert slab["status"] == "unobserved" and slab["value_m_or_percent"] == ""
    svg = open(out["svg"], encoding="utf-8").read()
    assert svg.startswith("<svg") and "未観測" in svg


def test_explicit_axis_and_input_validation(bridge_las):
    h = math.radians(30.0)
    start = [100.0 - 5 * math.cos(h), 200.0 - 5 * math.sin(h)]
    end = [100.0 + 5 * math.cos(h), 200.0 + 5 * math.sin(h)]
    r = bridge_sections(str(bridge_las), classes=[1, 2, 3], axis_classes=[2], axis=(start, end))
    assert r["axis"]["source"] == "explicit"
    assert next(row for row in r["table"] if row["item"] == "effective_width_m")["value"] == pytest.approx(
        EFFECTIVE, abs=0.05)
    with pytest.raises(ValueError):
        bridge_sections(str(bridge_las), spacing=0)
    with pytest.raises(ValueError):
        bridge_sections(str(bridge_las), axis=([0, 0], [0, 0]))
