"""Damage candidates on a synthetic slab deck with known patches."""

import csv
from pathlib import Path

import laspy
import numpy as np
import pytest

from ca.bridge_damage import bridge_damage

HALF_LENGTH, HALF_WIDTH, SLAB, STEP = 5.0, 3.0, 0.4, 0.02
GREY = (0.55, 0.55, 0.53)
PATCHES = {  # centre (along, lateral), half size, colour or recess
    "light": ((-2.0, 1.0), 0.12, (0.95, 0.95, 0.93)),
    "rust": ((1.5, -1.0), 0.12, (0.62, 0.38, 0.18)),
    "recess": ((2.5, 1.2), 0.12, None),
}


def _deck(path, *, soffit=True, colour=True):
    rng = np.random.default_rng(11)
    pts, rgb, cls = [], [], []
    for s in np.arange(-HALF_LENGTH, HALF_LENGTH, STEP):
        for u in np.arange(-HALF_WIDTH, HALF_WIDTH, STEP):
            pts.append((s, u, 0.0)); rgb.append(GREY); cls.append(2)
            if not soffit:
                continue
            z, c = -SLAB, GREY
            for kind, ((ps, pu), h, col) in PATCHES.items():
                if abs(s - ps) <= h and abs(u - pu) <= h:
                    if kind == "recess":
                        z = -SLAB + 0.03
                    else:
                        c = col
            # A dark straight seam along the deck: a joint between slab units, not damage.
            if abs(u - (-2.0)) < 0.02 and -4.0 < s < 4.0:
                c = (0.2, 0.2, 0.2)
            pts.append((s, u, z)); rgb.append(c); cls.append(2)
    pts = np.array(pts) + rng.normal(0, 0.002, (len(pts), 3))
    rgb = np.clip(np.array(rgb) + rng.normal(0, 0.01, (len(rgb), 3)), 0, 1)
    las = laspy.LasData(laspy.LasHeader(point_format=2 if colour else 0, version="1.2"))
    las.header.scales = [0.001] * 3
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    if colour:
        las.red, las.green, las.blue = (np.round(rgb[:, k] * 65535).astype(np.uint16) for k in range(3))
    las.classification = np.array(cls, dtype=np.uint8)
    las.write(str(path))
    return path


def test_patches_are_found_on_their_element_and_a_straight_seam_is_not(tmp_path):
    r = bridge_damage(str(_deck(tmp_path / "deck.las")), str(tmp_path / "out"), axis_classes=[2])
    assert r["underside"] == "no_girders_observed"
    found = {c["kind"]: c for c in r["candidates"]}
    assert set(found) == {"light", "rust", "recess"}, r["candidates"]
    # The axis may run either way along the deck: compare positions up to its sign.
    for kind, ((s, u), h, _) in PATCHES.items():
        a, lat = found[kind]["centre_m"]
        assert min(np.hypot(a - s, lat - u), np.hypot(a + s, lat + u)) < 0.08
        assert found[kind]["area_m2"] == pytest.approx((2 * h) ** 2, rel=0.5)
        assert found[kind]["element"] == "Ds0101"
    assert found["recess"]["contrast"] == pytest.approx(-0.03, abs=0.01)
    assert r["straight_seams_left_out"]["cells_along_m"].get("dark", 0) > 5
    with open(r["outputs"]["csv"], encoding="utf-8") as f:
        assert {row["kind"] for row in csv.DictReader(f)} == {"light", "rust", "recess"}
    svg = Path(r["outputs"]["svg"]).read_text(encoding="utf-8")
    assert "data:image/png;base64," in svg and "D001" in svg and Path(r["outputs"]["png"]).exists()


def test_a_deck_scanned_only_from_above_has_no_candidates(tmp_path):
    r = bridge_damage(str(_deck(tmp_path / "top.las", soffit=False)), str(tmp_path / "o"), axis_classes=[2])
    assert r["underside"] == "unobserved" and r["candidates"] == []
    assert "not searched" in r["note"]


def test_colour_is_required(tmp_path):
    with pytest.raises(ValueError, match="colour"):
        bridge_damage(str(_deck(tmp_path / "grey.las", colour=False)), axis_classes=[2])


def test_the_command_reports_candidates(tmp_path):
    from typer.testing import CliRunner

    from cloudanalyzer_cli.main import app

    result = CliRunner().invoke(app, ["bridge-damage", str(_deck(tmp_path / "c.las")), "-o", str(tmp_path / "out"),
                                      "--axis-classes", "2"])
    assert result.exit_code == 0, result.output
    assert "Candidates: 3" in result.output and (tmp_path / "out" / "damage_plan.svg").exists()
