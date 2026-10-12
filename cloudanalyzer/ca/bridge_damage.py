"""Damage candidates on a bridge deck soffit, located on the numbered plan elements.

The soffit seen in the plan is rasterised at 2 cm into colour (CIE L*a*b*) and height.
Each cell is compared with the median of its 0.5 m neighbourhood: much lighter cells
(efflorescence or whitening), much darker cells (water or dark staining), orange-brown
cells (rust staining) and cells receded into the slab (spalling or section loss) form
candidate regions. Bands along girders, cross beams, supports, deck edges and scan
holes are left out, where shading and haunches mimic all four, and long straight runs
along or across the deck (joints between slab units, formwork seams) are counted apart.
Cracks are narrower than the raster and are not searched. Candidates are for an
inspector to confirm on site or in the imagery; nothing is diagnosed.
"""

from __future__ import annotations

import base64
import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from ca.bridge import Axis, load_points
from ca.bridge_plan import bridge_plan, plan_svg

SCHEMA = "cloudanalyzer.bridge_damage.v1"

CELL = 0.02                # raster cell (m)
BACKGROUND = 0.5           # neighbourhood for the local median (m)
SOFFIT_BAND = 0.08         # returns within this of the soffit height count as soffit (m)
LIGHT_DL = 15.0            # lighter than the neighbourhood by this L*
DARK_DL = -15.0
RUST_DB, RUST_DA = 12.0, 4.0   # yellower and redder than the neighbourhood
RECESS_M = 0.012           # receded into the slab by this much (m)
MIN_AREA = 0.005           # smallest candidate (m^2)
MEMBER_MARGIN = 0.15       # left out on each side of girders, cross beams and supports (m)
SEAM_RUN = 0.4             # straight run along or across the deck that makes a seam (m)
SEAM_ELONGATION = 6.0      # length over width of a straight seam
SEAM_ANGLE = 10.0          # degrees from the axis or its normal for a joint or formwork seam
EDGE_MARGIN = 0.3          # left out inside the deck edges and ends (m)
HOLE_MARGIN = 0.1          # left out around cells without returns (m)

KINDS = {
    "light": ("遊離石灰・白色変色の候補", "#00bcd4"),
    "dark": ("漏水・暗色変色の候補", "#3949ab"),
    "rust": ("錆汁の候補", "#d93025"),
    "recess": ("剥離・欠損の候補", "#f9ab00"),
}


def _lab(rgb: np.ndarray) -> np.ndarray:
    """CIE L*a*b* (D65) of sRGB colours in [0, 1]."""
    c = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.column_stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])])


def _element_at(plan: dict[str, Any], along: float, lat: float) -> str | None:
    """The deck slab element containing a plan point (skewed rows, lateral columns)."""
    d = plan["deck"]
    skewed = along - d["skew_slope"] * (lat - 0.5 * (d["left_edge_m"] + d["right_edge_m"]))
    for e in plan["elements"]:
        if e["kind"] == "deck_slab":
            (a, b), (r, l) = e["along_m"], e["lateral_m"]
            if a <= skewed <= b and r <= lat <= l:
                return str(e["id"])
    return None


def bridge_damage(
    path: str,
    out_dir: str | None = None,
    *,
    plan: dict[str, Any] | None = None,
    classes: list[int] | None = None,
    axis_classes: list[int] | None = None,
    member_classes: list[int] | None = None,
    surface_classes: list[int] | None = None,
    supports: list[float] | None = None,
) -> dict[str, Any]:
    """Damage candidates on the deck soffit, with the plan element each lies in.

    `plan` is a :func:`ca.bridge_plan.bridge_plan` result for the same file; without it
    the plan is drawn with `classes`, `axis_classes`, `member_classes` and `supports`.
    `surface_classes` selects the points the soffit is taken from (default: all).
    """
    if plan is None:
        plan = bridge_plan(path, classes=classes, axis_classes=axis_classes, member_classes=member_classes,
                           supports=supports)
    data = load_points(path)
    result: dict[str, Any] = {"schema": SCHEMA, "source": str(Path(path).resolve()), "underside": plan["underside"],
                              "candidates": [], "thresholds": {
                                  "cell_m": CELL, "background_m": BACKGROUND, "light_dL": LIGHT_DL,
                                  "dark_dL": DARK_DL, "rust_db_da": [RUST_DB, RUST_DA], "recess_m": RECESS_M,
                                  "min_area_m2": MIN_AREA, "member_margin_m": MEMBER_MARGIN,
                                  "edge_margin_m": EDGE_MARGIN, "seam_elongation": SEAM_ELONGATION,
                                  "seam_angle_deg": SEAM_ANGLE}}
    d = plan["deck"]
    if plan["underside"] == "unobserved" or d["slab_thickness_m"] is None:
        result["note"] = "No returns beneath the deck: the soffit was not searched and no damage is implied."
        if out_dir:
            result["outputs"] = _write(result, plan, None, Path(out_dir))
        return result
    if "rgb" not in data:
        raise ValueError("the point cloud has no colour; damage candidates need RGB")
    xyz, rgb = data["xyz"], data["rgb"]
    cls = data.get("classification")
    if surface_classes and cls is None:
        raise ValueError("surface classes were given but the file has no classification")
    if surface_classes and cls is not None:
        keep = np.isin(cls, surface_classes)
        xyz, rgb = xyz[keep], rgb[keep]
    ax = Axis(np.array(plan["axis"]["origin_xy"]), np.array(plan["axis"]["direction_xy"]), "plan")
    rel = xyz[:, :2] - ax.origin
    along, lat, z = rel @ ax.direction, rel @ ax.lateral, xyz[:, 2]
    slope, mid = d["skew_slope"], 0.5 * (d["left_edge_m"] + d["right_edge_m"])
    skewed = along - slope * (lat - mid)
    on_deck = (lat > d["right_edge_m"]) & (lat < d["left_edge_m"]) & (skewed > d["start_m"]) & (skewed < d["end_m"])

    # The deck top over the inner width (clear of curbs and parapets), with a crown term,
    # through its upper layer of returns; the soffit lies a slab thickness below it.
    half = 0.5 * (d["left_edge_m"] - d["right_edge_m"])
    inner = on_deck & (np.abs(lat - mid) < 0.8 * half)

    def surface(a: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(a)), a, u - mid, (u - mid) ** 2, np.abs(u - mid)])

    top = inner & (z > np.percentile(z[inner], 50))
    design = surface(along[top], lat[top])
    coef = np.linalg.lstsq(design, z[top], rcond=None)[0]
    for _ in range(4):
        upper = np.abs(z[top] - design @ coef) < 0.04
        coef = np.linalg.lstsq(design[upper], z[top][upper], rcond=None)[0]
    depth = surface(along, lat) @ coef - z
    soffit = on_deck & (np.abs(depth - d["slab_thickness_m"]) < SOFFIT_BAND)
    if soffit.sum() < 1000:
        result["note"] = "Too few returns on the soffit to search for damage candidates."
        if out_dir:
            result["outputs"] = _write(result, plan, None, Path(out_dir))
        return result

    # A grid fixed in the axis frame: a deck edge moved by millimetres must not re-cut every cell.
    a0 = float(np.floor(along[soffit].min() / CELL) * CELL)
    u0 = float(np.floor(lat[soffit].min() / CELL) * CELL)
    ia = ((along[soffit] - a0) / CELL).astype(int)
    iu = ((lat[soffit] - u0) / CELL).astype(int)
    na, nu = int(ia.max()) + 1, int(iu.max()) + 1
    flat = iu * na + ia
    count = np.bincount(flat, minlength=na * nu).reshape(nu, na)
    colour = np.clip(rgb[soffit] / (65535.0 if rgb.max() > 255 else 255.0), 0, 1)
    lab = _lab(colour)

    def mean(values: np.ndarray) -> np.ndarray:
        total = np.bincount(flat, weights=values, minlength=na * nu).reshape(nu, na)
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)

    image = np.dstack([np.nan_to_num(mean(colour[:, k])) for k in range(3)])
    L, A, B = (mean(lab[:, k]) for k in range(3))
    height = mean(depth[soffit])
    valid = count >= 1
    size = int(round(BACKGROUND / CELL)) | 1

    def background(x: np.ndarray) -> np.ndarray:
        return np.asarray(ndimage.median_filter(np.where(valid, x, np.nanmedian(x[valid])), size=size))

    dL, dA, dB = L - background(L), A - background(A), B - background(B)
    dH = height - background(height)

    # Where shading, haunches and edges mimic damage: around members, deck edges and holes.
    ga = a0 + (np.arange(na) + 0.5) * CELL
    gu = u0 + (np.arange(nu) + 0.5) * CELL
    GA, GU = np.meshgrid(ga, gu)
    GS = GA - slope * (GU - mid)
    # Holes are gaps of several cells (scan shadows), not single empty cells between returns.
    holes, n_holes = ndimage.label(~ndimage.binary_closing(valid, iterations=2) & ~valid)
    sizes = ndimage.sum(np.ones_like(holes), holes, range(1, n_holes + 1))
    real = np.isin(holes, 1 + np.nonzero(np.asarray(sizes) >= 4)[0])
    usable = valid & ~ndimage.binary_dilation(real, iterations=max(1, int(HOLE_MARGIN / CELL)))
    usable &= (GU < d["left_edge_m"] - EDGE_MARGIN) & (GU > d["right_edge_m"] + EDGE_MARGIN)
    usable &= (GS > d["start_m"] + EDGE_MARGIN) & (GS < d["end_m"] - EDGE_MARGIN)
    for g in plan["girders"] + plan["partially_observed_members"]:
        usable &= np.abs(GU - g["lateral_m"]) > g["visible_width_m"] / 2 + MEMBER_MARGIN
    for c in plan["cross_beams"]:
        usable &= np.abs(GS - c["along_m"]) > c["visible_width_m"] / 2 + MEMBER_MARGIN
    for p in plan["supports"]:
        usable &= np.abs(GS - p["along_m"]) > p.get("width_m", 0.0) / 2 + MEMBER_MARGIN

    masks = {"light": dL > LIGHT_DL, "dark": dL < DARK_DL, "rust": (dB > RUST_DB) & (dA > RUST_DA),
             "recess": dH < -RECESS_M}
    min_cells = int(np.ceil(MIN_AREA / CELL**2))
    candidates: list[dict[str, Any]] = []
    seams: dict[str, int] = {}
    run = max(3, int(round(SEAM_RUN / CELL)))
    seam_length: dict[str, float] = {}
    for kind, mask in masks.items():
        mask = np.nan_to_num(mask, nan=0).astype(bool) & usable
        # Straight runs along or across the deck, even when broken into pieces, are joints
        # between slab units or formwork seams: take them out before forming regions.
        straight = np.zeros_like(mask)
        for line in (np.ones((1, run)), np.ones((run, 1))):
            bridged = ndimage.binary_closing(mask, structure=line[:, :3] if line.shape[0] == 1 else line[:3])
            straight |= ndimage.binary_opening(bridged, structure=line) & mask
        if straight.any():
            seam_length[kind] = float(straight.sum() * CELL)
            mask &= ~ndimage.binary_dilation(straight, iterations=2)
        mask = ndimage.binary_opening(mask)
        labels, n = ndimage.label(mask)
        for k, region in enumerate(ndimage.find_objects(labels)):
            if region is None:
                continue
            cells = labels[region] == k + 1
            if cells.sum() < min_cells:
                continue
            rows, cols = np.nonzero(cells)
            ys, xs = rows + region[0].start, cols + region[1].start
            ca, cu = float(ga[xs].mean()), float(gu[ys].mean())
            # Long straight runs along or across the deck are joints between slab units or
            # formwork seams, not damage: report them apart.
            pts = np.column_stack([xs, ys]).astype(float)
            evals, evecs = np.linalg.eigh(np.cov(pts.T) + np.eye(2) * 1e-9)
            elongation = float(np.sqrt(evals[1] / max(evals[0], 1e-9)))
            angle = float(np.degrees(np.arctan2(abs(evecs[1, 1]), abs(evecs[0, 1]))))
            if elongation >= SEAM_ELONGATION and min(angle, 90 - angle) <= SEAM_ANGLE:
                seams[kind] = seams.get(kind, 0) + 1
                continue
            metric = {"light": dL, "dark": dL, "rust": dB, "recess": dH}[kind][ys, xs]
            candidates.append({
                "kind": kind, "label_ja": KINDS[kind][0], "area_m2": float(cells.sum() * CELL**2),
                "centre_m": [ca, cu], "element": _element_at(plan, ca, cu),
                "along_m": [float(ga[xs].min() - CELL / 2), float(ga[xs].max() + CELL / 2)],
                "lateral_m": [float(gu[ys].min() - CELL / 2), float(gu[ys].max() + CELL / 2)],
                "contrast": float(np.median(metric)),
                "contrast_unit": "m" if kind == "recess" else ("b*" if kind == "rust" else "L*"),
            })
    candidates.sort(key=lambda c: (c["element"] or "~", c["centre_m"][0]))
    for k, c in enumerate(candidates):
        c["id"] = f"D{k + 1:03d}"
    result["candidates"] = candidates
    result["straight_seams_left_out"] = {"regions": seams, "cells_along_m": seam_length}
    result["soffit"] = {"points": int(soffit.sum()), "cells_with_returns": int(valid.sum()),
                        "cells_searched": int(usable.sum()), "area_searched_m2": float(usable.sum() * CELL**2)}
    result["note"] = ("Candidates are colour or height departures from the surrounding soffit, for an inspector "
                      "to confirm; formwork marks, shadows, wet patches and repairs can look the same.")
    if out_dir:
        result["outputs"] = _write(result, plan, (image[::-1], a0, a0 + na * CELL, u0, u0 + nu * CELL), Path(out_dir))
    return result


def _write(result: dict[str, Any], plan: dict[str, Any], ortho: tuple[np.ndarray, float, float, float, float] | None,
           out: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "damage.json", "csv": out / "damage.csv", "svg": out / "damage_plan.svg"}
    with open(paths["csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "label_ja", "element", "area_m2", "centre_along_m", "centre_lateral_m",
                    "contrast", "contrast_unit"])
        for c in result["candidates"]:
            w.writerow([c["id"], c["kind"], c["label_ja"], c["element"] or "", f"{c['area_m2']:.4f}",
                        f"{c['centre_m'][0]:.3f}", f"{c['centre_m'][1]:.3f}", f"{c['contrast']:.3f}",
                        c["contrast_unit"]])
    underlay = None
    if ortho is not None:
        import matplotlib.image as mpimg

        image, a0, a1, u0, u1 = ortho
        buf = io.BytesIO()
        mpimg.imsave(buf, np.clip(image, 0, 1), format="png")
        paths["png"] = out / "soffit_ortho.png"
        paths["png"].write_bytes(buf.getvalue())
        underlay = (base64.b64encode(buf.getvalue()).decode("ascii"), a0, a1, u0, u1)

    def overlay(X: Any, Y: Any, at: Any) -> list[str]:
        parts = []
        for c in result["candidates"]:
            (a, b), (lo, hi) = c["along_m"], c["lateral_m"]
            colour = KINDS[c["kind"]][1]
            parts.append(f'<rect x="{X(a):.1f}" y="{Y(hi):.1f}" width="{max(X(b) - X(a), 2):.1f}" '
                         f'height="{max(Y(lo) - Y(hi), 2):.1f}" fill="none" stroke="{colour}" stroke-width="2"/>')
            parts.append(f'<text x="{X(b) + 3:.1f}" y="{Y(hi) + 10:.1f}" font-size="10" fill="{colour}">'
                         f'{c["id"]}</text>')
        return parts

    legend = (" ".join(f'<tspan fill="{colour}">■ {label}</tspan>' for label, colour in KINDS.values())
              + f' · 候補 {len(result["candidates"])} 件（要確認・診断ではありません）')
    svg = plan_svg(plan, underlay=underlay, overlay=overlay, footer=legend)
    paths["svg"].write_text(svg, encoding="utf-8")
    paths["json"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}
