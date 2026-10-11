"""Bridge plan for damage-map bases: deck outline, observed girders and cross beams, numbered elements.

Only what the point cloud shows is drawn. Girders are found beneath the slab as
narrow members that run along the deck in most cross sections; cross beams as
members between girders that run parallel to the (possibly skewed) deck ends.
The deck slab is divided into elements by the deck ends, the cross beams and the
girders. Without returns beneath the deck nothing below it is drawn or assumed.

Element numbers have four digits: the row along the axis from the start, then the
column across from the left facing the end (the top of the drawing), each from 01.
Main girders are ``Mg``, cross beams ``Cr`` and deck slab elements ``Ds``.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ca.bridge import Axis, measure_section, structure

SCHEMA = "cloudanalyzer.bridge_plan.v1"

SOFFIT_MIN_SHARE = 0.5     # share of sections with a measured soffit for the underside to count as observed
MEMBER_BIN = 0.05          # histogram bin for member faces (m)
MEMBER_MIN_STEM = 0.2      # a member protrudes at least this far below the soffit
MEMBER_MAX_DEPTH = 2.5     # below the soffit; deeper returns are ground or water
MEMBER_MAX_WIDTH = 1.2     # wider runs are walls or ground, not webs
MEMBER_JOIN = 0.3          # peaks within this lateral distance are the same member
GIRDER_MIN_SHARE = 0.5     # sections in which a girder must appear
GIRDER_WEAK_SHARE = 0.1    # below this a cluster is noise; between, a partial observation
CROSS_MIN_SHARE = 0.5      # girder bays in which a cross beam must appear
WEB_CLEARANCE = 0.15       # beyond a girder face before cross-beam returns count
SUPPORT_BELOW = 0.5        # a support reaches at least this far below the lowest girder (or slab)
SUPPORT_SPAN = 3.0         # depth range below that searched for supports (m)
SUPPORT_MAX_WIDTH = 3.0    # along the axis; wider runs are ground or embankment
SUPPORT_MAX_HOLLOW = 2.0   # a pier's unscanned interior between its two faces
SUPPORT_MAX_LEAN = 0.5     # a pier's upper and lower halves stand within this along the axis
SUPPORT_MIN_COVER = 0.3    # share of the deck width the support occupies
SUPPORT_END_CLEARANCE = 2.0  # from the deck ends: closer is an abutment, not a pier
SUPPORT_OPEN = 1.5         # open space beneath the deck on both sides of a pier (m)


def _peaks(values: np.ndarray, lo: float, hi: float, min_count: int = 3,
           gap: float = MEMBER_BIN) -> list[tuple[float, float]]:
    """Centre and width of runs of occupied bins over [lo, hi]: a member's faces and
    bottom together, with empty stretches of at most `gap` inside."""
    if len(values) == 0 or hi <= lo:
        return []
    h, e = np.histogram(values, bins=np.arange(lo, hi + MEMBER_BIN, MEMBER_BIN))
    on = h >= max(min_count, 0.1 * h.max(initial=0))
    reach = max(1, int(round(gap / MEMBER_BIN)))
    out, i = [], 0
    while i < len(on):
        if not on[i]:
            i += 1
            continue
        j = i
        while j < len(on) and on[j:j + reach + 1].any():
            j += 1
        while not on[j - 1]:
            j -= 1
        out.append((float(0.5 * (e[i] + e[j])), float(e[j] - e[i])))
        i = j
    return out


def _clusters(rows: list[tuple[float, ...]], key: int) -> list[list[tuple[float, ...]]]:
    """Group rows whose `key` values lie within MEMBER_JOIN of their group's median, so
    scattered returns between two members cannot chain them into one."""
    rows = sorted(rows, key=lambda r: r[key])
    groups: list[list[tuple[float, ...]]] = []
    for r in rows:
        if groups and r[key] - float(np.median([g[key] for g in groups[-1]])) <= MEMBER_JOIN:
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


def _below_deck(points: np.ndarray, ax: Axis, sections: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    """Along, lateral and depth below the deck top of `points`, each from its nearest section's profile."""
    rel = points[:, :2] - ax.origin
    along, lat = rel @ ax.direction, rel @ ax.lateral
    stations = np.array([s["station_m"] for s in sections])
    # Sections are sorted along the axis: the nearest is one of the two around each point.
    k = np.clip(np.searchsorted(stations, along), 1, len(stations) - 1)
    nearest = np.where(np.abs(along - stations[k - 1]) <= np.abs(along - stations[k]), k - 1, k)
    depth = np.full(len(points), np.nan)
    for i, s in enumerate(sections):
        m = nearest == i
        depth[m] = np.interp(lat[m], s["profile"]["u"], s["profile"]["z"]) - points[m, 2]
    return along, lat, depth


def _supports(along: np.ndarray, lat: np.ndarray, depth: np.ndarray, top: float, deck: dict[str, float],
              slope: float) -> list[dict[str, Any]]:
    """Piers: vertical structure well below the superstructure, isolated from the deck ends.

    Returns between `top` and `top + SUPPORT_SPAN` below the deck top, across the deck
    width, are binned along the (skewed) axis. A support is a run at most
    SUPPORT_MAX_WIDTH long whose upper and lower halves stand at the same place (an
    abutment's sloped face leans), across a fair share of the width, with open space
    beneath the deck on both sides and at least SUPPORT_END_CLEARANCE from both deck ends.
    """
    mid = 0.5 * (deck["left"] + deck["right"])
    sel = (depth > top) & (depth < top + SUPPORT_SPAN) & (lat > deck["right"]) & (lat < deck["left"])
    skewed = along[sel] - slope * (lat[sel] - mid)
    out = []
    for centre, width in _peaks(skewed, deck["start"], deck["end"], min_count=20, gap=SUPPORT_MAX_HOLLOW):
        if width > SUPPORT_MAX_WIDTH or min(centre - deck["start"], deck["end"] - centre) < SUPPORT_END_CLEARANCE:
            continue
        on = np.abs(skewed - centre) <= width / 2
        d, u, sk = depth[sel][on], lat[sel][on], skewed[on]
        upper, lower = d < top + SUPPORT_SPAN / 2, d >= top + SUPPORT_SPAN / 2
        if upper.sum() < 10 or lower.sum() < 10:
            continue
        lean = abs(float(np.median(sk[upper]) - np.median(sk[lower])))
        bins = np.histogram(u, bins=np.arange(deck["right"], deck["left"] + 0.25, 0.25))[0]
        cover = float((bins > 0).mean())
        # Open on both sides: an abutment has its body on the side of its deck end.
        before = (skewed >= centre - width / 2 - SUPPORT_OPEN) & (skewed < centre - width / 2)
        after = (skewed > centre + width / 2) & (skewed <= centre + width / 2 + SUPPORT_OPEN)
        open_sides = max(before.sum(), after.sum()) <= 0.05 * on.sum()
        if lean <= SUPPORT_MAX_LEAN and cover >= SUPPORT_MIN_COVER and open_sides:
            out.append({"along_m": centre, "width_m": width, "lean_m": lean, "width_share": cover,
                        "source": "detected"})
    return out


def bridge_plan(
    path: str,
    out_dir: str | None = None,
    *,
    spacing: float = 0.25,
    thickness: float = 0.1,
    classes: list[int] | None = None,
    axis_classes: list[int] | None = None,
    member_classes: list[int] | None = None,
    axis: tuple[list[float], list[float]] | None = None,
    margin: float = 0.3,
    supports: list[float] | None = None,
) -> dict[str, Any]:
    """Plan of a bridge with numbered elements for damage-map bases.

    `classes` and `axis_classes` work as in :func:`ca.bridge.bridge_sections`.
    `member_classes` selects the points searched for girders and cross beams,
    for example girder and deck labels without the abutments, whose walls under
    the deck ends would otherwise read as end cross beams. `supports` gives the
    piers as distances from the start along the centreline (from a bridge register);
    without it, isolated vertical supports found beneath the deck are used.
    """
    if not (0 < spacing <= 5) or not (0 < thickness <= 1):
        raise ValueError("spacing must be in (0, 5] m and thickness in (0, 1] m")
    xyz, cls, pts, ax, ends, lo, hi = structure(path, classes, axis_classes, axis)
    if ends is None:
        raise ValueError("the deck ends could not be found; check the axis or classes")
    if member_classes and cls is None:
        raise ValueError("member classes were given but the file has no classification")
    members_pts = xyz[np.isin(cls, member_classes)] if member_classes and cls is not None else pts

    rel = pts[:, :2] - ax.origin
    order = np.argsort(rel @ ax.direction)
    pts_sorted = pts[order]
    along_sorted = (rel @ ax.direction)[order]
    lat_sorted = (rel @ ax.lateral)[order]
    sections = []
    for station in np.arange(lo + margin, hi - margin + 1e-9, spacing):
        a, b = np.searchsorted(along_sorted, [station - thickness / 2, station + thickness / 2], side="right")
        u, z = lat_sorted[a:b], pts_sorted[a:b, 2]
        m = measure_section(u, z)
        if m["status"] != "measured":
            continue
        sections.append({"station_m": float(station), "profile": m["deck_profile"],
                         "left_u": m["left"].get("outer_u", m["deck"]["hi"]),
                         "right_u": m["right"].get("outer_u", m["deck"]["lo"]),
                         "slab_m": m["dims"].get("slab_thickness_m")})
    if len(sections) < 2:
        raise ValueError("fewer than two cross sections show a deck surface")
    left_u = float(np.median([s["left_u"] for s in sections]))
    right_u = float(np.median([s["right_u"] for s in sections]))
    slabs = [s["slab_m"] for s in sections if s["slab_m"] is not None]
    underside = len(slabs) >= SOFFIT_MIN_SHARE * len(sections)
    slab = float(np.median(slabs)) if slabs else None

    slope, start_b = ends["start_line"]
    end_b = ends["end_line"][1]
    mid = 0.5 * (left_u + right_u)
    girders: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    cross: list[dict[str, Any]] = []
    if underside and slab is not None:
        along, lat, depth = _below_deck(members_pts, ax, sections)
        stem = (depth > slab + MEMBER_MIN_STEM) & (depth < slab + MEMBER_MAX_DEPTH)
        inside = (lat > right_u - 0.1) & (lat < left_u + 0.1)
        keep = np.nonzero(stem & inside)[0]
        keep = keep[np.argsort(along[keep])]
        keep_along = along[keep]
        # Girders: narrow members across each cross section that persist along the deck.
        peaks: list[tuple[float, ...]] = []
        for sec in sections:
            if sec["slab_m"] is None:
                continue
            a, b = np.searchsorted(keep_along, [sec["station_m"] - thickness / 2, sec["station_m"] + thickness / 2],
                                   side="right")
            idx = keep[a:b]
            for centre, width in _peaks(lat[idx], right_u - 0.2, left_u + 0.2):
                if width <= MEMBER_MAX_WIDTH:
                    on = idx[np.abs(lat[idx] - centre) <= width / 2 + MEMBER_BIN]
                    peaks.append((sec["station_m"], centre, width, float(np.percentile(depth[on], 95))))
        observed = len(slabs)
        for group in _clusters(peaks, 1):
            share = len({g[0] for g in group}) / observed
            info = {"lateral_m": float(np.median([g[1] for g in group])),
                    "visible_width_m": float(np.median([g[2] for g in group])),
                    "depth_below_deck_top_m": float(np.median([g[3] for g in group])),
                    "sections_observed": len({g[0] for g in group}), "sections_total": observed}
            if share >= GIRDER_MIN_SHARE:
                girders.append(info)
            elif share >= GIRDER_WEAK_SHARE:
                partial.append(info)
        # Numbered from the left facing the end: the top of the drawing.
        girders.sort(key=lambda g: -g["lateral_m"])
        for k, g in enumerate(girders):
            g["number"] = k + 1
        # Cross beams: members between adjacent girders, parallel to the deck ends.
        skewed = along - slope * (lat - mid)
        bays = list(zip(girders, girders[1:]))
        found: list[tuple[float, ...]] = []
        for b, (g_left, g_right) in enumerate(bays):
            a = g_right["lateral_m"] + g_right["visible_width_m"] / 2 + WEB_CLEARANCE
            c = g_left["lateral_m"] - g_left["visible_width_m"] / 2 - WEB_CLEARANCE
            sel = stem & (lat > a) & (lat < c)
            first, last = start_b + slope * mid, end_b + slope * mid
            for centre, width in _peaks(skewed[sel], first, last):
                if width <= MEMBER_MAX_WIDTH:
                    on = sel & (np.abs(skewed - centre) <= width / 2 + MEMBER_BIN)
                    found.append((float(b), centre, width, float(np.percentile(depth[on], 95))))
        for group in _clusters(found, 1):
            share = len({g[0] for g in group}) / max(len(bays), 1)
            if share >= CROSS_MIN_SHARE:
                cross.append({"along_m": float(np.median([g[1] for g in group])),
                              "visible_width_m": float(np.median([g[2] for g in group])),
                              "depth_below_deck_top_m": float(np.median([g[3] for g in group])),
                              "bays_observed": len({g[0] for g in group}), "bays_total": len(bays)})
        cross.sort(key=lambda c: c["along_m"])
        for k, c in enumerate(cross):
            c["number"] = k + 1

    start, end = start_b + slope * mid, end_b + slope * mid
    if supports is not None:
        if any(not (0 < v < end - start) for v in supports):
            raise ValueError("supports must lie between the deck ends, as distances from the start")
        piers = [{"along_m": start + float(v), "source": "explicit"} for v in sorted(supports)]
    else:
        p_along, p_lat, p_depth = _below_deck(pts, ax, sections)
        lowest = max([g["depth_below_deck_top_m"] for g in girders] + [slab or 0.0])
        piers = _supports(p_along, p_lat, p_depth, lowest + SUPPORT_BELOW,
                          {"start": start, "end": end, "left": left_u, "right": right_u}, slope)
    for k, p in enumerate(piers):
        p["label"] = f"P{k + 1}"
        p["from_start_m"] = p["along_m"] - start
    span_edges = [start] + [p["along_m"] for p in piers] + [end]
    spans = [{"number": k + 1, "from_m": a, "to_m": b, "length_m": b - a}
             for k, (a, b) in enumerate(zip(span_edges, span_edges[1:]))]
    # Rows between the deck ends, cross beams and supports (skewed along-axis positions at mid
    # width; a cross beam over a support is the same line), columns between the deck edges and
    # girders from the left.
    lines = sorted([c["along_m"] for c in cross]
                   + [p["along_m"] for p in piers
                      if all(abs(p["along_m"] - c["along_m"]) > MEMBER_JOIN for c in cross)])
    row_edges = [start] + lines + [end]

    def span_of(a: float, b: float) -> int:
        centre = 0.5 * (a + b)
        return int(next(sp["number"] for sp in spans if sp["from_m"] - 1e-9 <= centre <= sp["to_m"] + 1e-9))

    col_edges = [left_u] + [g["lateral_m"] for g in girders] + [right_u]
    elements: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(zip(row_edges, row_edges[1:])):
        for j, (l, r) in enumerate(zip(col_edges, col_edges[1:])):
            elements.append({"id": f"Ds{i + 1:02d}{j + 1:02d}", "kind": "deck_slab", "label_ja": "床版",
                             "row": i + 1, "column": j + 1, "span": span_of(a, b), "along_m": [a, b],
                             "lateral_m": [r, l],
                             "status": "observed_bounds" if girders or not underside else "single_panel"})
        for g in girders:
            elements.append({"id": f"Mg{i + 1:02d}{g['number']:02d}", "kind": "main_girder", "label_ja": "主桁",
                             "row": i + 1, "column": g["number"], "span": span_of(a, b), "along_m": [a, b],
                             "lateral_m": [g["lateral_m"], g["lateral_m"]], "status": "observed"})
    for c in cross:
        for j, (gl, gr) in enumerate(zip(girders, girders[1:])):
            elements.append({"id": f"Cr{c['number']:02d}{j + 1:02d}", "kind": "cross_beam", "label_ja": "横桁",
                             "row": c["number"], "column": j + 1, "span": span_of(c["along_m"], c["along_m"]),
                             "along_m": [c["along_m"], c["along_m"]],
                             "lateral_m": [gr["lateral_m"], gl["lateral_m"]], "status": "observed"})

    if not underside:
        below = "unobserved"
    elif girders:
        below = "girders_observed"
    else:
        below = "no_girders_observed"
    notes = {
        "unobserved": "No returns beneath most of the deck: girders and cross beams are neither drawn nor assumed; "
                      "the deck is one element.",
        "no_girders_observed": "The soffit was scanned and shows no member protruding 0.2 m or more below it "
                               "(a slab deck, or members outside the scan); the deck is divided only lengthwise.",
        "girders_observed": "Girders and cross beams are drawn where the scan shows them; a member hidden from "
                            "every scan position is missing and would shift the numbers after it.",
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "source": str(Path(path).resolve()),
        "points": int(len(xyz)),
        "classes": classes, "member_classes": member_classes,
        "axis": {"origin_xy": ax.origin.tolist(), "direction_xy": ax.direction.tolist(), "source": ax.source},
        "options": {"spacing_m": spacing, "thickness_m": thickness, "margin_m": margin},
        "deck": {"start_m": start, "end_m": end, "length_m": end - start, "skew_deg": ends["skew_deg"],
                 "skew_slope": slope, "left_edge_m": left_u, "right_edge_m": right_u,
                 "width_m": left_u - right_u, "slab_thickness_m": slab,
                 "sections_measured": len(sections), "sections_with_soffit": len(slabs)},
        "underside": below,
        "girders": girders,
        "partially_observed_members": partial,
        "cross_beams": cross,
        "supports": piers,
        "spans": spans,
        "elements": elements,
        "numbering": "Four digits: row along the axis from the start, then column across from the left facing the "
                     "end (the top of the drawing). Mg main girder, Cr cross beam, Ds deck slab. Rows continue "
                     "across spans; each element records its span. A1 and A2 are the start and end abutments, "
                     "P1, P2, ... the piers from the start.",
        "note": notes[below],
        "warnings": ([f"{len(partial)} member line(s) appear in only part of the sections and are not numbered; "
                      "check the scan before relying on the girder numbers."] if partial else []),
    }
    if out_dir:
        result["outputs"] = _write(result, Path(out_dir))
    return result


def _write(result: dict[str, Any], out: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "bridge_plan.json", "csv": out / "members.csv", "svg": out / "plan.svg"}
    with open(paths["csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "kind", "label_ja", "span", "row", "column", "along_from_m", "along_to_m",
                    "lateral_from_m", "lateral_to_m", "status"])
        for e in result["elements"]:
            w.writerow([e["id"], e["kind"], e["label_ja"], e["span"], e["row"], e["column"],
                        f"{e['along_m'][0]:.3f}", f"{e['along_m'][1]:.3f}",
                        f"{e['lateral_m'][0]:.3f}", f"{e['lateral_m'][1]:.3f}", e["status"]])
    paths["svg"].write_text(plan_svg(result), encoding="utf-8")
    paths["json"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def plan_svg(result: dict[str, Any], underlay: tuple[str, float, float, float, float] | None = None,
             overlay: Callable[..., list[str]] | None = None, footer: str | None = None) -> str:
    """The plan looking down through the deck: start on the left, the left side facing the end on top.

    `underlay` is a base64 PNG and the along/lateral range it covers (a0, a1, u0, u1), its first
    row at u1; `overlay(X, Y, at)` returns SVG elements drawn over the plan with its mapping;
    `footer` is an extra line of SVG text content under the notes."""
    d = result["deck"]
    slope, mid = d["skew_slope"], 0.5 * (d["left_edge_m"] + d["right_edge_m"])
    at = lambda s, u: s + slope * (u - mid)  # along-axis position of skewed station s at lateral u
    a_lo = min(at(d["start_m"], d["left_edge_m"]), at(d["start_m"], d["right_edge_m"])) - 2.0
    a_hi = max(at(d["end_m"], d["left_edge_m"]), at(d["end_m"], d["right_edge_m"])) + 1.6
    u_hi, u_lo = d["left_edge_m"] + 1.6, d["right_edge_m"] - (2.0 if len(result["spans"]) > 1 else 1.4)
    scale = 1100.0 / (a_hi - a_lo)
    W, H = 1100.0, (u_hi - u_lo) * scale + (120 if len(result["spans"]) > 1 else 80) + (22 if footer else 0)
    X = lambda a: (a - a_lo) * scale
    Y = lambda u: (u_hi - u) * scale
    P = lambda s, u: f"{X(at(s, u)):.1f},{Y(u):.1f}"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
             f'font-family="sans-serif" font-size="13">',
             '<defs><marker id="t" viewBox="0 0 2 10" refX="1" refY="5" markerWidth="2" markerHeight="10" '
             'orient="auto"><path d="M1 0V10" stroke="#1a73e8"/></marker></defs>',
             '<rect width="100%" height="100%" fill="white"/>']
    L, R, S, E = d["left_edge_m"], d["right_edge_m"], d["start_m"], d["end_m"]
    parts.append(f'<polygon points="{P(S, L)} {P(E, L)} {P(E, R)} {P(S, R)}" '
                 f'fill="{"none" if underlay else "#fdf6ec"}" stroke="#e8710a" stroke-width="2.5"/>')
    if underlay:
        png, a0, a1, u0, u1 = underlay
        parts.insert(-1, f'<image x="{X(a0):.1f}" y="{Y(u1):.1f}" width="{X(a1) - X(a0):.1f}" '
                         f'height="{Y(u0) - Y(u1):.1f}" preserveAspectRatio="none" '
                         f'href="data:image/png;base64,{png}"/>')
    hidden = 'stroke="#5f6368" stroke-width="1.6" stroke-dasharray="8 5"'
    for g in result["girders"]:
        u = g["lateral_m"]
        parts.append(f'<line x1="{X(at(S, u)):.1f}" y1="{Y(u):.1f}" x2="{X(at(E, u)):.1f}" y2="{Y(u):.1f}" {hidden}/>')
        parts.append(f'<text x="{X(at(S, u)) - 8:.1f}" y="{Y(u) + 4:.1f}" text-anchor="end" fill="#5f6368">'
                     f'G{g["number"]}</text>')
    for p in result["partially_observed_members"]:
        u = p["lateral_m"]
        parts.append(f'<line x1="{X(at(S, u)):.1f}" y1="{Y(u):.1f}" x2="{X(at(E, u)):.1f}" y2="{Y(u):.1f}" '
                     f'stroke="#d93025" stroke-width="1.2" stroke-dasharray="2 6"/>')
        parts.append(f'<text x="{X(at(E, u)) + 8:.1f}" y="{Y(u) + 4:.1f}" fill="#d93025">一部のみ観測</text>')
    if result["girders"]:
        top, bottom = result["girders"][0]["lateral_m"], result["girders"][-1]["lateral_m"]
        for c in result["cross_beams"]:
            s = c["along_m"]
            parts.append(f'<line x1="{X(at(s, top)):.1f}" y1="{Y(top):.1f}" x2="{X(at(s, bottom)):.1f}" '
                         f'y2="{Y(bottom):.1f}" {hidden}/>')
    for p in result["supports"]:
        s_ = p["along_m"]
        parts.append(f'<line x1="{X(at(s_, L)):.1f}" y1="{Y(L):.1f}" x2="{X(at(s_, R)):.1f}" y2="{Y(R):.1f}" '
                     f'stroke="#3c4043" stroke-width="2.2" stroke-dasharray="14 4 3 4"/>')
        parts.append(f'<text x="{X(at(s_, L)):.1f}" y="{Y(L) - 8:.1f}" text-anchor="middle" font-weight="bold" '
                     f'fill="#3c4043">{p["label"]}</text>')
    for label, s_ in (("A1", S), ("A2", E)):
        parts.append(f'<text x="{X(at(s_, L)):.1f}" y="{Y(L) - 8:.1f}" text-anchor="middle" font-weight="bold" '
                     f'fill="#3c4043">{label}</text>')
    for e in result["elements"]:
        (a, b), (r, l) = e["along_m"], e["lateral_m"]
        if e["kind"] == "deck_slab":
            sm, um = 0.5 * (a + b), 0.5 * (l + r)
            parts.append(f'<text x="{X(at(sm, um)):.1f}" y="{Y(um) + 5:.1f}" text-anchor="middle" '
                         f'font-size="15" fill="#202124">{e["id"]}</text>')
        elif e["kind"] == "main_girder":
            sm = 0.5 * (a + b)
            parts.append(f'<text x="{X(at(sm, l)):.1f}" y="{Y(l) - 5:.1f}" text-anchor="middle" font-size="11" '
                         f'fill="#5f6368">{e["id"]}</text>')
        else:
            um = 0.5 * (l + r)
            parts.append(f'<text x="{X(at(a, um)) + 4:.1f}" y="{Y(um) - 4:.1f}" font-size="11" fill="#5f6368" '
                         f'transform="rotate(-90 {X(at(a, um)) + 4:.1f} {Y(um) - 4:.1f})">{e["id"]}</text>')

    def dim(x1: float, y1: float, x2: float, y2: float, text: str, dx: float = 0, dy: float = -6) -> None:
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#1a73e8" '
                     f'stroke-width="1.2" marker-start="url(#t)" marker-end="url(#t)"/>')
        x, y = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
        # Vertical dimensions read along their line, beside it.
        turn = f' transform="rotate(-90 {x:.1f} {y:.1f})"' if abs(x1 - x2) < 1e-6 else ""
        parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" fill="#1a73e8"{turn}>{text}</text>')

    y_len = Y(L) - 40
    dim(X(at(S, mid)), y_len, X(at(E, mid)), y_len, f"橋長方向 床版長（中心線） {d['length_m'] * 1000:.0f}")
    x_w = X(a_hi) - 30
    dim(x_w, Y(L), x_w, Y(R), f"全幅 {d['width_m'] * 1000:.0f}", dx=16)
    edges = [L] + [g["lateral_m"] for g in result["girders"]] + [R]
    if result["girders"]:
        x_g = X(a_lo) + 40
        for u1, u2 in zip(edges, edges[1:]):
            dim(x_g, Y(u1), x_g, Y(u2), f"{(u1 - u2) * 1000:.0f}", dx=-10)
    if result["cross_beams"]:
        stations = [S] + [c["along_m"] for c in result["cross_beams"]] + [E]
        y_c = Y(R) + 50
        for s1, s2 in zip(stations, stations[1:]):
            dim(X(at(s1, mid)), y_c, X(at(s2, mid)), y_c, f"{(s2 - s1) * 1000:.0f}", dy=16)
    if len(result["spans"]) > 1:
        y_s = Y(R) + 75
        for sp in result["spans"]:
            dim(X(at(sp["from_m"], mid)), y_s, X(at(sp["to_m"], mid)), y_s,
                f"第{sp['number']}径間 {sp['length_m'] * 1000:.0f}", dy=16)
    parts.append(f'<text x="{X(at(S, R)):.1f}" y="{Y(R) + 22:.1f}" text-anchor="middle" '
                 f'font-weight="bold">起点</text>')
    parts.append(f'<text x="{X(at(E, R)):.1f}" y="{Y(R) + 22:.1f}" text-anchor="middle" '
                 f'font-weight="bold">終点</text>')
    if overlay:
        parts.extend(overlay(X, Y, at))
    y = H - 46 - (22 if footer else 0)
    below = {"unobserved": "下面は未観測：主桁・横桁は描いていません（仮定なし）",
             "no_girders_observed": "下面を観測：床版下に突出する部材なし（床版橋または計測範囲外）",
             "girders_observed": "主桁・横桁は点群で観測された位置（破線＝床版下の部材）"}[result["underside"]]
    parts.append(f'<text x="12" y="{y:.0f}">平面図（上から透視）· 起点を左、終点を向いて左側を上 · 単位 mm · '
                 f'斜角 {d["skew_deg"]:.1f}° · 要素番号＝橋軸方向（起点から）2桁＋直角方向（上から）2桁</text>')
    parts.append(f'<text x="12" y="{y + 22:.0f}" fill="#5f6368">{below}</text>')
    if result["warnings"]:
        parts.append(f'<text x="12" y="{y + 42:.0f}" fill="#d93025">一部の断面でのみ見える部材線があります。'
                     f'番号を使う前に点群を確認してください。</text>')
    if footer:
        parts.append(f'<text x="12" y="{y + (64 if result["warnings"] else 42):.0f}" font-size="12">{footer}</text>')
    parts.append("</svg>")
    return "\n".join(parts)
