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
from pathlib import Path
from typing import Any

import numpy as np

from ca.bridge import Axis, cut, measure_section, structure

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


def _peaks(values: np.ndarray, lo: float, hi: float, min_count: int = 3) -> list[tuple[float, float]]:
    """Centre and width of runs of occupied bins over [lo, hi]: a member's faces and
    bottom together, with at most one empty bin inside."""
    if len(values) == 0 or hi <= lo:
        return []
    h, e = np.histogram(values, bins=np.arange(lo, hi + MEMBER_BIN, MEMBER_BIN))
    on = h >= max(min_count, 0.1 * h.max(initial=0))
    out, i = [], 0
    while i < len(on):
        if not on[i]:
            i += 1
            continue
        j = i
        while j < len(on) and (on[j] or (j + 1 < len(on) and on[j + 1])):
            j += 1
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
    nearest = np.abs(along[:, None] - stations[None, :]).argmin(axis=1) if len(stations) else np.zeros(0, int)
    depth = np.full(len(points), np.nan)
    for k, s in enumerate(sections):
        m = nearest == k
        depth[m] = np.interp(lat[m], s["profile"]["u"], s["profile"]["z"]) - points[m, 2]
    return along, lat, depth


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
) -> dict[str, Any]:
    """Plan of a bridge with numbered elements for damage-map bases.

    `classes` and `axis_classes` work as in :func:`ca.bridge.bridge_sections`.
    `member_classes` selects the points searched for girders and cross beams,
    for example girder and deck labels without the abutments, whose walls under
    the deck ends would otherwise read as end cross beams.
    """
    if not (0 < spacing <= 5) or not (0 < thickness <= 1):
        raise ValueError("spacing must be in (0, 5] m and thickness in (0, 1] m")
    xyz, cls, pts, ax, ends, lo, hi = structure(path, classes, axis_classes, axis)
    if ends is None:
        raise ValueError("the deck ends could not be found; check the axis or classes")
    if member_classes and cls is None:
        raise ValueError("member classes were given but the file has no classification")
    members_pts = xyz[np.isin(cls, member_classes)] if member_classes and cls is not None else pts

    sections = []
    for station in np.arange(lo + margin, hi - margin + 1e-9, spacing):
        _, u, z = cut(pts, ax, float(station), thickness)
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
        # Girders: narrow members across each cross section that persist along the deck.
        peaks: list[tuple[float, ...]] = []
        for s in sections:
            if s["slab_m"] is None:
                continue
            sel = stem & inside & (np.abs(along - s["station_m"]) <= thickness / 2)
            for centre, width in _peaks(lat[sel], right_u - 0.2, left_u + 0.2):
                if width <= MEMBER_MAX_WIDTH:
                    on = sel & (np.abs(lat - centre) <= width / 2 + MEMBER_BIN)
                    peaks.append((s["station_m"], centre, width, float(np.percentile(depth[on], 95))))
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

    # Rows between the deck ends and cross beams (skewed along-axis positions at mid width),
    # columns between the deck edges and girders from the left.
    start, end = start_b + slope * mid, end_b + slope * mid
    row_edges = [start] + [c["along_m"] for c in cross] + [end]
    col_edges = [left_u] + [g["lateral_m"] for g in girders] + [right_u]
    elements: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(zip(row_edges, row_edges[1:])):
        for j, (l, r) in enumerate(zip(col_edges, col_edges[1:])):
            elements.append({"id": f"Ds{i + 1:02d}{j + 1:02d}", "kind": "deck_slab", "label_ja": "床版",
                             "row": i + 1, "column": j + 1, "along_m": [a, b], "lateral_m": [r, l],
                             "status": "observed_bounds" if girders or not underside else "single_panel"})
        for g in girders:
            elements.append({"id": f"Mg{i + 1:02d}{g['number']:02d}", "kind": "main_girder", "label_ja": "主桁",
                             "row": i + 1, "column": g["number"], "along_m": [a, b],
                             "lateral_m": [g["lateral_m"], g["lateral_m"]], "status": "observed"})
    for c in cross:
        for j, (gl, gr) in enumerate(zip(girders, girders[1:])):
            elements.append({"id": f"Cr{c['number']:02d}{j + 1:02d}", "kind": "cross_beam", "label_ja": "横桁",
                             "row": c["number"], "column": j + 1, "along_m": [c["along_m"], c["along_m"]],
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
        "elements": elements,
        "numbering": "Four digits: row along the axis from the start, then column across from the left facing the "
                     "end (the top of the drawing). Mg main girder, Cr cross beam, Ds deck slab.",
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
        w.writerow(["id", "kind", "label_ja", "row", "column", "along_from_m", "along_to_m",
                    "lateral_from_m", "lateral_to_m", "status"])
        for e in result["elements"]:
            w.writerow([e["id"], e["kind"], e["label_ja"], e["row"], e["column"],
                        f"{e['along_m'][0]:.3f}", f"{e['along_m'][1]:.3f}",
                        f"{e['lateral_m'][0]:.3f}", f"{e['lateral_m'][1]:.3f}", e["status"]])
    paths["svg"].write_text(plan_svg(result), encoding="utf-8")
    paths["json"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def plan_svg(result: dict[str, Any]) -> str:
    """The plan looking down through the deck: start on the left, the left side facing the end on top."""
    d = result["deck"]
    slope, mid = d["skew_slope"], 0.5 * (d["left_edge_m"] + d["right_edge_m"])
    at = lambda s, u: s + slope * (u - mid)  # along-axis position of skewed station s at lateral u
    a_lo = min(at(d["start_m"], d["left_edge_m"]), at(d["start_m"], d["right_edge_m"])) - 2.0
    a_hi = max(at(d["end_m"], d["left_edge_m"]), at(d["end_m"], d["right_edge_m"])) + 1.6
    u_hi, u_lo = d["left_edge_m"] + 1.6, d["right_edge_m"] - 1.4
    scale = 1100.0 / (a_hi - a_lo)
    W, H = 1100.0, (u_hi - u_lo) * scale + 80
    X = lambda a: (a - a_lo) * scale
    Y = lambda u: (u_hi - u) * scale
    P = lambda s, u: f"{X(at(s, u)):.1f},{Y(u):.1f}"
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
             f'font-family="sans-serif" font-size="13">',
             '<defs><marker id="t" viewBox="0 0 2 10" refX="1" refY="5" markerWidth="2" markerHeight="10" '
             'orient="auto"><path d="M1 0V10" stroke="#1a73e8"/></marker></defs>',
             '<rect width="100%" height="100%" fill="white"/>']
    L, R, S, E = d["left_edge_m"], d["right_edge_m"], d["start_m"], d["end_m"]
    parts.append(f'<polygon points="{P(S, L)} {P(E, L)} {P(E, R)} {P(S, R)}" fill="#fdf6ec" '
                 f'stroke="#e8710a" stroke-width="2.5"/>')
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
    parts.append(f'<text x="{X(at(S, R)):.1f}" y="{Y(R) + 22:.1f}" text-anchor="middle" '
                 f'font-weight="bold">起点</text>')
    parts.append(f'<text x="{X(at(E, R)):.1f}" y="{Y(R) + 22:.1f}" text-anchor="middle" '
                 f'font-weight="bold">終点</text>')
    y = H - 46
    below = {"unobserved": "下面は未観測：主桁・横桁は描いていません（仮定なし）",
             "no_girders_observed": "下面を観測：床版下に突出する部材なし（床版橋または計測範囲外）",
             "girders_observed": "主桁・横桁は点群で観測された位置（破線＝床版下の部材）"}[result["underside"]]
    parts.append(f'<text x="12" y="{y:.0f}">平面図（上から透視）· 起点を左、終点を向いて左側を上 · 単位 mm · '
                 f'斜角 {d["skew_deg"]:.1f}° · 要素番号＝橋軸方向（起点から）2桁＋直角方向（上から）2桁</text>')
    parts.append(f'<text x="12" y="{y + 22:.0f}" fill="#5f6368">{below}</text>')
    if result["warnings"]:
        parts.append(f'<text x="12" y="{y + 42:.0f}" fill="#d93025">一部の断面でのみ見える部材線があります。'
                     f'番号を使う前に点群を確認してください。</text>')
    parts.append("</svg>")
    return "\n".join(parts)
