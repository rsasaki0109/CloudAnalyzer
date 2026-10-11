"""Cross sections and a member dimension table for a bridge point cloud.

Sections are cut perpendicular to the bridge axis at a fixed spacing. Each
section is measured from geometry alone: the deck top line, the curbs (地覆)
at both ends, the parapets above them, the visible depth of the outer faces
and, only when returns exist beneath the deck, the slab thickness. Per-section
values are aggregated into a table that keeps observed values, lower bounds and
unobserved items apart: a scan taken from the deck sees no soffit, so the slab
thickness and interior girders are reported as unobserved, never assumed.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA = "cloudanalyzer.bridge_sections.v1"

# Geometry tolerances (metres).
SURFACE_BIN = 0.05          # lateral bin for surface layers
LAYER_GAP = 0.06            # height gap separating layers within a bin
SURFACE_STEP = 0.03         # height change allowed between adjacent bins
MAX_BIN_GAP = 4             # empty bins a surface may bridge (0.2 m)
CROWN_MIN = 0.015           # rise above the chord reported as a crown
DECK_MAX_SLOPE = 0.12       # deck cross slope considered (12 %)
DECK_GAP = 0.25             # largest lateral gap inside a curb top
DECK_MIN_RUN = 1.5          # shortest deck run accepted
CURB_SEARCH = 1.2           # lateral search beyond a deck end for a curb
CURB_MIN_RISE = 0.05        # a curb top sits at least this far above the deck
CURB_MAX_RISE = 0.6
CURB_LEVEL = 0.03           # height band of the curb top plateau
FACE_BAND = 0.08            # lateral band of the outer face
FACE_GAP = 0.1              # vertical gap that ends the outer face
SOFFIT_GAP = 0.08           # soffit returns lie at least this far below the deck top
SOFFIT_MIN_SHARE = 0.3      # share of deck bins that need soffit returns
SOFFIT_BIN = 0.25

ITEMS = [
    # key, Japanese label, status when present
    ("total_width_m", "全幅", "observed"),
    ("effective_width_m", "有効幅員（地覆内側間）", "observed"),
    ("cross_slope_percent", "横断勾配（全体の直線当てはめ、%）", "observed"),
    ("crown_height_m", "路面の拝み高さ（弦からの盛り上がり）", "observed"),
    ("left_cross_slope_percent", "横断勾配（左側、%）", "observed"),
    ("right_cross_slope_percent", "横断勾配（右側、%）", "observed"),
    ("left_curb_width_m", "地覆幅（左）", "observed"),
    ("right_curb_width_m", "地覆幅（右）", "observed"),
    ("left_curb_height_m", "地覆高（左）", "observed"),
    ("right_curb_height_m", "地覆高（右）", "observed"),
    ("left_parapet_height_m", "高欄高（左、地覆天端から）", "observed"),
    ("right_parapet_height_m", "高欄高（右、地覆天端から）", "observed"),
    ("left_outer_face_depth_m", "外面の可視深さ（左、床版上面から）", "lower_bound"),
    ("right_outer_face_depth_m", "外面の可視深さ（右、床版上面から）", "lower_bound"),
    ("slab_thickness_m", "床版厚", "observed"),
]


CROWN_ONLY = {"left_cross_slope_percent", "right_cross_slope_percent"}


@dataclass
class Axis:
    origin: np.ndarray      # XY point on the axis
    direction: np.ndarray   # unit XY vector along the bridge
    source: str

    @property
    def lateral(self) -> np.ndarray:
        # Left of the direction of travel along the axis.
        return np.array([-self.direction[1], self.direction[0]])


def load_points(path: str) -> dict[str, np.ndarray]:
    """XYZ plus classification and RGB when the file carries them."""
    p = Path(path)
    if p.suffix.lower() in {".las", ".laz"}:
        import laspy

        las = laspy.read(str(p))
        out = {"xyz": np.column_stack([las.x, las.y, las.z]).astype(float)}
        names = set(las.point_format.dimension_names)
        if "classification" in names:
            out["classification"] = np.asarray(las.classification).astype(np.int32)
        if {"red", "green", "blue"} <= names:
            out["rgb"] = np.column_stack([las.red, las.green, las.blue]).astype(np.float32)
        return out
    from ca.io import load_point_cloud

    cloud = load_point_cloud(str(p))
    out = {"xyz": np.asarray(cloud.points, dtype=float)}
    if cloud.has_colors():
        out["rgb"] = np.asarray(cloud.colors, dtype=np.float32)
    return out


def estimate_axis(xyz: np.ndarray) -> Axis:
    """Principal XY direction of the given points (the deck or the structure)."""
    xy = xyz[:, :2]
    origin = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - origin, full_matrices=False)
    direction = vt[0] / np.linalg.norm(vt[0])
    return Axis(origin, direction, "principal_xy_direction")


def refine_axis(xyz: np.ndarray, axis: Axis, iterations: int = 4) -> Axis:
    """Turn the axis parallel to the deck's side edges.

    A skewed deck is a parallelogram whose principal direction leans toward its
    long diagonal. Its side edges (along the curbs) are parallel to the bridge,
    so lines through the lateral extremes of the middle of the deck give the
    axis direction regardless of the skew of its ends.
    """
    xy = xyz[:, :2]
    for _ in range(iterations):
        rel = xy - axis.origin
        along, lat = rel @ axis.direction, rel @ axis.lateral
        lo, hi = np.percentile(along, [20, 80])
        bins = np.linspace(lo, hi, 13)
        centers, left, right = [], [], []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (along >= a) & (along < b)
            if m.sum() >= 50:
                centers.append(0.5 * (a + b))
                right.append(np.percentile(lat[m], 1))
                left.append(np.percentile(lat[m], 99))
        if len(centers) < 4:
            return axis
        slope = 0.5 * (np.polyfit(centers, left, 1)[0] + np.polyfit(centers, right, 1)[0])
        turn = math.atan(slope)
        c, s = math.cos(turn), math.sin(turn)
        d = axis.direction
        direction = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        axis = Axis(axis.origin, direction / np.linalg.norm(direction), "deck_side_edges")
        if abs(turn) < 1e-4:
            break
    return axis


def axis_from_points(start: Iterable[float], end: Iterable[float]) -> Axis:
    a = np.asarray(list(start), dtype=float)[:2]
    b = np.asarray(list(end), dtype=float)[:2]
    d = b - a
    n = np.linalg.norm(d)
    if not math.isfinite(n) or n < 1e-6:
        raise ValueError("axis start and end must be distinct XY points")
    return Axis(a, d / n, "explicit")


def _runs(u: np.ndarray, gap: float) -> list[tuple[int, int]]:
    """Index ranges [i, j) of sorted u split where consecutive gaps exceed `gap`."""
    if len(u) == 0:
        return []
    cuts = np.where(np.diff(u) > gap)[0] + 1
    bounds = np.concatenate([[0], cuts, [len(u)]])
    return [(int(bounds[k]), int(bounds[k + 1])) for k in range(len(bounds) - 1)]


def _layers(z: np.ndarray, gap: float) -> list[np.ndarray]:
    zs = np.sort(z)
    cuts = np.nonzero(np.diff(zs) > gap)[0] + 1
    return [part for part in np.split(zs, cuts) if len(part) >= 2]


def _deck_surface(u: np.ndarray, z: np.ndarray) -> dict[str, Any] | None:
    """The deck top: the highest of the long, smooth, near-horizontal surfaces.

    Points are binned laterally and split into height layers; layers continuing
    smoothly into the next bins form surfaces. A scanned soffit or the ground
    can be as long as the deck, so among surfaces at least 60 % as long as the
    longest, the highest is the deck. Crowned decks are smooth, not straight.
    """
    if len(u) < 30:
        return None
    lo_u = float(np.min(u))
    idx = np.floor((u - lo_u) / SURFACE_BIN).astype(int)
    order = np.argsort(idx, kind="stable")
    bins = np.split(order, np.nonzero(np.diff(idx[order]))[0] + 1)
    nodes = []          # (bin index, layer median z)
    for members in bins:
        b = int(idx[members[0]])
        for layer in _layers(z[members], LAYER_GAP):
            nodes.append((b, float(np.median(layer))))
    if not nodes:
        return None
    # Link each node to the closest smooth node in the previous few bins.
    chain_of: dict[int, int] = {}
    chains: list[list[int]] = []
    by_bin: dict[int, list[int]] = {}
    for k, (b, zk) in enumerate(nodes):
        best, best_dz = None, None
        for back in range(1, MAX_BIN_GAP + 2):
            for j in by_bin.get(b - back, []):
                du = back * SURFACE_BIN
                dz = abs(zk - nodes[j][1])
                if dz <= SURFACE_STEP + DECK_MAX_SLOPE * du and (best_dz is None or dz < best_dz):
                    best, best_dz = j, dz
            if best is not None:
                break
        if best is not None and chains[chain_of[best]][-1] == best:
            chain_of[k] = chain_of[best]
            chains[chain_of[k]].append(k)
        else:
            chain_of[k] = len(chains)
            chains.append([k])
        by_bin.setdefault(b, []).append(k)
    spans = [(nodes[c[-1]][0] - nodes[c[0]][0] + 1) * SURFACE_BIN for c in chains]
    longest = max(spans)
    if longest < DECK_MIN_RUN:
        return None
    candidates = [c for c, sp in zip(chains, spans) if sp >= max(DECK_MIN_RUN, 0.6 * longest)]
    deck = max(candidates, key=lambda c: float(np.median([nodes[k][1] for k in c])))
    pu = np.array([lo_u + (nodes[k][0] + 0.5) * SURFACE_BIN for k in deck])
    pz = np.array([nodes[k][1] for k in deck])
    slope, icpt = np.polyfit(pu, pz, 1)
    # The ends of the surface: its returns just beyond the last linked bins,
    # up to the curb faces, rather than bin centres.
    near = (u >= pu[0] - 2 * SURFACE_BIN) & (u <= pu[-1] + 2 * SURFACE_BIN)
    on = near & (np.abs(z - np.interp(u, pu, pz)) <= SURFACE_STEP)
    out: dict[str, Any] = {
        "lo": float(np.min(u[on])), "hi": float(np.max(u[on])),
        "slope": float(slope), "intercept": float(icpt),
        "rms": float(np.sqrt(np.mean((pz - (slope * pu + icpt)) ** 2))),
        "profile_u": pu, "profile_z": pz,
    }
    # A crown: the profile rises above its chord; report each side's slope.
    chord = np.interp(pu, [pu[0], pu[-1]], [pz[0], pz[-1]])
    top = int(np.argmax(pz - chord))
    out["crown_m"] = max(0.0, float(pz[top] - chord[top]))
    if out["crown_m"] >= CROWN_MIN and 1.0 <= pu[top] - pu[0] and 1.0 <= pu[-1] - pu[top]:
        out["right_slope"] = float(np.polyfit(pu[: top + 1], pz[: top + 1], 1)[0])
        out["left_slope"] = float(np.polyfit(pu[top:], pz[top:], 1)[0])
    return out


def _deck_z(deck: dict[str, Any], u: Any) -> Any:
    return np.interp(u, deck["profile_u"], deck["profile_z"])


def _side(u: np.ndarray, z: np.ndarray, deck: dict[str, Any], sign: int) -> dict[str, Any]:
    """Curb, parapet and outer face beyond one deck end (sign -1 right/low u, +1 left/high u)."""
    edge = deck["hi"] if sign > 0 else deck["lo"]
    deck_z = float(_deck_z(deck, edge))
    beyond = (u - edge) * sign
    near = (beyond > 0.0) & (beyond <= CURB_SEARCH)
    rise = z - deck_z
    out: dict[str, Any] = {"inner_u": float(edge), "deck_z": float(deck_z)}
    cand = near & (rise >= CURB_MIN_RISE) & (rise <= CURB_MAX_RISE)
    if cand.sum() < 10:
        return out
    # The curb top is the most populated height level just above the deck.
    hist, edges = np.histogram(rise[cand], bins=np.arange(CURB_MIN_RISE, CURB_MAX_RISE + 0.01, 0.01))
    peak = int(np.argmax(hist))
    level = 0.5 * (edges[peak] + edges[peak + 1])
    top = cand & (np.abs(rise - level) <= CURB_LEVEL)
    top_z = float(np.median(z[top]))
    top_beyond = np.sort(beyond[top])
    runs = _runs(top_beyond, DECK_GAP)
    a, b = runs[0]
    outer_beyond = float(top_beyond[b - 1])
    if outer_beyond - float(top_beyond[a]) < 0.05:
        return out
    outer_u = float(edge + sign * outer_beyond)
    out.update({"curb_top_z": top_z, "outer_u": outer_u,
                "curb_width_m": outer_beyond, "curb_height_m": top_z - deck_z})
    # Parapet: returns above the curb top over the curb.
    over = (beyond > 0.0) & (beyond <= outer_beyond + 0.05) & (z > top_z + 0.1)
    if over.sum() >= 10:
        out["parapet_height_m"] = float(np.percentile(z[over], 99) - top_z)
    # Outer face: returns hanging below the curb top at the outer edge.
    # Only returns continuing down from the curb top count: an abutment or web
    # seen further below, across a gap, is not part of this face.
    face = np.sort(z[(np.abs(u - outer_u) <= FACE_BAND) & (z <= top_z)])[::-1]
    if len(face) >= 5:
        gaps = np.nonzero(-np.diff(face) > FACE_GAP)[0]
        bottom = face[gaps[0]] if len(gaps) else face[-1]
        if deck_z - bottom > 0.02:  # the face reaches below the deck top
            out["outer_face_depth_m"] = float(deck_z - bottom)
    return out


def _soffit(u: np.ndarray, z: np.ndarray, deck: dict[str, Any]) -> dict[str, Any]:
    lo, hi = deck["lo"], deck["hi"]
    inner = (u > lo + 0.3) & (u < hi - 0.3)
    deck_z = _deck_z(deck, u)
    below = inner & (z < deck_z - SOFFIT_GAP) & (z > deck_z - 1.5)
    bins = np.arange(lo + 0.3, hi - 0.3 + SOFFIT_BIN, SOFFIT_BIN)
    if len(bins) < 2:
        return {"soffit_share": 0.0}
    seen = np.histogram(u[below], bins=bins)[0] > 0
    share = float(seen.mean())
    out: dict[str, Any] = {"soffit_share": share}
    if share >= SOFFIT_MIN_SHARE:
        # The first surface below the deck top: the highest returns in each bin.
        tops = []
        for k in range(len(bins) - 1):
            m = below & (u >= bins[k]) & (u < bins[k + 1])
            # The first surface below: the highest layer of at least 3 returns.
            layers = _layers((deck_z - z)[m], 0.03) if m.sum() >= 3 else []
            layers = [layer for layer in layers if len(layer) >= 3]
            if layers:
                tops.append(float(np.median(layers[0])))
        if len(tops) >= SOFFIT_MIN_SHARE * (len(bins) - 1):
            out["slab_thickness_m"] = float(np.median(tops))
    return out


def measure_section(u: np.ndarray, z: np.ndarray) -> dict[str, Any]:
    """Dimensions of one cross section given lateral offsets `u` and heights `z`."""
    deck = _deck_surface(u, z)
    if deck is None:
        return {"status": "no_deck_surface", "points": int(len(u))}
    left, right = _side(u, z, deck, +1), _side(u, z, deck, -1)
    soffit = _soffit(u, z, deck)
    dims: dict[str, float] = {
        "effective_width_m": deck["hi"] - deck["lo"],
        "cross_slope_percent": deck["slope"] * 100.0,
    }
    dims["crown_height_m"] = deck["crown_m"]
    if "left_slope" in deck:
        dims["left_cross_slope_percent"] = deck["left_slope"] * 100.0
        dims["right_cross_slope_percent"] = deck["right_slope"] * 100.0
    for name, side in (("left", left), ("right", right)):
        for key in ("curb_width_m", "curb_height_m", "parapet_height_m", "outer_face_depth_m"):
            if key in side:
                dims[f"{name}_{key}"] = side[key]
    if "outer_u" in left and "outer_u" in right:
        dims["total_width_m"] = left["outer_u"] - right["outer_u"]
    if "slab_thickness_m" in soffit:
        dims["slab_thickness_m"] = soffit["slab_thickness_m"]
    summary = {k: v for k, v in deck.items() if not k.startswith("profile_")}
    return {"status": "measured", "points": int(len(u)), "dims": dims,
            "deck": summary, "left": left, "right": right, "soffit_share": soffit["soffit_share"],
            "deck_profile": {"u": deck["profile_u"], "z": deck["profile_z"]}}


def cut(xyz: np.ndarray, axis: Axis, station: float, thickness: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points within `thickness/2` of the plane normal to the axis at `station`."""
    rel = xyz[:, :2] - axis.origin
    along = rel @ axis.direction
    sel = np.abs(along - station) <= thickness * 0.5
    idx = np.nonzero(sel)[0]
    return idx, rel[idx] @ axis.lateral, xyz[idx, 2]


def deck_ends(xyz: np.ndarray, axis: Axis, bin_m: float = 0.5) -> dict[str, Any] | None:
    """Start and end lines of the deck across its width, its length and skew.

    In each lateral bin the 1st/99th along-axis percentiles mark the ends; a
    straight line through each set gives the skew against the axis normal.
    """
    rel = xyz[:, :2] - axis.origin
    along, lat = rel @ axis.direction, rel @ axis.lateral
    lo, hi = np.percentile(lat, [2, 98])
    rows = []
    for a in np.arange(lo, hi, bin_m):
        m = (lat >= a) & (lat < a + bin_m)
        if m.sum() >= 20:
            rows.append((a + bin_m / 2, *np.percentile(along[m], [1, 99])))
    if len(rows) < 3:
        return None
    c, start, end = (np.array(v) for v in zip(*rows))

    def line(y: np.ndarray) -> np.ndarray:
        fit = np.polyfit(c, y, 1)
        ok = np.abs(y - np.polyval(fit, c)) <= max(0.1, 2.5 * np.std(y - np.polyval(fit, c)))
        return np.asarray(np.polyfit(c[ok], y[ok], 1))

    slope = 0.5 * (line(start)[0] + line(end)[0])
    # Percentiles within a bin cut into the extent, so the end positions come
    # from the few most extreme points once each is shifted along the skew.
    mid = float(np.median(lat))
    inside = (lat >= lo) & (lat <= hi)
    shifted = along[inside] - slope * (lat[inside] - mid)
    k = min(max(3, len(shifted) // 20000), len(shifted) - 1)
    first = float(np.partition(shifted, k)[k])
    last = float(-np.partition(-shifted, k)[k])
    edge = np.array([lo, hi]) - mid
    return {
        "lateral_range_m": [float(lo), float(hi)],
        "start_line": [float(slope), first - slope * mid],
        "end_line": [float(slope), last - slope * mid],
        "full_width_from_m": float(first + np.max(slope * edge)),
        "full_width_to_m": float(last + np.min(slope * edge)),
        "length_m": last - first,
        "skew_deg": float(np.degrees(np.arctan(slope))),
    }


def bridge_sections(
    path: str,
    out_dir: str | None = None,
    *,
    spacing: float = 1.0,
    thickness: float = 0.1,
    classes: list[int] | None = None,
    axis_classes: list[int] | None = None,
    axis: tuple[list[float], list[float]] | None = None,
    margin: float = 0.3,
) -> dict[str, Any]:
    """Cut sections along a bridge and aggregate a member dimension table.

    `classes` restricts measured points to those classification codes (for
    example structure classes, excluding vegetation). The axis is explicit, or
    the principal XY direction of `axis_classes` (else of the measured points).
    The deck ends come from the same points; sections are cut only where the
    full width lies between them, `margin` inside, so a skewed end never cuts
    a section partway across an abutment.
    """
    if not (0 < spacing <= 50) or not (0 < thickness <= 1):
        raise ValueError("spacing must be in (0, 50] m and thickness in (0, 1] m")
    data = load_points(path)
    xyz = data["xyz"]
    cls = data.get("classification")
    if (classes or axis_classes) and cls is None:
        raise ValueError("classes were given but the file has no classification")
    keep = np.isin(cls, classes) if classes and cls is not None else np.ones(len(xyz), bool)
    pts = xyz[keep]
    if len(pts) < 100:
        raise ValueError("fewer than 100 points to measure")
    ref = xyz[np.isin(cls, axis_classes)] if axis_classes and cls is not None else pts
    ax = axis_from_points(axis[0], axis[1]) if axis is not None else refine_axis(ref, estimate_axis(ref))
    ends = deck_ends(ref, ax)
    if ends is None:
        along = (pts[:, :2] - ax.origin) @ ax.direction
        lo, hi = (float(v) for v in np.percentile(along, [0.5, 99.5]))
    else:
        lo, hi = ends["full_width_from_m"], ends["full_width_to_m"]
    stations = np.arange(lo + margin, hi - margin + 1e-9, spacing)
    if len(stations) == 0:
        raise ValueError("no station lies inside the deck ends; check the axis or classes")
    sections = []
    for s in stations:
        _, u, z = cut(pts, ax, float(s), thickness)
        m = measure_section(u, z)
        m.pop("deck_profile", None)
        m["station_m"] = float(s)
        sections.append(m)
    table = _aggregate(sections)
    if ends is not None:
        table = [
            {"item": "deck_length_m", "label_ja": "床版の橋軸方向長さ（中心線、点群範囲）", "status": "observed",
             "value": ends["length_m"], "p10": None, "p90": None, "sections_observed": None,
             "sections_total": None},
            {"item": "skew_deg", "label_ja": "斜角（床版端線と橋軸直角方向のなす角、度）", "status": "observed",
             "value": ends["skew_deg"], "p10": None, "p90": None, "sections_observed": None,
             "sections_total": None},
        ] + table
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "source": str(Path(path).resolve()),
        "points": int(len(xyz)),
        "measured_points": int(len(pts)),
        "classes": classes,
        "axis": {"origin_xy": ax.origin.tolist(), "direction_xy": ax.direction.tolist(), "source": ax.source},
        "options": {"spacing_m": spacing, "thickness_m": thickness, "margin_m": margin},
        "along_extent_m": [float(lo), float(hi)],
        "deck_ends": ends,
        "sections": sections,
        "table": table,
        "unobserved": [row["item"] for row in table if row["status"] == "unobserved"],
        "note": ("Values are measured from the point cloud only. lower_bound items are visible extents, "
                 "not member sizes; unobserved items had no supporting returns and are not assumed."),
    }
    if out_dir:
        result["outputs"] = _write_outputs(result, pts, ax, Path(out_dir))
    return result


def _aggregate(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    measured = [s for s in sections if s["status"] == "measured"]
    rows = []
    for key, label, status in ITEMS:
        vals = np.array([s["dims"][key] for s in measured if key in s["dims"]], dtype=float)
        row: dict[str, Any] = {"item": key, "label_ja": label, "sections_observed": int(len(vals)),
                               "sections_total": len(sections)}
        if len(vals) == 0:
            # Side slopes exist only on a crowned deck; their absence is not a gap in the data.
            missing = "not_applicable" if key in CROWN_ONLY and measured else "unobserved"
            row.update({"status": missing, "value": None, "p10": None, "p90": None})
        else:
            row.update({"status": status, "value": float(np.median(vals)),
                        "p10": float(np.percentile(vals, 10)), "p90": float(np.percentile(vals, 90))})
        rows.append(row)
    return rows


def _write_outputs(result: dict[str, Any], pts: np.ndarray, ax: Axis, out: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "bridge_sections.json", "csv": out / "dimensions.csv", "svg": out / "section.svg"}
    with open(paths["csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "label_ja", "status", "value_m_or_percent", "p10", "p90",
                    "sections_observed", "sections_total"])
        for r in result["table"]:
            w.writerow([r["item"], r["label_ja"], r["status"],
                        "" if r["value"] is None else f"{r['value']:.3f}",
                        "" if r["p10"] is None else f"{r['p10']:.3f}",
                        "" if r["p90"] is None else f"{r['p90']:.3f}",
                        r["sections_observed"], r["sections_total"]])
    measured = [s for s in result["sections"] if s["status"] == "measured"]
    if measured:
        width = [s["dims"].get("total_width_m", s["dims"]["effective_width_m"]) for s in measured]
        rep = measured[int(np.argsort(width)[len(width) // 2])]
        _, u, z = cut(pts, ax, rep["station_m"], result["options"]["thickness_m"])
        profile = measure_section(u, z).get("deck_profile")
        paths["svg"].write_text(section_svg(u, z, rep, result["table"], profile), encoding="utf-8")
    else:
        paths.pop("svg")
    paths["json"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def section_svg(u: np.ndarray, z: np.ndarray, section: dict[str, Any], table: list[dict[str, Any]],
                profile: dict[str, np.ndarray] | None = None) -> str:
    """A cross-section drawing: points, measured outline and dimension lines (metres)."""
    deck, left, right = section["deck"], section["left"], section["right"]
    lo_u = min(right.get("outer_u", deck["lo"]), deck["lo"]) - 0.5
    hi_u = max(left.get("outer_u", deck["hi"]), deck["hi"]) + 0.5
    deck_z = deck["slope"] * 0.5 * (deck["lo"] + deck["hi"]) + deck["intercept"]
    lo_z, hi_z = deck_z - 1.5, deck_z + 1.6
    keep = (u >= lo_u) & (u <= hi_u) & (z >= lo_z) & (z <= hi_z)
    scale = 1000.0 / (hi_u - lo_u)            # px per metre
    W, H = 1000.0, (hi_z - lo_z) * scale + 120
    # Viewed from the start of the axis toward its end: lateral left (+u) on the left.
    X = lambda v: (hi_u - v) * scale  # noqa: E731
    Y = lambda v: (hi_z - v) * scale + 20  # noqa: E731
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
             f'font-family="sans-serif" font-size="13">',
             '<rect width="100%" height="100%" fill="white"/>']
    step = max(1, int(keep.sum() // 6000))
    # One path of zero-length round-capped segments keeps the file small.
    dots = "".join(f"M{X(a):.1f} {Y(b):.1f}h0" for a, b in zip(u[keep][::step], z[keep][::step]))
    parts.append(f'<path d="{dots}" stroke="#9aa0a6" stroke-width="1.8" stroke-linecap="round"/>')
    surface = list(zip(profile["u"], profile["z"])) if profile is not None else []
    outline = [(right.get("outer_u", deck["lo"]), right.get("curb_top_z", right["deck_z"])),
               (deck["lo"], right.get("curb_top_z", right["deck_z"])),
               (deck["lo"], right["deck_z"]), *surface, (deck["hi"], left["deck_z"]),
               (deck["hi"], left.get("curb_top_z", left["deck_z"])),
               (left.get("outer_u", deck["hi"]), left.get("curb_top_z", left["deck_z"]))]
    pts_s = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in outline)
    parts.append(f'<polyline points="{pts_s}" fill="none" stroke="#e8710a" stroke-width="2.5"/>')

    def dim(a: float, b: float, y: float, text: str) -> None:
        parts.append(f'<line x1="{X(a):.1f}" y1="{Y(y):.1f}" x2="{X(b):.1f}" y2="{Y(y):.1f}" '
                     f'stroke="#1a73e8" stroke-width="1.2" marker-start="url(#t)" marker-end="url(#t)"/>')
        parts.append(f'<text x="{(X(a) + X(b)) / 2:.1f}" y="{Y(y) - 5:.1f}" text-anchor="middle" '
                     f'fill="#1a73e8">{text}</text>')

    def vdim(at: float, z1: float, z2: float, text: str) -> None:
        parts.append(f'<line x1="{X(at):.1f}" y1="{Y(z1):.1f}" x2="{X(at):.1f}" y2="{Y(z2):.1f}" '
                     f'stroke="#1a73e8" stroke-width="1.2"/>')
        parts.append(f'<text x="{X(at) + 6:.1f}" y="{(Y(z1) + Y(z2)) / 2 + 4:.1f}" fill="#1a73e8">{text}</text>')

    parts.insert(2, '<defs><marker id="t" viewBox="0 0 2 10" refX="1" refY="5" markerWidth="2" '
                    'markerHeight="10" orient="auto"><path d="M1 0V10" stroke="#1a73e8"/></marker></defs>')
    d = section["dims"]
    dim(deck["lo"], deck["hi"], deck_z - 0.25, f"有効幅員 {d['effective_width_m'] * 1000:.0f}")
    if "total_width_m" in d:
        dim(right["outer_u"], left["outer_u"], deck_z + 1.35, f"全幅 {d['total_width_m'] * 1000:.0f}")
    for side, name in ((left, "左"), (right, "右")):
        if "curb_width_m" in side:
            a, b = sorted([side["inner_u"], side["outer_u"]])
            dim(a, b, side["curb_top_z"] + 0.12, f"{side['curb_width_m'] * 1000:.0f}")
    for side in (left, right):
        if "curb_height_m" in side:
            mid_u = 0.5 * (side["inner_u"] + side["outer_u"])
            vdim(mid_u, side["deck_z"], side["curb_top_z"], f"{side['curb_height_m'] * 1000:.0f}")
            if "parapet_height_m" in side:
                vdim(side["outer_u"], side["curb_top_z"], side["curb_top_z"] + side["parapet_height_m"],
                     f"{side['parapet_height_m'] * 1000:.0f}")
    if "slab_thickness_m" in d:
        at = deck["lo"] + 0.3 * (deck["hi"] - deck["lo"])   # clear of the width label
        top = float(deck["slope"] * at + deck["intercept"])
        vdim(at, top, top - d["slab_thickness_m"], f"床版厚 {d['slab_thickness_m'] * 1000:.0f}")
    unobserved = [r["label_ja"] for r in table if r["status"] == "unobserved"]
    y = H - 60
    parts.append(f'<text x="12" y="{y:.0f}">station {section["station_m"]:.2f} m · 起点側から終点方向を見た断面 · '
                 f'単位 mm · 点群のみから計測（仮定値なし）</text>')
    if unobserved:
        parts.append(f'<text x="12" y="{y + 22:.0f}" fill="#d93025">未観測: {"、".join(unobserved)}'
                     f'（点群に裏付けがないため値を出していません）</text>')
    parts.append("</svg>")
    return "\n".join(parts)
