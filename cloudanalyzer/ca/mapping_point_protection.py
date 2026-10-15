"""Conservative immutable XY neighborhoods for retained HD source traces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ca import mapping_job as jobs

PROTOCOL: dict[str, Any] = {
    "model": "retained_lane_convex_hulls_with_saved_audit_radius",
    "margin_m": 1e-6,
    "maximum_lanes": 256,
    "maximum_vertices": 65536,
    "maximum_hull_vertices": 4096,
    "all_heights": True,
    "protect_existing_holds_as_well_as_supported_traces": True,
    "independent_accuracy_established": False,
}


def plan(parent: dict[str, Any]) -> dict[str, Any]:
    """Hull includes both boundaries and explicit centers; derived centers lie inside."""
    jobs._verify(parent["files"]["editable_map"])
    jobs._verify(parent["quality_report"])
    ir = json.loads(Path(parent["files"]["editable_map"]["path"]).read_text())
    saved = json.loads(Path(parent["quality_report"]["path"]).read_text())
    audits = [
        saved["editable"],
        saved["reopened_osm"],
        *saved.get("ground_consensus", {}).values(),
    ]
    if len(audits) != 4 or any(not jobs._diagnose_audit(a)["complete"] for a in audits):
        raise ValueError("protected density needs four complete retained HD audits")
    radii = [a["quality"]["ground_radius_m"] for a in audits]
    if any(not np.isfinite(r) or not 0 < r <= 2 for r in radii):
        raise ValueError("invalid retained audit ground radius")
    radius = float(max(radii))
    boundaries = {b["id"]: b["geometry"] for b in ir["boundaries"]}
    if not 1 <= len(ir["lanes"]) <= PROTOCOL["maximum_lanes"]:
        raise ValueError("protected density lane count exceeds its bound")
    hulls = []
    vertices_total = 0
    hull_total = 0
    for lane in ir["lanes"]:
        lines = [
            boundaries[lane[s]["boundary"] if isinstance(lane[s], dict) else lane[s]]
            for s in ("left", "right")
        ]
        if lane.get("centerline") is not None:
            lines.append(lane["centerline"])
        arrays = [np.asarray(line, dtype=float) for line in lines]
        if any(
            a.ndim != 2
            or a.shape[1] != 3
            or len(a) < 2
            or not np.isfinite(a).all()
            or np.abs(a).max() > 1e6
            for a in arrays
        ):
            raise ValueError("protected density needs bounded valid retained curves")
        vertices = np.vstack(arrays)[:, :2]
        vertices_total += len(vertices)
        if vertices_total > PROTOCOL["maximum_vertices"]:
            raise ValueError("protected density curve vertices exceed their bound")
        try:
            hull = ConvexHull(vertices)
        except QhullError as error:
            raise ValueError("protected lane hull is degenerate") from error
        polygon = vertices[hull.vertices]
        hull_total += len(polygon)
        if hull_total > PROTOCOL["maximum_hull_vertices"]:
            raise ValueError("protected density hull vertices exceed their bound")
        hulls.append(
            {
                "lane": lane["id"],
                "vertices_xy": polygon.tolist(),
                "equations": hull.equations.tolist(),
            }
        )
    return {
        "protocol": PROTOCOL,
        "ground_radius_m": radius,
        "effective_halo_m": radius + PROTOCOL["margin_m"],
        "hulls": hulls,
        "source_hd": parent["files"]["editable_map"],
        "source_audits": parent["quality_report"],
    }


def mask(rows: np.ndarray, protection: dict[str, Any]) -> np.ndarray:
    """Preserve a superset of every retained trace's XY source-query disk."""
    if protection["protocol"] != PROTOCOL:
        raise ValueError("point protection protocol changed")
    halo = protection["effective_halo_m"]
    xy = np.column_stack([rows[n] for n in ("x", "y")])
    flags = np.zeros(len(rows), dtype=bool)
    for hull in protection["hulls"]:
        polygon = np.asarray(hull["vertices_xy"])
        equations = np.asarray(hull["equations"])
        lo, hi = polygon.min(axis=0) - halo, polygon.max(axis=0) + halo
        ids = np.flatnonzero(~flags & ((xy >= lo) & (xy <= hi)).all(axis=1))
        # Bound temporary half-plane arrays even for a dense ROI.
        for start in range(0, len(ids), 4096):
            chunk = ids[start : start + 4096]
            points = xy[chunk]
            close = np.all(
                points @ equations[:, :2].T + equations[:, 2] <= 1e-8, axis=1
            )
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                v = b - a
                t = np.clip(((points - a) @ v) / (v @ v), 0, 1)
                close |= np.sum((points - (a + t[:, None] * v)) ** 2, axis=1) <= halo**2
            flags[chunk[close]] = True
    return flags


def verify(protection: dict[str, Any], parent: dict[str, Any]) -> None:
    """Recompute from hashed complete audits/geometry, rather than trusting a mask."""
    if protection != plan(parent):
        raise ValueError(
            "protected density retained geometry or audit protocol changed"
        )
