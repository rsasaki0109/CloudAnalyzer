"""Bounded point-map observations at explicitly inspected gap profiles."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from ca import mapping_job as jobs, mapping_retry as retries
from ca.mapping_map_gaps import SCHEMA as GAP_SCHEMA

SCHEMA = "cloudanalyzer.mapping_map_points.v1"
MAX_REPORT_BYTES = 2 * 1024**2
PROTOCOL: dict[str, Any] = {"radius_m": .75, "boundary": "xy_distance_inclusive", "all_heights": True,
            "maximum_profiles": 8, "maximum_points_per_profile": 128,
            "maximum_map_points": 2_000_000, "maximum_map_bytes": 128 * 1024**2,
            "xy_cell_m": .2, "preview_selection": "evenly_spaced_in_source_record_order",
            "maximum_report_bytes": MAX_REPORT_BYTES,
            "raw_evidence_available": False, "fresh_source_audit_performed": False}


def validate(root: Path, cid: int, evidence: dict[str, Any], gap_ids: Any,
             stations: Any) -> dict[str, Any]:
    """Bind the requested stations to the selected, previously seen gap page."""
    job = jobs._load(root)
    if job["pointcloud"].get("origin") != "external_slam":
        raise ValueError("map-point inspection requires an imported external SLAM map")
    jobs._verify(evidence)
    saved = json.loads(Path(evidence["path"]).read_text(encoding="utf-8"))
    retries._verify_inputs(saved["inputs"])
    parent = retries._parent(job, cid)
    if (saved.get("schema") != GAP_SCHEMA or saved.get("candidate_id") != cid
            or saved["inputs"]["proposal"] != parent["corridor_proposal"]
            or saved["inputs"]["pointcloud_map"] != job["pointcloud"]["files"]["map"]):
        raise ValueError("map-point inspection needs this draft's saved-map gap evidence")
    if (not isinstance(gap_ids, list) or not 1 <= len(gap_ids) <= 8
            or any(type(i) is not int for i in gap_ids) or len(set(gap_ids)) != len(gap_ids)
            or not set(gap_ids) <= {g["id"] for g in saved["gaps"]}):
        raise ValueError("choose 1..8 distinct inspected gap IDs from the latest page")
    if (not isinstance(stations, list) or not 1 <= len(stations) <= PROTOCOL["maximum_profiles"]
            or any(type(s) not in (int, float) or abs(s) > 1e15 or not math.isfinite(s) for s in stations)
            or len(set(stations)) != len(stations)):
        raise ValueError("stations_m needs 1..8 distinct finite inspected profile stations")
    profiles = {p["station_m"]: p for g in saved["gaps"] if g["id"] in gap_ids for p in g["profiles"]}
    if not set(stations) <= profiles.keys():
        raise ValueError("inspect every requested profile station in the selected gaps first")
    pointcloud = job["pointcloud"]
    if (type(pointcloud["map_points"]) is not int
            or not 1 <= pointcloud["map_points"] <= PROTOCOL["maximum_map_points"]
            or pointcloud["files"]["map"]["bytes"] > PROTOCOL["maximum_map_bytes"]):
        raise ValueError("map-point inspection exceeds its map point/byte budget")
    return {"inputs": {**saved["inputs"], "gap_evidence": evidence},
            "profiles": [profiles[s] for s in stations], "map_points": pointcloud["map_points"]}


def inspect(root: Path, cid: int, evidence: dict[str, Any], gap_ids: list[int],
            stations: list[float]) -> dict[str, Any]:
    """Read the fixed point map, preserving all existing maps, poses and budgets.

    Nearby returns at all heights and their occupied XY cells are observations,
    not a new ground estimate, source audit or permission to fill the gap.
    """
    with jobs._locked(root):
        job = jobs._load(root)
        jobs._inputs(job)
        selected = validate(root, cid, evidence, gap_ids, stations)
        inputs = selected["inputs"]
        request = {"candidate_id": cid, "gap_ids": gap_ids, "stations_m": stations, "map_points": selected["map_points"],
                   "inputs": inputs, "inspection_protocol": PROTOCOL}
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        path = root / f"map-points-{cid:02d}-{key[:16]}.json"
        if path.exists():
            if path.stat().st_size > MAX_REPORT_BYTES:
                raise ValueError("map-point evidence exceeds its 2 MiB page budget")
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("schema") != SCHEMA or any(saved.get(k) != v for k, v in request.items()):
                raise ValueError("map-point evidence inputs changed")
        else:
            if shutil.disk_usage(root).free < 10 * 1024**3 + 2 * MAX_REPORT_BYTES:
                raise ValueError("insufficient disk space: map-point inspection must retain 10 GiB free")
            module = jobs.core()
            assert module is not None
            xyz = np.asarray(module.read(inputs["pointcloud_map"]["path"])["positions"])
            if (xyz.shape != (selected["map_points"], 3) or not np.isfinite(xyz).all()
                    or np.any(np.abs(xyz) > 1e15)):
                raise ValueError("point reader must preserve the imported map's finite XYZ records")
            observations = []
            for profile in selected["profiles"]:
                center = np.asarray(profile["trajectory"])
                ids = np.flatnonzero(np.sum((xyz[:, :2] - center[:2]) ** 2, axis=1) <= PROTOCOL["radius_m"] ** 2)
                nearby = xyz[ids]
                total = len(ids)
                n = min(total, PROTOCOL["maximum_points_per_profile"])
                chosen = np.linspace(0, total - 1, n, dtype=int) if n else np.empty(0, dtype=int)
                observations.append({"station_m": profile["station_m"], "trajectory": profile["trajectory"],
                    "saved_reference_ground_height_m": profile["reference_ground_height_m"],
                    "points_total": total, "points_limited": total > n,
                    "occupied_xy_cells": len(np.unique(np.floor(nearby[:, :2] / PROTOCOL["xy_cell_m"]), axis=0)),
                    "height_range_m": [float(nearby[:, 2].min()), float(nearby[:, 2].max())] if total else None,
                    "points": [{"source_record_index": int(ids[i]), "xyz": nearby[i].tolist()} for i in chosen]})
            saved = {"schema": SCHEMA, **request, "profiles": observations,
                     "map_points": selected["map_points"], "raw_evidence_available": False,
                     "fresh_source_audit_performed": False,
                     "note": "All-height point-map neighborhoods at saved gap profiles. Counts and previews "
                             "do not establish a ground level, missing physical road, source support, accuracy "
                             "or permission to regenerate the point map or repair HD geometry."}
            if len((json.dumps(saved, indent=2, allow_nan=False) + "\n").encode("utf-8")) > MAX_REPORT_BYTES:
                raise ValueError("map-point evidence exceeds its 2 MiB page budget")
            jobs._inputs(job)
            retries._verify_inputs(inputs)
            jobs._save(path, saved)
        retries._verify_inputs(inputs)
        return {"file": jobs._artifact(path), **{k: v for k, v in saved.items() if k != "inputs"}}
