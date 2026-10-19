"""Inspect an external map draft's saved gaps without extracting a raw recording."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from ca import mapping_job as jobs, mapping_retry as retries
from ca.mapping_geometry import preview_layout_width
from ca.mapping_hd_repair import occupancy

SCHEMA = "cloudanalyzer.mapping_map_gaps.v1"
MAX_REPORT_BYTES = 2 * 1024**2
PROTOCOL = {"gaps_per_page": 8, "profiles_per_gap": 8, "bands_per_profile": 16,
            "width_candidates_per_gap": 8, "sections_per_width_preview": 128,
            "ranges_per_width_preview": 16, "maximum_report_bytes": MAX_REPORT_BYTES,
            "raw_evidence_available": False, "fresh_source_audit_performed": False}


def _gap(interval: dict[str, Any], number: int, proposal: dict[str, Any],
         layout: dict[str, Any], parent: dict[str, Any]) -> dict[str, Any]:
    lo, hi = interval["from_m"], interval["to_m"]
    profiles = [p for p in proposal["profiles"] if lo - 2 <= p["station_m"] <= hi + 2]
    n = min(len(profiles), PROTOCOL["profiles_per_gap"])
    indices = [round(i * (len(profiles) - 1) / (n - 1)) for i in range(n)] if n > 1 else list(range(n))
    observed = [{**{k: p[k] for k in ("station_m", "trajectory", "heading_usable", "reference_ground_height_m")},
                 "bands": p["bands"][:PROTOCOL["bands_per_profile"]],
                 "bands_total": len(p["bands"]), "bands_limited": len(p["bands"]) > PROTOCOL["bands_per_profile"]}
                for p in (profiles[i] for i in indices)]
    candidates = [c for c in proposal["candidates"] if max(lo, c["from_m"]) < min(hi, c["to_m"])]
    widths = []
    for candidate in candidates[:PROTOCOL["width_candidates_per_gap"]]:
        sections = [s for s in candidate["sections"] if lo <= s["station_m"] <= hi]
        preview = {"sections": sections[:PROTOCOL["sections_per_width_preview"]], "total_sections": len(sections)}
        widths.append({"candidate_id": candidate["id"], "candidate_from_m": candidate["from_m"],
                       "candidate_to_m": candidate["to_m"], "layout_width_preview": preview_layout_width(preview, layout)})
    return {"id": number, **interval, "profiles": observed, "profiles_total": len(profiles),
            "profiles_limited": len(profiles) > len(observed), "width_observations": widths,
            "width_candidates_total": len(candidates), "width_candidates_limited": len(candidates) > len(widths),
            "retained_hd_occupancy": occupancy(parent, interval)}


def inspect(root: Path, cid: int, offset: int, layout: dict[str, Any]) -> dict[str, Any]:
    """Return one immutable page of saved extraction evidence and width observations.

    The retained draft's source proposal is used even after a later refinement.
    No point clouds are loaded and no native geometry/audit or raw extraction is
    performed. Saved profile gaps are not proof that the physical road is missing.
    These pages cannot authorize point-cloud retries or HD-repair trials.
    """
    if type(offset) is not int or offset < 0:
        raise ValueError("gap offset must be a nonnegative integer")
    with jobs._locked(root):
        job = jobs._load(root)
        if job["pointcloud"].get("origin") != "external_slam":
            raise ValueError("saved-map gap inspection requires an imported external SLAM map")
        jobs._inputs(job)
        parent = retries._parent(job, cid)
        inputs = {"source": job["source"], "layout": layout, "native": job["runtime"]["native"]["extension"],
                  "proposal": parent["corridor_proposal"], "audits": parent["quality_report"],
                  **{f"pointcloud_{k}": v for k, v in job["pointcloud"]["files"].items()},
                  **{f"parent_{k}": v for k, v in parent["files"].items()},
                  **{f"geometry_{k}": v for k, v in parent["geometry_inputs"].items()}}
        retries._verify_inputs(inputs)
        key = hashlib.sha256(json.dumps({"inputs": inputs, "protocol": PROTOCOL}, sort_keys=True).encode()).hexdigest()
        path = root / f"map-gaps-{cid:02d}-{offset}-{key[:16]}.json"
        if path.exists():
            if path.stat().st_size > MAX_REPORT_BYTES:
                raise ValueError("saved-map gap evidence exceeds its 2 MiB page budget")
            saved = json.loads(path.read_text(encoding="utf-8"))
            if (saved["inputs"] != inputs or saved["inspection_protocol"] != PROTOCOL
                    or saved.get("schema") != SCHEMA or saved.get("candidate_id") != cid or saved.get("offset") != offset
                    or saved.get("raw_evidence_available") is not False or saved.get("fresh_source_audit_performed") is not False):
                raise ValueError("saved-map gap evidence inputs changed")
        else:
            proposal = json.loads(Path(parent["corridor_proposal"]["path"]).read_text(encoding="utf-8"))
            report = json.loads(Path(parent["files"]["report"]["path"]).read_text(encoding="utf-8"))
            fixed_layout = json.loads(Path(layout["path"]).read_text(encoding="utf-8"))
            intervals = [i for i in report["station_disposition"] if i["status"] != "included_lane_hypothesis"]
            page = [_gap(interval, offset + i + 1, proposal, fixed_layout, parent)
                    for i, interval in enumerate(intervals[offset:offset + PROTOCOL["gaps_per_page"]])]
            saved = {"schema": SCHEMA, "candidate_id": cid, "inputs": inputs, "offset": offset,
                     "gaps_total": len(intervals), "gaps": page,
                     "next_offset": offset + len(page) if offset + len(page) < len(intervals) else None,
                     "source_extent": parent["extent"], "source_protocol": proposal["protocol"],
                     "inspection_protocol": PROTOCOL, "raw_evidence_available": False,
                     "fresh_source_audit_performed": False, "proposal_role": "retained_candidate_source_proposal",
                     "note": "Saved extraction profiles and width observations only. Read ambiguity, missing anchors, "
                             "path containment and uninspected/truncated evidence. These are not proven physical causes, "
                             "fresh ground audits, raw-log evidence or permission to retry the point map/repair HD."}
            size = len((json.dumps(saved, indent=2, allow_nan=False) + "\n").encode("utf-8"))
            if size > MAX_REPORT_BYTES:
                raise ValueError("saved-map gap evidence exceeds its 2 MiB page budget")
            if shutil.disk_usage(root).free < 10 * 1024**3 + 2 * size:
                raise ValueError("insufficient disk space: gap inspection must retain 10 GiB free")
            jobs._inputs(job)
            retries._verify_inputs(inputs)
            jobs._save(path, saved)
        retries._verify_inputs(saved["inputs"])
        return {"file": jobs._artifact(path), **{k: v for k, v in saved.items() if k != "inputs"}}
