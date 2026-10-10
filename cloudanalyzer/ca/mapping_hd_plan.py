"""Inspect observed gap curves before spending the shared HD generation budget."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ca import mapping_job as jobs, mapping_retry as retries
from ca.mapping_hd_repair import occupancy, validate
from ca.mapping_patch import _intervals, _read

PROTOCOL = {
    "maximum_intervals": 256,
    "intervals_per_page": 8,
    "maximum_source_samples_per_page_per_estimator": 100000,
    "endpoint_tolerance_m": 0.01,
    "observed_station_endpoints_only": True,
    "both_ground_estimators_required": True,
    "all_reference_trace_samples_required": True,
    "hd_generation_attempts_spent": 0,
    "lane_export_performed": False,
    "final_four_audit_patch_required": True,
}


def _curves(sections: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    left, right = [
        np.asarray([s[side] for s in sections], dtype=float)
        for side in ("left", "right")
    ]
    if any(
        a.shape != (2, 3) or not np.isfinite(a).all() or (np.abs(a) > 1e15).any()
        for a in (left, right)
    ):
        raise ValueError("plan intervals need two finite observed boundary sections")
    return left, right


def _index(
    parent: dict[str, Any],
    proposal: dict[str, Any],
    gaps: list[dict[str, Any]],
    layout: dict[str, Any],
) -> list[dict[str, Any]]:
    lanes = layout["lanes"]
    if (
        len(lanes) != 1
        or lanes[0]["direction"] != "forward"
        or not lanes[0]["one_way"]
        or lanes[0]["fraction"] != 1
    ):
        raise ValueError("HD gap planning currently requires one forward one-way lane")
    ir = _read(parent["files"]["editable_map"])
    boundaries = {b["id"]: b["geometry"] for b in ir["boundaries"]}
    intervals = _intervals(parent)

    def curve(lane: dict[str, Any], side: str) -> np.ndarray:
        ref = lane[side]
        points = np.asarray(boundaries[ref if type(ref) is int else ref["boundary"]])
        return (
            points[::-1]
            if isinstance(ref, dict) and ref.get("reversed", False)
            else points
        )

    offered = []
    for gap in gaps:
        occupied = occupancy(parent, gap)
        if not occupied["available"] or occupied["limited_in_response"]:
            raise ValueError("gap planning requires complete retained HD occupancy")
        for source in proposal["candidates"]:
            sections = source["sections"]
            for a, b in zip(sections, sections[1:]):
                lo, hi = a["station_m"], b["station_m"]
                if not any(
                    i["from_m"] <= lo < hi <= i["to_m"]
                    for i in occupied["unoccupied_intervals"]
                ):
                    continue
                left, right = _curves([a, b])
                widths = np.linalg.norm(left[:, :2] - right[:, :2], axis=1)
                if widths.min() + 1e-6 < lanes[0]["minimum_width_m"]:
                    continue
                links = []
                for old in ir["lanes"]:
                    start, end = intervals[old["id"]]
                    for direction, station, index, oldindex in [
                        ("from_retained", end, 0, -1),
                        ("to_retained", start, -1, 0),
                    ]:
                        if abs(station - (lo if index == 0 else hi)) > 1e-6:
                            continue
                        distances = [
                            float(np.linalg.norm(c[index] - curve(old, side)[oldindex]))
                            for c, side in [(left, "left"), (right, "right")]
                        ]
                        if max(distances) <= PROTOCOL["endpoint_tolerance_m"]:
                            links.append(
                                {
                                    "direction": direction,
                                    "retained_lane_id": old["id"],
                                    "boundary_endpoint_distances_m": distances,
                                }
                            )
                # This bound covers native endpoint-inclusive sampling for the derived
                # centre and both observed sides before allocating a native audit.
                lengths = [float(np.linalg.norm(c[1] - c[0])) for c in (left, right)]
                bound = sum(math.ceil(n / 0.5) + 1 for n in [*lengths, sum(lengths)])
                offered.append(
                    {
                        "candidate_id": source["id"],
                        "from_m": lo,
                        "to_m": hi,
                        "gap_id": gap["id"],
                        "observed_sections": [a, b],
                        "minimum_observed_span_m": float(widths.min()),
                        "retained_endpoint_links": links,
                        "source_sample_upper_bound": bound,
                        "source_ambiguous": any(
                            max(lo, i["from_m"]) < min(hi, i["to_m"])
                            for i in proposal["ambiguous_intervals"]
                        ),
                    }
                )
                if len(offered) > PROTOCOL["maximum_intervals"]:
                    raise ValueError(
                        "plan exceeds 256 intervals; choose fewer inspected gaps"
                    )
    offered.sort(
        key=lambda r: (
            -len(r["retained_endpoint_links"]),
            r["from_m"],
            r["candidate_id"],
        )
    )
    return [{"plan_interval_id": i + 1, **row} for i, row in enumerate(offered)]


def _inspection_ir(row: dict[str, Any], layout: dict[str, Any]) -> dict[str, Any]:
    """Transient reference-trace carrier; no lane building, export or adoption."""
    return {
        "format": "vectormap-ir",
        "version": 1,
        "metadata": {
            "name": "Read-only observed gap trace inspection",
            "attributes": {
                "cloudanalyzer:draft": "reference_trace_inspection",
                "cloudanalyzer:deployment_ready": "false",
            },
        },
        "boundaries": [
            {
                "id": i + 1,
                "kind": {"type": "other"},
                "geometry": [s[side] for s in row["observed_sections"]],
            }
            for i, side in enumerate(("left", "right"))
        ],
        "lanes": [
            {
                "id": 3,
                "kind": "driving",
                "left": 1,
                "right": 2,
                "one_way": True,
                "speed_limit": {"kmh": layout["speed_limit_kmh"]},
            }
        ],
        "roads": [],
        "topology": [],
    }


def inspect(
    root: Path,
    cid: int,
    evidence: dict[str, Any],
    gap_ids: list[int],
    layout_file: dict[str, Any],
    offset: int,
) -> dict[str, Any]:
    if type(offset) is not int or offset < 0:
        raise ValueError("plan offset must be a nonnegative integer")
    with jobs._locked(root):
        job = jobs._load(root)
        jobs._inputs(job)
        parent = retries._parent(job, cid)
        saved = validate(evidence, cid, gap_ids)
        jobs._verify(layout_file)
        inputs = {
            **retries._inputs(job, parent, layout_file),
            "gap_evidence": evidence,
            "raw_manifest": saved["raw_manifest"],
        }
        if saved["inputs"] != retries._inputs(job, parent, layout_file):
            raise ValueError("gap plan inputs differ from retained evidence")
        layout = _read(layout_file)
        proposal = _read(parent["corridor_proposal"])
        key = hashlib.sha256(
            json.dumps(
                {"inputs": inputs, "gap_ids": gap_ids, "protocol": PROTOCOL},
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        directory = root / f"hd-plan-{cid:02d}-{key}"
        directory.mkdir(exist_ok=True)
        index_path = directory / "index.json"
        if not index_path.exists():
            rows = _index(
                parent,
                proposal,
                [g for g in saved["gaps"] if g["id"] in gap_ids],
                layout,
            )
            jobs._save(
                index_path,
                {
                    "inputs": inputs,
                    "protocol": PROTOCOL,
                    "gap_ids": gap_ids,
                    "intervals": rows,
                },
            )
        index = _read(jobs._artifact(index_path))
        if (
            index["inputs"] != inputs
            or index["protocol"] != PROTOCOL
            or index["gap_ids"] != gap_ids
        ):
            raise ValueError("retained gap plan index changed")
        retries._verify_inputs(inputs)
        page_path = directory / f"page-{offset:04d}.json"
        if page_path.exists():
            page = _read(jobs._artifact(page_path))
            if page["index"] != jobs._artifact(index_path) or page["offset"] != offset:
                raise ValueError("retained gap plan page changed")
            for row in page["intervals"]:
                jobs._verify(row["source_audits_file"])
        else:
            rows = index["intervals"][offset : offset + PROTOCOL["intervals_per_page"]]
            if (
                sum(r["source_sample_upper_bound"] for r in rows)
                > PROTOCOL["maximum_source_samples_per_page_per_estimator"]
            ):
                raise ValueError("gap plan page exceeds source sampling budget")
            result = []
            native = jobs.core()
            assert native is not None
            for row in rows:
                audit_path = directory / f"source-{row['plan_interval_id']:04d}.json"
                if audit_path.exists():
                    saved_audits = _read(jobs._artifact(audit_path))
                    if (
                        saved_audits["inputs"] != inputs
                        or saved_audits["interval"] != row
                    ):
                        raise ValueError("retained reference source inspection changed")
                    audits = saved_audits["audits"]
                else:
                    with tempfile.TemporaryDirectory(
                        prefix=".source-curves-", dir=directory
                    ) as temp:
                        carrier = Path(temp) / "reference.json"
                        jobs._save(carrier, _inspection_ir(row, layout))
                        cloud = job["pointcloud"]["files"]["map"]["path"]
                        audits = {
                            "quantile": json.loads(
                                native.audit_vector_map_quality_details(
                                    cloud, str(carrier)
                                )
                            ),
                            "consensus": json.loads(
                                native.audit_vector_map_ground_consensus_details(
                                    cloud, str(carrier)
                                )
                            ),
                        }
                    jobs._inputs(job)
                    retries._verify_inputs(inputs)
                    jobs._save(
                        audit_path,
                        {"inputs": inputs, "interval": row, "audits": audits},
                    )
                diagnoses = {
                    name: jobs._diagnose_audit(a) for name, a in audits.items()
                }
                complete = all(d["complete"] for d in diagnoses.values())
                passed = complete and all(
                    len(a["quality"]["lanes"]) == 1
                    and all(
                        a["quality"]["lanes"][0][side]["samples"] > 0
                        and a["quality"]["lanes"][0][side]["samples"]
                        == a["quality"]["lanes"][0][side]["supported"]
                        and a["quality"]["lanes"][0][side]["start_supported"]
                        and a["quality"]["lanes"][0][side]["end_supported"]
                        for side in ("center", "left", "right")
                    )
                    for a in audits.values()
                )
                response = {
                    k: v
                    for k, v in row.items()
                    if k not in ("observed_sections", "source_sample_upper_bound")
                }
                response.update(
                    source_audits_file=jobs._artifact(audit_path),
                    reference_traces_fully_supported=passed,
                    reference_audits={
                        k: {
                            "complete": d["complete"],
                            "sample_totals": d["sample_totals"],
                            "protocol": d["protocol"],
                        }
                        for k, d in diagnoses.items()
                    },
                )
                result.append(response)
            page = {
                "schema": "cloudanalyzer.hd_gap_plan.v1",
                "index": jobs._artifact(index_path),
                "offset": offset,
                "inputs": inputs,
                "candidate_id": cid,
                "gap_ids": gap_ids,
                "protocol": PROTOCOL,
                "intervals": result,
                "intervals_total": len(index["intervals"]),
                "next_offset": (
                    offset + 8 if offset + 8 < len(index["intervals"]) else None
                ),
                "note": "Read-only reference-curve source checks; no lane builder, OSM exporter, point processing or HD attempt. Support does not prove road identity, full width or a final patch pass. Inspect the frozen child proposal, explicitly draft chosen observed ranges, then require all four exported/reopened audits and retention checks before adoption. Endpoint links are geometry hints requiring patch inspection.",
            }
            jobs._inputs(job)
            retries._verify_inputs(inputs)
            jobs._save(page_path, page)
        return {
            "file": jobs._artifact(page_path),
            "index_file": page["index"],
            **{k: v for k, v in page.items() if k not in ("inputs", "index", "offset")},
        }


def verify_observation(observation: dict[str, Any]) -> None:
    jobs._verify(observation["file"])
    saved = _read(observation["file"])
    jobs._verify(saved["index"])
    retries._verify_inputs(saved["inputs"])
    for row in saved["intervals"]:
        jobs._verify(row["source_audits_file"])
