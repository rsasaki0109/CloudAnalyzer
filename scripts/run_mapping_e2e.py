#!/usr/bin/env python3
"""Generate both maps from a bundled recording with no operator, end to end.

For each case this starts a mapping run from the raw recording and a fixed
layout hypothesis, then drives the run with a deterministic *reference client*
through the same actions a calling agent uses (refine, inspect, draft, finish):

0. refine once with ``association=trajectory_containing``, the run's own remedy
   for off-path bands that fragment the route (``--association
   all_supported_bands`` keeps the original proposal instead);
1. inspect every offered corridor candidate;
2. include each candidate interval that the run's own previews accept:
   ``layout_width_preview`` interpolated ranges that meet the layout width and
   contain the recorded path at every section, whose native
   ``layout_direction_preview`` passed, and whose sections all have a supported
   path level; overlapping intervals are resolved longest first;
3. defer every other candidate, draft once and finish with the audited draft.

The reference client is a reproducible test driver, not an adoption policy:
CloudAnalyzer itself still ranks nothing and the outputs remain drafts that need
review. The script writes ``summary.json`` with the inputs, timings, peak
memory, disk use, decisions, outputs, quality evidence and invariant checks,
plus a ``mapping-scorecard`` per case, and exits 1 when a check fails.
It also splits every unresolved station metre by reason: the run's own source
reasons, and for intervals the client deferred, the furthest preview check any
available candidate reached (see ``REJECTIONS``), and records the draft's
review holds: held traces and every sample where source ground lies below a
trace, with its location.

    python scripts/run_mapping_e2e.py --out /tmp/mapping-e2e            # april and june
    python scripts/run_mapping_e2e.py --case april --out /tmp/mapping-e2e
    python scripts/run_mapping_e2e.py --association all_supported_bands --out /tmp/mapping-e2e-unrefined

Requires the native core (``pip install ./rust/crates/ca-py``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "cloudanalyzer"))

CASES = {
    "april": {
        "recording": "web/public/samples/nclt-2012-04-29.mcap",
        "layout": "benchmarks/vector-map/nclt-agent-run/april/layout.json",
    },
    "june": {
        "recording": "web/public/samples/nclt-2012-06-15.mcap",
        "layout": "benchmarks/vector-map/nclt-agent-run/june/layout.json",
    },
}
INSPECT_BATCH = 8
ASSOCIATIONS = ("trajectory_containing", "all_supported_bands")
INCLUDE_REASON = (
    "Reference client: the run's width preview meets the layout width with the recorded path "
    "inside every section, the native direction preview passed and every section has a supported "
    "path level. This is an unverified lane hypothesis, not road identity or complete width."
)
DEFER_REASON = (
    "Reference client: no non-overlapping interval of this candidate passed the width, path, "
    "direction and surface-level previews. This is not evidence that a road is absent."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _peak_rss_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1 << 20) if sys.platform == "darwin" else peak / 1024, 1)


def _tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())


# Why the reference client rejects a width-preview interval, in check order.
REJECTIONS = (
    "narrower_than_layout",       # no interpolated range meets the layout width
    "path_outside_profile",       # the recorded path leaves a section of the range
    "direction_unresolved",       # the native direction preview did not pass
    "sections_not_previewed",     # the range extends past a truncated section preview
    "surface_level_unsupported",  # a section has no supported path level
)
LOST_TO_OVERLAP = "overlapping_interval_chosen"  # accepted, but a longer overlap won


def rejection(observation: dict[str, Any], interval: dict[str, Any]) -> str | None:
    """The first check that rejects ``interval`` of ``observation``, or None if accepted."""
    start, stop = interval["from_m"], interval["to_m"]
    if stop <= start:
        return "narrower_than_layout"
    if not interval.get("path_contained_at_all_sections"):
        return "path_outside_profile"
    passed = {
        (round(r["from_m"], 6), round(r["to_m"], 6))
        for r in observation["layout_direction_preview"].get("ranges", [])
        if r.get("direction_checks_passed") and not r.get("issues_limited")
    }
    if (round(start, 6), round(stop, 6)) not in passed:
        return "direction_unresolved"
    sections = observation["sections"]
    last_station = max((s["station_m"] for s in sections), default=float("-inf"))
    if observation.get("section_preview_limited") and last_station < stop:
        return "sections_not_previewed"
    covered = [s for s in sections if start <= s["station_m"] <= stop]
    if not covered or not all(s["path_level_supported"] for s in covered):
        return "surface_level_unsupported"
    return None


def accepted_intervals(observation: dict[str, Any]) -> list[tuple[float, float]]:
    """Intervals of one inspected candidate that every run preview accepts."""
    return [(r["from_m"], r["to_m"])
            for r in observation["layout_width_preview"].get("interpolated_ranges", [])
            if rejection(observation, r) is None]


def disposition_breakdown(disposition: list[dict[str, Any]],
                          observations: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Station metres by outcome, with a reason for every unresolved metre.

    ``source_deferred`` keeps the run's own source reasons. For intervals the
    reference client deferred although candidates existed, the reason is the
    furthest check any available candidate's covering width range reached.
    """
    order = REJECTIONS + (LOST_TO_OVERLAP,)
    included = 0.0
    source: dict[str, float] = {}
    client: dict[str, float] = {}
    other: dict[str, float] = {}
    for row in disposition:
        start, stop = row["from_m"], row["to_m"]
        length = stop - start
        if row["status"] == "included_geometry":
            included += length
        elif row["status"] == "source_deferred":
            key = "+".join(sorted(row.get("source_reasons") or [])) or "unspecified"
            source[key] = source.get(key, 0.0) + length
        elif row["status"] == "agent_deferred":
            best = -1
            for cid in row.get("available_candidate_ids") or []:
                observation = observations.get(cid)
                if observation is None:
                    continue
                best = max(best, 0)
                for interval in observation["layout_width_preview"].get("interpolated_ranges", []):
                    if interval["from_m"] <= start + 1e-6 and interval["to_m"] >= stop - 1e-6:
                        why = rejection(observation, interval)
                        best = max(best, len(order) - 1 if why is None else order.index(why))
            key = order[best] if best >= 0 else "no_inspected_candidate"
            client[key] = client.get(key, 0.0) + length
        else:
            other[row["status"]] = other.get(row["status"], 0.0) + length

    def rounded(values: dict[str, float]) -> dict[str, float]:
        return {k: round(v, 3) for k, v in sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))}

    return {"included_m": round(included, 3), "source_deferred_m": rounded(source),
            "client_deferred_m": rounded(client), "other_m": rounded(other)}


def choose(observations: dict[int, dict[str, Any]]) -> list[tuple[int, float, float]]:
    """Accepted intervals without overlap, longest first, deterministic ties."""
    pieces = sorted(
        ((stop - start, cid, start, stop) for cid, obs in observations.items()
         for start, stop in accepted_intervals(obs)),
        key=lambda p: (-p[0], p[1], p[2]),
    )
    chosen: list[tuple[int, float, float]] = []
    for _, cid, start, stop in pieces:
        if all(stop <= a or start >= b for _, a, b in chosen):
            chosen.append((cid, start, stop))
    return sorted(chosen, key=lambda c: c[1])


def review_hold_summary(diagnosis: dict[str, Any]) -> dict[str, Any] | None:
    """Held traces and below-trace samples from an audited draft's review holds."""
    holds = diagnosis.get("review_holds")
    if not holds:
        return None
    traces = holds.get("traces", [])
    held = [
        {"lane": t["lane"], "trace": t["trace"],
         "low_quantile_fraction": round(t["low_quantile"]["fraction"], 4),
         "ground_consensus_fraction": round(t["ground_consensus"]["fraction"], 4),
         "ground_consensus_mismatches": t["ground_consensus"]["height_mismatches"]}
        for t in traces if t.get("status") == "held"
    ]
    samples = [
        {"estimator": s["estimator"], "lane": s["lane"], "trace": s["trace"],
         "lane_station_m": round(s["lane_station_m"], 3), "xyz": [round(v, 3) for v in s["xyz"]],
         "below_trace_m": round(s["below_trace_m"], 3)}
        for s in holds.get("below_trace_samples", [])
    ]
    return {
        "traces_total": holds.get("traces_total", len(traces)),
        "held_traces_total": len(held),
        "unexplained_traces_total": holds.get("unexplained_traces_total"),
        "ground_estimators_disagree": holds.get("ground_estimators_disagree"),
        "held": held,
        "below_trace_samples_total": holds.get("below_trace_samples_total", len(samples)),
        "below_trace_samples": samples,
    }


def run_case(name: str, recording: Path, layout_path: Path, out: Path,
             association: str = "trajectory_containing") -> dict[str, Any]:
    from ca import mapping_run as mr
    from ca.mapping_scorecard import format_text, scorecard

    job = out / name / "job"
    if job.exists():
        raise SystemExit(f"{job} already exists; choose a new --out directory")
    job.parent.mkdir(parents=True, exist_ok=True)
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    timings: dict[str, float] = {}
    started = time.monotonic()

    tick = time.monotonic()
    state = mr.start_mapping_run(str(recording), str(job), layout)
    timings["start_s"] = time.monotonic() - tick
    if state["status"] != "needs_agent":
        raise RuntimeError(f"{name}: mapping run did not reach the agent loop: {state['status']}")
    if association == "trajectory_containing":
        tick = time.monotonic()
        revision = mr.inspect_mapping_run(str(job))["revision"]
        refined = mr.advance_mapping_run(
            str(job), {"type": "refine", "association": "trajectory_containing"},
            "Reference client re-extracts trajectory-containing bands once before inspecting.", revision)
        timings["refine_s"] = time.monotonic() - tick
        if refined["status"] != "needs_agent":
            raise RuntimeError(f"{name}: refinement did not return to the agent loop: {refined['status']}")

    tick = time.monotonic()
    offered: list[int] = []
    offset: int | None = 0
    while offset is not None:
        page = mr.inspect_mapping_run(str(job), offset)["candidate_index"]
        offered += [c["id"] for c in page["candidate_index"]]
        offset = page.get("next_offset")
    observations: dict[int, dict[str, Any]] = {}
    for i in range(0, len(offered), INSPECT_BATCH):
        revision = mr.inspect_mapping_run(str(job))["revision"]
        result = mr.advance_mapping_run(
            str(job), {"type": "inspect", "candidate_ids": offered[i:i + INSPECT_BATCH]},
            "Reference client inspects every offered candidate before deciding.", revision)
        observations.update({o["id"]: o for o in result["observation"]})
    timings["inspect_s"] = time.monotonic() - tick

    chosen = choose(observations)
    included_ids = {cid for cid, _, _ in chosen}
    decisions = [
        {"candidate_id": cid, "action": "include", "from_m": start, "to_m": stop,
         "endpoint_policy": "linear_interpolation", "reason": INCLUDE_REASON}
        for cid, start, stop in chosen
    ] + [{"candidate_id": cid, "action": "defer", "reason": DEFER_REASON}
         for cid in offered if cid not in included_ids]

    tick = time.monotonic()
    revision = mr.inspect_mapping_run(str(job))["revision"]
    drafted = mr.advance_mapping_run(
        str(job), {"type": "draft", "decisions": decisions},
        "Reference client drafts every accepted non-overlapping interval.", revision)
    timings["draft_s"] = time.monotonic() - tick
    draft = drafted.get("draft_result") or {}
    diagnosis = draft.get("diagnosis") or {}

    tick = time.monotonic()
    revision = mr.inspect_mapping_run(str(job))["revision"]
    finished = mr.advance_mapping_run(
        str(job), {"type": "finish", "candidate_id": diagnosis.get("candidate_id")},
        "Reference client finishes with the audited draft, or with none.", revision)
    timings["finish_s"] = time.monotonic() - tick
    timings["total_s"] = time.monotonic() - started
    output = finished.get("output") or {}
    artifacts = output.get("artifacts") or {}

    geometry_reports = sorted(job.glob("geometry-*/report.json"))
    breakdown = None
    if geometry_reports:
        geometry = json.loads(geometry_reports[-1].read_text(encoding="utf-8"))
        breakdown = disposition_breakdown(geometry.get("station_disposition", []), observations)

    card = scorecard(str(job))
    (out / name / "scorecard.json").write_text(json.dumps(card, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / name / "scorecard.txt").write_text(format_text(card), encoding="utf-8")

    def produced(key: str) -> bool:
        entry = artifacts.get(key)
        if not isinstance(entry, dict):
            return False
        path = Path(entry["path"])
        return path.is_file() and _sha256(path) == entry["sha256"]

    checks = {
        "every_offered_candidate_inspected": bool(offered) and set(observations) == set(offered),
        "at_least_one_interval_included": bool(chosen),
        "draft_audited": draft.get("status") == "audited_draft",
        "finished_with_both_maps": output.get("status") == "draft_needs_review",
        "point_map_and_trajectory_written": produced("map") and produced("trajectory"),
        "lanelet2_map_written": produced("hd_map"),
        "editable_and_reopened_audits_match": diagnosis.get("editable_and_reopened_match") is True,
        "not_marked_deployment_ready": finished.get("deployment_ready") is False,
    }
    extent = diagnosis.get("extent") or {}
    return {
        "case": name,
        "association": association,
        "inputs": {"recording": _file(recording), "layout": _file(layout_path)},
        "pointcloud": {k: state["pointcloud"].get(k) for k in ("options", "status") if k in state["pointcloud"]},
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "disk_bytes": _tree_bytes(out / name),
        "reference_client": {
            "human_interventions": 0,
            "offered_candidates": len(offered),
            "inspected_candidates": len(observations),
            "inspect_actions": -(-len(offered) // INSPECT_BATCH),
            "included_intervals": len(chosen),
            "included_station_m": round(sum(b - a for _, a, b in chosen), 3),
            "deferred_candidates": len(offered) - len(included_ids),
        },
        "draft": {"status": draft.get("status"), "error": draft.get("error"), "stage": draft.get("stage")},
        "output_status": output.get("status"),
        "artifacts": {k: {"sha256": v.get("sha256"), "bytes": v.get("bytes")}
                      for k, v in artifacts.items() if isinstance(v, dict)},
        "quality": {
            "extent": extent,
            "lane_specifications_total": diagnosis.get("lane_specifications_total"),
            "routes": (diagnosis.get("routes") or {}).get("after"),
            "editable_audit_samples": ((diagnosis.get("editable") or {}).get("sample_totals")),
            "pointcloud_quality_status": diagnosis.get("pointcloud_quality_status"),
            "station_disposition": breakdown,
            "review_holds": review_hold_summary(diagnosis),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate both maps from bundled recordings with no operator.")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--association", choices=ASSOCIATIONS, default=ASSOCIATIONS[0],
                        help="trajectory_containing (default) refines once before inspecting; "
                             "all_supported_bands keeps the original proposal")
    parser.add_argument("--case", choices=sorted(CASES), action="append",
                        help="Case to run (repeatable; default: all)")
    args = parser.parse_args(argv)
    names = args.case or sorted(CASES)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    import cloudanalyzer_core  # noqa: F401 -- fail early without the native core

    results = [run_case(n, REPO_ROOT / CASES[n]["recording"], REPO_ROOT / CASES[n]["layout"], out, args.association)
               for n in names]
    summary = {
        "schema": "cloudanalyzer.mapping_e2e.v1",
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "peak_rss_mib": _peak_rss_mib()},
        "cases": results,
        "passed": all(r["passed"] for r in results),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for r in results:
        failed = [k for k, ok in r["checks"].items() if not ok]
        client, extent = r["reference_client"], r["quality"]["extent"]
        print(f"{r['case']}: {'PASS' if r['passed'] else 'FAIL ' + ', '.join(failed)} | "
              f"{client['inspected_candidates']}/{client['offered_candidates']} inspected, "
              f"{client['included_intervals']} intervals ({client['included_station_m']} m), "
              f"retained {extent.get('retained_fraction', 0):.1%} of {extent.get('trajectory_length_m', 0):.1f} m | "
              f"{r['timings_s']['total_s']} s")
        breakdown = r["quality"]["station_disposition"]
        if breakdown:
            def parts(values: dict[str, float]) -> str:
                return ", ".join(f"{k} {v:g} m" for k, v in values.items()) or "none"
            holds = r["quality"]["review_holds"]
            if holds:
                worst = max((s["below_trace_m"] for s in holds["below_trace_samples"]), default=0)
                print(f"  review holds: {holds['held_traces_total']}/{holds['traces_total']} traces held; "
                      f"{holds['below_trace_samples_total']} below-trace samples (max {worst:.2f} m)")
            print(f"  unresolved: source [{parts(breakdown['source_deferred_m'])}]; "
                  f"reference client [{parts(breakdown['client_deferred_m'])}]")
    print(f"peak RSS {summary['runtime']['peak_rss_mib']} MiB; summary: {out / 'summary.json'}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
