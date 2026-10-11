"""Measure end-to-end HD draft coverage of a recording with a fixed, scripted agent policy.

The policy repeats the rule of the earlier NCLT runs, so native changes can be
compared without an agent's judgement in the loop: refine to trajectory-containing
association, inspect every candidate, include each interval whose two end sections
meet the fixed minimum lane width and defer the rest, then draft once.

    python scripts/mapping_e2e_coverage.py web/public/samples/nclt-2012-04-29.mcap runs/april 2.5
    python scripts/mapping_e2e_coverage.py web/public/samples/nclt-2012-06-15.mcap runs/june 2.0

Optional trailing `FROM:TO` station ranges are deferred as well. The summary is
printed and saved as `e2e-result.json` in the new run directory.
"""

import json
import sys
import time

from ca.mapping_run import advance_mapping_run, inspect_mapping_run, start_mapping_run


def main() -> None:
    source, out, width = sys.argv[1], sys.argv[2], float(sys.argv[3])
    deferred = [tuple(map(float, r.split(":"))) for r in sys.argv[4:]]
    layout = {"boundary_policy": "source_span_hypothesis", "speed_limit_kmh": 40.0,
              "reason": "Fixed unverified one-lane hypothesis for an end-to-end coverage comparison",
              "lanes": [{"direction": "forward", "kind": "driving", "one_way": True, "fraction": 1.0,
                         "minimum_width_m": width}]}
    start = time.time()
    run = start_mapping_run(source, out, layout)
    if run["status"] != "needs_agent":
        raise SystemExit(f"run did not start: {run['status']}")

    def act(action: dict, reason: str) -> dict:
        nonlocal run
        result = advance_mapping_run(out, action, reason, run["revision"])
        run = inspect_mapping_run(out)
        return result

    act({"type": "refine", "association": "trajectory_containing"},
        "Off-path bands fragment the route; test explicit path association with unchanged thresholds")
    proposal = json.load(open(run["candidate_index"]["file"]["path"]))
    ids = [c["id"] for c in proposal["candidates"]]
    for k in range(0, len(ids), 8):
        act({"type": "inspect", "candidate_ids": ids[k:k + 8]}, "Inspect every candidate before deciding ranges")

    decisions = []
    for candidate in proposal["candidates"]:
        runs: list[list] = []
        for a, b in zip(candidate["sections"], candidate["sections"][1:]):
            keep = (a["support_span_m"] >= width and b["support_span_m"] >= width
                    and not any(lo <= a["station_m"] < hi for lo, hi in deferred))
            if runs and runs[-1][0] == keep and runs[-1][2] == a["station_m"]:
                runs[-1][2] = b["station_m"]
            else:
                runs.append([keep, a["station_m"], b["station_m"]])
        for keep, lo, hi in runs:
            decisions.append({"candidate_id": candidate["id"], "action": "include" if keep else "defer",
                              "from_m": lo, "to_m": hi,
                              "reason": "Observed end sections meet the fixed minimum lane width; unverified lane hypothesis"
                              if keep else "An end section is narrower than the fixed width or deferred; no extrapolation"})
    draft = act({"type": "draft", "decisions": decisions},
                "Draft every inspected interval meeting the fixed width").get("draft_result", {})
    summary = {
        "source": source, "minimum_width_m": width, "deferred_ranges": deferred,
        "corridor": {k: proposal[k] for k in ("trajectory_length_m", "trajectory_covered_station_length_m",
                                               "without_candidate_station_length_m")},
        "candidates": len(ids),
        "included_m": sum(d["to_m"] - d["from_m"] for d in decisions if d["action"] == "include"),
        "draft_status": draft.get("status"), "draft_error": draft.get("error"),
        "extent": (draft.get("diagnosis") or {}).get("extent"), "seconds": round(time.time() - start, 1),
    }
    with open(f"{out}/e2e-result.json", "w") as f:
        json.dump({"summary": summary, "draft": draft}, f, indent=1, default=str)
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
