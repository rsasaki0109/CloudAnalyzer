"""Compare boundary jitter on identical inputs; this is not an accuracy metric.

Run vector_map_evaluate twice, disabling track_boundaries and fit_boundaries for
the baseline, then pass the two comparison.json files and an output JSON path.
Use the reference-distance metrics in their report.json files for accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def measure(data: dict) -> dict:
    turns = []
    displacements = []
    for road in data["generated"]:
        sources = road.get("source_boundaries") or road["boundaries"]
        for line, source in zip(road["boundaries"], sources):
            points = np.asarray(line, dtype=float)
            delta = np.diff(points[:, :2], axis=0)
            a, b = delta[:-1], delta[1:]
            valid = (np.linalg.norm(a, axis=1) > 1e-6) & (
                np.linalg.norm(b, axis=1) > 1e-6
            )
            cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            dot = np.sum(a * b, axis=1)
            turns.extend(np.degrees(np.abs(np.arctan2(cross[valid], dot[valid]))))
            original = np.asarray(source, dtype=float)
            if not np.array_equal(points[:, 2], original[:, 2]):
                raise ValueError("fitting changed source heights")
            displacements.extend(np.linalg.norm(points[:, :2] - original[:, :2], axis=1))
    if not turns:
        raise ValueError("no adjacent nonzero boundary segments")
    return {
        "adjacent_segment_turn_degrees": {
            "count": len(turns),
            "median": float(np.median(turns)),
            "p95": float(np.percentile(turns, 95)),
            "maximum": float(np.max(turns)),
        },
        "maximum_fit_xy_m": float(np.max(displacements)),
        "source_heights_unchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("fitted", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    fitted = json.loads(args.fitted.read_text())
    if baseline["reference"] != fitted["reference"] or baseline["cloud"] != fitted["cloud"]:
        raise ValueError("reference geometry or cloud preview differs")
    ignored = {"track_boundaries", "fit_boundaries"}
    if {k: v for k, v in baseline["options"].items() if k not in ignored} != {
        k: v for k, v in fitted["options"].items() if k not in ignored
    }:
        raise ValueError("extraction options differ beyond tracking and fitting")
    # Sampling must agree: changing vertex spacing would change this statistic.
    if baseline["trajectory"] != fitted["trajectory"]:
        raise ValueError("recorded trajectories differ")
    if len(baseline["generated"]) != len(fitted["generated"]):
        raise ValueError("supported stretches differ")
    for a, b in zip(baseline["generated"], fitted["generated"]):
        if a["reference"] != b["reference"]:
            raise ValueError("cross-section sampling differs")
        if len(a["boundaries"]) != len(b["boundaries"]):
            raise ValueError("lane counts differ")
    result = {
        "baseline": measure(baseline),
        "fitted": measure(fitted),
        "accuracy_metric": False,
        "limitations": "Lower heading variation can hide true corners or positional errors. "
        "Check the cloud and independent reference-distance scores separately.",
    }
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
