"""Compare source-footprint road generation on fixed inputs, without reference maps.

Counts concern regenerated road stretches, not the frozen junction/equipment map.
Deferred length is reported alongside support: filtering is not an accuracy score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vector_map_quality_audit import digest, verify_frozen_map


def summarize(quality: dict) -> dict:
    return {
        "checked_lanes": len(quality["lanes"]),
        "source_review_lane_ids": quality["low_support_lanes"],
        "sampled_points": quality["sampled_points"],
        "omitted_lanes": quality["omitted_lanes"],
        "malformed_lanes": quality["malformed_lanes"],
        "limited": quality["limited"],
    }


def evaluate(source: Path, cases: list[dict], out: Path, source_commit: str) -> dict:
    import cloudanalyzer_core as core

    if out.exists():
        raise FileExistsError("choose a new output directory")
    out.mkdir(parents=True)
    report = {
        "source_commit": source_commit,
        "native_sha256": digest(Path(core._core.__file__)),
        "source_sha256": digest(source),
        "reference_inputs": [],
        "artifact_newlines": "LF",
        "cases": [],
        "limitations": [
            "Road-only regeneration; frozen junctions/equipment and associations are not transferred.",
            "Support is the generation gate, not independent accuracy or road semantics.",
            "Lower review counts include omission of unsupported extent; inspect deferred length and fragments.",
            "Lane counts, traffic directions and width priors remain explicit operator inputs.",
            "Existing project development scenes, not strictly held-out generalization evidence.",
        ],
    }
    previous = {"before": None, "after": None}
    lane_ids = {"before": set(), "after": set()}
    for index, case in enumerate(cases):
        item = {
            "name": case["name"],
            "trajectory_sha256": digest(case["trajectory"]),
            "options": case["options"],
        }
        for mode in ("before", "after"):
            options = {**case["options"], "fit_source_surface": mode == "after"}
            try:
                built = json.loads(
                    core.build_vector_map(
                        str(source),
                        str(case["trajectory"]),
                        json.dumps(options),
                        existing_map=str(previous[mode]) if previous[mode] else None,
                    )
                )
            except ValueError as error:
                if mode == "before":
                    raise
                item[mode] = {"entire_path_deferred": True, "error": str(error)}
                continue
            vector_map = out / f"{mode}-road-{index}.json"
            vector_map.write_text(built["map_json"], encoding="utf-8", newline="\n")
            (out / f"{mode}-road-{index}.osm").write_text(
                built["osm"], encoding="utf-8", newline="\n"
            )
            projector = out / f"{mode}-projector-info.yaml"
            projector.write_text(
                built["projector_info"], encoding="utf-8", newline="\n"
            )
            audit = json.loads(
                core.audit_vector_map_quality(str(source), str(vector_map))
            )
            (out / f"{mode}-audit-{index}.json").write_text(
                json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n"
            )
            current = json.loads(built["map_json"])
            current_ids = {lane["id"] for lane in current["lanes"]}
            added = current_ids - lane_ids[mode]
            per_case = {
                **audit["quality"],
                "lanes": [
                    lane for lane in audit["quality"]["lanes"] if lane["lane"] in added
                ],
                "low_support_lanes": [
                    lane
                    for lane in audit["quality"]["low_support_lanes"]
                    if lane in added
                ],
            }
            # The lane subset must not inherit the cumulative map's sample count.
            per_case["sampled_points"] = sum(
                lane[side]["samples"]
                for lane in per_case["lanes"]
                for side in ("center", "left", "right")
            )
            item[mode] = {
                "map": vector_map.name,
                "map_sha256": digest(vector_map),
                "generated_length_m": built["report"]["extraction"]["generated_length"],
                "surface_fit": built["report"]["extraction"]["surface_fit"],
                "added_lane_ids": sorted(added),
                "source_quality": summarize(per_case),
                "validation": audit["validation"]["counts"],
            }
            # Freeze the mode's newly generated geometry before the next pass.
            previous[mode] = vector_map
            lane_ids[mode] = current_ids
            report[mode] = {
                "final_map": vector_map.name,
                "final_osm": f"{mode}-road-{index}.osm",
                "final_projector": projector.name,
                "source_quality": summarize(audit["quality"]),
            }
        report["cases"].append(item)
    for mode in ("before", "after"):
        report.setdefault(
            mode,
            {
                "final_map": None,
                "final_osm": None,
                "final_projector": None,
                "source_quality": None,
            },
        )
        report[mode]["generated_length_m"] = sum(
            c[mode].get("generated_length_m", 0.0) for c in report["cases"]
        )
    report["after"]["reported_deferred_length_m"] = sum(
        (c["after"].get("surface_fit") or {}).get("deferred_length_m", 0.0)
        for c in report["cases"]
    )
    report["after"]["entire_path_deferred_cases"] = [
        c["name"] for c in report["cases"] if c["after"].get("entire_path_deferred")
    ]
    fits = [c["after"].get("surface_fit") for c in report["cases"]]
    report["after"]["evaluated_path_length_m"] = (
        sum(fit["evaluated_path_length_m"] for fit in fits) if all(fits) else None
    )
    (out / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def run(prepared: Path, proof: Path, out: Path, source_commit: str) -> dict:
    _, manifest = verify_frozen_map(proof)
    source = prepared / "geometry.las"
    if digest(source) != manifest["input_cloud_sha256"]:
        raise ValueError("source differs from the frozen proof")
    builds = json.loads((proof / "roads-report.json").read_text())["builds"]
    cases = [
        {"name": b["name"], "trajectory": proof / b["csv"], "options": b["options"]}
        for b in builds
    ]
    report = evaluate(source, cases, out, source_commit)
    # This frozen generated draft is opened only AFTER generation, for regression
    # identity, not fitting. Original defaults must reproduce the same road IR.
    old = json.loads((proof / "roads-5.json").read_text())
    legacy = json.loads((out / report["before"]["final_map"]).read_text())
    if old != legacy:
        raise AssertionError(
            "legacy roads changed; inspect before reporting improvement"
        )
    report["legacy_generated_roads_exact"] = True
    report["frozen_media_osm_sha256"] = manifest["generated_sha256"]
    report["before"]["final_map_sha256"] = digest(out / report["before"]["final_map"])
    report["after"]["final_map_sha256"] = (
        digest(out / report["after"]["final_map"])
        if report["after"]["final_map"]
        else None
    )
    (out / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="mode", required=True)
    frozen = subcommands.add_parser("frozen", help="replay the frozen six-path proof")
    frozen.add_argument("prepared", type=Path)
    frozen.add_argument("proof", type=Path)
    scene = subcommands.add_parser("scene", help="compare one cloud and trajectory")
    scene.add_argument("cloud", type=Path)
    scene.add_argument("trajectory", type=Path)
    scene.add_argument("--name", required=True)
    scene.add_argument("--right-hand", action="store_true")
    scene.add_argument("--forward-lanes", type=int, default=1)
    scene.add_argument("--backward-lanes", type=int, default=1)
    scene.add_argument("--lane-width", type=float, default=3.5)
    scene.add_argument("--segment-length", type=float, default=50)
    for command in (frozen, scene):
        command.add_argument("out", type=Path)
        command.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.mode == "frozen":
        result = run(args.prepared, args.proof, args.out, args.source_commit)
    else:
        result = evaluate(
            args.cloud,
            [
                {
                    "name": args.name,
                    "trajectory": args.trajectory,
                    "options": {
                        "left_hand_traffic": not args.right_hand,
                        "forward_lanes": args.forward_lanes,
                        "backward_lanes": args.backward_lanes,
                        "lane_width": args.lane_width,
                        "segment_length": args.segment_length,
                    },
                }
            ],
            args.out,
            args.source_commit,
        )
    print(json.dumps({mode: result[mode] for mode in ("before", "after")}, indent=2))
