"""Freeze source-only native proof for the actual hard-intersection UI capture.

Read the existing prepared source cloud in place, never the separate map/labels.
Recorded poses keep their original timestamps and coordinates. Branch paths,
recorded-path intervals, lane priors, object types, associations and selected
geometric connections are explicit demo operator inputs, not recovered truth.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cloudanalyzer_core as core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    metadata = json.loads(
        (args.prepared / "preparation.json").read_text(encoding="utf-8")
    )
    if metadata["reference_inputs"]:
        raise ValueError("prepared source must not use reference inputs")
    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "web/media/vector-map-hard-intersection-inputs.json"
        ).read_text(encoding="utf-8")
    )
    if args.out.exists():
        raise FileExistsError("choose a new ignored proof directory")
    args.out.mkdir(parents=True)
    cloud = args.prepared / "geometry.las"
    previous = None
    builds = []
    for i, path in enumerate(config["paths"]):
        if path["source"] == "recorded":
            original = args.prepared / metadata["drives"][path["drive_index"]]["path"]
            lo, hi = path["relative_y_interval"]
            with original.open(encoding="utf-8", newline="") as stream:
                rows = [
                    row
                    for row in csv.DictReader(stream)
                    if lo <= float(row["y"]) - config["origin_xy"][1] <= hi
                ]
            points = [[row[k] for k in ("timestamp", "x", "y", "z")] for row in rows]
        else:
            points = [
                [j, config["origin_xy"][0] + x, config["origin_xy"][1] + y, 29.0]
                for j, (x, y) in enumerate(path["relative_xy"])
            ]
        drive = args.out / f"{path['source']}-{path['name']}.csv"
        with drive.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["timestamp", "x", "y", "z"])
            writer.writerows(points)
        built = json.loads(
            core.build_vector_map(
                str(cloud),
                str(drive),
                json.dumps(path["options"]),
                existing_map=previous,
            )
        )
        destination = args.out / f"roads-{i}.json"
        destination.write_text(built["map_json"], encoding="utf-8")
        previous = str(destination)
        builds.append(
            {
                "name": path["name"],
                "source": path["source"],
                "poses": len(points),
                "csv": drive.name,
                "options": path["options"],
                "report": built["report"],
            }
        )
    options = json.dumps(config["junction_options"])
    preview = json.loads(
        core.connect_vector_map_junctions(
            str(cloud), previous, options, preview_only=True
        )
    )
    (args.out / "junction-preview.json").write_text(
        json.dumps(preview, indent=2), encoding="utf-8"
    )
    available = {
        (c["from"], c["to"]) for c in preview["report"]["junctions"]["candidates"]
    }
    if not all(tuple(pair) in available for pair in config["reviewed_pairs"]):
        raise ValueError(
            "reviewed geometric branches changed; inspect source evidence again"
        )
    connected = json.loads(
        core.connect_vector_map_junctions(
            str(cloud),
            previous,
            options,
            lane_pairs=json.dumps(config["reviewed_pairs"]),
        )
    )
    roads = args.out / "generated-roads.json"
    roads.write_text(connected["map_json"], encoding="utf-8")
    (args.out / "roads-report.json").write_text(
        json.dumps({"builds": builds, "connections": connected["report"]}, indent=2),
        encoding="utf-8",
    )
    discovery_options = '{"scope":"ground_surface"}'
    preview = json.loads(
        core.discover_vector_map_features(str(cloud), str(roads), discovery_options)
    )
    (args.out / "ground-preview.json").write_text(
        json.dumps(preview, indent=2), encoding="utf-8"
    )
    candidates = {c["id"]: c for c in preview["report"]["discovery"]["candidates"]}
    confirmations = [
        {**c, "key": candidates[c["candidate"]]["key"]} for c in config["confirmations"]
    ]
    added = json.loads(
        core.discover_vector_map_features(
            str(cloud), str(roads), discovery_options, json.dumps(confirmations)
        )
    )
    (args.out / "reviewed.json").write_text(added["map_json"], encoding="utf-8")
    osm = args.out / "reviewed.osm"
    osm.write_text(added["osm"], encoding="utf-8")
    (args.out / "reviewed-report.json").write_text(
        json.dumps(added["report"], indent=2), encoding="utf-8"
    )
    replay = json.loads(
        core.discover_vector_map_features(
            str(cloud), str(osm), discovery_options, json.dumps(confirmations)
        )
    )
    assert all(c["reused"] for c in replay["report"]["additions"])
    assert added["report"]["validation"]["counts"]["errors"] == 0
    manifest = {
        "reference_inputs": [],
        "source_commit": args.source_commit,
        "native_sha256": hashlib.sha256(
            Path(core._core.__file__).read_bytes()
        ).hexdigest(),
        "input_cloud_sha256": hashlib.sha256(cloud.read_bytes()).hexdigest(),
        "prepared_metadata": metadata,
        "operator_inputs": config,
        "roundtrip_replay": replay["report"],
        "generated_sha256": hashlib.sha256(osm.read_bytes()).hexdigest(),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Native proof: {args.out}; points read in place, no map/labels read.")


if __name__ == "__main__":
    main()
