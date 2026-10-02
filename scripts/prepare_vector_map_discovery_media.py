"""Prepare external-source proof for the source-only equipment UI capture.

Paths and object/lane confirmations are explicit demo operator inputs. No surveyed
map, feature boxes or reference geometry are read. Raw points stay in place.
"""
import argparse
import json
from pathlib import Path

import cloudanalyzer_core as core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cloud", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if not args.cloud.is_file():
        raise FileNotFoundError(args.cloud)
    if args.out.exists():
        raise FileExistsError("choose a new ignored proof directory")
    args.out.mkdir(parents=True)
    paths = json.loads((Path(__file__).resolve().parents[1] / "web/media/vector-map-operator-paths.json").read_text())
    previous = None
    reports = []
    for i, points in enumerate(paths):
        drive = args.out / f"operator-path-{i}.csv"
        drive.write_text("timestamp,x,y,z\n" + "\n".join(f"{j},{','.join(map(str, p))}" for j, p in enumerate(points)))
        result = json.loads(core.build_vector_map(str(args.cloud), str(drive), '{"segment_length":0}', existing_map=previous))
        current = args.out / f"roads-{i}.json"
        current.write_text(result["map_json"])
        reports.append(result["report"])
        previous = str(current)
    preview = json.loads(core.connect_vector_map_junctions(str(args.cloud), previous, preview_only=True))
    pairs = [[c["from"], c["to"]] for c in preview["report"]["junctions"]["candidates"]]
    assert pairs == [[4, 9], [4, 15], [10, 5], [14, 5], [14, 9]]
    roads = json.loads(core.connect_vector_map_junctions(str(args.cloud), previous, lane_pairs=json.dumps(pairs)))
    generated = args.out / "generated-roads.json"
    generated.write_text(roads["map_json"])
    (args.out / "roads-report.json").write_text(json.dumps({"path_source": "operator-traced, not a recorded drive", "input_map": False, "builds": reports, "junctions": roads["report"]}, indent=2))
    options = '{"scope":"ground_surface"}'
    preview = json.loads(core.discover_vector_map_features(str(args.cloud), str(generated), options))
    (args.out / "ground-preview.json").write_text(json.dumps(preview, indent=2))
    candidates = {c["id"]: c for c in preview["report"]["discovery"]["candidates"]}
    # Reviews refer to already generated proposals, never detector inputs.
    assert candidates[93]["evidence"]["kind"] == "bright_bar"
    assert candidates[94]["evidence"]["kind"] == "repeated_paint"  # rejected in UI
    assert candidates[96]["evidence"]["kind"] == "elevated_panel"
    confirmations = [{"candidate": i, "key": candidates[i]["key"], "classification": kind, "lanes": [14]} for i, kind in [(93, "stop_line"), (96, "vehicle_signal")]]
    encoded = json.dumps(confirmations)
    (args.out / "confirmations.json").write_text(encoded)
    added = json.loads(core.discover_vector_map_features(str(args.cloud), str(generated), options, encoded))
    (args.out / "reviewed.json").write_text(added["map_json"])
    osm = args.out / "reviewed.osm"
    osm.write_text(added["osm"])
    (args.out / "reviewed-report.json").write_text(json.dumps(added["report"], indent=2))
    replay = json.loads(core.discover_vector_map_features(str(args.cloud), str(osm), options, encoded))
    assert all(a["reused"] for a in replay["report"]["additions"])
    (args.out / "roundtrip-report.json").write_text(json.dumps(replay["report"], indent=2))
    print(f"Source-only media proof written to {args.out}; raw cloud was not copied.")


if __name__ == "__main__":
    main()
