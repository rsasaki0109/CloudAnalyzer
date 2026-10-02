"""Regenerate source-footprint roads, reviewed connections and equipment.

Reuse the frozen source CSVs in place; no reference map or label supplies geometry.
The old generated road map is compared only after generation for regression identity.
Operator types, lane associations and selected manoeuvres are explicit draft inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vector_map_quality_audit import digest
from vector_map_surface_evaluate import run as compare_roads


def save(path: Path, value) -> None:
    text = value if isinstance(value, str) else json.dumps(value, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def path_ownership(comparison: dict) -> dict[int, str]:
    return {
        lane: case["name"]
        for case in comparison["cases"]
        for lane in case["after"]["added_lane_ids"]
    }


def check_selection(config: dict, comparison: dict, preview: dict) -> None:
    owners = path_ownership(comparison)
    available = {(c["from"], c["to"]): c for c in preview["candidates"]}
    for pair in config["reviewed_pairs"]:
        if owners[pair[0]] == owners[pair[1]]:
            raise ValueError(
                "do not reconnect a deferred interval within one input path"
            )
        candidate = available.get(tuple(pair))
        if (
            candidate is None
            or candidate.get("boundary_support") != [1.0, 1.0]
            or candidate["ground_support"] != 1.0
        ):
            raise ValueError(
                "selected connection lacks full centre and boundary source support"
            )
    if any(tuple(p) in available for p in config["deferred_requested_pairs"]):
        raise ValueError(
            "deferred requested turn changed; review original points again"
        )


def retained_roads(original: dict, changed: dict) -> bool:
    return all(
        {item["id"]: item for item in changed[key]}.get(item["id"]) == item
        for key in ("lanes", "boundaries")
        for item in original[key]
    )


def freeze(prepared: Path, proof: Path, out: Path, source_commit: str) -> dict:
    import cloudanalyzer_core as core

    if out.exists():
        raise FileExistsError("choose a new proof directory")
    out.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    config_path = root / "web/media/vector-map-supported-intersection-inputs.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    original_inputs = json.loads(
        (root / "web/media" / config["road_operator_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    frozen_manifest = json.loads((proof / "manifest.json").read_text(encoding="utf-8"))
    if frozen_manifest["operator_inputs"]["paths"] != original_inputs["paths"]:
        raise ValueError("source path priors changed; regenerate and review paths")
    comparison = compare_roads(prepared, proof, out / "road-comparison", source_commit)
    road_path = out / "road-comparison" / comparison["after"]["final_map"]
    road_map = json.loads(road_path.read_text())
    # A previous generated draft is a post-generation regression check, not an input.
    baseline = json.loads(
        (
            root
            / "benchmarks/vector-map/hard-intersection/source-footprint/after-road-5.json"
        ).read_text()
    )
    if road_map != baseline:
        raise ValueError(
            "source-footprint roads changed; review before retaining selections"
        )
    source = prepared / "geometry.las"
    options = json.dumps(config["junction_options"])
    legacy = json.loads(
        core.connect_vector_map_junctions(
            str(source), str(road_path), '{"max_gap":50}', preview_only=True
        )
    )
    checked = json.loads(
        core.connect_vector_map_junctions(
            str(source), str(road_path), options, preview_only=True
        )
    )
    check_selection(config, comparison, checked["report"]["junctions"])
    save(out / "junction-centre-preview.json", legacy["report"])
    save(out / "junction-preview.json", checked["report"])
    connected = json.loads(
        core.connect_vector_map_junctions(
            str(source),
            str(road_path),
            options,
            lane_pairs=json.dumps(config["reviewed_pairs"]),
        )
    )
    connected_map = json.loads(connected["map_json"])
    if not retained_roads(road_map, connected_map):
        raise AssertionError("connections changed retained source-footprint roads")
    connected_path = out / "connected.json"
    save(connected_path, connected["map_json"])
    save(out / "connections-report.json", connected["report"])
    preview = json.loads(
        core.discover_vector_map_features(
            str(source), str(connected_path), '{"scope":"ground_surface"}'
        )
    )
    save(out / "equipment-preview.json", preview["report"])
    candidates = {c["id"]: c for c in preview["report"]["discovery"]["candidates"]}
    confirmations = [
        {**c, "key": candidates[c["candidate"]]["key"]} for c in config["confirmations"]
    ]
    added = json.loads(
        core.discover_vector_map_features(
            str(source),
            str(connected_path),
            '{"scope":"ground_surface"}',
            json.dumps(confirmations),
        )
    )
    final = json.loads(added["map_json"])
    if not retained_roads(road_map, final) or not retained_roads(connected_map, final):
        raise AssertionError("equipment changed roads or connections")
    save(out / "reviewed.json", added["map_json"])
    save(out / "reviewed.osm", added["osm"])
    save(out / "map_projector_info.yaml", added["projector_info"])
    save(out / "reviewed-report.json", added["report"])
    quality = json.loads(
        core.audit_vector_map_quality(str(source), str(out / "reviewed.json"))
    )
    save(out / "quality.json", quality)
    if (
        quality["quality"]["low_support_lanes"]
        or quality["quality"]["omitted_lanes"]
        or quality["quality"]["malformed_lanes"]
        or quality["quality"]["limited"]
    ):
        raise ValueError(
            "final road graph still needs source review; do not publish capture"
        )
    if quality["validation"]["counts"]["errors"]:
        raise ValueError("final map has structural errors")
    replay = json.loads(
        core.discover_vector_map_features(
            str(source),
            str(out / "reviewed.osm"),
            '{"scope":"ground_surface"}',
            json.dumps(confirmations),
        )
    )
    if not all(a["reused"] for a in replay["report"]["additions"]):
        raise AssertionError("equipment replay after OSM import was not a no-op")
    manifest = {
        "reference_inputs": [],
        "source_commit": source_commit,
        "native_sha256": digest(Path(core._core.__file__)),
        "input_cloud_sha256": comparison["source_sha256"],
        "operator_inputs": config,
        "operator_inputs_sha256": digest(config_path),
        "path_operator_inputs": original_inputs["paths"],
        "source_roads_exact": True,
        "retained_roads_and_connections_exact": True,
        "road_length_m": comparison["after"]["generated_length_m"],
        "deferred_road_length_m": comparison["after"]["reported_deferred_length_m"],
        "legacy_centre_candidates": len(legacy["report"]["junctions"]["candidates"]),
        "full_curve_candidates": len(checked["report"]["junctions"]["candidates"]),
        "selected_connections": len(connected["report"]["junctions"]["added"]),
        "road_lanes": len(final["lanes"]),
        "crosswalks": len(final.get("crosswalks", [])),
        "stop_lines": len(final.get("stop_lines", [])),
        "signal_housings": len(final.get("traffic_signals", [])),
        "source_quality": quality["quality"],
        "validation": quality["validation"],
        "generated_sha256": digest(out / "reviewed.osm"),
        "roundtrip_replay": replay["report"],
        "limitations": [
            "Same development scene, not independent accuracy",
            "Missing road extent and disconnected fragments are retained as uncertainty",
            "Ground curves do not certify road interiors, clearance or lawful manoeuvres",
            "Paint geometry is observed; object types, controlled lanes and permitted turns need semantic review",
            "Proposal cap and unmatched/omitted equipment remain; signal lamps, states and stop associations are not inferred",
        ],
    }
    save(out / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "proof", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    result = freeze(args.prepared, args.proof, args.out, args.source_commit)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "road_lanes",
                    "crosswalks",
                    "stop_lines",
                    "signal_housings",
                    "selected_connections",
                    "deferred_road_length_m",
                )
            },
            indent=2,
        )
    )
