"""Freeze source-only RGB boundary experiments before reading survey references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from pyproj import Transformer

from vector_map_cross_scene_evaluate import boundary_distance, save
from vector_map_quality_audit import digest


def reference_document(path: Path, epsg: str | None) -> dict:
    if not epsg:
        import cloudanalyzer_core as core
        return json.loads(json.loads(core.edit_vector_map_relations(str(path)))["map_json"])
    root = ET.parse(path).getroot()
    transform = Transformer.from_crs("EPSG:4326", epsg, always_xy=True)
    nodes = {}
    for node in root.findall("node"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in node.findall("tag")}
        nodes[node.attrib["id"]] = [*transform.transform(float(node.attrib["lon"]), float(node.attrib["lat"])), float(tags["ele"])]
    ways = {w.attrib["id"]: [nodes[n.attrib["ref"]] for n in w.findall("nd")] for w in root.findall("way")}
    lanes = []
    for relation in root.findall("relation"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in relation.findall("tag")}
        if tags.get("type") != "lanelet" or tags.get("subtype") != "road":
            continue
        sides = {m.attrib["role"]: int(m.attrib["ref"]) for m in relation.findall("member") if m.attrib["role"] in ("left", "right")}
        if len(sides) == 2:
            lanes.append({"id": int(relation.attrib["id"]), "kind": "driving", **sides})
    return {"lanes": lanes, "boundaries": [{"id": int(k), "geometry": v} for k, v in ways.items()]}


def run(source: Path, config_path: Path, reference: Path, out: Path, commit: str, epsg: str | None) -> dict:
    import cloudanalyzer_core as core
    if out.exists():
        raise FileExistsError("choose a new output directory")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("reference_inputs") != []:
        raise ValueError("generation must declare empty reference inputs")
    out.mkdir(parents=True)
    before_hash = digest(source)
    modes = {"prior": [], "rgb": []}
    for mode in modes:
        previous = None
        for i, case in enumerate(config["cases"]):
            trajectory = Path(case["trajectory"])
            options = {**case["options"], "fit_source_surface": True, "observe_rgb_boundaries": mode == "rgb"}
            payload = json.loads(core.build_vector_map(str(source), str(trajectory), json.dumps(options), existing_map=str(previous) if previous else None))
            previous = out / f"{mode}-{i}.json"
            previous.write_text(payload["map_json"], encoding="utf-8", newline="\n")
            (out / f"{mode}-{i}.osm").write_text(payload["osm"], encoding="utf-8", newline="\n")
            audit = json.loads(core.audit_vector_map_quality(str(source), str(previous)))
            save(out / f"{mode}-{i}-audit.json", audit)
            modes[mode].append({"name": case["name"], "trajectory_sha256": digest(trajectory), "options": options,
                                "extraction": payload["report"]["extraction"], "map": previous.name})
    frozen = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
    save(out / "generation-freeze.json", {"reference_inputs": [], "artifact_sha256": frozen})
    # References are opened only after every generated map and source audit exists.
    surveyed = reference_document(reference, epsg)
    metrics = {}
    for mode, cases in modes.items():
        final = json.loads((out / cases[-1]["map"]).read_text(encoding="utf-8"))
        quality = json.loads((out / f"{mode}-{len(cases)-1}-audit.json").read_text(encoding="utf-8"))["quality"]
        metrics[mode] = {"boundary_distance": boundary_distance(final, surveyed),
                         "generated_path_m": sum(c["extraction"]["generated_length"] for c in cases),
                         "deferred_path_m": sum(c["extraction"]["surface_fit"]["deferred_length_m"] for c in cases),
                         "rgb_source_vertices": sum(c["extraction"]["rgb_paint_vertices"] for c in cases),
                         "lane_fragments": len(final["lanes"]), "source_flags": quality["low_support_lanes"],
                         "source_samples": quality["sampled_points"], "limited": quality["limited"]}
    result = {"source_commit": commit, "native_sha256": digest(Path(core._core.__file__)), "source_sha256": before_hash,
              "config_sha256": digest(config_path), "reference_sha256": digest(reference),
              "reference_transform": epsg or "native coordinate metadata", "generation_reference_inputs": [],
              "generation_artifact_sha256": frozen, "cases": modes, "metrics": metrics,
              "limitations": ["Known development scenes; no held-out or semantic accuracy claim.",
                 "Unpaired nearest surveyed boundary XY distances; no registration or paired lane identity.",
                 "Generation extent can change: compare deferred length and fragments alongside distances.",
                 "Source support is the generation gate, not independent accuracy.",
                 "Only narrow longitudinal RGB paint with observed dark flanks; short/dashed/occluded marks may be missed."]}
    assert digest(source) == before_hash
    assert all(digest(out / name) == sha for name, sha in frozen.items())
    save(out / "evaluation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "config", "reference", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--reference-epsg")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.config, args.reference, args.out, args.source_commit, args.reference_epsg)["metrics"], indent=2))
