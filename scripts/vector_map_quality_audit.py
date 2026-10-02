"""Audit the frozen generated intersection; references are opened only afterwards.

This is a development-scene geometry audit, not semantic precision/recall or
independent generalization. No annotations enter native source-coverage checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from hard_intersection_evaluate import (
    gated_assignment,
    read_reference,
    reference_center,
    symmetric_error,
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        result = hashlib.file_digest(stream, "sha256")
    return result.hexdigest()


def verify_frozen_map(proof: Path) -> tuple[Path, dict]:
    manifest = json.loads((proof / "manifest.json").read_text(encoding="utf-8"))
    if manifest["reference_inputs"]:
        raise ValueError("proof generation used reference inputs")
    vector_map = proof / "reviewed.osm"
    if digest(vector_map) != manifest["generated_sha256"]:
        raise ValueError("generated map changed after freezing")
    return vector_map, manifest


def generated_equipment(path: Path) -> dict:
    """Use generated Local metre coordinates, never zero export lat/lon."""
    root = ET.parse(path).getroot()
    nodes = {}
    for node in root.findall("node"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in node.findall("tag")}
        nodes[node.attrib["id"]] = [
            float(tags[k]) for k in ("local_x", "local_y", "ele")
        ]
    ways, families = {}, {"repeated_paint": [], "bright_bar": [], "elevated_panel": []}
    for way in root.findall("way"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in way.findall("tag")}
        item = {
            "id": way.attrib["id"],
            "geometry": [nodes[n.attrib["ref"]] for n in way.findall("nd")],
            "tags": tags,
        }
        ways[item["id"]] = item
        kind = {"stop_line": "bright_bar", "traffic_light": "elevated_panel"}.get(
            tags.get("type")
        )
        if kind:
            families[kind].append(item)
    for relation in root.findall("relation"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in relation.findall("tag")}
        if tags.get("type") != "lanelet" or tags.get("subtype") != "crosswalk":
            continue
        sides = {
            m.attrib["role"]: ways[m.attrib["ref"]]["geometry"]
            for m in relation.findall("member")
            if m.attrib["type"] == "way" and m.attrib["role"] in {"left", "right"}
        }
        ring = sides["left"] + list(reversed(sides["right"]))
        families["repeated_paint"].append(
            {"id": relation.attrib["id"], "geometry": ring + [ring[0]], "tags": tags}
        )
    return families


def equipment_comparison(generated: dict, references: dict) -> dict:
    report = {}
    for family, predictions in generated.items():
        refs = references[family]
        pairs = gated_assignment(
            [reference_center(p, family) for p in predictions],
            [reference_center(r, family) for r in refs],
            surface=family != "elevated_panel",
        )
        used_p, used_r = {i for i, _, _ in pairs}, {j for _, j, _ in pairs}
        report[family] = {
            "confirmed_objects": len(predictions),
            "reference_objects": len(refs),
            "nearby_correspondences": len(pairs),
            "unmatched_generated": [
                p["id"] for i, p in enumerate(predictions) if i not in used_p
            ],
            "reference_without_confirmed_match": [
                r["id"] for j, r in enumerate(refs) if j not in used_r
            ],
            "matches": [
                {
                    "generated": predictions[i]["id"],
                    "reference": refs[j]["id"],
                    "center_distance_m": d,
                    "geometry": symmetric_error(
                        predictions[i]["geometry"], refs[j]["geometry"]
                    ),
                }
                for i, j, d in pairs
            ],
        }
    return report


def run(
    source: Path, proof: Path, output: Path, dataset: Path | None, source_commit: str
) -> dict:
    import cloudanalyzer_core as core

    if output.exists():
        raise FileExistsError("choose a new report path")
    vector_map, manifest = verify_frozen_map(proof)
    if digest(source) != manifest["input_cloud_sha256"]:
        raise ValueError("source points differ from the frozen proof")
    native = json.loads(core.audit_vector_map_quality(str(source), str(vector_map)))
    report = {
        "kind": "generated_map_quality_development_audit",
        "generation_reference_inputs": [],
        "generated_source_commit": manifest["source_commit"],
        "audit_source_commit": source_commit,
        "audit_native_sha256": digest(Path(core._core.__file__)),
        "input_cloud_sha256": manifest["input_cloud_sha256"],
        "generated_osm_sha256": manifest["generated_sha256"],
        "native": native,
        "limitations": [
            "Not an independent held-out scene; used in development.",
            "Coverage checks original retained source returns, not road semantics or obstacle clearance.",
            "Operator paths, lane counts/widths and feature associations remain inputs.",
            "References are incomplete; unmatched objects are not proven false positives.",
            "Measured paint extents and mapped crossing footprints differ; no reference shape completion.",
        ],
    }
    # Freeze/report source-only results before opening separate evaluation geometry.
    source_only_hash = hashlib.sha256(
        json.dumps(native, sort_keys=True).encode()
    ).hexdigest()
    report["source_only_audit_sha256"] = source_only_hash
    if dataset:
        references, _, coordinate_audit = read_reference(
            dataset / "maps/lanelet2/jp_tokyo_takanawadai.osm"
        )
        report["coordinate_audit"] = coordinate_audit
        report["equipment"] = equipment_comparison(
            generated_equipment(vector_map), references
        )
        report["reference_osm_sha256"] = digest(
            dataset / "maps/lanelet2/jp_tokyo_takanawadai.osm"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("proof", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--dataset", type=Path, help="Optional post-generation reference comparison"
    )
    parser.add_argument(
        "--source-commit",
        required=True,
        help="Source used to build the installed native audit module",
    )
    args = parser.parse_args()
    result = run(args.source, args.proof, args.output, args.dataset, args.source_commit)
    q = result["native"]["quality"]
    print(
        f"{len(q['lanes'])} lanes checked; {len(q['low_support_lanes'])} need source review; {len(q['omitted_lanes'])} omitted."
    )
