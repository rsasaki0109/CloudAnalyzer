"""Separate source-only generation from a surveyed-context association audit.

References are opened only after generation artifacts are frozen. Masking a
query removes its target, not the surveyed geometry or other marking contexts;
this is a component audit, never end-to-end semantic precision/recall.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from hard_intersection_evaluate import gated_assignment, resample
from vector_map_quality_audit import digest
from vector_map_surface_evaluate import evaluate as evaluate_roads


def save(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def mask_target(document: dict, rule_id: int) -> dict:
    result = copy.deepcopy(document)
    rule = next(r for r in result["regulatory_elements"] if r["id"] == rule_id)
    rule["rule"].pop("stop_line", None)
    rule["rule"].pop("stop_lines", None)
    rule.pop("controlled_crosswalks", None)
    return result


def summarize_relations(records: list[dict]) -> dict:
    expected = correct = wrong = movement_changes = 0
    unresolved = mixed = limited = 0
    for r in records:
        p = r["proposal"]
        mixed += p["kind"] == "mixed"
        limited += p["limited"]
        unresolved += p["eligible_count"] == 0
        has_expected = False
        for c in p["candidates"]:
            if not c["eligible"]:
                continue
            target_matches = c["target_kind"] == "stop_line" and c["target_id"] == r["reference_stop"]
            lanes_match = set(c["lanes"]) == set(r["reviewed_lanes"])
            correct += target_matches and lanes_match
            wrong += not target_matches
            movement_changes += not lanes_match
            has_expected |= target_matches and lanes_match
        expected += has_expected
    return {"rules": len(records), "rules_with_expected_candidate": expected,
            "matching_target_and_movement_pairs": correct, "different_reference_target_pairs": wrong,
            "changed_movement_pairs": movement_changes, "rules_without_eligible_candidate": unresolved,
            "mixed_kind_rules_held": mixed, "limited_rules": limited}


def references(document: dict) -> dict:
    result = {k: [] for k in ("repeated_paint", "bright_bar", "elevated_panel")}
    for item in document["crosswalks"]:
        ring = item["left_edge"] + item["right_edge"][::-1]
        result["repeated_paint"].append({"id": item["id"], "geometry": ring})
    result["bright_bar"] = document["stop_lines"]
    result["elevated_panel"] = document["traffic_signals"]
    return result


def center(item: dict, family: str, prediction: bool = False) -> np.ndarray:
    item = item["evidence"] if prediction else item
    points = item["measurement"]["outline"] if prediction and family == "repeated_paint" else item["geometry"]
    p = np.asarray(points, dtype=float)
    c = (p.min(axis=0) + p.max(axis=0)) / 2
    if family == "elevated_panel":
        c[2] += (item.get("height") or 0.5) / 2
    return c


def compare_features(discovery: dict, surveyed: dict) -> dict:
    results = {}
    for family, refs in references(surveyed).items():
        proposals = [c for c in discovery["candidates"] if c["evidence"]["kind"] == family]
        pairs = gated_assignment([center(p, family, True) for p in proposals],
                                 [center(r, family) for r in refs], surface=family != "elevated_panel")
        results[family] = {"retained_proposals": len(proposals), "reference_objects": len(refs),
            "nearby_one_to_one_pairs": len(pairs),
            "pairs": [{"candidate": proposals[i]["id"], "reference": refs[j]["id"],
                       "center_distance_m": d} for i, j, d in pairs],
            "unmatched_proposals": len(proposals) - len(pairs), "unmatched_references": len(refs) - len(pairs)}
    return results


def boundary_distance(generated: dict, surveyed: dict) -> dict:
    def samples(doc):
        ids = set()
        for lane in doc["lanes"]:
            if lane["kind"] != "driving":
                continue
            for side in ("left", "right"):
                ref = lane[side]
                ids.add(ref if isinstance(ref, int) else ref["boundary"])
        return np.concatenate([resample(b["geometry"], 0.5) for b in doc["boundaries"] if b["id"] in ids])
    a, b = samples(generated), samples(surveyed)
    d = cKDTree(b[:, :2]).query(a[:, :2])[0]
    return {"generated_samples": len(d), "mean_xy_m": float(d.mean()),
            "p90_xy_m": float(np.quantile(d, .9)), "maximum_xy_m": float(d.max()),
            "fraction_within_0_5m": float((d <= .5).mean()),
            "role": "Unpaired nearest surveyed boundary distances; no fitted registration, lane pairing or full-scene completeness claim."}


def run(cloud: Path, inputs: Path, reference: Path, out: Path, source_commit: str) -> dict:
    import cloudanalyzer_core as core

    if out.exists():
        raise FileExistsError("choose a new output directory")
    config = json.loads(inputs.read_text(encoding="utf-8"))
    if config.get("reference_inputs") != []:
        raise ValueError("generation configuration must declare no reference inputs")
    out.mkdir(parents=True)
    source_hash = digest(cloud)
    cases = []
    for i, path in enumerate(config["operator_paths"]):
        csv = out / f"operator-path-{i}.csv"
        csv.write_text("timestamp,x,y,z\n" + "".join(f"{j},{x},{y},{z}\n" for j, (x,y,z) in enumerate(path)), encoding="utf-8", newline="\n")
        cases.append({"name": f"operator-path-{i}", "trajectory": csv, "options": config["build_options"]})
    roads = evaluate_roads(cloud, cases, out / "roads", source_commit)
    discovery_payload = json.loads(core.discover_vector_map_features(str(cloud), None,
                               json.dumps({"scope": "ground_surface"}), None))
    discovery = discovery_payload["report"]["discovery"]
    save(out / "discovery.json", discovery)
    final_name = roads["after"]["final_map"]
    final_map = out / "roads" / final_name if final_name else None
    junction_report = {"status": "no_source_supported_roads", "junctions": None}
    if final_map:
        preview = json.loads(core.connect_vector_map_junctions(str(cloud), str(final_map),
                             options=json.dumps({"min_ground_support":1, "check_boundary_support":True})))
        junction_report = preview["report"]
    save(out / "junction-preview.json", junction_report)
    # Write and hash the complete source-only artifacts BEFORE reading references.
    frozen = {p.relative_to(out).as_posix():digest(p) for p in out.rglob("*") if p.is_file()}
    save(out / "generation-freeze.json", {"reference_inputs": [], "artifact_sha256": frozen})
    surveyed_payload = json.loads(core.edit_vector_map_relations(str(reference)))
    surveyed = json.loads(surveyed_payload["map_json"])
    component = {"existing": [], "query_target_masked": []}
    path = out / "component-input.json"
    for r in surveyed["regulatory_elements"]:
        if r["rule"]["type"] != "traffic_light":
            continue
        for mode in component:
            document = surveyed if mode == "existing" else mask_target(surveyed, r["id"])
            save(path, document)
            before = digest(path)
            p = json.loads(core.propose_vector_map_relations(str(path), r["id"]))
            assert digest(path) == before and json.loads(p["map_json"]) == document, "preview changed input"
            component[mode].append({"rule_id":r["id"], "reference_stop":r["rule"].get("stop_line"),
                                   "reviewed_lanes":r["lanes"], "proposal":p["report"]["proposal"]})
    # Do not distribute a copied survey map with this report.
    path.unlink(missing_ok=True)
    save(out / "relation-proposals.json", component)
    result = {"source_commit": source_commit, "native_sha256":digest(Path(core._core.__file__)),
              "source_sha256":source_hash, "operator_inputs_sha256":digest(inputs),
              "reference_sha256":digest(reference), "generation_reference_inputs":[],
              "generation_artifact_sha256":frozen, "road_generation":roads,
              "discovery":{k:discovery[k] for k in ("source_points","windows","detected_candidates","limited","unsupported_windows")},
              "features":compare_features(discovery, surveyed),
              "boundary_distance":boundary_distance(json.loads(final_map.read_text(encoding="utf-8")),surveyed) if final_map else None,
              "relations":{mode:summarize_relations(rows) for mode, rows in component.items()},
              "reference_import_issues":surveyed_payload["report"]["import_issues"],
              "limitations":["Existing development scene, not held-out accuracy.",
               "Paths, lane counts, directions and width priors are explicit operator inputs; no automatic legal semantics.",
               "Source support is also a generation gate; report deferred extent alongside zero flags.",
               "Feature matches are nearby geometry, not semantic classification or precision/recall; references may be incomplete and previews capped.",
               "Association audit uses surveyed geometry and reviewed lanes. Masked mode removes only each query target; other reviewed marking contexts remain.",
               "Masking also removes the query's stop-marking context; missing context is held, not recreated from the answer.",
               "Mixed vehicle/pedestrian groups and distant targets remain held; unsigned normals cannot establish legal control or front face.",
               "Junctions are strict point-supported previews only; no legal connection choices are adopted."]}
    assert digest(cloud) == source_hash and all(digest(out/name)==h for name,h in frozen.items())
    save(out / "evaluation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cloud", "inputs", "reference", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    report = run(args.cloud, args.inputs, args.reference, args.out, args.source_commit)
    print(json.dumps({"relations": report["relations"], "discovery":report["discovery"],
                      "boundary_distance":report["boundary_distance"]}, indent=2))
