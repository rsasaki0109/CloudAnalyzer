"""Post-generation reference audit of an untuned baseline or development scene.

Unmatched proposals are NOT proven false positives: annotations derive from an
incomplete HDMap. Counts are proposal coverage, not semantic detector accuracy.
Never pair raw and labelled LAS by point number (their counts differ).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import laspy
import numpy as np
from pyproj import Transformer
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from hard_intersection_prepare import CHUNK, FirstVoxel
from hard_intersection_generate import save

CLASSES = {"repeated_paint": [41], "bright_bar": [42], "elevated_panel": [51, 52, 53]}


def resample(points, step=.1):
    p = np.asarray(points, dtype=float)
    result = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        if n > 10_000:
            raise ValueError("excessive reference geometry")
        result.extend(a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:])
    return np.asarray(result)


def geometry(candidate):
    e = candidate["evidence"]
    return np.asarray(e["measurement"]["outline"] if e["kind"] == "repeated_paint" else e["geometry"])


def center(candidate):
    p = geometry(candidate)
    c = (p.min(axis=0) + p.max(axis=0)) / 2
    if candidate["evidence"]["kind"] == "elevated_panel":
        c[2] += candidate["evidence"]["height"] / 2
    return c


def gated_assignment(predictions, references, gate=2., surface=False):
    """Maximum-cardinality gated one-to-one matching, then minimum distance."""
    p, r = np.asarray(predictions).reshape(-1, 3), np.asarray(references).reshape(-1, 3)
    if not len(p) or not len(r):
        return []
    delta = p[:, None, :] - r[None, :, :]
    distance = np.linalg.norm(delta[:, :, :2] if surface else delta, axis=2)
    valid = distance <= gate
    if surface:
        valid &= abs(delta[:, :, 2]) <= .75
    # Losing one match must cost more than EVERY possible matched-distance saving.
    penalty = (min(len(p), len(r)) + 1) * (gate + 1)
    cost = np.full((len(p), len(r) + len(p)), penalty)
    cost[:, :len(r)] = np.where(valid, distance, penalty * 3)
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(distance[i, j])) for i, j in zip(rows, cols) if j < len(r) and valid[i, j]]


def read_reference(path):
    root = ET.parse(path).getroot()
    projected = Transformer.from_crs("EPSG:4326", "EPSG:6677", always_xy=True)
    utm = Transformer.from_crs("EPSG:4326", "EPSG:32654", always_xy=True)
    nodes, mgrs_errors = {}, []
    for n in root.findall("node"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in n.findall("tag")}
        lon, lat = float(n.attrib["lon"]), float(n.attrib["lat"])
        nodes[n.attrib["id"]] = [*projected.transform(lon, lat), float(tags["ele"])]
        if "local_x" in tags and "local_y" in tags:
            u = np.mod(utm.transform(lon, lat), 100_000)
            mgrs_errors.append(float(np.linalg.norm(u - [float(tags["local_x"]), float(tags["local_y"])])))
    ways = {}
    refs = {k: [] for k in CLASSES}
    white = []
    for w in root.findall("way"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in w.findall("tag")}
        points = [nodes[n.attrib["ref"]] for n in w.findall("nd")]
        item = {"id": w.attrib["id"], "geometry": points, "tags": tags}
        ways[w.attrib["id"]] = item
        kind = {"stop_line": "bright_bar", "traffic_light": "elevated_panel"}.get(tags.get("type"))
        if kind:
            refs[kind].append(item)
        # Only EXPLICIT white attributes; unlabeled thin lines aren't inferred white.
        if tags.get("type") == "line_thin" and tags.get("color") == "white":
            white.append(item)
    for relation in root.findall("relation"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in relation.findall("tag")}
        if tags.get("type") == "lanelet" and tags.get("subtype") == "crosswalk":
            sides = {m.attrib["role"]: ways[m.attrib["ref"]]["geometry"] for m in relation.findall("member") if m.attrib["type"] == "way" and m.attrib["role"] in {"left", "right"}}
            points = sides["left"] + list(reversed(sides["right"]))
            refs["repeated_paint"].append({"id": relation.attrib["id"], "geometry": points + [points[0]], "tags": tags})
    audit = {"nodes": len(nodes), "local_xy_compared_nodes": len(mgrs_errors),
             "local_xy_vs_utm54_remainder_max_m": max(mgrs_errors) if mgrs_errors else None,
             "reference_transform": "OSM longitude/latitude -> EPSG:6677 (always_xy); no registration fit",
             "local_xy_frame": "UTM54 100km remainders inferred by checking all tagged nodes",
             "generator_frame": "source EPSG:6677; heights unchanged"}
    return refs, white, audit


def reference_center(item, kind):
    p = np.asarray(item["geometry"])
    c = (p.min(axis=0) + p.max(axis=0)) / 2
    if kind == "elevated_panel":
        c[2] += float(item["tags"]["height"]) / 2
    return c


def label_samples(path):
    """Separate per-class streaming passes bound occupancy memory to one bitset.

    Select first original labelled point in each 10cm voxel independently per class.
    This is spatial supervision only, not an index join with source geometry.
    """
    counts = np.zeros(256, dtype=np.int64)
    samples = {}
    codes = [21, 22, 41, 42, 51, 52, 53]
    for ci, code in enumerate(codes):
        pieces, total = [], 0
        with laspy.open(path) as reader:
            seen = FirstVoxel(reader.header.mins, reader.header.maxs)
            for chunk in reader.chunk_iterator(CHUNK):
                labels = np.asarray(chunk.user_data)
                if ci == 0:
                    counts += np.bincount(labels, minlength=256)
                mask = labels == code
                xyz = np.column_stack([chunk.x[mask], chunk.y[mask], chunk.z[mask]])
                selected = xyz[seen.retain(xyz)]
                total += len(selected)
                if total > 2_000_000:
                    raise ValueError("label representatives exceed memory cap")
                pieces.append(selected)
        samples[code] = np.concatenate(pieces) if pieces else np.empty((0, 3))
        print(f"UserData {code}: {counts[code]:,} annotated points, {total:,} spatial samples", flush=True)
        del seen
    return samples, {str(i): int(c) for i, c in enumerate(counts) if c}


def symmetric_error(prediction, reference):
    p, r = resample(prediction), resample(reference)
    a, b = cKDTree(r).query(p)[0], cKDTree(p).query(r)[0]
    return {"symmetric_mean_m": float((a.mean() + b.mean()) / 2),
            "hausdorff_m": float(max(a.max(), b.max()))}


def evaluate(dataset: Path, generated: Path, out: Path, development: bool = False):
    genfile = generated / "generation.json"
    expected = (generated / "generation.sha256").read_text(encoding="utf-8").strip()
    if hashlib.sha256(genfile.read_bytes()).hexdigest() != expected:
        raise ValueError("generation changed after freezing; run a new baseline")
    gen = json.loads(genfile.read_text(encoding="utf-8"))
    for drive in gen["drives"]:
        if "map" in drive and hashlib.sha256((generated / drive["map"]).read_bytes()).hexdigest() != drive["map_sha256"]:
            raise ValueError("road map changed after freezing; run a new baseline")
    refs, white, audit = read_reference(dataset / "maps/lanelet2/jp_tokyo_takanawadai.osm")
    samples, counts = label_samples(dataset / "annotation/semantic_pointcloud/jp_tokyo_takanawadai_class.las")
    with laspy.open(dataset / "pointcloud/jp_tokyo_takanawadai.las") as reader:
        lo, hi, raw_count = reader.header.mins, reader.header.maxs, reader.header.point_count
    report = {"source": json.loads((dataset / "manifest.json").read_text(encoding="utf-8")),
              "baseline_commit": gen["baseline_commit"], "generation_sha256": expected, "coordinate_audit": audit,
              "runtime": gen["runtime"],
              "evaluation_role": "development_scene" if development else "untuned_baseline",
              "raw_points": raw_count, "annotated_points": sum(counts.values()), "index_join": False,
              "user_data_counts": counts, "instance_gate_m": 2., "surface_z_gate_m": .75,
              "limitations": ["one intersection, no independent generalization estimate", "annotations derived from HDMap and incomplete",
                              "unmatched proposals are not proven negatives", "geometry proposals, no automatic semantic classification",
                              "no training, ground-truth registration or reference geometry supplied to generation",
                              "detector developed using this scene; not independent held-out generalization" if development else "untuned original baseline",
                              "10cm original-point decimation; fixed tile halos can still cause proposal differences",
                              "crosswalk outline is observed paint extent, reference is mapped crossing footprint"],
              "instances": {}, "road_drafts": []}
    for kind, codes in CLASSES.items():
        predictions = [c for c in gen["candidates"] if c["evidence"]["kind"] == kind]
        reference = [r for r in refs[kind] if np.all(reference_center(r, kind)[:2] >= lo[:2]) and np.all(reference_center(r, kind)[:2] <= hi[:2])]
        pairs = gated_assignment([center(p) for p in predictions], [reference_center(r, kind) for r in reference], surface=kind != "elevated_panel")
        labels = np.concatenate([samples[code] for code in codes])
        tree = cKDTree(labels) if len(labels) else None
        proposal_curves = [resample(geometry(p)) for p in predictions]
        support_audit = [{"tile": p["tile"], "id": p["id"],
                          "annotated_class_near_geometry_fraction_25cm": float(np.mean(tree.query(curve)[0] <= .25)) if tree else None}
                         for p, curve in zip(predictions, proposal_curves)]
        coverage = float(np.mean(cKDTree(np.concatenate(proposal_curves)).query(labels)[0] <= .35)) if proposal_curves and len(labels) else 0.
        matched = []
        for pi, ri, distance in pairs:
            p, r = predictions[pi], reference[ri]
            error = symmetric_error(geometry(p), r["geometry"])
            support = float(np.mean(tree.query(resample(geometry(p)))[0] <= .25)) if tree else None
            item = {"proposal": {"tile": p["tile"], "id": p["id"]}, "reference": r["id"],
                    "center_distance_m": distance, "geometry": error, "annotated_class_near_geometry_fraction_25cm": support}
            if kind == "elevated_panel":
                item["height_error_m"] = abs(p["evidence"]["height"] - float(r["tags"]["height"]))
            matched.append(item)
        used_p, used_r = {p for p, _, _ in pairs}, {r for _, r, _ in pairs}
        report["instances"][kind] = {"reference_total": len(refs[kind]), "reference_in_source_extent": len(reference),
            "owned_proposals": len(predictions), "matched": len(pairs), "reference_misses": [r["id"] for i, r in enumerate(reference) if i not in used_r],
            "unmatched_proposals": [{"tile": p["tile"], "id": p["id"]} for i, p in enumerate(predictions) if i not in used_p], "matches": matched,
            "proposal_annotation_proximity": support_audit, "annotated_class_spatial_sample_near_proposal_geometry_fraction_35cm": coverage,
            "proximity_meaning": "distance to sampled outline/bottom edge; not area overlap or semantic accuracy"}
    white_samples = np.concatenate([resample(w["geometry"]) for w in white])
    white_total = len(white_samples)
    white_samples = white_samples[np.all(white_samples[:, :2] >= lo[:2], axis=1) & np.all(white_samples[:, :2] <= hi[:2], axis=1)]
    report["white_reference"] = {"explicit_white_ways": len(white), "samples_total": white_total,
                                 "samples_in_source_xy_extent": len(white_samples), "sampling_step_m": .1}
    white_tree = cKDTree(white_samples)
    white_labels = np.concatenate([samples[21], samples[22]])
    for drive in gen["drives"]:
        entry = {"recording": drive["recording"], "trajectory": drive["path"]}
        if "error" in drive:
            entry["error"] = drive["error"]
        else:
            data = json.loads((generated / drive["map"]).read_text(encoding="utf-8"))
            curves = [resample(b["geometry"]) for b in data["boundaries"]]
            vertices = np.concatenate(curves)
            dist = white_tree.query(vertices)[0]
            truth_dist = cKDTree(vertices).query(white_samples)[0]
            label_dist = cKDTree(vertices).query(white_labels)[0]
            entry.update({"lanes": len(data["lanes"]), "extraction": drive["report"]["extraction"],
                          "boundary_distance_to_explicit_white_reference": {"median_m": float(np.median(dist)), "p90_m": float(np.quantile(dist, .9))},
                          "explicit_white_reference_length_sample_coverage_35cm": float(np.mean(truth_dist <= .35)),
                          "white_label_spatial_sample_coverage_35cm": float(np.mean(label_dist <= .35)),
                          "meaning": "geometry agreement only; each drive has lane-count/width priors, no white-line semantic classifier"})
        report["road_drafts"].append(entry)
    report["tile_status"] = [{k: v for k, v in t.items() if k in {"path", "error", "limited", "windows", "unsupported_windows", "owned"}} for t in gen["tiles"]]
    out.mkdir(exist_ok=False)
    save(out / "evaluation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("generated", type=Path)
    parser.add_argument("output", type=Path, help="NEW directory")
    parser.add_argument("--development", action="store_true", help="Mark a scene used to develop the detector, not an independent held-out test")
    args = parser.parse_args()
    evaluate(args.dataset, args.generated, args.output, args.development)
