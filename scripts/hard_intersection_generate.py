"""Run the fixed native baseline on prepared RAW geometry and recorded drives only.

No OSM, annotation LAS, semantic images, or semantic class fields are inputs.
Proposals are retained for evaluation, never auto-confirmed as traffic objects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import importlib.metadata
import platform
import time
from pathlib import Path

import numpy as np

BASELINE = "3c93d90996b071c78518d54ec21cb6d570484778"
OPTIONS = {"scope": "ground_surface", "corridor_radius": 12., "brightness_fraction": .65}


def owns(tile, candidate):
    center = (np.asarray(candidate["min"]) + candidate["max"]) / 2
    return bool(np.all(center[:2] >= tile["core_min"]) and np.all(center[:2] < tile["core_max"]))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def generate(prepared: Path, out: Path):
    import cloudanalyzer_core as core
    out.mkdir(exist_ok=False)
    prep = json.loads((prepared / "preparation.json").read_text(encoding="utf-8"))
    binaries = list(Path(core.__file__).parent.glob("*.pyd")) + list(Path(core.__file__).parent.glob("*.so"))
    report = {"baseline_commit": BASELINE, "options": OPTIONS, "reference_inputs": [],
              "semantic_classification": "none; geometry proposals require review",
              "preparation": prep, "tiles": [], "candidates": [], "drives": []}
    report["runtime"] = {"python": platform.python_version(), "numpy": np.__version__,
                         "core_version": importlib.metadata.version("cloudanalyzer-core"),
                         "native_binaries": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in binaries],
                         "note": "source commit is the intended baseline; installed binary provenance must be verified by its builder"}
    start = time.perf_counter()
    for tile in prep["tiles"]:
        try:
            payload = json.loads(core.discover_vector_map_features(str(prepared / tile["path"]), None, json.dumps(OPTIONS), None))
            discovery = payload["report"]["discovery"]
            save(out / (tile["path"] + ".json"), discovery)
            owned = [dict(c, tile=tile["path"]) for c in discovery["candidates"] if owns(tile, c)]
            report["candidates"].extend(owned)
            report["tiles"].append({"path": tile["path"], **{k: v for k, v in discovery.items() if k != "candidates"}, "owned": len(owned)})
            print(f"{tile['path']}: {len(owned)} owned / {len(discovery['candidates'])} preview candidates", flush=True)
        except ValueError as e:
            report["tiles"].append({"path": tile["path"], "error": str(e)})
            print(f"{tile['path']}: unsupported: {e}", flush=True)
    # Each recording is evaluated separately: no merging or truth-derived route selection.
    for drive in prep["drives"]:
        entry = dict(drive)
        try:
            payload = json.loads(core.build_vector_map(str(prepared / "geometry.las"), str(prepared / drive["path"]), "{}", None, None, None))
            save(out / (drive["path"] + ".report.json"), payload["report"])
            name = drive["path"] + ".map.json"
            (out / name).write_text(payload["map_json"], encoding="utf-8")
            (out / (drive["path"] + ".osm")).write_text(payload["osm"], encoding="utf-8")
            entry.update(map=name, map_sha256=hashlib.sha256((out / name).read_bytes()).hexdigest(), report=payload["report"])
            print(f"{drive['path']}: generated road draft", flush=True)
        except ValueError as e:
            entry["error"] = str(e)
            print(f"{drive['path']}: {e}", flush=True)
        report["drives"].append(entry)
    report["seconds"] = time.perf_counter() - start
    save(out / "generation.json", report)
    digest = hashlib.sha256((out / "generation.json").read_bytes()).hexdigest()
    (out / "generation.sha256").write_text(digest + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("output", type=Path, help="NEW directory")
    args = parser.parse_args()
    generate(args.prepared, args.output)
