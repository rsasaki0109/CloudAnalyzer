"""Compare complete resident MapEval with symmetric grouped covariance sums."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import struct
import sys
import time


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def info(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def report_sha(report):
    return hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source-root", "reference", "compared", "input-manifest", "baseline", "work-dir", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--structure-voxel-size", type=float, default=3.)
    options = [value.split("=")[0] for value in sys.argv[1:] if value.startswith("--")]
    require(len(options) == len(set(options)), "duplicate options")
    args = parser.parse_args()
    require(1 <= args.repeats <= 100, "invalid repeats")
    require(math.isfinite(args.structure_voxel_size) and args.structure_voxel_size >= 0, "invalid structure voxel size")
    require(not args.output.exists() and not args.work_dir.exists(), "output/work directory must be new")
    require(sha(args.baseline) == args.baseline_sha256, "baseline differs")
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.path.insert(0, str((args.source_root / "cloudanalyzer").resolve()))
    import numpy as np
    import scipy
    import scipy.spatial._ckdtree as tree_binary
    import open3d as o3d
    import ca.core.map_evaluate as current
    from ca.io import load_point_cloud
    spec = importlib.util.spec_from_file_location("map_covariance_scatter_baseline", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    for key in ["reference", "compared"]:
        require(sha(getattr(args, key)) == manifest["inputs"][key]["sha256"], key + " differs")
    original = (args.source_root / "web" / manifest["source"]["path"]).resolve()
    require(sha(original) == manifest["source"]["sha256"], "original source differs")
    inputs = {name: info(getattr(args, name)) for name in ["reference", "compared", "input_manifest", "baseline"]}
    inputs["originalSource"] = info(original)
    compiled_tree = info(tree_binary.__file__)
    files = ["cloudanalyzer/ca/core/map_evaluate.py", "cloudanalyzer/ca/io.py", "cloudanalyzer/ca/_rust.py",
             "cloudanalyzer/pyproject.toml", "cloudanalyzer/tests/test_map_nn_parallel.py", "cloudanalyzer/tests/test_map_scs_index.py", "cloudanalyzer/tests/test_map_covariance.py",
             "cloudanalyzer/scripts/benchmark_map_covariance.py"]
    hashes = {path: sha(args.source_root / path) for path in files}
    source_cloud, reference_cloud = load_point_cloud(str(args.compared)), load_point_cloud(str(args.reference))
    source, reference = np.asarray(source_cloud.points), np.asarray(reference_cloud.points)
    coordinate_hashes = {name: hashlib.sha256(array.astype("<f8", copy=False).tobytes()).hexdigest()
                         for name, array in [("compared", source), ("reference", reference)]}
    args.work_dir.mkdir(parents=True, exist_ok=False)

    def call(policy, capture=False):
        module = baseline if policy == "scatter" else current
        nn = module._min_distances_kdtree
        arrays, artifacts, scs_indexes, gaussians = [], {}, [], []
        build_gaussians = module.build_voxel_gaussians
        prepare = getattr(module, "_scs_spatial_index", None)
        if capture:
            def traced(a, b):
                values = nn(a, b)
                path = args.work_dir / (policy + "-distance-" + str(len(arrays)) + ".bin")
                with path.open("xb") as stream:
                    stream.write(np.asarray(values, dtype="<f8").tobytes())
                arrays.append({**info(path), "points": len(values)})
                return values

            def traced_gaussians(points, size):
                voxels = build_gaussians(points, size)
                path = args.work_dir / (policy + "-gaussians-" + str(len(gaussians)) + ".bin")
                packer = struct.Struct("<4q12d")
                with path.open("xb") as stream:
                    for key, voxel in voxels.items():
                        stream.write(packer.pack(*key, voxel.num_points, *voxel.mu, *voxel.sigma.ravel()))
                gaussians.append({**info(path), "voxels": len(voxels), "recordBytes": packer.size})
                return voxels

            module.build_voxel_gaussians = traced_gaussians
            module._min_distances_kdtree = traced
            if prepare is not None:
                def traced_prepare(distances, radius):
                    prepared = prepare(distances, radius)
                    if prepared is not None:
                        tree = prepared[0]
                        scs_indexes.append({"cells": len(distances), "radius": radius,
                            "coordinateBytes": tree.data.nbytes, "indexBytes": tree.indices.nbytes, "nativeNodes": tree.size})
                    return prepared

                module._scs_spatial_index = traced_prepare
        protocol = module.MapEvalProtocol(voxel_size_m=args.structure_voxel_size) if args.structure_voxel_size else None
        request = module.MapEvaluateRequest(estimated_points=source, reference_points=reference,
                    structure_voxel_size=args.structure_voxel_size, map_eval_protocol=protocol,
                    artifact_dir=str(args.work_dir / policy) if capture else None)
        start = time.perf_counter_ns()
        try:
            result = module.evaluate_map(request)
        finally:
            module._min_distances_kdtree = nn
            module.build_voxel_gaussians = build_gaussians
            if prepare is not None:
                module._scs_spatial_index = prepare
        millis = (time.perf_counter_ns() - start) / 1e6
        report = asdict(result)
        for key, path in report["artifacts"].items():
            if key.endswith("_ply"):
                artifacts[key] = {**info(path), "file": Path(path).name}
        for key, metadata in artifacts.items():
            report["artifacts"][key] = metadata["file"]
        return {"millis": millis, "report": report, "reportSha256": report_sha(report),
                "distances": arrays, "artifacts": artifacts, "scsIndexes": scs_indexes, "gaussians": gaussians}

    parity = {policy: call(policy, True) for policy in ["scatter", "grouped"]}
    require(parity["scatter"]["report"] == parity["grouped"]["report"], "complete reports differ")
    require(len(parity["scatter"]["distances"]) == len(parity["grouped"]["distances"]) == 2, "unexpected distance count")
    for a, b in zip(parity["scatter"]["distances"], parity["grouped"]["distances"], strict=True):
        require(a["sha256"] == b["sha256"] and a["points"] == b["points"], "full f64 distances differ")
    require(len(parity["scatter"]["artifacts"]) == len(parity["grouped"]["artifacts"]) == 2, "unexpected artifact count")
    for key, a in parity["scatter"]["artifacts"].items():
        b = parity["grouped"]["artifacts"][key]
        require((a["sha256"], a["bytes"]) == (b["sha256"], b["bytes"]), "complete PLY bytes differ")
    require(parity["scatter"]["scsIndexes"] == parity["grouped"]["scsIndexes"], "SCS index differs")
    require(len(parity["scatter"]["gaussians"]) == len(parity["grouped"]["gaussians"]) == 2 * int(args.structure_voxel_size > 0), "Gaussian count differs")
    for a, b in zip(parity["scatter"]["gaussians"], parity["grouped"]["gaussians"], strict=True):
        require((a["sha256"], a["bytes"], a["voxels"]) == (b["sha256"], b["bytes"], b["voxels"]), "full voxel Gaussian bits differ")
    expected_indexes = int(args.structure_voxel_size > 0 and parity["grouped"]["report"]["metrics"].get("n_awd_voxels", 0) >= current._SCS_INDEX_MIN_VOXELS)
    require(len(parity["grouped"]["scsIndexes"]) == expected_indexes, "unexpected SCS index count")
    first = {policy: call(policy) for policy in ["scatter", "grouped"]}
    expected = first["scatter"]["report"]
    require(first["grouped"]["report"] == expected, "first reports differ")
    no_ply_report = {**parity["scatter"]["report"], "artifacts": {key: value for key, value in
                     parity["scatter"]["report"]["artifacts"].items() if not key.endswith("_ply")}}
    require(expected == no_ply_report, "artifact choice changed metrics")

    def pair(index):
        order = ["scatter", "grouped"] if index % 2 == 0 else ["grouped", "scatter"]
        observed = {policy: call(policy) for policy in order}
        require(all(item["report"] == expected for item in observed.values()), "timed report differs")
        return {"order": order, "millis": {key: value["millis"] for key, value in observed.items()},
                "reportHashes": {key: value["reportSha256"] for key, value in observed.items()}}

    print(json.dumps({"stage": "parity_complete", "source": manifest["source"]["path"]}), flush=True)
    warmups = [pair(index) for index in range(2)]
    trials = [pair(index + 2) for index in range(args.repeats)]
    a, b = [item["millis"]["scatter"] for item in trials], [item["millis"]["grouped"] for item in trials]
    summary = {"scatterMedianMillis": statistics.median(a), "groupedMedianMillis": statistics.median(b),
               "pairedShorteningPercent": statistics.median([100 * (1 - y / x) for x, y in zip(a, b)]),
               "pairedSavedMedianMillis": statistics.median([x - y for x, y in zip(a, b)]), "fasterTrials": sum(x > y for x, y in zip(a, b))}
    for path, expected_hash in hashes.items():
        require(sha(args.source_root / path) == expected_hash, "source changed: " + path)
    for item in [*inputs.values(), compiled_tree]:
        require(sha(item["path"]) == item["sha256"], "input/library changed")
    for name, array in [("compared", source), ("reference", reference)]:
        require(hashlib.sha256(array.astype("<f8", copy=False).tobytes()).hexdigest() == coordinate_hashes[name], "input coordinates changed")
    report = {"schemaVersion": 1, "comparison": "map-covariance", "createdAt": datetime.now(timezone.utc).isoformat(),
              "sourceHashes": hashes, "inputs": inputs, "inputManifest": manifest, "coordinateHashes": coordinate_hashes,
              "compiledTree": compiled_tree, "environment": {"python": platform.python_version(), "numpy": np.__version__,
                 "scipy": scipy.__version__, "open3d": o3d.__version__, "platform": platform.system(), "logicalCpus": os.cpu_count(),
                 "ompNumThreads": os.environ["OMP_NUM_THREADS"], "openblasNumThreads": os.environ["OPENBLAS_NUM_THREADS"]},
              "policy": {"nnWorkers": -1, "nnParallelMinPoints": current._PARALLEL_NN_MIN_POINTS, "scsIndexMinVoxels": current._SCS_INDEX_MIN_VOXELS, "scsExactSpan": current._SCS_INDEX_EXACT_SPAN},
              "repeats": args.repeats, "warmupRounds": 2, "structureVoxelSize": args.structure_voxel_size,
              "parity": parity, "firstUse": first, "warmups": warmups, "trials": trials, "summary": summary,
              "accuracy": {"fullF64ArraysChecked": 4, "plyOutputsChecked": 4, "gaussianMapsChecked": 4 * int(args.structure_voxel_size > 0), "reportsChecked": 2 * (2 + 2 + args.repeats)},
              "note": "Control already includes parallel NN and sparse integer SCS lookup. Complete resident MapEval includes "
                      "fresh point trees/queries, thresholds, voxel grouping, covariance, Wasserstein and SCS including index "
                      "construction/destruction. Grouped covariance retains point-order f64 sums and computes six symmetric "
                      "components. Nonfinite/extreme centered values retain scatter. Source sampling/coordinates are fixed; "
                      "no generated data. Loading, imports, hashing, Gaussian/f64/PLY parity tracing and CLI writes are outside "
                      "clocks. Two warm-up pairs and an excluded first pair precede alternating timed pairs. No outliers removed. "
                      "No persistent cache, GPU or streaming speed claim. Array payload sizes are not peak RSS."}


    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
