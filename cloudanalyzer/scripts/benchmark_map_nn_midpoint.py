"""Compare complete resident MapEval with compact midpoint point trees."""

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
    spec = importlib.util.spec_from_file_location("map_nn_balanced_baseline", args.baseline)
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
    files = ['cloudanalyzer/ca/core/map_evaluate.py', 'cloudanalyzer/ca/io.py', 'cloudanalyzer/ca/_rust.py', 'cloudanalyzer/pyproject.toml', 'cloudanalyzer/tests/test_map_nn_parallel.py', 'cloudanalyzer/tests/test_map_scs_index.py', 'cloudanalyzer/tests/test_map_covariance.py', 'cloudanalyzer/tests/test_map_grouping.py', 'cloudanalyzer/tests/test_map_stream_groups.py', 'cloudanalyzer/tests/test_map_downsample_centroids.py', 'cloudanalyzer/tests/test_map_downsample_small_groups.py', 'cloudanalyzer/tests/test_map_mean_sums.py', 'cloudanalyzer/tests/test_map_wasserstein_batch.py', 'cloudanalyzer/tests/test_map_scs_batch.py', 'cloudanalyzer/tests/test_map_nn_midpoint.py', 'cloudanalyzer/scripts/benchmark_map_nn_midpoint.py']
    hashes = {path: sha(args.source_root / path) for path in files}
    source_cloud, reference_cloud = load_point_cloud(str(args.compared)), load_point_cloud(str(args.reference))
    source, reference = np.asarray(source_cloud.points), np.asarray(reference_cloud.points)
    coordinate_hashes = {name: hashlib.sha256(array.astype("<f8", copy=False).tobytes()).hexdigest()
                         for name, array in [("compared", source), ("reference", reference)]}
    args.work_dir.mkdir(parents=True, exist_ok=False)

    def call(policy, capture=False):
        module = baseline if policy == "balanced" else current
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

    parity = {policy: call(policy, True) for policy in ["balanced", "midpoint"]}
    require(parity["balanced"]["report"] == parity["midpoint"]["report"], "complete reports differ")
    require(len(parity["balanced"]["distances"]) == len(parity["midpoint"]["distances"]) == 2, "unexpected distance count")
    for a, b in zip(parity["balanced"]["distances"], parity["midpoint"]["distances"], strict=True):
        require(a["sha256"] == b["sha256"] and a["points"] == b["points"], "full f64 distances differ")
    require(len(parity["balanced"]["artifacts"]) == len(parity["midpoint"]["artifacts"]) == 2, "unexpected artifact count")
    for key, a in parity["balanced"]["artifacts"].items():
        b = parity["midpoint"]["artifacts"][key]
        require((a["sha256"], a["bytes"]) == (b["sha256"], b["bytes"]), "complete PLY bytes differ")
    require(parity["balanced"]["scsIndexes"] == parity["midpoint"]["scsIndexes"], "SCS index differs")
    require(len(parity["balanced"]["gaussians"]) == len(parity["midpoint"]["gaussians"]) == 2 * int(args.structure_voxel_size > 0), "Gaussian count differs")
    for a, b in zip(parity["balanced"]["gaussians"], parity["midpoint"]["gaussians"], strict=True):
        require((a["sha256"], a["bytes"], a["voxels"]) == (b["sha256"], b["bytes"], b["voxels"]), "full voxel Gaussian bits differ")
    expected_indexes = int(args.structure_voxel_size > 0 and parity["midpoint"]["report"]["metrics"].get("n_awd_voxels", 0) >= current._SCS_INDEX_MIN_VOXELS)
    require(len(parity["midpoint"]["scsIndexes"]) == expected_indexes, "unexpected SCS index count")
    first = {policy: call(policy) for policy in ["balanced", "midpoint"]}
    expected = first["balanced"]["report"]
    require(first["midpoint"]["report"] == expected, "first reports differ")
    no_ply_report = {**parity["balanced"]["report"], "artifacts": {key: value for key, value in
                     parity["balanced"]["report"]["artifacts"].items() if not key.endswith("_ply")}}
    require(expected == no_ply_report, "artifact choice changed metrics")

    def pair(index):
        order = ["balanced", "midpoint"] if index % 2 == 0 else ["midpoint", "balanced"]
        observed = {policy: call(policy) for policy in order}
        require(all(item["report"] == expected for item in observed.values()), "timed report differs")
        return {"order": order, "millis": {key: value["millis"] for key, value in observed.items()},
                "reportHashes": {key: value["reportSha256"] for key, value in observed.items()}}

    print(json.dumps({"stage": "parity_complete", "source": manifest["source"]["path"]}), flush=True)
    warmups = [pair(index) for index in range(2)]
    trials = [pair(index + 2) for index in range(args.repeats)]
    a, b = [item["millis"]["balanced"] for item in trials], [item["millis"]["midpoint"] for item in trials]
    summary = {"balancedMedianMillis": statistics.median(a), "midpointMedianMillis": statistics.median(b),
               "pairedShorteningPercent": statistics.median([100 * (1 - y / x) for x, y in zip(a, b)]),
               "pairedSavedMedianMillis": statistics.median([x - y for x, y in zip(a, b)]), "fasterTrials": sum(x > y for x, y in zip(a, b))}
    reducer = None
    if args.structure_voxel_size > 0:
        dtype = np.dtype([("key", "<i8", (3,)), ("count", "<i8"), ("mu", "<f8", (3,)), ("sigma", "<f8", (3, 3))])
        gaussian_maps = []
        for metadata in parity["balanced"]["gaussians"]:
            rows = np.fromfile(metadata["path"], dtype=dtype)
            gaussian_maps.append({tuple(int(axis) for axis in row["key"]): baseline.VoxelGaussian(
                row["mu"].copy(), row["sigma"].copy(), int(row["count"])) for row in rows})
        estimated_map, reference_map = gaussian_maps
        distance_captures = {}
        packer = struct.Struct("<3qd")
        for policy, module in [("balanced", baseline), ("midpoint", current)]:
            values = dict(module._voxel_wasserstein_distances(estimated_map, reference_map, 100))
            path = args.work_dir / (policy + "-voxel-distances.bin")
            with path.open("xb") as stream:
                for key, value in values.items():
                    stream.write(packer.pack(*key, value))
            distance_captures[policy] = {**info(path), "voxels": len(values), "recordBytes": packer.size}
        require(distance_captures["balanced"]["sha256"] == distance_captures["midpoint"]["sha256"], "Gaussian distances differ")
        distances = values
        centers = [(key, len(row)) for key, row in zip(distances, baseline._scs_neighbor_values(distances, 5), strict=True) if row]
        def balanced_terms():
            return np.asarray(list(baseline._scs_terms(distances, 5)), dtype=np.float64)
        def midpoint_terms():
            return np.asarray(list(current._scs_terms(distances, 5)), dtype=np.float64)
        functions = {"balanced": balanced_terms, "midpoint": midpoint_terms}
        term_packer = struct.Struct("<3qQd")
        def term_bytes(terms):
            return b"".join(term_packer.pack(*key, count, value) for (key, count), value in zip(centers, terms, strict=True))
        captures = {}
        for policy, function in functions.items():
            terms = function()
            path = args.work_dir / (policy + "-scs-terms.bin")
            with path.open("xb") as stream:
                stream.write(term_bytes(terms))
            captures[policy] = {**info(path), "centers": len(terms), "recordBytes": term_packer.size}
        require(captures["balanced"]["sha256"] == captures["midpoint"]["sha256"], "complete SCS terms differ")
        require(float(np.nanmean(terms)) == parity["balanced"]["report"]["metrics"]["scs"], "captured SCS differs from full result")
        reducer = {"distances": distance_captures, "parity": captures}
    def nn_pair(index):
        order = ["balanced", "midpoint"] if index % 2 == 0 else ["midpoint", "balanced"]
        observed = {}
        for policy in order:
            module = baseline if policy == "balanced" else current
            start = time.perf_counter_ns()
            values = [module._min_distances_kdtree(source, reference), module._min_distances_kdtree(reference, source)]
            observed[policy] = (time.perf_counter_ns() - start) / 1e6
            for values_array, metadata in zip(values, parity[policy]["distances"], strict=True):
                require(hashlib.sha256(values_array.astype("<f8", copy=False).tobytes()).hexdigest() == metadata["sha256"], "NN distances changed")
            del values, values_array
        return {"order": order, "millis": observed}
    nn_first = nn_pair(0)
    nn_warmups = [nn_pair(index) for index in range(2)]
    nn_trials = [nn_pair(index + 2) for index in range(args.repeats)]
    a, b = [p["millis"]["balanced"] for p in nn_trials], [p["millis"]["midpoint"] for p in nn_trials]
    nn_reducer = {"firstUse": nn_first, "warmups": nn_warmups, "trials": nn_trials,
        "summary": {"balancedMedianMillis": statistics.median(a), "midpointMedianMillis": statistics.median(b),
            "pairedShorteningPercent": statistics.median([100 * (1 - y / x) for x, y in zip(a, b)]),
            "pairedSavedMedianMillis": statistics.median([x - y for x, y in zip(a, b)]), "fasterTrials": sum(x > y for x, y in zip(a, b))}}
    for path, expected_hash in hashes.items():
        require(sha(args.source_root / path) == expected_hash, "source changed: " + path)
    for item in [*inputs.values(), compiled_tree]:
        require(sha(item["path"]) == item["sha256"], "input/library changed")
    for name, array in [("compared", source), ("reference", reference)]:
        require(hashlib.sha256(array.astype("<f8", copy=False).tobytes()).hexdigest() == coordinate_hashes[name], "input coordinates changed")
    report = {"schemaVersion": 1, "comparison": "map-nn-midpoint", "createdAt": datetime.now(timezone.utc).isoformat(),
              "sourceHashes": hashes, "inputs": inputs, "inputManifest": manifest, "coordinateHashes": coordinate_hashes,
              "compiledTree": compiled_tree, "environment": {"python": platform.python_version(), "numpy": np.__version__,
                 "scipy": scipy.__version__, "open3d": o3d.__version__, "platform": platform.system(), "logicalCpus": os.cpu_count(),
                 "ompNumThreads": os.environ["OMP_NUM_THREADS"], "openblasNumThreads": os.environ["OPENBLAS_NUM_THREADS"]},
              "policy": {"nnWorkers": -1, "nnParallelMinPoints": current._PARALLEL_NN_MIN_POINTS, "scsIndexMinVoxels": current._SCS_INDEX_MIN_VOXELS, "scsExactSpan": current._SCS_INDEX_EXACT_SPAN, "meanBincountMinPoints": current._VOXEL_MEAN_BINCOUNT_MIN_POINTS,
                         "wassersteinBatchSize": current._WASSERSTEIN_BATCH_SIZE,
                         "wassersteinBatchMinPairs": current._WASSERSTEIN_BATCH_MIN_PAIRS,
                         "wassersteinSafeMagnitude": current._WASSERSTEIN_BATCH_SAFE_MAGNITUDE, "numpyErrorState": np.geterr(), "scsTermBatchMinVoxels": current._SCS_TERM_BATCH_MIN_VOXELS,
                         "scsTermBatchMinCenters": current._SCS_TERM_BATCH_MIN_CENTERS,
                         "scsTermBatchMinSameSize": current._SCS_TERM_BATCH_MIN_SAME_SIZE,
                         "scsTermBatchMaxCenters": current._SCS_TERM_BATCH_MAX_CENTERS,
                         "scsTermBatchMaxValues": current._SCS_TERM_BATCH_MAX_VALUES, "nnMidpointMinPoints": current._MIDPOINT_NN_MIN_POINTS, "nnLeafSize": 16, "nnCompactNodes": True},
              "repeats": args.repeats, "warmupRounds": 2, "structureVoxelSize": args.structure_voxel_size,
              "structuralParity": reducer, "nnReducer": nn_reducer, "parity": parity, "firstUse": first, "warmups": warmups, "trials": trials, "summary": summary,
              "accuracy": {"fullF64ArraysChecked": 4, "plyOutputsChecked": 4, "gaussianMapsChecked": 4 * int(args.structure_voxel_size > 0), "reportsChecked": 2 * (2 + 2 + args.repeats), "fullGaussianDistanceArraysChecked": 2 * int(args.structure_voxel_size > 0), "fullScsTermArraysChecked": 2 * int(args.structure_voxel_size > 0)},
              "note": "Control already includes all preceding MapEval optimizations, including batched SCS statistics. "
                      "Current uses balanced_tree=False for exact point NN trees only when both maps have at least "
                      "65536 points. Compact bounds, leaf size 16, query parameters and the existing worker threshold "
                      "remain. Smaller/asymmetric workloads retain preceding tree construction. Complete resident "
                      "clocks include fresh NN tree setup/query/destruction, thresholds, grouping, means/covariances, "
                      "Wasserstein and SCS/index setup/destruction. Alignment/downsampling are disabled; original "
                      "coordinates/sampling stay fixed. Loading/imports/hashing/Gaussian/f64/PLY parity tracing and CLI "
                      "writes are outside clocks. Two warm-ups and an excluded first pair precede five default alternating "
                      "timed pairs. All observations remain. Separate nnReducer clocks include both fresh exact point "
                      "trees/queries/destruction and output arrays, exclude statistics/structural work and checking/hashing. "
                      "Structural parity is captured without an unchanged-component speed comparison. No persistent "
                      "reference cache, streaming speed, GPU or peak-RSS claim."}


    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
