"""Compare complete resident MapEval with bounded Gaussian Wasserstein batches."""

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
    spec = importlib.util.spec_from_file_location("map_wasserstein_scalar_baseline", args.baseline)
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
    files = ['cloudanalyzer/ca/core/map_evaluate.py', 'cloudanalyzer/ca/io.py', 'cloudanalyzer/ca/_rust.py', 'cloudanalyzer/pyproject.toml', 'cloudanalyzer/tests/test_map_nn_parallel.py', 'cloudanalyzer/tests/test_map_scs_index.py', 'cloudanalyzer/tests/test_map_covariance.py', 'cloudanalyzer/tests/test_map_grouping.py', 'cloudanalyzer/tests/test_map_stream_groups.py', 'cloudanalyzer/tests/test_map_downsample_centroids.py', 'cloudanalyzer/tests/test_map_downsample_small_groups.py', 'cloudanalyzer/tests/test_map_mean_sums.py', 'cloudanalyzer/tests/test_map_wasserstein_batch.py', 'cloudanalyzer/scripts/benchmark_map_wasserstein_batch.py']
    hashes = {path: sha(args.source_root / path) for path in files}
    source_cloud, reference_cloud = load_point_cloud(str(args.compared)), load_point_cloud(str(args.reference))
    source, reference = np.asarray(source_cloud.points), np.asarray(reference_cloud.points)
    coordinate_hashes = {name: hashlib.sha256(array.astype("<f8", copy=False).tobytes()).hexdigest()
                         for name, array in [("compared", source), ("reference", reference)]}
    args.work_dir.mkdir(parents=True, exist_ok=False)

    def call(policy, capture=False):
        module = baseline if policy == "scalar" else current
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

    parity = {policy: call(policy, True) for policy in ["scalar", "batched"]}
    require(parity["scalar"]["report"] == parity["batched"]["report"], "complete reports differ")
    require(len(parity["scalar"]["distances"]) == len(parity["batched"]["distances"]) == 2, "unexpected distance count")
    for a, b in zip(parity["scalar"]["distances"], parity["batched"]["distances"], strict=True):
        require(a["sha256"] == b["sha256"] and a["points"] == b["points"], "full f64 distances differ")
    require(len(parity["scalar"]["artifacts"]) == len(parity["batched"]["artifacts"]) == 2, "unexpected artifact count")
    for key, a in parity["scalar"]["artifacts"].items():
        b = parity["batched"]["artifacts"][key]
        require((a["sha256"], a["bytes"]) == (b["sha256"], b["bytes"]), "complete PLY bytes differ")
    require(parity["scalar"]["scsIndexes"] == parity["batched"]["scsIndexes"], "SCS index differs")
    require(len(parity["scalar"]["gaussians"]) == len(parity["batched"]["gaussians"]) == 2 * int(args.structure_voxel_size > 0), "Gaussian count differs")
    for a, b in zip(parity["scalar"]["gaussians"], parity["batched"]["gaussians"], strict=True):
        require((a["sha256"], a["bytes"], a["voxels"]) == (b["sha256"], b["bytes"], b["voxels"]), "full voxel Gaussian bits differ")
    expected_indexes = int(args.structure_voxel_size > 0 and parity["batched"]["report"]["metrics"].get("n_awd_voxels", 0) >= current._SCS_INDEX_MIN_VOXELS)
    require(len(parity["batched"]["scsIndexes"]) == expected_indexes, "unexpected SCS index count")
    first = {policy: call(policy) for policy in ["scalar", "batched"]}
    expected = first["scalar"]["report"]
    require(first["batched"]["report"] == expected, "first reports differ")
    no_ply_report = {**parity["scalar"]["report"], "artifacts": {key: value for key, value in
                     parity["scalar"]["report"]["artifacts"].items() if not key.endswith("_ply")}}
    require(expected == no_ply_report, "artifact choice changed metrics")

    def pair(index):
        order = ["scalar", "batched"] if index % 2 == 0 else ["batched", "scalar"]
        observed = {policy: call(policy) for policy in order}
        require(all(item["report"] == expected for item in observed.values()), "timed report differs")
        return {"order": order, "millis": {key: value["millis"] for key, value in observed.items()},
                "reportHashes": {key: value["reportSha256"] for key, value in observed.items()}}

    print(json.dumps({"stage": "parity_complete", "source": manifest["source"]["path"]}), flush=True)
    warmups = [pair(index) for index in range(2)]
    trials = [pair(index + 2) for index in range(args.repeats)]
    a, b = [item["millis"]["scalar"] for item in trials], [item["millis"]["batched"] for item in trials]
    summary = {"scalarMedianMillis": statistics.median(a), "batchedMedianMillis": statistics.median(b),
               "pairedShorteningPercent": statistics.median([100 * (1 - y / x) for x, y in zip(a, b)]),
               "pairedSavedMedianMillis": statistics.median([x - y for x, y in zip(a, b)]), "fasterTrials": sum(x > y for x, y in zip(a, b))}
    reducer = None
    if args.structure_voxel_size > 0:
        dtype = np.dtype([("key", "<i8", (3,)), ("count", "<i8"), ("mu", "<f8", (3,)), ("sigma", "<f8", (3, 3))])
        gaussian_maps = []
        for metadata in parity["scalar"]["gaussians"]:
            rows = np.fromfile(metadata["path"], dtype=dtype)
            gaussian_maps.append({tuple(int(axis) for axis in row["key"]): baseline.VoxelGaussian(
                row["mu"].copy(), row["sigma"].copy(), int(row["count"])) for row in rows})
        estimated_map, reference_map = gaussian_maps
        def scalar_distances():
            return {key: baseline.wasserstein_distance_gaussian(a.mu, a.sigma, b.mu, b.sigma)
                    for key, a in estimated_map.items() if (b := reference_map.get(key)) is not None
                    and a.num_points >= 100 and b.num_points >= 100}
        def batched_distances():
            return dict(current._voxel_wasserstein_distances(estimated_map, reference_map, 100))
        functions = {"scalar": scalar_distances, "batched": batched_distances}
        packer = struct.Struct("<3qd")
        def distance_bytes(values):
            return b"".join(packer.pack(*key, value) for key, value in values.items())
        captures = {}
        for policy, function in functions.items():
            values = function()
            path = args.work_dir / (policy + "-voxel-distances.bin")
            with path.open("xb") as stream:
                stream.write(distance_bytes(values))
            captures[policy] = {**info(path), "voxels": len(values), "recordBytes": packer.size}
        require(captures["scalar"]["sha256"] == captures["batched"]["sha256"], "complete Gaussian distances differ")
        expected_hash = captures["scalar"]["sha256"]
        def reducer_pair(index):
            order = ["scalar", "batched"] if index % 2 == 0 else ["batched", "scalar"]
            observed = {}
            for policy in order:
                start = time.perf_counter_ns()
                values = functions[policy]()
                observed[policy] = (time.perf_counter_ns() - start) / 1e6
                require(hashlib.sha256(distance_bytes(values)).hexdigest() == expected_hash, "Gaussian distances changed")
                del values
            return {"order": order, "millis": observed}
        stage_first = reducer_pair(0)
        stage_warmups = [reducer_pair(index) for index in range(2)]
        stage_trials = [reducer_pair(index + 2) for index in range(args.repeats)]
        a, b = [p["millis"]["scalar"] for p in stage_trials], [p["millis"]["batched"] for p in stage_trials]
        reducer = {"parity": captures, "firstUse": stage_first, "warmups": stage_warmups, "trials": stage_trials,
            "summary": {"scalarMedianMillis": statistics.median(a), "batchedMedianMillis": statistics.median(b),
                "pairedShorteningPercent": statistics.median([100 * (1 - y / x) for x, y in zip(a, b)]),
                "pairedSavedMedianMillis": statistics.median([x - y for x, y in zip(a, b)]), "fasterTrials": sum(x > y for x, y in zip(a, b))}}
    for path, expected_hash in hashes.items():
        require(sha(args.source_root / path) == expected_hash, "source changed: " + path)
    for item in [*inputs.values(), compiled_tree]:
        require(sha(item["path"]) == item["sha256"], "input/library changed")
    for name, array in [("compared", source), ("reference", reference)]:
        require(hashlib.sha256(array.astype("<f8", copy=False).tobytes()).hexdigest() == coordinate_hashes[name], "input coordinates changed")
    report = {"schemaVersion": 1, "comparison": "map-wasserstein-batch", "createdAt": datetime.now(timezone.utc).isoformat(),
              "sourceHashes": hashes, "inputs": inputs, "inputManifest": manifest, "coordinateHashes": coordinate_hashes,
              "compiledTree": compiled_tree, "environment": {"python": platform.python_version(), "numpy": np.__version__,
                 "scipy": scipy.__version__, "open3d": o3d.__version__, "platform": platform.system(), "logicalCpus": os.cpu_count(),
                 "ompNumThreads": os.environ["OMP_NUM_THREADS"], "openblasNumThreads": os.environ["OPENBLAS_NUM_THREADS"]},
              "policy": {"nnWorkers": -1, "nnParallelMinPoints": current._PARALLEL_NN_MIN_POINTS, "scsIndexMinVoxels": current._SCS_INDEX_MIN_VOXELS, "scsExactSpan": current._SCS_INDEX_EXACT_SPAN, "meanBincountMinPoints": current._VOXEL_MEAN_BINCOUNT_MIN_POINTS,
                         "wassersteinBatchSize": current._WASSERSTEIN_BATCH_SIZE,
                         "wassersteinBatchMinPairs": current._WASSERSTEIN_BATCH_MIN_PAIRS,
                         "wassersteinSafeMagnitude": current._WASSERSTEIN_BATCH_SAFE_MAGNITUDE, "numpyErrorState": np.geterr()},
              "repeats": args.repeats, "warmupRounds": 2, "structureVoxelSize": args.structure_voxel_size,
              "wassersteinReducer": reducer, "parity": parity, "firstUse": first, "warmups": warmups, "trials": trials, "summary": summary,
              "accuracy": {"fullF64ArraysChecked": 4, "plyOutputsChecked": 4, "gaussianMapsChecked": 4 * int(args.structure_voxel_size > 0), "reportsChecked": 2 * (2 + 2 + args.repeats), "fullGaussianDistanceArraysChecked": 2 * int(args.structure_voxel_size > 0)},
              "note": "Control already includes all preceding MapEval optimizations. Current batches eligible Gaussian "
                      "Wasserstein calculations in at most 256 supported pairs, preserving the scalar formula, matrix "
                      "operations and per-vector mean dot. Batches below 128 pairs, nonstandard/nonfinite/extreme arrays "
                      "or nondefault NumPy error settings retain scalar calls. Complete resident clocks include fresh "
                      "NN queries, thresholds, grouping, means/covariances, Wasserstein and SCS/index setup/destruction. "
                      "Alignment/downsampling are disabled; original coordinates/sampling stay fixed. Loading, imports, "
                      "hashing, Gaussian/f64/PLY parity tracing and CLI writes are outside clocks. Two warm-up pairs and "
                      "an excluded first pair precede five default alternating timed pairs. All observations remain. "
                      "Separate wassersteinReducer clocks use saved full Gaussian maps, include supported-pair selection "
                      "and distance dictionary construction, exclude SCS/moment building and checking/hashing. No new "
                      "worker pool/cache, streaming speed, GPU or peak-RSS claim."}


    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
