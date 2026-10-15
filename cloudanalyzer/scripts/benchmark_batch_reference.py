"""Compare whole batch evaluation against archived stateless Python sources.

Uses three unchanged-coordinate selections (strides 1, 2, 4) of an existing
real-source compared cloud. Every timed invocation loads files, prepares/drops
its trees, runs both distance directions and computes complete batch summaries.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import platform
import statistics
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


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report_sha(report):
    data = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source-root", "reference", "compared", "input-manifest", "baseline",
                 "baseline-evaluate", "build-index", "work-dir", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--baseline-evaluate-sha256", required=True)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=10)
    options = [value.split("=")[0] for value in sys.argv[1:] if value.startswith("--")]
    require(len(options) == len(set(options)), "duplicate options")
    args = parser.parse_args()
    require(1 <= args.threads <= 64 and 1 <= args.repeats <= 100, "invalid threads/repeats")
    require(not args.output.exists() and not args.work_dir.exists(), "output/work directory must be new")
    require(sha(args.baseline) == args.baseline_sha256, "batch baseline differs")
    require(sha(args.baseline_evaluate) == args.baseline_evaluate_sha256, "evaluation baseline differs")
    os.environ["RAYON_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["CA_DISABLE_RUST_CORE"] = "0"
    sys.path.insert(0, str((args.source_root / "cloudanalyzer").resolve()))
    import numpy as np
    import open3d as o3d
    import cloudanalyzer_core._core as extension
    import ca.batch as current
    from ca.io import load_point_cloud
    evaluation = importlib.import_module("ca.evaluate")
    logging.getLogger("ca").setLevel(logging.ERROR)
    require(hasattr(extension, "PreparedReference"), "extension lacks prepared references")
    build = json.loads(args.build_index.read_text(encoding="utf-8"))
    extension_info = info(extension.__file__)
    require(extension_info["sha256"] == build["extension"]["sha256"], "extension differs from build index")
    for path, expected in build["sourceHashes"].items():
        require(sha(args.source_root / path) == expected, "native source differs: " + path)
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    for key in ["reference", "compared"]:
        require(sha(getattr(args, key)) == manifest["inputs"][key]["sha256"], key + " differs")
    original = (args.source_root / "web" / manifest["source"]["path"]).resolve()
    require(sha(original) == manifest["source"]["sha256"], "original source differs")
    inputs = {name: info(getattr(args, name)) for name in
              ["reference", "compared", "input_manifest", "baseline", "baseline_evaluate", "build_index"]}
    inputs["originalSource"] = info(original)
    files = ["cloudanalyzer/ca/batch.py", "cloudanalyzer/ca/evaluate.py", "cloudanalyzer/ca/metrics.py",
             "cloudanalyzer/ca/io.py", "cloudanalyzer/ca/_rust.py", "cloudanalyzer/ca/pareto.py",
             "cloudanalyzer/ca/log.py", "cloudanalyzer/pyproject.toml", "cloudanalyzer/tests/test_batch_reference.py",
             "cloudanalyzer/scripts/benchmark_batch_reference.py"]
    hashes = {path: sha(args.source_root / path) for path in files}
    baseline = load_module("batch_reference_baseline", args.baseline)
    old_evaluation = load_module("batch_reference_old_evaluation", args.baseline_evaluate)
    baseline.evaluate = old_evaluation.evaluate
    args.work_dir.mkdir(parents=True, exist_ok=False)
    candidates = args.work_dir / "candidates"
    candidates.mkdir()
    cloud = load_point_cloud(str(args.compared))
    points = np.asarray(cloud.points)
    candidate_info = []
    for stride in [1, 2, 4]:
        path = candidates / ("stride-" + str(stride) + ".ply")
        selected = o3d.geometry.PointCloud()
        selected.points = o3d.utility.Vector3dVector(points[::stride])
        require(o3d.io.write_point_cloud(str(path), selected, write_ascii=False), "candidate write failed")
        decoded = np.asarray(load_point_cloud(str(path)).points)
        require(np.array_equal(decoded.view(np.uint64), points[::stride].view(np.uint64)), "candidate coordinates changed")
        candidate_info.append({**info(path), "stride": stride, "points": len(decoded),
                               "xyzF64Sha256": hashlib.sha256(decoded.astype("<f8", copy=False).tobytes()).hexdigest()})
    del cloud, points, selected, decoded

    def call(policy, capture=False):
        module = baseline if policy == "fresh" else current
        eval_module = old_evaluation if policy == "fresh" else evaluation
        nn, loader = eval_module.compute_nn_distance, module.load_point_cloud
        eval_loader = eval_module.load_point_cloud
        prepare = getattr(module, "prepare_nn_reference", None)
        arrays, loads, capacities = [], [], []

        def capture_array(values):
            path = args.work_dir / (policy + "-distance-" + str(len(arrays)) + ".bin")
            data = np.asarray(values, dtype="<f8").tobytes()
            with path.open("xb") as stream:
                stream.write(data)
            arrays.append({**info(path), "points": len(values)})

        if capture:
            def traced_nn(source, target):
                values = nn(source, target)
                capture_array(values)
                return values

            def traced_load(path):
                loads.append(path)
                return loader(path)

            eval_module.compute_nn_distance = traced_nn
            module.load_point_cloud = eval_module.load_point_cloud = traced_load
            if prepare is not None:
                def traced_prepare(target):
                    native = prepare(target)
                    require(native is not None, "missing prepared reference")
                    capacities.append(native.retained_bytes)

                    class TracedReference:
                        def distances(self, source):
                            values = native.distances(source)
                            capture_array(values)
                            return values

                    return TracedReference()

                module.prepare_nn_reference = traced_prepare
        start = time.perf_counter_ns()
        try:
            report = module.batch_evaluate(str(candidates), str(args.reference), min_auc=.95, max_chamfer=.02)
        finally:
            eval_module.compute_nn_distance = nn
            module.load_point_cloud, eval_module.load_point_cloud = loader, eval_loader
            if prepare is not None:
                module.prepare_nn_reference = prepare
        millis = (time.perf_counter_ns() - start) / 1e6
        return {"millis": millis, "report": report, "reportSha256": report_sha(report),
                "distances": arrays, "loads": loads, "referenceCapacities": capacities}

    parity = {policy: call(policy, True) for policy in ["fresh", "reuse"]}
    require(parity["fresh"]["report"] == parity["reuse"]["report"], "complete reports differ")
    for a, b in zip(parity["fresh"]["distances"], parity["reuse"]["distances"], strict=True):
        require(a["sha256"] == b["sha256"] and a["points"] == b["points"], "full f64 distances differ")
    require(len(parity["fresh"]["distances"]) == 6, "unexpected distance count")
    require(parity["fresh"]["loads"].count(str(args.reference)) == 4, "unexpected baseline loads")
    require(parity["reuse"]["loads"].count(str(args.reference)) == 1, "unexpected current loads")
    require(not parity["fresh"]["referenceCapacities"] and len(parity["reuse"]["referenceCapacities"]) == 1,
            "unexpected reference construction count")
    expected = parity["fresh"]["report"]
    first = {policy: call(policy) for policy in ["fresh", "reuse"]}
    require(all(item["report"] == expected for item in first.values()), "first reports differ")

    def pair(index):
        order = ["fresh", "reuse"] if index % 2 == 0 else ["reuse", "fresh"]
        observed = {policy: call(policy) for policy in order}
        require(all(item["report"] == expected for item in observed.values()), "timed report differs")
        return {"order": order, "millis": {key: value["millis"] for key, value in observed.items()},
                "reportHashes": {key: value["reportSha256"] for key, value in observed.items()}}

    warmups = [pair(index) for index in range(5)]
    trials = [pair(index + 5) for index in range(args.repeats)]
    a = [item["millis"]["fresh"] for item in trials]
    b = [item["millis"]["reuse"] for item in trials]
    summary = {"freshMedianMillis": statistics.median(a), "reuseMedianMillis": statistics.median(b),
               "pairedShorteningPercent": statistics.median([100 * (1 - y / x) for x, y in zip(a, b)]),
               "pairedSavedMedianMillis": statistics.median([x - y for x, y in zip(a, b)]),
               "fasterTrials": sum(x > y for x, y in zip(a, b))}
    for path, expected_hash in {**build["sourceHashes"], **hashes}.items():
        require(sha(args.source_root / path) == expected_hash, "source changed: " + path)
    for item in [*inputs.values(), *candidate_info, extension_info]:
        require(sha(item["path"]) == item["sha256"], "input/build changed")
    report = {"schemaVersion": 1, "comparison": "batch-reference-reuse", "createdAt": datetime.now(timezone.utc).isoformat(),
              "sourceHashes": hashes, "nativeSourceHashes": build["sourceHashes"], "inputs": inputs, "inputManifest": manifest,
              "candidates": candidate_info, "extension": extension_info,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "open3d": o3d.__version__,
                              "platform": platform.system(), "logicalCpus": os.cpu_count(), "rayonNumThreads": args.threads,
                              "ompNumThreads": args.threads},
              "threads": args.threads, "repeats": args.repeats, "warmupRounds": 5, "parity": parity,
              "firstUse": first, "warmups": warmups, "trials": trials, "summary": summary,
              "accuracy": {"fullF64ArraysChecked": 12, "batchReportsChecked": 2 * (2 + 5 + args.repeats)},
              "memory": {"referenceNativePayloadBytes": parity["reuse"]["referenceCapacities"][0]},
              "note": "Whole batch function includes file discovery and loading, per-invocation reference preparation/destruction, "
                      "six bidirectional queries, all evaluation summaries and quality gates. Three candidates select unchanged "
                      "real compared coordinates at strides 1/2/4; they are subsets of one source, not independent surveys. "
                      "Full-f64 parity and excluded first pairs precede five warm-up and alternating timed pairs. "
                      "Candidate writes, imports, hashing, correctness checks and CLI report serialization are outside clocks. "
                      "No outliers removed; warmed OS file cache, clocks and scheduling remain uncontrolled. "
                      "Native capacity is payload, not RSS. No browser/GPU speed claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
