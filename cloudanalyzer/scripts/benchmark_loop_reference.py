"""Measure scoped reference reuse in before/after map reports on actual ICP data."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
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
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def report_sha(report):
    data = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source-root", "reference", "compared", "input-manifest", "baseline", "baseline-evaluate",
                 "build-index", "registration-record", "work-dir", "output"]:
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
    require(sha(args.baseline) == args.baseline_sha256, "report baseline differs")
    require(sha(args.baseline_evaluate) == args.baseline_evaluate_sha256, "evaluation baseline differs")
    os.environ["RAYON_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["CA_DISABLE_RUST_CORE"] = "0"
    sys.path.insert(0, str((args.source_root / "cloudanalyzer").resolve()))
    import numpy as np
    import open3d as o3d
    import cloudanalyzer_core._core as extension
    import ca.loop_closure_report as current
    from ca.io import load_point_cloud
    evaluation = importlib.import_module("ca.evaluate")
    require(hasattr(extension, "PreparedReference"), "extension lacks prepared references")
    build = json.loads(args.build_index.read_text(encoding="utf-8"))
    extension_info = info(extension.__file__)
    require(extension_info["sha256"] == build["extension"]["sha256"], "extension differs from build index")
    for path, expected in build["sourceHashes"].items():
        require(sha(args.source_root / path) == expected, "native source differs: " + path)
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    prior = json.loads(args.registration_record.read_text(encoding="utf-8"))
    require(prior["comparison"] == "scan-match-distance-reuse", "unexpected registration record")
    require(prior["inputManifest"] == manifest, "registration manifest differs")
    for key in ["reference", "compared"]:
        require(sha(getattr(args, key)) == manifest["inputs"][key]["sha256"] == prior["inputs"][key]["sha256"], key + " differs")
    original = (args.source_root / "web" / manifest["source"]["path"]).resolve()
    require(sha(original) == manifest["source"]["sha256"], "original source differs")
    snapshot = prior["registration"]["snapshot"]
    require(sha(snapshot["path"]) == snapshot["sha256"], "registration snapshot differs")
    inputs = {name: info(getattr(args, name)) for name in
              ["reference", "compared", "input_manifest", "baseline", "baseline_evaluate", "build_index", "registration_record"]}
    inputs["originalSource"], inputs["registrationSnapshot"] = info(original), info(snapshot["path"])
    files = ["cloudanalyzer/ca/loop_closure_report.py", "cloudanalyzer/ca/evaluate.py", "cloudanalyzer/ca/metrics.py",
             "cloudanalyzer/ca/io.py", "cloudanalyzer/ca/_rust.py", "cloudanalyzer/ca/posegraph.py",
             "cloudanalyzer/ca/trajectory.py", "cloudanalyzer/pyproject.toml", "cloudanalyzer/tests/test_loop_reference.py",
             "cloudanalyzer/scripts/benchmark_loop_reference.py"]
    hashes = {path: sha(args.source_root / path) for path in files}
    baseline = load_module("loop_reference_baseline", args.baseline)
    old_evaluation = load_module("loop_reference_old_evaluation", args.baseline_evaluate)
    baseline.evaluate_map = old_evaluation.evaluate
    with np.load(snapshot["path"]) as data:
        before, reference = load_point_cloud(str(args.compared)), load_point_cloud(str(args.reference))
        for key, cloud in [("initial", before), ("reference", reference)]:
            require(np.array_equal(data[key].view(np.uint64), np.asarray(cloud.points).view(np.uint64)), "recorded points differ: " + key)
        matrix = np.array(prior["registration"]["transformation"])
        require(np.array_equal(matrix.view(np.uint64), data["transformation"].view(np.uint64)), "recorded matrix differs")
        before.transform(matrix)
        require(np.array_equal(np.asarray(before.points).view(np.uint64), data["aligned"].view(np.uint64)), "aligned coordinates differ")
        args.work_dir.mkdir(parents=True, exist_ok=False)
        after_path = args.work_dir / "after-registered.ply"
        require(o3d.io.write_point_cloud(str(after_path), before, write_ascii=False), "aligned PLY write failed")
        decoded = load_point_cloud(str(after_path))
        require(np.array_equal(np.asarray(decoded.points).view(np.uint64), data["aligned"].view(np.uint64)), "saved aligned bits differ")
    after_info = info(after_path)
    del before, reference, decoded

    def call(policy, capture=False):
        module = baseline if policy == "fresh" else current
        eval_module = old_evaluation if policy == "fresh" else evaluation
        nn, loader = eval_module.compute_nn_distance, eval_module.load_point_cloud
        map_eval = module.evaluate_map if policy == "fresh" else module._evaluate_clouds
        prepare = getattr(module, "prepare_nn_reference", None)
        arrays, loads, capacities, evaluations = [], [], [], []

        def capture_array(values):
            path = args.work_dir / (policy + "-distance-" + str(len(arrays)) + ".bin")
            with path.open("xb") as stream:
                stream.write(np.asarray(values, dtype="<f8").tobytes())
            arrays.append({**info(path), "points": len(values)})

        if capture:
            def traced_nn(source, target):
                values = nn(source, target)
                capture_array(values)
                return values

            def traced_load(path):
                loads.append(path)
                return loader(path)

            def traced_evaluate(*positional, **kwargs):
                result = map_eval(*positional, **kwargs)
                evaluations.append(result)
                return result

            eval_module.compute_nn_distance, eval_module.load_point_cloud = traced_nn, traced_load
            if policy == "fresh":
                module.evaluate_map = traced_evaluate
            else:
                current_loader = module.load_point_cloud
                module.load_point_cloud, module._evaluate_clouds = traced_load, traced_evaluate

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
            report = module.build_loop_closure_report(before_map=str(args.compared), after_map=str(after_path),
                        reference_map=str(args.reference), gate=module.LoopClosureGate(min_auc_gain=.001, max_after_chamfer=.2))
        finally:
            eval_module.compute_nn_distance, eval_module.load_point_cloud = nn, loader
            if policy == "fresh":
                module.evaluate_map = map_eval
            elif capture:
                module.load_point_cloud, module._evaluate_clouds = current_loader, map_eval
                module.prepare_nn_reference = prepare
        millis = (time.perf_counter_ns() - start) / 1e6
        return {"millis": millis, "report": report, "reportSha256": report_sha(report), "evaluations": evaluations,
                "distances": arrays, "loads": loads, "referenceCapacities": capacities}

    parity = {policy: call(policy, True) for policy in ["fresh", "reuse"]}
    require(parity["fresh"]["report"] == parity["reuse"]["report"], "complete reports differ")
    require(parity["fresh"]["evaluations"] == parity["reuse"]["evaluations"], "complete evaluations differ")
    for a, b in zip(parity["fresh"]["distances"], parity["reuse"]["distances"], strict=True):
        require(a["sha256"] == b["sha256"] and a["points"] == b["points"], "full f64 distances differ")
    require(len(parity["fresh"]["distances"]) == 4, "unexpected distance count")
    for index in [0, 1]:
        require(parity["fresh"]["distances"][index * 2]["sha256"] == prior["firstUse"]["fresh"]["capturedDistances"][index]["sha256"], "independent forward distances differ")
    require(parity["fresh"]["loads"].count(str(args.reference)) == 2, "unexpected baseline loads")
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
    a, b = [item["millis"]["fresh"] for item in trials], [item["millis"]["reuse"] for item in trials]
    summary = {"freshMedianMillis": statistics.median(a), "reuseMedianMillis": statistics.median(b),
               "pairedShorteningPercent": statistics.median([100 * (1 - y / x) for x, y in zip(a, b)]),
               "pairedSavedMedianMillis": statistics.median([x - y for x, y in zip(a, b)]), "fasterTrials": sum(x > y for x, y in zip(a, b))}
    for path, expected_hash in {**build["sourceHashes"], **hashes}.items():
        require(sha(args.source_root / path) == expected_hash, "source changed: " + path)
    for item in [*inputs.values(), after_info, extension_info]:
        require(sha(item["path"]) == item["sha256"], "input/build changed")
    report = {"schemaVersion": 1, "comparison": "loop-reference-reuse", "createdAt": datetime.now(timezone.utc).isoformat(),
              "sourceHashes": hashes, "nativeSourceHashes": build["sourceHashes"], "inputs": inputs, "inputManifest": manifest,
              "after": after_info, "transformation": matrix.tolist(), "extension": extension_info,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "open3d": o3d.__version__,
                              "platform": platform.system(), "logicalCpus": os.cpu_count(), "rayonNumThreads": args.threads, "ompNumThreads": args.threads},
              "threads": args.threads, "repeats": args.repeats, "warmupRounds": 5, "parity": parity, "firstUse": first,
              "warmups": warmups, "trials": trials, "summary": summary,
              "accuracy": {"fullF64ArraysChecked": 8, "reportsChecked": 2 * (2 + 5 + args.repeats), "fullEvaluationsChecked": 4},
              "memory": {"referenceNativePayloadBytes": parity["reuse"]["referenceCapacities"][0]},
              "note": "Complete map-only report function includes real file loading, per-invocation reference preparation/destruction, "
                      "four directional searches, all evaluation statistics/deltas and gates. Before/after are original measured "
                      "input and an archived actual ICP output, verified by reapplying its matrix bit-for-bit. This measures report "
                      "cost, not loop-closure accuracy or ICP speed. Optional trajectory/posegraph sections are not supplied. "
                      "Imports, aligned-file preparation, hashing, parity tracing and CLI serialization/writes are outside clocks. "
                      "Excluded first pair and five warm-up pairs precede alternating timed pairs; no outliers removed. "
                      "OS file caches are warmed; clocks and scheduling are uncontrolled. Native payload is not RSS. No GPU claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
