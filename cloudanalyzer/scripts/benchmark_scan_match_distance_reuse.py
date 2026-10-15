"""Measure scan-match diagnostic/artifact work with a fixed real ICP result.

Preloading and actual registration are excluded. Both production workflow
versions receive the same full source/reference and measured registration.
The work directory must be new; only its generated artifact files are reused.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def info(path):
    return {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size, "sha256": sha(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ["source-root", "reference", "compared", "input-manifest", "baseline", "work-dir", "output"]:
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=10)
    options = [value.split("=")[0] for value in sys.argv[1:] if value.startswith("--")]
    require(len(options) == len(set(options)), "duplicate options")
    args = parser.parse_args()
    require(1 <= args.threads <= 64 and 1 <= args.repeats <= 100, "invalid threads/repeats")
    require(not args.output.exists() and not args.work_dir.exists(), "output/work directory must be new")
    require(sha(args.baseline) == args.baseline_sha256, "baseline fingerprint differs")
    os.environ["RAYON_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["CA_DISABLE_RUST_CORE"] = "0"
    sys.path.insert(0, str((args.source_root / "cloudanalyzer").resolve()))
    import numpy as np
    import open3d as o3d
    import matplotlib
    import ca.scan_match_debug as current
    from ca.io import load_point_cloud
    from ca.registration import register_detailed
    import cloudanalyzer_core._core as core

    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    require(manifest["schemaVersion"] == 1, "unexpected manifest")
    for key in ["reference", "compared"]:
        path = getattr(args, key)
        require(sha(path) == manifest["inputs"][key]["sha256"], key + " fingerprint differs")
        require(path.stat().st_size == manifest["inputs"][key]["bytes"], key + " length differs")
    original = (args.source_root / "web" / manifest["source"]["path"]).resolve()
    require(sha(original) == manifest["source"]["sha256"], "original measured source differs")
    spec = importlib.util.spec_from_file_location("scan_match_preceding", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    files = ["cloudanalyzer/ca/scan_match_debug.py", "cloudanalyzer/ca/metrics.py",
             "cloudanalyzer/ca/registration.py", "cloudanalyzer/ca/io.py", "cloudanalyzer/ca/visualization.py",
             "cloudanalyzer/ca/_rust.py", "cloudanalyzer/pyproject.toml",
             "cloudanalyzer/tests/test_scan_match_debug.py",
             "cloudanalyzer/scripts/benchmark_scan_match_distance_reuse.py"]
    source_hashes = {path: sha(args.source_root / path) for path in files}
    inputs = {"reference": info(args.reference), "compared": info(args.compared),
              "manifest": info(args.input_manifest), "originalSource": info(original), "baseline": info(args.baseline)}
    extension = info(core.__file__)
    scan = load_point_cloud(str(args.compared))
    reference = load_point_cloud(str(args.reference))
    require(len(scan.points) == manifest["inputs"]["compared"]["points"], "query count differs")
    require(len(reference.points) == manifest["inputs"]["reference"]["points"], "reference count differs")
    require(np.isfinite(np.asarray(scan.points)).all() and np.isfinite(np.asarray(reference.points)).all(), "nonfinite inputs")
    initial = o3d.geometry.PointCloud(scan)
    working_map = o3d.geometry.PointCloud(reference)
    registration = register_detailed(initial, working_map, method="icp", max_correspondence_distance=1.)
    args.work_dir.mkdir(parents=True, exist_ok=False)
    np.savez(args.work_dir / "registration.npz", initial=np.asarray(initial.points),
             aligned=np.asarray(registration.transformed.points), reference=np.asarray(working_map.points),
             initial_normals=np.asarray(initial.normals), aligned_normals=np.asarray(registration.transformed.normals),
             reference_normals=np.asarray(working_map.normals), transformation=registration.transformation)
    registration_info = {"method": "icp", "maxCorrespondenceDistance": 1., "fitness": registration.fitness,
                         "rmse": registration.rmse, "transformation": registration.transformation.tolist(),
                         "snapshot": info(args.work_dir / "registration.npz")}

    def frozen_load(path):
        return scan if path == str(args.compared) else reference

    def frozen_registration(source, target, **kwargs):
        # Real registration adds these normals to its inputs. Reproduce them for
        # identical PLY fields while retaining the actual aligned output.
        source.normals = o3d.utility.Vector3dVector(np.asarray(initial.normals))
        target.normals = o3d.utility.Vector3dVector(np.asarray(working_map.normals))
        return registration

    for module in [baseline, current]:
        module.load_point_cloud = frozen_load
        module.register_detailed = frozen_registration

    def call(policy, capture=False):
        module = baseline if policy == "fresh" else current
        function = module.compute_nn_distance
        captured = []
        if capture:
            def traced(source, target):
                values = function(source, target)
                captured.append(values)
                return values
            module.compute_nn_distance = traced
        start = time.perf_counter_ns()
        try:
            result = module.run_scan_match_debug(str(args.compared), str(args.reference), method="icp",
                                                threshold=.5, artifact_dir=str(args.work_dir / policy))
        finally:
            module.compute_nn_distance = function
        millis = (time.perf_counter_ns() - start) / 1e6
        artifacts = {key: {"file": Path(path).name, "bytes": Path(path).stat().st_size, "sha256": sha(path)}
                     for key, path in result["artifacts"].items()}
        # Directory paths differ by policy; preserve all other report fields.
        result["artifacts"] = {key: value["file"] for key, value in artifacts.items()}
        raw = [{"bytes": values.nbytes, "sha256": hashlib.sha256(values.astype("<f8",copy=False).tobytes()).hexdigest()}
               for values in captured]
        return {"millis": millis, "report": result, "artifacts": artifacts, "capturedDistances": raw}

    first = {policy: call(policy, capture=True) for policy in ["fresh", "reuse"]}
    require(first["fresh"]["report"] == first["reuse"]["report"], "diagnostic reports differ")
    require(first["fresh"]["artifacts"] == first["reuse"]["artifacts"], "complete artifact bytes differ")
    old, new = first["fresh"]["capturedDistances"], first["reuse"]["capturedDistances"]
    require(len(old) == 4 and len(new) == 2, "unexpected number of nearest-neighbour calls")
    require(old[:2] == new and old[2:] == new, "full f64 distance arrays differ")
    expected_report = first["fresh"]["report"]
    expected_artifacts = first["fresh"]["artifacts"]

    def pair(index):
        order = ["fresh", "reuse"] if index % 2 == 0 else ["reuse", "fresh"]
        observations = {policy: call(policy) for policy in order}
        for observation in observations.values():
            require(observation["report"] == expected_report, "report differs from preceding workflow")
            require(observation["artifacts"] == expected_artifacts, "PLY bytes differ from preceding workflow")
        return {"order": order, "millis": {key: value["millis"] for key,value in observations.items()},
                "artifactHashes": {key: value["artifacts"] for key,value in observations.items()}}

    warmups = [pair(index) for index in range(5)]
    trials = [pair(index+5) for index in range(args.repeats)]
    a = [trial["millis"]["fresh"] for trial in trials]
    b = [trial["millis"]["reuse"] for trial in trials]
    summary = {"freshMedianMillis": statistics.median(a), "reuseMedianMillis": statistics.median(b),
               "pairedShorteningPercent": statistics.median([100*(1-y/x) for x,y in zip(a,b)]),
               "pairedSavedMedianMillis": statistics.median([x-y for x,y in zip(a,b)]),
               "fasterTrials": sum(x>y for x,y in zip(a,b))}
    for path, expected in source_hashes.items():
        require(sha(args.source_root / path) == expected, "source changed: " + path)
    for item in [*inputs.values(), extension, registration_info["snapshot"]]:
        require(sha(item["path"]) == item["sha256"], "input/build changed")
    report = {"schemaVersion":1, "comparison":"scan-match-distance-reuse", "createdAt":datetime.now(timezone.utc).isoformat(),
              "sourceHashes":source_hashes, "inputs":inputs, "inputManifest":manifest, "extension":extension,
              "environment":{"python":platform.python_version(), "numpy":np.__version__, "open3d":o3d.__version__,
                             "matplotlib":matplotlib.__version__, "platform":platform.system(), "logicalCpus":os.cpu_count(),
                             "rayonNumThreads":args.threads, "ompNumThreads":args.threads},
              "registration":registration_info, "threads":args.threads, "repeats":args.repeats, "warmupRounds":5,
              "firstUse":first, "warmups":warmups, "trials":trials, "summary":summary,
              "accuracy":{"fullF64ArraysChecked":6, "plyOutputsChecked":6*(1+5+args.repeats),
                          "report":expected_report,"artifacts":expected_artifacts},
              "memory":{"additionalRetainedBeforeDistanceBytes":len(scan.points)*8},
              "note":"Diagnostic/preprocessing/color/PLY-write interval of production run_scan_match_debug, "
                     "using one actual full-source ICP result fixed across both workflows. Source loading, normal "
                     "estimation and ICP solving are excluded; frozen result/normal assignment and workflow copies "
                     "are included. First-use observations trace/retain distance arrays for correctness and are "
                     "excluded from performance summaries. Output hashing/checks are excluded. No outliers removed. Generated artifact "
                     "paths in the new work directory are overwritten between trials. This does not measure full "
                     "registration/CLI speed or change the registration algorithm, NN kernel or thread dispatch."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output":str(args.output),"summary":summary,"accuracy":report["accuracy"]["plyOutputsChecked"]}),flush=True)


if __name__ == "__main__":
    main()
