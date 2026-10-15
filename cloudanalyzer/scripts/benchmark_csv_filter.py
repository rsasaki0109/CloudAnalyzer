"""Compare complete CSV point-cloud loads using unchanged measured coordinates.

Run each source/pass in a fresh process. Parsing, real file I/O and owned
Open3D conversion run normally; only finite-filter batching changes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
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
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def info(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ["source-root", "baseline", "input-manifest", "csv", "coordinate-oracle", "work-dir", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--dataset", required=True, choices=["pandaset019", "autoware-planning", "tokyo-takanawadai"])
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--start-policy", choices=["per_row", "batched"], default="per_row")
    options = [value.split("=")[0] for value in sys.argv[1:] if value.startswith("--")]
    require(len(options) == len(set(options)), "duplicate options")
    args = parser.parse_args()
    require(1 <= args.threads <= 64 and 1 <= args.repeats <= 100, "invalid threads/repeats")
    require(not args.output.exists() and not args.work_dir.exists(), "output/work directory must be new")
    require(sha(args.baseline) == args.baseline_sha256, "I/O baseline differs")
    args.source_root = args.source_root.resolve()
    os.environ["CA_DISABLE_RUST_CORE"] = "1"
    os.environ["RAYON_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source_root / "cloudanalyzer"))

    import _csv
    import csv
    import numpy as np
    import open3d as o3d
    import ca.io as current

    require(current.core() is None, "optional core must be disabled")
    spec = importlib.util.spec_from_file_location("io_per_row_csv_baseline", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    entry = manifest["sources"][args.dataset]
    require(sha(args.csv) == entry["csv"]["sha256"] and args.csv.stat().st_size == entry["csv"]["bytes"], "CSV differs")
    require(sha(args.coordinate_oracle) == entry["coordinateOracle"]["sha256"], "coordinate oracle differs")
    require(args.coordinate_oracle.stat().st_size == 24 * entry["points"], "coordinate size differs")
    require(args.csv.suffix.lower() == ".csv", "CSV extension required")
    with args.csv.open(newline="", encoding="utf-8") as stream:
        first = next(csv.reader(stream))
        if entry["header"] is not None:
            require(first == entry["header"], "CSV header differs")
        else:
            require(len(first) == 3 and all(np.isfinite(float(value)) for value in first), "numeric first record differs")
    inputs = dict(csv=info(args.csv), oracle=info(args.coordinate_oracle), manifest=info(args.input_manifest), baseline=info(args.baseline))
    for key in ["geometry", "originalSource", "sourceManifest"]:
        item = entry[key]
        require(sha(item["path"]) == item["sha256"], "source provenance differs: " + key)
        inputs[key] = info(item["path"])
    source_manifest = json.loads(Path(inputs["sourceManifest"]["path"]).read_text(encoding="utf-8"))
    require(source_manifest == entry["provenance"], "original manifest differs")
    require(entry["points"] == source_manifest["inputs"]["compared"]["points"], "point count differs")
    oracle = np.memmap(args.coordinate_oracle, mode="r", dtype="<f8", shape=(entry["points"], 3))
    geometry = o3d.io.read_point_cloud(inputs["geometry"]["path"])
    require(np.asarray(geometry.points).tobytes() == oracle.tobytes(), "original PLY coordinates differ")
    del geometry
    paths = ["cloudanalyzer/ca/io.py", "cloudanalyzer/ca/_rust.py", "cloudanalyzer/ca/__init__.py",
             "cloudanalyzer/pyproject.toml", "cloudanalyzer/tests/test_io_csv_batch_filter.py",
             "cloudanalyzer/tests/test_io.py", "cloudanalyzer/tests/test_io_binary_xyz_ply.py",
             "cloudanalyzer/tests/test_io_las_packed_coordinates.py", "cloudanalyzer/tests/test_compare.py",
             "cloudanalyzer/tests/test_cli.py", "cloudanalyzer/scripts/benchmark_csv_filter.py"]
    source_hashes = {path: sha(args.source_root / path) for path in paths}
    parser_file = getattr(_csv, "__file__", None)
    if parser_file is None:
        # Windows embeds _csv in the Python DLL rather than a separate module.
        runtime = Path(sys.base_prefix) / f"python{sys.version_info.major}{sys.version_info.minor}.dll"
        parser_file = runtime if runtime.is_file() else Path(sys.executable)
    libraries = dict(open3d=info(o3d.cpu.pybind.__file__), csv=info(csv.__file__),
                     csvParser=dict(**info(parser_file), moduleOrigin=_csv.__spec__.origin))
    args.work_dir.mkdir(parents=True, exist_ok=False)
    modules = dict(per_row=baseline, batched=current)
    names = [args.start_policy, "batched" if args.start_policy == "per_row" else "per_row"]

    def call(policy, full_check=False):
        start = time.perf_counter_ns()
        cloud = modules[policy].load_point_cloud(str(args.csv))
        elapsed = (time.perf_counter_ns() - start) / 1e6
        points = np.asarray(cloud.points)
        signature = dict(shape=list(points.shape), dtype=points.dtype.str,
                         xyzSha256=hashlib.sha256(memoryview(points)).hexdigest(),
                         colors=len(cloud.colors), normals=len(cloud.normals), covariances=len(cloud.covariances))
        require(signature == dict(shape=[entry["points"], 3], dtype="<f8",
                                  xyzSha256=entry["coordinateOracle"]["sha256"], colors=0, normals=0, covariances=0),
                "complete loaded output differs")
        if full_check:
            for start in range(0, len(points), 131071):
                require(points[start:start+131071].tobytes() == oracle[start:start+131071].tobytes(), "full XYZ bytes differ")
        return dict(millis=elapsed, output=signature)

    parity = {policy: call(policy, full_check=True) for policy in names}
    print(json.dumps(dict(stage="parity_complete", dataset=args.dataset)), flush=True)

    def pair(index):
        order = names if index % 2 == 0 else list(reversed(names))
        observed = {policy: call(policy) for policy in order}
        return dict(order=order, observations=observed)

    first = pair(0)
    warmups = [pair(index + 1) for index in range(2)]
    trials = []
    for index in range(args.repeats):
        trial = pair(index)
        trials.append(trial)
        print(json.dumps(dict(stage="trial_complete", trial=index+1,
                             millis={key: value["millis"] for key, value in trial["observations"].items()})), flush=True)
    a = [item["observations"]["per_row"]["millis"] for item in trials]
    b = [item["observations"]["batched"]["millis"] for item in trials]
    summary = dict(perRowMedianMillis=statistics.median(a), batchedMedianMillis=statistics.median(b),
                   pairedShorteningPercent=statistics.median(100*(1-y/x) for x, y in zip(a, b)),
                   pairedSavedMedianMillis=statistics.median(x-y for x, y in zip(a, b)),
                   fasterTrials=sum(x > y for x, y in zip(a, b)))
    for path, expected in source_hashes.items():
        require(sha(args.source_root / path) == expected, "source changed: " + path)
    for item in [*inputs.values(), *libraries.values()]:
        require(sha(item["path"]) == item["sha256"], "input/library changed")
    record = dict(schemaVersion=1, comparison="csv-batch-finite-filter", createdAt=datetime.now(timezone.utc).isoformat(),
                  dataset=args.dataset, inputManifest=manifest, inputs=inputs, sourceHashes=source_hashes, libraries=libraries,
                  environment=dict(platform=platform.system(), python=platform.python_version(), numpy=np.__version__, open3d=o3d.__version__,
                                   logicalCpus=os.cpu_count(), rayonNumThreads=args.threads, ompNumThreads=args.threads,
                                   openblasNumThreads=1, disableRustCore=True, csvBatchRows=current._CSV_POINT_BATCH_SIZE),
                  startPolicy=args.start_policy, repeats=args.repeats, warmupRounds=2, parity=parity,
                  firstUntraced=first, warmups=warmups, trials=trials, summary=summary,
                  note="Complete load_point_cloud CSV function: real file parsing, Python float conversions, finite validation, "
                       "XYZ assembly, owned Open3D conversion, empty check and internal cleanup. Returned-cloud release, output "
                       "hashing/byte validation, imports, fingerprints, CSV export and CLI serialization excluded from timing. "
                       "Only finite-filter batching changes. Full existing compared subsets of measured source clouds are "
                       "exported to CSV without changing any f64 coordinate bits or order; they are not native sensor CSV "
                       "logs or independent surveys. No generated points, perturbation, further thinning, replaced parser "
                       "or read result. All measured inputs are finite; nonfinite/malformed semantics are tested separately. "
                       "Excluded full byte-parity pair, separate first pair and two warm-up pairs precede five default "
                       "alternating timed pairs. Fresh process per dataset/pass; first policy balanced across passes; all "
                       "observations retained. Warm pages. Full cloud remains in memory; this is not a streaming API. "
                       "No cold CLI, peak RSS, whole comparison, registration or GPU speed claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), summary=summary)), flush=True)


if __name__ == "__main__":
    main()
