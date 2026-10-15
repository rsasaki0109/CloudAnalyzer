"""Compare complete CSV saves of unchanged measured point clouds."""

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


def same_bytes(first, second):
    with Path(first).open("rb") as a, Path(second).open("rb") as b:
        while True:
            block = a.read(1 << 20)
            require(block == b.read(1 << 20), "CSV bytes differ")
            if not block:
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ["source-root", "baseline", "input-manifest", "work-dir", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--dataset", required=True, choices=["pandaset019", "autoware-planning", "tokyo-takanawadai"])
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--start-policy", choices=["whole_rows", "batch_rows"], default="whole_rows")
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
    spec = importlib.util.spec_from_file_location("io_whole_csv_write_baseline", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    entry = manifest["sources"][args.dataset]
    inputs = dict(manifest=info(args.input_manifest), baseline=info(args.baseline))
    for key in ["geometry", "coordinateOracle", "originalSource", "sourceManifest"]:
        item = entry[key]
        require(sha(item["path"]) == item["sha256"], "source provenance differs: " + key)
        inputs[key] = info(item["path"])
    source_manifest = json.loads(Path(inputs["sourceManifest"]["path"]).read_text(encoding="utf-8"))
    require(source_manifest == entry["provenance"], "original manifest differs")
    require(entry["points"] == source_manifest["inputs"]["compared"]["points"], "point count differs")
    require(inputs["coordinateOracle"]["bytes"] == 24 * entry["points"], "coordinate size differs")
    oracle = np.memmap(inputs["coordinateOracle"]["path"], mode="r", dtype="<f8", shape=(entry["points"], 3))
    cloud = o3d.io.read_point_cloud(inputs["geometry"]["path"])
    require(np.asarray(cloud.points).tobytes() == oracle.tobytes(), "original PLY coordinates differ")
    paths = ["cloudanalyzer/ca/io.py", "cloudanalyzer/ca/_rust.py", "cloudanalyzer/ca/__init__.py",
             "cloudanalyzer/ca/convert.py", "cloudanalyzer/pyproject.toml",
             "cloudanalyzer/tests/test_io_csv_write_batches.py", "cloudanalyzer/tests/test_io.py",
             "cloudanalyzer/tests/test_cli.py", "cloudanalyzer/scripts/benchmark_csv_writer.py"]
    source_hashes = {path: sha(args.source_root / path) for path in paths}
    parser_file = getattr(_csv, "__file__", None)
    if parser_file is None:
        runtime = Path(sys.base_prefix) / f"python{sys.version_info.major}{sys.version_info.minor}.dll"
        parser_file = runtime if runtime.is_file() else Path(sys.executable)
    libraries = dict(open3d=info(o3d.cpu.pybind.__file__), csv=info(csv.__file__),
                     csvParser=dict(**info(parser_file), moduleOrigin=_csv.__spec__.origin),
                     numpyCore=info(np._core._multiarray_umath.__file__))
    args.work_dir.mkdir(parents=True, exist_ok=False)
    modules = dict(whole_rows=baseline, batch_rows=current)
    names = [args.start_policy, "batch_rows" if args.start_policy == "whole_rows" else "whole_rows"]
    expected = None
    calls = 0

    def call(policy, keep=False):
        nonlocal expected, calls
        calls += 1
        path = args.work_dir / f"{calls:02d}-{policy}.csv"
        require(not path.exists(), "save destination exists")
        start = time.perf_counter_ns()
        result = modules[policy].save_point_cloud(str(path), cloud)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        require(result is None, "save return value differs")
        signature = dict(bytes=path.stat().st_size, sha256=sha(path))
        if expected is None:
            expected = signature
        require(signature == expected, "full CSV fingerprint differs")
        if keep:
            return dict(millis=elapsed, output=signature, path=str(path.resolve()))
        path.unlink()
        return dict(millis=elapsed, output=signature)

    parity = {policy: call(policy, keep=True) for policy in names}
    same_bytes(parity["whole_rows"]["path"], parity["batch_rows"]["path"])
    with Path(parity["whole_rows"]["path"]).open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        require(next(reader) == ["x", "y", "z"], "output header differs")
        count = 0
        for row, coordinates in zip(reader, oracle, strict=True):
            parsed = np.asarray([float(value) for value in row], dtype="<f8")
            require(parsed.tobytes() == coordinates.tobytes(), "CSV coordinate bits differ")
            count += 1
        require(count == entry["points"], "CSV point count differs")
    print(json.dumps(dict(stage="parity_complete", dataset=args.dataset)), flush=True)

    def pair(index):
        order = names if index % 2 == 0 else list(reversed(names))
        observed = {policy: call(policy) for policy in order}
        return dict(order=order, observations=observed)

    first = pair(0)
    warmups = [pair(1)]
    trials = []
    for index in range(args.repeats):
        trial = pair(index)
        trials.append(trial)
        print(json.dumps(dict(stage="trial_complete", dataset=args.dataset, trial=index+1,
                             millis={key: value["millis"] for key, value in trial["observations"].items()})), flush=True)
    a = [item["observations"]["whole_rows"]["millis"] for item in trials]
    b = [item["observations"]["batch_rows"]["millis"] for item in trials]
    summary = dict(wholeRowsMedianMillis=statistics.median(a), batchRowsMedianMillis=statistics.median(b),
                   pairedShorteningPercent=statistics.median(100*(1-y/x) for x, y in zip(a, b)),
                   pairedSavedMedianMillis=statistics.median(x-y for x, y in zip(a, b)),
                   fasterTrials=sum(x > y for x, y in zip(a, b)))
    require(np.asarray(cloud.points).tobytes() == oracle.tobytes(), "source cloud changed")
    for path, expected_hash in source_hashes.items():
        require(sha(args.source_root / path) == expected_hash, "source changed: " + path)
    for item in [*inputs.values(), *libraries.values()]:
        require(sha(item["path"]) == item["sha256"], "input/library changed")
    record = dict(schemaVersion=1, comparison="csv-batched-write", createdAt=datetime.now(timezone.utc).isoformat(),
                  dataset=args.dataset, provenance=entry["provenance"], inputs=inputs, sourceHashes=source_hashes, libraries=libraries,
                  environment=dict(platform=platform.system(), python=platform.python_version(), numpy=np.__version__, open3d=o3d.__version__,
                                   logicalCpus=os.cpu_count(), rayonNumThreads=args.threads, ompNumThreads=args.threads,
                                   openblasNumThreads=1, disableRustCore=True, csvWriteBatchRows=current._CSV_WRITE_BATCH_SIZE),
                  startPolicy=args.start_policy, repeats=args.repeats, warmupRounds=1, points=entry["points"], parity=parity,
                  firstUntraced=first, warmups=warmups, trials=trials, summary=summary,
                  note="Complete save_point_cloud CSV function: path/parent handling, actual UTF-8 text-file open, "
                       "canonical XYZ header, unchanged Python float formatting via csv.writer, all data writes, "
                       "internal cleanup and close/flush. No fsync durability guarantee. Source loading, imports, "
                       "fingerprints, output hashing/validation and removal are excluded. Candidate copies full XYZ "
                       "into a stable numeric snapshot and materializes at most 65536 Python rows per batch; old "
                       "writer materializes all Python rows. Entire existing measured compared subsets are used, "
                       "without further thinning, generated coordinates or perturbation; not independent surveys. "
                       "Excluded full-byte parity pair, first pair and one warm-up pair precede five default "
                       "alternating timed pairs. Every save is hashed, both parity files are retained and checked "
                       "byte-for-byte; all baseline parity CSV coordinates round-trip bitwise to the full oracle. "
                       "Other outputs are removed outside timing. Fresh process per source/pass, warm pages. "
                       "No peak RSS, cold CLI, complete conversion, other-format, parallel or GPU speed claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(stage="complete", dataset=args.dataset, summary=summary)), flush=True)


if __name__ == "__main__":
    main()
