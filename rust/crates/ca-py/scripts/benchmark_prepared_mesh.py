"""Compare complete Python C2M calls using an archived real-source native oracle.

Run in a release-wheel environment. No input thinning or coordinate conversion.
Checks and serialization run outside the API clocks. Output is create-only.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import struct
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_info(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["geometry", "compared", "input-manifest", "native-report", "oracle-f64", "source-root", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=10)
    raw_options = [value.split("=")[0] for value in sys.argv[1:] if value.startswith("--")]
    require(len(raw_options) == len(set(raw_options)), "duplicate options are not allowed")
    args = parser.parse_args()
    require(1 <= args.threads <= 64 and 1 <= args.repeats <= 100, "invalid threads or repeats")
    require(not args.output.exists(), "refusing to overwrite an existing report")
    require(sys.byteorder == "little", "oracle uses little-endian f64")
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import cloudanalyzer_core as cc
    import cloudanalyzer_core._core as core

    native = json.loads(args.native_report.read_text(encoding="utf-8"))
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    require(native["comparison"] == "native-c2m-retained-par", "unexpected native oracle report")
    require(manifest == native["inputManifest"], "input manifest differs from native oracle")
    require(digest(args.input_manifest) == native["inputManifestSha256"], "manifest fingerprint differs")
    require(digest(args.geometry) == native["geometry"]["sha256"], "geometry differs from original TIN")
    require(digest(args.compared) == native["inputs"]["compared"]["sha256"], "query input differs")
    require(digest(args.oracle_f64) == native["oracle"]["distanceSha256"], "distance oracle differs")
    source = (args.source_root / "web" / manifest["source"]["path"]).resolve()
    require(digest(source) == manifest["source"]["sha256"], "original measured source differs")
    paths = list(native["builds"]["native"]["sourceHashes"])
    paths += ["rust/crates/ca-py/Cargo.toml", "rust/crates/ca-py/Cargo.lock",
              "rust/crates/ca-py/pyproject.toml", "rust/crates/ca-py/python/cloudanalyzer_core/__init__.py",
              "rust/crates/ca-py/python/cloudanalyzer_core/_core.pyi",
              "rust/crates/ca-py/tests/test_prepared_mesh.py",
              "rust/crates/ca-py/scripts/benchmark_prepared_mesh.py"]
    paths += [str(path.relative_to(args.source_root)).replace("\\", "/")
              for directory, pattern in [("rust/crates/ca-py/src", "*.rs"),
                                         ("rust/crates/ca-py/python/cloudanalyzer_core", "*.py")]
              for path in sorted((args.source_root / directory).glob(pattern))]
    source_hashes = {path: digest(args.source_root / path) for path in paths}
    for path, old in native["builds"]["native"]["sourceHashes"].items():
        if path != "rust/crates/ca-py/src/lib.rs":
            require(source_hashes[path] == old, "native core/config changed: " + path)
    inputs = {"geometry": file_info(args.geometry), "compared": file_info(args.compared),
              "manifest": file_info(args.input_manifest), "originalSource": file_info(source),
              "oracleF64": file_info(args.oracle_f64), "nativeReport": file_info(args.native_report)}
    extension = file_info(Path(core.__file__))
    package_dir = Path(cc.__file__).parent
    for path in (args.source_root / "rust/crates/ca-py/python/cloudanalyzer_core").glob("*"):
        if path.is_file() and path.suffix in {".py", ".pyi"}:
            require(digest(path) == digest(package_dir / path.name), "installed package source differs: " + path.name)
    vertices, triangles = cc.read_mesh(str(args.geometry))
    points = cc.read(str(args.compared))["positions"]
    require(vertices.shape == (native["mesh"]["vertices"], 3), "vertex count differs")
    require(triangles.shape == (native["mesh"]["triangles"], 3), "triangle count differs")
    require(points.shape == (native["mesh"]["queryPoints"], 3), "point count differs")
    require(np.isfinite(points).all() and np.isfinite(vertices).all(), "expected finite real coordinates")
    header = ("ply\nformat binary_little_endian 1.0\ncomment CloudAnalyzer Web\n"
              f"element vertex {len(points)}\nproperty double x\nproperty double y\nproperty double z\n"
              "property float scalar_C2M\nend_header\n").encode()
    record_dtype = np.dtype([("xyz", "<f8", (3,)), ("distance", "<f4")])

    def output_info(values):
        require(values.shape == (len(points),) and values.dtype == np.float64, "invalid output array")
        raw = values.astype("<f8", copy=False).tobytes()
        records = np.empty(len(points), dtype=record_dtype)
        records["xyz"] = points
        records["distance"] = values
        ply = header + records.tobytes()
        return {"distanceBytes": len(raw), "distanceSha256": hashlib.sha256(raw).hexdigest(),
                "scalarPlyBytes": len(ply), "scalarPlySha256": hashlib.sha256(ply).hexdigest()}

    def check(values):
        info = output_info(values)
        for key, value in info.items():
            require(value == native["oracle"][key], "unsigned oracle mismatch: " + key)
        return info

    def statistics_bits(values):
        # Sequential Python float operations preserve the native f64 reduction order.
        total = squared = 0.0
        for value in values:
            value = float(value)
            total += value
            squared += value * value
        mean = total / len(values)
        sorted_values = np.sort(values)
        mid = len(values) // 2
        median = float(sorted_values[mid])
        if len(values) % 2 == 0:
            median = .5 * (float(sorted_values[mid - 1]) + median)
        stats = [float(values.min()), float(values.max()), mean, math.sqrt(squared / len(values)),
                 math.sqrt(max(squared / len(values) - mean * mean, 0.0)), median]
        return [struct.pack(">d", value).hex() for value in stats]

    def fresh(signed=False):
        return cc.cloud_to_mesh(points, vertices, triangles, signed=signed)

    def timed(call):
        start = time.perf_counter_ns()
        values = call()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        return elapsed, values

    fresh_first, fresh_values = timed(fresh)
    prepare_start = time.perf_counter_ns()
    prepared = cc.PreparedMesh(vertices, triangles)
    prepare_ms = (time.perf_counter_ns() - prepare_start) / 1e6
    retained_bytes = prepared.retained_bytes
    retained_first, retained_values = timed(lambda: prepared.distances(points, signed=False))
    first_checks = [check(fresh_values), check(retained_values)]
    stats_bits = statistics_bits(retained_values)
    require(stats_bits == native["oracle"]["statisticsBits"], "statistics differ from native oracle")
    del fresh_values, retained_values
    signed_fresh = fresh(signed=True)
    signed_retained = prepared.distances(points)
    require(np.array_equal(signed_fresh.view(np.uint64), signed_retained.view(np.uint64)), "signed f64 differs")
    signed_info = output_info(signed_retained)
    require(output_info(signed_fresh) == signed_info, "signed scalar PLY differs")
    signed_stats = statistics_bits(signed_fresh)
    require(statistics_bits(signed_retained) == signed_stats, "signed statistics differ")
    del signed_fresh, signed_retained

    def pair(index):
        order = ["fresh", "retained"] if index % 2 == 0 else ["retained", "fresh"]
        timings, values = {}, {}
        for policy in order:
            call = fresh if policy == "fresh" else lambda: prepared.distances(points, signed=False)
            timings[policy], values[policy] = timed(call)
        checks = {policy: check(values[policy]) for policy in order}
        require(prepared.retained_bytes == retained_bytes, "mesh retained capacity changed")
        return {"order": order, "millis": timings, "outputs": checks}

    warmups = [pair(index) for index in range(5)]
    trials = [pair(index + 5) for index in range(args.repeats)]
    fresh_times = [trial["millis"]["fresh"] for trial in trials]
    retained_times = [trial["millis"]["retained"] for trial in trials]
    savings = [a - b for a, b in zip(fresh_times, retained_times)]
    shortening = [100 * (1 - b / a) for a, b in zip(fresh_times, retained_times)]
    saved_median = statistics.median(savings)
    summary = {"freshMedianMillis": statistics.median(fresh_times),
               "retainedMedianMillis": statistics.median(retained_times),
               "pairedShorteningPercent": statistics.median(shortening),
               "pairedSavedMedianMillis": saved_median,
               "fasterTrials": sum(value > 0 for value in savings),
               "setupAmortizationCallsEstimate": math.ceil(prepare_ms / saved_median) if saved_median > 0 else None}
    # Recheck provenance after all timings. Never omit or remove slow observations.
    for path, expected in source_hashes.items():
        require(digest(args.source_root / path) == expected, "source changed during measurement: " + path)
    for info in [*inputs.values(), extension]:
        require(digest(Path(info["path"])) == info["sha256"], "input/build changed during measurement")
    report = {"schemaVersion": 1, "comparison": "python-c2m-prepared-mesh",
              "createdAt": datetime.now(timezone.utc).isoformat(), "threads": args.threads,
              "repeats": args.repeats, "warmupRounds": 5, "sourceHashes": source_hashes,
              "extension": extension, "inputs": inputs, "inputManifest": manifest,
              "mesh": {**native["mesh"], "threads": args.threads},
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "package": cc.__version__, "maturin": importlib.metadata.version("maturin"),
                              "platform": platform.system(), "architecture": platform.machine(),
                              "logicalCpus": os.cpu_count(), "rayonNumThreads": os.environ["RAYON_NUM_THREADS"]},
              "preparation": {"elapsedMillis": prepare_ms, "retainedNativeBytes": retained_bytes,
                              "callerGeometryArrayBytes": vertices.nbytes + triangles.nbytes},
              "firstUse": {"freshMillis": fresh_first, "retainedQueryMillis": retained_first,
                           "retainedWithSetupMillis": prepare_ms + retained_first, "outputs": first_checks},
              "accuracy": {"unsignedNativeOracle": native["oracle"], "statisticsBits": stats_bits,
                           "unsignedOutputsChecked": 2 * (1 + 5 + args.repeats),
                           "signedFreshRetainedOutputsChecked": 2, "signedOutputs": signed_info,
                           "signedStatisticsBits": signed_stats},
              "warmups": warmups, "trials": trials, "summary": summary,
              "note": "Whole Python API calls include NumPy-to-Rust copies and NumPy output creation. "
                      "Unsigned hot calls exclude one-time preparation, loading, TIN construction, serialization, "
                      "statistics, hashing and final mesh destruction. Finite full-source f64, sequential statistics "
                      "and canonical scalar PLY bytes checked outside clocks. No outliers removed. "
                      "Retained bytes are native vector payload capacity, excluding allocator overhead and RSS."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "summary": summary, "preparation": report["preparation"]}))


if __name__ == "__main__":
    main()
