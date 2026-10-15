"""Measure complete LAS/LAZ loads through the optional-core-disabled reader.

Use full existing measured files and an independently decoded full XYZ oracle.
Run each dataset/pass in a fresh process with new output/work destinations.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
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
    for name in ["source-root", "baseline", "input-manifest", "coordinate-oracle", "work-dir", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--dataset", required=True, choices=["autoware-planning", "autzen"])
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--start-policy", choices=["strided", "packed"], default="strided")
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
    import laspy
    import lazrs
    import numpy as np
    import open3d as o3d
    import ca.io as current

    require(current.core() is None, "optional native reader must be disabled")
    spec = importlib.util.spec_from_file_location("io_strided_las_baseline", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    require(baseline.core() is None, "baseline native reader must be disabled")
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    entry = manifest["sources"][args.dataset]
    source = args.source_root / entry["path"]
    require(sha(source) == entry["sha256"] and source.stat().st_size == entry["bytes"], "source differs")
    require(sha(args.coordinate_oracle) == entry["coordinateOracle"]["sha256"], "coordinate oracle differs")
    require(args.coordinate_oracle.stat().st_size == 24 * entry["points"], "coordinate size differs")
    with laspy.open(source) as reader:
        require(reader.header.point_count == entry["points"] and reader.header.point_format.id == entry["pointFormat"],
                "source header differs")
        require(reader.header.scales.tolist() == entry["scales"] and reader.header.offsets.tolist() == entry["offsets"],
                "source scaling differs")
    oracle = np.memmap(args.coordinate_oracle, mode="r", dtype="<f8", shape=(entry["points"], 3))
    paths = ["cloudanalyzer/ca/io.py", "cloudanalyzer/ca/_rust.py", "cloudanalyzer/ca/__init__.py",
             "cloudanalyzer/pyproject.toml", "cloudanalyzer/tests/test_io_las_packed_coordinates.py",
             "cloudanalyzer/tests/test_io.py", "cloudanalyzer/tests/test_io_binary_xyz_ply.py",
             "cloudanalyzer/scripts/benchmark_las_coordinates.py"]
    source_hashes = {path: sha(args.source_root / path) for path in paths}
    package_root = Path(laspy.__file__).parent
    laspy_hashes = {str(path.relative_to(package_root)): sha(path) for path in sorted(package_root.rglob("*.py"))}
    decoder = list(Path(lazrs.__file__).parent.glob("*.pyd")) + list(Path(lazrs.__file__).parent.glob("*.so"))
    require(len(decoder) == 1, "expected one lazrs extension")
    libraries = dict(open3d=info(o3d.cpu.pybind.__file__), lazrs=info(decoder[0]), lazrsWrapper=info(lazrs.__file__))
    inputs = dict(source=info(source), manifest=info(args.input_manifest), oracle=info(args.coordinate_oracle), baseline=info(args.baseline))
    provenance = entry["provenance"]
    if "originalPath" in provenance:
        original = args.source_root / provenance["originalPath"]
        require(sha(original) == provenance["originalSha256"], "original measured source differs")
        inputs["originalSource"] = info(original)
    if "recordPath" in provenance:
        prior = args.source_root / provenance["recordPath"]
        require(sha(prior) == provenance["recordSha256"], "source provenance differs")
        inputs["provenance"] = info(prior)
    args.work_dir.mkdir(parents=True, exist_ok=False)
    modules = dict(strided=baseline, packed=current)
    names = [args.start_policy, "packed" if args.start_policy == "strided" else "strided"]

    def call(policy, full_check=False):
        start = time.perf_counter_ns()
        cloud = modules[policy].load_point_cloud(str(source))
        elapsed = (time.perf_counter_ns() - start) / 1e6
        points = np.asarray(cloud.points)
        signature = dict(shape=list(points.shape), dtype=points.dtype.str,
                         xyzSha256=hashlib.sha256(memoryview(points)).hexdigest(),
                         colors=len(cloud.colors), normals=len(cloud.normals), covariances=len(cloud.covariances))
        require(signature == dict(shape=[entry["points"], 3], dtype="<f8",
                                  xyzSha256=entry["coordinateOracle"]["sha256"], colors=0, normals=0, covariances=0),
                "complete loaded output differs")
        if full_check:
            for start in range(0, len(points), 250000):
                require(points[start:start+250000].tobytes() == oracle[start:start+250000].tobytes(),
                        "full XYZ bytes differ")
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
    a = [item["observations"]["strided"]["millis"] for item in trials]
    b = [item["observations"]["packed"]["millis"] for item in trials]
    summary = dict(stridedMedianMillis=statistics.median(a), packedMedianMillis=statistics.median(b),
                   pairedShorteningPercent=statistics.median(100 * (1-y/x) for x, y in zip(a, b)),
                   pairedSavedMedianMillis=statistics.median(x-y for x, y in zip(a, b)),
                   fasterTrials=sum(x > y for x, y in zip(a, b)))
    for path, expected in source_hashes.items():
        require(sha(args.source_root / path) == expected, "source changed: " + path)
    for path, expected in laspy_hashes.items():
        require(sha(package_root / path) == expected, "laspy changed: " + path)
    for item in [*inputs.values(), *libraries.values()]:
        require(sha(item["path"]) == item["sha256"], "input/library changed")
    record = dict(schemaVersion=1, comparison="las-packed-coordinates", createdAt=datetime.now(timezone.utc).isoformat(),
                  dataset=args.dataset, inputManifest=manifest, inputs=inputs, sourceHashes=source_hashes,
                  laspySourceRoot=str(package_root), laspySourceHashes=laspy_hashes, libraries=libraries,
                  environment=dict(platform=platform.system(), python=platform.python_version(), numpy=np.__version__,
                                   open3d=o3d.__version__, laspy=laspy.__version__, lazrs=importlib.metadata.version("lazrs"),
                                   logicalCpus=os.cpu_count(), rayonNumThreads=args.threads, ompNumThreads=args.threads,
                                   openblasNumThreads=1, disableRustCore=True,
                                   availableLazBackends=[str(value) for value in laspy.LazBackend if value.is_available()]),
                  startPolicy=args.start_policy, repeats=args.repeats, warmupRounds=2, parity=parity,
                  firstUntraced=first, warmups=warmups, trials=trials, summary=summary,
                  note="Complete load_point_cloud call with optional Rust core disabled: actual whole-file LAS/LAZ decoding, "
                       "header coordinate scaling, XYZ assembly, owned Open3D conversion, empty check and internal cleanup. "
                       "Returned cloud release, full-output hashing/validation, imports, fingerprinting and CLI serialization "
                       "excluded from timing. Same full existing measured source in decoded order; no generated geometry, "
                       "new thinning, perturbation or decoder replacement. Both policies check all output coordinates and "
                       "empty attributes after every call. Excluded full byte-parity pair, separate first pair, two warm-up "
                       "pairs, then five default alternating timed pairs. Fresh process per dataset/pass, balanced initial "
                       "policy across passes, warmed file pages. No native-reader, streaming/spatial COPC, whole comparison, "
                       "cold CLI, peak RSS or GPU speed claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), summary=summary)), flush=True)


if __name__ == "__main__":
    main()
