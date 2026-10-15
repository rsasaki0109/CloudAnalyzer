"""Measure complete real-source snapshot comparisons with packed RGB rows.

Run each source/pass in a fresh process, using new work/output destinations.
The baseline is the archived preceding visualization module. Both policies
execute the current comparison, actual I/O/NN/statistics and PNG renderer.
"""

import argparse
from datetime import datetime, timezone
import hashlib
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
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def info(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source-root", "native-source-root", "reference", "compared",
                 "input-manifest", "baseline", "build-index", "prior-colors",
                 "work-dir", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=5)
    options = [value.split("=")[0] for value in sys.argv[1:] if value.startswith("--")]
    require(len(options) == len(set(options)), "duplicate options")
    args = parser.parse_args()
    require(1 <= args.threads <= 64 and 1 <= args.repeats <= 100, "invalid threads/repeats")
    require(not args.output.exists() and not args.work_dir.exists(), "output/work directory must be new")
    require(sha(args.baseline) == args.baseline_sha256, "visualization baseline differs")
    os.environ["RAYON_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["CA_DISABLE_RUST_CORE"] = "0"
    os.environ["CLOUDANALYZER_RENDER_BACKEND"] = "matplotlib"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str((args.source_root / "cloudanalyzer").resolve()))

    import matplotlib
    import numpy as np
    import open3d as o3d
    import cloudanalyzer_core
    import cloudanalyzer_core._core as core
    import ca.compare as comparison
    from ca._rust import core as selected_core

    require(selected_core().nearest_distances is core.nearest_distances, "native NN is not selected")
    logging.getLogger("ca").setLevel(logging.ERROR)
    spec = importlib.util.spec_from_file_location("visualization_strided_baseline", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    policies = dict(strided=baseline.colorize, packed=comparison.colorize)

    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    prior = json.loads(args.prior_colors.read_text(encoding="utf-8"))
    require(prior["comparison"] == "compare-requested-colors", "unexpected preceding record")
    require(prior["inputManifest"] == manifest, "preceding manifest differs")
    require(prior["sourceHashes"]["cloudanalyzer/ca/visualization.py"] == args.baseline_sha256,
            "preceding visualization differs")
    for key in ["reference", "compared"]:
        require(sha(getattr(args, key)) == manifest["inputs"][key]["sha256"]
                == prior["inputs"][key]["sha256"], key + " differs")
    distance_record = prior["inputs"]["distance_record"]
    require(sha(distance_record["path"]) == distance_record["sha256"], "distance record differs")
    distance_prior = json.loads(Path(distance_record["path"]).read_text(encoding="utf-8"))
    require(distance_prior["inputManifest"] == manifest, "distance manifest differs")
    snapshot = distance_prior["registration"]["snapshot"]
    require(sha(snapshot["path"]) == snapshot["sha256"], "coordinate snapshot differs")
    with np.load(snapshot["path"]) as data:
        coordinates = {key: data[key] for key in ["initial", "reference"]}
    for key, path in [("initial", args.compared), ("reference", args.reference)]:
        cloud = comparison.load_point_cloud(str(path))
        require(np.asarray(cloud.points).tobytes() == coordinates[key].tobytes(), "coordinates differ")
        require(not cloud.has_colors() and not cloud.has_normals() and not cloud.has_covariances(),
                "unexpected input attributes")
    del cloud
    original = (args.source_root / "web" / manifest["source"]["path"]).resolve()
    require(sha(original) == manifest["source"]["sha256"], "original source differs")
    build = json.loads(args.build_index.read_text(encoding="utf-8"))
    extension = info(core.__file__)
    require(extension["sha256"] == build["extension"]["sha256"], "native extension differs")
    for path, expected in build["sourceHashes"].items():
        require(sha(args.native_source_root / path) == expected, "archived native source differs: " + path)

    files = [*sorted((args.source_root / "cloudanalyzer/ca").glob("*.py")),
             args.source_root / "cloudanalyzer/pyproject.toml",
             args.source_root / "cloudanalyzer/tests/test_visualization_packed_colors.py",
             args.source_root / "cloudanalyzer/tests/test_visualization.py",
             args.source_root / "cloudanalyzer/tests/test_compare_color_consumers.py",
             Path(__file__).resolve()]
    hashes = {path.relative_to(args.source_root.resolve()).as_posix(): sha(path) for path in files}
    inputs = {key: info(getattr(args, key)) for key in ["reference", "compared", "input_manifest",
                                                      "baseline", "build_index", "prior_colors"]}
    inputs.update(originalSource=info(original), coordinateSnapshot=info(snapshot["path"]),
                  distanceRecord=info(distance_record["path"]))
    libraries = dict(open3d=info(o3d.cpu.pybind.__file__), wrapper=info(cloudanalyzer_core.__file__))
    args.work_dir.mkdir(parents=True, exist_ok=False)
    count = 0

    def call(policy, capture=False, reports=False):
        nonlocal count
        count += 1
        destination = args.work_dir / f"{count:03d}-{policy}"
        destination.mkdir(exist_ok=False)
        image = destination / "snapshot.png"
        query = comparison.compute_nn_distance
        color = policies[policy]
        captured = {}

        def traced_query(source, target):
            require(np.asarray(source.points).tobytes() == coordinates["initial"].tobytes(), "query source differs")
            require(np.asarray(target.points).tobytes() == coordinates["reference"].tobytes(), "query target differs")
            require("distance" not in captured, "extra query")
            result = query(source, target)
            captured["distance"] = result.copy()
            return result

        def traced_color(cloud, values):
            require("colors" not in captured, "extra color call")
            result = color(cloud, values)
            captured["colors"] = np.asarray(cloud.colors).copy()
            return result

        comparison.colorize = traced_color if capture else color
        if capture:
            comparison.compute_nn_distance = traced_query
        kwargs = dict(method=None, threshold=.5, snapshot_path=str(image))
        if reports:
            kwargs.update(json_path=str(destination / "report.json"), report_path=str(destination / "report.md"))
        start = time.perf_counter_ns()
        try:
            report = comparison.run_compare(str(args.compared), str(args.reference), **kwargs)
        finally:
            comparison.compute_nn_distance = query
            comparison.colorize = policies["packed"]
        elapsed = (time.perf_counter_ns() - start) / 1e6
        saved = {}
        if capture:
            require(set(captured) == {"distance", "colors"}, "query/color call count differs")
            for key, array in captured.items():
                path = destination / (key + "-f64.bin")
                with path.open("xb") as stream:
                    stream.write(array.astype("<f8", copy=False).tobytes())
                saved[key] = info(path)
        output_files = dict(png=info(image))
        if reports:
            output_files.update({key: info(destination / ("report." + key)) for key in ["json", "md"]})
        return dict(millis=elapsed, report=report, captured=saved, outputFiles=output_files)

    names = list(policies)
    parity = {policy: call(policy, capture=True, reports=True) for policy in names}
    expected_report = parity["strided"]["report"]
    expected_png = parity["strided"]["outputFiles"]["png"]["sha256"]

    def check(item):
        require(item["report"] == expected_report, "complete report differs")
        require(item["outputFiles"]["png"]["sha256"] == expected_png, "PNG bytes differ")

    for item in parity.values():
        check(item)
        for key in ["distance", "colors"]:
            require(item["captured"][key]["sha256"] == prior["parity"]["always"]["captured"][key]["sha256"],
                    "full preceding " + key + " bytes differ")
        require(item["captured"]["colors"]["bytes"] == 24 * manifest["inputs"]["compared"]["points"],
                "color payload differs")
    for key in ["json", "md"]:
        require(parity["strided"]["outputFiles"][key]["sha256"]
                == parity["packed"]["outputFiles"][key]["sha256"], "written report bytes differ")
    print(json.dumps(dict(stage="parity_complete")), flush=True)

    def pair(index):
        order = names if index % 2 == 0 else list(reversed(names))
        observed = {policy: call(policy) for policy in order}
        for item in observed.values():
            check(item)
        return dict(order=order, observations=observed)

    first = pair(0)
    warmups = [pair(index + 1) for index in range(2)]
    trials = []
    for index in range(args.repeats):
        trial = pair(index + 3)
        trials.append(trial)
        print(json.dumps(dict(stage="trial_complete", trial=index+1,
                             millis={key: value["millis"] for key, value in trial["observations"].items()})), flush=True)
    a = [item["observations"]["strided"]["millis"] for item in trials]
    b = [item["observations"]["packed"]["millis"] for item in trials]
    summary = dict(stridedMedianMillis=statistics.median(a), packedMedianMillis=statistics.median(b),
                   pairedShorteningPercent=statistics.median(100 * (1 - y/x) for x, y in zip(a, b)),
                   pairedSavedMedianMillis=statistics.median(x-y for x, y in zip(a, b)),
                   fasterTrials=sum(x > y for x, y in zip(a, b)))
    for path, expected in hashes.items():
        require(sha(args.source_root / path) == expected, "Python source changed: " + path)
    for path, expected in build["sourceHashes"].items():
        require(sha(args.native_source_root / path) == expected, "archived native source changed: " + path)
    for item in [*inputs.values(), extension, *libraries.values()]:
        require(sha(item["path"]) == item["sha256"], "input/build/library changed")
    report = dict(schemaVersion=1, comparison="snapshot-packed-rgb", createdAt=datetime.now(timezone.utc).isoformat(),
                  sourceHashes=hashes, nativeSourceRoot=str(args.native_source_root.resolve()),
                  nativeSourceHashes=build["sourceHashes"], inputs=inputs, inputManifest=manifest,
                  extension=extension, libraries=libraries,
                  environment=dict(python=platform.python_version(), numpy=np.__version__, open3d=o3d.__version__,
                                   matplotlib=matplotlib.__version__, platform=platform.system(), logicalCpus=os.cpu_count(),
                                   rayonNumThreads=args.threads, ompNumThreads=args.threads, openblasNumThreads=1,
                                   renderBackend="matplotlib", imageWidth=1920, imageHeight=1080, rendererMaxPoints=200000),
                  repeats=args.repeats, warmupRounds=2, parity=parity, firstUntraced=first,
                  warmups=warmups, trials=trials, summary=summary,
                  note="Complete run_compare with method=None, threshold=.5 and PNG output, using full unchanged real-source "
                       "geometry and actual native NN. Normalization/colormap/owned RGB assignment, deterministic Matplotlib "
                       "projection, PNG encoding/write and function cleanup included. Renderer retains its existing 200000-point "
                       "display cap; distances and coloring use all compared points. Only colorize changes between policies. "
                       "Directory creation, imports/hashing, output validation, CLI serialization, traced parity and extra "
                       "JSON/Markdown writes are excluded from timing summaries. Excluded first untraced pair after parity, "
                       "two warm-up pairs, then five default alternating timed pairs, starting packed. Every report and PNG "
                       "is checked. Fresh process per source/pass, warmed pages. Native sources come from the installed "
                       "binary's archived build, not the later checkout. No cold-process, peak RSS, default registered "
                       "comparison, interactive Open3D renderer, GPU or independent-survey accuracy claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), summary=summary)), flush=True)


if __name__ == "__main__":
    main()
