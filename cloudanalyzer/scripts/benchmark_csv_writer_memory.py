"""Measure traced allocations for one complete CSV save in a fresh process.

Run separately from wall-clock trials. This measures tracemalloc allocations,
including NumPy's registered array buffers, rather than process peak RSS.
"""

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import tracemalloc

from benchmark_csv_writer import info, require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ["source-root", "timing-record", "work-dir", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--policy", choices=["whole_rows", "batch_rows"], required=True)
    options = [value.split("=")[0] for value in sys.argv[1:] if value.startswith("--")]
    require(len(options) == len(set(options)), "duplicate options")
    args = parser.parse_args()
    require(not args.output.exists() and not args.work_dir.exists(), "output/work directory must be new")
    record = json.loads(args.timing_record.read_text(encoding="utf-8"))
    require(record["comparison"] == "csv-batched-write", "incorrect timing record")
    args.source_root = args.source_root.resolve()
    for relative, expected in record["sourceHashes"].items():
        require(sha(args.source_root / relative) == expected, "source differs: " + relative)
    for item in [*record["inputs"].values(), *record["libraries"].values()]:
        require(sha(item["path"]) == item["sha256"], "input/library differs")
    os.environ["CA_DISABLE_RUST_CORE"] = "1"
    os.environ["OMP_NUM_THREADS"] = os.environ["RAYON_NUM_THREADS"] = str(record["environment"]["ompNumThreads"])
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.source_root / "cloudanalyzer"))
    import numpy as np
    import open3d as o3d
    import ca.io as current

    baseline_path = Path(record["inputs"]["baseline"]["path"])
    spec = importlib.util.spec_from_file_location("io_csv_memory_baseline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    require(current.core() is None, "optional core must be disabled")
    require(info(o3d.cpu.pybind.__file__)["sha256"] == record["libraries"]["open3d"]["sha256"], "loaded Open3D differs")
    require(info(np._core._multiarray_umath.__file__)["sha256"] == record["libraries"]["numpyCore"]["sha256"], "loaded NumPy differs")
    cloud = o3d.io.read_point_cloud(record["inputs"]["geometry"]["path"])
    oracle = np.memmap(record["inputs"]["coordinateOracle"]["path"], mode="r", dtype="<f8", shape=(record["points"], 3))
    require(np.asarray(cloud.points).tobytes() == oracle.tobytes(), "input coordinates differ")
    controller = info(__file__)
    timing = info(args.timing_record)
    args.work_dir.mkdir(parents=True, exist_ok=False)
    path = args.work_dir / "points.csv"
    module = baseline if args.policy == "whole_rows" else current
    gc.collect()
    tracemalloc.start()
    try:
        result = module.save_point_cloud(str(path), cloud)
        residual, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    require(result is None, "save return value differs")
    output = info(path)
    require({key: output[key] for key in ["bytes", "sha256"]} == record["parity"][args.policy]["output"], "CSV output differs")
    require(np.asarray(cloud.points).tobytes() == oracle.tobytes(), "source cloud changed")
    for relative, expected in record["sourceHashes"].items():
        require(sha(args.source_root / relative) == expected, "source changed: " + relative)
    for item in [*record["inputs"].values(), *record["libraries"].values(), controller, timing]:
        require(sha(item["path"]) == item["sha256"], "input/library/controller changed")
    result = dict(schemaVersion=1, comparison="csv-write-traced-allocations", dataset=record["dataset"],
                  points=record["points"], policy=args.policy, peakTracedBytes=peak, residualTracedBytes=residual,
                  timingRecord=timing, controller=controller, output=output, environment=record["environment"],
                  note="One fresh process per dataset/policy; cloud load, imports, validation and GC collection "
                       "precede tracing. Complete save_point_cloud runs under tracemalloc, including snapshot "
                       "copy, row materialization, formatting, actual file writes and close. No wall-clock "
                       "measurement is taken. Peak includes registered NumPy buffers and traced Python "
                       "allocations, but excludes untraced native/OS buffers, source-cloud storage and tracer "
                       "metadata. This is not process peak RSS, total application memory or a time result.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(dataset=record["dataset"], policy=args.policy, peakTracedBytes=peak)), flush=True)


if __name__ == "__main__":
    main()
