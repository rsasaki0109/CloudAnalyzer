"""Compare full Python nearest-distance calls against a frozen preceding wheel.

Use mode oracle with the old wheel first, then compare with a prepared-reference
wheel. Source archives, extension binaries and original real inputs are verified.
All hashing, statistics and canonical scalar PLY reconstruction are outside clocks.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import struct
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def info(path):
    return {"path":str(Path(path).resolve()),"bytes":Path(path).stat().st_size,"sha256":sha(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["oracle","compare"],required=True)
    for option in ["source-root","reference","compared","input-manifest","build-index","build-root","work-dir","output"]:
        parser.add_argument("--"+option,type=Path,required=True)
    parser.add_argument("--oracle",type=Path)
    parser.add_argument("--threads",type=int,default=12)
    parser.add_argument("--repeats",type=int,default=10)
    options = [arg.split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")]
    require(len(options)==len(set(options)),"duplicate options")
    args = parser.parse_args()
    require(1<=args.threads<=64 and 1<=args.repeats<=100,"invalid threads/repeats")
    require(not args.output.exists() and not args.work_dir.exists(),"output/work directory must be new")
    require((args.mode=="compare") == (args.oracle is not None),"compare requires oracle; oracle mode omits it")
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import cloudanalyzer_core as cc
    import cloudanalyzer_core._core as core

    build = json.loads(args.build_index.read_text(encoding="utf-8"))
    for path,expected in build["sourceHashes"].items():
        require(sha(args.build_root/path)==expected,"archived source fingerprint differs: "+path)
    extension = info(core.__file__)
    require(extension["sha256"]==build["extension"]["sha256"],"loaded extension differs from build")
    package = Path(cc.__file__).parent
    archived_package = args.build_root/"rust/crates/ca-py/python/cloudanalyzer_core"
    for source in archived_package.glob("*"):
        if source.is_file() and source.suffix in {".py",".pyi"}:
            require(sha(source)==sha(package/source.name),"loaded Python package differs: "+source.name)
    if args.mode=="compare":
        for path,expected in build["sourceHashes"].items():
            require(sha(args.source_root/path)==expected,"current source differs: "+path)
    manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    require(manifest["schemaVersion"]==1,"unexpected manifest")
    for key in ["reference","compared"]:
        path = getattr(args,key)
        require(sha(path)==manifest["inputs"][key]["sha256"],key+" fingerprint differs")
        require(path.stat().st_size==manifest["inputs"][key]["bytes"],key+" length differs")
    original = (args.source_root/"web"/manifest["source"]["path"]).resolve()
    require(sha(original)==manifest["source"]["sha256"],"original measured source differs")
    inputs = {"reference":info(args.reference),"compared":info(args.compared),
              "manifest":info(args.input_manifest),"originalSource":info(original),"buildIndex":info(args.build_index)}
    reference = cc.read(str(args.reference))["positions"]
    source = cc.read(str(args.compared))["positions"]
    require(len(reference)==manifest["inputs"]["reference"]["points"],"reference count differs")
    require(len(source)==manifest["inputs"]["compared"]["points"],"query count differs")
    require(200000<=len(source)<=1000000 and np.isfinite(source).all() and np.isfinite(reference).all(),"expected finite full real inputs")
    header = ("ply\nformat binary_little_endian 1.0\ncomment CloudAnalyzer Web\n"
              f"element vertex {len(source)}\nproperty double x\nproperty double y\nproperty double z\n"
              "property float scalar_C2C\nend_header\n").encode()

    def output(values):
        require(values.dtype==np.float64 and values.shape==(len(source),),"invalid result array")
        raw = values.astype("<f8",copy=False).tobytes()
        records = np.empty(len(source),dtype=[("xyz","<f8",(3,)),("distance","<f4")])
        records["xyz"],records["distance"] = source,values
        ply = header+records.tobytes()
        return {"distanceBytes":len(raw),"distanceSha256":hashlib.sha256(raw).hexdigest(),
                "scalarPlyBytes":len(ply),"scalarPlySha256":hashlib.sha256(ply).hexdigest()},raw,ply

    def stats(values):
        total = squared = 0.
        for value in values:
            value = float(value)
            total += value
            squared += value*value
        n = len(values)
        mean = total/n
        ordered = np.sort(values)
        median = float(ordered[n//2]) if n%2 else .5*(float(ordered[n//2-1])+float(ordered[n//2]))
        result = [float(values.min()),float(values.max()),mean,math.sqrt(squared/n),
                  math.sqrt(max(squared/n-mean*mean,0.)),median]
        return [struct.pack(">d",value).hex() for value in result]

    def verify_files():
        for path,expected in build["sourceHashes"].items():
            require(sha(args.build_root/path)==expected,"archive changed: "+path)
            if args.mode=="compare":
                require(sha(args.source_root/path)==expected,"current source changed: "+path)
        for item in [*inputs.values(),extension]:
            require(sha(item["path"])==item["sha256"],"input/build changed")

    metadata = {"schemaVersion":1,"createdAt":datetime.now(timezone.utc).isoformat(),"inputs":inputs,
                "inputManifest":manifest,"extension":extension,"build":build,
                "scriptSha256":sha(__file__),"threads":args.threads,
                "environment":{"python":platform.python_version(),"numpy":np.__version__,"package":cc.__version__,
                               "platform":platform.system(),"architecture":platform.machine(),"logicalCpus":os.cpu_count()}}
    if args.mode=="oracle":
        values = cc.nearest_distances(source,reference)
        result,raw,ply = output(values)
        statistics_bits = stats(values)
        args.work_dir.mkdir(parents=True,exist_ok=False)
        (args.work_dir/"distances-f64.bin").write_bytes(raw)
        (args.work_dir/"distances.ply").write_bytes(ply)
        verify_files()
        report = {**metadata,"comparison":"python-c2c-reference-oracle","result":result,"statisticsBits":statistics_bits,
                  "distanceFile":info(args.work_dir/"distances-f64.bin"),"plyFile":info(args.work_dir/"distances.ply")}
    else:
        oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
        require(oracle["comparison"]=="python-c2c-reference-oracle","unexpected oracle")
        require(oracle["inputManifest"]==manifest,"oracle manifest differs")
        for key in ["reference","compared","manifest","originalSource"]:
            require(inputs[key]["sha256"]==oracle["inputs"][key]["sha256"],"oracle input differs: "+key)
        require(build["compiler"]==oracle["build"]["compiler"],"compiler differs")
        require(metadata["environment"]==oracle["environment"],"runtime environment differs")
        changed = {"rust/crates/ca-core/src/distance.rs","rust/crates/ca-py/src/lib.rs",
                   "rust/crates/ca-py/python/cloudanalyzer_core/__init__.py","rust/crates/ca-py/python/cloudanalyzer_core/_core.pyi"}
        for path,old in oracle["build"]["sourceHashes"].items():
            if path not in changed:
                require(build["sourceHashes"][path]==old,"unrelated source/config changed: "+path)
        for item in [oracle["distanceFile"],oracle["plyFile"]]:
            require(sha(item["path"])==item["sha256"],"oracle output changed")
        inputs["oracle"] = info(args.oracle)
        args.work_dir.mkdir(parents=True,exist_ok=False)

        def check(values):
            result,_,_ = output(values)
            require(result==oracle["result"],"complete result differs from preceding oracle")
            return result

        def timed(call):
            start = time.perf_counter_ns()
            values = call()
            return (time.perf_counter_ns()-start)/1e6,values

        fresh_ms,fresh_values = timed(lambda:cc.nearest_distances(source,reference))
        start = time.perf_counter_ns()
        prepared = cc.PreparedReference(reference)
        setup_ms = (time.perf_counter_ns()-start)/1e6
        capacity = prepared.retained_bytes
        retained_ms,retained_values = timed(lambda:prepared.distances(source))
        first = {"freshMillis":fresh_ms,"retainedQueryMillis":retained_ms,"retainedWithSetupMillis":setup_ms+retained_ms,
                 "outputs":[check(fresh_values),check(retained_values)]}
        require(stats(fresh_values)==stats(retained_values)==oracle["statisticsBits"],"six f64 statistics differ")
        # Reverse the original measured coordinates to exercise changed query order.
        # No generated/transformed positions enter this check or any performance call.
        reverse = prepared.distances(source[::-1])
        require(np.array_equal(reverse.view(np.uint64),retained_values[::-1].view(np.uint64)),"changed query order differs")
        del fresh_values,retained_values,reverse

        def pair(index):
            order = ["fresh","retained"] if index%2==0 else ["retained","fresh"]
            timings,values = {},{}
            for policy in order:
                call = (lambda:cc.nearest_distances(source,reference)) if policy=="fresh" else (lambda:prepared.distances(source))
                timings[policy],values[policy] = timed(call)
            checked = {policy:check(values[policy]) for policy in order}
            require(prepared.retained_bytes==capacity,"retained tree capacity changed")
            return {"order":order,"millis":timings,"outputs":checked}

        warmups = [pair(index) for index in range(5)]
        trials = [pair(index+5) for index in range(args.repeats)]
        a = [trial["millis"]["fresh"] for trial in trials]
        b = [trial["millis"]["retained"] for trial in trials]
        saved = statistics.median([x-y for x,y in zip(a,b)])
        summary = {"freshMedianMillis":statistics.median(a),"retainedMedianMillis":statistics.median(b),
                   "pairedShorteningPercent":statistics.median([100*(1-y/x) for x,y in zip(a,b)]),
                   "pairedSavedMedianMillis":saved,"fasterTrials":sum(x>y for x,y in zip(a,b)),
                   "setupAmortizationCallsEstimate":math.ceil(setup_ms/saved) if saved>0 else None}
        verify_files()
        for item in [oracle["distanceFile"],oracle["plyFile"]]:
            require(sha(item["path"])==item["sha256"],"oracle output changed during comparison")
        report = {**metadata,"comparison":"python-c2c-prepared-reference","oracle":oracle,
                  "repeats":args.repeats,"warmupRounds":5,"firstUse":first,"warmups":warmups,"trials":trials,"summary":summary,
                  "preparation":{"elapsedMillis":setup_ms,"retainedNativeBytes":capacity,"callerReferenceArrayBytes":reference.nbytes},
                  "accuracy":{"fullOutputsChecked":2*(1+5+args.repeats),"statisticsBits":oracle["statisticsBits"],"reversedQueryOrderChecked":True},
                  "note":"Whole Python calls include source-array copies, Morton work, parallel searches, scatter and NumPy output creation. "
                         "Fresh calls also copy the reference and build/drop its tree. One-time retained setup and final object destruction "
                         "are outside repeated-call clocks. Loading, first/warm-up calls, statistics, canonical PLY serialization and hashing "
                         "are excluded. Complete f64/PLY bytes match an independent preceding release wheel. No outliers removed. "
                         "Native retained bytes are vector payload capacity, excluding caller arrays, query/result scratch, allocator overhead and RSS. "
                         "This opt-in API does not change CLI/browser/WASM/GPU selection."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output":str(args.output),"comparison":report["comparison"],"summary":report.get("summary")}),flush=True)


if __name__=="__main__":
    main()
