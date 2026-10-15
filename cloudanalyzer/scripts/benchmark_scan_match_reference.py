"""Compare complete scan-match diagnostics with per-invocation tree preparation.

Loading and the recorded real ICP solve are excluded; tree construction,
preprocessing, both changed query batches, summaries and cleanup are timed.
Full PLY parity is checked outside clocks when artifact timing is disabled.
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
    return {"path":str(Path(path).resolve()),"bytes":Path(path).stat().st_size,"sha256":sha(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ["source-root","reference","compared","input-manifest","baseline","registration-record","work-dir","output"]:
        parser.add_argument("--"+option,type=Path,required=True)
    parser.add_argument("--baseline-sha256",required=True)
    parser.add_argument("--threads",type=int,default=12)
    parser.add_argument("--repeats",type=int,default=10)
    parser.add_argument("--with-artifacts",action="store_true")
    options=[value.split("=")[0] for value in sys.argv[1:] if value.startswith("--")]
    require(len(options)==len(set(options)),"duplicate options")
    args=parser.parse_args()
    require(1<=args.threads<=64 and 1<=args.repeats<=100,"invalid threads/repeats")
    require(not args.output.exists() and not args.work_dir.exists(),"output/work directory must be new")
    require(sha(args.baseline)==args.baseline_sha256,"baseline fingerprint differs")
    os.environ["RAYON_NUM_THREADS"]=os.environ["OMP_NUM_THREADS"]=str(args.threads)
    os.environ["CA_DISABLE_RUST_CORE"]="0"
    sys.path.insert(0,str((args.source_root/"cloudanalyzer").resolve()))
    import numpy as np
    import open3d as o3d
    import matplotlib
    import cloudanalyzer_core as cc
    import cloudanalyzer_core._core as core
    import ca.scan_match_debug as current
    from ca.io import load_point_cloud
    from ca.registration import RegistrationResult

    require(hasattr(cc,"PreparedReference"),"extension lacks prepared references")
    manifest=json.loads(args.input_manifest.read_text(encoding="utf-8"))
    prior=json.loads(args.registration_record.read_text(encoding="utf-8"))
    require(prior["comparison"]=="scan-match-distance-reuse","unexpected registration record")
    require(manifest==prior["inputManifest"],"registration manifest differs")
    for key in ["reference","compared"]:
        require(sha(getattr(args,key))==manifest["inputs"][key]["sha256"]==prior["inputs"][key]["sha256"],key+" differs")
    original=(args.source_root/"web"/manifest["source"]["path"]).resolve()
    require(sha(original)==manifest["source"]["sha256"],"original measured source differs")
    snapshot=prior["registration"]["snapshot"]
    require(sha(snapshot["path"])==snapshot["sha256"],"recorded registration snapshot differs")
    data=np.load(snapshot["path"])
    scan=load_point_cloud(str(args.compared))
    reference=load_point_cloud(str(args.reference))
    for key,cloud in [("initial",scan),("reference",reference)]:
        require(np.array_equal(data[key].view(np.uint64),np.asarray(cloud.points).view(np.uint64)),"recorded points differ: "+key)
    matrix=np.array(prior["registration"]["transformation"])
    require(np.array_equal(matrix.view(np.uint64),data["transformation"].view(np.uint64)),"recorded matrix differs")
    transformed=o3d.geometry.PointCloud(scan)
    transformed.normals=o3d.utility.Vector3dVector(data["initial_normals"])
    transformed.transform(matrix)
    require(np.array_equal(np.asarray(transformed.points).view(np.uint64),data["aligned"].view(np.uint64)),"recorded aligned points differ")
    require(np.array_equal(np.asarray(transformed.normals).view(np.uint64),data["aligned_normals"].view(np.uint64)),"recorded aligned normals differ")
    registration=RegistrationResult(transformed,prior["registration"]["fitness"],prior["registration"]["rmse"],matrix)
    spec=importlib.util.spec_from_file_location("scan_match_reference_preceding",args.baseline)
    baseline=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    files=["cloudanalyzer/ca/scan_match_debug.py","cloudanalyzer/ca/metrics.py","cloudanalyzer/ca/registration.py",
           "cloudanalyzer/ca/io.py","cloudanalyzer/ca/visualization.py","cloudanalyzer/ca/_rust.py","cloudanalyzer/pyproject.toml",
           "cloudanalyzer/tests/test_scan_match_debug.py","cloudanalyzer/tests/test_metrics.py",
           "cloudanalyzer/scripts/benchmark_scan_match_reference.py"]
    hashes={path:sha(args.source_root/path) for path in files}
    inputs={"reference":info(args.reference),"compared":info(args.compared),"manifest":info(args.input_manifest),
            "originalSource":info(original),"baseline":info(args.baseline),"registrationRecord":info(args.registration_record),
            "registrationSnapshot":info(snapshot["path"])}
    extension=info(core.__file__)
    args.work_dir.mkdir(parents=True,exist_ok=False)

    def fixed_registration(source,target,**kwargs):
        source.normals=o3d.utility.Vector3dVector(data["initial_normals"])
        target.normals=o3d.utility.Vector3dVector(data["reference_normals"])
        return registration

    for module in [baseline,current]:
        module.load_point_cloud=lambda path:scan if path==str(args.compared) else reference
        module.register_detailed=fixed_registration

    def call(policy,with_artifacts,capture=False):
        module=baseline if policy=="fresh" else current
        function=module.compute_nn_distance
        prepare=getattr(module,"prepare_nn_reference",None)
        captured,capacities=[],[]
        if capture:
            def traced(source,target):
                values=function(source,target)
                captured.append(values)
                return values
            module.compute_nn_distance=traced
            if prepare is not None:
                def traced_prepare(target):
                    native=prepare(target)
                    require(native is not None,"reference preparation missing")
                    capacities.append(native.retained_bytes)
                    class TracedReference:
                        def __init__(self,native):self.native=native
                        def distances(self,source):
                            values=self.native.distances(source)
                            captured.append(values)
                            return values
                    return TracedReference(native)
                module.prepare_nn_reference=traced_prepare
        start=time.perf_counter_ns()
        try:
            result=module.run_scan_match_debug(str(args.compared),str(args.reference),method="icp",threshold=.5,
                                              artifact_dir=str(args.work_dir/policy) if with_artifacts else None)
        finally:
            module.compute_nn_distance=function
            if prepare is not None:module.prepare_nn_reference=prepare
        millis=(time.perf_counter_ns()-start)/1e6
        artifacts={key:{"file":Path(path).name,"bytes":Path(path).stat().st_size,"sha256":sha(path)} for key,path in result["artifacts"].items()}
        result["artifacts"]={key:value["file"] for key,value in artifacts.items()}
        arrays=[{"bytes":values.nbytes,"sha256":hashlib.sha256(values.astype("<f8",copy=False).tobytes()).hexdigest()} for values in captured]
        return {"millis":millis,"report":result,"artifacts":artifacts,"capturedDistances":arrays,"referenceCapacities":capacities}

    # One complete artifact correctness pair, outside all performance summaries.
    parity={policy:call(policy,True,capture=True) for policy in ["fresh","reuse"]}
    require(parity["fresh"]["report"]==parity["reuse"]["report"],"complete report differs")
    require(parity["fresh"]["artifacts"]==parity["reuse"]["artifacts"],"complete PLY bytes differ")
    require(parity["fresh"]["capturedDistances"]==parity["reuse"]["capturedDistances"],"full f64 arrays differ")
    require(len(parity["fresh"]["capturedDistances"])==2,"unexpected queries")
    require(parity["fresh"]["capturedDistances"]==prior["firstUse"]["fresh"]["capturedDistances"][:2],"preceding independent distances differ")
    require(parity["fresh"]["referenceCapacities"]==[] and len(parity["reuse"]["referenceCapacities"])==1,"unexpected construction count")
    first={policy:call(policy,args.with_artifacts) for policy in ["fresh","reuse"]}
    expected=first["fresh"]["report"]
    require(first["reuse"]["report"]==expected,"first reports differ")
    expected_fields={key:value for key,value in parity["fresh"]["report"].items() if key!="artifacts"}
    require({key:value for key,value in expected.items() if key!="artifacts"}==expected_fields,"artifact choice changed metrics")

    def pair(index):
        order=["fresh","reuse"] if index%2==0 else ["reuse","fresh"]
        observed={policy:call(policy,args.with_artifacts) for policy in order}
        for value in observed.values():
            require(value["report"]==expected,"timed report differs")
            require(value["artifacts"]==first["fresh"]["artifacts"],"timed PLY bytes differ")
        return {"order":order,"millis":{policy:value["millis"] for policy,value in observed.items()},
                "artifactHashes":{policy:value["artifacts"] for policy,value in observed.items()}}

    warmups=[pair(index) for index in range(5)]
    trials=[pair(index+5) for index in range(args.repeats)]
    a=[pair["millis"]["fresh"] for pair in trials]
    b=[pair["millis"]["reuse"] for pair in trials]
    summary={"freshMedianMillis":statistics.median(a),"reuseMedianMillis":statistics.median(b),
             "pairedShorteningPercent":statistics.median([100*(1-y/x) for x,y in zip(a,b)]),
             "pairedSavedMedianMillis":statistics.median([x-y for x,y in zip(a,b)]),"fasterTrials":sum(x>y for x,y in zip(a,b))}
    for path,expected_hash in hashes.items():require(sha(args.source_root/path)==expected_hash,"source changed: "+path)
    for item in [*inputs.values(),extension]:require(sha(item["path"])==item["sha256"],"input/build changed")
    report={"schemaVersion":1,"comparison":"scan-match-reference-reuse","createdAt":datetime.now(timezone.utc).isoformat(),
            "sourceHashes":hashes,"inputs":inputs,"inputManifest":manifest,"extension":extension,
            "environment":{"python":platform.python_version(),"numpy":np.__version__,"open3d":o3d.__version__,"matplotlib":matplotlib.__version__,
                           "platform":platform.system(),"logicalCpus":os.cpu_count(),"rayonNumThreads":args.threads,"ompNumThreads":args.threads},
            "threads":args.threads,"repeats":args.repeats,"warmupRounds":5,"artifactsTimed":args.with_artifacts,
            "artifactParity":parity,"firstUse":first,"warmups":warmups,"trials":trials,"summary":summary,
            "accuracy":{"fullF64ArraysChecked":4,"reportsChecked":2*(1+5+args.repeats),
                        "plyOutputsChecked":6+(6*(1+5+args.repeats) if args.with_artifacts else 0)},
            "memory":{"referenceNativePayloadBytes":parity["reuse"]["referenceCapacities"][0]},
            "note":"Complete diagnostic workflow interval includes per-invocation reference preparation and destruction, both NN searches, "
                   "cloud/normal/result copies, summaries and thresholds. Input loading and the saved actual ICP solve are excluded. "
                   "Coloring/PLY writes are timed only when artifactsTimed is true; a full artifact/f64 parity pair always runs outside summaries. "
                   "Five alternating warm-up pairs and excluded first observations precede timed pairs. No outliers removed. "
                   "Hashing/checks are outside clocks. No full CLI/registration or browser/GPU speed claim. Native payload bytes exclude RSS and scratch."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output":str(args.output),"summary":summary,"artifactsTimed":args.with_artifacts}),flush=True)


if __name__=="__main__":main()
