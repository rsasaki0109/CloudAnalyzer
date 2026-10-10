"""Compare anchor modes on identical source intervals, after generation freeze."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

from hard_intersection_evaluate import resample
from vector_map_cross_scene_evaluate import boundary_distance
from vector_map_quality_audit import digest


def save(path: Path, value):
    path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8",newline="\n")


def reference_document(path: Path, epsg: str | None):
    if not epsg:
        import cloudanalyzer_core as core
        return json.loads(json.loads(core.edit_vector_map_relations(str(path)))["map_json"])
    root=ET.parse(path).getroot()
    transform=Transformer.from_crs("EPSG:4326",epsg,always_xy=True)
    nodes={}
    for node in root.findall("node"):
        tags={t.attrib["k"]:t.attrib["v"] for t in node.findall("tag")}
        nodes[node.attrib["id"]]=[*transform.transform(float(node.attrib["lon"]),float(node.attrib["lat"])),float(tags["ele"])]
    ways={int(w.attrib["id"]):[nodes[n.attrib["ref"]] for n in w.findall("nd")] for w in root.findall("way")}
    lanes=[]
    for relation in root.findall("relation"):
        tags={t.attrib["k"]:t.attrib["v"] for t in relation.findall("tag")}
        if tags.get("type")!="lanelet" or tags.get("subtype")!="road":
            continue
        sides={m.attrib["role"]:int(m.attrib["ref"]) for m in relation.findall("member") if m.attrib["role"] in ("left","right")}
        if len(sides)==2:
            lanes.append({"id":int(relation.attrib["id"]),"kind":"driving",**sides})
    return {"lanes":lanes,"boundaries":[{"id":k,"geometry":v} for k,v in ways.items()]}


def intervals(profiles: dict) -> dict:
    result={}
    for road in profiles["roads"]:
        reference=np.asarray(road["reference"])
        if "operator_reference" in road:
            operator=np.asarray(road["operator_reference"])
            shift=np.asarray((profiles["extraction"].get("trace_alignment") or {}).get("shift_xy", [0.,0.]))
            if operator.shape != reference.shape or not np.allclose(reference[:,:2]-operator[:,:2], shift, rtol=0, atol=1e-8):
                raise ValueError("operator reference does not invert the reported source translation")
            reference=operator
        lines=np.asarray(road["boundaries"])
        for k in range(len(reference)-1):
            key=tuple(np.round(reference[k:k+2,:2].ravel(),6))
            if key in result:
                raise ValueError("ambiguous repeated source interval")
            length=float(np.linalg.norm(reference[k+1,:2]-reference[k,:2]))
            result[key]=(length,lines[:,k,:],lines[:,k+1,:])
    return result


def paired(before: dict, after: dict, target: cKDTree) -> dict:
    a,b=intervals(before),intervals(after)
    common=set(a)&set(b)
    old,new=[],[]
    retained=0.
    for key in sorted(common):
        length,ap,aq=a[key]
        other,bp,bq=b[key]
        if ap.shape!=bp.shape or not np.isclose(length,other):
            raise ValueError("source boundary slots/interval lengths changed")
        t=np.linspace(0,1,max(1,int(np.ceil(length/.5)))+1)
        old.append((ap[:,None,:]+(aq-ap)[:,None,:]*t[None,:,None]).reshape(-1,3))
        new.append((bp[:,None,:]+(bq-bp)[:,None,:]*t[None,:,None]).reshape(-1,3))
        retained+=length
    result={"common_source_path_m":retained,"before_only_source_path_m":sum(v[0] for k,v in a.items() if k not in b),
            "after_only_source_path_m":sum(v[0] for k,v in b.items() if k not in a),"common_intervals":len(common),
            "role":"Same source intervals and configured boundary slots; every after point evaluated against the SAME nearest survey sample selected for its before point. Not semantic lane correspondence."}
    if not old:
        return {**result,"samples":0,"before":None,"after":None}
    p,q=np.concatenate(old),np.concatenate(new)
    distance,ids=target.query(p[:,:2])
    moved=np.linalg.norm(q[:,:2]-target.data[ids],axis=1)
    def stats(d):
        return {"mean_xy_m":float(d.mean()),"p90_xy_m":float(np.quantile(d,.9)),"maximum_xy_m":float(d.max()),"fraction_within_0_5m":float((d<=.5).mean())}
    return {**result,"samples":len(p),"before":stats(distance),"after":stats(moved),
            "mean_boundary_displacement_m":float(np.linalg.norm(p[:,:2]-q[:,:2],axis=1).mean()),
            "maximum_height_change_m":float(np.abs(p[:,2]-q[:,2]).max())}


def assert_profile_geometry(profiles: dict, document: dict) -> None:
    """The measured profile polylines must be present in the actual editable map."""
    built = [np.asarray(b["geometry"]) for b in document["boundaries"]]
    for road in profiles["roads"]:
        for line in road["boundaries"]:
            p = np.asarray(line)
            if not any(p.shape == q.shape and (np.allclose(p, q, rtol=0, atol=1e-8)
                       or np.allclose(p, q[::-1], rtol=0, atol=1e-8)) for q in built):
                raise ValueError("measured profile is absent from the actual generated map")


def run(source: Path, config: Path, reference: Path, out: Path, executable: Path, commit: str, epsg: str | None, *, evaluate_frozen: bool = False):
    if out.exists() and not evaluate_frozen:
        raise FileExistsError("choose a new output directory")
    source_hash=digest(source)
    config_hash=digest(config)
    executable_hash=digest(executable)
    configuration=json.loads(config.read_text(encoding="utf-8"))
    if not evaluate_frozen:
        subprocess.run([str(executable.resolve()),str(source),str(config),str(out)],check=True)
    for i in range(len(configuration["cases"])):
        for mode in ("before", "after"):
            assert_profile_geometry(
                json.loads((out / f"{mode}-{i}-profiles.json").read_text()),
                json.loads((out / f"{mode}-{i}.json").read_text()),
            )
    # Freeze ALL source outputs before opening any reference geometry.
    if evaluate_frozen:
        freeze = json.loads((out / "generation-freeze.json").read_text(encoding="utf-8"))
        if (freeze["reference_inputs"] != [] or freeze["source_sha256"] != source_hash
                or freeze["config_sha256"] != config_hash or freeze["executable_sha256"] != executable_hash):
            raise ValueError("frozen generation inputs differ")
        frozen = freeze["artifact_sha256"]
        if not frozen or any(Path(k).name != k or digest(out / k) != sha for k, sha in frozen.items()):
            raise ValueError("frozen generation artifacts differ")
        required = {f"{mode}-{i}{suffix}" for i in range(len(configuration["cases"]))
                    for mode in ("before", "after") for suffix in (".json", ".osm", "-profiles.json", "-audit.json")}
        required |= {f"trajectory-{i}.csv" for i in range(len(configuration["cases"]))} | {"comparison.json"}
        if not required <= frozen.keys():
            raise ValueError("generation freeze is incomplete")
    else:
        frozen={p.name:digest(p) for p in out.iterdir() if p.is_file()}
        save(out/"generation-freeze.json",{"reference_inputs":[],"source_sha256":source_hash,"config_sha256":config_hash,"executable_sha256":executable_hash,"artifact_sha256":frozen})
    surveyed=reference_document(reference,epsg)
    used={lane[side] if isinstance(lane[side],int) else lane[side]["boundary"] for lane in surveyed["lanes"] if lane["kind"]=="driving" for side in ("left","right")}
    target=cKDTree(np.concatenate([resample(b["geometry"],.5) for b in surveyed["boundaries"] if b["id"] in used])[:,:2])
    count=len(configuration["cases"])
    paired_cases=[]
    corridors=[]
    metrics={}
    for i,case in enumerate(configuration["cases"]):
        p=[json.loads((out/f"{mode}-{i}-profiles.json").read_text()) for mode in ("before","after")]
        paired_cases.append({"name":case["name"],**paired(*p,target)})
        if configuration.get("comparison") in {"curb_trace_alignment", "paint_corridor", "paint_divider", "lane_edge_inference"}:
            from vector_map_corridor_evaluate import corridor_comparison
            corridors.append({"name":case["name"], **corridor_comparison(*p, surveyed, case["options"], include_slots=configuration.get("comparison") in {"paint_divider", "lane_edge_inference"})})
    for mode in ("before","after"):
        audits=[json.loads((out/f"{mode}-{i}-audit.json").read_text()) for i in range(count)]
        final=json.loads((out/f"{mode}-{count-1}.json").read_text())
        metrics[mode]={"boundary_distance":boundary_distance(final,surveyed),
                       "generated_path_m":sum(r["extraction"]["generated_length"] for r in audits),
                       "deferred_path_m":sum(r["extraction"]["surface_fit"]["deferred_length_m"] for r in audits),
                       "ignored_coverage_anchor_candidates":sum(r["extraction"]["coverage_edge_anchor_candidates_ignored"] for r in audits),
                       "lane_fragments":len(final["lanes"]),"source_flags":audits[-1]["quality"]["low_support_lanes"],
                       "source_samples":audits[-1]["quality"]["sampled_points"],"limited":audits[-1]["quality"]["limited"]}
    report={"source_commit":commit,"executable_sha256":executable_hash,"evaluation_script_sha256":digest(Path(__file__)),"source_sha256":source_hash,
            "config_sha256":config_hash,"reference_sha256":digest(reference),"reference_transform":epsg or "native coordinates",
            "generation_reference_inputs":[],"profile_geometry_verified_in_built_maps":True,"generation_artifact_sha256":frozen,"metrics":metrics,"paired_source_intervals":paired_cases,
            "limitations":["Known development scenes, not held-out accuracy.","Full-map distances are unpaired nearest sampled surveyed driving-boundary XY.",
                "Common intervals are matched using source coordinates; survey targets selected for before points stay fixed for after points. Lane identities are not established.",
                "Endpoint samples repeat across intervals; sampled reference discretization affects distances.","Source support is also a generation gate; report deferred extent separately.",
                "Coverage-edge candidate geometry remains available; only its use to shift inferred lanes is disabled."]}
    if corridors:
        report["fixed_reference_corridors"] = corridors
        report["corridor_evaluation_script_sha256"] = digest(Path(__file__).with_name("vector_map_corridor_evaluate.py"))
        report["limitations"] += ["Trace translation can change which nearby survey boundary is closest. Retained before-nearest targets and ordered adjacent-lane correspondences are different diagnostics; report both.", "Corridor assignment is selected jointly using BEFORE only; incomplete reference intervals and unsupported configurations are held and their extent reported."]
    assert digest(source)==source_hash and digest(config)==config_hash and digest(executable)==executable_hash
    assert all(digest(out/k)==v for k,v in frozen.items())
    save(out/"evaluation.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","config","reference","out","executable"):
        parser.add_argument(name,type=Path)
    parser.add_argument("--source-commit",required=True)
    parser.add_argument("--reference-epsg")
    parser.add_argument("--evaluate-frozen", action="store_true", help="Verify and evaluate already frozen generation; never regenerate")
    args=parser.parse_args()
    report=run(args.source,args.config,args.reference,args.out,args.executable,args.source_commit,args.reference_epsg,evaluate_frozen=args.evaluate_frozen)
    print(json.dumps({"metrics":report["metrics"],"paired":report["paired_source_intervals"]},indent=2))
