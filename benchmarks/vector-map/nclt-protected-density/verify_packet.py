"""Check protected point replacement, unchanged HD audits and the unadopted gap."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from ca import mapping_job as jobs, mapping_point_protection as protection
from ca.mapping_local_points import records, mask
from ca.mapping_patch import _checks

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--generated-root", type=Path)
args = parser.parse_args()


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


manifest = read(root / "files-sha256.json")
actual = {
    str(p.relative_to(root))
    for p in root.rglob("*")
    if p.is_file() and p.name != "files-sha256.json" and "__pycache__" not in p.parts
}
assert actual == manifest.keys()
for name, expected in manifest.items():
    assert sha(root / name) == expected, name
v, preview, update, before, after, proposal, output, parent = [
    read(root / (n + ".json"))
    for n in (
        "verification",
        "preview",
        "local-update",
        "before-source-quality",
        "after-source-quality",
        "proposal",
        "output",
        "parent-output",
    )
]
assert not v["adopted"] and not output["pointcloud_retry_decision"]["adopted"]
assert v["retained_checks_pass"] and read(root / "checks.json")["passes"]
reference = root.parent / "nclt-repair-continuation"
ir = read(reference / "vector_map.json")
assert (
    sha(reference / "vector_map.json") == preview["protection"]["source_hd"]["sha256"]
)
expected = protection.plan(
    {
        "files": {"editable_map": jobs._artifact(reference / "vector_map.json")},
        "quality_report": jobs._artifact(root / "before-source-quality.json"),
    }
)
for key in ("protocol", "ground_radius_m", "effective_halo_m", "hulls"):
    assert expected[key] == preview["protection"][key]
checks = _checks(before, after, {l["id"] for l in ir["lanes"]}, set())
assert checks == read(root / "checks.json")
normalized = deepcopy(before)
for name in ("editable", "reopened_osm"):
    normalized[name]["quality"]["cloud_points"] = after[name]["quality"]["cloud_points"]
    normalized["ground_consensus"][name]["quality"]["cloud_points"] = after[
        "ground_consensus"
    ][name]["quality"]["cloud_points"]
assert normalized == after
assert not any(max(c["from_m"], 8) < min(c["to_m"], 14) for c in proposal["candidates"])
assert (
    v["missing_interval"] == [8, 14]
    and not v["eligible_child_candidate_in_selected_gap"]
)
for key in (
    "map",
    "graph",
    "trajectory",
    "hd_map",
    "hd_editable_map",
    "hd_report",
    "hd_projector",
    "hd_source_audits",
):
    assert output["artifacts"][key] == parent["artifacts"][key]
assert output["continuation"]["session_spent_attempts"] == v["session_hd_attempts"] == 0
assert output["continuation"]["cumulative_spent_attempts"] == 8
assert v["source_extent_delivered_m"] == 118 and v["longest_route_delivered_m"] == 66
_, base = records(root / "inside-baseline.ply")
_, trial = records(root / "inside-protected-trial.ply")
a, b = [protection.mask(rows, preview["protection"]) for rows in (base, trial)]
assert len(base) == 4399 and len(trial) == 6193
assert base[a].tobytes() == trial[b].tobytes() and int(a.sum()) == 2460
assert int((~a).sum()) == 1939 and int((~b).sum()) == 3733
assert v["baseline_points"] == 646309 and v["trial_points"] == 648103
assert not v["independent_accuracy_established"] and not v["full_extent_goal_met"]
old = read(reference / "local-update.json")
assert (
    old["full_fusion_candidate"]["sha256"] == update["full_fusion_candidate"]["sha256"]
)
assert old["effective_bounds_xy"] == update["effective_bounds_xy"]
assert not read(reference / "local-checks.json")["passes"]
if args.generated_root:
    generated = args.generated_root.resolve()
    job = jobs._load(generated)
    for artifact in job["continuation_inputs"].values():
        jobs._verify(artifact)
    stage = job["pointcloud_retry"]
    report = read(Path(stage["local_update"]["report"]["path"]))
    protection.verify(report["protection"], job["attempts"][0])
    _, fullbase = records(Path(report["baseline_map"]["path"]))
    _, fulltrial = records(Path(report["updated_map"]["path"]))
    _, fusion = records(Path(report["full_fusion_candidate"]["path"]))

    def keep(rows):
        inside = mask(rows, report["effective_bounds_xy"])
        flags = ~inside
        flags[inside] = protection.mask(rows[inside], report["protection"])
        return flags

    a, b, c = [keep(rows) for rows in (fullbase, fulltrial, fusion)]
    assert fullbase[a].tobytes() == fulltrial[b].tobytes()
    assert fusion[~c].tobytes() == fulltrial[~b].tobytes()
    assert (
        hashlib.sha256(fullbase[a].tobytes()).hexdigest()
        == v["retained_records_sha256"]
    )
    assert (
        fullbase[mask(fullbase, v["previous_point_box"])].tobytes()
        == fulltrial[mask(fulltrial, v["previous_point_box"])].tobytes()
    )
    assert (
        stage["status"] == "ready"
        and read(generated / "run.json")["output"]["artifacts"]["map"]
        == job["pointcloud"]["files"]["map"]
    )
print(
    "Verified protected trial: all retained HD audit evidence unchanged; selected gap unresolved and prior pair retained."
    + (
        " Full retained records verified."
        if args.generated_root
        else " Full record claims require --generated-root."
    )
)
