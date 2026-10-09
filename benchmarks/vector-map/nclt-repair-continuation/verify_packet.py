"""Verify a rejected second repair and retention of the previously delivered pair."""

import argparse
import hashlib
import json
from pathlib import Path

from ca import mapping_job as jobs
from ca.mapping_connections import edges
from ca.mapping_geometry import verify_lane_roundtrip
from ca.mapping_local_points import records, mask
from ca.mapping_patch import _checks

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--generated-root", type=Path)
args = parser.parse_args()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read(root / "files-sha256.json")
actual = {
    str(p.relative_to(root))
    for p in root.rglob("*")
    if p.is_file() and p.name != "files-sha256.json" and "__pycache__" not in p.parts
}
assert actual == manifest.keys()
for name, expected in manifest.items():
    assert sha(root / name) == expected, name
v, output, parent, seed = [
    read(root / (n + ".json"))
    for n in ("verification", "output", "parent-output", "seed")
]
assert not v["adopted"] and output["candidate_id"] == 1
assert not output["pointcloud_retry_decision"]["adopted"]
for key in (
    "map",
    "graph",
    "trajectory",
    "hd_map",
    "hd_editable_map",
    "hd_projector",
    "hd_report",
    "hd_source_audits",
):
    assert output["artifacts"][key] == parent["artifacts"][key]
assert output["continuation"]["session_spent_attempts"] == 0
assert (
    output["continuation"]["cumulative_spent_attempts"]
    == v["cumulative_hd_attempts"]
    == 8
)
assert v["new_session_budget"] == v["allocated_to_rejected_child"] == 6
ir = read(root / "vector_map.json")
assert {51, 54} <= {l["id"] for l in ir["lanes"]}
assert {(51, 54), (54, 9)} <= edges(ir)
retained, rejected = [
    read(root / (n + "-source-quality.json")) for n in ("retained", "rejected")
]
checks = _checks(retained, rejected, {l["id"] for l in ir["lanes"]}, set())
assert checks == read(root / "local-checks.json") and not checks["passes"]
assert checks["holds"] == v["holds"]
assert any("6:left" in h for h in checks["holds"]) and any(
    "39:center" in h for h in checks["holds"]
)
for audits in (retained, rejected):
    for audit in (
        audits["editable"],
        audits["reopened_osm"],
        *audits["ground_consensus"].values(),
    ):
        assert jobs._diagnose_audit(audit)["complete"]
assert (
    v["source_extent_before_m"]
    == v["source_extent_after_m"]
    == seed["extent"]["generated_length_m"]
    == 118
)
assert v["gained_source_length_m"] == v["lost_source_length_m"] == 0
assert v["retained_route_components"] == 12 and v["retained_longest_route_m"] == 66
for mode, count in (
    ("retained", v["before_inside_points"]),
    ("rejected", v["rejected_inside_points"]),
):
    _, rows = records(root / f"inside-{mode}.ply")
    assert len(rows) == count and mask(rows, v["point_box"]).all()
assert (
    not v["independent_accuracy_improvement_established"]
    and not v["full_extent_goal_met"]
)
# Validate the exact exported map, not just its saved source-audit summaries.
core = jobs.core()
assert core is not None
reopened = json.loads(
    json.loads(core.edit_vector_map_relations(str(root / "lanelet2_map.osm")))[
        "map_json"
    ]
)
verify_lane_roundtrip(ir, reopened)
assert edges(ir) == edges(reopened)
if args.generated_root:
    generated = args.generated_root.resolve()
    job = jobs._load(generated)
    for artifact in job["continuation_inputs"].values():
        jobs._verify(artifact)
    run = read(generated / "run.json")
    old = read(Path(job["continuation"]["parent_run"]) / "run.json")
    assert run["status"] == "finished"
    for key in ("map", "graph", "trajectory", "hd_map", "hd_editable_map"):
        assert run["output"]["artifacts"][key] == old["output"]["artifacts"][key]
        jobs._verify(run["output"]["artifacts"][key])
    update = read(Path(job["pointcloud_retry"]["local_update"]["report"]["path"]))
    _, before = records(Path(update["baseline_map"]["path"]))
    _, trial = records(Path(update["updated_map"]["path"]))
    _, fusion = records(Path(update["full_fusion_candidate"]["path"]))
    a, b, c = [mask(rows, v["point_box"]) for rows in (before, trial, fusion)]
    assert before[~a].tobytes() == trial[~b].tobytes()
    assert fusion[c].tobytes() == trial[b].tobytes()
    assert (
        hashlib.sha256(before[~a].tobytes()).hexdigest() == v["outside_records_sha256"]
    )
    assert (
        before[mask(before, v["previous_point_box"])].tobytes()
        == trial[mask(trial, v["previous_point_box"])].tobytes()
    )
    assert (
        len(before) == v["retained_map_points"]
        and len(trial) == v["rejected_map_points"]
    )
print(
    "Verified rejected second density trial, retained previous map/lanes/connector and cumulative budget."
    + (
        " Full outside records and immutable lineage verified."
        if args.generated_root
        else " Outside-byte claims require --generated-root."
    )
)
