"""Verify the adopted HD-only addition; optionally verify the full generated map pair."""

import argparse
import hashlib
import json
from pathlib import Path

from ca import mapping_job as jobs
from ca.mapping_patch import _checks, _preserved

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--generated-root", type=Path)
args = parser.parse_args()


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read(root / "files-sha256.json")
actual = {
    str(p.relative_to(root))
    for p in root.rglob("*")
    if p.is_file() and p.name != "files-sha256.json" and "__pycache__" not in p.parts
}
assert actual == manifest.keys()
for name, expected in manifest.items():
    assert digest(root / name) == expected, name
(
    before,
    after,
    audits_before,
    audits_after,
    checks,
    comparison,
    output,
    verification,
    repair,
) = [
    read(root / (name + ".json"))
    for name in (
        "before-vector_map",
        "vector_map",
        "before-source-quality",
        "after-source-quality",
        "checks",
        "comparison",
        "output",
        "verification",
        "hd-repair",
    )
]
old_ids = {lane["id"] for lane in before["lanes"]}
new_ids = {lane["id"] for lane in after["lanes"]} - old_ids
assert new_ids == {57}
_preserved(before, after, set())
assert _checks(audits_before, audits_after, old_ids, new_ids) == checks
assert checks["passes"] and not checks["holds"]
assert comparison["retry_strategy"] == "hd_only"
assert (
    comparison["pointcloud_artifacts_identical"]
    and comparison["source_proposal_identical"]
)
assert (
    comparison["gained_source_length_m"] == 2
    and comparison["lost_source_length_m"] == 0
)
assert comparison["gained_source_intervals"] == [{"from_m": 118, "to_m": 120}]
for mode in ("before", "after"):
    assert comparison[mode]["routes"]["longest_route_station_span_m"] == 66
assert comparison["before"]["extent"]["generated_length_m"] == 118
assert comparison["after"]["extent"]["generated_length_m"] == 120
assert comparison["before"]["routes"]["connected_components"] == 12
assert comparison["after"]["routes"]["connected_components"] == 13
assert (
    output["hd_repair_decision"]["adopted"]
    and not output["hd_repair_decision"]["pointcloud_regenerated"]
)
assert output["continuation"]["session_spent_attempts"] == 5
assert output["continuation"]["cumulative_spent_attempts"] == 13
assert output["artifacts"]["map"] == repair["pointcloud"]["files"]["map"]
assert repair["proposal"] == comparison["inputs"]["after_lineage_proposal"]
for key, artifact in repair["pointcloud"]["files"].items():
    assert artifact == comparison["inputs"][f"after_pointcloud_{key}"]
    assert artifact == comparison["inputs"][f"after_lineage_pointcloud_{key}"]
assert (
    not verification["full_extent_goal_met"]
    and not verification["independent_accuracy_established"]
)
for audit in [
    audits_after["editable"],
    audits_after["reopened_osm"],
    *audits_after["ground_consensus"].values(),
]:
    assert not audit["quality"]["limited"]
    lane = next(l for l in audit["quality"]["lanes"] if l["lane"] == 57)
    assert sum(lane[role]["samples"] for role in ("center", "left", "right")) == 26
    assert all(
        lane[role]["samples"] == lane[role]["supported"]
        for role in ("center", "left", "right")
    )
initial = read(root / "initial-addition-source-quality.json")
assert all(
    not a["quality"]["low_support_lanes"]
    for a in [
        initial["editable"],
        initial["reopened_osm"],
        *initial["ground_consensus"].values(),
    ]
)
occupied = read(root / "occupied-gap-observation.json")
gap = next(g for g in occupied["gaps"] if g["id"] == 21)
assert gap["retained_hd_occupancy"]["overlaps"] == [
    {"lane_id": 45, "from_m": 184, "to_m": 188}
]
assert gap["retained_hd_occupancy"]["unoccupied_intervals_total"] == 0
assert (
    read(root / "rejected-patch.json")["validation_error"]
    == "gap patch would overlap a retained lane or connector station interval"
)
assert read(root / "connection-proposal.json")["candidates"] == []

if args.generated_root:
    generated = args.generated_root.resolve()
    job, child = jobs._load(generated), jobs._load(generated / "hd-repair")
    jobs._inputs(job)
    jobs._inputs(child)
    assert job["pointcloud"] == child["pointcloud"]
    delivered = read(generated / "run.json")["output"]
    for key, artifact in job["pointcloud"]["files"].items():
        jobs._verify(artifact)
        assert delivered["artifacts"][key] == artifact
    seed, candidate = job["attempts"][0], child["attempts"][4]
    assert seed["corridor_proposal"] == candidate["corridor_proposal"]

    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        if isinstance(value, str):
            for prefix, tag in [
                (generated, "run"),
                (generated.parent / "path-refinement-final-native", "native"),
                (root.parents[2], "repo"),
                (generated.parent, "env"),
            ]:
                if value.startswith(str(prefix) + "/"):
                    return str(Path(tag) / Path(value).relative_to(prefix))
        return value

    assert portable(
        json.loads(Path(seed["corridor_proposal"]["path"]).read_text())
    ) == read(root / "source-proposal.json")
    native = jobs.core()
    for kind, file in [("editable", "editable_map"), ("reopened_osm", "map")]:
        assert (
            json.loads(
                native.audit_vector_map_quality_details(
                    job["pointcloud"]["files"]["map"]["path"],
                    candidate["files"][file]["path"],
                )
            )
            == audits_after[kind]
        )
        assert (
            json.loads(
                native.audit_vector_map_ground_consensus_details(
                    job["pointcloud"]["files"]["map"]["path"],
                    candidate["files"][file]["path"],
                )
            )
            == audits_after["ground_consensus"][kind]
        )
print(
    "HD-only repair packet verified"
    + (
        " with full retained point map and repeated native audits"
        if args.generated_root
        else " (full point files require --generated-root)"
    )
)
