"""Rebuild source planning decisions and verify the adopted partial HD map."""

import argparse
import hashlib
import json
from pathlib import Path

from ca import mapping_job as jobs
from ca.mapping_hd_plan import _index
from ca.mapping_patch import _checks, _preserved
from ca.mapping_connections import edges

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--generated-root", type=Path)
args = parser.parse_args()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = read(root / "files-sha256.json")
assert {
    str(p.relative_to(root))
    for p in root.rglob("*")
    if p.is_file() and p.name != "files-sha256.json" and "__pycache__" not in p.parts
} == manifest.keys()
for name, expected in manifest.items():
    assert sha(root / name) == expected, name
v = read(root / "verification.json")
before = read(root / "before-vector_map.json")
after = read(root / "vector_map.json")
a = read(root / "before-source-quality.json")
b = read(root / "after-source-quality.json")
check = read(root / "checks.json")
comparison = read(root / "comparison.json")
output = read(root / "output.json")
old = {l["id"] for l in before["lanes"]}
new = {l["id"] for l in after["lanes"]} - old
assert new == {62, 63}
_preserved(before, after, {(36, 62), (62, 63)})
assert edges(after) == edges(before) | {(36, 62), (62, 63)}
assert _checks(a, b, old, new) == check and check["passes"] and not check["holds"]
assert (
    comparison["gained_source_length_m"] == 4
    and comparison["lost_source_length_m"] == 0
)
assert (
    comparison["before"]["extent"]["generated_length_m"] == 120
    and comparison["after"]["extent"]["generated_length_m"] == 124
)
assert (
    comparison["pointcloud_artifacts_identical"]
    and comparison["source_proposal_identical"]
)
assert (
    output["hd_repair_decision"]["adopted"]
    and not output["hd_repair_decision"]["pointcloud_regenerated"]
)
assert (
    output["continuation"]["session_spent_attempts"] == 3
    and output["continuation"]["cumulative_spent_attempts"] == 16
)
for key, artifact in v["pointcloud"].items():
    assert output["artifacts"][key] == artifact
source_file = (root / v["source_proposal_packet"]).resolve()
assert sha(source_file) == v["source_proposal_packet_sha256"]
proposal = read(source_file)
report = read(root / "before-report.json")
layout = read(root / "layout-hypothesis.json")
parent = {
    "files": {
        "editable_map": jobs._artifact(root / "before-vector_map.json"),
        "report": jobs._artifact(root / "before-report.json"),
    }
}
gaps = [
    {"id": i + 1, **r}
    for i, r in enumerate(
        [
            r
            for r in report["station_disposition"]
            if r["status"] != "included_lane_hypothesis"
        ]
    )
]
assert len(gaps) == 31
all_rows = []
for batch in ("0", "0-next", "1"):
    page = read(root / ("plan-" + batch + ".json"))
    index = read(root / ("index-" + batch + ".json"))
    rebuilt = _index(
        parent, proposal, [g for g in gaps if g["id"] in index["gap_ids"]], layout
    )
    assert rebuilt == index["intervals"]
    assert (
        page["protocol"]["hd_generation_attempts_spent"] == 0
        and not page["protocol"]["lane_export_performed"]
    )
    for row in page["intervals"]:
        saved = read(root / f"source-{batch}-{row['plan_interval_id']}.json")
        expected = next(
            r
            for r in index["intervals"]
            if r["plan_interval_id"] == row["plan_interval_id"]
        )
        assert saved["interval"] == expected
        audits = saved["audits"]
        complete = all(
            jobs._diagnose_audit(audit)["complete"] for audit in audits.values()
        )
        supported = complete and all(
            len(audit["quality"]["lanes"]) == 1
            and all(
                l[side]["samples"] > 0
                and l[side]["supported"] == l[side]["samples"]
                and l[side]["start_supported"]
                and l[side]["end_supported"]
                for l in audit["quality"]["lanes"]
                for side in ("center", "left", "right")
            )
            for audit in audits.values()
        )
        assert supported == row["reference_traces_fully_supported"]
        if supported:
            lid = 62 if row["from_m"] == 222 else 63
            for estimator, actual in [
                ("quantile", b["editable"]),
                ("consensus", b["ground_consensus"]["editable"]),
            ]:
                generated = next(
                    l for l in actual["quality"]["lanes"] if l["lane"] == lid
                )
                assert all(
                    audits[estimator]["quality"]["lanes"][0][role] == generated[role]
                    for role in ("center", "left", "right")
                )
        all_rows.append(row)
assert (
    len(all_rows) == 11
    and sum(r["reference_traces_fully_supported"] for r in all_rows) == 2
)
assert [
    (r["from_m"], r["to_m"]) for r in all_rows if r["reference_traces_fully_supported"]
] == [(222, 224), (224, 226)]
for audit in [b["editable"], b["reopened_osm"], *b["ground_consensus"].values()]:
    assert not audit["quality"]["limited"]
    additions = [l for l in audit["quality"]["lanes"] if l["lane"] in new]
    assert (
        sum(
            l[side]["samples"]
            for l in additions
            for side in ("center", "left", "right")
        )
        == 45
    )
    assert all(
        l[side]["supported"] == l[side]["samples"]
        for l in additions
        for side in ("center", "left", "right")
    )
routes = read(root / "routes.json")
for mode, span in [("before", 2), ("after", 6)]:
    assert (
        routes[mode]["longest_route_station_span_m"] == 66
        and routes[mode]["connected_components"] == 13
    )
    assert (
        next(r for r in routes[mode]["routes"] if 36 in r["lane_ids"])["station_span_m"]
        == span
    )
assert not v["full_extent_goal_met"] and not v["independent_accuracy_established"]
if args.generated_root:
    generated = args.generated_root.resolve()
    job = jobs._load(generated)
    child = jobs._load(generated / "hd-repair")
    jobs._inputs(job)
    jobs._inputs(child)
    assert job["pointcloud"] == child["pointcloud"]
    for artifact in job["pointcloud"]["files"].values():
        jobs._verify(artifact)
    native = jobs.core()
    candidate = child["attempts"][2]
    for kind, file in [("editable", "editable_map"), ("reopened_osm", "map")]:
        assert (
            json.loads(
                native.audit_vector_map_quality_details(
                    job["pointcloud"]["files"]["map"]["path"],
                    candidate["files"][file]["path"],
                )
            )
            == b[kind]
        )
        assert (
            json.loads(
                native.audit_vector_map_ground_consensus_details(
                    job["pointcloud"]["files"]["map"]["path"],
                    candidate["files"][file]["path"],
                )
            )
            == b["ground_consensus"][kind]
        )
print(
    "HD repair preflight packet verified"
    + (
        " with full point files and repeated native audits"
        if args.generated_root
        else " (full point files require --generated-root)"
    )
)
