"""Verify explicit MCP application, source decisions and retained HD maps."""

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
parser.add_argument("--april-generated-root", type=Path)
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
policy = read(root / "application-policy.json")
assert policy["interval_ids"] == [3, 11] and policy["connect_endpoints"] is True
assert [(r["from_m"], r["to_m"]) for r in policy["selected_intervals"]] == [
    (222, 224),
    (224, 226),
]
assert all(
    r["reference_traces_fully_supported"] and not r["source_ambiguous"]
    for r in policy["selected_intervals"]
)
repeat = read(root / "repeat-verification.json")
assert repeat["ready_state_hashes"] == repeat["repeated_state_hashes"]
originals = read(root / "original-state-verification.json")
assert (
    originals["before"] == originals["after"] and originals["original_states_unchanged"]
)
root_history, child_history = read(root / "root-history.json"), read(
    root / "child-history.json"
)
assert [h["action"]["type"] for h in root_history[-3:]] == [
    "repair_hd",
    "compare_retry",
    "finish_retry",
]
assert [h["action"]["type"] for h in child_history] == [
    "inspect",
    "draft",
    "inspect_patch",
    "patch_gaps",
    "finish",
]
assert root_history[-1]["reason"].startswith("Explicitly adopt")
assert "hd_application_policy" in output["artifacts"]
april = root / "april"
va = read(april / "verification.json")
ba, aa = read(april / "before-vector_map.json"), read(april / "vector_map.json")
qa, qb = read(april / "before-source-quality.json"), read(
    april / "after-source-quality.json"
)
old_a = {lane["id"] for lane in ba["lanes"]}
new_a = {lane["id"] for lane in aa["lanes"]} - old_a
assert sorted(new_a) == va["new_lane_ids"] and len(new_a) == 7
_preserved(ba, aa, {tuple(pair) for pair in va["added_edges"]})
assert edges(aa) == edges(ba) | {tuple(pair) for pair in va["added_edges"]}
assert _checks(qa, qb, old_a, new_a) == read(april / "checks.json")
assert read(april / "checks.json")["passes"]
ca = read(april / "comparison.json")
assert ca["gained_source_length_m"] == 14 and ca["lost_source_length_m"] == 0
assert ca["before"]["extent"]["generated_length_m"] == 110
assert ca["after"]["extent"]["generated_length_m"] == 124
assert ca["pointcloud_artifacts_identical"] and ca["source_proposal_identical"]
rejected = read(april / "rejection-verification.json")
assert rejected["before"] == rejected["after"] and rejected["hd_attempts_spent"] == 0
assert not rejected["child_created"] and not rejected["policy_created"]
pa = read(april / "application-policy.json")
assert pa["interval_ids"] == list(range(1, 8)) and pa["connect_endpoints"]
assert all(
    r["reference_traces_fully_supported"] and not r["source_ambiguous"]
    for r in pa["selected_intervals"]
)
assert [(r["from_m"], r["to_m"]) for r in pa["selected_intervals"]] == [
    (n, n + 2) for n in range(184, 198, 2)
]
assert va["actual_new_attempts"] == va["attempt_budget"] == 3
out_a = read(april / "output.json")
assert (
    out_a["hd_repair_decision"]["adopted"]
    and "hd_application_policy" in out_a["artifacts"]
)
proposal_a = read(april / "source-proposal.json")
report_a = read(april / "before-report.json")
gaps_a = [
    {"id": i + 1, **r}
    for i, r in enumerate(
        [
            r
            for r in report_a["station_disposition"]
            if r["status"] != "included_lane_hypothesis"
        ]
    )
]
for phase in ["april-plan0", "april-plan1", "april-plan1-next"]:
    page = read(april / (phase + ".json"))
    index = read(april / (phase + "-index.json"))
    parent = {
        "files": {
            "editable_map": jobs._artifact(april / "before-vector_map.json"),
            "report": jobs._artifact(april / "before-report.json"),
        }
    }
    assert (
        _index(
            parent,
            proposal_a,
            [g for g in gaps_a if g["id"] in index["gap_ids"]],
            read(april / "layout-hypothesis.json"),
        )
        == index["intervals"]
    )
    for row in page["intervals"]:
        saved = read(
            april / (phase + "-source-" + str(row["plan_interval_id"]) + ".json")
        )
        assert saved["interval"] == next(
            r
            for r in index["intervals"]
            if r["plan_interval_id"] == row["plan_interval_id"]
        )
        supported = all(
            jobs._diagnose_audit(a)["complete"]
            and all(
                l[side]["samples"] > 0
                and l[side]["supported"] == l[side]["samples"]
                and l[side]["start_supported"]
                and l[side]["end_supported"]
                for l in a["quality"]["lanes"]
                for side in ("left", "right", "center")
            )
            for a in saved["audits"].values()
        )
        assert supported == row["reference_traces_fully_supported"]
if args.april_generated_root:
    generated = args.april_generated_root.resolve()
    job = jobs._load(generated)
    child = jobs._load(generated / "hd-repair")
    jobs._inputs(job)
    jobs._inputs(child)
    assert job["pointcloud"] == child["pointcloud"] and len(child["attempts"]) == 3
    candidate = child["attempts"][2]
    native = jobs.core()
    for kind, file in [("editable", "editable_map"), ("reopened_osm", "map")]:
        assert (
            json.loads(
                native.audit_vector_map_quality_details(
                    job["pointcloud"]["files"]["map"]["path"],
                    candidate["files"][file]["path"],
                )
            )
            == qb[kind]
        )
        assert (
            json.loads(
                native.audit_vector_map_ground_consensus_details(
                    job["pointcloud"]["files"]["map"]["path"],
                    candidate["files"][file]["path"],
                )
            )
            == qb["ground_consensus"][kind]
        )
print(
    "HD plan application packet verified"
    + (
        " with full point files and repeated native audits"
        if args.generated_root
        else " (full point files require --generated-root)"
    )
)
