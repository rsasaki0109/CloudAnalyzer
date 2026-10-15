"""Verify bounded HD height evidence; optionally audit full generated maps."""

import argparse
import hashlib
import json
from pathlib import Path

from ca.mapping_connections import edges, route_metrics
from ca.mapping_geometry import verify_lane_roundtrip
from ca.mapping_heights import _checks as height_checks, _preserved as height_preserved
from ca.mapping_local_points import inside_geometry, mask, records
from ca.mapping_patch import _checks as patch_checks, _preserved as root_preserved

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--generated-root", type=Path)
args = parser.parse_args()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def totals(audits):
    return {
        name: {
            "samples": audit["quality"]["sampled_points"],
            "supported": sum(
                lane[side]["supported"]
                for lane in audit["quality"]["lanes"]
                for side in ("center", "left", "right")
            ),
        }
        for name, audit in [
            ("legacy", audits["editable"]),
            ("consensus", audits["ground_consensus"]["editable"]),
        ]
    }


manifest = read(root / "files-sha256.json")
actual = {
    str(p.relative_to(root))
    for p in root.rglob("*")
    if p.is_file() and p.name != "files-sha256.json" and "__pycache__" not in p.parts
}
assert actual == manifest.keys(), "packet membership changed"
for name, expected in manifest.items():
    assert sha(root / name) == expected, name

for run in read(root / "verification.json")["runs"]:
    folder = root / run["label"]
    before = read(folder / "before/vector_map.json")
    addition = read(folder / "addition/vector_map.json")
    after = read(folder / "after/vector_map.json")
    old_ids = {lane["id"] for lane in before["lanes"]}
    new_ids = {lane["id"] for lane in after["lanes"]} - old_ids
    assert new_ids == set(run["addition_lane_ids"]) and len(new_ids) == 1
    root_preserved(before, after)
    assert edges(before) == edges(after)
    box = run["effective_bounds_xy"]
    old_boundaries = {b["id"] for b in before["boundaries"]}
    assert all(
        inside_geometry(b["geometry"], box)
        for b in after["boundaries"]
        if b["id"] not in old_boundaries
    )
    before_audits = read(folder / "before/source-quality.json")
    local_audits = read(folder / "local-audits.json")
    final_audits = read(folder / "after/source-quality.json")
    checks = patch_checks(before_audits, local_audits, old_ids, set())
    assert checks == read(folder / "local-checks.json") and checks["passes"]
    checks = patch_checks(before_audits, final_audits, old_ids, new_ids)
    assert checks == read(folder / "patch-checks.json") and checks["passes"]
    assert totals(before_audits) == run["baseline_hd_audits"]
    assert totals(local_audits) == run["local_attempt_hd_audits"]
    preview, update = read(folder / "local-preview.json"), read(
        folder / "local-update.json"
    )
    assert (
        preview["request"]["strategy"] == "density" and preview["eligible_frames"] == []
    )
    assert (
        preview["outside_records_sha256"]
        == update["outside_records_sha256"]
        == run["outside_records_sha256"]
    )
    assert not update["holds"] and update["effective_bounds_xy"] == box
    assert update["updated_map"] == run["final_point_map"]
    for name, count in [
        ("before", run["before_inside_points"]),
        ("candidate", run["candidate_inside_points"]),
    ]:
        _, rows = records(folder / f"inside-{name}.ply")
        assert len(rows) == count and mask(rows, box).all()
    point_report = read(folder / "pointcloud-report.json")
    ids = point_report["original_retained_frame_ids"]
    assert point_report["added_frame_ids"] == [] and run["added_frames"] == 0
    assert (
        ids == sorted(set(ids))
        and len(ids)
        == run["original_retained_frames"]
        == point_report["nodes"]
        == point_report["scans"]
    )
    assert point_report["full_fusion_map_points"] == run["full_fusion_map_points"]
    for mode, ir in [("before", before), ("after", after)]:
        report = read(folder / mode / "report.json")
        intervals = {int(k): tuple(v) for k, v in report["lane_intervals"].items()}
        assert (
            route_metrics(ir, intervals)
            == run[f"{mode}_routes"]
            == report["routes"]["after"]
        )
        assert report["extent"]["generated_length_m"] == run[f"{mode}_hd_extent_m"]
    comparison = read(folder / "comparison.json")
    assert comparison["retry_strategy"] == "local_density"
    assert comparison["gained_source_length_m"] == run["gained_source_length_m"]
    assert comparison["lost_source_length_m"] == run["lost_source_length_m"] == 0
    addition_audits = read(folder / "addition/source-quality.json")
    assert totals(addition_audits) == run["addition_before_height_audits"]
    hp = read(folder / "height-preview.json")
    assert len(hp["vertices"]) == run["height_preview_vertices"]
    modes = ["before", "addition", "after"]
    if run["label"] == "june":
        modes.append("height")
        fixed = read(folder / "height/vector_map.json")
        edits = run["height_edits"]
        assert (
            len(edits) == 1
            and edits[0]["boundary_id"] == 2
            and edits[0]["vertex_index"] == 1
            and edits[0]["delta_z_m"] == 0.075
        )
        height_preserved(addition, fixed, edits)
        assert read(folder / "height-trial.json") == fixed
        repaired_audits = read(folder / "height/source-quality.json")
        assert repaired_audits == read(folder / "height-audits.json")
        hc = height_checks(addition_audits, repaired_audits)
        assert hc == read(folder / "height-checks.json") and hc["passes"]
        assert totals(repaired_audits) == run["addition_after_height_audits"]
        report = read(folder / "height/report.json")
        assert (
            report["height_edits"] == edits
            and report["height_inputs"]["pointcloud_map"] == update["updated_map"]
        )
        assert (
            report["built_segments"]
            == read(folder / "addition/report.json")["built_segments"]
        )
        assert run["family_hd_attempts"] == run["family_budget"] == 7
    else:
        assert hp["vertices"] == [] and run["height_edits"] == []
        assert (
            run["addition_before_height_audits"] == run["addition_after_height_audits"]
        )
        assert run["family_hd_attempts"] == 6
    actions = read(folder / "agent-actions.json")
    assert (
        actions["root"]["output"]["pointcloud_retry_decision"]["adopted"]
        == run["adopted"]
        is True
    )
    assert actions["root"]["output"]["artifacts"]["map"] == update["updated_map"]
    edits_count = sum(
        h["action"]["type"] == "edit_heights" for h in actions["child"]["actions"]
    )
    assert edits_count == (1 if run["label"] == "june" else 0)
    if args.generated_root:
        import cloudanalyzer_core as native

        generated = args.generated_root / f"nclt-hd-height-{run['label']}"

        def generated_file(artifact):
            path = generated / Path(artifact["path"]).relative_to("run")
            assert (
                path.stat().st_size == artifact["bytes"]
                and sha(path) == artifact["sha256"]
            )
            return path

        _, base = records(generated_file(update["baseline_map"]))
        _, candidate = records(generated_file(update["full_fusion_candidate"]))
        local_path = generated_file(update["updated_map"])
        _, merged = records(local_path)
        a, b, c = mask(base, box), mask(candidate, box), mask(merged, box)
        assert base.dtype == candidate.dtype == merged.dtype
        assert base[~a].tobytes() == merged[~c].tobytes()
        assert candidate[b].tobytes() == merged[c].tobytes()
        assert (
            hashlib.sha256(merged[~c].tobytes()).hexdigest()
            == run["outside_records_sha256"]
        )
        assert (
            len(merged[~c]) == run["outside_points"]
            and len(merged) == run["final_map_points"]
        )
        for name, rows in [("before", base[a]), ("candidate", candidate[b])]:
            assert records(folder / f"inside-{name}.ply")[1].tobytes() == rows.tobytes()
        graph = native.PoseGraph.from_g2o(
            generated_file(run["baseline_point_files"]["graph"]).read_text()
        )
        assert list(graph.node_ids) == ids
        for key in ("graph", "trajectory"):
            relative = Path(
                point_report["outputs"]["g2o" if key == "graph" else "kitti"]
            ).relative_to("run")
            assert (
                sha(generated / relative) == run["baseline_point_files"][key]["sha256"]
            )
        for mode in modes:
            ir = read(folder / mode / "vector_map.json")
            osm = json.loads(
                json.loads(
                    native.edit_vector_map_relations(
                        str(folder / mode / "lanelet2_map.osm")
                    )
                )["map_json"]
            )
            verify_lane_roundtrip(ir, osm)
            assert edges(ir) == edges(osm)
            repeated = {
                key: json.loads(
                    native.audit_vector_map_quality_details(
                        str(local_path), str(folder / mode / filename)
                    )
                )
                for key, filename in [
                    ("editable", "vector_map.json"),
                    ("reopened_osm", "lanelet2_map.osm"),
                ]
            }
            repeated["ground_consensus"] = {
                key: json.loads(
                    native.audit_vector_map_ground_consensus_details(
                        str(local_path), str(folder / mode / filename)
                    )
                )
                for key, filename in [
                    ("editable", "vector_map.json"),
                    ("reopened_osm", "lanelet2_map.osm"),
                ]
            }
            expected = (
                local_audits
                if mode == "before"
                else read(folder / mode / "source-quality.json")
            )
            assert repeated == expected, mode
print(
    "Verified two saved trials: HD gains 4/6 m, losses 0/0 m; height edits 0/1; all four combined gates pass."
)
