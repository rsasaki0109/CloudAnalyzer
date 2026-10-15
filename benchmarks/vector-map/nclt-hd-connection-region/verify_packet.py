"""Verify separate point/HD envelopes; optionally check full generated maps."""

import argparse
import hashlib
import json
from pathlib import Path

from ca.mapping_connections import edges, route_metrics
from ca.mapping_geometry import verify_lane_roundtrip
from ca.mapping_local_points import inside_geometry, mask, records
from ca.mapping_patch import _checks, _preserved

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
assert actual == manifest.keys(), "packet membership changed"
for name, expected in manifest.items():
    assert sha(root / name) == expected, name
v = read(root / "verification.json")
before = read(root / "patched/vector_map.json")
after = read(root / "connected/vector_map.json")
old_ids = {lane["id"] for lane in before["lanes"]}
new_ids = {lane["id"] for lane in after["lanes"]} - old_ids
assert new_ids == {v["connector_lane_id"]} == {54}
extra = edges(after) - edges(before)
assert extra == {tuple(pair) for pair in v["added_edges"]} == {(51, 54), (54, 9)}
_preserved(before, after, extra)
old_boundaries = {row["id"] for row in before["boundaries"]}
new_boundaries = [row for row in after["boundaries"] if row["id"] not in old_boundaries]
assert all(inside_geometry(row["geometry"], v["hd_box"]) for row in new_boundaries)
assert any(
    not inside_geometry(row["geometry"], v["point_box"]) for row in new_boundaries
)
before_audits = read(root / "patched/source-quality.json")
after_audits = read(root / "connected/source-quality.json")
checks = _checks(before_audits, after_audits, old_ids, new_ids)
assert checks == read(root / "connection-checks.json") and checks["passes"]
for name, audit in [
    ("legacy", after_audits["editable"]),
    ("consensus", after_audits["ground_consensus"]["editable"]),
]:
    lane = next(row for row in audit["quality"]["lanes"] if row["lane"] == 54)
    assert lane == v["connector_audits"][name]
    assert sum(lane[side]["samples"] for side in ("center", "left", "right")) == 40
    assert all(
        lane[side]["supported"] == lane[side]["samples"]
        and lane[side]["start_supported"]
        and lane[side]["end_supported"]
        for side in ("center", "left", "right")
    )
proposal = read(root / "region-preview.json")
region = proposal["region"]
assert region["bounds_xy"] == v["hd_box"] == [9, -7, 22, 5]
assert region["point_bounds_xy"] == v["point_box"] == [12.0, -3.0, 22.0, 5.0]
assert region["repair_lanes_only"] and region["point_map_unchanged"]
assert {(row["from"], row["to"]) for row in proposal["candidates"]} == {(51, 9)}
assert all(
    inside_geometry(row[side], v["hd_box"])
    for row in proposal["candidates"]
    for side in ("center", "left", "right")
)
normal = read(root / "default-preview.json")
assert normal["candidates"] == []
assert any(
    row["from"] == 51
    and row["to"] == 9
    and row["holds"] == ["outside_local_point_update_bounds"]
    for row in normal["rejected"]
)
for mode, ir, metrics in [
    ("patched", before, v["before_routes"]),
    ("connected", after, v["after_routes"]),
]:
    report = read(root / mode / "report.json")
    intervals = {int(k): tuple(value) for k, value in report["lane_intervals"].items()}
    assert route_metrics(ir, intervals) == report["routes"]["after"] == metrics
    assert metrics["longest_route_station_span_m"] == 66
    assert report["extent"]["generated_length_m"] == 118
assert (
    v["before_routes"]["connected_components"] == 13
    and v["after_routes"]["connected_components"] == 12
)
assert v["local_chain"] in v["after_routes"]["routes"]
assert v["local_chain"] == {
    "lane_ids": [51, 54, 9],
    "from_m": 22,
    "to_m": 36,
    "station_span_m": 14,
}
assert (
    v["source_extent_before_connection_m"]
    == v["source_extent_after_connection_m"]
    == 118
)
for name in ("patched", "connected"):
    comparison = read(root / f"comparison-{name}.json")
    assert comparison["gained_source_length_m"] == v["source_intervals_gained_m"] == 6
    assert comparison["lost_source_length_m"] == v["source_intervals_lost_m"] == 0
assert (
    v["point_map_before_connection"]
    == v["point_map_after_connection"]
    == v["pointcloud_files"]["map"]
)
update = read(root / "local-update.json")
assert (
    update["effective_bounds_xy"] == v["point_box"]
    and update["updated_map"] == v["point_map_after_connection"]
)
assert update["outside_records_sha256"] == v["outside_records_sha256"]
assert (
    read(root / "local-checks.json")["passes"]
    and read(root / "height-checks.json")["passes"]
)
for name, count in [
    ("before", v["before_inside_points"]),
    ("candidate", v["candidate_inside_points"]),
]:
    _, rows = records(root / f"inside-{name}.ply")
    assert len(rows) == count and mask(rows, v["point_box"]).all()
point_report = read(root / "pointcloud-report.json")
ids = point_report["original_retained_frame_ids"]
assert (
    point_report["added_frame_ids"] == []
    and len(ids) == len(set(ids)) == v["original_retained_frames"] == 195
)
assert point_report["full_fusion_map_points"] == v["full_fusion_map_points"]
assert v["family_hd_attempts"] == v["family_budget"] == 8
for key in ("graph", "trajectory"):
    assert (
        v["baseline_point_files"][key]["sha256"] == v["pointcloud_files"][key]["sha256"]
    )
actions = read(root / "agent-actions.json")
assert (
    actions["root"]["output"]["pointcloud_retry_decision"]["adopted"]
    == v["adopted"]
    is True
)
assert actions["root"]["output"]["artifacts"]["map"] == v["point_map_after_connection"]
assert (
    actions["root"]["output"]["artifacts"]["hd_connection_proposal"]
    == read(root / "connected/report.json")["connection_proposal"]
)
assert any(
    row["action"]["type"] == "inspect_connection_region"
    and row["action"]["bounds_xy"] == v["hd_box"]
    for row in actions["child"]["actions"]
)

if args.generated_root:
    import cloudanalyzer_core as native

    generated = args.generated_root / "nclt-hd-region-june"

    def generated_file(artifact):
        path = generated / Path(artifact["path"]).relative_to("run")
        assert (
            path.stat().st_size == artifact["bytes"] and sha(path) == artifact["sha256"]
        )
        return path

    _, base = records(generated_file(update["baseline_map"]))
    _, candidate = records(generated_file(update["full_fusion_candidate"]))
    local_path = generated_file(v["point_map_after_connection"])
    _, local = records(local_path)
    a, b, c = [mask(rows, v["point_box"]) for rows in (base, candidate, local)]
    assert base.dtype == candidate.dtype == local.dtype
    assert (
        base[~a].tobytes() == local[~c].tobytes()
        and candidate[b].tobytes() == local[c].tobytes()
    )
    assert (
        hashlib.sha256(local[~c].tobytes()).hexdigest() == v["outside_records_sha256"]
    )
    assert (
        len(local[~c]) == v["outside_points"]
        and len(local) == v["delivered_map_points"]
    )
    graph = native.PoseGraph.from_g2o(
        generated_file(v["baseline_point_files"]["graph"]).read_text()
    )
    assert list(graph.node_ids) == ids
    for artifact in v["pointcloud_files"].values():
        generated_file(artifact)
    for mode, ir, expected in [
        ("patched", before, before_audits),
        ("connected", after, after_audits),
    ]:
        osm = json.loads(
            json.loads(
                native.edit_vector_map_relations(str(root / mode / "lanelet2_map.osm"))
            )["map_json"]
        )
        verify_lane_roundtrip(ir, osm)
        assert edges(ir) == edges(osm)
        repeated = {
            key: json.loads(
                native.audit_vector_map_quality_details(
                    str(local_path), str(root / mode / filename)
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
                    str(local_path), str(root / mode / filename)
                )
            )
            for key, filename in [
                ("editable", "vector_map.json"),
                ("reopened_osm", "lanelet2_map.osm"),
            ]
        }
        assert repeated == expected, mode
print(
    "Verified HD-only connector 51 → 54 → 9: 14 m local chain; point file fixed; all four gates pass; source extent stays 118 m."
)
