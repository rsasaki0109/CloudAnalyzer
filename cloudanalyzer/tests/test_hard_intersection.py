"""Bounded, label-independent preparation and honest one-scene reference metrics."""
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import laspy
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import hard_intersection_fetch as fetch
import hard_intersection_prepare as prep
import hard_intersection_generate as gen
import hard_intersection_evaluate as audit
import vector_map_quality_audit as quality
import vector_map_surface_evaluate as surface


def test_surface_evaluation_reports_all_deferred_and_counts_only_added_lane_samples(tmp_path, monkeypatch):
    source = tmp_path / "cloud"
    trajectory = tmp_path / "trajectory"
    source.write_bytes(b"fixed source"); trajectory.write_bytes(b"fixed path")
    ids = []
    def build(cloud, path, options, existing_map=None):
        if json.loads(options)["fit_source_surface"]:
            raise ValueError("No source-supported road stretches")
        ids.append(len(ids) + 1)
        return json.dumps({"map_json": json.dumps({"lanes": [{"id": i} for i in ids]}), "osm": "<osm/>", "projector_info": "projector_type: Local\n", "report": {"extraction": {"generated_length": 10, "surface_fit": None}}})
    def audit(cloud, path):
        sides = {s: {"samples": 1} for s in ("center", "left", "right")}
        return json.dumps({"quality": {"lanes": [{"lane": i, **sides} for i in ids], "low_support_lanes": ids, "sampled_points": 3 * len(ids), "omitted_lanes": [], "malformed_lanes": [], "limited": False}, "validation": {"counts": {"errors": 0}}})
    monkeypatch.setitem(sys.modules, "cloudanalyzer_core", SimpleNamespace(_core=SimpleNamespace(__file__=str(source)), build_vector_map=build, audit_vector_map_quality=audit))
    result = surface.evaluate(source, [{"name": name, "trajectory": trajectory, "options": {}} for name in ("a", "b")], tmp_path / "out", "test")
    assert result["after"]["final_map"] is None
    assert result["after"]["source_quality"] is None
    assert result["after"]["evaluated_path_length_m"] is None
    assert result["after"]["entire_path_deferred_cases"] == ["a", "b"]
    assert [c["before"]["source_quality"]["sampled_points"] for c in result["cases"]] == [3, 3]
    assert result["before"]["source_quality"]["sampled_points"] == 6


def test_quality_audit_uses_generated_local_coordinates_and_closed_paint(tmp_path):
    path = tmp_path / "map.osm"
    nodes = "".join(f'<node id="{i}" lat="0" lon="0"><tag k="local_x" v="{x}"/><tag k="local_y" v="{y}"/><tag k="ele" v="2"/></node>' for i, (x,y) in enumerate([(100,200),(100,204),(106,200),(106,204)],1))
    path.write_text('<osm>'+nodes+'<way id="10"><nd ref="1"/><nd ref="2"/></way><way id="11"><nd ref="3"/><nd ref="4"/></way><relation id="12"><member type="way" ref="10" role="left"/><member type="way" ref="11" role="right"/><tag k="type" v="lanelet"/><tag k="subtype" v="crosswalk"/></relation></osm>', encoding="utf-8")
    objects = quality.generated_equipment(path)
    ring = objects["repeated_paint"][0]["geometry"]
    assert ring[0] == ring[-1] == [100,200,2]
    assert len(ring) == 5
    comparison = quality.equipment_comparison(objects, objects)
    assert comparison["repeated_paint"]["nearby_correspondences"] == 1
    assert comparison["repeated_paint"]["matches"][0]["geometry"]["hausdorff_m"] == 0


def test_quality_frozen_map_guard_rejects_changed_map_and_reference_inputs(tmp_path):
    osm = tmp_path / "reviewed.osm"
    osm.write_bytes(b"<osm/>")
    manifest = {"reference_inputs": [], "generated_sha256": hashlib.sha256(osm.read_bytes()).hexdigest()}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert quality.verify_frozen_map(tmp_path)[0] == osm
    osm.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after freezing"):
        quality.verify_frozen_map(tmp_path)
    osm.write_bytes(b"<osm/>")
    manifest["reference_inputs"] = ["reference.osm"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="used reference"):
        quality.verify_frozen_map(tmp_path)


def test_closed_outline_sampling_includes_last_edge_without_mutating_prediction():
    candidate = {"evidence": {"kind": "repeated_paint", "measurement": {"outline": [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]}}}
    original = json.dumps(candidate)
    closed = audit.geometry(candidate, True)
    np.testing.assert_array_equal(closed[0], closed[-1])
    target = np.array([0, 1, 0])
    assert np.min(np.linalg.norm(audit.resample(closed) - target, axis=1)) < 1e-9
    assert np.min(np.linalg.norm(audit.resample(audit.geometry(candidate)) - target, axis=1)) >= 1.
    assert json.dumps(candidate) == original
    candidate["evidence"]["measurement"]["outline"].append([0, 0, 0])
    assert len(audit.geometry(candidate, True)) == 5
    line = {"evidence": {"kind": "bright_bar", "geometry": [[0, 0, 0], [2, 0, 0]]}}
    assert len(audit.geometry(line, True)) == 2


def test_voxel_selection_is_chunk_invariant_original_and_bounded():
    xyz = np.array([[0, 0, 0], [.05, 0, 0], [.2, .1, .1], [.25, .1, .1], [.1, .2, .2], [0, 0, 0]])
    whole = prep.FirstVoxel([0, 0, 0], [1, 1, 1]).retain(xyz)
    seen = prep.FirstVoxel([0, 0, 0], [1, 1, 1])
    chunks = np.concatenate([seen.retain(xyz[:3]), seen.retain(xyz[3:]) + 3])
    np.testing.assert_array_equal(whole, chunks)
    np.testing.assert_array_equal(whole, [0, 2, 4])
    with pytest.raises(ValueError, match="budget"):
        prep.FirstVoxel([0, 0, 0], [1000, 1000, 1000], max_bytes=1)
    with pytest.raises(ValueError, match="nonfinite"):
        seen.retain(np.array([[np.nan, 0, 0]]))
    with pytest.raises(ValueError, match="outside"):
        seen.retain(np.array([[-1, 0, 0]]))


def trajectory(path, rows):
    path.write_text("Time[s],Easting[m],Northing[m],Height[m],Roll[deg],Pitch[deg],Yaw[deg]\n" + "".join(f"{t},{x},{y},1,0,0,0\n" for t, x, y in rows), encoding="utf-8")


def test_trajectory_splits_outside_extent_and_preserves_recorded_time(tmp_path):
    path = tmp_path / "drive.txt"
    trajectory(path, [(0, 1, 1), (1, 1.1, 1), (2, 2, 1), (3, 10, 1), (4, 3, 1), (5, 4, 1)])
    assert prep.trajectory_segments(path, [0, 0], [5, 5]) == [[[0, 1, 1, 1], [2, 2, 1, 1]], [[4, 3, 1, 1], [5, 4, 1, 1]]]
    trajectory(path, [(1, 1, 1), (1, 2, 1)])
    with pytest.raises(ValueError, match="nonincreasing"):
        prep.trajectory_segments(path, [0, 0], [5, 5])


def test_prepare_retains_exact_attributes_clears_semantics_and_ignores_reference(tmp_path, monkeypatch):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [.001] * 3
    cloud = laspy.LasData(header)
    cloud.x = [0, .04, .2, .25, .4]
    cloud.y = [0, 0, .2, .2, .4]
    cloud.z = [1] * 5
    cloud.red = [10, 20, 30, 40, 50]
    cloud.intensity = [100, 200, 300, 400, 500]
    cloud.user_data = [41, 52, 21, 42, 11]
    cloud.classification = [1, 2, 3, 4, 5]
    source = tmp_path / "raw.las"
    cloud.write(source)
    before = source.read_bytes()
    drives = tmp_path / "drives"
    drives.mkdir()
    trajectory(drives / "drive.txt", [(0, 0, 0), (1, .4, .4)])
    # A deliberately invalid reference must never be read during preparation.
    (tmp_path / "reference.osm").write_text("not XML")
    monkeypatch.setattr(prep, "CHUNK", 2)
    result = prep.prepare(source, drives, tmp_path / "out")
    retained = laspy.read(tmp_path / "out/geometry.las")
    np.testing.assert_array_equal(retained.X, cloud.X[[0, 2, 4]])
    np.testing.assert_array_equal(retained.red, [10, 30, 50])
    np.testing.assert_array_equal(retained.intensity, [100, 300, 500])
    assert not np.any(retained.user_data) and not np.any(retained.classification)
    assert result["reference_inputs"] == [] and result["retained_points"] == 3
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        prep.prepare(source, drives, tmp_path / "out")


def test_tile_ownership_half_open_has_no_boundary_duplicates():
    candidate = {"min": [20, 1, 0], "max": [20, 1, 2]}
    assert not gen.owns({"core_min": [0, 0], "core_max": [20, 20]}, candidate)
    assert gen.owns({"core_min": [20, 0], "core_max": [40, 20]}, candidate)


def test_assignment_maximizes_valid_pairs_before_minimizing_distance():
    # Choosing the shortest first leaves one unmatched; valid two-pair solution wins.
    pairs = audit.gated_assignment([[0, 0, 0], [1.9, 0, 0]], [[.1, 0, 0], [-1.8, 0, 0]])
    assert {(p, r) for p, r, _ in pairs} == {(0, 1), (1, 0)}
    assert audit.gated_assignment([[0, 0, 2]], [[0, 0, 0]], surface=True) == []
    assert len(audit.gated_assignment([[0, 0, 2]], [[0, 0, 0]])) == 1
    assert audit.gated_assignment([], [[0, 0, 0]]) == []
    assert len(audit.gated_assignment([[0, 0, 0]] * 3, [[0, 0, 0]])) == 1


def test_reference_transform_uses_longitude_latitude_not_incompatible_local_xy(tmp_path):
    path = tmp_path / "reference.osm"
    path.write_text('<osm><node id="46" lat="35.63200892633" lon="139.73048038912"><tag k="local_x" v="85048.9657"/><tag k="local_y" v="43876.1014"/><tag k="ele" v="28.5"/></node><node id="47" lat="35.63200892633" lon="139.73048038912"><tag k="ele" v="29"/></node><way id="1"><nd ref="46"/><nd ref="47"/><tag k="type" v="stop_line"/></way></osm>', encoding="utf-8")
    refs, _, report = audit.read_reference(path)
    np.testing.assert_allclose(refs["bright_bar"][0]["geometry"][0], [-9315.5623467, -40821.7142752, 28.5], atol=.001)
    assert report["local_xy_vs_utm54_remainder_max_m"] < .001


def test_label_sampling_counts_distinct_file_without_index_join(tmp_path, monkeypatch):
    h = laspy.LasHeader(point_format=3, version="1.2")
    h.scales = [.001] * 3
    d = laspy.LasData(h)
    d.x, d.y, d.z = [0, .01, .2, .21], [0] * 4, [1] * 4
    d.user_data = [21, 21, 0, 52]
    d.classification = [22] * 4  # Standard classification is NOT the semantic field.
    path = tmp_path / "labels.las"
    d.write(path)
    monkeypatch.setattr(audit, "CHUNK", 1)
    samples, counts = audit.label_samples(path)
    assert counts == {"0": 1, "21": 2, "52": 1}
    assert len(samples[21]) == 1 and len(samples[42]) == 0 and len(samples[52]) == 1


def item(data, lfs=True):
    return {"path": "test.bin", "size": len(data), **({"lfs": {"oid": hashlib.sha256(data).hexdigest()}} if lfs else {"oid": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()})}


@pytest.mark.parametrize("lfs", [True, False])
def test_download_resumes_verified_partial_and_rechecks_cached_hash(tmp_path, monkeypatch, lfs):
    data = b"abcdef"
    (tmp_path / "test.bin.part").write_bytes(data[:2])
    def response(req, timeout):
        assert req.get_header("Range") == "bytes=2-5"
        result = io.BytesIO(data[2:])
        result.status = 206
        result.headers = {"Content-Range": "bytes 2-5/6"}
        return result
    monkeypatch.setattr(fetch.urllib.request, "urlopen", response)
    monkeypatch.setattr(fetch.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    assert fetch.download(tmp_path, item(data, lfs))["sha256"] == hashlib.sha256(data).hexdigest()
    (tmp_path / "test.bin").write_bytes(b"ghijkl")
    with pytest.raises(ValueError, match="hash mismatch"):
        fetch.download(tmp_path, item(data, lfs))


@pytest.mark.parametrize("status,content_range,body", [(200, None, b"abcdef"), (206, "bytes 0-5/9", b"abcdef"), (206, "bytes 0-5/6", b"abc")])
def test_download_rejects_unbounded_wrong_or_short_response(tmp_path, monkeypatch, status, content_range, body):
    def response(*args, **kwargs):
        result = io.BytesIO(body)
        result.status = status
        result.headers = {"Content-Range": content_range}
        return result
    monkeypatch.setattr(fetch.urllib.request, "urlopen", response)
    monkeypatch.setattr(fetch.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
    with pytest.raises(ValueError):
        fetch.download(tmp_path, item(b"abcdef"))
    assert not (tmp_path / "test.bin").exists()


def test_low_disk_download_fails_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="6 GiB"):
        fetch.download(tmp_path, item(b"abcdef"))


def test_generation_never_supplies_reference_or_confirmation_and_freezes_maps(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    tile = {"path": "tile.las", "core_min": [0, 0], "core_max": [20, 20]}
    (prepared / "preparation.json").write_text(json.dumps({"tiles": [tile], "drives": [{"path": "drive.csv"}]}), encoding="utf-8")
    calls = []
    def discover(cloud, map_path, options, confirmations):
        assert cloud == str(prepared / "tile.las") and map_path is None and confirmations is None
        assert json.loads(options) == gen.OPTIONS
        calls.append("discover")
        return json.dumps({"report": {"discovery": {"candidates": [{"min": [1, 1, 1], "max": [2, 2, 2]}]}}})
    def build(cloud, drive, options, reference, georeference, existing):
        assert cloud == str(prepared / "geometry.las") and drive == str(prepared / "drive.csv")
        assert (options, reference, georeference, existing) == ("{}", None, None, None)
        calls.append("build")
        return json.dumps({"report": {}, "map_json": "{}", "osm": "<osm/>"})
    monkeypatch.setitem(sys.modules, "cloudanalyzer_core", SimpleNamespace(__file__=str(tmp_path / "__init__.py"), discover_vector_map_features=discover, build_vector_map=build))
    monkeypatch.setattr(gen.importlib.metadata, "version", lambda _: "0.1.0")
    out = tmp_path / "generated"
    result = gen.generate(prepared, out, source_commit="1" * 40)
    assert result["baseline_commit"] == "1" * 40
    assert calls == ["discover", "build"] and result["reference_inputs"] == []
    assert len(result["candidates"]) == 1
    (out / "drive.csv.map.json").write_text("tampered", encoding="utf-8")
    # No dataset exists: integrity failure must precede reading any reference.
    with pytest.raises(ValueError, match="road map changed"):
        audit.evaluate(tmp_path / "absent-dataset", out, tmp_path / "evaluation")
    (out / "generation.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="generation changed"):
        audit.evaluate(tmp_path / "absent-dataset", out, tmp_path / "evaluation")
