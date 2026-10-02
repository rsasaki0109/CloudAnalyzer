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
