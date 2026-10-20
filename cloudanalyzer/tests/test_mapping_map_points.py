"""Point-map evidence stays bounded, tied to seen profiles and read-only."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ca import mapping_job as jobs, mapping_run as runs, mapping_map_points as points
from tests.test_mapping_import import external_pair, job_backend, _partial_external_draft


@pytest.fixture
def inspected_points(external_pair, tmp_path, monkeypatch):
    cloud = external_pair[0]
    cloud.write_text("ply\nformat ascii 1.0\nelement vertex 4\nproperty double x\n"
                     "property double y\nproperty double z\nend_header\n"
                     "6 0 1\n6.5 0 2\n6.75 0 3\n6.751 0 4\n")
    root = tmp_path / "import"
    run = _partial_external_draft(external_pair, root, monkeypatch)
    run = runs.advance_mapping_run(str(root), {"type": "inspect_gaps", "candidate_id": 2, "offset": 0},
                                   "Read saved gap profiles", run["revision"])
    monkeypatch.setattr(jobs.core(), "read", pytest.importorskip("cloudanalyzer_core").read, raising=False)
    return root, run, external_pair


def _action(**changes):
    return {"type": "inspect_map_points", "candidate_id": 2, "gap_ids": [1], "stations_m": [6., 8.], **changes}


def _advance(root, action=None):
    run = runs.inspect_mapping_run(str(root))
    return runs.advance_mapping_run(str(root), action or _action(), "Inspect the fixed map returns", run["revision"])


def test_map_points_keep_all_heights_inclusive_radius_and_original_record_indices(inspected_points, monkeypatch):
    root, before_run, pair = inspected_points
    parent = jobs._load(root)["attempts"][-1]
    files = [root / "job.json", pair[0], pair[1], *[Path(f["path"]) for f in parent["files"].values()]]
    before = {p: p.read_bytes() for p in files}
    native_read = jobs.core().read
    calls = []
    def read(path):
        calls.append(path)
        return native_read(path)
    monkeypatch.setattr(jobs.core(), "read", read)
    for name in ("propose_road_corridors", "audit_vector_map_quality_details", "audit_vector_map_ground_consensus_details"):
        monkeypatch.setattr(jobs.core(), name, lambda *a: pytest.fail("point inspection must not extract or audit"))
    run = _advance(root)
    observed = run["map_point_observation"]
    near, empty = observed["profiles"]
    assert near["station_m"] == 6. and near["trajectory"] == [6., 0., 1.]
    assert near["points_total"] == near["occupied_xy_cells"] == 3
    assert near["height_range_m"] == [1., 3.] and not near["points_limited"]
    assert near["points"] == [{"source_record_index": i, "xyz": xyz}
                              for i, xyz in enumerate([[6., 0., 1.], [6.5, 0., 2.], [6.75, 0., 3.]])]
    assert near["saved_reference_ground_height_m"] == 1.
    assert empty["points_total"] == empty["occupied_xy_cells"] == 0 and empty["points"] == []
    assert empty["height_range_m"] is None and not empty["points_limited"]
    assert not observed["raw_evidence_available"] and not observed["fresh_source_audit_performed"]
    assert run["remaining_attempts"] == before_run["remaining_attempts"]
    assert all(p.read_bytes() == data for p, data in before.items())
    assert "map_point_observation" not in run["history"][-1]
    again = _advance(root)["map_point_observation"]
    assert again == observed and len(calls) == 1 and len(list(root.glob("map-points-*.json"))) == 1


@pytest.mark.parametrize("changes", [
    {"gap_ids": []}, {"gap_ids": [True]}, {"gap_ids": [1, 1]}, {"gap_ids": [99]},
    {"stations_m": []}, {"stations_m": None}, {"stations_m": [True]}, {"stations_m": ["6"]},
    {"stations_m": [float("nan")]}, {"stations_m": [float("inf")]}, {"stations_m": [10**400]},
    {"stations_m": [6., 6.]}, {"stations_m": list(range(9))}, {"stations_m": [5.]},
    {"candidate_id": 1}, {"candidate_id": True},
])
def test_unseen_or_invalid_map_queries_do_not_allocate_history(inspected_points, monkeypatch, changes):
    root, run, _ = inspected_points
    before = {p: (root / p).read_bytes() for p in ("job.json", "run.json")}
    monkeypatch.setattr(jobs.core(), "read", lambda *a: pytest.fail("invalid query must not read points"))
    with pytest.raises(ValueError, match="gap|station|profile|audited"):
        runs.advance_mapping_run(str(root), _action(**changes), "Reject an uninspected query", run["revision"])
    assert all((root / p).read_bytes() == data for p, data in before.items())
    assert not list(root.glob("map-points-*"))


def test_map_points_require_a_seen_gap_page(external_pair, tmp_path, monkeypatch):
    root = tmp_path / "import"
    run = _partial_external_draft(external_pair, root, monkeypatch)
    before = (root / "run.json").read_bytes()
    with pytest.raises(ValueError, match="inspect gaps through this run"):
        runs.advance_mapping_run(str(root), _action(), "No prior gap page", run["revision"])
    assert (root / "run.json").read_bytes() == before


def test_map_points_resume_reuses_completed_observation_without_reading_map_twice(inspected_points, monkeypatch):
    root, run, _ = inspected_points
    inspect = points.inspect
    def interrupted(*args, **kwargs):
        inspect(*args, **kwargs)
        raise KeyboardInterrupt()
    monkeypatch.setattr(points, "inspect", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _advance(root)
    assert runs.inspect_mapping_run(str(root))["status"] == "interrupted"
    monkeypatch.setattr(points, "inspect", inspect)
    monkeypatch.setattr(jobs.core(), "read", lambda *a: pytest.fail("completed query must be cached"))
    resumed = _advance(root, {"type": "resume"})
    assert resumed["status"] == "needs_agent" and resumed["map_point_observation"]["profiles"][0]["points_total"] == 3
    assert resumed["remaining_attempts"] == run["remaining_attempts"]
    assert len(list(root.glob("map-points-*"))) == 1


def test_map_point_evidence_tampering_is_detected_in_retained_history(inspected_points):
    root, _, _ = inspected_points
    observed = _advance(root)["map_point_observation"]
    path = Path(observed["file"]["path"])
    saved = json.loads(path.read_text())
    saved["profiles"][0]["points"][0]["xyz"][2] += 1.
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="changed"):
        runs.inspect_mapping_run(str(root))


@pytest.mark.parametrize("fault", ["count", "nonfinite", "bounds"])
def test_map_point_reader_cannot_silently_drop_or_change_coordinates(inspected_points, monkeypatch, fault):
    root, _, _ = inspected_points
    xyz = np.asarray(jobs.core().read(str(inspected_points[2][0]))["positions"]).copy()
    if fault == "count":
        xyz = xyz[:-1]
    elif fault == "nonfinite":
        xyz[0, 2] = np.nan
    else:
        xyz[0, 0] = 1e16
    monkeypatch.setattr(jobs.core(), "read", lambda _: {"positions": xyz})
    before = (root / "job.json").read_bytes()
    with pytest.raises(ValueError, match="finite XYZ records"):
        _advance(root)
    assert not list(root.glob("map-points-*"))
    assert (root / "job.json").read_bytes() == before
    assert runs.inspect_mapping_run(str(root))["status"] == "interrupted"


@pytest.mark.parametrize("limit", ["maximum_map_points", "maximum_map_bytes"])
def test_map_point_input_limits_reject_before_native_load_and_history(inspected_points, monkeypatch, limit):
    root, run, _ = inspected_points
    monkeypatch.setitem(points.PROTOCOL, limit, 1)
    monkeypatch.setattr(jobs.core(), "read", lambda *a: pytest.fail("oversized map must not be loaded"))
    before = (root / "run.json").read_bytes()
    with pytest.raises(ValueError, match="map point/byte budget"):
        runs.advance_mapping_run(str(root), _action(), "Respect bounded input", run["revision"])
    assert (root / "run.json").read_bytes() == before


@pytest.mark.parametrize("guard", ["space", "output"])
def test_map_point_storage_limits_leave_no_observation_or_changed_job(inspected_points, monkeypatch, guard):
    root, run, _ = inspected_points
    if guard == "space":
        monkeypatch.setattr(points.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 1024**3))
        monkeypatch.setattr(jobs.core(), "read", lambda *a: pytest.fail("space guard must precede map loading"))
    else:
        monkeypatch.setattr(points, "MAX_REPORT_BYTES", 8)
    before = (root / "job.json").read_bytes(), (root / "run.json").read_bytes()
    with pytest.raises(ValueError, match="disk space|page budget"):
        points.inspect(root, 2, run["gap_observation"]["file"], [1], [6.])
    assert not list(root.glob("map-points-*"))
    assert before == ((root / "job.json").read_bytes(), (root / "run.json").read_bytes())


def test_map_point_previews_report_truncation_and_cover_source_order(external_pair, tmp_path, monkeypatch):
    n = 300
    rows = [[6. + i / 1000, 0., float(i)] for i in range(n)]
    external_pair[0].write_text("ply\nformat ascii 1.0\nelement vertex 300\nproperty double x\n"
                                "property double y\nproperty double z\nend_header\n"
                                + ''.join(' '.join(map(str, row)) + '\n' for row in rows))
    root = tmp_path / "import"
    run = _partial_external_draft(external_pair, root, monkeypatch)
    run = runs.advance_mapping_run(str(root), {"type": "inspect_gaps", "candidate_id": 2, "offset": 0},
                                   "Read truncated-map context", run["revision"])
    monkeypatch.setattr(jobs.core(), "read", pytest.importorskip("cloudanalyzer_core").read, raising=False)
    profile = _advance(root)["map_point_observation"]["profiles"][0]
    assert profile["points_total"] == n and profile["points_limited"] and len(profile["points"]) == 128
    ids = [p["source_record_index"] for p in profile["points"]]
    assert ids[0] == 0 and ids[-1] == n - 1 and ids == sorted(set(ids))
    assert profile["height_range_m"] == [0., 299.]
    np.testing.assert_allclose([p["xyz"] for p in profile["points"]], np.asarray(rows)[ids])
