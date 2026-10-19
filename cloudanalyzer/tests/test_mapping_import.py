"""External outputs enter the existing drafting loop without rerunning SLAM."""
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
from typer.testing import CliRunner

from ca import mapping_import as imports, mapping_job as jobs, mapping_run as runs
from cloudanalyzer_cli.main import app
from tests.test_mapping_job import job_backend, _attach_lane_native, _run_layout


@pytest.fixture
def external_pair(tmp_path, monkeypatch, job_backend):
    cloud = tmp_path / "external.ply"
    cloud.write_text("ply\nformat ascii 1.0\nelement vertex 4\nproperty double x\n"
                     "property double y\nproperty double z\nend_header\n"
                     "0 -1 1\n0 1 1\n10 -1 1\n10 1 1\n")
    trajectory = tmp_path / "external.tum"
    trajectory.write_text("100 0 0 1 0 0 0 1\n101 10 0 1 0 0 0 1\n")
    provenance = {"backend": "fixture-export", "version": "fixed-fixture",
                  "source": "test fixture", "license": "test fixture",
                  "coordinate_frame": "fixture_map", "length_unit": "metres", "up_axis": "z",
                  "pose_convention": "world_from_sensor", "time_unit": "seconds",
                  "time_basis": "fixture seconds", "configuration": {"deskew": False}}
    monkeypatch.setattr(imports.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    return cloud, trajectory, provenance


def _start(pair, root):
    cloud, trajectory, provenance = pair
    return runs.start_mapping_run_from_slam(str(cloud), str(trajectory), str(root),
                                           provenance, _run_layout())


def test_import_preserves_map_times_frame_and_provenance_without_running_slam(external_pair, tmp_path, monkeypatch):
    cloud, trajectory, provenance = external_pair
    before = cloud.read_bytes(), trajectory.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("external imports must not run odometry or fusion")

    monkeypatch.setattr(jobs, "odometry", forbidden)
    monkeypatch.setattr(jobs, "fix_session", forbidden)
    root = tmp_path / "import"
    run = _start(external_pair, root)
    assert run["status"] == "needs_agent" and run["revision"] == 0
    assert run["pointcloud"]["quality_status"] == "imported_unverified"
    assert run["pointcloud"]["coordinate_frame"] == "fixture_map"
    assert run["pointcloud"]["files"]["map"] == jobs._artifact(cloud)
    assert run["pointcloud"]["files"]["source_trajectory"] == jobs._artifact(trajectory)
    assert not (root / "map.ply").exists() and not (root / "odometry").exists()
    np.testing.assert_array_equal(np.loadtxt(root / "trajectory.txt")[:, [3, 7, 11]], [[0, 0, 1], [10, 0, 1]])
    declaration = json.loads((root / "slam-import.json").read_text())
    assert declaration["producer_declaration"] == provenance
    assert not declaration["alignment_applied"] and not declaration["timestamps_rebased"]
    assert (cloud.read_bytes(), trajectory.read_bytes()) == before
    assert run["pointcloud_retry_allowed"] is False
    assert set(run["action_contract"]) == runs.FROZEN_MAP_ACTIONS
    assert jobs.inspect_mapping_job(str(root))["storage"]["point_map_copied"] is False
    assert run["pointcloud"]["input_recording_coverage"] == "not_verified"


@pytest.mark.parametrize("rows", [
    "0 0 0 0 0 0 0 1\n0 1 0 0 0 0 0 1\n",
    "1 0 0 0 0 0 0 1\n0 1 0 0 0 0 0 1\n",
    "0 0 0 0 0 0 0 0\n1 1 0 0 0 0 0 1\n",
    "0 nan 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n",
    "0 0 0 0 0 0 0 1\n",
    "0 0 0 0 0 0 1\n1 1 0 0 0 0 1\n",
    "0 -1e308 0 0 0 0 0 1\n1 1e308 0 0 0 0 0 1\n",
])
def test_invalid_poses_are_rejected_without_creating_an_output(external_pair, tmp_path, rows):
    external_pair[1].write_text(rows)
    root = tmp_path / "import"
    with pytest.raises(ValueError, match="TUM"):
        _start(external_pair, root)
    assert not root.exists()


def test_frame_units_are_explicit_and_not_silently_converted(external_pair, tmp_path):
    external_pair[2]["length_unit"] = "centimetres"
    with pytest.raises(ValueError, match="metre-frame"):
        _start(external_pair, tmp_path / "import")
    assert not (tmp_path / "import").exists()


def test_disk_guard_rejects_before_loading_or_creating_outputs(external_pair, tmp_path, monkeypatch):
    monkeypatch.setattr(imports.shutil, "disk_usage", lambda _: SimpleNamespace(free=imports.MIN_FREE_BYTES))
    monkeypatch.setattr(imports, "load_frame", lambda _: pytest.fail("low-space import loaded the map"))
    with pytest.raises(ValueError, match="disk space"):
        _start(external_pair, tmp_path / "import")
    assert not (tmp_path / "import").exists()


def test_input_size_guard_does_not_create_or_copy_large_files(external_pair, tmp_path, monkeypatch):
    monkeypatch.setattr(imports, "MAX_MAP_BYTES", 8)
    with pytest.raises(ValueError, match="size limit"):
        _start(external_pair, tmp_path / "import")
    assert not (tmp_path / "import").exists()


def test_changed_external_input_and_recording_actions_are_rejected(external_pair, tmp_path):
    root = tmp_path / "import"
    _start(external_pair, root)
    before = (root / "run.json").read_bytes(), (root / "job.json").read_bytes()
    with pytest.raises(ValueError, match="frozen external"):
        runs.advance_mapping_run(str(root), {"type": "inspect_gaps", "candidate_id": 2, "offset": 0}, "Unavailable raw log", 0)
    assert ((root / "run.json").read_bytes(), (root / "job.json").read_bytes()) == before
    from ca import mapping_retry
    with pytest.raises(ValueError, match="raw-recording evidence"):
        mapping_retry._raw_source(root, jobs._load(root))
    assert not (root / "gap-source").exists()
    external_pair[1].write_text("100 0 0 0 0 0 0 1\n101 20 0 0 0 0 0 1\n")
    with pytest.raises(ValueError, match="changed"):
        runs.inspect_mapping_run(str(root))


def test_external_pair_drafts_finishes_and_exports_with_existing_quality_holds(external_pair, tmp_path):
    _attach_lane_native()
    from ca import mapping_bundle
    root = tmp_path / "import"
    run = _start(external_pair, root)
    run = runs.advance_mapping_run(str(root), {"type": "inspect", "candidate_ids": [1]}, "Read source sections", run["revision"])
    run = runs.advance_mapping_run(str(root), {"type": "draft", "decisions": [
        {"candidate_id": 1, "action": "include", "reason": "Retain inspected source extent"}]}, "Fixed layout", run["revision"])
    assert run["draft_result"]["status"] == "audited_draft"
    run = runs.advance_mapping_run(str(root), {"type": "finish", "candidate_id": 2}, "Deliver with source holds", run["revision"])
    assert run["output"]["status"] == "draft_needs_review" and not run["output"]["deployment_ready"]
    assert run["output"]["diagnosis"]["review_holds"]["traces_total"] > 0
    assert {"map", "trajectory", "graph", "source_trajectory", "slam_import", "hd_map"} <= run["output"]["artifacts"].keys()
    native = pytest.importorskip("cloudanalyzer_core")
    graph = native.PoseGraph.from_g2o((root / "poses.g2o").read_text())
    np.testing.assert_allclose(np.asarray(graph.poses())[:, :3], np.loadtxt(root / "trajectory.txt").reshape(-1, 3, 4), atol=1e-10, rtol=0)
    target = tmp_path / "external-review.zip"
    mapping_bundle.export_mapping_run(str(root), str(target), "Test fixture")
    inspected = mapping_bundle.inspect_mapping_bundle(str(target))
    assert not inspected["deployment_ready"]
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        imported = json.loads(archive.read(manifest["roles"]["slam_import"]))
    assert imported["producer_declaration"] == external_pair[2]


def test_cli_and_mcp_expose_external_import(external_pair, tmp_path):
    cloud, trajectory, provenance = external_pair
    declaration, layout = tmp_path / "provenance.json", tmp_path / "layout.json"
    declaration.write_text(json.dumps(provenance))
    layout.write_text(json.dumps(_run_layout()))
    result = CliRunner().invoke(app, ["mapping-import", str(cloud), "--trajectory", str(trajectory),
        "--out", str(tmp_path / "cli-import"), "--provenance", str(declaration), "--layout", str(layout)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "needs_agent"
    from ca.mcp_server import TOOLS
    assert runs.start_mapping_run_from_slam in TOOLS
