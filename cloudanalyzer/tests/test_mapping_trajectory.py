"""Reference comparison must expose regressions and preserve mapping artifacts."""
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from ca import mapping_job as jobs, mapping_trajectory as comparison
from cloudanalyzer_cli.main import app

PROVENANCE = {"source": "Synthetic reference for contract verification", "license": "MIT",
              "frame": "sensor origin, xyz metres", "time_basis": "seconds, same clock", "used_for_generation": False}


@pytest.fixture
def fixture(tmp_path):
    if jobs.core() is None:
        pytest.skip("needs native PoseGraph parser")
    root = tmp_path / "job"
    root.mkdir()
    source = root / "source.mcap"
    source.write_bytes(b"synthetic source provenance")
    points = np.array([[k, (k % 3) * .5, 0.] for k in range(7)])
    original = points.copy()
    original[:, 2] = np.arange(7)**2 * .08
    initial = root / "original.tum"
    truth = tmp_path / "reference.tum"
    q = np.tile([0., 0., 0., 1.], (7, 1))
    comparison._write(initial, np.arange(7.), original, q)
    comparison._write(truth, np.arange(7.), points, q)
    ids = [0, 2, 4, 6]
    graph = root / "corrected.g2o"
    graph.write_text("".join(f"VERTEX_SE3:QUAT {i} {' '.join(map(str, points[i]))} 0 0 0 1\n" for i in ids))
    poses = np.tile(np.eye(4), (len(ids), 1, 1))
    poses[:, :3, 3] = points[ids]
    corrected = root / "corrected.txt"
    np.savetxt(corrected, poses[:, :3].reshape(len(ids), 12), fmt="%.17g")
    cloud = root / "map.ply"
    cloud.write_bytes(b"unchanged point-map sentinel")
    job = {"schema": jobs.SCHEMA, "source": jobs._artifact(source), "attempts": [{"id": 1}],
           "selected": 1, "pointcloud": {"quality_status": "generated_unverified",
               "files": {"map": jobs._artifact(cloud), "graph": jobs._artifact(graph), "trajectory": jobs._artifact(corrected)},
               "source_motion": {"trajectory": jobs._artifact(initial)}}}
    jobs._save(root / "job.json", job)
    return root, truth, tmp_path / "report.json"


def _run(fixture, **kwargs):
    root, truth, out = fixture
    return comparison.evaluate_mapping_trajectory(str(root), str(truth), PROVENANCE, str(out), **kwargs)


def test_same_original_ids_detect_improvement_and_preserve_every_job_file(fixture):
    root, truth, out = fixture
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    result = _run(fixture)
    assert result["coverage"]["matched_original_frame_ids"] == [0, 2, 4, 6]
    assert result["coverage"]["retained_pose_fraction"] == 1.
    assert result["results"]["original"]["ate"]["rmse"] > .1
    assert result["results"]["corrected"]["ate"]["rmse"] < 1e-10
    assert result["change"]["ate_rmse_m_corrected_minus_original"] < -.1
    stored = json.loads(out.read_text())
    assert stored["results"]["original"]["matched_trajectory"]["timestamps"] == stored["results"]["corrected"]["matched_trajectory"]["timestamps"]
    assert "matched_trajectory" not in result["results"]["original"]
    assert "error_series" not in result["results"]["original"]
    assert "quality_gate" not in stored["results"]["original"]
    assert result["reference_independence"] == "caller_declared_not_used_for_generation"
    assert result["protocol"]["held_out_alignment"] is False
    assert json.loads(out.read_text())["inputs"]["reference"] == jobs._artifact(truth)
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


def test_reports_a_regression_instead_of_changing_or_selecting_the_map(fixture):
    root, truth, _ = fixture
    rows = np.loadtxt(truth)
    rows[:, 3] = np.arange(7)**2 * .08
    np.savetxt(truth, rows, fmt="%.17g")
    result = _run(fixture)
    assert result["change"]["ate_rmse_m_corrected_minus_original"] > .1
    assert json.loads((root / "job.json").read_text())["selected"] == 1


def test_reports_partial_coverage_without_extrapolating(fixture):
    _, truth, _ = fixture
    np.savetxt(truth, np.loadtxt(truth)[:5], fmt="%.17g")
    result = _run(fixture)
    assert result["coverage"]["matched_original_frame_ids"] == [0, 2, 4]
    assert result["coverage"]["retained_pose_fraction"] == .75
    assert result["coverage"]["matched_duration_s"] == 4.


def test_interpolates_reference_at_the_original_scan_times(fixture):
    _, truth, _ = fixture
    points = np.loadtxt(truth)
    rows = []
    for row in points:
        for offset in (-.01, .01):
            sample = row.copy()
            sample[0] += offset
            rows.append(sample)
    np.savetxt(truth, rows, fmt="%.17g")
    result = _run(fixture)
    assert result["coverage"]["matched_original_frame_ids"] == [0, 2, 4, 6]
    assert result["coverage"]["max_nearest_reference_delta_s"] == pytest.approx(.01)
    assert result["results"]["corrected"]["ate"]["rmse"] < 1e-10


def test_rejects_an_unconstrained_straight_reference(fixture):
    _, truth, out = fixture
    rows = np.loadtxt(truth)
    rows[:, 2:4] = 0.
    np.savetxt(truth, rows, fmt="%.17g")
    with pytest.raises(ValueError, match="cannot constrain"):
        _run(fixture)
    assert not out.exists()


def test_used_generation_reference_is_explicitly_nonindependent(fixture):
    root, truth, out = fixture
    result = comparison.evaluate_mapping_trajectory(str(root), str(truth), {**PROVENANCE, "used_for_generation": True}, str(out))
    assert result["reference_independence"] == "not_independent"


def test_known_input_reuse_overrides_an_incorrect_independence_declaration(fixture):
    root, truth, _ = fixture
    truth.write_bytes((root / "original.tum").read_bytes())
    result = _run(fixture)
    assert result["reference_independence"] == "not_independent"
    assert result["reference_matches_recorded_inputs"] == ["source_motion_trajectory"]


def test_saves_no_report_if_the_output_path_is_created_during_evaluation(fixture, monkeypatch):
    out = fixture[2]
    evaluate = comparison.evaluate_trajectory
    def create(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        out.write_text("competing report")
        return result
    monkeypatch.setattr(comparison, "evaluate_trajectory", create)
    with pytest.raises(FileExistsError):
        _run(fixture)
    assert out.read_text() == "competing report"
    assert not list(out.parent.glob(".ca-trajectory-*"))


@pytest.mark.parametrize("provenance", [{}, {**PROVENANCE, "used_for_generation": "false"}, {**PROVENANCE, "frame": ""}, {**PROVENANCE, "extra": 1}])
def test_requires_complete_explicit_reference_provenance(fixture, provenance):
    root, truth, out = fixture
    with pytest.raises(ValueError):
        comparison.evaluate_mapping_trajectory(str(root), str(truth), provenance, str(out))
    assert not out.exists()


@pytest.mark.parametrize("tolerance", [0., float("nan"), float("inf"), True, 1.01])
def test_rejects_invalid_time_policy(fixture, tolerance):
    with pytest.raises(ValueError, match="max_time_delta"):
        _run(fixture, max_time_delta=tolerance)
    assert not fixture[2].exists()


def test_rejects_clock_shift_before_publishing(fixture):
    _, truth, out = fixture
    rows = np.loadtxt(truth)
    rows[:, 0] += .5
    np.savetxt(truth, rows)
    with pytest.raises(ValueError, match="reference-supported"):
        _run(fixture)
    assert not out.exists()


def test_does_not_overwrite_reports_or_write_inside_the_job(fixture):
    root, truth, out = fixture
    out.write_text("keep existing report")
    with pytest.raises(FileExistsError):
        _run(fixture)
    assert out.read_text() == "keep existing report"
    with pytest.raises(ValueError, match="outside"):
        comparison.evaluate_mapping_trajectory(str(root), str(truth), PROVENANCE, str(root / "report.json"))


def test_changed_frozen_motion_is_rejected(fixture):
    root, _, out = fixture
    with (root / "original.tum").open("a") as stream:
        stream.write("# changed\n")
    with pytest.raises(ValueError, match="changed"):
        _run(fixture)
    assert not out.exists()


def test_change_during_evaluation_is_rejected(fixture, monkeypatch):
    root, _, out = fixture
    evaluate = comparison.evaluate_trajectory
    def change(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        (root / "map.ply").write_bytes(b"interleaved map mutation")
        return result
    monkeypatch.setattr(comparison, "evaluate_trajectory", change)
    with pytest.raises(ValueError, match="changed"):
        _run(fixture)
    assert not out.exists()


def test_corrected_graph_must_match_the_frozen_trajectory(fixture):
    root, _, out = fixture
    path = root / "corrected.txt"
    rows = np.loadtxt(path)
    rows[0, 3] += 1.
    np.savetxt(path, rows)
    job = json.loads((root / "job.json").read_text())
    job["pointcloud"]["files"]["trajectory"] = jobs._artifact(path)
    jobs._save(root / "job.json", job)
    with pytest.raises(ValueError, match="do not agree"):
        _run(fixture)
    assert not out.exists()


def test_rejects_legacy_motion_without_inventing_frame_timestamps(fixture):
    root, _, out = fixture
    job = json.loads((root / "job.json").read_text())
    del job["pointcloud"]["source_motion"]
    jobs._save(root / "job.json", job)
    with pytest.raises(ValueError, match="source_motion"):
        _run(fixture)
    assert not out.exists()


def test_cli_emits_the_same_report_contract(fixture):
    root, truth, out = fixture
    provenance = truth.with_suffix(".json")
    provenance.write_text(json.dumps(PROVENANCE))
    result = CliRunner().invoke(app, ["mapping-trajectory-evaluate", str(root), "--reference", str(truth),
                                      "--provenance", str(provenance), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["schema"] == comparison.SCHEMA


def _all_graph_nodes(fixture):
    root, truth, _ = fixture
    rows = np.loadtxt(truth)
    points = rows[:, 1:4]
    graph = root / "corrected.g2o"
    graph.write_text("".join(f"VERTEX_SE3:QUAT {i} {' '.join(map(str, p))} 0 0 0 1\n" for i, p in enumerate(points)))
    poses = np.tile(np.eye(4), (len(points), 1, 1))
    poses[:, :3, 3] = points
    corrected = root / "corrected.txt"
    np.savetxt(corrected, poses[:, :3].reshape(len(points), 12), fmt="%.17g")
    job = json.loads((root / "job.json").read_text())
    job["pointcloud"]["files"].update(graph=jobs._artifact(graph), trajectory=jobs._artifact(corrected))
    jobs._save(root / "job.json", job)


def test_prefix_alignment_excludes_evaluated_reference_samples(fixture):
    _all_graph_nodes(fixture)
    root, truth, _ = fixture
    first = _run(fixture, alignment_prefix_fraction=4/7)
    assert first["protocol"]["held_out_alignment"] is True
    assert first["coverage"]["alignment_original_frame_ids"] == [0, 1, 2, 3]
    assert first["coverage"]["evaluated_original_frame_ids"] == [4, 5, 6]
    assert first["coverage"]["matched_poses"] == 7 and first["coverage"]["evaluated_poses"] == 3
    assert first["results"]["corrected"]["ate"]["rmse"] < 1e-10
    rows = np.loadtxt(truth)
    rows[4:, 2:4] += [4., 1.5]
    np.savetxt(truth, rows, fmt="%.17g")
    second = comparison.evaluate_mapping_trajectory(str(root), str(truth), PROVENANCE,
               str(truth.parent / "suffix-changed.json"), alignment_prefix_fraction=4/7)
    for name in ("original", "corrected"):
        assert second["results"][name]["alignment"] == first["results"][name]["alignment"]
    assert second["results"]["corrected"]["ate"]["rmse"] == pytest.approx(np.hypot(4., 1.5))


def test_prefix_alignment_applies_the_same_rigid_frame_to_orientations(fixture):
    from scipy.spatial.transform import Rotation
    _all_graph_nodes(fixture)
    _, truth, _ = fixture
    rows = np.loadtxt(truth)
    rotation = Rotation.from_euler('z', .3)
    rows[:, 1:4] = rotation.apply(rows[:, 1:4]) + [100., -20., 4.]
    rows[:, 4:] = rotation.as_quat()
    np.savetxt(truth, rows, fmt="%.17g")
    result = _run(fixture, alignment_prefix_fraction=4/7)
    assert result["results"]["corrected"]["ate"]["rmse"] < 1e-10
    assert result["results"]["corrected"]["ate_rotation"]["rmse"] < 1e-5
    np.testing.assert_allclose(result["results"]["corrected"]["alignment"]["rotation_matrix"], rotation.as_matrix(), atol=1e-10)


@pytest.mark.parametrize("fraction", [0., float('nan'), float('inf'), True, 1.01])
def test_rejects_invalid_prefix_fraction(fixture, fraction):
    with pytest.raises(ValueError, match="alignment_prefix_fraction"):
        _run(fixture, alignment_prefix_fraction=fraction)
    assert not fixture[2].exists()


def test_prefix_alignment_needs_enough_fit_and_evaluation_samples(fixture):
    with pytest.raises(ValueError, match="3 fitting and 3 evaluated"):
        _run(fixture, alignment_prefix_fraction=.5)
    assert not fixture[2].exists()


def test_prefix_alignment_rejects_an_unconstrained_fit_even_if_suffix_turns(fixture):
    _all_graph_nodes(fixture)
    _, truth, out = fixture
    rows = np.loadtxt(truth)
    rows[:4, 2:4] = 0.
    np.savetxt(truth, rows, fmt='%.17g')
    with pytest.raises(ValueError, match='prefix cannot constrain'):
        _run(fixture, alignment_prefix_fraction=4/7)
    assert not out.exists()


def test_nclt_sensor_transform_and_reference_bracket_preparation(fixture):
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[2] / 'scripts/prepare_nclt_reference.py'
    spec = importlib.util.spec_from_file_location('nclt_reference', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = np.array([1e6, 20., 30., -1., .1, -.2, .3])
    def xyz_rotation(angles):
        r, p, y = angles
        rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        return rz @ ry @ rx
    body = np.eye(4)
    body[:3, :3] = xyz_rotation(row[4:])
    body[:3, 3] = row[1:4]
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = xyz_rotation(np.radians(module.BODY_VEL[3:]))
    extrinsic[:3, 3] = module.BODY_VEL[:3]
    flip = np.diag([1., -1., -1., 1.])
    np.testing.assert_allclose(module.sensor_pose(row), flip @ body @ extrinsic @ flip, atol=1e-12)
    root, _, _ = fixture
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    raw = root.parent / 'groundtruth.csv'
    rows = [np.array([t * 1e6, t, t % 3, 0., .1, -.2, .3]) for t in np.arange(-.01, 6.02, .02)]
    np.savetxt(raw, rows, delimiter=',', fmt='%.17g')
    out = root.parent / 'reference-prepared'
    result = module.prepare(root, raw, out, 'https://example.test/nclt.csv')
    selected = np.loadtxt(out / 'reference-raw-rows.csv', delimiter=',')
    tum = np.loadtxt(out / 'reference.tum')
    np.testing.assert_allclose(tum[:, 0], selected[:, 0] / 1e6, atol=0.)
    for index, raw_row in enumerate(selected):
        np.testing.assert_allclose(tum[index, 1:4], module.sensor_pose(raw_row)[:3, 3], atol=1e-12)
    assert result['selected_bracket_rows'] <= 2 * result['original_scan_poses']
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    with pytest.raises(FileExistsError):
        module.prepare(root, raw, out, 'https://example.test/nclt.csv')
