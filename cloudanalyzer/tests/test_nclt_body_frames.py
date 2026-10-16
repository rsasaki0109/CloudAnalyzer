"""Documented NCLT frame/lever-arm geometry and fresh reference provenance."""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


@pytest.fixture
def frames(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("nclt_body_frames")


@pytest.mark.parametrize("rpy", [[0., 0., 0.], [.3, -.2, .5], [-.7, .6, -2.]])
def test_body_origin_and_point_coordinates_agree_with_scipy_and_documented_axes(frames, rpy):
    position = np.array([10., -2., 3.])
    point_native = np.array([1., 2., -.4])
    axes = np.diag([1., -1., -1.])
    rotation = Rotation.from_euler("xyz", rpy).as_matrix()
    actual = frames.body_pose_z_up(position, rpy)
    # Synchronized scan coordinates already refer to the axle, rather than a
    # displaced LiDAR origin: no mounting translation appears in this identity.
    np.testing.assert_allclose(actual[:3, :3] @ (axes @ point_native) + actual[:3, 3],
                               axes @ (rotation @ point_native + position), atol=1e-12)
    np.testing.assert_array_equal(actual[:3, 3], axes @ position)
    up = frames.body_up_from_rpy(*rpy[:2])
    np.testing.assert_allclose(up, axes @ rotation.T @ [0., 0., -1.], atol=1e-12)
    assert np.linalg.norm(up) == pytest.approx(1.)


def test_native_imu_axes_origin_and_gravity_include_the_documented_lever_arm(frames):
    rpy, position = [.3, -.2, .5], np.array([10., -2., 3.])
    native_rotation = Rotation.from_euler("xyz", rpy).as_matrix()
    axes = np.diag([1., -1., -1.])
    body = frames.body_pose_z_up(position, rpy)
    imu = frames.imu_pose_from_body_z_up(body)
    np.testing.assert_allclose(imu[:3, :3], axes @ native_rotation, atol=1e-12)
    np.testing.assert_allclose(imu[:3, 3], axes @ (position + native_rotation @ [-.11, -.18, -.71]), atol=1e-12)
    # Specific force for a stationary native IMU cancels z-up world gravity.
    gravity = np.array([0., 0., -9.81])
    specific_force = native_rotation.T @ [0., 0., -9.81]
    np.testing.assert_allclose(imu[:3, :3] @ specific_force + gravity, 0., atol=1e-12)
    np.testing.assert_array_equal(body, frames.body_pose_z_up(position, rpy))


@pytest.mark.parametrize("change", ["shape", "reflection", "homogeneous", "nan"])
def test_invalid_pose_conversions_are_rejected(frames, change):
    pose = np.eye(4)
    if change == "shape":
        pose = pose[:3]
    elif change == "reflection":
        pose[2, 2] = -1.
    elif change == "homogeneous":
        pose[3, 0] = 1.
    else:
        pose[0, 3] = np.nan
    with pytest.raises(ValueError):
        frames.imu_pose_from_body_z_up(pose)
    with pytest.raises(ValueError):
        frames.body_pose_z_up([np.nan, 0., 0.], [0., 0., 0.])


def test_fresh_reference_uses_body_origin_and_records_helper_without_mutating_inputs(frames, tmp_path):
    from ca import mapping_job as jobs

    prepare = importlib.import_module("prepare_nclt_body_reference")
    folder = tmp_path / "job"
    folder.mkdir()
    motion = folder / "motion.tum"
    np.savetxt(motion, np.column_stack(([100., 100.1, 100.2], np.zeros((3, 3)),
                                       np.tile([0., 0., 0., 1.], (3, 1)))))
    motion_artifact = prepare.artifact(motion)
    (folder / "job.json").write_text(json.dumps({
        "schema": jobs.SCHEMA,
        "pointcloud": {"source_motion": {"trajectory": motion_artifact}},
    }), encoding="utf-8")
    raw = tmp_path / "groundtruth.csv"
    rows = np.column_stack((100_000_000 + np.arange(21) * 10_000,
                            np.arange(21) * .02, np.arange(21) * -.01, np.ones(21),
                            np.arange(21) * .003, np.arange(21) * -.002, np.arange(21) * .01))
    np.savetxt(raw, rows, delimiter=",")
    inputs = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [motion, raw, folder / "job.json"]}
    output = tmp_path / "reference"
    result = prepare.prepare(folder, raw, output, "https://example.invalid/nclt.csv")
    selected = np.loadtxt(output / "reference-raw-rows.csv", delimiter=",")
    actual = np.loadtxt(output / "reference.tum")
    for source, row in zip(selected, actual, strict=True):
        expected = frames.body_pose_z_up(source[1:4], source[4:7])
        np.testing.assert_allclose(row[1:4], expected[:3, 3], atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(row[4:]).as_matrix(), expected[:3, :3], atol=1e-12)
    assert result["schema"] == "cloudanalyzer.nclt_body_reference_preparation.v1"
    assert result["frame_id"] == "nclt_body_z_up"
    assert result["inputs"]["frame_module"] == prepare.artifact(SCRIPTS / "nclt_body_frames.py")
    provenance = json.loads((output / "reference-provenance.json").read_text())
    assert provenance["frame_id"] == "nclt_body_z_up" and not provenance["used_for_generation"]
    for path, digest in inputs.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        prepare.prepare(folder, raw, output, "https://example.invalid/nclt.csv")
