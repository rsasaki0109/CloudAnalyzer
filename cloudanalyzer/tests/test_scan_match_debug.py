"""Tests for scan matching diagnostics."""

import weakref
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest

from ca import _rust
import ca.metrics as metrics
import ca.scan_match_debug as debug
from ca.metrics import summarize, threshold_stats
from ca.registration import RegistrationResult
from ca.scan_match_debug import run_scan_match_debug
from ca.visualization import colorize


def test_run_scan_match_debug_reports_before_after(source_and_target_files, tmp_path):
    src, tgt = source_and_target_files
    artifact_dir = tmp_path / "artifacts"

    result = run_scan_match_debug(
        scan_path=src,
        map_path=tgt,
        method="icp",
        max_correspondence_distance=0.5,
        threshold=0.05,
        artifact_dir=str(artifact_dir),
    )

    assert result["method"] == "icp"
    assert result["registration"]["fitness"] >= 0.0
    assert np.asarray(result["registration"]["final_transform"]).shape == (4, 4)
    assert result["distance_before"]["stats"]["mean"] >= result["distance_after"]["stats"]["mean"]
    assert result["distance_before"]["threshold"]["threshold"] == 0.05
    assert (artifact_dir / "scan_initial_error.ply").exists()
    assert (artifact_dir / "scan_aligned_error.ply").exists()
    assert (artifact_dir / "map_debug.ply").exists()


def test_run_scan_match_debug_crops_map(source_and_target_files):
    src, tgt = source_and_target_files

    result = run_scan_match_debug(
        scan_path=src,
        map_path=tgt,
        method="icp",
        max_correspondence_distance=0.5,
        crop_margin=0.2,
    )

    assert result["preprocess"]["map_points_used"] <= result["preprocess"]["map_points_raw"]
    assert result["preprocess"]["crop_margin"] == 0.2


@pytest.mark.parametrize("disable_rust", [False, True])
@pytest.mark.parametrize("method", ["icp", "gicp"])
@pytest.mark.parametrize("write_artifacts", [False, True])
@pytest.mark.parametrize("threshold", [None, 0.05])
def test_distances_are_shared_by_metrics_and_artifact_colors(
    source_and_target_files, tmp_path, monkeypatch, disable_rust, method, write_artifacts, threshold,
):
    if not disable_rust:
        pytest.importorskip("cloudanalyzer_core")
    monkeypatch.setenv("CA_DISABLE_RUST_CORE", "1" if disable_rust else "0")
    _rust._load.cache_clear()
    original_nn = debug.compute_nn_distance
    original_prepare = debug.prepare_nn_reference
    calls = []
    preparations = []

    def counted_nn(source, target):
        values = original_nn(source, target)
        calls.append((o3d.geometry.PointCloud(source), o3d.geometry.PointCloud(target), values.copy()))
        return values

    def counted_prepare(target):
        prepared = original_prepare(target)
        if prepared is None:
            return None
        preparations.append(1)
        frozen_target = o3d.geometry.PointCloud(target)

        class TracedReference:
            def __init__(self, native):
                self.native = native

            def distances(self, source):
                values = self.native.distances(source)
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(source)
                calls.append((cloud, frozen_target, values.copy()))
                return values

        return TracedReference(prepared)

    monkeypatch.setattr(debug, "compute_nn_distance", counted_nn)
    monkeypatch.setattr(debug, "prepare_nn_reference", counted_prepare)
    source, target = source_and_target_files
    out_dir = tmp_path / "shared-distances"
    try:
        result = run_scan_match_debug(
            source, target, method=method, max_correspondence_distance=0.5,
            threshold=threshold, artifact_dir=str(out_dir) if write_artifacts else None,
        )
        assert len(calls) == 2
        supported = not disable_rust and hasattr(_rust.core(), "PreparedReference")
        assert len(preparations) == int(supported)
        assert not np.array_equal(calls[0][2], calls[1][2])
        for index, block in enumerate(["distance_before", "distance_after"]):
            cloud, reference, observed = calls[index]
            expected = original_nn(cloud, reference)
            np.testing.assert_array_equal(observed.view(np.uint64), expected.view(np.uint64))
            assert result[block]["stats"] == summarize(expected)
            if threshold is None:
                assert "threshold" not in result[block]
            else:
                assert result[block]["threshold"] == threshold_stats(expected, threshold)
            if write_artifacts:
                name = "scan_initial_error.ply" if index == 0 else "scan_aligned_error.ply"
                saved = o3d.io.read_point_cloud(str(out_dir / name))
                colored = colorize(o3d.geometry.PointCloud(cloud), expected)
                np.testing.assert_array_equal(np.asarray(saved.points), np.asarray(colored.points))
                # PLY stores RGB as bytes; reproduce Open3D's conversion.
                expected_rgb = np.floor(np.asarray(colored.colors) * 255 + 0.5).astype(np.uint8)
                actual_rgb = np.rint(np.asarray(saved.colors) * 255).astype(np.uint8)
                np.testing.assert_array_equal(actual_rgb, expected_rgb)
        if not write_artifacts:
            assert result["artifacts"] == {} and not out_dir.exists()
    finally:
        _rust._load.cache_clear()


def test_registration_failure_does_not_publish_artifacts(
    source_and_target_files, tmp_path, monkeypatch,
):
    def fail_registration(*args, **kwargs):
        raise RuntimeError("registration failed")

    monkeypatch.setattr(debug, "register_detailed", fail_registration)
    source, target = source_and_target_files
    out_dir = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="registration failed"):
        run_scan_match_debug(source, target, artifact_dir=str(out_dir))
    assert not out_dir.exists()


@pytest.mark.parametrize("write_artifacts", [False, True])
def test_distance_array_lifetime(source_and_target_files, tmp_path, monkeypatch, write_artifacts):
    arrays = []
    original_nn = debug.compute_nn_distance

    def tracked_nn(source, target):
        values = original_nn(source, target)
        arrays.append(weakref.ref(values))
        return values

    def fixed_registration(source, target, **kwargs):
        assert (arrays[0]() is not None) == write_artifacts
        return RegistrationResult(o3d.geometry.PointCloud(source), 1., 0., np.eye(4))

    monkeypatch.setattr(debug, "compute_nn_distance", tracked_nn)
    monkeypatch.setattr(debug, "prepare_nn_reference", lambda target: None)
    monkeypatch.setattr(debug, "register_detailed", fixed_registration)
    source, target = source_and_target_files
    run_scan_match_debug(source, target, artifact_dir=str(tmp_path / "lifetime") if write_artifacts else None)
    assert len(arrays) == 2 and all(reference() is None for reference in arrays)


@pytest.mark.parametrize("write_artifacts", [False, True])
def test_reference_and_arrays_are_released_at_the_end_of_search(
    source_and_target_files, tmp_path, monkeypatch, write_artifacts,
):
    cc = pytest.importorskip("cloudanalyzer_core")
    if not hasattr(cc, "PreparedReference"):
        pytest.skip("extension does not support owned references")
    trees, arrays = [], []

    class TrackedReference:
        def __init__(self, target):
            self.native = cc.PreparedReference(np.asarray(target.points))

        def distances(self, source):
            result = self.native.distances(source)
            arrays.append(weakref.ref(result))
            return result

    def prepare(target):
        prepared = TrackedReference(target)
        trees.append(weakref.ref(prepared))
        return prepared

    def register(source, target, **kwargs):
        assert trees[0]() is not None
        assert (arrays[0]() is not None) == write_artifacts
        return RegistrationResult(o3d.geometry.PointCloud(source), 1., 0., np.eye(4))

    original_write = debug._write_colored_cloud

    def write(*args):
        assert trees[0]() is None
        return original_write(*args)

    monkeypatch.setattr(debug, "prepare_nn_reference", prepare)
    monkeypatch.setattr(debug, "register_detailed", register)
    monkeypatch.setattr(debug, "_write_colored_cloud", write)
    source, target = source_and_target_files
    run_scan_match_debug(source, target, artifact_dir=str(tmp_path / "reference") if write_artifacts else None)
    assert len(trees) == 1 and trees[0]() is None
    assert len(arrays) == 2 and all(array() is None for array in arrays)


@pytest.mark.parametrize("failure", ["construction", "after_query"])
def test_reference_failures_propagate_without_artifacts(
    source_and_target_files, tmp_path, monkeypatch, failure,
):
    class FailingReference:
        calls = 0

        def distances(self, source):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("reference query failed")
            return np.ones(len(source))

    def prepare(target):
        if failure == "construction":
            raise RuntimeError("reference construction failed")
        return FailingReference()

    monkeypatch.setattr(debug, "prepare_nn_reference", prepare)
    monkeypatch.setattr(debug, "register_detailed", lambda source, target, **kwargs:
                        RegistrationResult(o3d.geometry.PointCloud(source), 1., 0., np.eye(4)))
    source, target = source_and_target_files
    out_dir = tmp_path / "failed-reference"
    with pytest.raises(RuntimeError, match="reference .* failed"):
        run_scan_match_debug(source, target, artifact_dir=str(out_dir))
    assert not out_dir.exists()


def test_older_extension_keeps_stateless_distance_dispatch(source_and_target_files, monkeypatch):
    cc = pytest.importorskip("cloudanalyzer_core")
    calls = []

    def nearest(source, target):
        calls.append(1)
        return cc.nearest_distances(source, target)

    monkeypatch.setattr(metrics, "core", lambda: SimpleNamespace(nearest_distances=nearest))
    source, target = source_and_target_files
    result = run_scan_match_debug(source, target, method="icp")
    assert len(calls) == 2 and result["artifacts"] == {}
