"""Scoped batch reference reuse and exact evaluation compatibility."""

import importlib
import weakref
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import pytest

from ca import _rust
import ca.batch as batch
import ca.metrics as metrics

evaluation = importlib.import_module("ca.evaluate")


@pytest.fixture
def batch_inputs(tmp_path, identical_pcd, shifted_pcd):
    directory = tmp_path / "candidates"
    directory.mkdir()
    reference = tmp_path / "reference.ply"
    assert o3d.io.write_point_cloud(str(reference), identical_pcd)
    for name, cloud in [("a", identical_pcd), ("b", shifted_pcd)]:
        assert o3d.io.write_point_cloud(str(directory / (name + ".ply")), cloud)
    return directory, reference


@pytest.mark.parametrize("backend", ["native", "legacy", "open3d"])
@pytest.mark.parametrize("thresholds", [None, [1., .01, .1], [.1], []])
def test_exact_bidirectional_results_and_one_reference_load(
    batch_inputs, monkeypatch, backend, thresholds,
):
    if backend != "open3d":
        cc = pytest.importorskip("cloudanalyzer_core")
        if backend == "native" and not hasattr(cc, "PreparedReference"):
            pytest.skip("extension does not support owned references")
        if backend == "legacy":
            monkeypatch.setattr(metrics, "core", lambda: SimpleNamespace(nearest_distances=cc.nearest_distances))
    monkeypatch.setenv("CA_DISABLE_RUST_CORE", "1" if backend == "open3d" else "0")
    _rust._load.cache_clear()
    directory, reference = batch_inputs
    files = sorted(directory.glob("*.ply"))
    load = batch.load_point_cloud
    nn = evaluation.compute_nn_distance
    prepare = batch.prepare_nn_reference
    expected, expected_arrays = [], []
    for file in files:
        expected.append(evaluation.evaluate(str(file), str(reference), thresholds))
        source, target = load(str(file)), load(str(reference))
        expected_arrays.extend([nn(source, target), nn(target, source)])
    loads, trees, arrays, reports = [], [], [], []

    def tracked_load(path):
        loads.append(path)
        return load(path)

    def tracked_nn(source, target):
        values = nn(source, target)
        arrays.append(values.copy())
        return values

    def tracked_prepare(target):
        native = prepare(target)
        if native is None:
            return None
        trees.append(1)

        class TracedReference:
            def distances(self, source):
                values = native.distances(source)
                arrays.append(values.copy())
                return values

        return TracedReference()

    def tracked_evaluate(*args, **kwargs):
        report = evaluation._evaluate_clouds(*args, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr(batch, "load_point_cloud", tracked_load)
    monkeypatch.setattr(batch, "prepare_nn_reference", tracked_prepare)
    monkeypatch.setattr(batch, "_evaluate_clouds", tracked_evaluate)
    monkeypatch.setattr(evaluation, "compute_nn_distance", tracked_nn)
    try:
        result = batch.batch_evaluate(str(directory), str(reference), thresholds=thresholds)
        assert reports == expected
        assert len(result) == (0 if thresholds == [] else 2)
        assert loads == [str(reference)] + [str(file) for file in files]
        assert len(trees) == int(backend == "native")
        assert len(arrays) == 4
        for observed, wanted in zip(arrays, expected_arrays, strict=True):
            np.testing.assert_array_equal(observed.view(np.uint64), wanted.view(np.uint64))
    finally:
        _rust._load.cache_clear()


def test_single_file_avoids_extra_tree(batch_inputs, monkeypatch):
    directory, reference = batch_inputs
    (directory / "b.ply").unlink()

    def unnecessary_tree(target):
        pytest.fail("single-file batch constructed an extra tree")

    monkeypatch.setattr(batch, "prepare_nn_reference", unnecessary_tree)
    result = batch.batch_evaluate(str(directory), str(reference))
    assert len(result) == 1 and result[0]["auc"] == 1.


def test_empty_batch_does_not_load_missing_reference(tmp_path):
    assert batch.batch_evaluate(str(tmp_path), str(tmp_path / "missing.ply")) == []


def test_invalid_candidate_does_not_stop_remaining_files(batch_inputs, caplog):
    directory, reference = batch_inputs
    (directory / "ab.csv").write_text("invalid\n", encoding="utf-8")
    result = batch.batch_evaluate(str(directory), str(reference))
    assert [item["path"] for item in result] == [str(directory / "a.ply"), str(directory / "b.ply")]
    assert "Skipped" in caplog.text


def test_reference_is_a_batch_snapshot_and_reloaded_next_call(batch_inputs, monkeypatch, shifted_pcd):
    directory, reference = batch_inputs
    files = sorted(directory.glob("*.ply"))
    expected = [evaluation.evaluate(str(file), str(reference)) for file in files]
    reports = []
    original = batch._evaluate_clouds

    def update_reference(*args, **kwargs):
        report = original(*args, **kwargs)
        reports.append(report)
        if len(reports) == 1:
            assert o3d.io.write_point_cloud(str(reference), shifted_pcd)
        return report

    monkeypatch.setattr(batch, "_evaluate_clouds", update_reference)
    batch.batch_evaluate(str(directory), str(reference))
    assert reports == expected
    expected_next = [evaluation.evaluate(str(file), str(reference)) for file in files]
    reports.clear()
    batch.batch_evaluate(str(directory), str(reference))
    assert reports == expected_next and reports != expected


@pytest.mark.parametrize("failure", [None, "construction", "query", "recoverable_query"])
def test_tree_and_sources_are_released_on_success_or_failure(batch_inputs, monkeypatch, failure):
    directory, reference = batch_inputs
    trees, sources, arrays = [], [], []
    original_load = batch.load_point_cloud
    original_nn = evaluation.compute_nn_distance

    class TrackedReference:
        def __init__(self, target):
            self.target = target
            self.calls = 0

        def distances(self, source):
            self.calls += 1
            if self.calls == 1 and failure in {"query", "recoverable_query"}:
                error = ValueError if failure == "recoverable_query" else RuntimeError
                raise error("reference query failed")
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(source)
            values = original_nn(cloud, self.target)
            arrays.append(weakref.ref(values))
            return values

    def prepare(target):
        if failure == "construction":
            raise RuntimeError("reference construction failed")
        tree = TrackedReference(target)
        trees.append(weakref.ref(tree))
        return tree

    def load(path):
        # No previous candidate survives the start of the next file's loading.
        assert all(source() is None for source in sources)
        cloud = original_load(path)
        if path != str(reference):
            sources.append(weakref.ref(cloud))
        return cloud

    def recommend(*args, **kwargs):
        assert all(tree() is None for tree in trees)
        assert all(source() is None for source in sources)
        assert all(array() is None for array in arrays)

    monkeypatch.setattr(batch, "prepare_nn_reference", prepare)
    monkeypatch.setattr(batch, "load_point_cloud", load)
    monkeypatch.setattr(batch, "mark_quality_size_recommended", recommend)
    if failure in {"construction", "query"}:
        with pytest.raises(RuntimeError, match="reference .* failed"):
            batch.batch_evaluate(str(directory), str(reference))
    else:
        result = batch.batch_evaluate(str(directory), str(reference))
        assert len(result) == (1 if failure else 2)
    assert all(tree() is None for tree in trees)
    assert all(source() is None for source in sources)
    assert all(array() is None for array in arrays)
