"""Before/after map reports share one scoped immutable reference."""

import importlib
from types import SimpleNamespace
import weakref

import numpy as np
import open3d as o3d
import pytest

from ca import _rust
import ca.loop_closure_report as loop
import ca.metrics as metrics

evaluation = importlib.import_module("ca.evaluate")


@pytest.fixture
def map_inputs(tmp_path, identical_pcd, shifted_pcd):
    paths = [tmp_path / name for name in ["before.ply", "after.ply", "reference.ply"]]
    for path, cloud in zip(paths, [shifted_pcd, identical_pcd, identical_pcd], strict=True):
        assert o3d.io.write_point_cloud(str(path), cloud)
    return tuple(str(path) for path in paths)


@pytest.mark.parametrize("backend", ["native", "legacy", "open3d"])
@pytest.mark.parametrize("thresholds", [None, [.5, .01, .1], [.1], []])
def test_full_evaluations_and_bidirectional_f64_parity(map_inputs, monkeypatch, backend, thresholds):
    if backend != "open3d":
        cc = pytest.importorskip("cloudanalyzer_core")
        if backend == "native" and not hasattr(cc, "PreparedReference"):
            pytest.skip("extension does not support owned references")
        if backend == "legacy":
            monkeypatch.setattr(metrics, "core", lambda: SimpleNamespace(nearest_distances=cc.nearest_distances))
    monkeypatch.setenv("CA_DISABLE_RUST_CORE", "1" if backend == "open3d" else "0")
    _rust._load.cache_clear()
    before, after, reference = map_inputs
    load, nn, prepare = loop.load_point_cloud, evaluation.compute_nn_distance, loop.prepare_nn_reference
    expected = [evaluation.evaluate(path, reference, thresholds) for path in [before, after]]
    expected_arrays = []
    target = load(reference)
    for path in [before, after]:
        source = load(path)
        expected_arrays.extend([nn(source, target), nn(target, source)])
    arrays, loads, preparations, reports = [], [], [], []

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
        preparations.append(1)

        class TracedReference:
            def distances(self, source):
                values = native.distances(source)
                arrays.append(values.copy())
                return values

        return TracedReference()

    def tracked_evaluate(*args, **kwargs):
        result = evaluation._evaluate_clouds(*args, **kwargs)
        reports.append(result)
        return result

    monkeypatch.setattr(loop, "load_point_cloud", tracked_load)
    monkeypatch.setattr(loop, "prepare_nn_reference", tracked_prepare)
    monkeypatch.setattr(loop, "_evaluate_clouds", tracked_evaluate)
    monkeypatch.setattr(evaluation, "compute_nn_distance", tracked_nn)
    try:
        if thresholds == []:
            with pytest.raises(ValueError, match=r"max\(\).*empty"):
                loop.build_loop_closure_report(before_map=before, after_map=after, reference_map=reference, thresholds=thresholds)
        else:
            result = loop.build_loop_closure_report(before_map=before, after_map=after, reference_map=reference, thresholds=thresholds)
            for key, wanted in zip(["before", "after"], expected, strict=True):
                for metric in ["auc", "chamfer_distance", "hausdorff_distance"]:
                    assert result["map"][key][metric] == wanted[metric]
                assert result["map"][key]["best_f1"] == max(wanted["f1_scores"], key=lambda score: score["f1"])
            assert result["quality_gate"] is None
        assert reports == expected
        assert loads == [before, reference, after]
        assert len(preparations) == int(backend == "native")
        assert len(arrays) == 4
        for observed, wanted in zip(arrays, expected_arrays, strict=True):
            np.testing.assert_array_equal(observed.view(np.uint64), wanted.view(np.uint64))
    finally:
        _rust._load.cache_clear()


def test_reference_snapshot_is_reloaded_on_next_report(map_inputs, monkeypatch, shifted_pcd):
    before, after, reference = map_inputs
    original = loop._evaluate_clouds
    expected = [evaluation.evaluate(path, reference) for path in [before, after]]
    reports = []

    def update_reference(*args, **kwargs):
        result = original(*args, **kwargs)
        reports.append(result)
        if len(reports) == 1:
            assert o3d.io.write_point_cloud(reference, shifted_pcd)
        return result

    monkeypatch.setattr(loop, "_evaluate_clouds", update_reference)
    loop.build_loop_closure_report(before_map=before, after_map=after, reference_map=reference)
    assert reports == expected
    next_expected = [evaluation.evaluate(path, reference) for path in [before, after]]
    reports.clear()
    loop.build_loop_closure_report(before_map=before, after_map=after, reference_map=reference)
    assert reports == next_expected and reports != expected


@pytest.mark.parametrize("failure", [None, "construction", "first_query", "second_query", "after_load"])
def test_reference_lifetime_and_failure_cleanup(map_inputs, monkeypatch, failure):
    before, after, reference = map_inputs
    trees, clouds, arrays = [], [], []
    original_load, nn = loop.load_point_cloud, evaluation.compute_nn_distance

    class TrackedReference:
        def __init__(self, target):
            self.target = target
            self.calls = 0

        def distances(self, source):
            self.calls += 1
            if failure == ("first_query" if self.calls == 1 else "second_query"):
                raise RuntimeError("reference query failed")
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(source)
            values = nn(cloud, self.target)
            arrays.append(weakref.ref(values))
            return values

    def prepare(target):
        if failure == "construction":
            raise RuntimeError("reference construction failed")
        tree = TrackedReference(target)
        trees.append(weakref.ref(tree))
        return tree

    def load(path):
        if path == after:
            assert clouds[0]() is None
            assert trees[0]() is not None
            if failure == "after_load":
                raise FileNotFoundError("after map missing")
        cloud = original_load(path)
        if path != reference:
            clouds.append(weakref.ref(cloud))
        return cloud

    def optional_section(**kwargs):
        assert all(tree() is None for tree in trees)
        assert all(cloud() is None for cloud in clouds)
        assert all(array() is None for array in arrays)
        return {"summary": {"ok": True}}

    monkeypatch.setattr(loop, "prepare_nn_reference", prepare)
    monkeypatch.setattr(loop, "load_point_cloud", load)
    monkeypatch.setattr(loop, "validate_posegraph_session", optional_section)
    if failure is not None:
        error = FileNotFoundError if failure == "after_load" else RuntimeError
        with pytest.raises(error):
            loop.build_loop_closure_report(before_map=before, after_map=after, reference_map=reference, before_g2o="unused")
    else:
        result = loop.build_loop_closure_report(before_map=before, after_map=after, reference_map=reference, before_g2o="unused")
        assert result["posegraph_session"]["before"]["summary"]["ok"]
    assert all(tree() is None for tree in trees)
    assert all(cloud() is None for cloud in clouds)
    assert all(array() is None for array in arrays)


def test_missing_before_map_still_fails_before_reference_load(monkeypatch):
    calls = []

    def missing(path):
        calls.append(path)
        raise FileNotFoundError(path)

    monkeypatch.setattr(loop, "load_point_cloud", missing)
    with pytest.raises(FileNotFoundError):
        loop.build_loop_closure_report(before_map="before", after_map="after", reference_map="reference")
    assert calls == ["before"]
