"""Core-promotion smoke: the map_evaluate contract is reachable from ca.core.

After Phase 21 the request/result + adopted NNThreshold strategy live in
``ca.core.map_evaluate``. ``ca.experiments.map_evaluate.{common,nn_thresholds}``
keep working as thin re-exports for back-compat; this file pins the new
public surface so a future refactor can't quietly drop it.
"""

from __future__ import annotations

import numpy as np

from ca.core.map_evaluate import (
    MapEvaluateRequest,
    MapEvaluateResult,
    NNThresholdMapEvaluateStrategy,
    evaluate_map,
    voxel_downsample,
)


def _make_request() -> MapEvaluateRequest:
    rng = np.random.default_rng(0)
    ref = rng.uniform(-3.0, 3.0, size=(200, 3))
    est = ref + rng.normal(0, 0.02, size=ref.shape)
    return MapEvaluateRequest(
        estimated_points=est,
        reference_points=ref,
        thresholds_m=(0.1, 0.05),
    )


def test_core_evaluate_map_default_strategy_returns_nn_thresholds() -> None:
    result = evaluate_map(_make_request())
    assert isinstance(result, MapEvaluateResult)
    assert result.strategy == "nn_thresholds"
    assert result.metric_family == "reference_based_nn_thresholds"
    assert result.reference_required is True


def test_core_evaluate_map_metrics_populated() -> None:
    result = evaluate_map(_make_request())
    assert "chamfer_m" in result.metrics
    assert "accuracy@0.100m" in result.metrics
    assert "completeness@0.050m" in result.metrics
    assert "fscore@0.100m" in result.metrics
    assert result.metrics["chamfer_m"] >= 0.0


def test_core_strategy_matches_experiment_reexport() -> None:
    """The experiment-side import must produce identical numbers; the
    experimental module is now just a re-export from core."""
    from ca.experiments.map_evaluate.nn_thresholds import (
        NNThresholdMapEvaluateStrategy as ExpStrategy,
    )

    req = _make_request()
    core_result = NNThresholdMapEvaluateStrategy().evaluate(req)
    exp_result = ExpStrategy().evaluate(req)
    assert core_result.metrics == exp_result.metrics
    assert core_result.metric_family == exp_result.metric_family


def test_core_voxel_downsample_helper_visible() -> None:
    """The shared voxel_downsample helper is exported from core."""
    pts = np.array([[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [10.0, 10.0, 10.0]])
    out = voxel_downsample(pts, voxel_size=0.5)
    assert out.shape == (2, 3)


def test_core_evaluate_map_rejects_missing_reference() -> None:
    """NN-threshold strategy requires reference_points."""
    rng = np.random.default_rng(1)
    bad_req = MapEvaluateRequest(
        estimated_points=rng.uniform(-1.0, 1.0, size=(50, 3)),
        reference_points=None,
    )
    import pytest

    with pytest.raises(ValueError, match="reference_points"):
        evaluate_map(bad_req)
