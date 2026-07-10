"""Core-promotion smoke: the map_evaluate contract is reachable from ca.core.

After Phase 21 the request/result + adopted NNThreshold strategy live in
``ca.core.map_evaluate``. ``ca.experiments.map_evaluate.{common,nn_thresholds}``
keep working as thin re-exports for back-compat; this file pins the new
public surface so a future refactor can't quietly drop it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ca.core.map_evaluate import (
    MapEvaluateRequest,
    MapEvaluateResult,
    NNThresholdMapEvaluateStrategy,
    evaluate_map,
    compute_voxel_wasserstein_metrics,
    voxel_downsample,
    wasserstein_distance_gaussian,
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
    with pytest.raises(ValueError, match="reference_points"):
        evaluate_map(bad_req)


def test_gaussian_wasserstein_matches_closed_form_for_diagonal_covariances() -> None:
    mu1 = np.array([0.0, 0.0, 0.0])
    mu2 = np.array([1.0, 2.0, 2.0])
    sigma1 = np.diag([1.0, 4.0, 9.0])
    sigma2 = np.diag([4.0, 9.0, 16.0])
    # W2^2 = ||mu1-mu2||^2 + sum((sqrt(var1)-sqrt(var2))^2).
    assert wasserstein_distance_gaussian(mu1, sigma1, mu2, sigma2) == pytest.approx(
        np.sqrt(12.0)
    )


def test_gaussian_wasserstein_is_symmetric_for_correlated_covariances() -> None:
    sigma1 = np.array([[2.0, 0.7, 0.1], [0.7, 1.0, 0.2], [0.1, 0.2, 0.5]])
    sigma2 = np.array([[1.5, -0.2, 0.3], [-0.2, 0.8, 0.1], [0.3, 0.1, 1.2]])
    forward = wasserstein_distance_gaussian(np.zeros(3), sigma1, np.ones(3), sigma2)
    reverse = wasserstein_distance_gaussian(np.ones(3), sigma2, np.zeros(3), sigma1)
    assert forward == pytest.approx(reverse, rel=1e-10)


def test_awd_scs_identical_dense_neighbor_voxels_are_zero() -> None:
    rng = np.random.default_rng(4)
    first = rng.uniform([0.05, 0.05, 0.05], [0.45, 0.45, 0.45], size=(120, 3))
    second = rng.uniform([0.55, 0.05, 0.05], [0.95, 0.45, 0.45], size=(120, 3))
    points = np.vstack([first, second])
    metrics = compute_voxel_wasserstein_metrics(
        points, points.copy(), voxel_size=0.5, neighbor_radius=1
    )
    assert metrics["awd_m"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["scs"] == pytest.approx(0.0)
    assert metrics["n_awd_voxels"] == 2
    assert metrics["n_scs_voxels"] == 2


def test_awd_scs_sparse_or_empty_inputs_are_explicitly_unavailable() -> None:
    sparse = np.array([[0.0, 0.0, 0.0]])
    empty = np.empty((0, 3))
    for estimated, reference in ((sparse, sparse), (empty, empty)):
        metrics = compute_voxel_wasserstein_metrics(
            estimated, reference, voxel_size=1.0
        )
        assert np.isnan(metrics["awd_m"])
        assert np.isnan(metrics["scs"])
        assert metrics["n_awd_voxels"] == 0


def test_awd_rejects_non_positive_voxel_size() -> None:
    with pytest.raises(ValueError, match="voxel_size"):
        compute_voxel_wasserstein_metrics(
            np.empty((0, 3)), np.empty((0, 3)), voxel_size=0.0
        )
