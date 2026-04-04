"""Tests for the experiment-driven ground segmentation evaluation workflow."""

from ca.core import GroundEvaluateRequest, evaluate_ground
from ca.experiments.ground_evaluate import get_ground_evaluate_strategies
from ca.experiments.ground_evaluate.common import build_default_datasets
from ca.experiments.ground_evaluate.evaluate import run_ground_evaluate_experiment


def test_stable_core_evaluates_perfect_segmentation():
    dataset = build_default_datasets()[0]

    result = evaluate_ground(dataset.request)

    assert result.strategy == "voxel_confusion"
    assert result.f1 >= dataset.expected_min_f1


def test_all_experimental_strategies_share_the_same_contract():
    request = build_default_datasets()[1].request
    assert isinstance(request, GroundEvaluateRequest)

    strategies = get_ground_evaluate_strategies()
    assert len(strategies) == 3
    for strategy in strategies:
        result = strategy.evaluate(request)
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1 <= 1.0
        assert 0.0 <= result.iou <= 1.0


def test_run_experiment_returns_complete_report():
    report = run_ground_evaluate_experiment(repetitions=1)

    assert report["problem"]["name"] == "ground_segmentation_evaluate"
    assert len(report["datasets"]) == 3
    assert len(report["strategy_summaries"]) == 3
    assert report["decision"]["stabilized_core_strategy"] == "voxel_confusion"
    assert all(
        "composite_rank" in summary for summary in report["strategy_summaries"]
    )
