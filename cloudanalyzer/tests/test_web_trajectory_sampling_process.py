"""Tests for the experiment-driven workflow around web trajectory sampling."""

import numpy as np

from ca.core.web_trajectory_sampling import WebTrajectorySamplingRequest, reduce_trajectory_for_web
from ca.experiments.web_trajectory_sampling import get_strategies
from ca.experiments.web_trajectory_sampling.evaluate import run_web_trajectory_sampling_experiment


def _make_positions(count: int = 120) -> tuple[np.ndarray, np.ndarray]:
    positions = np.column_stack(
        [
            np.linspace(0.0, 12.0, count),
            np.sin(np.linspace(0.0, 4.0 * np.pi, count)),
            np.zeros(count),
        ]
    )
    timestamps = np.linspace(0.0, 12.0, count)
    return timestamps, positions


def test_stable_core_respects_budget_and_preserve_indices():
    timestamps, positions = _make_positions(180)
    result = reduce_trajectory_for_web(
        positions=positions,
        timestamps=timestamps,
        max_points=20,
        preserve_indices=(50, 120),
    )
    assert result.strategy == "turn_aware"
    assert result.reduced_points <= 20
    assert 50 in result.kept_indices.tolist()
    assert 120 in result.kept_indices.tolist()


def test_all_experimental_strategies_share_the_same_contract():
    timestamps, positions = _make_positions(150)
    request = WebTrajectorySamplingRequest(
        positions=positions,
        timestamps=timestamps,
        max_points=18,
        preserve_indices=(40, 90),
    )
    for strategy in get_strategies():
        result = strategy.reduce(request)
        assert result.strategy == strategy.name
        assert result.design == strategy.design
        assert result.reduced_points <= 18


def test_experiment_report_contains_three_variants():
    report = run_web_trajectory_sampling_experiment(repetitions=1)
    assert len(report["strategy_summaries"]) == 3
    assert report["decision"]["selected_experiment"] in {
        item["strategy"] for item in report["strategy_summaries"]
    }
