"""Tests for the experiment-driven workflow around web progressive loading."""

import numpy as np

from ca.core.web_progressive_loading import (
    WebProgressiveLoadingRequest,
    plan_progressive_loading_for_web,
)
from ca.experiments.web_progressive_loading import get_strategies
from ca.experiments.web_progressive_loading.evaluate import (
    run_web_progressive_loading_experiment,
)


def _make_positions(count: int = 180) -> np.ndarray:
    return np.column_stack(
        [
            np.linspace(0.0, 18.0, count),
            np.sin(np.linspace(0.0, 4.0 * np.pi, count)),
            np.zeros(count),
        ]
    )


def test_stable_core_returns_initial_payload_plus_chunks():
    result = plan_progressive_loading_for_web(
        positions=_make_positions(180),
        initial_points=24,
        chunk_points=32,
    )
    assert result.strategy == "distance_shells"
    assert result.initial_points == 24
    assert result.total_displayed_points == 180


def test_all_experimental_strategies_share_the_same_contract():
    request = WebProgressiveLoadingRequest(
        positions=_make_positions(150),
        initial_points=20,
        chunk_points=28,
    )
    for strategy in get_strategies():
        result = strategy.plan(request)
        assert result.strategy == strategy.name
        assert result.design == strategy.design
        assert result.initial_points <= 20


def test_experiment_report_contains_three_variants():
    report = run_web_progressive_loading_experiment(repetitions=1)
    assert len(report["strategy_summaries"]) == 3
    assert report["decision"]["selected_experiment"] in {
        item["strategy"] for item in report["strategy_summaries"]
    }
