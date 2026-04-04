"""Tests for the experiment-driven failed-check triage workflow."""

from ca.core import CheckTriageRequest, rank_failed_checks
from ca.experiments.check_triage import get_check_triage_strategies
from ca.experiments.check_triage.common import build_default_datasets
from ca.experiments.check_triage.evaluate import run_check_triage_experiment


def test_stable_core_ranks_most_severe_failure_first():
    dataset = build_default_datasets()[0]

    result = rank_failed_checks(dataset.request)

    assert result.strategy == "severity_weighted"
    assert result.ranked_items[0].check_id == "integrated-run"


def test_all_experimental_strategies_share_the_same_contract():
    request = build_default_datasets()[1].request
    assert isinstance(request, CheckTriageRequest)

    for strategy in get_check_triage_strategies():
        result = strategy.rank(request)
        assert result.strategy == strategy.name
        assert result.design == strategy.design
        assert len(result.ranked_items) == len(request.failed_items)


def test_experiment_report_contains_three_variants():
    report = run_check_triage_experiment(repetitions=1)

    assert report["problem"]["name"] == "check_regression_triage"
    assert len(report["strategy_summaries"]) == 3
    assert report["decision"]["selected_experiment"] in {
        item["strategy"] for item in report["strategy_summaries"]
    }
