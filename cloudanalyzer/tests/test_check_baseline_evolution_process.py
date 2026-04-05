"""Tests for the experiment-driven baseline evolution workflow."""

from ca.core import BaselineEvolutionRequest, decide_baseline_evolution, summarize_baseline_evolution
from ca.experiments.check_baseline_evolution import (
    get_check_baseline_evolution_strategies,
)
from ca.experiments.check_baseline_evolution.common import build_default_datasets
from ca.experiments.check_baseline_evolution.evaluate import (
    run_check_baseline_evolution_experiment,
)


def test_stable_core_promotes_after_stable_window():
    dataset = build_default_datasets()[0]

    result = decide_baseline_evolution(dataset.request)

    assert result.strategy == "stability_window"
    assert result.decision == "promote"


def test_all_experimental_strategies_share_the_same_contract():
    request = build_default_datasets()[1].request
    assert isinstance(request, BaselineEvolutionRequest)

    for strategy in get_check_baseline_evolution_strategies():
        result = strategy.decide(request)
        assert result.strategy == strategy.name
        assert result.design == strategy.design
        assert result.decision in {"promote", "keep", "reject"}


def test_summary_builder_parses_check_results_into_keep_decision():
    history_result = {
        "config_path": "/tmp/history-1.json",
        "project": "baseline-evolution-test",
        "summary": {
            "passed": True,
            "failed_check_ids": [],
        },
        "checks": [
            {
                "id": "mapping-postprocess",
                "kind": "artifact",
                "passed": True,
                "summary": {
                    "auc": 0.958,
                    "chamfer_distance": 0.018,
                },
                "result": {
                    "quality_gate": {
                        "min_auc": 0.95,
                        "max_chamfer": 0.02,
                    }
                },
            },
            {
                "id": "localization-run",
                "kind": "trajectory",
                "passed": True,
                "summary": {
                    "ate_rmse": 0.44,
                    "rpe_rmse": 0.17,
                    "drift_endpoint": 0.15,
                    "coverage_ratio": 0.92,
                },
                "result": {
                    "quality_gate": {
                        "max_ate": 0.5,
                        "max_rpe": 0.2,
                        "max_drift": 0.2,
                        "min_coverage": 0.9,
                    }
                },
            },
        ],
    }
    candidate_result = {
        "config_path": "/tmp/candidate.json",
        "project": "baseline-evolution-test",
        "summary": {
            "passed": True,
            "failed_check_ids": [],
            "triage": {
                "items": [],
            },
        },
        "checks": [
            {
                "id": "mapping-postprocess",
                "kind": "artifact",
                "passed": True,
                "summary": {
                    "auc": 0.975,
                    "chamfer_distance": 0.014,
                },
                "result": {
                    "quality_gate": {
                        "min_auc": 0.95,
                        "max_chamfer": 0.02,
                    }
                },
            },
            {
                "id": "localization-run",
                "kind": "trajectory",
                "passed": True,
                "summary": {
                    "ate_rmse": 0.34,
                    "rpe_rmse": 0.12,
                    "drift_endpoint": 0.11,
                    "coverage_ratio": 0.95,
                },
                "result": {
                    "quality_gate": {
                        "max_ate": 0.5,
                        "max_rpe": 0.2,
                        "max_drift": 0.2,
                        "min_coverage": 0.9,
                    }
                },
            },
        ],
    }

    summary = summarize_baseline_evolution(candidate_result, [history_result])

    assert summary["candidate_label"] == "candidate"
    assert summary["history_labels"] == ["history-1"]
    assert summary["strategy"] == "stability_window"
    assert summary["decision"] == "keep"


def test_experiment_report_contains_three_variants():
    report = run_check_baseline_evolution_experiment(repetitions=1)

    assert report["problem"]["name"] == "check_baseline_evolution"
    assert len(report["strategy_summaries"]) == 3
    assert report["decision"]["selected_experiment"] in {
        item["strategy"] for item in report["strategy_summaries"]
    }
