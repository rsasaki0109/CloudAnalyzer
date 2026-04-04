"""Tests for the experiment-driven workflow around `ca init-check`."""

from pathlib import Path

from ca.core import CheckScaffoldRequest, render_check_scaffold
from ca.experiments.check_scaffolding import get_check_scaffolding_strategies
from ca.experiments.check_scaffolding.evaluate import (
    build_default_profile_cases,
    run_check_scaffolding_experiment,
    write_report_docs,
)


def test_stable_core_renders_supported_profile():
    result = render_check_scaffold(profile="integrated")

    assert result.strategy == "static_profiles"
    assert "integrated-run" in result.yaml_text
    assert "summary_output_json" in result.yaml_text


def test_all_experimental_strategies_share_the_same_contract():
    for strategy in get_check_scaffolding_strategies():
        result = strategy.render(CheckScaffoldRequest(profile="mapping"))
        assert result.strategy == strategy.name
        assert result.design == strategy.design
        assert "mapping-postprocess" in result.yaml_text


def test_experiment_report_and_docs_generation(tmp_path: Path):
    report = run_check_scaffolding_experiment(
        profiles=build_default_profile_cases(),
        repetitions=1,
    )

    assert report["problem"]["name"] == "check_scaffolding"
    assert len(report["strategy_summaries"]) == 3
    assert report["decision"]["stabilized_core_strategy"] == "static_profiles"

    write_report_docs(report, tmp_path)

    assert (tmp_path / "experiments.md").exists()
    assert (tmp_path / "decisions.md").exists()
    assert (tmp_path / "interfaces.md").exists()
    assert "Strategy Comparison" in (tmp_path / "experiments.md").read_text(encoding="utf-8")
