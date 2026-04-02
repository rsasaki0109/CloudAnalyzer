"""Tests for progressive-loading evaluator and report generation."""

from pathlib import Path

import numpy as np

from ca.experiments.web_progressive_loading.evaluate import (
    ProgressiveDatasetCase,
    _rank,
    benchmark_strategy_on_dataset,
    build_default_datasets,
    render_decision_section,
    render_experiment_section,
    render_interface_section,
    run_web_progressive_loading_experiment,
    static_source_analysis,
    summarize_strategy_results,
)
from ca.experiments.web_progressive_loading.grid_tiles import GridTilesStrategy


class TestDatasetsAndAnalysis:
    def test_build_default_datasets(self):
        datasets = build_default_datasets()
        assert [dataset.name for dataset in datasets] == [
            "corridor_run",
            "clustered_yard",
            "multi_level_room",
        ]
        assert all(dataset.positions.shape[0] > dataset.initial_points for dataset in datasets)

    def test_static_source_analysis_returns_metrics(self, tmp_path: Path):
        path = tmp_path / "sample.py"
        path.write_text(
            '"""doc"""\n\n'
            "def run(value: int) -> int:\n"
            '    """fn"""\n'
            "    if value > 0:\n"
            "        return value\n"
            "    return 0\n",
            encoding="utf-8",
        )
        analysis = static_source_analysis(path)
        assert analysis["function_count"] == 1
        assert analysis["branch_count"] == 1


class TestHelpers:
    def test_rank_forward_and_reverse(self):
        values = {"a": 2.0, "b": 1.0, "c": 3.0}
        assert _rank(values) == {"b": 1, "a": 2, "c": 3}
        assert _rank(values, reverse=True) == {"c": 1, "a": 2, "b": 3}


class TestBenchmarkAndReport:
    def test_benchmark_returns_progressive_metrics(self):
        positions = np.column_stack(
            [np.linspace(0.0, 9.0, 120), np.zeros(120), np.zeros(120)]
        )
        dataset = ProgressiveDatasetCase(
            name="line",
            description="line",
            positions=positions,
            initial_points=20,
            chunk_points=25,
        )
        metrics = benchmark_strategy_on_dataset(GridTilesStrategy(), dataset, repetitions=1)
        assert metrics["chunk_count"] >= 0
        assert metrics["initial_coverage_p95"] >= 0.0
        assert metrics["progressive_coverage_auc"] >= 0.0

    def test_summarize_strategy_results(self):
        rows = [
            {
                "strategy": "a",
                "design": "grid",
                "module": "a.py",
                "runtime_ms": 1.0,
                "initial_coverage_p95": 0.1,
                "progressive_coverage_auc": 0.08,
                "chunk_size_std": 2.0,
                "initial_ratio": 0.2,
            },
            {
                "strategy": "b",
                "design": "functional",
                "module": "b.py",
                "runtime_ms": 2.0,
                "initial_coverage_p95": 0.2,
                "progressive_coverage_auc": 0.12,
                "chunk_size_std": 1.0,
                "initial_ratio": 0.2,
            },
        ]
        analysis = {
            "a": {"readability_score": 80.0, "extensibility_score": 70.0},
            "b": {"readability_score": 75.0, "extensibility_score": 60.0},
        }
        summaries = summarize_strategy_results(rows, analysis)
        assert len(summaries) == 2
        assert all("composite_rank" in item for item in summaries)

    def test_run_experiment_and_render_sections(self):
        report = run_web_progressive_loading_experiment(repetitions=1)
        assert report["problem"]["name"] == "web_progressive_loading"
        assert len(report["strategy_summaries"]) == 3
        assert report["decision"]["stabilized_core_strategy"] == "distance_shells"

        assert "web_progressive_loading" in render_experiment_section(report)
        assert "distance_shells" in render_decision_section(report)
        assert "WebProgressiveLoadingRequest" in render_interface_section(report)
