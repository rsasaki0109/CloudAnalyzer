"""Tests for trajectory simplification evaluator and reporting."""

from pathlib import Path

import numpy as np

from ca.experiments.web_trajectory_sampling.evaluate import (
    _rank,
    _summary,
    TrajectoryDatasetCase,
    benchmark_strategy_on_dataset,
    build_default_datasets,
    render_experiment_section,
    run_web_trajectory_sampling_experiment,
    static_source_analysis,
    summarize_strategy_results,
)
from ca.experiments.web_trajectory_sampling.turn_aware import TurnAwareStrategy


class TestHelpers:
    def test_summary_handles_empty_and_non_empty_arrays(self):
        assert _summary(np.array([])) == {"mean": 0.0, "p95": 0.0, "max": 0.0}
        summary = _summary(np.array([1.0, 2.0, 3.0]))
        assert summary["mean"] == 2.0
        assert summary["max"] == 3.0

    def test_rank_forward_and_reverse(self):
        values = {"a": 2.0, "b": 1.0, "c": 3.0}
        assert _rank(values) == {"b": 1, "a": 2, "c": 3}
        assert _rank(values, reverse=True) == {"c": 1, "a": 2, "b": 3}


class TestDatasetsAndAnalysis:
    def test_build_default_datasets(self):
        datasets = build_default_datasets()
        assert [dataset.name for dataset in datasets] == [
            "straight_corridor",
            "right_angle_turn",
            "switchback",
        ]
        assert all(dataset.positions.shape[0] > dataset.max_points for dataset in datasets)

    def test_static_source_analysis_returns_metrics(self, tmp_path: Path):
        path = tmp_path / "sample.py"
        path.write_text(
            '"""doc"""\n\n'
            "class Example:\n"
            '    """cls"""\n'
            "    def run(self, value: int) -> int:\n"
            '        """fn"""\n'
            "        if value > 0:\n"
            "            return value\n"
            "        return 0\n",
            encoding="utf-8",
        )
        analysis = static_source_analysis(path)
        assert analysis["function_count"] == 1
        assert analysis["class_count"] == 1
        assert analysis["branch_count"] == 1


class TestBenchmarkAndReport:
    def test_benchmark_returns_quality_metrics(self):
        positions = np.column_stack([np.linspace(0.0, 9.0, 120), np.zeros(120), np.zeros(120)])
        dataset = TrajectoryDatasetCase(
            name="line",
            description="line",
            timestamps=np.linspace(0.0, 12.0, 120),
            positions=positions,
            max_points=20,
            preserve_indices=(60,),
        )
        metrics = benchmark_strategy_on_dataset(TurnAwareStrategy(), dataset, repetitions=1)
        assert metrics["reduced_points"] <= 20
        assert metrics["mean_error"] >= 0.0
        assert metrics["preserve_ratio"] == 1.0

    def test_summarize_strategy_results(self):
        rows = [
            {
                "strategy": "a",
                "design": "functional",
                "module": "a.py",
                "runtime_ms": 1.0,
                "mean_error": 0.2,
                "p95_error": 0.3,
                "path_length_delta": 0.1,
                "preserve_ratio": 1.0,
            },
            {
                "strategy": "b",
                "design": "oop",
                "module": "b.py",
                "runtime_ms": 2.0,
                "mean_error": 0.1,
                "p95_error": 0.2,
                "path_length_delta": 0.05,
                "preserve_ratio": 0.9,
            },
        ]
        analysis = {
            "a": {"readability_score": 80.0, "extensibility_score": 70.0},
            "b": {"readability_score": 75.0, "extensibility_score": 60.0},
        }
        summaries = summarize_strategy_results(rows, analysis)
        assert len(summaries) == 2
        assert all("composite_rank" in item for item in summaries)

    def test_run_experiment_and_render_section(self):
        report = run_web_trajectory_sampling_experiment(repetitions=1)
        assert report["problem"]["name"] == "web_trajectory_sampling"
        assert len(report["strategy_summaries"]) == 3
        assert report["decision"]["stabilized_core_strategy"] == "turn_aware"

        section = render_experiment_section(report)
        assert "web_trajectory_sampling" in section
        assert "Strategy Comparison" in section
        assert "turn_aware" in section
