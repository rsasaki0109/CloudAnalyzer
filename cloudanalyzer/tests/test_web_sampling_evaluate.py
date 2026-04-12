"""Tests for the experiment evaluator and report generation."""

from pathlib import Path

import numpy as np
import open3d as o3d

from ca.experiments.web_sampling.evaluate import (
    DatasetCase,
    _make_point_cloud,
    _rank,
    _summarize_distances,
    benchmark_strategy_on_dataset,
    build_default_datasets,
    render_decisions_markdown,
    render_experiments_markdown,
    render_interfaces_markdown,
    run_web_sampling_experiment,
    static_source_analysis,
    summarize_strategy_results,
    write_report_docs,
)
from ca.experiments.web_sampling.object_random import RandomBudgetSamplingStrategy


def _dataset_from_points(name: str, points: np.ndarray, max_points: int) -> DatasetCase:
    return DatasetCase(
        name=name,
        description=f"{name} dataset",
        point_cloud=_make_point_cloud(points),
        max_points=max_points,
    )


class TestDatasetBuilders:
    def test_make_point_cloud_preserves_shape(self):
        points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        point_cloud = _make_point_cloud(points)

        assert isinstance(point_cloud, o3d.geometry.PointCloud)
        assert len(point_cloud.points) == 2

    def test_build_default_datasets_returns_three_cases(self):
        datasets = build_default_datasets()

        assert [dataset.name for dataset in datasets] == [
            "structured_plane",
            "clustered_room",
            "corridor_scan",
        ]
        assert [dataset.max_points for dataset in datasets] == [1800, 1600, 2000]


class TestMetricHelpers:
    def test_summarize_distances_for_empty_input(self):
        summary = _summarize_distances(np.array([]))
        assert summary == {"mean": 0.0, "p95": 0.0, "max": 0.0}

    def test_summarize_distances_for_non_empty_input(self):
        summary = _summarize_distances(np.array([1.0, 2.0, 3.0, 4.0]))
        assert summary["mean"] == 2.5
        assert summary["max"] == 4.0
        assert summary["p95"] >= 3.0

    def test_rank_supports_forward_and_reverse_order(self):
        values = {"a": 3.0, "b": 1.0, "c": 2.0}

        assert _rank(values, reverse=False) == {"b": 1, "c": 2, "a": 3}
        assert _rank(values, reverse=True) == {"a": 1, "c": 2, "b": 3}


class TestStaticAnalysis:
    def test_static_source_analysis_returns_expected_keys(self, tmp_path: Path):
        module = tmp_path / "sample_module.py"
        module.write_text(
            '"""module docs"""\n\n'
            "class Sample:\n"
            '    """class docs"""\n'
            "    def run(self, value: int) -> int:\n"
            '        """method docs"""\n'
            "        if value > 0:\n"
            "            return value\n"
            "        return 0\n",
            encoding="utf-8",
        )

        analysis = static_source_analysis(module)

        assert analysis["loc"] > 0
        assert analysis["function_count"] == 1
        assert analysis["class_count"] == 1
        assert analysis["branch_count"] == 1
        assert 0.0 <= analysis["readability_score"] <= 100.0
        assert 0.0 <= analysis["extensibility_score"] <= 100.0


class TestBenchmarkAndSummary:
    def test_benchmark_strategy_on_dataset_returns_comparable_metrics(self):
        rng = np.random.default_rng(42)
        dataset = _dataset_from_points(
            "tiny",
            rng.random((200, 3)),
            max_points=40,
        )

        metrics = benchmark_strategy_on_dataset(
            strategy=RandomBudgetSamplingStrategy(),
            dataset=dataset,
            repetitions=2,
        )

        assert metrics["dataset"] == "tiny"
        assert metrics["reduced_points"] == 40
        assert metrics["runtime_ms"] >= 0.0
        assert metrics["coverage_mean"] >= 0.0
        assert metrics["fidelity_mean"] >= 0.0
        assert metrics["retained_ratio"] == 0.2

    def test_summarize_strategy_results_ranks_multiple_dimensions(self):
        report_rows = [
            {
                "strategy": "fast",
                "design": "oop",
                "module": "fast.py",
                "runtime_ms": 1.0,
                "chamfer_mean": 0.3,
                "coverage_p95": 0.4,
                "retained_ratio": 0.2,
            },
            {
                "strategy": "fast",
                "design": "oop",
                "module": "fast.py",
                "runtime_ms": 1.2,
                "chamfer_mean": 0.2,
                "coverage_p95": 0.3,
                "retained_ratio": 0.2,
            },
            {
                "strategy": "accurate",
                "design": "functional",
                "module": "accurate.py",
                "runtime_ms": 3.5,
                "chamfer_mean": 0.1,
                "coverage_p95": 0.2,
                "retained_ratio": 0.25,
            },
        ]
        analysis_rows = {
            "fast": {"readability_score": 80.0, "extensibility_score": 55.0},
            "accurate": {"readability_score": 70.0, "extensibility_score": 65.0},
        }

        summaries = summarize_strategy_results(report_rows, analysis_rows)

        assert len(summaries) == 2
        assert {item["strategy"] for item in summaries} == {"fast", "accurate"}
        assert all("composite_rank" in item for item in summaries)
        assert summaries[0]["composite_rank"] <= summaries[1]["composite_rank"]


class TestExperimentExecutionAndDocs:
    def test_run_web_sampling_experiment_produces_full_report(self):
        rng = np.random.default_rng(8)
        dataset = _dataset_from_points("mini", rng.random((180, 3)), max_points=45)

        report = run_web_sampling_experiment(datasets=[dataset], repetitions=1)

        assert report["problem"]["name"] == "web_point_cloud_reduction"
        assert len(report["datasets"]) == 1
        assert len(report["results"]) == 3
        assert len(report["analysis"]) == 3
        assert len(report["strategy_summaries"]) == 3
        assert report["decision"]["selected_experiment"] in {
            item["strategy"] for item in report["strategy_summaries"]
        }

    def test_renderers_include_current_direct_adoption_wording(self):
        rng = np.random.default_rng(9)
        dataset = _dataset_from_points("mini", rng.random((160, 3)), max_points=30)
        report = run_web_sampling_experiment(datasets=[dataset], repetitions=1)

        experiments_md = render_experiments_markdown(report)
        decisions_md = render_decisions_markdown(report)
        interfaces_md = render_interfaces_markdown(report)

        assert "Strategy Comparison" in experiments_md
        assert "random_budget" in experiments_md
        assert (
            "adopted directly as the current core strategy" in decisions_md
            or "stabilized core form of" in decisions_md
        )
        assert (
            "adopted directly in core" in interfaces_md
            or "stabilized lineage" in interfaces_md
        )

    def test_write_report_docs_writes_all_expected_files(self, tmp_path: Path):
        rng = np.random.default_rng(10)
        dataset = _dataset_from_points("mini", rng.random((160, 3)), max_points=30)
        report = run_web_sampling_experiment(datasets=[dataset], repetitions=1)

        write_report_docs(report, tmp_path)

        experiments_path = tmp_path / "experiments.md"
        decisions_path = tmp_path / "decisions.md"
        interfaces_path = tmp_path / "interfaces.md"

        assert experiments_path.exists()
        assert decisions_path.exists()
        assert interfaces_path.exists()
        assert "Shared Inputs" in experiments_path.read_text(encoding="utf-8")
        assert "Trigger To Re-run" in decisions_path.read_text(encoding="utf-8")
        assert "Current Minimal Interface" in interfaces_path.read_text(encoding="utf-8")
