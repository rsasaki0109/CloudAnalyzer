"""Tests for the experiment-driven web sampling workflow."""

from pathlib import Path

import numpy as np
import open3d as o3d

from ca.core.web_sampling import reduce_point_cloud_for_web
from ca.experiments.web_sampling import get_web_sampling_strategies
from ca.experiments.web_sampling.evaluate import (
    DatasetCase,
    run_web_sampling_experiment,
    write_report_docs,
)


def _make_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(float))
    return point_cloud


class TestWebSamplingProcess:
    def test_core_reducer_respects_budget(self):
        rng = np.random.default_rng(0)
        point_cloud = _make_point_cloud(rng.random((500, 3)))

        result = reduce_point_cloud_for_web(point_cloud, max_points=120, label="test")

        assert result.reduced_points <= 120
        assert result.original_points == 500
        assert result.strategy == "random_budget"

    def test_all_experimental_strategies_respect_budget(self):
        from ca.core.web_sampling import WebSampleRequest

        rng = np.random.default_rng(2)
        point_cloud = _make_point_cloud(rng.random((400, 3)))

        for strategy in get_web_sampling_strategies():
            result = strategy.reduce(
                WebSampleRequest(point_cloud=point_cloud, max_points=90, label="test")
            )
            assert result.reduced_points <= 90
            assert result.original_points == 400
            assert result.strategy == strategy.name

    def test_experiment_report_and_docs_generation(self, tmp_path: Path):
        rng = np.random.default_rng(3)
        dataset = DatasetCase(
            name="tiny_cloud",
            description="Small deterministic dataset for report generation.",
            point_cloud=_make_point_cloud(rng.random((320, 3))),
            max_points=80,
        )

        report = run_web_sampling_experiment(datasets=[dataset], repetitions=1)

        assert report["problem"]["name"] == "web_point_cloud_reduction"
        assert len(report["strategy_summaries"]) == 3
        assert report["decision"]["stabilized_core_strategy"] == "random_budget"

        write_report_docs(report, tmp_path)

        assert (tmp_path / "experiments.md").exists()
        assert (tmp_path / "decisions.md").exists()
        assert (tmp_path / "interfaces.md").exists()
        assert "Strategy Comparison" in (tmp_path / "experiments.md").read_text(encoding="utf-8")
