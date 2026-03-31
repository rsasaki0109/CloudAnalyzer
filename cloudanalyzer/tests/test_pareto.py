"""Tests for ca.pareto module."""

from ca.pareto import (
    mark_quality_size_pareto,
    mark_quality_size_recommended,
    quality_size_pareto_results,
    recommended_quality_size_result,
)


class TestQualitySizePareto:
    def test_returns_non_dominated_results(self):
        results = [
            {"path": "a.pcd", "auc": 0.99, "compression": {"size_ratio": 0.2}},
            {"path": "b.pcd", "auc": 0.80, "compression": {"size_ratio": 0.5}},
            {"path": "c.pcd", "auc": 0.97, "compression": {"size_ratio": 0.15}},
            {"path": "d.pcd", "auc": 1.00, "compression": {"size_ratio": 0.4}},
        ]

        frontier = quality_size_pareto_results(results)

        assert [item["path"] for item in frontier] == ["c.pcd", "a.pcd", "d.pcd"]

    def test_marks_items_in_place(self):
        results = [
            {"path": "a.pcd", "auc": 0.99, "compression": {"size_ratio": 0.2}},
            {"path": "b.pcd", "auc": 0.80, "compression": {"size_ratio": 0.5}},
        ]

        mark_quality_size_pareto(results)

        assert results[0]["compression"]["pareto_optimal"] is True
        assert results[1]["compression"]["pareto_optimal"] is False

    def test_recommends_smallest_pareto_candidate(self):
        results = [
            {
                "path": "a.pcd",
                "auc": 0.99,
                "chamfer_distance": 0.01,
                "compression": {"size_ratio": 0.2},
            },
            {
                "path": "b.pcd",
                "auc": 1.00,
                "chamfer_distance": 0.02,
                "compression": {"size_ratio": 0.4},
            },
        ]

        recommended = recommended_quality_size_result(results)

        assert recommended is not None
        assert recommended["path"] == "a.pcd"

    def test_recommendation_respects_quality_gate(self):
        results = [
            {
                "path": "a.pcd",
                "auc": 0.99,
                "chamfer_distance": 0.01,
                "quality_gate": {"passed": False},
                "compression": {"size_ratio": 0.2},
            },
            {
                "path": "b.pcd",
                "auc": 0.98,
                "chamfer_distance": 0.01,
                "quality_gate": {"passed": True},
                "compression": {"size_ratio": 0.3},
            },
        ]

        recommended = recommended_quality_size_result(results)

        assert recommended is not None
        assert recommended["path"] == "b.pcd"

    def test_marks_recommended_item_in_place(self):
        results = [
            {
                "path": "a.pcd",
                "auc": 0.99,
                "chamfer_distance": 0.01,
                "compression": {"size_ratio": 0.2},
            },
            {
                "path": "b.pcd",
                "auc": 1.00,
                "chamfer_distance": 0.02,
                "compression": {"size_ratio": 0.4},
            },
        ]

        mark_quality_size_recommended(results)

        assert results[0]["compression"]["recommended"] is True
        assert results[1]["compression"]["recommended"] is False
