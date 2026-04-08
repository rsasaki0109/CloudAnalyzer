"""Tests for ca.report module."""

import json

from ca.report import (
    make_batch_html,
    make_batch_markdown,
    make_batch_summary,
    make_ground_html,
    make_ground_markdown,
    make_json,
    make_markdown,
    save_batch_report,
    save_ground_report,
    save_json,
)


SAMPLE_STATS = {"mean": 0.05, "median": 0.04, "max": 0.12, "min": 0.001, "std": 0.02}
SAMPLE_BATCH_RESULTS = [
    {
        "path": "a.pcd",
        "num_points": 100,
        "reference_path": "ref.pcd",
        "reference_points": 100,
        "chamfer_distance": 0.001,
        "hausdorff_distance": 0.01,
        "auc": 0.99,
        "best_f1": {"threshold": 0.1, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        "f1_scores": [],
        "inspect": {
            "web_heatmap": "ca web a.pcd ref.pcd --heatmap",
            "heatmap3d": "ca heatmap3d a.pcd ref.pcd -o a_vs_ref_heatmap.png",
        },
        "compression": {
            "source_size_bytes": 1000,
            "baseline_path": "a.pcd",
            "baseline_size_bytes": 1000,
            "compressed_path": "a.cloudini",
            "compressed_size_bytes": 200,
            "size_ratio": 0.2,
            "space_saving_ratio": 0.8,
        },
    },
    {
        "path": "b.pcd",
        "num_points": 100,
        "reference_path": "ref.pcd",
        "reference_points": 100,
        "chamfer_distance": 0.05,
        "hausdorff_distance": 0.2,
        "auc": 0.8,
        "best_f1": {"threshold": 0.2, "precision": 0.8, "recall": 0.9, "f1": 0.8471},
        "f1_scores": [],
        "inspect": {
            "web_heatmap": "ca web b.pcd ref.pcd --heatmap",
            "heatmap3d": "ca heatmap3d b.pcd ref.pcd -o b_vs_ref_heatmap.png",
        },
        "compression": {
            "source_size_bytes": 900,
            "baseline_path": "b.pcd",
            "baseline_size_bytes": 900,
            "compressed_path": "b.cloudini",
            "compressed_size_bytes": 450,
            "size_ratio": 0.5,
            "space_saving_ratio": 0.5,
        },
    },
]
SAMPLE_BATCH_RESULTS_WITH_GATE = [
    {
        **SAMPLE_BATCH_RESULTS[0],
        "quality_gate": {
            "passed": True,
            "min_auc": 0.95,
            "max_chamfer": 0.02,
            "reasons": [],
        },
    },
    {
        **SAMPLE_BATCH_RESULTS[1],
        "quality_gate": {
            "passed": False,
            "min_auc": 0.95,
            "max_chamfer": 0.02,
            "reasons": ["AUC 0.8000 < min_auc 0.9500"],
        },
    },
]
SAMPLE_GROUND_RESULT = {
    "estimated_ground_path": "estimated_ground.pcd",
    "estimated_nonground_path": "estimated_nonground.pcd",
    "reference_ground_path": "reference_ground.pcd",
    "reference_nonground_path": "reference_nonground.pcd",
    "voxel_size": 0.5,
    "counts": {
        "estimated_ground_points": 95,
        "estimated_nonground_points": 105,
        "reference_ground_points": 100,
        "reference_nonground_points": 100,
    },
    "confusion_matrix": {"tp": 90, "fp": 5, "fn": 10, "tn": 95},
    "precision": 0.9474,
    "recall": 0.9000,
    "f1": 0.9231,
    "iou": 0.8571,
    "accuracy": 0.9250,
    "quality_gate": {
        "passed": True,
        "min_precision": 0.9,
        "min_recall": 0.9,
        "min_f1": 0.9,
        "min_iou": 0.8,
        "reasons": [],
    },
}


class TestMakeJson:
    def test_structure(self):
        data = make_json(1000, 2000, 0.95, 0.01, SAMPLE_STATS)
        assert data["source_points"] == 1000
        assert data["target_points"] == 2000
        assert data["fitness"] == 0.95
        assert data["rmse"] == 0.01
        assert data["distance_stats"] == SAMPLE_STATS

    def test_no_registration(self):
        data = make_json(100, 200, None, None, SAMPLE_STATS)
        assert data["fitness"] is None
        assert data["rmse"] is None


class TestSaveJson:
    def test_writes_valid_json(self, tmp_path):
        data = make_json(100, 200, 0.9, 0.02, SAMPLE_STATS)
        path = tmp_path / "out.json"
        save_json(data, str(path))
        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "out.json"
        save_json({"test": 1}, str(path))
        assert path.exists()


class TestMakeMarkdown:
    def test_contains_sections(self, tmp_path):
        data = make_json(100, 200, 0.95, 0.01, SAMPLE_STATS)
        path = tmp_path / "report.md"
        make_markdown(data, str(path))
        content = path.read_text()
        assert "# CloudAnalyzer Report" in content
        assert "## Registration" in content
        assert "## Distance Stats" in content
        assert "## Point Counts" in content
        assert "0.9500" in content  # fitness
        assert "Source: 100" in content

    def test_no_registration_section(self, tmp_path):
        data = make_json(100, 200, None, None, SAMPLE_STATS)
        path = tmp_path / "report.md"
        make_markdown(data, str(path))
        content = path.read_text()
        assert "## Registration" not in content
        assert "## Distance Stats" in content


class TestBatchReports:
    def test_batch_summary(self):
        summary = make_batch_summary(SAMPLE_BATCH_RESULTS, "ref.pcd")
        assert summary["total_files"] == 2
        assert summary["mean_auc"] == 0.895
        assert summary["best_auc"]["path"] == "a.pcd"
        assert summary["worst_chamfer"]["path"] == "b.pcd"
        assert summary["quality_gate"] is None
        assert summary["compression"]["mean_size_ratio"] == 0.35
        assert summary["compression"]["best_size_ratio"]["path"] == "a.pcd"
        assert summary["compression"]["pareto_optimal_count"] == 1
        assert summary["compression"]["pareto_optimal_items"][0]["path"] == "a.pcd"
        assert summary["compression"]["recommended_item"]["path"] == "a.pcd"

    def test_batch_summary_with_quality_gate(self):
        summary = make_batch_summary(
            SAMPLE_BATCH_RESULTS_WITH_GATE,
            "ref.pcd",
            min_auc=0.95,
            max_chamfer=0.02,
        )
        assert summary["quality_gate"]["pass_count"] == 1
        assert summary["quality_gate"]["fail_count"] == 1
        assert summary["quality_gate"]["failed_paths"] == ["b.pcd"]

    def test_batch_markdown(self, tmp_path):
        path = tmp_path / "batch.md"
        make_batch_markdown(SAMPLE_BATCH_RESULTS, "ref.pcd", str(path))
        content = path.read_text()
        plot_path = tmp_path / "batch_quality_vs_size.png"
        assert "# CloudAnalyzer Batch Evaluation Report" in content
        assert "Mean AUC: 0.8950" in content
        assert "## Compression" in content
        assert "Mean Size Ratio: 0.3500" in content
        assert "Pareto Candidates: 1" in content
        assert "Pareto Paths: a.pcd" in content
        assert "Recommended: a.pcd (size=0.2000, auc=0.9900)" in content
        assert "| a.pcd | 100 | 0.0010 | 0.9900 | 0.2000 | Yes | Yes | 1.0000 | 0.10 |" in content
        assert "![Quality vs Size](batch_quality_vs_size.png)" in content
        assert "## Inspection Commands" in content
        assert "`ca web a.pcd ref.pcd --heatmap`" in content
        assert "Snapshot: `ca heatmap3d a.pcd ref.pcd -o a_vs_ref_heatmap.png`" in content
        assert plot_path.exists()

    def test_batch_markdown_with_quality_gate(self, tmp_path):
        path = tmp_path / "batch.md"
        make_batch_markdown(
            SAMPLE_BATCH_RESULTS_WITH_GATE,
            "ref.pcd",
            str(path),
            min_auc=0.95,
            max_chamfer=0.02,
        )
        content = path.read_text()
        assert "## Quality Gate" in content
        assert "Pass: 1" in content
        assert "Fail: 1" in content
        assert "Recommendation Rule: smallest candidate on the gate-filtered Pareto frontier" in content
        assert "| b.pcd | 100 | 0.0500 | 0.8000 | 0.5000 |  |  | 0.8471 | 0.20 | FAIL |" in content

    def test_batch_html(self, tmp_path):
        path = tmp_path / "batch.html"
        make_batch_html(SAMPLE_BATCH_RESULTS, "ref.pcd", str(path))
        content = path.read_text()
        plot_path = tmp_path / "batch_quality_vs_size.png"
        assert "<title>CloudAnalyzer Batch Evaluation Report</title>" in content
        assert "<td>a.pcd</td>" in content
        assert "Mean AUC" in content
        assert "Mean Size Ratio" in content
        assert "<th>Size Ratio</th>" in content
        assert "<th>Pareto</th>" in content
        assert "<th>Recommended</th>" in content
        assert "<td>0.2000</td>" in content
        assert "<td>Yes</td>" in content
        assert "Pareto Candidates" in content
        assert "Recommended" in content
        assert 'class="summary-table"' in content
        assert 'class="summary-chip summary-chip-pareto"' in content
        assert 'class="summary-chip summary-chip-recommended"' in content
        assert 'class="summary-chip-count"' in content
        assert 'summary-chip-count">1<' in content
        assert 'id="summary-show-pass"' not in content
        assert 'data-summary-action="pareto"' in content
        assert 'data-summary-action="recommended"' in content
        assert 'id="summary-show-failed"' not in content
        assert 'id="summary-show-pareto"' in content
        assert 'id="summary-show-recommended"' in content
        assert 'id="quick-show-pass"' in content
        assert 'data-quick-action="pass" disabled' in content
        assert 'id="quick-show-failed"' in content
        assert 'data-quick-action="failed" disabled' in content
        assert 'id="quick-show-pareto"' in content
        assert 'id="quick-show-recommended"' in content
        assert 'id="quick-reset-view"' in content
        assert 'data-quick-action="pareto"' in content
        assert "applyQuickAction('pass')" in content
        assert "applyQuickAction('pareto')" in content
        assert "applyQuickAction('recommended')" in content
        assert 'id="sort-results-control"' in content
        assert 'id="sort-results"' in content
        assert 'class="filter-control" id="sort-results-control"' in content
        assert 'id="filter-pass-only-control"' in content
        assert 'class="filter-control filter-control-disabled" id="filter-pass-only-control"' in content
        assert 'id="filter-failed-only-control"' in content
        assert 'class="filter-control filter-control-disabled" id="filter-failed-only-control"' in content
        assert 'id="filter-recommended-only-control"' in content
        assert 'class="filter-control" id="filter-recommended-only-control"' in content
        assert 'id="filter-pareto-only-control"' in content
        assert 'class="filter-control" id="filter-pareto-only-control"' in content
        assert 'value="auc-desc"' in content
        assert 'value="auc-asc"' in content
        assert 'value="chamfer-asc"' in content
        assert 'value="size-ratio-asc"' in content
        assert 'value="failed-first" disabled' in content
        assert 'value="recommended-first"' in content
        assert 'id="filter-pass-only" disabled' in content
        assert 'id="filter-recommended-only"' in content
        assert 'id="filter-pareto-only"' in content
        assert 'id="filter-failed-only" disabled' in content
        assert 'id="reset-filters"' in content
        assert 'id="filter-summary"' in content
        assert "refreshResultsView()" in content
        assert "resetFilters()" in content
        assert "updateActionStates(activeAction)" in content
        assert "updateFilterControlStates(passEnabled, failedEnabled, paretoEnabled, recommendedEnabled, sortValue)" in content
        assert "control.classList.toggle('filter-control-active', enabled);" in content
        assert "Filters: ${activeFilters.join(', ')}" in content
        assert "Sort: ${sortValue}" in content
        assert ".filter-control-active" in content
        assert ".filter-control-active select" in content
        assert ".filter-control-disabled" in content
        assert "sortKey === 'recommended' && sortDirection === 'first'" in content
        assert "sortKey === 'failed' && sortDirection === 'first'" in content
        assert 'data-pareto="true"' in content
        assert 'data-recommended="true"' in content
        assert 'alt="Quality vs Size plot"' in content
        assert "Inspection Commands" in content
        assert "ca web a.pcd ref.pcd --heatmap" in content
        assert "ca heatmap3d a.pcd ref.pcd -o a_vs_ref_heatmap.png" in content
        assert "copyCommand(" in content
        assert ">Copy</button>" in content
        assert plot_path.exists()

    def test_batch_html_with_quality_gate(self, tmp_path):
        path = tmp_path / "batch.html"
        make_batch_html(
            SAMPLE_BATCH_RESULTS_WITH_GATE,
            "ref.pcd",
            str(path),
            min_auc=0.95,
            max_chamfer=0.02,
        )
        content = path.read_text()
        assert "<th>Status</th>" in content
        assert "<td>FAIL</td>" in content
        assert "Max Chamfer" in content
        assert 'class="fail-row"' in content
        assert "pareto-row" in content
        assert "recommended-row" in content
        assert 'class="summary-chip summary-chip-pass"' in content
        assert 'class="summary-chip summary-chip-failed"' in content
        assert 'summary-chip-count">1<' in content
        assert 'id="summary-show-pass"' in content
        assert 'id="summary-show-failed"' in content
        assert 'id="summary-show-pareto"' in content
        assert 'id="summary-show-recommended"' in content
        assert 'id="quick-show-pass"' in content
        assert 'id="quick-show-failed"' in content
        assert 'id="sort-results-control"' in content
        assert 'class="filter-control" id="sort-results-control"' in content
        assert 'id="filter-pass-only-control"' in content
        assert 'class="filter-control" id="filter-pass-only-control"' in content
        assert 'id="filter-failed-only-control"' in content
        assert 'class="filter-control" id="filter-failed-only-control"' in content
        assert 'id="filter-recommended-only-control"' in content
        assert 'class="filter-control" id="filter-recommended-only-control"' in content
        assert 'id="filter-pareto-only-control"' in content
        assert 'class="filter-control" id="filter-pareto-only-control"' in content
        assert 'id="filter-pass-only"' in content
        assert 'id="filter-failed-only"' in content
        assert 'id="filter-pareto-only"' in content
        assert 'id="sort-results"' in content
        assert 'value="failed-first"' in content
        assert 'value="recommended-first"' in content
        assert 'data-passed="true"' in content
        assert 'data-failed="true"' in content
        assert 'data-pareto="true"' in content
        assert 'id="results-table-body"' in content
        assert 'id="inspection-table-body"' in content
        assert "row.dataset.passed" in content
        assert "row.dataset.failed" in content
        assert "row.dataset.pareto" in content
        assert "compareRows(a, b, sortValue)" in content
        assert "setSortValue('auc-desc')" in content
        assert "setSortValue('auc-asc')" in content
        assert "summary-chip-active" in content
        assert "quick-action-chip-active" in content
        assert "updateFilterControlStates(passEnabled, failedEnabled, paretoEnabled, recommendedEnabled, sortValue)" in content
        assert "sortKey === 'recommended' && sortDirection === 'first'" in content
        assert "sortKey === 'failed' && sortDirection === 'first'" in content

    def test_batch_report_dispatch(self, tmp_path):
        path = tmp_path / "batch.html"
        save_batch_report(SAMPLE_BATCH_RESULTS, "ref.pcd", str(path))
        assert path.exists()

    def test_batch_report_invalid_extension(self, tmp_path):
        path = tmp_path / "batch.txt"
        try:
            save_batch_report(SAMPLE_BATCH_RESULTS, "ref.pcd", str(path))
        except ValueError as e:
            assert "Unsupported report format" in str(e)
        else:
            raise AssertionError("Expected ValueError for unsupported extension")


class TestGroundReports:
    def test_ground_markdown(self, tmp_path):
        path = tmp_path / "ground.md"
        make_ground_markdown(SAMPLE_GROUND_RESULT, str(path))
        content = path.read_text()
        assert "# CloudAnalyzer Ground Segmentation Report" in content
        assert "## Metrics" in content
        assert "Precision: 0.9474" in content
        assert "| Ground | 90 | 5 |" in content
        assert "## Quality Gate" in content
        assert "Status: PASS" in content

    def test_ground_html(self, tmp_path):
        path = tmp_path / "ground.html"
        make_ground_html(SAMPLE_GROUND_RESULT, str(path))
        content = path.read_text()
        assert "<title>CloudAnalyzer Ground Segmentation Report</title>" in content
        assert "<h2>Confusion Matrix</h2>" in content
        assert "<td>90</td><td>5</td>" in content
        assert "Quality Gate" in content
        assert "PASS" in content

    def test_ground_report_dispatch(self, tmp_path):
        path = tmp_path / "ground.html"
        save_ground_report(SAMPLE_GROUND_RESULT, str(path))
        assert path.exists()

    def test_ground_report_invalid_extension(self, tmp_path):
        path = tmp_path / "ground.txt"
        try:
            save_ground_report(SAMPLE_GROUND_RESULT, str(path))
        except ValueError as e:
            assert "Unsupported report format" in str(e)
        else:
            raise AssertionError("Expected ValueError for unsupported extension")
