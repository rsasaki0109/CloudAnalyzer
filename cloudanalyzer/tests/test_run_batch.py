"""Tests for combined run-batch evaluation."""

from pathlib import Path

import open3d as o3d
import pytest

from ca.report import make_run_batch_summary, save_run_batch_report
from ca.run_evaluate import evaluate_run_batch


def _write_csv_trajectory(path: Path, rows: list[tuple[float, float, float, float]]) -> str:
    lines = ["timestamp,x,y,z"]
    lines.extend(f"{timestamp},{x},{y},{z}" for timestamp, x, y, z in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return str(path)


class TestEvaluateRunBatch:
    def test_evaluates_all_runs(self, tmp_path, identical_pcd, shifted_pcd):
        map_dir = tmp_path / "maps"
        map_reference_dir = tmp_path / "map_refs"
        trajectory_dir = tmp_path / "trajs"
        trajectory_reference_dir = tmp_path / "traj_refs"
        map_dir.mkdir()
        map_reference_dir.mkdir()
        trajectory_dir.mkdir()
        trajectory_reference_dir.mkdir()

        o3d.io.write_point_cloud(str(map_dir / "a.pcd"), identical_pcd)
        o3d.io.write_point_cloud(str(map_reference_dir / "a.pcd"), identical_pcd)
        _write_csv_trajectory(
            trajectory_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            trajectory_reference_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        o3d.io.write_point_cloud(str(map_dir / "b.pcd"), shifted_pcd)
        o3d.io.write_point_cloud(str(map_reference_dir / "b.pcd"), identical_pcd)
        _write_csv_trajectory(
            trajectory_dir / "b.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.3, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            trajectory_reference_dir / "b.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.0, 0.0, 0.0)],
        )

        results = evaluate_run_batch(
            str(map_dir),
            str(map_reference_dir),
            str(trajectory_dir),
            str(trajectory_reference_dir),
            min_auc=0.95,
            max_chamfer=0.05,
            max_drift=0.2,
        )

        assert len(results) == 2
        assert results[0]["id"] == "a"
        assert results[0]["overall_quality_gate"]["passed"] is True
        assert results[1]["id"] == "b"
        assert results[1]["overall_quality_gate"]["passed"] is False
        assert any("Map:" in reason for reason in results[1]["overall_quality_gate"]["reasons"])
        assert any("Trajectory:" in reason for reason in results[1]["overall_quality_gate"]["reasons"])


class TestRunBatchReport:
    def test_summary_and_report(self, tmp_path, identical_pcd, shifted_pcd):
        map_dir = tmp_path / "maps"
        map_reference_dir = tmp_path / "map_refs"
        trajectory_dir = tmp_path / "trajs"
        trajectory_reference_dir = tmp_path / "traj_refs"
        map_dir.mkdir()
        map_reference_dir.mkdir()
        trajectory_dir.mkdir()
        trajectory_reference_dir.mkdir()

        o3d.io.write_point_cloud(str(map_dir / "a.pcd"), identical_pcd)
        o3d.io.write_point_cloud(str(map_reference_dir / "a.pcd"), identical_pcd)
        _write_csv_trajectory(
            trajectory_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            trajectory_reference_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        o3d.io.write_point_cloud(str(map_dir / "b.pcd"), shifted_pcd)
        o3d.io.write_point_cloud(str(map_reference_dir / "b.pcd"), identical_pcd)
        _write_csv_trajectory(
            trajectory_dir / "b.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.3, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            trajectory_reference_dir / "b.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.0, 0.0, 0.0)],
        )

        results = evaluate_run_batch(
            str(map_dir),
            str(map_reference_dir),
            str(trajectory_dir),
            str(trajectory_reference_dir),
            min_auc=0.95,
            max_chamfer=0.05,
            max_drift=0.2,
        )
        summary = make_run_batch_summary(
            results,
            str(map_reference_dir),
            str(trajectory_reference_dir),
            min_auc=0.95,
            max_chamfer=0.05,
            max_drift=0.2,
        )

        assert summary["total_runs"] == 2
        assert summary["quality_gate"]["pass_count"] == 1
        assert summary["quality_gate"]["fail_count"] == 1
        assert summary["quality_gate"]["map_fail_count"] == 1
        assert summary["quality_gate"]["trajectory_fail_count"] == 1
        assert summary["quality_gate"]["failed_ids"] == ["b"]

        markdown_path = tmp_path / "run_batch.md"
        html_path = tmp_path / "run_batch.html"
        save_run_batch_report(
            results,
            str(map_reference_dir),
            str(trajectory_reference_dir),
            str(markdown_path),
            min_auc=0.95,
            max_chamfer=0.05,
            max_drift=0.2,
        )
        save_run_batch_report(
            results,
            str(map_reference_dir),
            str(trajectory_reference_dir),
            str(html_path),
            min_auc=0.95,
            max_chamfer=0.05,
            max_drift=0.2,
        )

        markdown = markdown_path.read_text()
        html = html_path.read_text()
        assert "# CloudAnalyzer Run Batch Evaluation Report" in markdown
        assert "## Quality Gate" in markdown
        assert "Mean Map AUC" in markdown
        assert "Map Failures: 1" in markdown
        assert "Trajectory Failures: 1" in markdown
        assert "Run viewer:" in markdown
        assert "Map heatmap:" in markdown
        assert "Combined:" in markdown
        assert "<title>CloudAnalyzer Run Batch Evaluation Report</title>" in html
        assert "Results" in html
        assert "Inspection Commands" in html
        assert "<th>Map Status</th>" in html
        assert "<th>Trajectory Status</th>" in html
        assert "<th>Overall</th>" in html
        assert "ca web" in html
        assert "ca run-evaluate" in html

    def test_html_filters_and_sort(self, tmp_path, identical_pcd, shifted_pcd):
        map_dir = tmp_path / "maps"
        map_reference_dir = tmp_path / "map_refs"
        trajectory_dir = tmp_path / "trajs"
        trajectory_reference_dir = tmp_path / "traj_refs"
        map_dir.mkdir()
        map_reference_dir.mkdir()
        trajectory_dir.mkdir()
        trajectory_reference_dir.mkdir()

        o3d.io.write_point_cloud(str(map_dir / "a.pcd"), identical_pcd)
        o3d.io.write_point_cloud(str(map_reference_dir / "a.pcd"), identical_pcd)
        _write_csv_trajectory(
            trajectory_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            trajectory_reference_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        o3d.io.write_point_cloud(str(map_dir / "b.pcd"), shifted_pcd)
        o3d.io.write_point_cloud(str(map_reference_dir / "b.pcd"), identical_pcd)
        _write_csv_trajectory(
            trajectory_dir / "b.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.3, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            trajectory_reference_dir / "b.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.0, 0.0, 0.0)],
        )

        results = evaluate_run_batch(
            str(map_dir),
            str(map_reference_dir),
            str(trajectory_dir),
            str(trajectory_reference_dir),
            min_auc=0.95,
            max_chamfer=0.05,
            max_drift=0.2,
        )

        html_path = tmp_path / "run_batch.html"
        save_run_batch_report(
            results,
            str(map_reference_dir),
            str(trajectory_reference_dir),
            str(html_path),
            min_auc=0.95,
            max_chamfer=0.05,
            max_drift=0.2,
        )

        html = html_path.read_text()
        assert 'class="summary-table"' in html
        assert 'id="run-batch-summary-show-pass"' in html
        assert 'id="run-batch-summary-show-failed"' in html
        assert 'id="run-batch-summary-show-map-failed"' in html
        assert 'id="run-batch-summary-show-trajectory-failed"' in html
        assert 'id="run-batch-quick-show-pass"' in html
        assert 'id="run-batch-quick-show-failed"' in html
        assert 'id="run-batch-quick-show-map-failed"' in html
        assert 'id="run-batch-quick-show-trajectory-failed"' in html
        assert 'id="run-batch-quick-reset-view"' in html
        assert 'id="run-batch-sort-results-control"' in html
        assert 'id="run-batch-sort-results"' in html
        assert 'value="map-auc-desc"' in html
        assert 'value="map-chamfer-asc"' in html
        assert 'value="traj-ate-desc"' in html
        assert 'value="traj-drift-desc"' in html
        assert 'value="coverage-asc"' in html
        assert 'id="run-batch-filter-pass-only-control"' in html
        assert 'id="run-batch-filter-failed-only-control"' in html
        assert 'id="run-batch-filter-map-failed-only-control"' in html
        assert 'id="run-batch-filter-trajectory-failed-only-control"' in html
        assert 'id="run-batch-filter-summary"' in html
        assert 'id="run-batch-results-table-body"' in html
        assert 'id="run-batch-inspection-table-body"' in html
        assert "<th>Map Status</th>" in html
        assert "<th>Trajectory Status</th>" in html
        assert "<th>Overall</th>" in html
        assert 'data-map-failed="true"' in html
        assert 'data-trajectory-failed="true"' in html
        assert 'data-passed="true"' in html
        assert "ca run-evaluate" in html
        assert "--trajectory-reference" in html
        assert "applyRunBatchQuickAction('failed')" in html
        assert "applyRunBatchQuickAction('map-failed')" in html
        assert "applyRunBatchQuickAction('trajectory-failed')" in html
        assert "refreshRunBatchResultsView()" in html
        assert "compareRunBatchRows(a, b, sortValue)" in html
        assert "updateRunBatchActionStates(activeAction)" in html
        assert "updateRunBatchFilterControlStates(passEnabled, failedEnabled, mapFailedEnabled, trajectoryFailedEnabled, sortValue)" in html
        assert "Filters: ${activeFilters.join(', ')}" in html
        assert "Sort: ${sortValue}" in html
