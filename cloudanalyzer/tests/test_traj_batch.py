"""Tests for trajectory batch evaluation."""

import pytest

from ca.batch import trajectory_batch_evaluate
from ca.report import make_trajectory_batch_summary, save_trajectory_batch_report


def _write_csv_trajectory(path, rows):
    lines = ["timestamp,x,y,z"]
    lines.extend(f"{timestamp},{x},{y},{z}" for timestamp, x, y, z in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return str(path)


class TestTrajectoryBatchEvaluate:
    def test_evaluates_all_files(self, tmp_path):
        estimated_dir = tmp_path / "estimated"
        reference_dir = tmp_path / "reference"
        _write_csv_trajectory(
            estimated_dir / "a.csv",
            [(0.0, 0.1, 0.0, 0.0), (1.0, 1.1, 0.0, 0.0), (2.0, 2.1, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            estimated_dir / "b.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.4, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            reference_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            reference_dir / "b.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        results = trajectory_batch_evaluate(str(estimated_dir), str(reference_dir))

        assert len(results) == 2
        assert results[0]["inspect"]["traj_evaluate"].startswith("ca traj-evaluate ")
        assert results[0]["inspect"]["traj_evaluate_aligned"].startswith("ca traj-evaluate ")
        assert results[0]["coverage_ratio"] == pytest.approx(1.0)

    def test_align_origin(self, tmp_path):
        estimated_dir = tmp_path / "estimated"
        reference_dir = tmp_path / "reference"
        _write_csv_trajectory(
            estimated_dir / "a.csv",
            [(0.0, 5.0, -2.0, 1.0), (1.0, 6.0, -2.0, 1.0), (2.0, 7.0, -2.0, 1.0)],
        )
        _write_csv_trajectory(
            reference_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        results = trajectory_batch_evaluate(
            str(estimated_dir),
            str(reference_dir),
            align_origin=True,
        )

        assert results[0]["alignment"]["mode"] == "origin"
        assert results[0]["ate"]["rmse"] == pytest.approx(0.0, abs=1e-8)

    def test_align_rigid(self, tmp_path):
        estimated_dir = tmp_path / "estimated"
        reference_dir = tmp_path / "reference"
        _write_csv_trajectory(
            estimated_dir / "a.csv",
            [
                (0.0, 5.0, -3.0, 0.0),
                (1.0, 5.0, -2.0, 0.0),
                (2.0, 4.0, -2.0, 0.0),
                (3.0, 4.0, -1.0, 0.0),
            ],
        )
        _write_csv_trajectory(
            reference_dir / "a.csv",
            [
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 1.0, 0.0, 0.0),
                (2.0, 1.0, 1.0, 0.0),
                (3.0, 2.0, 1.0, 0.0),
            ],
        )

        results = trajectory_batch_evaluate(
            str(estimated_dir),
            str(reference_dir),
            align_rigid=True,
        )

        assert results[0]["alignment"]["mode"] == "rigid"
        assert results[0]["ate"]["rmse"] == pytest.approx(0.0, abs=1e-8)

    def test_quality_gate(self, tmp_path):
        estimated_dir = tmp_path / "estimated"
        reference_dir = tmp_path / "reference"
        _write_csv_trajectory(
            estimated_dir / "a.csv",
            [(0.0, 0.1, 0.0, 0.0), (1.0, 1.1, 0.0, 0.0), (2.0, 2.1, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            estimated_dir / "b.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.5, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            reference_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            reference_dir / "b.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        results = trajectory_batch_evaluate(
            str(estimated_dir),
            str(reference_dir),
            max_ate=0.2,
        )

        assert results[0]["quality_gate"]["passed"] is True
        assert results[1]["quality_gate"]["passed"] is False
        assert results[1]["quality_gate"]["reasons"]

    def test_quality_gate_max_drift(self, tmp_path):
        estimated_dir = tmp_path / "estimated"
        reference_dir = tmp_path / "reference"
        _write_csv_trajectory(
            estimated_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.3, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            reference_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.0, 0.0, 0.0)],
        )

        results = trajectory_batch_evaluate(
            str(estimated_dir),
            str(reference_dir),
            max_drift=0.2,
        )

        assert results[0]["drift"]["endpoint"] == pytest.approx(0.3)
        assert results[0]["quality_gate"]["passed"] is False
        assert results[0]["quality_gate"]["max_drift"] == pytest.approx(0.2)
        assert any("Endpoint Drift" in reason for reason in results[0]["quality_gate"]["reasons"])

    def test_quality_gate_min_coverage(self, tmp_path):
        estimated_dir = tmp_path / "estimated"
        reference_dir = tmp_path / "reference"
        _write_csv_trajectory(
            estimated_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            reference_dir / "a.csv",
            [
                (0.0, 0.0, 0.0, 0.0),
                (0.5, 0.5, 0.0, 0.0),
                (1.0, 1.0, 0.0, 0.0),
                (1.5, 1.5, 0.0, 0.0),
                (2.0, 2.0, 0.0, 0.0),
            ],
        )

        results = trajectory_batch_evaluate(
            str(estimated_dir),
            str(reference_dir),
            max_time_delta=0.05,
            min_coverage=0.8,
        )

        assert results[0]["coverage_ratio"] == pytest.approx(0.6)
        assert results[0]["quality_gate"]["passed"] is False
        assert results[0]["quality_gate"]["min_coverage"] == pytest.approx(0.8)
        assert any("Coverage" in reason for reason in results[0]["quality_gate"]["reasons"])


class TestTrajectoryBatchReport:
    def test_summary_and_report(self, tmp_path):
        estimated_dir = tmp_path / "estimated"
        reference_dir = tmp_path / "reference"
        _write_csv_trajectory(
            estimated_dir / "a.csv",
            [(0.0, 0.1, 0.0, 0.0), (1.0, 1.1, 0.0, 0.0), (2.0, 2.1, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            reference_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        results = trajectory_batch_evaluate(str(estimated_dir), str(reference_dir))
        summary = make_trajectory_batch_summary(
            results,
            str(reference_dir),
            max_drift=0.5,
            min_coverage=0.8,
        )

        assert summary["total_files"] == 1
        assert summary["mean_ate_rmse"] == pytest.approx(0.1)
        assert summary["quality_gate"]["max_drift"] == pytest.approx(0.5)
        assert summary["quality_gate"]["min_coverage"] == pytest.approx(0.8)

        markdown_path = tmp_path / "traj_batch.md"
        html_path = tmp_path / "traj_batch.html"
        save_trajectory_batch_report(
            results,
            str(reference_dir),
            str(markdown_path),
            max_drift=0.5,
            min_coverage=0.8,
        )
        save_trajectory_batch_report(
            results,
            str(reference_dir),
            str(html_path),
            max_drift=0.5,
            min_coverage=0.8,
        )

        markdown = markdown_path.read_text()
        html = html_path.read_text()
        assert "# CloudAnalyzer Trajectory Batch Evaluation Report" in markdown
        assert "## Inspection Commands" in markdown
        assert "Rigid:" in markdown
        assert "Max Drift: 0.5000" in markdown
        assert "Min Coverage: 80.0%" in markdown
        assert "<title>CloudAnalyzer Trajectory Batch Evaluation Report</title>" in html
        assert "ca traj-evaluate" in html
        assert "Inspection Commands" in html
        assert "Max Drift" in html
        assert "Min Coverage" in html

    def test_html_filters_and_sort(self, tmp_path):
        estimated_dir = tmp_path / "estimated"
        reference_dir = tmp_path / "reference"
        _write_csv_trajectory(
            estimated_dir / "a.csv",
            [(0.0, 0.1, 0.0, 0.0), (1.0, 1.1, 0.0, 0.0), (2.0, 2.1, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            estimated_dir / "b.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            reference_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            reference_dir / "b.csv",
            [
                (0.0, 0.0, 0.0, 0.0),
                (0.5, 0.5, 0.0, 0.0),
                (1.0, 1.0, 0.0, 0.0),
                (1.5, 1.5, 0.0, 0.0),
                (2.0, 2.0, 0.0, 0.0),
            ],
        )

        results = trajectory_batch_evaluate(
            str(estimated_dir),
            str(reference_dir),
            max_ate=0.2,
            min_coverage=0.8,
        )

        html_path = tmp_path / "traj_batch.html"
        save_trajectory_batch_report(
            results,
            str(reference_dir),
            str(html_path),
            max_ate=0.2,
            min_coverage=0.8,
        )

        html = html_path.read_text()
        assert 'class="summary-table"' in html
        assert 'id="trajectory-summary-show-pass"' in html
        assert 'id="trajectory-summary-show-failed"' in html
        assert 'id="trajectory-summary-show-low-coverage"' in html
        assert 'id="trajectory-quick-show-pass"' in html
        assert 'id="trajectory-quick-show-failed"' in html
        assert 'id="trajectory-quick-show-low-coverage"' in html
        assert 'id="trajectory-quick-reset-view"' in html
        assert "Low Coverage (&lt;80%)" in html
        assert 'id="trajectory-sort-results-control"' in html
        assert 'id="trajectory-sort-results"' in html
        assert 'value="ate-desc"' in html
        assert 'value="coverage-asc"' in html
        assert 'value="drift-desc"' in html
        assert 'id="trajectory-filter-pass-only-control"' in html
        assert 'id="trajectory-filter-failed-only-control"' in html
        assert 'id="trajectory-filter-low-coverage-only-control"' in html
        assert 'id="trajectory-filter-summary"' in html
        assert 'id="trajectory-results-table-body"' in html
        assert 'id="trajectory-inspection-table-body"' in html
        assert 'data-low-coverage="true"' in html
        assert 'data-passed="true"' in html
        assert "applyTrajectoryQuickAction('pass')" in html
        assert "applyTrajectoryQuickAction('failed')" in html
        assert "applyTrajectoryQuickAction('low-coverage')" in html
        assert "refreshTrajectoryResultsView()" in html
        assert "compareTrajectoryRows(a, b, sortValue)" in html
        assert "updateTrajectoryActionStates(activeAction)" in html
        assert "updateTrajectoryFilterControlStates(passEnabled, failedEnabled, lowCoverageEnabled, sortValue)" in html
        assert "Filters: ${activeFilters.join(', ')}" in html
        assert "Sort: ${sortValue}" in html
