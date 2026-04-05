"""Tests for trajectory evaluation."""

import pytest

from ca.report import save_trajectory_report
from ca.trajectory import evaluate_trajectory, load_trajectory


def _write_csv_trajectory(path, rows, with_header=True):
    lines = []
    if with_header:
        lines.append("timestamp,x,y,z")
    lines.extend(f"{timestamp},{x},{y},{z}" for timestamp, x, y, z in rows)
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _write_tum_trajectory(path, rows):
    lines = [
        f"{timestamp} {x} {y} {z} 0 0 0 1"
        for timestamp, x, y, z in rows
    ]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


class TestLoadTrajectory:
    def test_load_csv_with_header(self, tmp_path):
        path = _write_csv_trajectory(
            tmp_path / "traj.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)],
        )
        result = load_trajectory(path)

        assert result["format"] == "csv"
        assert result["num_poses"] == 2
        assert result["timestamps"].tolist() == [0.0, 1.0]

    def test_load_tum(self, tmp_path):
        path = _write_tum_trajectory(
            tmp_path / "traj.tum",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)],
        )
        result = load_trajectory(path)

        assert result["format"] == "tum"
        assert result["num_poses"] == 2
        assert result["positions"].tolist() == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

    def test_rejects_non_monotonic_timestamps(self, tmp_path):
        path = _write_csv_trajectory(
            tmp_path / "traj.csv",
            [(0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)],
        )
        with pytest.raises(ValueError, match="strictly increasing"):
            load_trajectory(path)


class TestEvaluateTrajectory:
    def test_constant_offset_has_ate_but_zero_rpe(self, tmp_path):
        reference = _write_csv_trajectory(
            tmp_path / "ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.0, 0.0, 0.0)],
        )
        estimated = _write_csv_trajectory(
            tmp_path / "est.csv",
            [(0.0, 0.1, 0.0, 0.0), (1.0, 1.1, 0.0, 0.0), (2.0, 2.1, 0.0, 0.0), (3.0, 3.1, 0.0, 0.0)],
        )

        result = evaluate_trajectory(estimated, reference, max_time_delta=0.05)

        assert result["matching"]["matched_poses"] == 4
        assert result["matching"]["coverage_ratio"] == pytest.approx(1.0)
        assert result["ate"]["rmse"] == pytest.approx(0.1)
        assert result["rpe_translation"]["rmse"] == pytest.approx(0.0, abs=1e-8)
        assert result["drift"]["endpoint"] == pytest.approx(0.0, abs=1e-8)
        assert len(result["matched_trajectory"]["timestamps"]) == 4
        assert len(result["error_series"]["rpe_translation"]) == 3
        assert result["quality_gate"] is None
        # Constant +X offset along an X-axis trajectory is purely longitudinal
        assert result["lateral"]["rmse"] == pytest.approx(0.0, abs=1e-8)
        assert result["longitudinal"]["rmse"] == pytest.approx(0.1, abs=1e-4)

    def test_lateral_offset_along_x_trajectory(self, tmp_path):
        # Reference moves along X axis; estimated is offset in Y (purely lateral)
        reference = _write_csv_trajectory(
            tmp_path / "ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.0, 0.0, 0.0)],
        )
        estimated = _write_csv_trajectory(
            tmp_path / "est.csv",
            [(0.0, 0.0, 0.2, 0.0), (1.0, 1.0, 0.2, 0.0), (2.0, 2.0, 0.2, 0.0), (3.0, 3.0, 0.2, 0.0)],
        )

        result = evaluate_trajectory(estimated, reference, max_time_delta=0.05)

        assert result["lateral"]["rmse"] == pytest.approx(0.2, abs=1e-4)
        assert result["longitudinal"]["rmse"] == pytest.approx(0.0, abs=1e-4)
        assert "lateral_errors" in result["matched_trajectory"]
        assert "longitudinal_errors" in result["matched_trajectory"]

    def test_lateral_quality_gate(self, tmp_path):
        reference = _write_csv_trajectory(
            tmp_path / "ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        estimated = _write_csv_trajectory(
            tmp_path / "est.csv",
            [(0.0, 0.0, 0.3, 0.0), (1.0, 1.0, 0.3, 0.0), (2.0, 2.0, 0.3, 0.0)],
        )

        result = evaluate_trajectory(estimated, reference, max_time_delta=0.05, max_lateral=0.2)

        assert result["quality_gate"] is not None
        assert result["quality_gate"]["passed"] is False
        assert any("Lateral" in r for r in result["quality_gate"]["reasons"])

    def test_interpolates_estimated_positions(self, tmp_path):
        reference = _write_csv_trajectory(
            tmp_path / "ref.csv",
            [
                (0.0, 0.0, 0.0, 0.0),
                (0.5, 0.5, 0.0, 0.0),
                (1.0, 1.0, 0.0, 0.0),
                (1.5, 1.5, 0.0, 0.0),
                (2.0, 2.0, 0.0, 0.0),
            ],
        )
        estimated = _write_csv_trajectory(
            tmp_path / "est.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        result = evaluate_trajectory(estimated, reference, max_time_delta=0.5)

        assert result["matching"]["matched_poses"] == 5
        assert result["ate"]["rmse"] == pytest.approx(0.0, abs=1e-8)
        assert result["rpe_translation"]["rmse"] == pytest.approx(0.0, abs=1e-8)

    def test_align_origin_removes_constant_translation_offset(self, tmp_path):
        reference = _write_csv_trajectory(
            tmp_path / "ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        estimated = _write_csv_trajectory(
            tmp_path / "est.csv",
            [(0.0, 5.0, -2.0, 1.0), (1.0, 6.0, -2.0, 1.0), (2.0, 7.0, -2.0, 1.0)],
        )

        result = evaluate_trajectory(
            estimated,
            reference,
            max_time_delta=0.05,
            align_origin=True,
        )

        assert result["alignment"]["mode"] == "origin"
        assert result["alignment"]["translation"] == pytest.approx([-5.0, 2.0, -1.0])
        assert result["ate"]["rmse"] == pytest.approx(0.0, abs=1e-8)
        assert result["rpe_translation"]["rmse"] == pytest.approx(0.0, abs=1e-8)

    def test_align_rigid_removes_rotation_and_translation_offset(self, tmp_path):
        reference = _write_csv_trajectory(
            tmp_path / "ref.csv",
            [
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 1.0, 0.0, 0.0),
                (2.0, 1.0, 1.0, 0.0),
                (3.0, 2.0, 1.0, 0.0),
            ],
        )
        estimated = _write_csv_trajectory(
            tmp_path / "est.csv",
            [
                (0.0, 5.0, -3.0, 0.0),
                (1.0, 5.0, -2.0, 0.0),
                (2.0, 4.0, -2.0, 0.0),
                (3.0, 4.0, -1.0, 0.0),
            ],
        )

        result = evaluate_trajectory(
            estimated,
            reference,
            max_time_delta=0.05,
            align_rigid=True,
        )

        assert result["alignment"]["mode"] == "rigid"
        assert result["ate"]["rmse"] == pytest.approx(0.0, abs=1e-8)
        assert result["rpe_translation"]["rmse"] == pytest.approx(0.0, abs=1e-8)

    def test_alignment_modes_are_mutually_exclusive(self, tmp_path):
        reference = _write_csv_trajectory(
            tmp_path / "ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)],
        )
        estimated = _write_csv_trajectory(
            tmp_path / "est.csv",
            [(0.0, 1.0, 0.0, 0.0), (1.0, 2.0, 0.0, 0.0)],
        )

        with pytest.raises(ValueError, match="mutually exclusive"):
            evaluate_trajectory(
                estimated,
                reference,
                max_time_delta=0.05,
                align_origin=True,
                align_rigid=True,
            )

    def test_quality_gate_can_fail(self, tmp_path):
        reference = _write_csv_trajectory(
            tmp_path / "ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.0, 0.0, 0.0)],
        )
        estimated = _write_csv_trajectory(
            tmp_path / "est.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.3, 0.0, 0.0), (3.0, 3.9, 0.0, 0.0)],
        )

        result = evaluate_trajectory(
            estimated,
            reference,
            max_time_delta=0.05,
            max_ate=0.2,
            max_rpe=0.2,
        )

        assert result["quality_gate"] is not None
        assert result["quality_gate"]["passed"] is False
        assert "ATE RMSE" in result["quality_gate"]["reasons"][0]
        assert any("RPE RMSE" in reason for reason in result["quality_gate"]["reasons"])

    def test_quality_gate_can_fail_on_coverage(self, tmp_path):
        reference = _write_csv_trajectory(
            tmp_path / "ref.csv",
            [
                (0.0, 0.0, 0.0, 0.0),
                (0.5, 0.5, 0.0, 0.0),
                (1.0, 1.0, 0.0, 0.0),
                (1.5, 1.5, 0.0, 0.0),
                (2.0, 2.0, 0.0, 0.0),
            ],
        )
        estimated = _write_csv_trajectory(
            tmp_path / "est.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        result = evaluate_trajectory(
            estimated,
            reference,
            max_time_delta=0.05,
            min_coverage=0.8,
        )

        assert result["matching"]["coverage_ratio"] == pytest.approx(0.6)
        assert result["quality_gate"] is not None
        assert result["quality_gate"]["passed"] is False
        assert result["quality_gate"]["min_coverage"] == pytest.approx(0.8)
        assert any("Coverage" in reason for reason in result["quality_gate"]["reasons"])

    def test_quality_gate_can_fail_on_drift(self, tmp_path):
        reference = _write_csv_trajectory(
            tmp_path / "ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.0, 0.0, 0.0)],
        )
        estimated = _write_csv_trajectory(
            tmp_path / "est.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.3, 0.0, 0.0)],
        )

        result = evaluate_trajectory(
            estimated,
            reference,
            max_time_delta=0.05,
            max_drift=0.2,
        )

        assert result["drift"]["endpoint"] == pytest.approx(0.3)
        assert result["quality_gate"] is not None
        assert result["quality_gate"]["passed"] is False
        assert result["quality_gate"]["max_drift"] == pytest.approx(0.2)
        assert any("Endpoint Drift" in reason for reason in result["quality_gate"]["reasons"])

    def test_requires_matching_poses(self, tmp_path):
        reference = _write_csv_trajectory(
            tmp_path / "ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)],
        )
        estimated = _write_csv_trajectory(
            tmp_path / "est.csv",
            [(10.0, 10.0, 0.0, 0.0), (11.0, 11.0, 0.0, 0.0)],
        )

        with pytest.raises(ValueError, match="matched poses"):
            evaluate_trajectory(estimated, reference, max_time_delta=0.1)


class TestTrajectoryReport:
    def test_save_markdown_and_html(self, tmp_path):
        reference = _write_csv_trajectory(
            tmp_path / "ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        estimated = _write_csv_trajectory(
            tmp_path / "est.csv",
            [(0.0, 0.1, 0.0, 0.0), (1.0, 1.1, 0.0, 0.0), (2.0, 2.1, 0.0, 0.0)],
        )
        result = evaluate_trajectory(
            estimated,
            reference,
            max_time_delta=0.05,
            max_ate=0.2,
            max_drift=0.5,
            min_coverage=0.8,
        )

        markdown_path = tmp_path / "trajectory_report.md"
        html_path = tmp_path / "trajectory_report.html"
        markdown_overlay = tmp_path / "trajectory_report_trajectory_overlay.png"
        markdown_errors = tmp_path / "trajectory_report_trajectory_errors.png"
        html_overlay = tmp_path / "trajectory_report_trajectory_overlay.png"
        html_errors = tmp_path / "trajectory_report_trajectory_errors.png"
        save_trajectory_report(result, str(markdown_path))
        save_trajectory_report(result, str(html_path))

        markdown = markdown_path.read_text()
        html = html_path.read_text()
        assert "# CloudAnalyzer Trajectory Evaluation Report" in markdown
        assert "## Visualizations" in markdown
        assert "- Alignment: none" in markdown
        assert "## Absolute Trajectory Error (ATE)" in markdown
        assert "## Worst RPE Segments" in markdown
        assert "![Trajectory Overlay](trajectory_report_trajectory_overlay.png)" in markdown
        assert "![Trajectory Errors](trajectory_report_trajectory_errors.png)" in markdown
        assert "<title>CloudAnalyzer Trajectory Evaluation Report</title>" in html
        assert "Alignment" in html
        assert "Trajectory overlay plot" in html
        assert "Trajectory error plot" in html
        assert "Worst ATE Samples" in html
        assert "Quality Gate: PASS" in html
        assert "- Max Drift: 0.5000" in markdown
        assert "- Min Coverage: 80.0%" in markdown
        assert "Min Coverage" in html
        assert "Max Drift" in html
        assert markdown_overlay.exists()
        assert markdown_errors.exists()
        assert html_overlay.exists()
        assert html_errors.exists()
