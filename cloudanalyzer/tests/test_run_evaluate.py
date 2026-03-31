"""Tests for combined run evaluation."""

from pathlib import Path

import open3d as o3d
import pytest

from ca.report import save_run_report
from ca.run_evaluate import evaluate_run


def _write_csv_trajectory(path: Path, rows: list[tuple[float, float, float, float]]) -> str:
    lines = ["timestamp,x,y,z"]
    lines.extend(f"{timestamp},{x},{y},{z}" for timestamp, x, y, z in rows)
    path.write_text("\n".join(lines) + "\n")
    return str(path)


class TestEvaluateRun:
    def test_pass_result_and_inspection_commands(self, tmp_path, identical_pcd):
        map_path = tmp_path / "map.pcd"
        map_reference = tmp_path / "map_ref.pcd"
        o3d.io.write_point_cloud(str(map_path), identical_pcd)
        o3d.io.write_point_cloud(str(map_reference), identical_pcd)
        trajectory_path = _write_csv_trajectory(
            tmp_path / "traj.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        trajectory_reference = _write_csv_trajectory(
            tmp_path / "traj_ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        result = evaluate_run(
            str(map_path),
            str(map_reference),
            trajectory_path,
            trajectory_reference,
            min_auc=0.95,
            max_ate=0.1,
            max_drift=0.1,
            min_coverage=1.0,
        )

        assert result["map"]["auc"] == pytest.approx(1.0)
        assert result["trajectory"]["ate"]["rmse"] == pytest.approx(0.0, abs=1e-8)
        assert result["overall_quality_gate"]["passed"] is True
        assert result["inspect"]["run_evaluate"].startswith("ca run-evaluate ")
        assert result["inspect"]["run_web"].startswith("ca web ")
        assert result["inspect"]["map"]["web_heatmap"].startswith("ca web ")
        assert result["inspect"]["trajectory"]["traj_evaluate_rigid"].startswith("ca traj-evaluate ")

    def test_overall_gate_can_fail(self, tmp_path, identical_pcd, shifted_pcd):
        map_path = tmp_path / "map.pcd"
        map_reference = tmp_path / "map_ref.pcd"
        o3d.io.write_point_cloud(str(map_path), shifted_pcd)
        o3d.io.write_point_cloud(str(map_reference), identical_pcd)
        trajectory_path = _write_csv_trajectory(
            tmp_path / "traj.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.3, 0.0, 0.0)],
        )
        trajectory_reference = _write_csv_trajectory(
            tmp_path / "traj_ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.0, 0.0, 0.0)],
        )

        result = evaluate_run(
            str(map_path),
            str(map_reference),
            trajectory_path,
            trajectory_reference,
            min_auc=0.95,
            max_chamfer=0.05,
            max_drift=0.2,
        )

        assert result["map"]["quality_gate"]["passed"] is False
        assert result["trajectory"]["quality_gate"]["passed"] is False
        assert result["overall_quality_gate"]["passed"] is False
        assert any("Map:" in reason for reason in result["overall_quality_gate"]["reasons"])
        assert any("Trajectory:" in reason for reason in result["overall_quality_gate"]["reasons"])


class TestRunReport:
    def test_markdown_and_html(self, tmp_path, identical_pcd):
        map_path = tmp_path / "map.pcd"
        map_reference = tmp_path / "map_ref.pcd"
        o3d.io.write_point_cloud(str(map_path), identical_pcd)
        o3d.io.write_point_cloud(str(map_reference), identical_pcd)
        trajectory_path = _write_csv_trajectory(
            tmp_path / "traj.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        trajectory_reference = _write_csv_trajectory(
            tmp_path / "traj_ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        result = evaluate_run(
            str(map_path),
            str(map_reference),
            trajectory_path,
            trajectory_reference,
            min_auc=0.95,
            max_ate=0.1,
            max_drift=0.1,
            min_coverage=1.0,
        )
        markdown_path = tmp_path / "run_report.md"
        html_path = tmp_path / "run_report.html"
        save_run_report(result, str(markdown_path))
        save_run_report(result, str(html_path))

        markdown = markdown_path.read_text()
        html = html_path.read_text()
        assert "# CloudAnalyzer Run Evaluation Report" in markdown
        assert "## Map Quality" in markdown
        assert "## Trajectory Quality" in markdown
        assert "Run viewer:" in markdown
        assert "Map heatmap:" in markdown
        assert "Trajectory rigid:" in markdown
        assert "<title>CloudAnalyzer Run Evaluation Report</title>" in html
        assert "Overall Summary" in html
        assert "Map Quality" in html
        assert "Trajectory Quality" in html
        assert "Map F1 curve" in html
        assert "Trajectory overlay plot" in html
        assert "ca web" in html
        assert "ca web" in html
        assert (tmp_path / "run_report_map_f1.png").exists()
        assert (tmp_path / "run_report_trajectory_overlay.png").exists()
        assert (tmp_path / "run_report_trajectory_errors.png").exists()
