"""Tests for CLI interface."""

import json
from pathlib import Path
from textwrap import dedent

import pytest
from typer.testing import CliRunner

from ca.core import load_check_suite
from cli.main import app

runner = CliRunner()


def _write_csv_trajectory(path: Path, rows: list[tuple[float, float, float, float]]) -> str:
    lines = ["timestamp,x,y,z"]
    lines.extend(f"{timestamp},{x},{y},{z}" for timestamp, x, y, z in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _write_config(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")
    return str(path)


class TestCLI:
    def test_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "v0.1.0" in result.output

    def test_compare_basic(self, source_and_target_files):
        src, tgt = source_and_target_files
        result = runner.invoke(app, ["compare", src, tgt, "--register", "gicp"])
        assert result.exit_code == 0
        assert "Loading source" in result.output
        assert "Done." in result.output

    def test_compare_no_registration(self, source_and_target_files):
        src, tgt = source_and_target_files
        result = runner.invoke(app, ["compare", src, tgt, "--register", "none"])
        assert result.exit_code == 0
        assert "Registering" not in result.output

    def test_compare_with_json(self, source_and_target_files, tmp_path):
        src, tgt = source_and_target_files
        json_out = str(tmp_path / "cli_out.json")
        result = runner.invoke(app, ["compare", src, tgt, "--json", json_out])
        assert result.exit_code == 0
        assert (tmp_path / "cli_out.json").exists()

    def test_compare_file_not_found(self):
        result = runner.invoke(app, ["compare", "/no/file.pcd", "/no/other.pcd"])
        assert result.exit_code == 1

    def test_info(self, sample_pcd_file):
        result = runner.invoke(app, ["info", sample_pcd_file])
        assert result.exit_code == 0
        assert "Points:" in result.output
        assert "100" in result.output

    def test_info_file_not_found(self):
        result = runner.invoke(app, ["info", "/no/file.pcd"])
        assert result.exit_code == 1

    def test_diff(self, source_and_target_files):
        src, tgt = source_and_target_files
        result = runner.invoke(app, ["diff", src, tgt])
        assert result.exit_code == 0
        assert "Mean:" in result.output

    def test_diff_file_not_found(self):
        result = runner.invoke(app, ["diff", "/no/a.pcd", "/no/b.pcd"])
        assert result.exit_code == 1

    def test_compare_with_threshold(self, source_and_target_files):
        src, tgt = source_and_target_files
        result = runner.invoke(app, ["compare", src, tgt, "--register", "none", "--threshold", "0.05"])
        assert result.exit_code == 0
        assert "Threshold" in result.output

    def test_diff_with_threshold(self, source_and_target_files):
        src, tgt = source_and_target_files
        result = runner.invoke(app, ["diff", src, tgt, "--threshold", "0.05"])
        assert result.exit_code == 0
        assert "Exceed:" in result.output

    def test_downsample(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "down.pcd")
        result = runner.invoke(app, ["downsample", sample_pcd_file, "-o", output, "-v", "0.3"])
        assert result.exit_code == 0
        assert "Reduction:" in result.output

    def test_downsample_missing_output(self, sample_pcd_file):
        result = runner.invoke(app, ["downsample", sample_pcd_file])
        assert result.exit_code != 0

    def test_merge(self, source_and_target_files, tmp_path):
        src, tgt = source_and_target_files
        output = str(tmp_path / "merged.pcd")
        result = runner.invoke(app, ["merge", src, tgt, "-o", output])
        assert result.exit_code == 0
        assert "Total:" in result.output

    def test_convert(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "out.ply")
        result = runner.invoke(app, ["convert", sample_pcd_file, output])
        assert result.exit_code == 0
        assert ".pcd" in result.output
        assert ".ply" in result.output

    def test_crop(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "cropped.pcd")
        result = runner.invoke(app, [
            "crop", sample_pcd_file, "-o", output,
            "--x-min", "0", "--y-min", "0", "--z-min", "0",
            "--x-max", "0.5", "--y-max", "0.5", "--z-max", "0.5",
        ])
        assert result.exit_code == 0
        assert "Cropped:" in result.output

    def test_stats(self, sample_pcd_file):
        result = runner.invoke(app, ["stats", sample_pcd_file])
        assert result.exit_code == 0
        assert "Density:" in result.output
        assert "Spacing mean:" in result.output

    def test_normals(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "normals.ply")
        result = runner.invoke(app, ["normals", sample_pcd_file, "-o", output])
        assert result.exit_code == 0
        assert "Saved:" in result.output

    def test_filter(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "filtered.pcd")
        result = runner.invoke(app, ["filter", sample_pcd_file, "-o", output])
        assert result.exit_code == 0
        assert "Filtered:" in result.output

    def test_sample(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "sampled.pcd")
        result = runner.invoke(app, ["sample", sample_pcd_file, "-o", output, "-n", "50"])
        assert result.exit_code == 0
        assert "Sampled:" in result.output

    def test_align(self, source_and_target_files, tmp_path):
        src, tgt = source_and_target_files
        output = str(tmp_path / "aligned.pcd")
        result = runner.invoke(app, ["align", src, tgt, "-o", output])
        assert result.exit_code == 0
        assert "Total:" in result.output

    def test_batch(self, tmp_path, simple_pcd):
        import open3d as o3d
        o3d.io.write_point_cloud(str(tmp_path / "a.pcd"), simple_pcd)
        o3d.io.write_point_cloud(str(tmp_path / "b.pcd"), simple_pcd)
        result = runner.invoke(app, ["batch", str(tmp_path)])
        assert result.exit_code == 0
        assert "Total files: 2" in result.output

    def test_batch_evaluate(self, tmp_path, identical_pcd, shifted_pcd):
        import open3d as o3d

        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        reference = tmp_path / "reference.pcd"
        o3d.io.write_point_cloud(str(reference), identical_pcd)
        o3d.io.write_point_cloud(str(batch_dir / "a.pcd"), identical_pcd)
        o3d.io.write_point_cloud(str(batch_dir / "b.pcd"), shifted_pcd)

        result = runner.invoke(app, ["batch", str(batch_dir), "--evaluate", str(reference)])

        assert result.exit_code == 0
        assert "Reference:" in result.output
        assert "AUC=" in result.output
        assert "Best F1=" in result.output

    def test_batch_evaluate_format_json(self, tmp_path, identical_pcd):
        import open3d as o3d

        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        reference = tmp_path / "reference.pcd"
        o3d.io.write_point_cloud(str(reference), identical_pcd)
        o3d.io.write_point_cloud(str(batch_dir / "a.pcd"), identical_pcd)

        result = runner.invoke(
            app,
            ["batch", str(batch_dir), "--evaluate", str(reference), "--format-json"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["auc"] == 1.0
        assert data[0]["best_f1"]["f1"] == 1.0
        assert data[0]["inspect"]["web_heatmap"].startswith("ca web ")
        assert data[0]["inspect"]["heatmap3d"].startswith("ca heatmap3d ")

    def test_batch_evaluate_compression_metrics(self, tmp_path, identical_pcd):
        import open3d as o3d

        batch_dir = tmp_path / "decoded"
        compressed_dir = tmp_path / "compressed"
        baseline_dir = tmp_path / "original"
        batch_dir.mkdir()
        compressed_dir.mkdir()
        baseline_dir.mkdir()
        reference = tmp_path / "reference.pcd"
        o3d.io.write_point_cloud(str(reference), identical_pcd)
        o3d.io.write_point_cloud(str(batch_dir / "a.pcd"), identical_pcd)
        o3d.io.write_point_cloud(str(baseline_dir / "a.pcd"), identical_pcd)
        (compressed_dir / "a.cloudini").write_bytes(b"1234567890")

        result = runner.invoke(
            app,
            [
                "batch",
                str(batch_dir),
                "--evaluate",
                str(reference),
                "--compressed-dir",
                str(compressed_dir),
                "--baseline-dir",
                str(baseline_dir),
            ],
        )

        assert result.exit_code == 0
        assert "Size=" in result.output
        assert "Mean Size Ratio:" in result.output
        assert "Pareto Candidates:" in result.output
        assert "Recommended:" in result.output

    def test_batch_evaluate_compression_metrics_format_json(self, tmp_path, identical_pcd, shifted_pcd):
        import open3d as o3d

        batch_dir = tmp_path / "decoded"
        compressed_dir = tmp_path / "compressed"
        baseline_dir = tmp_path / "original"
        batch_dir.mkdir()
        compressed_dir.mkdir()
        baseline_dir.mkdir()
        reference = tmp_path / "reference.pcd"
        o3d.io.write_point_cloud(str(reference), identical_pcd)
        o3d.io.write_point_cloud(str(batch_dir / "a.pcd"), identical_pcd)
        o3d.io.write_point_cloud(str(batch_dir / "b.pcd"), shifted_pcd)
        o3d.io.write_point_cloud(str(baseline_dir / "a.pcd"), identical_pcd)
        o3d.io.write_point_cloud(str(baseline_dir / "b.pcd"), shifted_pcd)
        (compressed_dir / "a.cloudini").write_bytes(b"1234567890")
        (compressed_dir / "b.cloudini").write_bytes(b"123456789012345678901234567890")

        result = runner.invoke(
            app,
            [
                "batch",
                str(batch_dir),
                "--evaluate",
                str(reference),
                "--compressed-dir",
                str(compressed_dir),
                "--baseline-dir",
                str(baseline_dir),
                "--format-json",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        by_name = {Path(item["path"]).name: item for item in data}
        assert by_name["a.pcd"]["compression"]["pareto_optimal"] is True
        assert by_name["b.pcd"]["compression"]["pareto_optimal"] is False
        assert by_name["a.pcd"]["compression"]["recommended"] is True
        assert by_name["b.pcd"]["compression"]["recommended"] is False

    def test_batch_evaluate_report(self, tmp_path, identical_pcd):
        import open3d as o3d

        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        reference = tmp_path / "reference.pcd"
        report = tmp_path / "batch_report.md"
        o3d.io.write_point_cloud(str(reference), identical_pcd)
        o3d.io.write_point_cloud(str(batch_dir / "a.pcd"), identical_pcd)

        result = runner.invoke(
            app,
            ["batch", str(batch_dir), "--evaluate", str(reference), "--report", str(report)],
        )

        assert result.exit_code == 0
        assert report.exists()
        assert "CloudAnalyzer Batch Evaluation Report" in report.read_text()
        assert "Report:" in result.output

    def test_batch_evaluate_quality_gate(self, tmp_path, identical_pcd, shifted_pcd):
        import open3d as o3d

        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        reference = tmp_path / "reference.pcd"
        o3d.io.write_point_cloud(str(reference), identical_pcd)
        o3d.io.write_point_cloud(str(batch_dir / "a.pcd"), identical_pcd)
        o3d.io.write_point_cloud(str(batch_dir / "b.pcd"), shifted_pcd)

        result = runner.invoke(
            app,
            [
                "batch",
                str(batch_dir),
                "--evaluate",
                str(reference),
                "--min-auc",
                "0.95",
                "--max-chamfer",
                "0.02",
            ],
        )

        assert result.exit_code == 1
        assert "PASS" in result.output
        assert "FAIL" in result.output
        assert "Quality Gate: pass=1 fail=1" in result.output

    def test_batch_evaluate_quality_gate_format_json(self, tmp_path, identical_pcd):
        import open3d as o3d

        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        reference = tmp_path / "reference.pcd"
        o3d.io.write_point_cloud(str(reference), identical_pcd)
        o3d.io.write_point_cloud(str(batch_dir / "a.pcd"), identical_pcd)

        result = runner.invoke(
            app,
            [
                "batch",
                str(batch_dir),
                "--evaluate",
                str(reference),
                "--min-auc",
                "1.01",
                "--format-json",
            ],
        )

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data[0]["quality_gate"]["passed"] is False
        assert data[0]["inspect"]["web_heatmap"].startswith("ca web ")
        assert data[0]["inspect"]["heatmap3d"].startswith("ca heatmap3d ")

    def test_batch_thresholds_requires_evaluate(self, tmp_path):
        result = runner.invoke(app, ["batch", str(tmp_path), "--thresholds", "0.1,0.2"])

        assert result.exit_code == 1

    def test_batch_report_requires_evaluate(self, tmp_path):
        result = runner.invoke(app, ["batch", str(tmp_path), "--report", "batch.md"])

        assert result.exit_code == 1

    def test_batch_quality_gate_requires_evaluate(self, tmp_path):
        result = runner.invoke(app, ["batch", str(tmp_path), "--min-auc", "0.9"])

        assert result.exit_code == 1

    def test_batch_compressed_dir_requires_evaluate(self, tmp_path):
        result = runner.invoke(app, ["batch", str(tmp_path), "--compressed-dir", str(tmp_path)])

        assert result.exit_code == 1

    def test_traj_evaluate(self, tmp_path):
        estimated = _write_csv_trajectory(
            tmp_path / "estimated.csv",
            [(0.0, 0.1, 0.0, 0.0), (1.0, 1.1, 0.0, 0.0), (2.0, 2.1, 0.0, 0.0)],
        )
        reference = _write_csv_trajectory(
            tmp_path / "reference.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        result = runner.invoke(app, ["traj-evaluate", estimated, reference])

        assert result.exit_code == 0
        assert "Matched:" in result.output
        assert "ATE RMSE:" in result.output
        assert "RPE RMSE:" in result.output

    def test_traj_evaluate_format_json(self, tmp_path):
        estimated = _write_csv_trajectory(
            tmp_path / "estimated.csv",
            [(0.0, 0.1, 0.0, 0.0), (1.0, 1.1, 0.0, 0.0), (2.0, 2.1, 0.0, 0.0)],
        )
        reference = _write_csv_trajectory(
            tmp_path / "reference.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        result = runner.invoke(app, ["traj-evaluate", estimated, reference, "--format-json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["matching"]["matched_poses"] == 3
        assert data["ate"]["rmse"] == pytest.approx(0.1)

    def test_traj_evaluate_align_origin(self, tmp_path):
        estimated = _write_csv_trajectory(
            tmp_path / "estimated.csv",
            [(0.0, 5.0, -2.0, 1.0), (1.0, 6.0, -2.0, 1.0), (2.0, 7.0, -2.0, 1.0)],
        )
        reference = _write_csv_trajectory(
            tmp_path / "reference.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        result = runner.invoke(
            app,
            ["traj-evaluate", estimated, reference, "--align-origin", "--format-json"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["alignment"]["mode"] == "origin"
        assert data["ate"]["rmse"] == pytest.approx(0.0, abs=1e-8)

    def test_traj_evaluate_align_rigid(self, tmp_path):
        estimated = _write_csv_trajectory(
            tmp_path / "estimated.csv",
            [
                (0.0, 5.0, -3.0, 0.0),
                (1.0, 5.0, -2.0, 0.0),
                (2.0, 4.0, -2.0, 0.0),
                (3.0, 4.0, -1.0, 0.0),
            ],
        )
        reference = _write_csv_trajectory(
            tmp_path / "reference.csv",
            [
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 1.0, 0.0, 0.0),
                (2.0, 1.0, 1.0, 0.0),
                (3.0, 2.0, 1.0, 0.0),
            ],
        )

        result = runner.invoke(
            app,
            ["traj-evaluate", estimated, reference, "--align-rigid", "--format-json"],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["alignment"]["mode"] == "rigid"
        assert data["ate"]["rmse"] == pytest.approx(0.0, abs=1e-8)

    def test_traj_evaluate_report_and_quality_gate(self, tmp_path):
        estimated = _write_csv_trajectory(
            tmp_path / "estimated.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.5, 0.0, 0.0)],
        )
        reference = _write_csv_trajectory(
            tmp_path / "reference.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        report = tmp_path / "trajectory_report.html"

        result = runner.invoke(
            app,
            [
                "traj-evaluate",
                estimated,
                reference,
                "--max-ate",
                "0.1",
                "--report",
                str(report),
            ],
        )

        assert result.exit_code == 1
        assert "Quality Gate: FAIL" in result.output
        assert report.exists()
        assert "CloudAnalyzer Trajectory Evaluation Report" in report.read_text()

    def test_traj_evaluate_min_coverage_quality_gate(self, tmp_path):
        estimated = _write_csv_trajectory(
            tmp_path / "estimated.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )
        reference = _write_csv_trajectory(
            tmp_path / "reference.csv",
            [
                (0.0, 0.0, 0.0, 0.0),
                (0.5, 0.5, 0.0, 0.0),
                (1.0, 1.0, 0.0, 0.0),
                (1.5, 1.5, 0.0, 0.0),
                (2.0, 2.0, 0.0, 0.0),
            ],
        )

        result = runner.invoke(
            app,
            [
                "traj-evaluate",
                estimated,
                reference,
                "--max-time-delta",
                "0.05",
                "--min-coverage",
                "0.8",
                "--format-json",
            ],
        )

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["quality_gate"]["passed"] is False
        assert data["quality_gate"]["min_coverage"] == pytest.approx(0.8)
        assert any("Coverage" in reason for reason in data["quality_gate"]["reasons"])

    def test_traj_evaluate_max_drift_quality_gate(self, tmp_path):
        estimated = _write_csv_trajectory(
            tmp_path / "estimated.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.3, 0.0, 0.0)],
        )
        reference = _write_csv_trajectory(
            tmp_path / "reference.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.0, 0.0, 0.0)],
        )

        result = runner.invoke(
            app,
            [
                "traj-evaluate",
                estimated,
                reference,
                "--max-drift",
                "0.2",
                "--format-json",
            ],
        )

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["quality_gate"]["passed"] is False
        assert data["quality_gate"]["max_drift"] == pytest.approx(0.2)
        assert any("Endpoint Drift" in reason for reason in data["quality_gate"]["reasons"])

    def test_traj_batch(self, tmp_path):
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

        result = runner.invoke(
            app,
            ["traj-batch", str(estimated_dir), "--reference-dir", str(reference_dir)],
        )

        assert result.exit_code == 0
        assert "Mean ATE RMSE:" in result.output
        assert "Mean Coverage:" in result.output
        assert "Total files: 1" in result.output

    def test_traj_batch_align_rigid(self, tmp_path):
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

        result = runner.invoke(
            app,
            [
                "traj-batch",
                str(estimated_dir),
                "--reference-dir",
                str(reference_dir),
                "--align-rigid",
                "--format-json",
            ],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data[0]["alignment"]["mode"] == "rigid"
        assert data[0]["ate"]["rmse"] == pytest.approx(0.0, abs=1e-8)

    def test_traj_batch_format_json_and_quality_gate(self, tmp_path):
        estimated_dir = tmp_path / "estimated"
        reference_dir = tmp_path / "reference"
        _write_csv_trajectory(
            estimated_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.5, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            reference_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0)],
        )

        result = runner.invoke(
            app,
            [
                "traj-batch",
                str(estimated_dir),
                "--reference-dir",
                str(reference_dir),
                "--max-ate",
                "0.1",
                "--format-json",
            ],
        )

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data[0]["quality_gate"]["passed"] is False
        assert data[0]["inspect"]["traj_evaluate"].startswith("ca traj-evaluate ")

    def test_traj_batch_min_coverage_quality_gate(self, tmp_path):
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

        result = runner.invoke(
            app,
            [
                "traj-batch",
                str(estimated_dir),
                "--reference-dir",
                str(reference_dir),
                "--max-time-delta",
                "0.05",
                "--min-coverage",
                "0.8",
                "--format-json",
            ],
        )

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data[0]["quality_gate"]["passed"] is False
        assert data[0]["quality_gate"]["min_coverage"] == pytest.approx(0.8)
        assert any("Coverage" in reason for reason in data[0]["quality_gate"]["reasons"])

    def test_traj_batch_max_drift_quality_gate(self, tmp_path):
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

        result = runner.invoke(
            app,
            [
                "traj-batch",
                str(estimated_dir),
                "--reference-dir",
                str(reference_dir),
                "--max-drift",
                "0.2",
                "--format-json",
            ],
        )

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data[0]["quality_gate"]["passed"] is False
        assert data[0]["quality_gate"]["max_drift"] == pytest.approx(0.2)
        assert any("Endpoint Drift" in reason for reason in data[0]["quality_gate"]["reasons"])

    def test_run_evaluate(self, tmp_path, identical_pcd):
        import open3d as o3d

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
        report = tmp_path / "run_report.html"

        result = runner.invoke(
            app,
            [
                "run-evaluate",
                str(map_path),
                str(map_reference),
                trajectory_path,
                trajectory_reference,
                "--min-auc",
                "0.95",
                "--max-ate",
                "0.1",
                "--report",
                str(report),
            ],
        )

        assert result.exit_code == 0
        assert "Map:" in result.output
        assert "Trajectory:" in result.output
        assert "Overall Quality Gate: PASS" in result.output
        assert report.exists()

    def test_run_evaluate_format_json_and_quality_gate(self, tmp_path, identical_pcd, shifted_pcd):
        import open3d as o3d

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

        result = runner.invoke(
            app,
            [
                "run-evaluate",
                str(map_path),
                str(map_reference),
                trajectory_path,
                trajectory_reference,
                "--min-auc",
                "0.95",
                "--max-chamfer",
                "0.05",
                "--max-drift",
                "0.2",
                "--format-json",
            ],
        )

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data["map"]["quality_gate"]["passed"] is False
        assert data["trajectory"]["quality_gate"]["passed"] is False
        assert data["overall_quality_gate"]["passed"] is False
        assert any("Map:" in reason for reason in data["overall_quality_gate"]["reasons"])
        assert any("Trajectory:" in reason for reason in data["overall_quality_gate"]["reasons"])

    def test_run_batch(self, tmp_path, identical_pcd):
        import open3d as o3d

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

        result = runner.invoke(
            app,
            [
                "run-batch",
                str(map_dir),
                "--map-reference-dir",
                str(map_reference_dir),
                "--trajectory-dir",
                str(trajectory_dir),
                "--trajectory-reference-dir",
                str(trajectory_reference_dir),
            ],
        )

        assert result.exit_code == 0
        assert "Mean Map AUC:" in result.output
        assert "Mean Trajectory ATE RMSE:" in result.output
        assert "Total runs: 1" in result.output

    def test_run_batch_quality_gate_text_output(self, tmp_path, identical_pcd, shifted_pcd):
        import open3d as o3d

        map_dir = tmp_path / "maps"
        map_reference_dir = tmp_path / "map_refs"
        trajectory_dir = tmp_path / "trajs"
        trajectory_reference_dir = tmp_path / "traj_refs"
        map_dir.mkdir()
        map_reference_dir.mkdir()
        trajectory_dir.mkdir()
        trajectory_reference_dir.mkdir()

        o3d.io.write_point_cloud(str(map_dir / "a.pcd"), shifted_pcd)
        o3d.io.write_point_cloud(str(map_reference_dir / "a.pcd"), identical_pcd)
        _write_csv_trajectory(
            trajectory_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.3, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            trajectory_reference_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.0, 0.0, 0.0)],
        )

        result = runner.invoke(
            app,
            [
                "run-batch",
                str(map_dir),
                "--map-reference-dir",
                str(map_reference_dir),
                "--trajectory-dir",
                str(trajectory_dir),
                "--trajectory-reference-dir",
                str(trajectory_reference_dir),
                "--min-auc",
                "0.95",
                "--max-chamfer",
                "0.05",
                "--max-drift",
                "0.2",
            ],
        )

        assert result.exit_code == 1
        assert "Map=FAIL" in result.output
        assert "Trajectory=FAIL" in result.output
        assert "FAIL" in result.output
        assert "Quality Gate: pass=0 fail=1" in result.output
        assert "Map Failures: 1" in result.output
        assert "Trajectory Failures: 1" in result.output

    def test_run_batch_format_json_and_quality_gate(self, tmp_path, identical_pcd, shifted_pcd):
        import open3d as o3d

        map_dir = tmp_path / "maps"
        map_reference_dir = tmp_path / "map_refs"
        trajectory_dir = tmp_path / "trajs"
        trajectory_reference_dir = tmp_path / "traj_refs"
        map_dir.mkdir()
        map_reference_dir.mkdir()
        trajectory_dir.mkdir()
        trajectory_reference_dir.mkdir()

        o3d.io.write_point_cloud(str(map_dir / "a.pcd"), shifted_pcd)
        o3d.io.write_point_cloud(str(map_reference_dir / "a.pcd"), identical_pcd)
        _write_csv_trajectory(
            trajectory_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.3, 0.0, 0.0)],
        )
        _write_csv_trajectory(
            trajectory_reference_dir / "a.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0), (2.0, 2.0, 0.0, 0.0), (3.0, 3.0, 0.0, 0.0)],
        )

        result = runner.invoke(
            app,
            [
                "run-batch",
                str(map_dir),
                "--map-reference-dir",
                str(map_reference_dir),
                "--trajectory-dir",
                str(trajectory_dir),
                "--trajectory-reference-dir",
                str(trajectory_reference_dir),
                "--min-auc",
                "0.95",
                "--max-chamfer",
                "0.05",
                "--max-drift",
                "0.2",
                "--format-json",
            ],
        )

        assert result.exit_code == 1
        data = json.loads(result.output)
        assert data[0]["overall_quality_gate"]["passed"] is False
        assert any("Map:" in reason for reason in data[0]["overall_quality_gate"]["reasons"])
        assert any("Trajectory:" in reason for reason in data[0]["overall_quality_gate"]["reasons"])
        assert data[0]["inspect"]["run_evaluate"].startswith("ca run-evaluate ")

    def test_web_heatmap_flag(self, tmp_path, identical_pcd, monkeypatch):
        import cli.main as cli_main
        import open3d as o3d

        source = tmp_path / "source.pcd"
        target = tmp_path / "target.pcd"
        o3d.io.write_point_cloud(str(source), identical_pcd)
        o3d.io.write_point_cloud(str(target), identical_pcd)

        called = {}

        def fake_web_serve(
            paths,
            port,
            max_points,
            open_browser,
            heatmap,
            trajectory_path=None,
            trajectory_reference_path=None,
            trajectory_max_time_delta=0.05,
            trajectory_align_origin=False,
            trajectory_align_rigid=False,
        ):
            called["paths"] = paths
            called["port"] = port
            called["max_points"] = max_points
            called["open_browser"] = open_browser
            called["heatmap"] = heatmap
            called["trajectory_path"] = trajectory_path
            called["trajectory_reference_path"] = trajectory_reference_path
            called["trajectory_max_time_delta"] = trajectory_max_time_delta
            called["trajectory_align_origin"] = trajectory_align_origin
            called["trajectory_align_rigid"] = trajectory_align_rigid

        monkeypatch.setattr(cli_main, "web_serve", fake_web_serve)

        result = runner.invoke(
            app,
            ["web", str(source), str(target), "--heatmap", "--no-browser", "--port", "9000"],
        )

        assert result.exit_code == 0
        assert called["paths"] == [str(source), str(target)]
        assert called["port"] == 9000
        assert called["open_browser"] is False
        assert called["heatmap"] is True
        assert called["trajectory_path"] is None

    def test_web_trajectory_overlay_flags(self, tmp_path, identical_pcd, monkeypatch):
        import cli.main as cli_main
        import open3d as o3d

        source = tmp_path / "source.pcd"
        target = tmp_path / "target.pcd"
        trajectory = _write_csv_trajectory(
            tmp_path / "traj.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)],
        )
        trajectory_reference = _write_csv_trajectory(
            tmp_path / "traj_ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)],
        )
        o3d.io.write_point_cloud(str(source), identical_pcd)
        o3d.io.write_point_cloud(str(target), identical_pcd)

        called = {}

        def fake_web_serve(
            paths,
            port,
            max_points,
            open_browser,
            heatmap,
            trajectory_path=None,
            trajectory_reference_path=None,
            trajectory_max_time_delta=0.05,
            trajectory_align_origin=False,
            trajectory_align_rigid=False,
        ):
            called["paths"] = paths
            called["port"] = port
            called["max_points"] = max_points
            called["open_browser"] = open_browser
            called["heatmap"] = heatmap
            called["trajectory_path"] = trajectory_path
            called["trajectory_reference_path"] = trajectory_reference_path
            called["trajectory_max_time_delta"] = trajectory_max_time_delta
            called["trajectory_align_origin"] = trajectory_align_origin
            called["trajectory_align_rigid"] = trajectory_align_rigid

        monkeypatch.setattr(cli_main, "web_serve", fake_web_serve)

        result = runner.invoke(
            app,
            [
                "web",
                str(source),
                str(target),
                "--heatmap",
                "--trajectory",
                trajectory,
                "--trajectory-reference",
                trajectory_reference,
                "--trajectory-max-time-delta",
                "0.1",
                "--trajectory-align-rigid",
                "--no-browser",
            ],
        )

        assert result.exit_code == 0
        assert called["paths"] == [str(source), str(target)]
        assert called["heatmap"] is True
        assert called["trajectory_path"] == trajectory
        assert called["trajectory_reference_path"] == trajectory_reference
        assert called["trajectory_max_time_delta"] == pytest.approx(0.1)
        assert called["trajectory_align_origin"] is False
        assert called["trajectory_align_rigid"] is True

    def test_web_export_flags(self, tmp_path, identical_pcd, monkeypatch):
        import cli.main as cli_main
        import open3d as o3d

        source = tmp_path / "source.pcd"
        target = tmp_path / "target.pcd"
        output_dir = tmp_path / "site"
        trajectory = _write_csv_trajectory(
            tmp_path / "traj.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)],
        )
        trajectory_reference = _write_csv_trajectory(
            tmp_path / "traj_ref.csv",
            [(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)],
        )
        o3d.io.write_point_cloud(str(source), identical_pcd)
        o3d.io.write_point_cloud(str(target), identical_pcd)

        called = {}

        def fake_web_export_static_bundle(
            paths,
            output_dir,
            max_points,
            heatmap,
            trajectory_path=None,
            trajectory_reference_path=None,
            trajectory_max_time_delta=0.05,
            trajectory_align_origin=False,
            trajectory_align_rigid=False,
        ):
            called["paths"] = paths
            called["output_dir"] = output_dir
            called["max_points"] = max_points
            called["heatmap"] = heatmap
            called["trajectory_path"] = trajectory_path
            called["trajectory_reference_path"] = trajectory_reference_path
            called["trajectory_max_time_delta"] = trajectory_max_time_delta
            called["trajectory_align_origin"] = trajectory_align_origin
            called["trajectory_align_rigid"] = trajectory_align_rigid
            output_root = Path(output_dir)
            return {
                "output_dir": output_dir,
                "viewer_mode": "heatmap",
                "data_json": str(output_root / "data.json"),
                "chunk_count": 2,
                "display_points": 1234,
            }

        monkeypatch.setattr(cli_main, "web_export_static_bundle", fake_web_export_static_bundle)

        result = runner.invoke(
            app,
            [
                "web-export",
                str(source),
                str(target),
                "--output-dir",
                str(output_dir),
                "--heatmap",
                "--trajectory",
                trajectory,
                "--trajectory-reference",
                trajectory_reference,
                "--trajectory-align-origin",
            ],
        )

        assert result.exit_code == 0
        assert called["paths"] == [str(source), str(target)]
        assert called["output_dir"] == str(output_dir)
        assert called["heatmap"] is True
        assert called["trajectory_path"] == trajectory
        assert called["trajectory_reference_path"] == trajectory_reference
        assert called["trajectory_align_origin"] is True
        assert called["trajectory_align_rigid"] is False
        assert "Exported:" in result.output
        assert "Viewer mode:  heatmap" in result.output

    def test_density_map(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "density.png")
        result = runner.invoke(app, ["density-map", sample_pcd_file, "-o", output])
        assert result.exit_code == 0
        assert "Max density:" in result.output

    def test_info_format_json(self, sample_pcd_file):
        result = runner.invoke(app, ["info", sample_pcd_file, "--format-json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert data["num_points"] == 100

    def test_check_uses_default_cloudanalyzer_yaml(self, tmp_path, identical_pcd, monkeypatch):
        import open3d as o3d

        map_path = tmp_path / "map.pcd"
        map_reference = tmp_path / "map_ref.pcd"
        o3d.io.write_point_cloud(str(map_path), identical_pcd)
        o3d.io.write_point_cloud(str(map_reference), identical_pcd)
        _write_config(
            tmp_path / "cloudanalyzer.yaml",
            f"""
            defaults:
              report_dir: qa/reports
              json_dir: qa/results
            checks:
              - id: default-artifact
                kind: artifact
                source: {map_path.name}
                reference: {map_reference.name}
                gate:
                  min_auc: 0.95
                  max_chamfer: 0.01
            """,
        )

        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["check"])

        assert result.exit_code == 0
        assert "[PASS] default-artifact (artifact)" in result.output
        assert "Summary JSON:" not in result.output
        assert (tmp_path / "qa" / "reports" / "default-artifact.html").exists()
        assert (tmp_path / "qa" / "results" / "default-artifact.json").exists()

    def test_check_format_json_and_output_json(self, tmp_path, identical_pcd):
        import open3d as o3d

        map_path = tmp_path / "map.pcd"
        map_reference = tmp_path / "map_ref.pcd"
        o3d.io.write_point_cloud(str(map_path), identical_pcd)
        o3d.io.write_point_cloud(str(map_reference), identical_pcd)
        config = _write_config(
            tmp_path / "cloudanalyzer.yaml",
            f"""
            checks:
              - id: perception-output
                kind: artifact
                source: {map_path.name}
                reference: {map_reference.name}
                gate:
                  min_auc: 0.95
            """,
        )
        summary_json = tmp_path / "summary.json"

        result = runner.invoke(
            app,
            ["check", config, "--format-json", "--output-json", str(summary_json)],
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["summary"]["passed"] is True
        assert data["checks"][0]["id"] == "perception-output"
        assert summary_json.exists()

    def test_check_returns_exit_code_one_for_failed_gate(self, tmp_path, identical_pcd, shifted_pcd):
        import open3d as o3d

        map_path = tmp_path / "map.pcd"
        map_reference = tmp_path / "map_ref.pcd"
        o3d.io.write_point_cloud(str(map_path), shifted_pcd)
        o3d.io.write_point_cloud(str(map_reference), identical_pcd)
        config = _write_config(
            tmp_path / "cloudanalyzer.yaml",
            f"""
            checks:
              - id: failing-artifact
                kind: artifact
                source: {map_path.name}
                reference: {map_reference.name}
                gate:
                  min_auc: 0.99
                  max_chamfer: 0.01
            """,
        )

        result = runner.invoke(app, ["check", config])

        assert result.exit_code == 1
        assert "[FAIL] failing-artifact (artifact)" in result.output

    def test_init_check_writes_integrated_template(self, tmp_path):
        config_path = tmp_path / "cloudanalyzer.yaml"

        result = runner.invoke(app, ["init-check", str(config_path)])

        assert result.exit_code == 0
        assert config_path.exists()
        assert "Profile: integrated" in result.output

        suite = load_check_suite(str(config_path))
        assert suite.project == "localization-mapping-perception"
        assert [check.kind for check in suite.checks] == [
            "artifact",
            "trajectory",
            "artifact",
            "run",
        ]

    def test_init_check_writes_profile_template(self, tmp_path):
        config_path = tmp_path / "mapping.yaml"

        result = runner.invoke(
            app,
            ["init-check", str(config_path), "--profile", "mapping"],
        )

        assert result.exit_code == 0
        suite = load_check_suite(str(config_path))
        assert suite.project == "mapping-qa"
        assert len(suite.checks) == 1
        assert suite.checks[0].kind == "artifact"
        assert suite.checks[0].check_id == "mapping-postprocess"

    def test_init_check_refuses_to_overwrite_without_force(self, tmp_path):
        config_path = tmp_path / "cloudanalyzer.yaml"
        config_path.write_text("existing\n", encoding="utf-8")

        result = runner.invoke(app, ["init-check", str(config_path)])

        assert result.exit_code == 1
        assert "Refusing to overwrite existing file" in result.output
        assert config_path.read_text(encoding="utf-8") == "existing\n"

    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "CloudAnalyzer" in result.output
