"""Tests for ca.plot module."""

from pathlib import Path

from ca.evaluate import evaluate
from ca.plot import heatmap3d, plot_multi_f1, plot_quality_vs_size


class TestPlotMultiF1:
    def test_creates_plot(self, source_and_target_files, tmp_path):
        src, tgt = source_and_target_files
        r1 = evaluate(src, tgt, thresholds=[0.05, 0.1, 0.5])
        r2 = evaluate(src, src, thresholds=[0.05, 0.1, 0.5])
        output = str(tmp_path / "multi_f1.png")
        plot_multi_f1([r1, r2], ["shifted", "identical"], output)
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0


class TestHeatmap3d:
    def test_creates_snapshot(self, source_and_target_files, tmp_path):
        src, tgt = source_and_target_files
        output = str(tmp_path / "heatmap.png")
        result = heatmap3d(src, tgt, output)
        assert Path(output).exists()
        assert result["num_points"] == 100
        assert result["mean_distance"] > 0


class TestPlotQualityVsSize:
    def test_creates_plot(self, tmp_path):
        results = [
            {
                "path": "a.pcd",
                "auc": 0.99,
                "compression": {"size_ratio": 0.2, "pareto_optimal": True},
                "quality_gate": {"passed": True},
            },
            {
                "path": "b.pcd",
                "auc": 0.8,
                "compression": {"size_ratio": 0.5, "pareto_optimal": False},
                "quality_gate": {"passed": False},
            },
        ]
        output = str(tmp_path / "quality_vs_size.png")
        plot_quality_vs_size(results, output)
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0


class TestHeatmap3dCLI:
    def test_basic(self, source_and_target_files, tmp_path):
        from typer.testing import CliRunner
        from cli.main import app
        runner = CliRunner()
        src, tgt = source_and_target_files
        output = str(tmp_path / "heatmap.png")
        result = runner.invoke(app, ["heatmap3d", src, tgt, "-o", output])
        assert result.exit_code == 0
        assert "Mean distance:" in result.output
