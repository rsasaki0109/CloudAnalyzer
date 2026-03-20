"""Tests for ca.evaluate module."""

import pytest

from ca.evaluate import evaluate, _f1_at_threshold
import numpy as np


class TestF1AtThreshold:
    def test_perfect_match(self):
        dist = np.zeros(100)
        result = _f1_at_threshold(dist, dist, 0.1)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_no_match(self):
        dist = np.ones(100) * 10.0
        result = _f1_at_threshold(dist, dist, 0.1)
        assert result["f1"] == pytest.approx(0.0)

    def test_partial_match(self):
        dist_s2t = np.array([0.05, 0.05, 0.5, 0.5])
        dist_t2s = np.array([0.05, 0.5, 0.5])
        result = _f1_at_threshold(dist_s2t, dist_t2s, 0.1)
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(1.0 / 3.0)


class TestEvaluate:
    def test_identical_clouds(self, sample_pcd_file):
        result = evaluate(sample_pcd_file, sample_pcd_file)
        assert result["chamfer_distance"] == pytest.approx(0.0, abs=1e-8)
        assert result["auc"] == pytest.approx(1.0)
        for s in result["f1_scores"]:
            assert s["f1"] == pytest.approx(1.0)

    def test_shifted_clouds(self, source_and_target_files):
        src, tgt = source_and_target_files
        result = evaluate(src, tgt)
        assert result["chamfer_distance"] > 0
        assert result["hausdorff_distance"] > 0
        assert 0 <= result["auc"] <= 1.0
        assert len(result["f1_scores"]) == 6  # default thresholds

    def test_custom_thresholds(self, source_and_target_files):
        src, tgt = source_and_target_files
        result = evaluate(src, tgt, thresholds=[0.01, 0.05, 0.5])
        assert len(result["f1_scores"]) == 3

    def test_bidirectional_stats(self, source_and_target_files):
        src, tgt = source_and_target_files
        result = evaluate(src, tgt)
        assert "source_to_target" in result["distance_stats"]
        assert "target_to_source" in result["distance_stats"]

    def test_point_counts(self, source_and_target_files):
        src, tgt = source_and_target_files
        result = evaluate(src, tgt)
        assert result["source_points"] == 100
        assert result["target_points"] == 100

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            evaluate("/no/file.pcd", "/no/other.pcd")


class TestEvaluateCLI:
    def test_basic(self, source_and_target_files):
        from typer.testing import CliRunner
        from cli.main import app
        runner = CliRunner()
        src, tgt = source_and_target_files
        result = runner.invoke(app, ["evaluate", src, tgt])
        assert result.exit_code == 0
        assert "Chamfer Distance:" in result.output
        assert "Hausdorff Distance:" in result.output
        assert "AUC (F1):" in result.output
        assert "F1 Scores:" in result.output

    def test_custom_thresholds(self, source_and_target_files):
        from typer.testing import CliRunner
        from cli.main import app
        runner = CliRunner()
        src, tgt = source_and_target_files
        result = runner.invoke(app, ["evaluate", src, tgt, "-t", "0.01,0.1,1.0"])
        assert result.exit_code == 0
        assert "d=0.01" in result.output
        assert "d=0.10" in result.output
        assert "d=1.00" in result.output

    def test_format_json(self, source_and_target_files):
        from typer.testing import CliRunner
        from cli.main import app
        import json
        runner = CliRunner()
        src, tgt = source_and_target_files
        result = runner.invoke(app, ["evaluate", src, tgt, "--format-json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "chamfer_distance" in data
        assert "auc" in data
