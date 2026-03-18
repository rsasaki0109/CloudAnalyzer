"""Tests for --output-json option across CLI commands."""

import json

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


class TestOutputJson:
    def test_info_output_json(self, sample_pcd_file, tmp_path):
        json_path = str(tmp_path / "info.json")
        result = runner.invoke(app, ["info", sample_pcd_file, "--output-json", json_path])
        assert result.exit_code == 0
        data = json.loads(open(json_path).read())
        assert data["num_points"] == 100

    def test_diff_output_json(self, source_and_target_files, tmp_path):
        src, tgt = source_and_target_files
        json_path = str(tmp_path / "diff.json")
        result = runner.invoke(app, ["diff", src, tgt, "--output-json", json_path])
        assert result.exit_code == 0
        data = json.loads(open(json_path).read())
        assert "distance_stats" in data

    def test_downsample_output_json(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "down.pcd")
        json_path = str(tmp_path / "down.json")
        result = runner.invoke(app, ["downsample", sample_pcd_file, "-o", output, "--output-json", json_path])
        assert result.exit_code == 0
        data = json.loads(open(json_path).read())
        assert "reduction_ratio" in data

    def test_stats_output_json(self, sample_pcd_file, tmp_path):
        json_path = str(tmp_path / "stats.json")
        result = runner.invoke(app, ["stats", sample_pcd_file, "--output-json", json_path])
        assert result.exit_code == 0
        data = json.loads(open(json_path).read())
        assert "density" in data

    def test_filter_output_json(self, sample_pcd_file, tmp_path):
        output = str(tmp_path / "filtered.pcd")
        json_path = str(tmp_path / "filter.json")
        result = runner.invoke(app, ["filter", sample_pcd_file, "-o", output, "--output-json", json_path])
        assert result.exit_code == 0
        data = json.loads(open(json_path).read())
        assert "removed_points" in data
