"""Tests for CLI interface."""

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


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

    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "CloudAnalyzer" in result.output
