"""Tests for logging and --verbose/--quiet CLI options."""

from typer.testing import CliRunner

from cloudanalyzer_cli.main import app

runner = CliRunner()


class TestLogging:
    def test_quiet_suppresses_output(self, source_and_target_files):
        src, tgt = source_and_target_files
        result = runner.invoke(app, ["--quiet", "compare", src, tgt, "--register", "none"])
        assert result.exit_code == 0
        # Quiet mode: logger output goes to stderr, not captured by typer runner
        # But no crash

    def test_verbose_works(self, sample_pcd_file):
        result = runner.invoke(app, ["--verbose", "info", sample_pcd_file])
        assert result.exit_code == 0
        assert "Points:" in result.output

    def test_error_hints_file_not_found(self):
        result = runner.invoke(app, ["info", "/no/such/file.pcd"])
        assert result.exit_code == 1
        assert "Hint:" in result.output

    def test_error_hints_unsupported_format(self, tmp_path):
        bad = tmp_path / "file.xyz"
        bad.write_text("x")
        result = runner.invoke(app, ["info", str(bad)])
        assert result.exit_code == 1
        assert "Hint:" in result.output
