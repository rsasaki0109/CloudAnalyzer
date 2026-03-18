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

    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "CloudAnalyzer" in result.output
