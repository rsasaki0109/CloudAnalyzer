"""Tests for consolidated experiment-process docs generation."""

from pathlib import Path

from ca.experiments.process_docs import (
    build_experiment_reports,
    render_decisions_markdown,
    render_experiments_markdown,
    render_interfaces_markdown,
    write_process_docs,
)


class TestProcessDocs:
    def test_build_experiment_reports_returns_all_slices(self):
        reports = build_experiment_reports(repetitions=1)

        assert [report["problem"]["name"] for report in reports] == [
            "web_point_cloud_reduction",
            "web_trajectory_sampling",
            "web_progressive_loading",
            "check_scaffolding",
        ]

    def test_renderers_include_all_experiment_sections(self):
        reports = build_experiment_reports(repetitions=1)

        experiments_md = render_experiments_markdown(reports)
        decisions_md = render_decisions_markdown(reports)
        interfaces_md = render_interfaces_markdown(reports)

        assert "web_point_cloud_reduction" in experiments_md
        assert "web_trajectory_sampling" in experiments_md
        assert "web_progressive_loading" in experiments_md
        assert "check_scaffolding" in experiments_md
        assert "Trigger To Re-run" in decisions_md
        assert "cloudanalyzer/ca/core/web_sampling.py" in interfaces_md
        assert "cloudanalyzer/ca/core/web_trajectory_sampling.py" in interfaces_md
        assert "cloudanalyzer/ca/core/web_progressive_loading.py" in interfaces_md
        assert "cloudanalyzer/ca/core/check_scaffolding.py" in interfaces_md

    def test_write_process_docs_writes_consolidated_files(self, tmp_path: Path):
        reports = build_experiment_reports(repetitions=1)

        write_process_docs(reports, tmp_path)

        experiments_path = tmp_path / "experiments.md"
        decisions_path = tmp_path / "decisions.md"
        interfaces_path = tmp_path / "interfaces.md"

        assert experiments_path.exists()
        assert decisions_path.exists()
        assert interfaces_path.exists()
        assert "web_trajectory_sampling" in experiments_path.read_text(encoding="utf-8")
        assert "web_progressive_loading" in experiments_path.read_text(encoding="utf-8")
        assert "check_scaffolding" in experiments_path.read_text(encoding="utf-8")
        assert "web_point_cloud_reduction" in decisions_path.read_text(encoding="utf-8")
        assert "Stable interfaces keep only" in interfaces_path.read_text(encoding="utf-8")
