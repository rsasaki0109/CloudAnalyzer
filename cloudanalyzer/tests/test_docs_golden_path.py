"""Regression checks for the public golden-path documentation."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
SLAM_TUTORIAL = REPO_ROOT / "docs" / "tutorial-slam-benchmark.md"
SLAM_SMOKE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "slam-benchmark-smoke.yml"


def test_readme_leads_with_the_web_viewer_and_links_the_cli():
    text = README.read_text(encoding="utf-8")

    assert "Point cloud viewer and analyzer that runs in your browser." in text
    assert "https://rsasaki0109.github.io/CloudAnalyzer/app/" in text
    assert (REPO_ROOT / "docs/images/web-viewer.png").is_file()
    assert (REPO_ROOT / "docs/images/social-preview.png").is_file()
    # The CI quality-gate CLI stays one click away, with its golden path.
    assert "[`cloudanalyzer/`](cloudanalyzer/)" in text
    cli = (REPO_ROOT / "cloudanalyzer" / "README.md").read_text(encoding="utf-8")
    assert "docs/tutorial-slam-benchmark.md" in cli


def test_slam_benchmark_tutorial_uses_checked_in_smoke_suite():
    text = SLAM_TUTORIAL.read_text(encoding="utf-8")

    assert "ca benchmark info benchmarks/slam/synthetic-figure8/suite.yaml" in text
    assert "--map benchmarks/slam/synthetic-figure8/sample_outputs/map_pass.pcd" in text
    assert "--trajectory benchmarks/slam/synthetic-figure8/sample_outputs/trajectory_pass.tum" in text
    assert "--out qa/synthetic-figure8" in text
    assert "qa/synthetic-figure8/metrics.json" in text
    assert "cloudanalyzer.slam_run_drivers" in text

    assert (REPO_ROOT / "benchmarks/slam/synthetic-figure8/suite.yaml").is_file()
    assert (REPO_ROOT / "benchmarks/slam/synthetic-figure8/sample_outputs/map_pass.pcd").is_file()
    assert (REPO_ROOT / "benchmarks/slam/synthetic-figure8/sample_outputs/trajectory_pass.tum").is_file()


def test_slam_benchmark_smoke_workflow_runs_readme_golden_path():
    text = SLAM_SMOKE_WORKFLOW.read_text(encoding="utf-8")

    assert "name: SLAM Benchmark Smoke" in text
    assert "ca benchmark info benchmarks/slam/synthetic-figure8/suite.yaml" in text
    assert "ca benchmark eval benchmarks/slam/synthetic-figure8/suite.yaml" in text
    assert "--map benchmarks/slam/synthetic-figure8/sample_outputs/map_pass.pcd" in text
    assert "--trajectory benchmarks/slam/synthetic-figure8/sample_outputs/trajectory_pass.tum" in text
    assert "--out qa/synthetic-figure8" in text
    assert "qa/synthetic-figure8/manifest.lock.yaml" in text
    assert "qa/synthetic-figure8/metrics.json" in text
    assert "qa/synthetic-figure8/provenance.json" in text
    assert "qa/synthetic-figure8/report.html" in text
    assert "qa/synthetic-figure8/summary.md" in text
    assert "overall_quality_gate" in text
    assert "actions/upload-artifact@v6" in text
