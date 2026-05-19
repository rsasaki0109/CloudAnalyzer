"""Tests for the PR-comment Markdown renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ca.pr_comment import build_pr_comment
from cloudanalyzer_cli.main import app


REPO_ROOT = Path(__file__).resolve().parents[2]
PASS_SUMMARY = REPO_ROOT / "benchmarks/public/stanford-bunny-mini/expected/suite-pass.summary.json"
REGRESSION_SUMMARY = (
    REPO_ROOT / "benchmarks/public/stanford-bunny-mini/expected/suite-regression.summary.json"
)


@pytest.fixture
def pass_summary() -> dict:
    return json.loads(PASS_SUMMARY.read_text(encoding="utf-8"))


@pytest.fixture
def regression_summary() -> dict:
    return json.loads(REGRESSION_SUMMARY.read_text(encoding="utf-8"))


def test_check_suite_pass(pass_summary: dict) -> None:
    md = build_pr_comment(pass_summary)
    assert "## CloudAnalyzer QA: PASS" in md
    assert "4/4 passed, 0 failed" in md
    # No failed rows → no triage block
    assert "### Recommended triage" not in md


def test_check_suite_fail_lists_failed_checks(regression_summary: dict) -> None:
    md = build_pr_comment(regression_summary)
    assert "## CloudAnalyzer QA: FAIL" in md
    assert "0/4 passed, 4 failed" in md
    assert "`mapping-postprocess`" in md
    assert "`localization-run`" in md
    assert "### Recommended triage" in md
    # Reasons surfaced
    assert "AUC 0.5282 < min_auc 0.9700" in md


def test_check_suite_with_baseline_shows_deltas(
    regression_summary: dict, pass_summary: dict
) -> None:
    md = build_pr_comment(regression_summary, baseline=pass_summary)
    # AUC dropped → ↓ arrow next to it
    assert "AUC=0.5282 (was 1.0000 ↓)" in md
    # Chamfer went up → ↑ arrow
    assert "Chamfer=0.0394 (was 0.0015 ↑)" in md
    # Coverage formatted as percentage on both sides
    assert "Coverage=81.2% (was 100.0% ↓)" in md


def test_single_run_pass() -> None:
    payload = {
        "overall_quality_gate": {"passed": True, "reasons": []},
        "map": {
            "auc": 0.9975,
            "chamfer_distance": 0.0316,
            "best_f1": {"f1": 1.0},
        },
        "trajectory": {
            "ate": {"rmse": 0.0877},
            "rpe_translation": {"rmse": 0.1255},
            "drift": {"endpoint": 0.0964},
            "matching": {"coverage_ratio": 1.0},
        },
        "benchmark": {
            "suite": "synthetic-figure8",
            "version": 1,
            "sequence": "default",
        },
    }
    md = build_pr_comment(payload)
    assert "## CloudAnalyzer QA: PASS" in md
    assert "`synthetic-figure8`" in md
    assert "sequence: `default`" in md
    assert "Map AUC: 0.9975" in md
    assert "Trajectory ATE: 0.0877" in md
    # No failed gates section
    assert "**Failed gates:**" not in md


def test_single_run_fail_includes_reasons() -> None:
    payload = {
        "overall_quality_gate": {
            "passed": False,
            "reasons": ["Map: AUC 0.40 < min_auc 0.95"],
        },
        "map": {"auc": 0.4},
        "trajectory": {"ate": {"rmse": 0.6}},
    }
    md = build_pr_comment(payload)
    assert "## CloudAnalyzer QA: FAIL" in md
    assert "**Failed gates:**" in md
    assert "Map: AUC 0.40 < min_auc 0.95" in md


def test_single_run_baseline_delta() -> None:
    current = {
        "overall_quality_gate": {"passed": True, "reasons": []},
        "map": {"auc": 0.95, "chamfer_distance": 0.04},
        "trajectory": {"ate": {"rmse": 0.1}},
    }
    baseline = {
        "overall_quality_gate": {"passed": True, "reasons": []},
        "map": {"auc": 0.99, "chamfer_distance": 0.02},
        "trajectory": {"ate": {"rmse": 0.05}},
    }
    md = build_pr_comment(current, baseline=baseline)
    assert "Map AUC: 0.9500 (was 0.9900 ↓)" in md
    assert "Map Chamfer: 0.0400 (was 0.0200 ↑)" in md
    assert "Trajectory ATE: 0.1000 (was 0.0500 ↑)" in md


def test_unknown_shape_raises() -> None:
    with pytest.raises(ValueError, match="Unrecognized"):
        build_pr_comment({"unrelated": True})


def test_cli_writes_to_stdout() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["report-pr-comment", str(REGRESSION_SUMMARY)])
    assert result.exit_code == 0, result.output
    assert "## CloudAnalyzer QA: FAIL" in result.output


def test_cli_writes_to_file(tmp_path: Path) -> None:
    output = tmp_path / "pr.md"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "report-pr-comment",
            str(REGRESSION_SUMMARY),
            "--baseline",
            str(PASS_SUMMARY),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    md = output.read_text(encoding="utf-8")
    assert "## CloudAnalyzer QA: FAIL" in md
    # Baseline delta present in saved file
    assert "(was 1.0000 ↓)" in md


def test_cli_rejects_unknown_shape(tmp_path: Path) -> None:
    payload = tmp_path / "bad.json"
    payload.write_text(json.dumps({"random": "object"}), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["report-pr-comment", str(payload)])
    assert result.exit_code == 1
    assert "Unrecognized" in result.output or "Unrecognized" in result.stderr
