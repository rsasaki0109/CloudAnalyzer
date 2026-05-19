"""Tests for the QA history time-series view (`ca history`)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ca.bundle import pack_bundle
from ca.history import (
    HISTORY_VERSION,
    HistoryEntry,
    build_history_series,
    discover_bundles,
    extract_history_entry,
    render_history_json,
    render_history_markdown,
)
from cloudanalyzer_cli.main import app


# ---------------------------------------------------------------- helpers


def _make_check_summary(
    tmp_path: Path,
    *,
    auc: float = 0.99,
    chamfer: float = 0.01,
    passed: bool = True,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    report = tmp_path / "report.html"
    report.write_text("<html>ok</html>", encoding="utf-8")
    summary = {
        "project": "test-project",
        "summary": {
            "total_checks": 1,
            "failed_checks": 0 if passed else 1,
            "passed_checks": 1 if passed else 0,
            "passed": passed,
        },
        "checks": [
            {
                "id": "demo-check",
                "kind": "artifact",
                "passed": passed,
                "report_path": str(report),
                "summary": {
                    "auc": auc,
                    "chamfer_distance": chamfer,
                    "passed": passed,
                },
            }
        ],
    }
    out = tmp_path / "summary.json"
    out.write_text(json.dumps(summary), encoding="utf-8")
    return out


def _make_single_run_summary(
    tmp_path: Path,
    *,
    auc: float = 0.95,
    ate: float = 0.10,
    drift: float = 0.05,
    passed: bool = True,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    summary = {
        "overall_quality_gate": {"passed": passed, "reasons": []},
        "map": {
            "auc": auc,
            "chamfer_distance": 0.05,
            "best_f1": {"f1": 0.88},
        },
        "trajectory": {
            "ate": {"rmse": ate},
            "rpe_translation": {"rmse": 0.02},
            "drift": {"endpoint": drift},
            "matching": {"coverage_ratio": 0.94},
        },
    }
    out = tmp_path / "single.json"
    out.write_text(json.dumps(summary), encoding="utf-8")
    return out


def _pack_with_created_at(
    summary_path: Path,
    bundle_path: Path,
    *,
    created_at: str,
    git_commit: str | None = None,
    pr_number: str | None = None,
    project: str | None = None,
) -> None:
    """Pack a bundle, then rewrite metadata.created_at so we control ordering."""
    pack_bundle(
        str(summary_path),
        str(bundle_path),
        project=project,
        git_commit=git_commit,
        pr_number=pr_number,
    )
    # Open the zip and rewrite metadata.json with the desired created_at.
    with zipfile.ZipFile(bundle_path, mode="r") as zf:
        with zf.open("metadata.json") as fp:
            metadata = json.loads(io.TextIOWrapper(fp, encoding="utf-8").read())
        other_files = {
            name: zf.read(name) for name in zf.namelist() if name != "metadata.json"
        }
    metadata["created_at"] = created_at
    with zipfile.ZipFile(bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(metadata))
        for name, payload in other_files.items():
            zf.writestr(name, payload)


# ---------------------------------------------------------------- extract


def test_extract_single_run_entry(tmp_path: Path) -> None:
    summary = _make_single_run_summary(tmp_path, auc=0.91, ate=0.20)
    bundle = tmp_path / "qa.zip"
    _pack_with_created_at(
        summary, bundle, created_at="2026-05-19T10:00:00+00:00", git_commit="abc12345deadbeef"
    )
    entry = extract_history_entry(bundle)
    assert isinstance(entry, HistoryEntry)
    assert entry.summary_kind == "single_run"
    assert entry.overall_passed is True
    assert entry.git_commit == "abc12345deadbeef"
    assert entry.metrics["map.auc"] == pytest.approx(0.91)
    assert entry.metrics["trajectory.ate.rmse"] == pytest.approx(0.20)
    assert entry.metrics["trajectory.matching.coverage_ratio"] == pytest.approx(0.94)


def test_extract_check_suite_entry(tmp_path: Path) -> None:
    summary = _make_check_summary(tmp_path, auc=0.97, chamfer=0.012)
    bundle = tmp_path / "suite.zip"
    _pack_with_created_at(
        summary, bundle, created_at="2026-05-19T11:00:00+00:00", project="test-project"
    )
    entry = extract_history_entry(bundle)
    assert entry.summary_kind == "check_suite"
    assert entry.overall_passed is True
    assert "demo-check" in entry.per_check_passed
    assert entry.per_check_kind["demo-check"] == "artifact"
    assert entry.per_check_metrics["demo-check"]["auc"] == pytest.approx(0.97)
    assert entry.per_check_metrics["demo-check"]["chamfer_distance"] == pytest.approx(0.012)


def test_extract_missing_bundle_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_history_entry(tmp_path / "does-not-exist.zip")


# ---------------------------------------------------------------- build_history_series


def test_build_series_sorts_by_created_at(tmp_path: Path) -> None:
    paths: list[Path] = []
    for idx, (ts, auc) in enumerate(
        [
            ("2026-05-19T12:00:00+00:00", 0.85),
            ("2026-05-17T12:00:00+00:00", 0.95),  # oldest
            ("2026-05-18T12:00:00+00:00", 0.90),
        ]
    ):
        sub = tmp_path / f"s{idx}"
        summary = _make_single_run_summary(sub, auc=auc)
        bundle = sub / "qa.zip"
        _pack_with_created_at(summary, bundle, created_at=ts)
        paths.append(bundle)
    entries = build_history_series([str(p) for p in paths])
    assert [e.created_at for e in entries] == [
        "2026-05-17T12:00:00+00:00",
        "2026-05-18T12:00:00+00:00",
        "2026-05-19T12:00:00+00:00",
    ]
    # AUC trend: 0.95 → 0.90 → 0.85 (regressing).
    assert [e.metrics["map.auc"] for e in entries] == pytest.approx([0.95, 0.90, 0.85])


def test_build_series_rejects_mixed_kinds(tmp_path: Path) -> None:
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    single = _make_single_run_summary(a_dir)
    suite = _make_check_summary(b_dir)
    a_bundle = a_dir / "a.zip"
    b_bundle = b_dir / "b.zip"
    _pack_with_created_at(single, a_bundle, created_at="2026-05-18T00:00:00+00:00")
    _pack_with_created_at(suite, b_bundle, created_at="2026-05-19T00:00:00+00:00")
    with pytest.raises(ValueError, match="mixed summary_kinds"):
        build_history_series([str(a_bundle), str(b_bundle)])


def test_build_series_requires_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_history_series([])


# ---------------------------------------------------------------- rendering


def test_render_markdown_single_run(tmp_path: Path) -> None:
    paths: list[Path] = []
    for idx, (ts, auc, passed) in enumerate(
        [
            ("2026-05-19T08:00:00+00:00", 0.92, True),
            ("2026-05-19T09:00:00+00:00", 0.84, False),
        ]
    ):
        sub = tmp_path / f"s{idx}"
        summary = _make_single_run_summary(sub, auc=auc, passed=passed)
        bundle = sub / "qa.zip"
        _pack_with_created_at(
            summary, bundle, created_at=ts, git_commit=f"commit{idx:08d}"
        )
        paths.append(bundle)
    entries = build_history_series([str(p) for p in paths])
    md = render_history_markdown(entries)
    assert "CloudAnalyzer QA history" in md
    assert "**Bundles**: 2 (single_run)" in md
    # Old row first.
    assert md.index("2026-05-19T08:00:00") < md.index("2026-05-19T09:00:00")
    # PASS for first, FAIL for second.
    assert "✅" in md and "❌" in md
    # Metric columns present.
    assert "Map AUC" in md
    assert "Traj ATE" in md


def test_render_markdown_check_suite_section_per_check(tmp_path: Path) -> None:
    paths: list[Path] = []
    for idx, (ts, auc, passed) in enumerate(
        [
            ("2026-05-19T01:00:00+00:00", 0.98, True),
            ("2026-05-19T02:00:00+00:00", 0.94, True),
            ("2026-05-19T03:00:00+00:00", 0.80, False),
        ]
    ):
        sub = tmp_path / f"s{idx}"
        summary = _make_check_summary(sub, auc=auc, passed=passed)
        bundle = sub / "qa.zip"
        _pack_with_created_at(summary, bundle, created_at=ts, project="test-project")
        paths.append(bundle)
    entries = build_history_series([str(p) for p in paths])
    md = render_history_markdown(entries)
    assert "**Bundles**: 3 (check_suite)" in md
    assert "**Project**: `test-project`" in md
    assert "### `demo-check` (artifact)" in md
    # Order: oldest → newest.
    pos0 = md.index("2026-05-19T01:00:00")
    pos1 = md.index("2026-05-19T02:00:00")
    pos2 = md.index("2026-05-19T03:00:00")
    assert pos0 < pos1 < pos2


def test_render_markdown_empty() -> None:
    md = render_history_markdown([])
    assert "_No bundles supplied._" in md


def test_render_json_payload(tmp_path: Path) -> None:
    sub = tmp_path / "s"
    summary = _make_single_run_summary(sub, auc=0.90)
    bundle = sub / "qa.zip"
    _pack_with_created_at(summary, bundle, created_at="2026-05-19T00:00:00+00:00")
    entries = build_history_series([str(bundle)])
    payload = render_history_json(entries)
    assert payload["history_version"] == HISTORY_VERSION
    assert payload["summary_kind"] == "single_run"
    assert len(payload["entries"]) == 1
    assert payload["entries"][0]["metrics"]["map.auc"] == pytest.approx(0.90)


# ---------------------------------------------------------------- discover


def test_discover_bundles_returns_sorted(tmp_path: Path) -> None:
    for name in ["b.zip", "a.zip", "c.zip"]:
        (tmp_path / name).write_bytes(b"placeholder")
    (tmp_path / "ignored.txt").write_text("nope", encoding="utf-8")
    found = discover_bundles(tmp_path)
    assert [p.name for p in found] == ["a.zip", "b.zip", "c.zip"]


def test_discover_bundles_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        discover_bundles(tmp_path / "nope")


# ---------------------------------------------------------------- CLI


def test_cli_history_renders_markdown(tmp_path: Path) -> None:
    paths: list[Path] = []
    for idx, ts in enumerate(
        ["2026-05-19T05:00:00+00:00", "2026-05-19T06:00:00+00:00"]
    ):
        sub = tmp_path / f"s{idx}"
        summary = _make_single_run_summary(sub, auc=0.90 + idx * 0.01)
        bundle = sub / "qa.zip"
        _pack_with_created_at(summary, bundle, created_at=ts)
        paths.append(bundle)
    runner = CliRunner()
    result = runner.invoke(app, ["history", *[str(p) for p in paths]])
    assert result.exit_code == 0, result.output
    assert "CloudAnalyzer QA history" in result.output
    assert "Map AUC" in result.output


def test_cli_history_format_json(tmp_path: Path) -> None:
    sub = tmp_path / "s"
    summary = _make_single_run_summary(sub)
    bundle = sub / "qa.zip"
    _pack_with_created_at(summary, bundle, created_at="2026-05-19T00:00:00+00:00")
    runner = CliRunner()
    result = runner.invoke(app, ["history", str(bundle), "--format-json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["history_version"] == HISTORY_VERSION
    assert payload["summary_kind"] == "single_run"


def test_cli_history_from_dir(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundles"
    bundle_dir.mkdir()
    for idx, ts in enumerate(
        ["2026-05-19T07:00:00+00:00", "2026-05-19T08:00:00+00:00"]
    ):
        sub = tmp_path / f"src{idx}"
        summary = _make_single_run_summary(sub, auc=0.85 + idx * 0.02)
        bundle = bundle_dir / f"qa-{idx}.zip"
        _pack_with_created_at(summary, bundle, created_at=ts)
    runner = CliRunner()
    result = runner.invoke(app, ["history", "--from-dir", str(bundle_dir)])
    assert result.exit_code == 0, result.output
    assert "**Bundles**: 2 (single_run)" in result.output


def test_cli_history_requires_input(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["history"])
    assert result.exit_code != 0
    assert "provide at least one bundle path" in result.output


def test_cli_history_write_to_output(tmp_path: Path) -> None:
    sub = tmp_path / "s"
    summary = _make_single_run_summary(sub)
    bundle = sub / "qa.zip"
    _pack_with_created_at(summary, bundle, created_at="2026-05-19T00:00:00+00:00")
    out = tmp_path / "report.md"
    runner = CliRunner()
    result = runner.invoke(app, ["history", str(bundle), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "CloudAnalyzer QA history" in out.read_text(encoding="utf-8")
