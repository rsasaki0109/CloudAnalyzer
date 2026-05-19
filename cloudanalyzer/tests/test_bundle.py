"""Tests for ca.bundle (QA result archive)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ca.bundle import (
    BUNDLE_VERSION,
    pack_bundle,
    show_bundle,
    unpack_bundle,
)
from cloudanalyzer_cli.main import app


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_REGRESSION = REPO_ROOT / "benchmarks/public/stanford-bunny-mini/expected/suite-regression.summary.json"
SUITE_PASS = REPO_ROOT / "benchmarks/public/stanford-bunny-mini/expected/suite-pass.summary.json"


# ---------------------------------------------------------------- helpers


def _make_check_summary(tmp_path: Path) -> Path:
    """Build a tiny check-suite summary with one referenced artifact."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    report = tmp_path / "report.html"
    report.write_text("<html>ok</html>", encoding="utf-8")
    json_artifact = tmp_path / "result.json"
    json_artifact.write_text(json.dumps({"ok": True}), encoding="utf-8")
    summary = {
        "project": "test-project",
        "summary": {"total_checks": 1, "failed_checks": 0, "passed": True},
        "checks": [
            {
                "id": "demo-check",
                "kind": "artifact",
                "passed": True,
                "report_path": str(report),
                "json_path": str(json_artifact),
                "summary": {"auc": 0.99, "chamfer_distance": 0.01, "passed": True},
            }
        ],
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def _make_single_run_summary(tmp_path: Path) -> Path:
    summary = {
        "overall_quality_gate": {"passed": True, "reasons": []},
        "map": {"auc": 0.99, "chamfer_distance": 0.01},
        "trajectory": {"ate": {"rmse": 0.1}},
    }
    path = tmp_path / "single.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    return path


# ---------------------------------------------------------------- pack


def test_pack_check_suite_collects_artifacts(tmp_path: Path) -> None:
    summary = _make_check_summary(tmp_path)
    bundle = tmp_path / "qa.zip"

    meta = pack_bundle(str(summary), str(bundle), project="proj-override")

    assert meta.bundle_version == BUNDLE_VERSION
    assert meta.summary_kind == "check_suite"
    assert meta.project == "proj-override"
    assert len(meta.artifacts) == 2  # report.html + result.json
    with zipfile.ZipFile(bundle) as zf:
        names = sorted(zf.namelist())
    assert "metadata.json" in names
    assert "summary.json" in names
    assert any("demo-check" in n and n.endswith(".html") for n in names)
    assert any("demo-check" in n and n.endswith(".json") and "metadata" not in n for n in names)


def test_pack_check_suite_with_baseline(tmp_path: Path) -> None:
    summary = _make_check_summary(tmp_path)
    baseline = _make_check_summary(tmp_path / "b")
    bundle = tmp_path / "qa.zip"
    meta = pack_bundle(str(summary), str(bundle), baseline_path=str(baseline))
    assert meta.has_baseline is True
    with zipfile.ZipFile(bundle) as zf:
        assert "baseline-summary.json" in zf.namelist()


def test_pack_single_run_no_artifacts(tmp_path: Path) -> None:
    summary = _make_single_run_summary(tmp_path)
    bundle = tmp_path / "qa.zip"
    meta = pack_bundle(str(summary), str(bundle))
    assert meta.summary_kind == "single_run"
    assert meta.artifacts == []


def test_pack_skips_missing_artifact_paths(tmp_path: Path) -> None:
    """Summary references a file that doesn't exist; bundle should still pack."""
    summary_data = {
        "project": "p",
        "summary": {"total_checks": 1, "failed_checks": 0, "passed": True},
        "checks": [
            {
                "id": "demo",
                "kind": "artifact",
                "passed": True,
                "report_path": "/nonexistent/report.html",
                "json_path": "/also/missing.json",
                "summary": {"auc": 1.0, "passed": True},
            }
        ],
    }
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(summary_data), encoding="utf-8")
    bundle = tmp_path / "qa.zip"
    meta = pack_bundle(str(summary), str(bundle))
    assert meta.artifacts == []
    with zipfile.ZipFile(bundle) as zf:
        assert "summary.json" in zf.namelist()


def test_pack_unknown_summary_shape_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"random": "object"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized"):
        pack_bundle(str(bad), str(tmp_path / "qa.zip"))


def test_pack_notes_and_metadata(tmp_path: Path) -> None:
    summary = _make_single_run_summary(tmp_path)
    bundle = tmp_path / "qa.zip"
    meta = pack_bundle(
        str(summary),
        str(bundle),
        project="rover-3",
        git_commit="deadbeef",
        pr_number="42",
        runner_id="gha-9876",
        notes={"dataset": "newer-college-mini", "voxel": "0.05"},
    )
    assert meta.git_commit == "deadbeef"
    assert meta.notes == {"dataset": "newer-college-mini", "voxel": "0.05"}
    with zipfile.ZipFile(bundle) as zf:
        raw = json.loads(zf.read("metadata.json").decode("utf-8"))
    assert raw["pr_number"] == "42"
    assert raw["notes"]["dataset"] == "newer-college-mini"


def test_pack_collision_disambiguates(tmp_path: Path) -> None:
    """Two checks pointing at the same filename should both end up in the bundle."""
    art = tmp_path / "report.html"
    art.write_text("x", encoding="utf-8")
    art2 = tmp_path / "second" / "report.html"
    art2.parent.mkdir()
    art2.write_text("y", encoding="utf-8")
    summary = {
        "project": "p",
        "summary": {"total_checks": 2, "failed_checks": 0, "passed": True},
        "checks": [
            {
                "id": "demo",
                "kind": "artifact",
                "passed": True,
                "report_path": str(art),
                "json_path": str(art2),  # same basename
                "summary": {"auc": 1.0, "passed": True},
            }
        ],
    }
    sp = tmp_path / "summary.json"
    sp.write_text(json.dumps(summary), encoding="utf-8")
    bundle = tmp_path / "qa.zip"
    meta = pack_bundle(str(sp), str(bundle))
    assert len({a.archive_path for a in meta.artifacts}) == 2


# ---------------------------------------------------------------- unpack / show


def test_round_trip(tmp_path: Path) -> None:
    summary = _make_check_summary(tmp_path)
    bundle = tmp_path / "qa.zip"
    pack_bundle(str(summary), str(bundle), project="p")
    extract_dir = tmp_path / "out"
    meta = unpack_bundle(str(bundle), str(extract_dir))
    assert meta.project == "p"
    assert (extract_dir / "summary.json").is_file()
    assert (extract_dir / "metadata.json").is_file()
    assert (extract_dir / "reports").is_dir()


def test_unpack_rejects_unsafe_paths(tmp_path: Path) -> None:
    """Bundle with `..` in an archive entry should error rather than escape the output dir."""
    summary = _make_single_run_summary(tmp_path)
    bundle = tmp_path / "qa.zip"
    pack_bundle(str(summary), str(bundle))
    # Tamper with the bundle to add a path-traversal entry.
    bad_bundle = tmp_path / "bad.zip"
    with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(bad_bundle, "w") as dst:
        for name in src.namelist():
            dst.writestr(name, src.read(name))
        dst.writestr("../escape.txt", b"oops")
    with pytest.raises(ValueError, match="unsafe archive entry"):
        unpack_bundle(str(bad_bundle), str(tmp_path / "out"))


def test_show_returns_metadata_and_toc(tmp_path: Path) -> None:
    summary = _make_check_summary(tmp_path)
    bundle = tmp_path / "qa.zip"
    pack_bundle(str(summary), str(bundle), project="p")
    info = show_bundle(str(bundle))
    assert info["metadata"]["bundle_version"] == BUNDLE_VERSION
    paths = {entry["path"] for entry in info["contents"]}
    assert "summary.json" in paths
    assert "metadata.json" in paths


def test_pack_real_check_summary(tmp_path: Path) -> None:
    """Use the bundled stanford-bunny-mini summary as a realistic fixture."""
    bundle = tmp_path / "qa.zip"
    meta = pack_bundle(
        str(SUITE_REGRESSION),
        str(bundle),
        baseline_path=str(SUITE_PASS),
        project="dogfood",
    )
    assert meta.summary_kind == "check_suite"
    # Real bundle should pull in the per-check reports/results files.
    assert any(a.archive_path.endswith(".html") for a in meta.artifacts)
    assert any(a.archive_path.endswith(".json") for a in meta.artifacts)


# ---------------------------------------------------------------- CLI


def test_cli_pack_show_unpack(tmp_path: Path) -> None:
    summary = _make_check_summary(tmp_path)
    bundle = tmp_path / "qa.zip"
    runner = CliRunner()

    # pack
    result = runner.invoke(
        app,
        [
            "bundle",
            "pack",
            str(summary),
            "--output",
            str(bundle),
            "--project",
            "proj",
            "--commit",
            "abc1234",
            "--note",
            "dataset=demo",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Kind:     check_suite" in result.output
    assert "Notes:" in result.output

    # show
    result = runner.invoke(app, ["bundle", "show", str(bundle)])
    assert result.exit_code == 0, result.output
    assert "Project:              proj" in result.output
    assert "abc1234" in result.output

    # unpack
    out_dir = tmp_path / "extracted"
    result = runner.invoke(
        app, ["bundle", "unpack", str(bundle), "--output", str(out_dir)]
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "summary.json").is_file()


def test_cli_pack_bad_note_format(tmp_path: Path) -> None:
    summary = _make_single_run_summary(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "bundle",
            "pack",
            str(summary),
            "--output",
            str(tmp_path / "qa.zip"),
            "--note",
            "no_equals_sign",
        ],
    )
    assert result.exit_code == 1
    assert "key=value" in result.output or "key=value" in result.stderr


def test_cli_show_json(tmp_path: Path) -> None:
    summary = _make_single_run_summary(tmp_path)
    bundle = tmp_path / "qa.zip"
    pack_bundle(str(summary), str(bundle), project="p")
    runner = CliRunner()
    result = runner.invoke(app, ["bundle", "show", str(bundle), "--format-json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["metadata"]["summary_kind"] == "single_run"
    assert isinstance(payload["contents"], list)
