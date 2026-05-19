"""Tests for ca.bundle (QA result archive)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ca.bundle import (
    BUNDLE_VERSION,
    diff_bundles,
    pack_bundle,
    render_diff_markdown,
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


# ---------------------------------------------------------------- diff


def test_diff_same_bundle_no_warnings(tmp_path: Path) -> None:
    summary = _make_check_summary(tmp_path)
    bundle = tmp_path / "qa.zip"
    pack_bundle(str(summary), str(bundle), project="p", git_commit="abc")
    diff = diff_bundles(str(bundle), str(bundle))
    assert diff["warnings"] == []
    assert diff["old"]["bundle_path"] == diff["new"]["bundle_path"]


def test_diff_metadata_mismatches(tmp_path: Path) -> None:
    summary = _make_check_summary(tmp_path)
    old_bundle = tmp_path / "old.zip"
    new_bundle = tmp_path / "new.zip"
    pack_bundle(
        str(summary),
        str(old_bundle),
        project="p",
        git_commit="abc",
        pr_number="1",
        notes={"dataset": "v1"},
    )
    pack_bundle(
        str(summary),
        str(new_bundle),
        project="p",
        git_commit="def",
        pr_number="2",
        notes={"dataset": "v2", "voxel": "0.05"},
    )
    diff = diff_bundles(str(old_bundle), str(new_bundle))
    msgs = "\n".join(diff["warnings"])
    assert "git_commit" in msgs
    assert "pr_number" in msgs
    assert "notes.dataset" in msgs
    assert "notes.voxel" in msgs  # only on the new side


def test_diff_rejects_mismatched_summary_kind(tmp_path: Path) -> None:
    check_summary = _make_check_summary(tmp_path)
    single = _make_single_run_summary(tmp_path)
    old_bundle = tmp_path / "old.zip"
    new_bundle = tmp_path / "new.zip"
    pack_bundle(str(check_summary), str(old_bundle))
    pack_bundle(str(single), str(new_bundle))
    with pytest.raises(ValueError, match="different summary_kind"):
        diff_bundles(str(old_bundle), str(new_bundle))


def test_diff_real_bundles_renders_delta_table(tmp_path: Path) -> None:
    """End-to-end: pack the bundled stanford fixtures and render a diff."""
    old_bundle = tmp_path / "pass.zip"
    new_bundle = tmp_path / "regression.zip"
    pack_bundle(str(SUITE_PASS), str(old_bundle), project="demo", git_commit="abc")
    pack_bundle(str(SUITE_REGRESSION), str(new_bundle), project="demo", git_commit="def")
    diff = diff_bundles(str(old_bundle), str(new_bundle))
    md = render_diff_markdown(diff)
    assert "## CloudAnalyzer Bundle Diff" in md
    # The PR-comment block should be embedded under the bundle header.
    assert "## CloudAnalyzer QA:" in md
    # AUC dropped pass -> regression, so the renderer should show ↓.
    assert "AUC=0.5282 (was 1.0000 ↓)" in md


def test_cli_bundle_diff(tmp_path: Path) -> None:
    old_bundle = tmp_path / "old.zip"
    new_bundle = tmp_path / "new.zip"
    pack_bundle(str(SUITE_PASS), str(old_bundle), project="p", git_commit="abc")
    pack_bundle(str(SUITE_REGRESSION), str(new_bundle), project="p", git_commit="def")

    runner = CliRunner()
    out_file = tmp_path / "diff.md"
    result = runner.invoke(
        app,
        ["bundle", "diff", str(old_bundle), str(new_bundle), "--output", str(out_file)],
    )
    assert result.exit_code == 0, result.output
    assert out_file.exists()
    md = out_file.read_text(encoding="utf-8")
    assert "## CloudAnalyzer Bundle Diff" in md
    assert "git_commit" in md  # mismatch warning rendered


def test_cli_bundle_diff_format_json(tmp_path: Path) -> None:
    summary = _make_check_summary(tmp_path)
    bundle = tmp_path / "qa.zip"
    pack_bundle(str(summary), str(bundle), project="p")
    runner = CliRunner()
    result = runner.invoke(
        app, ["bundle", "diff", str(bundle), str(bundle), "--format-json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["warnings"] == []
    assert payload["old"]["metadata"]["bundle_version"] == BUNDLE_VERSION
    assert payload["new"]["metadata"]["bundle_version"] == BUNDLE_VERSION
