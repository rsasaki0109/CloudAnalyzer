"""Tests for static leaderboard generation from benchmark report bundles."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from ca.leaderboard import LEADERBOARD_SCHEMA_VERSION, build_leaderboard_from_bundles
from cloudanalyzer_cli.main import app


def _write_bundle(
    root: Path,
    name: str,
    *,
    method: str | None = None,
    gate: dict | None = None,
    passed: bool = True,
) -> Path:
    bundle = root / name
    bundle.mkdir(parents=True)
    gate = gate or {"max_ate": 0.3, "min_auc": 0.95}
    metrics = {
        "benchmark": {
            "suite": "synthetic-figure8",
            "version": 1,
            "sequence": "default",
            "gate": gate,
        },
        "map": {
            "auc": 0.997,
            "chamfer_distance": 0.031,
        },
        "trajectory": {
            "ate": {"rmse": 0.08},
            "rpe_translation": {"rmse": 0.11},
            "drift": {"endpoint": 0.09},
            "matching": {"coverage_ratio": 1.0},
        },
        "overall_quality_gate": {
            "passed": passed,
            "reasons": [] if passed else ["ATE 0.0800 > max_ate 0.0500"],
        },
    }
    provenance = {
        "schema_version": "cloudanalyzer.benchmark_report_bundle.v0.1",
        "cloudanalyzer_version": "0.4.0",
        "summary_kind": "benchmark_run",
        "parameters": {"voxel_size": "0.5"},
    }
    if method is not None:
        provenance["method"] = method
    manifest_lock = {
        "schema_version": "cloudanalyzer.benchmark_report_bundle.v0.1",
        "suite": {
            "name": "synthetic-figure8",
            "version": 1,
            "sequence": "default",
            "source_path": "benchmarks/slam/synthetic-figure8/suite.yaml",
        },
        "gate": gate,
        "inputs": {
            "candidate_map": {
                "path": "qa/run/map.ply",
                "sha256": "a" * 64,
                "size_bytes": 100,
            },
            "candidate_trajectory": {
                "path": "qa/run/trajectory.tum",
                "sha256": "b" * 64,
                "size_bytes": 200,
            },
        },
        "outputs": {
            "metrics": "metrics.json",
            "summary": "summary.md",
            "report": "report.html",
            "provenance": "provenance.json",
            "manifest_lock": "manifest.lock.yaml",
        },
    }
    (bundle / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (bundle / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (bundle / "manifest.lock.yaml").write_text(
        yaml.safe_dump(manifest_lock, sort_keys=False),
        encoding="utf-8",
    )
    (bundle / "summary.md").write_text("# Summary\n", encoding="utf-8")
    (bundle / "report.html").write_text("<!doctype html><title>Report</title>\n", encoding="utf-8")
    return bundle


def test_build_leaderboard_from_bundles_copies_static_site(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path, "kiss-icp__synthetic-figure8")
    site = tmp_path / "site"

    payload = build_leaderboard_from_bundles([bundle], site)

    assert payload["schema_version"] == LEADERBOARD_SCHEMA_VERSION
    assert payload["errors"] == []
    assert payload["warnings"] == []
    assert payload["rows"][0]["method"] == "kiss-icp"
    assert payload["rows"][0]["dataset"] == "synthetic-figure8"
    assert payload["rows"][0]["gate_status"] == "pass"
    assert payload["rows"][0]["metrics"]["ate_rmse_m"] == 0.08
    assert payload["rows"][0]["links"]["report_html"] == "runs/kiss-icp__synthetic-figure8/report.html"
    assert payload["rows"][0]["artifact_hash"]

    assert (site / "results.json").is_file()
    assert (site / "index.html").is_file()
    assert (site / "runs" / "kiss-icp__synthetic-figure8" / "report.html").is_file()
    assert str(tmp_path) not in (site / "results.json").read_text(encoding="utf-8")


def test_build_leaderboard_warns_on_incomparable_rows(tmp_path: Path) -> None:
    first = _write_bundle(tmp_path, "method-a", gate={"max_ate": 0.3})
    second = _write_bundle(tmp_path, "method-b", gate={"max_ate": 0.1})

    payload = build_leaderboard_from_bundles([first, second], tmp_path / "site")

    assert payload["errors"] == []
    assert payload["warnings"][0]["code"] == "incomparable_rows"
    assert payload["warnings"][0]["run_ids"] == ["method-a", "method-b"]


def test_leaderboard_build_cli_outputs_json(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path, "kiss-icp__synthetic-figure8", method="kiss-icp")
    site = tmp_path / "site"
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "leaderboard",
            "build",
            str(bundle),
            "--out",
            str(site),
            "--format-json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["schema_version"] == LEADERBOARD_SCHEMA_VERSION
    assert payload["rows"][0]["method"] == "kiss-icp"
    assert (site / "index.html").is_file()
