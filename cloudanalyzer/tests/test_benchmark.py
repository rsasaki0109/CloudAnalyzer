"""Tests for the SLAM benchmark suite runner (`ca benchmark`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ca.benchmark import (
    GATE_KEYS,
    BenchmarkSuite,
    evaluate_benchmark_run,
    load_benchmark_suite,
)
from cloudanalyzer_cli.main import app


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
SUITE_PATH = REPO_ROOT / "benchmarks" / "slam" / "synthetic-figure8" / "suite.yaml"

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_synthetic_slam_suite import build as build_synthetic_suite  # noqa: E402


@pytest.fixture(scope="module")
def synthetic_suite_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Regenerate the synthetic suite into a temp dir.

    Keeps the test independent of whether the checked-in copy is up to date
    and gives us a clean place to mutate sample files for failure scenarios.
    """
    out = tmp_path_factory.mktemp("synthetic-figure8")
    build_synthetic_suite(out)
    return out


def test_loader_resolves_paths_and_gate(synthetic_suite_dir: Path) -> None:
    suite = load_benchmark_suite(synthetic_suite_dir / "suite.yaml")
    assert suite.name == "synthetic-figure8"
    assert suite.version == 1
    assert "default" in suite.sequences
    seq = suite.sequences["default"]
    assert seq.reference_map_path.exists()
    assert seq.reference_trajectory_path.exists()
    assert seq.sample_map_path is not None and seq.sample_map_path.exists()
    assert seq.sample_trajectory_path is not None and seq.sample_trajectory_path.exists()
    assert set(suite.gate).issubset(set(GATE_KEYS))
    assert suite.gate["min_auc"] == pytest.approx(0.95)


def test_load_rejects_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_benchmark_suite(empty)


def test_load_rejects_unknown_gate_key(tmp_path: Path) -> None:
    (tmp_path / "ref.pcd").write_text("placeholder", encoding="utf-8")
    (tmp_path / "ref.tum").write_text("0 0 0 0 0 0 0 1\n", encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "version: 1\nname: x\ndescription: y\nsequences:\n"
        "  s:\n    description: d\n    reference_map: ref.pcd\n"
        "    reference_trajectory: ref.tum\n"
        "gate:\n  not_a_gate_key: 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="gate key"):
        load_benchmark_suite(bad)


def test_evaluate_benchmark_run_passes_with_sample(synthetic_suite_dir: Path) -> None:
    suite = load_benchmark_suite(synthetic_suite_dir / "suite.yaml")
    seq = suite.resolve_sequence(None)
    result = evaluate_benchmark_run(
        suite,
        str(seq.sample_map_path),
        str(seq.sample_trajectory_path),
    )
    assert result["benchmark"]["suite"] == "synthetic-figure8"
    assert result["benchmark"]["sequence"] == "default"
    assert result["benchmark"]["gate"]["min_auc"] == pytest.approx(0.95)
    assert result["overall_quality_gate"]["passed"] is True


def test_evaluate_benchmark_run_fails_when_gate_tightened(
    synthetic_suite_dir: Path,
) -> None:
    suite = load_benchmark_suite(synthetic_suite_dir / "suite.yaml")
    seq = suite.resolve_sequence(None)
    # Drive RPE gate below the sample's measured value to force a FAIL.
    result = evaluate_benchmark_run(
        suite,
        str(seq.sample_map_path),
        str(seq.sample_trajectory_path),
        gate_overrides={"max_rpe": 0.05},
    )
    assert result["overall_quality_gate"]["passed"] is False
    assert any("RPE" in reason for reason in result["overall_quality_gate"]["reasons"])


def test_evaluate_benchmark_run_unknown_sequence(synthetic_suite_dir: Path) -> None:
    suite = load_benchmark_suite(synthetic_suite_dir / "suite.yaml")
    seq = suite.resolve_sequence(None)
    with pytest.raises(ValueError, match="not in benchmark suite"):
        evaluate_benchmark_run(
            suite,
            str(seq.sample_map_path),
            str(seq.sample_trajectory_path),
            sequence="missing",
        )


def test_cli_info_json(synthetic_suite_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["benchmark", "info", str(synthetic_suite_dir / "suite.yaml"), "--format-json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["name"] == "synthetic-figure8"
    assert "default" in payload["sequences"]


def test_cli_eval_pass(synthetic_suite_dir: Path) -> None:
    runner = CliRunner()
    suite_yaml = synthetic_suite_dir / "suite.yaml"
    sample_map = synthetic_suite_dir / "sample_outputs" / "map_pass.pcd"
    sample_traj = synthetic_suite_dir / "sample_outputs" / "trajectory_pass.tum"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "eval",
            str(suite_yaml),
            "--map",
            str(sample_map),
            "--trajectory",
            str(sample_traj),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Overall Quality Gate: PASS" in result.output


def test_cli_eval_gate_override_fails(synthetic_suite_dir: Path) -> None:
    runner = CliRunner()
    suite_yaml = synthetic_suite_dir / "suite.yaml"
    sample_map = synthetic_suite_dir / "sample_outputs" / "map_pass.pcd"
    sample_traj = synthetic_suite_dir / "sample_outputs" / "trajectory_pass.tum"
    result = runner.invoke(
        app,
        [
            "benchmark",
            "eval",
            str(suite_yaml),
            "--map",
            str(sample_map),
            "--trajectory",
            str(sample_traj),
            "--gate",
            "max_rpe=0.05",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output
