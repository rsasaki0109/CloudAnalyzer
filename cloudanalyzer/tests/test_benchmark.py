"""Tests for the SLAM benchmark suite runner (`ca benchmark`)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from ca.benchmark import (
    GATE_KEYS,
    REPORT_BUNDLE_SCHEMA_VERSION,
    BenchmarkSuite,
    evaluate_benchmark_run,
    load_benchmark_suite,
    materialize_suite,
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


def test_cli_eval_out_writes_report_bundle(
    synthetic_suite_dir: Path,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    suite_yaml = synthetic_suite_dir / "suite.yaml"
    sample_map = synthetic_suite_dir / "sample_outputs" / "map_pass.pcd"
    sample_traj = synthetic_suite_dir / "sample_outputs" / "trajectory_pass.tum"
    out_dir = tmp_path / "qa" / "synthetic-figure8"

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
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"Bundle: {out_dir}" in result.output
    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "summary.md").is_file()
    assert (out_dir / "report.html").is_file()
    assert (out_dir / "manifest.lock.yaml").is_file()
    assert (out_dir / "provenance.json").is_file()

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["overall_quality_gate"]["passed"] is True
    assert metrics["benchmark"]["suite"] == "synthetic-figure8"
    assert str(tmp_path) not in json.dumps(metrics)

    lock = yaml.safe_load((out_dir / "manifest.lock.yaml").read_text(encoding="utf-8"))
    assert lock["schema_version"] == REPORT_BUNDLE_SCHEMA_VERSION
    assert lock["outputs"] == {
        "metrics": "metrics.json",
        "summary": "summary.md",
        "report": "report.html",
        "report_assets": [
            "report_map_f1.png",
            "report_trajectory_errors.png",
            "report_trajectory_overlay.png",
        ],
        "provenance": "provenance.json",
        "manifest_lock": "manifest.lock.yaml",
    }
    assert lock["inputs"]["candidate_map"]["sha256"]
    assert lock["inputs"]["candidate_trajectory"]["sha256"]

    provenance = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["schema_version"] == REPORT_BUNDLE_SCHEMA_VERSION
    assert provenance["summary_kind"] == "benchmark_run"
    assert provenance["overall_quality_gate"]["passed"] is True
    assert provenance["artifacts"]["report_assets"] == lock["outputs"]["report_assets"]

    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "CloudAnalyzer QA" in summary


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


def _count_tum_lines(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def test_materialize_suite_round_trip(
    synthetic_suite_dir: Path, tmp_path: Path
) -> None:
    ref_map = synthetic_suite_dir / "reference" / "map.pcd"
    ref_traj = synthetic_suite_dir / "reference" / "trajectory.tum"
    out_dir = tmp_path / "suite"

    suite = materialize_suite(
        out_dir,
        name="round-trip",
        description="Round-trip test suite",
        reference_map=ref_map,
        reference_trajectory=ref_traj,
        sequence_name="seq0",
        sequence_description="Seq0 description",
        license="MIT (synthetic)",
        voxel_size=0.0,  # plain copy
        max_poses=None,
        gate={"min_auc": 0.9, "max_ate": 0.4, "not_a_key": 1.0},
    )

    # suite.yaml landed where we asked
    assert suite.source_path == (out_dir / "suite.yaml").resolve()
    assert suite.name == "round-trip"
    assert "seq0" in suite.sequences

    # Reference files actually exist on disk and contain the source bytes.
    seq = suite.sequences["seq0"]
    assert seq.reference_map_path.is_file()
    assert seq.reference_trajectory_path.is_file()
    assert seq.reference_map_path.read_bytes() == ref_map.read_bytes()
    assert _count_tum_lines(seq.reference_trajectory_path) == _count_tum_lines(ref_traj)

    # Unknown gate keys are dropped silently; known ones are preserved.
    assert "not_a_key" not in suite.gate
    assert suite.gate["min_auc"] == pytest.approx(0.9)
    assert suite.gate["max_ate"] == pytest.approx(0.4)


def test_materialize_suite_voxel_downsample_shrinks_map(
    synthetic_suite_dir: Path, tmp_path: Path
) -> None:
    import open3d as o3d

    ref_map = synthetic_suite_dir / "reference" / "map.pcd"
    ref_traj = synthetic_suite_dir / "reference" / "trajectory.tum"

    original_points = len(o3d.io.read_point_cloud(str(ref_map)).points)
    assert original_points > 0

    suite = materialize_suite(
        tmp_path / "suite",
        name="voxel",
        description="Voxel downsample test",
        reference_map=ref_map,
        reference_trajectory=ref_traj,
        voxel_size=2.0,
    )
    seq = suite.resolve_sequence(None)
    downsampled_points = len(o3d.io.read_point_cloud(str(seq.reference_map_path)).points)
    assert 0 < downsampled_points < original_points


def test_materialize_suite_max_poses_subsamples(
    synthetic_suite_dir: Path, tmp_path: Path
) -> None:
    ref_map = synthetic_suite_dir / "reference" / "map.pcd"
    ref_traj = synthetic_suite_dir / "reference" / "trajectory.tum"
    original_lines = _count_tum_lines(ref_traj)
    assert original_lines >= 50  # synthetic suite generates 200 poses

    suite = materialize_suite(
        tmp_path / "suite",
        name="subsample",
        description="Trajectory subsample test",
        reference_map=ref_map,
        reference_trajectory=ref_traj,
        max_poses=20,
    )
    seq = suite.resolve_sequence(None)
    kept = _count_tum_lines(seq.reference_trajectory_path)
    assert kept == 20


def test_materialize_suite_with_sample_outputs(
    synthetic_suite_dir: Path, tmp_path: Path
) -> None:
    ref_map = synthetic_suite_dir / "reference" / "map.pcd"
    ref_traj = synthetic_suite_dir / "reference" / "trajectory.tum"
    sample_map = synthetic_suite_dir / "sample_outputs" / "map_pass.pcd"
    sample_traj = synthetic_suite_dir / "sample_outputs" / "trajectory_pass.tum"

    suite = materialize_suite(
        tmp_path / "suite",
        name="with-sample",
        description="Sample outputs test",
        reference_map=ref_map,
        reference_trajectory=ref_traj,
        sample_map=sample_map,
        sample_trajectory=sample_traj,
        gate={"min_auc": 0.95},
    )
    seq = suite.resolve_sequence(None)
    assert seq.sample_map_path is not None and seq.sample_map_path.exists()
    assert (
        seq.sample_trajectory_path is not None and seq.sample_trajectory_path.exists()
    )

    # End-to-end: the materialized suite is immediately runnable via evaluate_benchmark_run.
    result = evaluate_benchmark_run(
        suite,
        str(seq.sample_map_path),
        str(seq.sample_trajectory_path),
    )
    assert result["benchmark"]["suite"] == "with-sample"
    assert result["overall_quality_gate"]["passed"] is True


def test_prepare_newer_college_mini_smoke(
    synthetic_suite_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive `prepare_newer_college_mini.py` against synthetic GT files.

    The script itself is a thin wrapper around `materialize_suite`; we
    only need to confirm the CLI plumbing (arg parsing, default gate,
    description rendering) produces a loadable suite.
    """
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "prepare_newer_college_mini.py"
    spec = importlib.util.spec_from_file_location("prepare_ncm", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "ncm-suite"
    argv = [
        "prepare_newer_college_mini.py",
        "--reference-map",
        str(synthetic_suite_dir / "reference" / "map.pcd"),
        "--reference-trajectory",
        str(synthetic_suite_dir / "reference" / "trajectory.tum"),
        "--sequence",
        "short_experiment",
        "--voxel",
        "0.5",
        "--max-poses",
        "50",
        "--gate",
        "min_auc=0.80",
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    suite = load_benchmark_suite(output / "suite.yaml")
    assert suite.name == "newer-college-mini"
    assert "short_experiment" in suite.sequences
    seq = suite.sequences["short_experiment"]
    assert seq.reference_map_path.exists()
    assert seq.reference_trajectory_path.exists()
    # Default gate from the wrapper, then overridden via --gate.
    assert suite.gate["min_auc"] == pytest.approx(0.80)
    assert suite.gate["max_chamfer"] == pytest.approx(0.30)  # wrapper default


def test_prepare_newer_college_mini_rejects_unknown_gate(
    synthetic_suite_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "prepare_newer_college_mini.py"
    spec = importlib.util.spec_from_file_location("prepare_ncm_bad", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    argv = [
        "prepare_newer_college_mini.py",
        "--reference-map",
        str(synthetic_suite_dir / "reference" / "map.pcd"),
        "--reference-trajectory",
        str(synthetic_suite_dir / "reference" / "trajectory.tum"),
        "--gate",
        "not_a_real_gate=0.5",
        "--output",
        str(tmp_path / "bad-suite"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="unknown gate key"):
        module.main()


def _load_kitti_mini_module() -> Any:
    import importlib.util

    script_path = REPO_ROOT / "scripts" / "prepare_kitti_mini.py"
    spec = importlib.util.spec_from_file_location("prepare_kitti_mini", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_kitti_pose_file(tmp_path: Path, n_poses: int = 30) -> Path:
    """Build a tiny KITTI 12-float pose file driving a planar arc."""
    import numpy as np

    out = tmp_path / "kitti_poses.txt"
    lines: list[str] = []
    for i in range(n_poses):
        theta = i * 0.1
        cos, sin = float(np.cos(theta)), float(np.sin(theta))
        # 3x4 row-major: rotation around Z + translation along the arc.
        row0 = [cos, -sin, 0.0, float(i) * 0.5]
        row1 = [sin, cos, 0.0, float(i) * 0.2]
        row2 = [0.0, 0.0, 1.0, 0.0]
        lines.append(" ".join(f"{v:.6f}" for v in row0 + row1 + row2))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def test_kitti_pose_conversion_round_trip(tmp_path: Path) -> None:
    """KITTI 12-float → TUM conversion preserves translations and rotation magnitude."""
    import numpy as np

    module = _load_kitti_mini_module()
    poses_path = _make_kitti_pose_file(tmp_path, n_poses=10)
    tum_path = tmp_path / "out.tum"

    count = module.kitti_poses_to_tum(poses_path, tum_path, frame_rate_hz=10.0)
    assert count == 10
    lines = [
        line.split()
        for line in tum_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 10
    # Timestamps are evenly spaced at 0.1 s.
    timestamps = [float(parts[0]) for parts in lines]
    assert timestamps[0] == pytest.approx(0.0)
    assert timestamps[1] - timestamps[0] == pytest.approx(0.1)
    # Translations recover the KITTI tx/ty/tz columns exactly.
    for i, parts in enumerate(lines):
        assert float(parts[1]) == pytest.approx(i * 0.5, abs=1e-5)
        assert float(parts[2]) == pytest.approx(i * 0.2, abs=1e-5)
        assert float(parts[3]) == pytest.approx(0.0, abs=1e-5)
        # Quaternion is unit-norm.
        qx, qy, qz, qw = (float(v) for v in parts[4:8])
        assert qx * qx + qy * qy + qz * qz + qw * qw == pytest.approx(1.0, abs=1e-5)
    # First pose is identity → qw≈1, qx=qy=qz≈0.
    first = lines[0]
    assert float(first[7]) == pytest.approx(1.0, abs=1e-5)
    assert float(first[4]) == pytest.approx(0.0, abs=1e-5)
    assert float(first[5]) == pytest.approx(0.0, abs=1e-5)
    assert float(first[6]) == pytest.approx(0.0, abs=1e-5)


def test_kitti_pose_conversion_rejects_malformed(tmp_path: Path) -> None:
    module = _load_kitti_mini_module()
    bad = tmp_path / "bad_poses.txt"
    bad.write_text("1.0 2.0 3.0\n", encoding="utf-8")  # only 3 fields, needs 12
    with pytest.raises(ValueError, match="expected 12"):
        module.kitti_poses_to_tum(bad, tmp_path / "out.tum")

    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no KITTI poses"):
        module.kitti_poses_to_tum(empty, tmp_path / "out.tum")


def test_prepare_kitti_mini_with_kitti_poses(
    synthetic_suite_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive `prepare_kitti_mini.py` with a KITTI 12-float pose file."""
    module = _load_kitti_mini_module()
    poses_path = _make_kitti_pose_file(tmp_path, n_poses=30)
    output = tmp_path / "kitti-suite"
    argv = [
        "prepare_kitti_mini.py",
        "--kitti-poses",
        str(poses_path),
        "--reference-map",
        str(synthetic_suite_dir / "reference" / "map.pcd"),
        "--sequence",
        "sequence_00",
        "--voxel",
        "0.5",
        "--max-poses",
        "20",
        "--gate",
        "min_auc=0.88",
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    suite = load_benchmark_suite(output / "suite.yaml")
    assert suite.name == "kitti-mini"
    assert "sequence_00" in suite.sequences
    seq = suite.sequences["sequence_00"]
    assert seq.reference_map_path.exists()
    assert seq.reference_trajectory_path.exists()
    # Wrapper default gate, then `--gate` override on top.
    assert suite.gate["min_auc"] == pytest.approx(0.88)
    assert suite.gate["max_ate"] == pytest.approx(5.00)  # KITTI default


def test_prepare_kitti_mini_with_pre_converted_tum(
    synthetic_suite_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass-through path: user provides an already-TUM trajectory."""
    module = _load_kitti_mini_module()
    output = tmp_path / "kitti-suite-tum"
    argv = [
        "prepare_kitti_mini.py",
        "--reference-trajectory",
        str(synthetic_suite_dir / "reference" / "trajectory.tum"),
        "--reference-map",
        str(synthetic_suite_dir / "reference" / "map.pcd"),
        "--voxel",
        "0.5",
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    suite = load_benchmark_suite(output / "suite.yaml")
    assert suite.name == "kitti-mini"
    seq = suite.resolve_sequence(None)
    assert seq.reference_trajectory_path.exists()


def test_prepare_kitti_mini_rejects_unknown_gate(
    synthetic_suite_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_kitti_mini_module()
    poses_path = _make_kitti_pose_file(tmp_path, n_poses=5)
    argv = [
        "prepare_kitti_mini.py",
        "--kitti-poses",
        str(poses_path),
        "--reference-map",
        str(synthetic_suite_dir / "reference" / "map.pcd"),
        "--gate",
        "not_a_gate=0.5",
        "--output",
        str(tmp_path / "bad-kitti-suite"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="unknown gate key"):
        module.main()


def test_prepare_kitti_mini_requires_pose_source(
    synthetic_suite_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Either --kitti-poses or --reference-trajectory must be provided."""
    module = _load_kitti_mini_module()
    argv = [
        "prepare_kitti_mini.py",
        "--reference-map",
        str(synthetic_suite_dir / "reference" / "map.pcd"),
        "--output",
        str(tmp_path / "no-pose-suite"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        module.main()
