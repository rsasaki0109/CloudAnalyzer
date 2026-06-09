"""Tests for the config-driven ``rendered`` (3DGS) check kind."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest

from ca.core import load_check_suite, run_check_suite
from ca.core.rendered_evaluate import RenderedEvalResult
from ca.pr_comment import build_pr_comment

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = REPO_ROOT / "benchmarks" / "3dgs" / "synthetic-room"


def _gs_available() -> bool:
    try:
        import gsplat  # noqa: F401
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _write_config(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")
    return path


def _mock_rendered_result(*, psnr: float = 40.0, ssim: float = 0.99, auc: float = 0.9) -> RenderedEvalResult:
    return RenderedEvalResult(
        photometric={
            "summary": {
                "psnr_mean": psnr,
                "ssim_mean": ssim,
                "lpips_mean": None,
                "pairs_evaluated": 4,
                "pairs_missing_in_reference": 0,
                "pairs_size_mismatch": 0,
            },
            "pairs": [],
            "metadata": {},
        },
        geometry={
            "auc": auc,
            "chamfer_distance": 0.05,
            "source_points": 100,
            "target_points": 100,
            "f1_scores": [],
        },
        renderer={"backend": "gsplat", "frames_rendered": 4},
        metadata={},
    )


def test_rendered_check_config_loads(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "cloudanalyzer.yaml",
        """
        checks:
          - id: splat
            kind: rendered
            splat: scene.ply
            cameras: transforms.json
            reference_dir: refs/
            reference_pointcloud: ref.pcd
            gate:
              min_psnr: 25.0
              max_chamfer: 0.2
        """,
    )

    suite = load_check_suite(str(config))
    spec = suite.checks[0]
    assert spec.kind == "rendered"
    assert spec.gate["min_psnr"] == pytest.approx(25.0)
    assert spec.gate["max_chamfer"] == pytest.approx(0.2)


def test_rendered_check_passes_with_mock(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "cloudanalyzer.yaml",
        """
        checks:
          - id: splat
            kind: rendered
            splat: scene.ply
            cameras: transforms.json
            reference_dir: refs/
            reference_pointcloud: ref.pcd
            gate:
              min_psnr: 30.0
              min_ssim: 0.9
              max_chamfer: 0.1
        """,
    )

    with patch(
        "ca.core.rendered_evaluate.rendered_evaluate",
        return_value=_mock_rendered_result(),
    ):
        result = run_check_suite(load_check_suite(str(config)))

    assert result["summary"]["passed"] is True
    check = result["checks"][0]
    assert check["kind"] == "rendered"
    assert check["passed"] is True
    assert check["summary"]["psnr_mean"] == pytest.approx(40.0)
    assert check["summary"]["auc"] == pytest.approx(0.9)


def test_rendered_check_fails_geometry_gate_without_reference(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "cloudanalyzer.yaml",
        """
        checks:
          - id: splat
            kind: rendered
            splat: scene.ply
            cameras: transforms.json
            reference_dir: refs/
            gate:
              min_auc: 0.5
        """,
    )

    with patch(
        "ca.core.rendered_evaluate.rendered_evaluate",
        return_value=RenderedEvalResult(
            photometric={
                "summary": {
                    "psnr_mean": 40.0,
                    "ssim_mean": 0.99,
                    "lpips_mean": None,
                    "pairs_evaluated": 4,
                    "pairs_missing_in_reference": 0,
                    "pairs_size_mismatch": 0,
                },
                "pairs": [],
                "metadata": {},
            },
            geometry=None,
            renderer={"backend": "gsplat"},
            metadata={},
        ),
    ):
        result = run_check_suite(load_check_suite(str(config)))

    check = result["checks"][0]
    assert check["passed"] is False
    reasons = check["result"]["quality_gate"]["reasons"]
    assert any("reference_pointcloud" in reason for reason in reasons)


def test_rendered_check_writes_report(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "cloudanalyzer.yaml",
        """
        defaults:
          report_dir: qa/reports
        checks:
          - id: splat
            kind: rendered
            splat: scene.ply
            cameras: transforms.json
            reference_dir: refs/
            gate:
              min_psnr: 20.0
        """,
    )

    with patch(
        "ca.core.rendered_evaluate.rendered_evaluate",
        return_value=_mock_rendered_result(),
    ):
        result = run_check_suite(load_check_suite(str(config)))

    report_path = tmp_path / "qa" / "reports" / "splat.html"
    assert result["checks"][0]["report_path"] == str(report_path.resolve())
    assert report_path.exists()
    assert "CloudAnalyzer Rendered 3DGS Evaluation Report" in report_path.read_text()


def test_rendered_check_pr_comment_shows_psnr(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "cloudanalyzer.yaml",
        """
        checks:
          - id: splat
            kind: rendered
            splat: scene.ply
            cameras: transforms.json
            reference_dir: refs/
            gate:
              min_psnr: 20.0
        """,
    )

    with patch(
        "ca.core.rendered_evaluate.rendered_evaluate",
        return_value=_mock_rendered_result(),
    ):
        summary = run_check_suite(load_check_suite(str(config)))

    comment = build_pr_comment(summary)
    assert "PSNR" in comment
    assert "splat" in comment


@pytest.mark.skipif(not _gs_available(), reason="cloudanalyzer[gs] with CUDA required")
@pytest.mark.skipif(not DEMO_ROOT.is_dir(), reason="synthetic-room demo missing")
def test_rendered_check_integration_synthetic_room(tmp_path: Path) -> None:
    config = _write_config(
        tmp_path / "cloudanalyzer.yaml",
        f"""
        checks:
          - id: room
            kind: rendered
            splat: {DEMO_ROOT / "gaussians_dense.ply"}
            cameras: {DEMO_ROOT / "transforms.json"}
            reference_dir: {DEMO_ROOT / "reference"}
            reference_pointcloud: {DEMO_ROOT / "reference.pcd"}
            gate:
              min_ssim: 0.9
              max_chamfer: 0.2
        """,
    )

    result = run_check_suite(load_check_suite(str(config)))
    check = result["checks"][0]
    assert check["passed"] is True
    assert check["summary"]["ssim_mean"] > 0.9
    assert check["summary"]["chamfer_distance"] < 0.2
