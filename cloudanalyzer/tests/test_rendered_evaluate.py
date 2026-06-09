"""Tests for ``ca rendered-evaluate`` and ``ca.core.rendered_evaluate``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_ROOT = REPO_ROOT / "benchmarks" / "3dgs" / "synthetic-room"


def _gs_available() -> bool:
    try:
        import gsplat  # noqa: F401
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not DEMO_ROOT.is_dir(), reason="synthetic-room demo missing")
def test_rendered_evaluate_missing_gs_shows_hint() -> None:
    from ca.core.rendered_evaluate import RenderedEvalRequest, rendered_evaluate

    request = RenderedEvalRequest(
        splat_path=DEMO_ROOT / "gaussians_dense.ply",
        cameras_path=DEMO_ROOT / "transforms.json",
        reference_dir=DEMO_ROOT / "reference",
    )
    with patch("ca.core.gs_renderer.require_gsplat", side_effect=ValueError("need gs")):
        with pytest.raises(ValueError, match="need gs"):
            rendered_evaluate(request)


@pytest.mark.skipif(not _gs_available(), reason="cloudanalyzer[gs] with CUDA required")
@pytest.mark.skipif(not DEMO_ROOT.is_dir(), reason="synthetic-room demo missing")
def test_rendered_evaluate_self_consistency_high_psnr() -> None:
    from ca.core.rendered_evaluate import RenderedEvalRequest, rendered_evaluate

    result = rendered_evaluate(
        RenderedEvalRequest(
            splat_path=DEMO_ROOT / "gaussians_dense.ply",
            cameras_path=DEMO_ROOT / "transforms.json",
            reference_dir=DEMO_ROOT / "reference",
            reference_pointcloud=DEMO_ROOT / "reference.pcd",
            metrics=("psnr", "ssim"),
        )
    )
    summary = result.photometric["summary"]
    assert summary["pairs_evaluated"] >= 4
    assert summary["ssim_mean"] is not None
    assert summary["ssim_mean"] > 0.95
    pair_psnr = [p["psnr"] for p in result.photometric["pairs"]]
    assert pair_psnr and all(v == float("inf") or v > 35.0 for v in pair_psnr)
    assert result.geometry is not None
    assert result.geometry["chamfer_distance"] < 0.15


@pytest.mark.skipif(not _gs_available(), reason="cloudanalyzer[gs] with CUDA required")
@pytest.mark.skipif(not DEMO_ROOT.is_dir(), reason="synthetic-room demo missing")
def test_rendered_evaluate_cli_smoke() -> None:
    cmd = [
        sys.executable,
        "-m",
        "cloudanalyzer_cli.main",
        "rendered-evaluate",
        str(DEMO_ROOT / "gaussians_dense.ply"),
        str(DEMO_ROOT / "reference"),
        "--cameras",
        str(DEMO_ROOT / "transforms.json"),
        "--metrics",
        "psnr",
        "--max-pairs",
        "2",
        "--format-json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["photometric"]["summary"]["pairs_evaluated"] == 2
