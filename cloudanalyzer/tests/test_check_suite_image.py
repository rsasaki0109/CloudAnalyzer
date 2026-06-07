"""Phase 31: tests for the config-driven ``image`` (photometric) check kind.

Wires ``ca.core.image_evaluate`` into ``ca check`` as a new check kind with a
PSNR/SSIM gate, report output, triage ranking, and PR-comment deltas. The
image-pair fixture is built at test time (small synthetic data, no committed
binaries), mirroring ``tests/test_image_evaluate.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import numpy as np

from ca.core import load_check_suite, run_check_suite
from ca.pr_comment import build_pr_comment


def _build_image_pair_fixture(
    base_dir: Path,
    *,
    n_pairs: int = 3,
    size: int = 64,
    noise_sigma: float = 0.02,
    seed: int = 7,
) -> tuple[Path, Path]:
    """Synthesize a ``(rendered, reference)`` image-pair set under ``base_dir``."""
    import matplotlib.pyplot as plt

    rendered_dir = base_dir / "rendered"
    reference_dir = base_dir / "reference"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 1.0, size)
    xx, yy = np.meshgrid(xs, xs)
    for i in range(n_pairs):
        ref = np.clip(np.stack([xx, yy, 0.5 + 0.0 * xx], axis=-1), 0.0, 1.0)
        ren = np.clip(ref + rng.normal(0, noise_sigma, ref.shape), 0.0, 1.0)
        plt.imsave(str(reference_dir / f"img_{i:02d}.png"), ref)
        plt.imsave(str(rendered_dir / f"img_{i:02d}.png"), ren)
    return rendered_dir, reference_dir


def _write_config(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")
    return path


def test_image_check_passes_lenient_gate(tmp_path: Path) -> None:
    rendered, reference = _build_image_pair_fixture(tmp_path)
    config = _write_config(
        tmp_path / "cloudanalyzer.yaml",
        f"""
        checks:
          - id: photo
            kind: image
            rendered_dir: {rendered.relative_to(tmp_path)}
            reference_dir: {reference.relative_to(tmp_path)}
            gate:
              min_psnr: 20.0
              min_ssim: 0.5
        """,
    )

    result = run_check_suite(load_check_suite(str(config)))

    assert result["summary"]["passed"] is True
    assert result["summary"]["gated_checks"] == 1
    check = result["checks"][0]
    assert check["kind"] == "image"
    assert check["passed"] is True
    assert check["summary"]["pairs_evaluated"] == 3
    assert check["summary"]["psnr_mean"] > 20.0
    assert 0.5 <= check["summary"]["ssim_mean"] <= 1.0


def test_image_check_fails_strict_psnr_gate_and_triage(tmp_path: Path) -> None:
    # Heavy noise drives PSNR well below an unrealistically strict floor.
    rendered, reference = _build_image_pair_fixture(tmp_path, noise_sigma=0.2)
    config = _write_config(
        tmp_path / "cloudanalyzer.yaml",
        f"""
        checks:
          - id: photo
            kind: image
            rendered_dir: {rendered.relative_to(tmp_path)}
            reference_dir: {reference.relative_to(tmp_path)}
            gate:
              min_psnr: 60.0
        """,
    )

    result = run_check_suite(load_check_suite(str(config)))

    assert result["summary"]["passed"] is False
    assert result["summary"]["failed_check_ids"] == ["photo"]
    check = result["checks"][0]
    assert check["passed"] is False
    gate = check["result"]["quality_gate"]
    assert gate["passed"] is False
    assert any("PSNR" in reason for reason in gate["reasons"])
    # Triage should rank the failed image dimension.
    triage = result["summary"]["triage"]
    assert "photo" in triage["ranked_ids"]


def test_image_check_unmatched_pairs_fail_gate(tmp_path: Path) -> None:
    rendered, reference = _build_image_pair_fixture(tmp_path)
    # Wipe references so no pair matches -> 0 pairs evaluated.
    for ref_file in reference.glob("*.png"):
        ref_file.unlink()
    config = _write_config(
        tmp_path / "cloudanalyzer.yaml",
        f"""
        checks:
          - id: photo
            kind: image
            rendered_dir: {rendered.relative_to(tmp_path)}
            reference_dir: {reference.relative_to(tmp_path)}
            gate:
              min_psnr: 20.0
        """,
    )

    result = run_check_suite(load_check_suite(str(config)))

    check = result["checks"][0]
    assert check["passed"] is False
    assert check["summary"]["pairs_evaluated"] == 0
    assert any("0 pairs" in reason for reason in check["result"]["quality_gate"]["reasons"])


def test_image_check_writes_report(tmp_path: Path) -> None:
    rendered, reference = _build_image_pair_fixture(tmp_path)
    config = _write_config(
        tmp_path / "cloudanalyzer.yaml",
        f"""
        defaults:
          report_dir: qa/reports
        checks:
          - id: photo
            kind: image
            rendered_dir: {rendered.relative_to(tmp_path)}
            reference_dir: {reference.relative_to(tmp_path)}
            gate:
              min_psnr: 20.0
        """,
    )

    result = run_check_suite(load_check_suite(str(config)))

    report_path = tmp_path / "qa" / "reports" / "photo.html"
    assert result["checks"][0]["report_path"] == str(report_path.resolve())
    assert report_path.exists()
    html = report_path.read_text()
    assert "Image Evaluation Report" in html
    assert "PSNR" in html


def test_image_check_pr_comment_shows_psnr_delta(tmp_path: Path) -> None:
    rendered, reference = _build_image_pair_fixture(tmp_path)
    config = _write_config(
        tmp_path / "cloudanalyzer.yaml",
        f"""
        checks:
          - id: photo
            kind: image
            rendered_dir: {rendered.relative_to(tmp_path)}
            reference_dir: {reference.relative_to(tmp_path)}
            gate:
              min_psnr: 20.0
        """,
    )
    summary = run_check_suite(load_check_suite(str(config)))

    # Fabricate a baseline with a higher PSNR so a downward delta renders.
    baseline = json.loads(json.dumps(summary))
    baseline["checks"][0]["summary"]["psnr_mean"] = (
        summary["checks"][0]["summary"]["psnr_mean"] + 5.0
    )

    comment = build_pr_comment(summary, baseline=baseline)
    assert "PSNR" in comment
    assert "↓" in comment  # current PSNR is below the inflated baseline
