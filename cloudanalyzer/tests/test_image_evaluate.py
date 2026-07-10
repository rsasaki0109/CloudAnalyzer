"""Phase 30: tests for ``ca.core.image_evaluate`` and ``ca image-evaluate``.

Covers:

- PSNR / SSIM scalar functions on known inputs (identity, additive
  Gaussian noise of known magnitude, channel-wise consistency).
- ``image_evaluate`` against a small synthetic image pair built at
  test time (no committed binary fixtures).
- The CLI subprocess path: ``ca image-evaluate`` returns aggregate
  metrics for the same fixture.

The fixture builder lives in this module on purpose — the synthetic
data is small (4 × 80×80 RGB) and a separate ``scripts/`` script would
be overkill for this scale.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_frequency_consistency_identical_and_artifact() -> None:
    from ca.core.image_evaluate import frequency_consistency

    reference = np.zeros((16, 16, 3), dtype=np.float64)
    reference[4:12, 4:12] = 0.75
    checkerboard = (np.indices((16, 16)).sum(axis=0) % 2).astype(np.float64)
    candidate = np.repeat(checkerboard[..., None], 3, axis=2)

    assert frequency_consistency(reference, reference) == pytest.approx(0.0)
    score = frequency_consistency(candidate, reference)
    assert 0.0 < score <= 1.0


def test_frequency_consistency_flat_reference_policy() -> None:
    from ca.core.image_evaluate import frequency_consistency

    flat = np.zeros((12, 12, 3), dtype=np.float64)
    nonflat = flat.copy()
    nonflat[6, 6] = 1.0

    assert frequency_consistency(flat, flat) == pytest.approx(0.0)
    assert frequency_consistency(nonflat, flat) == pytest.approx(1.0)

from ca.core.image_evaluate import (
    DREAMSIM_INSTALL_HINT,
    ImageEvalRequest,
    dreamsim_distance,
    image_evaluate,
    psnr,
    ssim,
)


# ---------------------------------------------------------------------------
# Tiny synthetic image-pair fixture (regenerated for every test that needs
# it). Deterministic via fixed RNG seed.
# ---------------------------------------------------------------------------


def _build_image_pair_fixture(
    base_dir: Path,
    *,
    n_pairs: int = 4,
    size: int = 80,
    noise_sigma: float = 0.02,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Synthesize a ``(rendered, reference)`` image pair under ``base_dir``."""

    import matplotlib.pyplot as plt

    rendered_dir = base_dir / "rendered"
    reference_dir = base_dir / "reference"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    for i in range(n_pairs):
        # A reference: smooth gradient with a colored blob. Deterministic.
        xs = np.linspace(0.0, 1.0, size)
        ys = np.linspace(0.0, 1.0, size)
        xx, yy = np.meshgrid(xs, ys)
        ref = np.stack([xx, yy, 0.5 + 0.0 * xx], axis=-1)
        ref = np.clip(ref, 0.0, 1.0)
        # Rendered: reference plus Gaussian noise (known PSNR/SSIM range).
        ren = ref + rng.normal(0, noise_sigma, ref.shape)
        ren = np.clip(ren, 0.0, 1.0)
        plt.imsave(str(reference_dir / f"img_{i:02d}.png"), ref)
        plt.imsave(str(rendered_dir / f"img_{i:02d}.png"), ren)
    return rendered_dir, reference_dir


# ---------------------------------------------------------------------------
# Scalar metric correctness.
# ---------------------------------------------------------------------------


def test_psnr_identical_inputs_is_infinite() -> None:
    rng = np.random.default_rng(0)
    a = rng.random((50, 50, 3))
    assert psnr(a, a) == float("inf")


def test_psnr_known_mse_matches_closed_form() -> None:
    a = np.full((10, 10), 0.5)
    b = np.full((10, 10), 0.3)
    # MSE = 0.04; PSNR = 10*log10(1/0.04) = 10*log10(25) ≈ 13.9794
    val = psnr(a, b)
    assert val == pytest.approx(10 * np.log10(1.0 / 0.04), abs=1e-9)


def test_psnr_rejects_shape_mismatch() -> None:
    a = np.zeros((4, 4))
    b = np.zeros((4, 5))
    with pytest.raises(ValueError):
        psnr(a, b)


def test_ssim_identical_inputs_is_one() -> None:
    rng = np.random.default_rng(0)
    a = rng.random((60, 60, 3))
    val = ssim(a, a)
    assert val == pytest.approx(1.0, abs=1e-6)


def test_ssim_small_noise_close_to_one() -> None:
    rng = np.random.default_rng(0)
    a = rng.random((60, 60))
    b = a + rng.normal(0, 0.01, a.shape)
    b = np.clip(b, 0.0, 1.0)
    val = ssim(a, b)
    # Sub-1% noise on random texture stays near 1 (well above the gate
    # most photometric tasks would care about).
    assert 0.85 <= val <= 1.0


def test_ssim_handles_color_via_channelwise_mean() -> None:
    rng = np.random.default_rng(0)
    a = rng.random((40, 40, 3))
    b = a.copy()
    b[..., 0] = np.clip(b[..., 0] + 0.1, 0.0, 1.0)
    val_color = ssim(a, b)
    # Channel-wise mean: 1 + 1 + degraded_red, all averaged.
    val_per_channel_expected = (1.0 + 1.0 + ssim(a[..., 0], b[..., 0])) / 3.0
    assert val_color == pytest.approx(val_per_channel_expected, abs=1e-6)


# ---------------------------------------------------------------------------
# image_evaluate function on a synthetic image-pair set.
# ---------------------------------------------------------------------------


def test_image_evaluate_pairs_by_filename(tmp_path: Path) -> None:
    rendered, reference = _build_image_pair_fixture(tmp_path)
    result = image_evaluate(
        ImageEvalRequest(rendered_dir=rendered, reference_dir=reference)
    )

    assert result.summary["pairs_evaluated"] == 4
    assert result.summary["pairs_missing_in_reference"] == 0
    assert result.summary["pairs_size_mismatch"] == 0

    # Every pair gets a PSNR and SSIM number.
    for entry in result.pairs:
        assert "psnr" in entry
        assert "ssim" in entry
        assert np.isfinite(entry["psnr"])
        assert 0.0 < entry["ssim"] <= 1.0

    # Aggregates are populated.
    assert result.summary["psnr_mean"] is not None
    assert result.summary["ssim_mean"] is not None
    assert result.summary["psnr_min"] <= result.summary["psnr_max"]


def test_image_evaluate_skips_missing_reference(tmp_path: Path) -> None:
    rendered, reference = _build_image_pair_fixture(tmp_path, n_pairs=3)
    # Drop one reference file -> that pair must be counted as missing,
    # not silently included.
    (reference / "img_00.png").unlink()
    result = image_evaluate(
        ImageEvalRequest(rendered_dir=rendered, reference_dir=reference)
    )
    assert result.summary["pairs_evaluated"] == 2
    assert result.summary["pairs_missing_in_reference"] == 1


def test_image_evaluate_skips_size_mismatch(tmp_path: Path) -> None:
    rendered, reference = _build_image_pair_fixture(tmp_path, n_pairs=3)
    # Overwrite one reference with a differently-sized image.
    import matplotlib.pyplot as plt

    plt.imsave(
        str(reference / "img_00.png"),
        np.zeros((40, 40, 3), dtype=np.float64),
    )
    result = image_evaluate(
        ImageEvalRequest(rendered_dir=rendered, reference_dir=reference)
    )
    assert result.summary["pairs_evaluated"] == 2
    assert result.summary["pairs_size_mismatch"] == 1


def test_image_evaluate_rejects_unknown_metric(tmp_path: Path) -> None:
    rendered, reference = _build_image_pair_fixture(tmp_path)
    with pytest.raises(ValueError, match="Unknown metric"):
        image_evaluate(
            ImageEvalRequest(
                rendered_dir=rendered,
                reference_dir=reference,
                metrics=("psnr", "made-up-metric"),
            )
        )


def test_image_evaluate_max_pairs_caps_iteration(tmp_path: Path) -> None:
    rendered, reference = _build_image_pair_fixture(tmp_path, n_pairs=5)
    result = image_evaluate(
        ImageEvalRequest(
            rendered_dir=rendered, reference_dir=reference, max_pairs=2
        )
    )
    assert result.summary["pairs_evaluated"] == 2


def test_image_evaluate_frequency_consistency_metadata(tmp_path: Path) -> None:
    rendered, reference = _build_image_pair_fixture(tmp_path, n_pairs=1)
    result = image_evaluate(
        ImageEvalRequest(
            rendered_dir=rendered,
            reference_dir=reference,
            metrics=("frequency_consistency",),
        )
    )

    assert result.summary["frequency_consistency_mean"] is not None
    assert 0.0 <= result.pairs[0]["frequency_consistency"] <= 1.0
    contract = result.metadata["frequency_consistency"]
    assert contract["filter"] == "5x5 Laplacian-of-Gaussian"
    assert contract["padding"] == "zero"
    assert contract["flat_reference_policy"] == (
        "both flat = 0; candidate non-flat = 1"
    )


def test_dreamsim_metric_uses_injected_backend_without_model_download(tmp_path: Path) -> None:
    rendered, reference = _build_image_pair_fixture(tmp_path, n_pairs=2)
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def fake_dreamsim(candidate: np.ndarray, target: np.ndarray) -> float:
        calls.append((candidate.shape, target.shape))
        return float(np.mean(np.abs(candidate - target)))

    result = image_evaluate(
        ImageEvalRequest(
            rendered_dir=rendered,
            reference_dir=reference,
            metrics=("dreamsim_distance",),
            metric_functions={"dreamsim_distance": fake_dreamsim},
        )
    )
    assert len(calls) == 2
    assert result.summary["dreamsim_distance_mean"] is not None
    assert all("dreamsim_distance" in pair for pair in result.pairs)


def test_dreamsim_missing_dependency_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    original_import = builtins.__import__

    def reject_dreamsim(name: str, *args: object, **kwargs: object) -> object:
        if name == "dreamsim":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_dreamsim)
    with pytest.raises(ValueError, match="cloudanalyzer\\[perceptual\\]"):
        dreamsim_distance(np.zeros((4, 4, 3)), np.zeros((4, 4, 3)))
    assert "downloads model weights" in DREAMSIM_INSTALL_HINT


# ---------------------------------------------------------------------------
# CLI integration.
# ---------------------------------------------------------------------------


def test_cli_image_evaluate_emits_summary(tmp_path: Path) -> None:
    rendered, reference = _build_image_pair_fixture(tmp_path)
    output_json = tmp_path / "out.json"
    cmd = [
        sys.executable,
        "-m",
        "cloudanalyzer_cli.main",
        "image-evaluate",
        str(rendered),
        str(reference),
        "--output-json",
        str(output_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)

    payload = json.loads(output_json.read_text())
    assert payload["summary"]["pairs_evaluated"] == 4
    assert payload["metadata"]["metrics"] == ["psnr", "ssim"]
    assert len(payload["pairs"]) == 4
    # Aggregates make sense.
    assert payload["summary"]["psnr_mean"] > 20.0
    assert 0.5 <= payload["summary"]["ssim_mean"] <= 1.0


def test_cli_image_evaluate_format_json(tmp_path: Path) -> None:
    rendered, reference = _build_image_pair_fixture(tmp_path, n_pairs=2)
    cmd = [
        sys.executable,
        "-m",
        "cloudanalyzer_cli.main",
        "image-evaluate",
        str(rendered),
        str(reference),
        "--metrics",
        "psnr",
        "--format-json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    # --format-json must produce parseable JSON on stdout.
    payload = json.loads(proc.stdout)
    assert payload["summary"]["pairs_evaluated"] == 2
    assert payload["metadata"]["metrics"] == ["psnr"]
