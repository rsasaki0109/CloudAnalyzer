"""Stable contract for ``ca image-evaluate``: score a set of rendered
images against a reference (ground-truth) set on standard photometric
metrics (PSNR + SSIM).

This is the first photometric eval module in CloudAnalyzer. It is the
foundation for a future ``ca rendered-evaluate`` that would tie 3DGS
PLY → rendered images → photometric metrics into the existing
``ca run-evaluate`` style pipeline. For now, it scores **two
already-rendered image directories** that share filenames.

Both PSNR and SSIM are implemented in pure numpy + ``scipy.ndimage`` so
no new heavy deps land in CloudAnalyzer's import graph; the only
new touch is ``scipy.ndimage.gaussian_filter`` (scipy is already a
required dep).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ImageEvalRequest:
    """Inputs to a single image-pair-set evaluation run."""

    rendered_dir: Path
    """Directory containing the candidate (rendered) images."""

    reference_dir: Path
    """Directory containing the ground-truth images. Pairs are matched
    by filename (basename + extension)."""

    metrics: tuple[str, ...] = ("psnr", "ssim")
    """Which photometric metrics to compute. Order is preserved in the
    summary aggregates."""

    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")
    """Filename extensions to discover under ``rendered_dir``."""

    ssim_window_size: int = 11
    """Side length of the Gaussian window used in SSIM (Wang & Bovik
    2004 use 11)."""

    ssim_sigma: float = 1.5
    """Standard deviation of the Gaussian window used in SSIM."""

    max_pairs: int | None = None
    """Optional cap on number of pairs evaluated. ``None`` evaluates all."""


@dataclass(slots=True)
class ImageEvalResult:
    """Outputs of a single image-pair-set evaluation run."""

    pairs: list[dict[str, Any]] = field(default_factory=list)
    """One entry per matched ``(rendered, reference)`` pair. Each
    entry carries: ``filename``, ``shape``, and one float per metric
    in :class:`ImageEvalRequest.metrics`."""

    summary: dict[str, Any] = field(default_factory=dict)
    """Aggregate stats (per-metric mean / median / min / max) plus
    counts: ``pairs_evaluated``, ``pairs_missing_in_reference``,
    ``pairs_size_mismatch``."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Snapshot of the request knobs and computation backend. Echoed
    into ``ca image-evaluate --format-json`` so two runs can be proven
    to share config."""


# ---------------------------------------------------------------------------
# Image IO helpers
# ---------------------------------------------------------------------------


def _load_image_rgb(path: Path) -> np.ndarray:
    """Load an image as an ``(H, W, 3)`` float64 array in ``[0, 1]``.

    Uses matplotlib.pyplot.imread, which is already a required CloudAnalyzer
    dependency (no new imports). Alpha channels are dropped; grayscale
    images are promoted by replicating the single channel.
    """

    import matplotlib.pyplot as plt

    arr = plt.imread(str(path))
    # imread returns float in [0,1] for PNG/JPEG and uint8 for some formats.
    if arr.dtype.kind in {"u", "i"}:
        arr = arr.astype(np.float64) / 255.0
    else:
        arr = arr.astype(np.float64, copy=False)

    if arr.ndim == 2:
        # Grayscale -> replicate to RGB.
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        # Drop alpha.
        arr = arr[..., :3]
    elif arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported image shape {arr.shape} for {path}")

    return np.clip(arr, 0.0, 1.0)


def _list_image_files(
    directory: Path, extensions: tuple[str, ...]
) -> list[Path]:
    """Return image files under ``directory`` matching ``extensions``,
    sorted lexicographically."""

    if not directory.is_dir():
        raise ValueError(f"image directory not found: {directory}")
    exts = {e.lower() for e in extensions}
    hits = [p for p in directory.iterdir() if p.suffix.lower() in exts]
    return sorted(hits)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def psnr(rendered: np.ndarray, reference: np.ndarray, *, max_value: float = 1.0) -> float:
    """Peak signal-to-noise ratio for images in ``[0, max_value]``.

    Identical inputs return ``+inf``.
    """

    if rendered.shape != reference.shape:
        raise ValueError(
            f"PSNR input shape mismatch: {rendered.shape} vs {reference.shape}"
        )
    diff = rendered - reference
    mse = float(np.mean(diff * diff))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * float(np.log10((max_value * max_value) / mse))


def ssim(
    rendered: np.ndarray,
    reference: np.ndarray,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
    max_value: float = 1.0,
) -> float:
    """Structural similarity index (Wang & Bovik 2004).

    Operates on ``(H, W)`` grayscale or ``(H, W, C)`` color arrays;
    color inputs return the per-channel mean. The Gaussian window is
    applied via ``scipy.ndimage.gaussian_filter`` (existing dep).
    """

    if rendered.shape != reference.shape:
        raise ValueError(
            f"SSIM input shape mismatch: {rendered.shape} vs {reference.shape}"
        )

    # SSIM constants for max_value=1.0 input (Wang & Bovik defaults).
    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2

    if rendered.ndim == 3:
        # Channel-wise mean.
        return float(
            np.mean(
                [
                    ssim(
                        rendered[..., k],
                        reference[..., k],
                        window_size=window_size,
                        sigma=sigma,
                        max_value=max_value,
                    )
                    for k in range(rendered.shape[-1])
                ]
            )
        )

    from scipy.ndimage import gaussian_filter

    # Truncate the Gaussian to keep its effective support near ``window_size``.
    radius = (window_size - 1) // 2
    truncate = float(radius) / float(sigma) if sigma > 0 else 1.0

    mu_x = gaussian_filter(rendered, sigma=sigma, truncate=truncate)
    mu_y = gaussian_filter(reference, sigma=sigma, truncate=truncate)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = gaussian_filter(rendered * rendered, sigma=sigma, truncate=truncate) - mu_x_sq
    sigma_y_sq = gaussian_filter(reference * reference, sigma=sigma, truncate=truncate) - mu_y_sq
    sigma_xy = gaussian_filter(rendered * reference, sigma=sigma, truncate=truncate) - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = num / den
    return float(np.mean(ssim_map))


# ---------------------------------------------------------------------------
# Driver entry point
# ---------------------------------------------------------------------------


_METRIC_FUNCS: dict[str, Any] = {
    "psnr": psnr,
    "ssim": ssim,
}


def image_evaluate(request: ImageEvalRequest) -> ImageEvalResult:
    """Score every image in ``rendered_dir`` against its same-named
    counterpart in ``reference_dir`` on each requested metric.

    Unknown metric names raise :class:`ValueError`. Files present in
    ``rendered_dir`` but missing from ``reference_dir`` are counted in
    ``summary.pairs_missing_in_reference`` and skipped; size-mismatched
    pairs are counted in ``summary.pairs_size_mismatch`` and skipped.
    """

    for m in request.metrics:
        if m not in _METRIC_FUNCS:
            raise ValueError(
                f"Unknown metric '{m}'. Available: {sorted(_METRIC_FUNCS)}"
            )

    rendered_files = _list_image_files(request.rendered_dir, request.extensions)
    if request.max_pairs is not None:
        rendered_files = rendered_files[: int(request.max_pairs)]

    pairs: list[dict[str, Any]] = []
    missing = 0
    size_mismatch = 0

    for rendered_path in rendered_files:
        reference_path = request.reference_dir / rendered_path.name
        if not reference_path.is_file():
            missing += 1
            continue
        rendered_img = _load_image_rgb(rendered_path)
        reference_img = _load_image_rgb(reference_path)
        if rendered_img.shape != reference_img.shape:
            size_mismatch += 1
            continue

        entry: dict[str, Any] = {
            "filename": rendered_path.name,
            "shape": list(rendered_img.shape),
        }
        for m in request.metrics:
            fn = _METRIC_FUNCS[m]
            if m == "ssim":
                value = fn(
                    rendered_img,
                    reference_img,
                    window_size=request.ssim_window_size,
                    sigma=request.ssim_sigma,
                )
            else:
                value = fn(rendered_img, reference_img)
            entry[m] = float(value)
        pairs.append(entry)

    summary: dict[str, Any] = {
        "pairs_evaluated": len(pairs),
        "pairs_missing_in_reference": missing,
        "pairs_size_mismatch": size_mismatch,
    }
    for m in request.metrics:
        # Filter out infs so the aggregate stays meaningful when a pair
        # is bit-identical (PSNR=+inf).
        raw = [p[m] for p in pairs if np.isfinite(p[m])]
        if raw:
            summary[f"{m}_mean"] = float(fmean(raw))
            summary[f"{m}_median"] = float(median(raw))
            summary[f"{m}_min"] = float(min(raw))
            summary[f"{m}_max"] = float(max(raw))
        else:
            summary[f"{m}_mean"] = None
            summary[f"{m}_median"] = None
            summary[f"{m}_min"] = None
            summary[f"{m}_max"] = None

    metadata: dict[str, Any] = {
        "metrics": list(request.metrics),
        "ssim_window_size": int(request.ssim_window_size),
        "ssim_sigma": float(request.ssim_sigma),
        "extensions": list(request.extensions),
        "rendered_dir": str(request.rendered_dir),
        "reference_dir": str(request.reference_dir),
        "backend": "scipy.ndimage.gaussian_filter",
    }

    return ImageEvalResult(pairs=pairs, summary=summary, metadata=metadata)


__all__ = [
    "ImageEvalRequest",
    "ImageEvalResult",
    "image_evaluate",
    "psnr",
    "ssim",
]
