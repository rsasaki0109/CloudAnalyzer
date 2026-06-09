"""Optional gsplat backend for rendering 3D Gaussian Splatting PLY files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from ca.core.cameras import CameraFrame, c2w_to_viewmat
from ca.geometry import _read_ply_vertices

GS_INSTALL_HINT = (
    "3DGS rendering requires optional dependencies.\n"
    'Install with: pip install "cloudanalyzer[gs]"'
)

_SH_C0 = 0.28209479177387814
_GAUSSIAN_SCALE_FIELDS = ("scale_0", "scale_1", "scale_2")
_GAUSSIAN_ROT_FIELDS = ("rot_0", "rot_1", "rot_2", "rot_3")
_GAUSSIAN_DC_FIELDS = ("f_dc_0", "f_dc_1", "f_dc_2")


@dataclass(frozen=True, slots=True)
class GaussianSplatScene:
    means: np.ndarray
    scales_log: np.ndarray
    quats_wxyz: np.ndarray
    opacity_logits: np.ndarray
    colors_rgb: np.ndarray
    sh_degree: int | None


def require_gsplat() -> Any:
    try:
        from gsplat.rendering import rasterization

        return rasterization
    except ImportError as exc:
        raise ValueError(GS_INSTALL_HINT) from exc


def require_torch() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ValueError(GS_INSTALL_HINT) from exc


def _sigmoid(values: np.ndarray) -> np.ndarray:
    out = 1.0 / (1.0 + np.exp(-values))
    return cast(np.ndarray, out.astype(np.float64, copy=False))


def load_gaussian_splat_ply(path: str | Path) -> GaussianSplatScene:
    """Load a standard 3DGS export PLY for rendering."""

    path = Path(path)
    wanted = (
        "x",
        "y",
        "z",
        "opacity",
        *_GAUSSIAN_SCALE_FIELDS,
        *_GAUSSIAN_ROT_FIELDS,
        *_GAUSSIAN_DC_FIELDS,
    )
    fields = _read_ply_vertices(path, wanted)
    for axis in ("x", "y", "z", "opacity"):
        if axis not in fields:
            raise ValueError(f"{path}: 3DGS PLY missing `{axis}` property")
    for name in _GAUSSIAN_SCALE_FIELDS + _GAUSSIAN_ROT_FIELDS:
        if name not in fields:
            raise ValueError(f"{path}: 3DGS PLY missing `{name}` property")

    means = np.column_stack([fields["x"], fields["y"], fields["z"]]).astype(np.float64)
    scales_log = np.column_stack([fields[name] for name in _GAUSSIAN_SCALE_FIELDS]).astype(
        np.float64
    )
    quats_wxyz = np.column_stack([fields[name] for name in _GAUSSIAN_ROT_FIELDS]).astype(
        np.float64
    )
    opacity_logits = np.asarray(fields["opacity"], dtype=np.float64)

    if all(name in fields for name in _GAUSSIAN_DC_FIELDS):
        dc = np.column_stack([fields[name] for name in _GAUSSIAN_DC_FIELDS]).astype(np.float64)
        colors_rgb = np.clip(0.5 + _SH_C0 * dc, 0.0, 1.0)
        sh_degree = 0
    else:
        colors_rgb = np.full((means.shape[0], 3), 0.75, dtype=np.float64)
        sh_degree = None

    return GaussianSplatScene(
        means=means,
        scales_log=scales_log,
        quats_wxyz=quats_wxyz,
        opacity_logits=opacity_logits,
        colors_rgb=colors_rgb,
        sh_degree=sh_degree,
    )


def _select_device(torch: Any, requested: str | None) -> Any:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def render_gaussian_views(
    scene: GaussianSplatScene,
    frames: tuple[CameraFrame, ...],
    output_dir: str | Path,
    *,
    opacity_threshold: float | None = None,
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
    device: str | None = None,
) -> list[Path]:
    """Rasterize ``scene`` at each camera frame and write PNGs under ``output_dir``."""

    rasterization = require_gsplat()
    torch = require_torch()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    means = scene.means
    scales_log = scene.scales_log
    quats = scene.quats_wxyz
    opacity_logits = scene.opacity_logits
    colors_rgb = scene.colors_rgb

    if opacity_threshold is not None:
        keep = _sigmoid(opacity_logits) >= float(opacity_threshold)
        means = means[keep]
        scales_log = scales_log[keep]
        quats = quats[keep]
        opacity_logits = opacity_logits[keep]
        colors_rgb = colors_rgb[keep]

    if means.shape[0] == 0:
        raise ValueError("No splats left after opacity filtering")

    dev = _select_device(torch, device)
    means_t = torch.tensor(means, dtype=torch.float32, device=dev)
    scales_t = torch.exp(torch.tensor(scales_log, dtype=torch.float32, device=dev))
    quats_t = torch.tensor(quats, dtype=torch.float32, device=dev)
    opacities_t = torch.sigmoid(torch.tensor(opacity_logits, dtype=torch.float32, device=dev))

    if scene.sh_degree is None:
        colors_t = torch.tensor(colors_rgb, dtype=torch.float32, device=dev)
        sh_degree = None
    else:
        colors_t = torch.tensor(colors_rgb, dtype=torch.float32, device=dev).unsqueeze(1)
        sh_degree = scene.sh_degree

    background = torch.tensor(background_rgb, dtype=torch.float32, device=dev)

    written: list[Path] = []
    import matplotlib.pyplot as plt

    for frame in frames:
        viewmat = torch.tensor(
            c2w_to_viewmat(frame.c2w)[None, ...],
            dtype=torch.float32,
            device=dev,
        )
        kmat = torch.tensor(
            [[[frame.fx, 0.0, frame.cx], [0.0, frame.fy, frame.cy], [0.0, 0.0, 1.0]]],
            dtype=torch.float32,
            device=dev,
        )
        render_colors, _alphas, _meta = rasterization(
            means_t,
            quats_t,
            scales_t,
            opacities_t,
            colors_t,
            viewmat,
            kmat,
            int(frame.width),
            int(frame.height),
            sh_degree=sh_degree,
            backgrounds=background,
        )
        image = render_colors[0].detach().cpu().numpy()
        image = np.clip(image, 0.0, 1.0)
        out_path = output_dir / frame.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(out_path), image)
        written.append(out_path)
    return written


__all__ = [
    "GS_INSTALL_HINT",
    "GaussianSplatScene",
    "load_gaussian_splat_ply",
    "render_gaussian_views",
    "require_gsplat",
    "require_torch",
]
