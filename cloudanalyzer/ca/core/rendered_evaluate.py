"""Stable contract for ``ca rendered-evaluate``.

Render a 3D Gaussian Splatting PLY at supplied camera poses, score the
images photometrically via :mod:`ca.core.image_evaluate`, and optionally
run cross-representation geometry QA via :func:`ca.geometry.evaluate_geometry`.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ca.core.cameras import load_cameras
from ca.core.gs_renderer import (
    GS_INSTALL_HINT,
    load_gaussian_splat_ply,
    render_gaussian_views,
)
from ca.core.image_evaluate import ImageEvalRequest, image_evaluate
from ca.geometry import evaluate_geometry


@dataclass(slots=True)
class RenderedEvalRequest:
    """Inputs to a rendered 3DGS evaluation run."""

    splat_path: Path
    cameras_path: Path
    reference_dir: Path
    metrics: tuple[str, ...] = ("psnr", "ssim")
    reference_pointcloud: Path | None = None
    opacity_threshold: float | None = None
    geometry_representation: str = "gaussian-points"
    geometry_opacity_threshold: float | None = None
    geometry_voxel: float | None = None
    geometry_splat_method: str = "centers"
    geometry_splat_samples: int = 8
    geometry_thresholds: list[float] | None = None
    render_device: str | None = None
    keep_rendered_dir: Path | None = None
    max_pairs: int | None = None


@dataclass(slots=True)
class RenderedEvalResult:
    photometric: dict[str, Any]
    geometry: dict[str, Any] | None = None
    renderer: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def rendered_evaluate(request: RenderedEvalRequest) -> RenderedEvalResult:
    """Render ``splat_path`` and score against ``reference_dir``."""

    if not request.splat_path.is_file():
        raise FileNotFoundError(f"3DGS PLY not found: {request.splat_path}")
    if not request.reference_dir.is_dir():
        raise ValueError(f"reference directory not found: {request.reference_dir}")

    cameras = load_cameras(request.cameras_path)
    scene = load_gaussian_splat_ply(request.splat_path)

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    rendered_dir = request.keep_rendered_dir
    if rendered_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="ca-rendered-")
        rendered_dir = Path(temp_dir.name)

    assert rendered_dir is not None
    rendered_dir.mkdir(parents=True, exist_ok=True)

    try:
        written = render_gaussian_views(
            scene,
            cameras.frames,
            rendered_dir,
            opacity_threshold=request.opacity_threshold,
            device=request.render_device,
        )
    except ValueError as exc:
        if GS_INSTALL_HINT.splitlines()[0] in str(exc):
            raise
        raise

    image_result = image_evaluate(
        ImageEvalRequest(
            rendered_dir=rendered_dir,
            reference_dir=request.reference_dir,
            metrics=request.metrics,
            max_pairs=request.max_pairs,
        )
    )

    geometry_result: dict[str, Any] | None = None
    if request.reference_pointcloud is not None:
        geometry_result = evaluate_geometry(
            str(request.splat_path),
            str(request.reference_pointcloud),
            representation=request.geometry_representation,
            opacity_threshold=request.geometry_opacity_threshold,
            voxel_size=request.geometry_voxel,
            thresholds=request.geometry_thresholds,
            splat_method=request.geometry_splat_method,
            splat_samples=request.geometry_splat_samples,
        )

    photometric = {
        "summary": image_result.summary,
        "pairs": image_result.pairs,
        "metadata": image_result.metadata,
    }

    renderer = {
        "backend": "gsplat",
        "frames_rendered": len(written),
        "rendered_dir": str(rendered_dir),
        "camera_source": cameras.source,
        "camera_path": cameras.source_path,
        "opacity_threshold": request.opacity_threshold,
        "splat_count": int(scene.means.shape[0]),
        "sh_degree": scene.sh_degree,
    }

    metadata = {
        "splat_path": str(request.splat_path),
        "cameras_path": str(request.cameras_path),
        "reference_dir": str(request.reference_dir),
        "reference_pointcloud": (
            str(request.reference_pointcloud) if request.reference_pointcloud else None
        ),
        "metrics": list(request.metrics),
    }

    return RenderedEvalResult(
        photometric=photometric,
        geometry=geometry_result,
        renderer=renderer,
        metadata=metadata,
    )


def rendered_evaluate_to_dict(result: RenderedEvalResult) -> dict[str, Any]:
    """JSON-serializable payload for CLI / reports."""

    payload: dict[str, Any] = {
        "photometric": result.photometric,
        "renderer": result.renderer,
        "metadata": result.metadata,
    }
    if result.geometry is not None:
        payload["geometry"] = result.geometry
    return payload


__all__ = [
    "RenderedEvalRequest",
    "RenderedEvalResult",
    "rendered_evaluate",
    "rendered_evaluate_to_dict",
]
