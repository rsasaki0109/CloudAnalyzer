"""Stable contract for ``ca rendered-evaluate``.

Render a 3D Gaussian Splatting PLY at supplied camera poses, score the
images photometrically via :mod:`ca.core.image_evaluate`, and optionally
run cross-representation geometry QA via :func:`ca.geometry.evaluate_geometry`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ca.core.cameras import load_cameras
from ca.core.gs_renderer import (
    GS_INSTALL_HINT,
    load_gaussian_splat_ply,
    render_gaussian_views,
)
from ca.core.image_evaluate import ImageEvalRequest, image_evaluate
from ca.geometry import evaluate_geometry


RENDER_PROTOCOL_VERSION = "cloudanalyzer.rendered_eval.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _image_manifest(directory: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        entries.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return entries


def _camera_manifest(cameras: Any) -> dict[str, Any]:
    return {
        "source": cameras.source,
        "source_path": cameras.source_path,
        "convention": "camera_to_world_opengl; gsplat_colmap_viewmat",
        "frames": [
            {
                "name": frame.name,
                "width": frame.width,
                "height": frame.height,
                "fx": float(frame.fx),
                "fy": float(frame.fy),
                "cx": float(frame.cx),
                "cy": float(frame.cy),
                "c2w": np.asarray(frame.c2w, dtype=float).round(12).tolist(),
            }
            for frame in cameras.frames
        ],
    }


def _optional_runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for module_name in ("torch", "gsplat"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[module_name] = None
    return versions


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
    skip_render: bool = False
    max_pairs: int | None = None
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ssim_window_size: int = 11
    ssim_sigma: float = 1.5


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
    if request.skip_render:
        if rendered_dir is None:
            raise ValueError(
                "skip_render requires rendered_dir (pre-rendered PNG directory)"
            )
        if not rendered_dir.is_dir():
            raise ValueError(f"pre-rendered directory not found: {rendered_dir}")
        written = sorted(rendered_dir.glob("*.png"))
        if not written:
            raise ValueError(f"no PNG renders found under {rendered_dir}")
        renderer_backend = "pre-rendered"
    else:
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
                background_rgb=request.background_rgb,
                device=request.render_device,
            )
        except ValueError as exc:
            if GS_INSTALL_HINT.splitlines()[0] in str(exc):
                raise
            raise
        renderer_backend = "gsplat"

    image_result = image_evaluate(
        ImageEvalRequest(
            rendered_dir=rendered_dir,
            reference_dir=request.reference_dir,
            metrics=request.metrics,
            ssim_window_size=request.ssim_window_size,
            ssim_sigma=request.ssim_sigma,
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

    camera_manifest = _camera_manifest(cameras)
    reference_manifest = _image_manifest(request.reference_dir)
    protocol = {
        "name": RENDER_PROTOCOL_VERSION,
        "input": {
            "splat_path": str(request.splat_path),
            "splat_sha256": _sha256_file(request.splat_path),
            "camera_manifest": camera_manifest,
            "camera_sha256": _sha256_file(Path(cameras.source_path)),
            "reference_manifest": reference_manifest,
            "reference_manifest_sha256": _sha256_json(reference_manifest),
        },
        "image": {
            "pairing": "relative_filename",
            "value_range": [0.0, 1.0],
            "metrics": list(request.metrics),
            "ssim_window_size": request.ssim_window_size,
            "ssim_sigma": request.ssim_sigma,
            "max_pairs": request.max_pairs,
        },
        "render": {
            "backend": renderer_backend,
            "device": request.render_device,
            "background_rgb": list(request.background_rgb),
            "opacity_threshold": request.opacity_threshold,
            "frames": len(written),
        },
        "runtime": _optional_runtime_versions(),
    }
    protocol["sha256"] = _sha256_json(protocol)

    renderer = {
        "backend": renderer_backend,
        "frames_rendered": len(written),
        "rendered_dir": str(rendered_dir),
        "camera_source": cameras.source,
        "camera_path": cameras.source_path,
        "opacity_threshold": request.opacity_threshold,
        "background_rgb": list(request.background_rgb),
        "render_device": request.render_device,
        "splat_count": int(scene.means.shape[0]),
        "sh_degree": scene.sh_degree,
        "protocol_version": RENDER_PROTOCOL_VERSION,
        "protocol_sha256": protocol["sha256"],
    }

    metadata = {
        "splat_path": str(request.splat_path),
        "cameras_path": str(request.cameras_path),
        "reference_dir": str(request.reference_dir),
        "reference_pointcloud": (
            str(request.reference_pointcloud) if request.reference_pointcloud else None
        ),
        "metrics": list(request.metrics),
        "evaluation_protocol": protocol,
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
    "RENDER_PROTOCOL_VERSION",
    "RenderedEvalRequest",
    "RenderedEvalResult",
    "rendered_evaluate",
    "rendered_evaluate_to_dict",
]
