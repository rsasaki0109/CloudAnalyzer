#!/usr/bin/env python3
"""Generate the synthetic-room 3DGS demo under benchmarks/3dgs/.

Layout produced::

    benchmarks/3dgs/synthetic-room/
    ├── README.md
    ├── reference.pcd
    ├── transforms.json
    ├── reference/               # ground-truth renders for ca rendered-evaluate
    ├── gaussians.ply
    └── gaussians_dense.ply

Both PLY files are ASCII for reviewable diffs. Reference PNGs are generated
via gsplat when ``cloudanalyzer[gs]`` is installed (required in CI).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "cloudanalyzer"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "3dgs" / "synthetic-room"
_SH_C0 = 0.28209479177387814
_RENDER_WIDTH = 256
_RENDER_HEIGHT = 256
_RENDER_FOV_X = 0.85


def _logit(alpha: float) -> float:
    alpha = float(np.clip(alpha, 1e-4, 1.0 - 1e-4))
    return math.log(alpha / (1.0 - alpha))


def _planar_room(seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xs = np.linspace(-5.0, 5.0, 14)
    ys = np.linspace(-5.0, 5.0, 14)
    xx, yy = np.meshgrid(xs, ys)
    floor = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    floor += rng.normal(0, 0.01, size=floor.shape)

    wall_x = np.linspace(-4.5, 4.5, 30)
    wall_z = np.linspace(0.0, 1.5, 6)
    wall_xs, wall_zs = np.meshgrid(wall_x, wall_z)
    pos = np.column_stack(
        [wall_xs.ravel(), np.full(wall_xs.size, 5.0), wall_zs.ravel()]
    )
    neg = np.column_stack(
        [wall_xs.ravel(), np.full(wall_xs.size, -5.0), wall_zs.ravel()]
    )
    pos += rng.normal(0, 0.01, size=pos.shape)
    neg += rng.normal(0, 0.01, size=neg.shape)

    return np.vstack([floor, pos, neg]).astype(np.float64)


def _rgb_to_sh_dc(rgb: np.ndarray) -> np.ndarray:
    return (rgb - 0.5) / _SH_C0


def _position_colors(points: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-color from normalized xyz."""

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    norm = (points - mins) / span
    colors = np.clip(
        np.stack(
            [
                0.25 + 0.55 * norm[:, 0],
                0.20 + 0.50 * norm[:, 1],
                0.30 + 0.45 * norm[:, 2],
            ],
            axis=1,
        ),
        0.05,
        0.95,
    )
    return colors.astype(np.float64)


def _write_pcd(points: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def _write_ascii_gaussian_ply(
    points: np.ndarray,
    opacity_logits: np.ndarray,
    scales_log: np.ndarray,
    quats_wxyz: np.ndarray,
    sh_dc: np.ndarray,
    path: Path,
) -> None:
    n = points.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        "end_header",
    ]
    rows = [
        (
            f"{x:.6f} {y:.6f} {z:.6f} {op:.6f} "
            f"{s0:.6f} {s1:.6f} {s2:.6f} "
            f"{rw:.6f} {rx:.6f} {ry:.6f} {rz:.6f} "
            f"{dc0:.6f} {dc1:.6f} {dc2:.6f}"
        )
        for (x, y, z), op, (s0, s1, s2), (rw, rx, ry, rz), (dc0, dc1, dc2) in zip(
            points, opacity_logits, scales_log, quats_wxyz, sh_dc
        )
    ]
    path.write_text("\n".join(header + rows) + "\n", encoding="ascii")


def _orbit_c2w(angle_rad: float, radius: float = 8.0, height: float = 2.5) -> np.ndarray:
    eye = np.array(
        [radius * math.sin(angle_rad), radius * math.cos(angle_rad), height],
        dtype=np.float64,
    )
    target = np.array([0.0, 0.0, 0.4], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up_cam = np.cross(right, forward)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = up_cam
    c2w[:3, 2] = -forward
    c2w[:3, 3] = eye
    return c2w


def _write_transforms_json(output_dir: Path, n_views: int = 8) -> None:
    fl_x = 0.5 * _RENDER_WIDTH / math.tan(0.5 * _RENDER_FOV_X)
    frames = []
    for index in range(n_views):
        angle = (2.0 * math.pi * index) / float(n_views)
        frames.append(
            {
                "file_path": f"view_{index:02d}.png",
                "transform_matrix": _orbit_c2w(angle).tolist(),
            }
        )
    payload = {
        "camera_angle_x": _RENDER_FOV_X,
        "w": _RENDER_WIDTH,
        "h": _RENDER_HEIGHT,
        "fl_x": fl_x,
        "fl_y": fl_x,
        "cx": _RENDER_WIDTH / 2.0,
        "cy": _RENDER_HEIGHT / 2.0,
        "frames": frames,
    }
    (output_dir / "transforms.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _render_reference_views(output_dir: Path, splat_path: Path) -> None:
    from ca.core.cameras import load_cameras
    from ca.core.gs_renderer import load_gaussian_splat_ply, render_gaussian_views

    cameras = load_cameras(output_dir / "transforms.json")
    scene = load_gaussian_splat_ply(splat_path)
    reference_dir = output_dir / "reference"
    if reference_dir.exists():
        for child in reference_dir.glob("*.png"):
            child.unlink()
    render_gaussian_views(scene, cameras.frames, reference_dir, device="cuda")


README_TEMPLATE = """\
# synthetic-room (3DGS demo)

A tiny Gaussian Splatting-style sample to smoke-test ``ca geometry-evaluate`` and
``ca rendered-evaluate`` without external data. Regenerate deterministically with
``scripts/build_synthetic_3dgs_demo.py`` (requires ``pip install 'cloudanalyzer[gs]'``
for reference PNG generation).

| File | Purpose |
|---|---|
| `reference.pcd` | Reference scan for geometry QA |
| `gaussians.ply` | Mixed-opacity 3DGS export |
| `gaussians_dense.ply` | High-opacity-only sanity case |
| `transforms.json` | nerfstudio camera poses for rendering |
| `reference/` | Ground-truth PNG renders of ``gaussians_dense.ply`` |

Example commands:

```bash
# Geometry-only QA
ca geometry-evaluate benchmarks/3dgs/synthetic-room/gaussians_dense.ply \\
                    benchmarks/3dgs/synthetic-room/reference.pcd

# Render + photometric + geometry combined report
ca rendered-evaluate benchmarks/3dgs/synthetic-room/gaussians_dense.ply \\
    benchmarks/3dgs/synthetic-room/reference \\
    --cameras benchmarks/3dgs/synthetic-room/transforms.json \\
    --reference-pointcloud benchmarks/3dgs/synthetic-room/reference.pcd \\
    --metrics psnr,ssim,lpips --report /tmp/rendered-report.html

# CI gate (kind: rendered) - dogfooded in rendered-self-qa.yml
ca check benchmarks/3dgs/synthetic-room/configs/suite-rendered.cloudanalyzer.yaml
```
"""


def _identity_quaternions(n: int) -> np.ndarray:
    quats = np.zeros((n, 4), dtype=np.float64)
    quats[:, 0] = 1.0
    return quats


def build(output_dir: Path, *, skip_renders: bool = False) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(17)
    reference = _planar_room()
    _write_pcd(reference, output_dir / "reference.pcd")

    high_mask = rng.random(reference.shape[0]) < 0.5
    centers_high = reference[high_mask] + rng.normal(0, 0.01, size=(int(high_mask.sum()), 3))
    centers_low = reference[~high_mask] + rng.normal(0, 0.4, size=(int((~high_mask).sum()), 3))

    points = np.vstack([centers_high, centers_low])
    high_alpha = rng.uniform(0.7, 0.95, size=centers_high.shape[0])
    low_alpha = rng.uniform(0.02, 0.18, size=centers_low.shape[0])
    alpha = np.concatenate([high_alpha, low_alpha])
    logits = np.array([_logit(a) for a in alpha], dtype=np.float64)

    n = points.shape[0]
    log_sigma_high = np.log(0.02)
    log_sigma_low = np.log(0.04)
    scales_log = np.empty((n, 3), dtype=np.float64)
    scales_log[: centers_high.shape[0]] = log_sigma_high
    scales_log[centers_high.shape[0] :] = log_sigma_low
    quats = _identity_quaternions(n)
    sh_dc = _rgb_to_sh_dc(_position_colors(points))

    order = rng.permutation(n)
    points = points[order]
    logits = logits[order]
    scales_log = scales_log[order]
    quats = quats[order]
    sh_dc = sh_dc[order]

    _write_ascii_gaussian_ply(
        points, logits, scales_log, quats, sh_dc, output_dir / "gaussians.ply"
    )

    dense_count = centers_high.shape[0]
    dense_sh = _rgb_to_sh_dc(_position_colors(centers_high))
    _write_ascii_gaussian_ply(
        centers_high,
        np.array([_logit(a) for a in high_alpha], dtype=np.float64),
        np.full((dense_count, 3), log_sigma_high, dtype=np.float64),
        _identity_quaternions(dense_count),
        dense_sh,
        output_dir / "gaussians_dense.ply",
    )

    _write_transforms_json(output_dir)
    dense_path = output_dir / "gaussians_dense.ply"
    if not skip_renders:
        _render_reference_views(output_dir, dense_path)

    (output_dir / "README.md").write_text(README_TEMPLATE, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--skip-renders",
        action="store_true",
        help="Skip gsplat reference PNG generation (geometry/PLY only).",
    )
    args = parser.parse_args()
    build(args.output, skip_renders=args.skip_renders)
    print(f"Wrote synthetic-room 3DGS demo to {args.output}")


if __name__ == "__main__":
    main()
