#!/usr/bin/env python3
"""Generate the synthetic-room 3DGS demo under benchmarks/3dgs/.

Layout produced::

    benchmarks/3dgs/synthetic-room/
    ├── README.md
    ├── reference.pcd          # planar room with two walls (same shape as the SLAM demo)
    ├── gaussians.ply          # 3DGS-style PLY: xyz + opacity (logit), ~50% high alpha
    └── gaussians_dense.ply    # same scene but only high-opacity splats (sanity case)

Both PLY files are written as ASCII so the diff is reviewable and ``ca
geometry-evaluate`` tests can read them without binary plumbing. Real 3DGS
exports are binary little-endian; the geometry adapter supports both.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import open3d as o3d


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "3dgs" / "synthetic-room"


def _logit(alpha: float) -> float:
    """Inverse sigmoid; 3DGS stores opacity in logit space."""
    alpha = float(np.clip(alpha, 1e-4, 1.0 - 1e-4))
    return math.log(alpha / (1.0 - alpha))


def _planar_room(seed: int = 17) -> np.ndarray:
    """Same shape as the SLAM demo's room but seeded independently."""
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


def _write_pcd(points: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def _write_ascii_gaussian_ply(
    points: np.ndarray, opacity_logits: np.ndarray, path: Path
) -> None:
    if points.shape[0] != opacity_logits.shape[0]:
        raise ValueError("points and opacities must have equal length")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property float opacity",
        "end_header",
    ]
    rows = [
        f"{x:.6f} {y:.6f} {z:.6f} {op:.6f}"
        for (x, y, z), op in zip(points, opacity_logits)
    ]
    path.write_text("\n".join(header + rows) + "\n", encoding="ascii")


README_TEMPLATE = """\
# synthetic-room (3DGS demo)

A tiny Gaussian Splatting-style sample to smoke-test ``ca geometry-evaluate``
without external data. Both PLY files are deterministic; rerun
``scripts/build_synthetic_3dgs_demo.py`` to reproduce them byte-for-byte.

| File | Shape | Purpose |
|---|---|---|
| `reference.pcd` | planar room + two walls | Reference scan that all geometry evaluations score against |
| `gaussians.ply` | reference centers + small noise; mixed opacity | Half the splats are "high alpha" (rendered alpha ≥ 0.6); the rest are "low alpha" (≤ 0.2) so the opacity filter has something to drop |
| `gaussians_dense.ply` | only the high-alpha half | Sanity case: should score very close to the reference even without opacity filtering |

Example commands:

```bash
# Auto-detect representation; no filtering — sees all splats.
ca geometry-evaluate benchmarks/3dgs/synthetic-room/gaussians.ply \\
                    benchmarks/3dgs/synthetic-room/reference.pcd

# Filter out low-alpha splats before scoring (much closer to the reference).
ca geometry-evaluate benchmarks/3dgs/synthetic-room/gaussians.ply \\
                    benchmarks/3dgs/synthetic-room/reference.pcd \\
                    --opacity-threshold 0.5
```
"""


def build(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(17)
    reference = _planar_room()
    _write_pcd(reference, output_dir / "reference.pcd")

    # Build splats: each reference point becomes a splat with small noise.
    # Half get high alpha (clean), half get drift + low alpha (the "noise"
    # that an opacity threshold is supposed to discard).
    high_mask = rng.random(reference.shape[0]) < 0.5

    centers_high = reference[high_mask] + rng.normal(0, 0.01, size=(int(high_mask.sum()), 3))
    centers_low = reference[~high_mask] + rng.normal(0, 0.4, size=(int((~high_mask).sum()), 3))

    points = np.vstack([centers_high, centers_low])
    high_alpha = rng.uniform(0.7, 0.95, size=centers_high.shape[0])
    low_alpha = rng.uniform(0.02, 0.18, size=centers_low.shape[0])
    alpha = np.concatenate([high_alpha, low_alpha])
    logits = np.array([_logit(a) for a in alpha], dtype=np.float64)

    # Deterministic shuffle so the file ordering exercises the opacity
    # filter (otherwise high-alpha splats would all come first).
    order = rng.permutation(points.shape[0])
    points = points[order]
    logits = logits[order]

    _write_ascii_gaussian_ply(points, logits, output_dir / "gaussians.ply")

    # Dense variant = only high-alpha splats, preserved order.
    _write_ascii_gaussian_ply(
        centers_high,
        np.array([_logit(a) for a in high_alpha], dtype=np.float64),
        output_dir / "gaussians_dense.ply",
    )

    (output_dir / "README.md").write_text(README_TEMPLATE, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory"
    )
    args = parser.parse_args()
    build(args.output)
    print(f"Wrote synthetic-room 3DGS demo to {args.output}")


if __name__ == "__main__":
    main()
