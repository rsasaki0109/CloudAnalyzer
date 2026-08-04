#!/usr/bin/env python3
"""Build the point-cloud triptych used in the root README.

The source clouds are the deterministic perception demo artifacts.  Keep the
sample seed fixed so the checked-in figure is stable across regenerations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import open3d as o3d

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/images/readme-pointcloud-triptych.png"
SEED = 20260804
MAX_POINTS = 14_000

SOURCES = (
    (
        "REFERENCE",
        ROOT / "docs/demo/perception/reference_scene.pcd",
        "61,781 pts · frozen baseline",
        "#8be9fd",
    ),
    (
        "DEEP BASELINE",
        ROOT / "docs/demo/perception/candidates/deep_baseline.pcd",
        "34,370 pts · AUC 0.9515 · PASS",
        "#83e6c2",
    ),
    (
        "NON-DEEP BASELINE",
        ROOT / "docs/demo/perception/candidates/nondeep_baseline.pcd",
        "16,906 pts · AUC 0.6648 · FAIL",
        "#ff9f68",
    ),
)


def load_points(path: Path, rng: np.random.Generator) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float64)
    if points.size == 0:
        raise RuntimeError(f"point cloud is empty: {path}")
    if len(points) > MAX_POINTS:
        indices = rng.choice(len(points), size=MAX_POINTS, replace=False)
        points = points[np.sort(indices)]
    return points


def render(output: Path) -> None:
    rng = np.random.default_rng(SEED)
    clouds = [load_points(path, rng) for _, path, _, _ in SOURCES]
    all_points = np.concatenate(clouds, axis=0)
    z_min, z_max = np.percentile(all_points[:, 2], [1, 99])
    norm = Normalize(vmin=z_min, vmax=z_max, clip=True)
    cmap = mpl.colormaps["viridis"]

    fig = plt.figure(figsize=(18, 6.4), dpi=120, facecolor="#0b1020")
    fig.text(
        0.045,
        0.925,
        "POINT-CLOUD REGRESSION / RELLIS-3D FRAME 000001",
        color="#aab8d5",
        fontsize=12,
        fontweight="bold",
        family="DejaVu Sans",
        ha="left",
        va="top",
        alpha=0.95,
    )
    fig.text(
        0.045,
        0.865,
        "Same reference scene. Different artifact quality. One deterministic gate.",
        color="#f6f8ff",
        fontsize=22,
        fontweight="bold",
        family="DejaVu Sans",
        ha="left",
        va="top",
    )

    for index, ((label, _, detail, accent), points) in enumerate(zip(SOURCES, clouds)):
        ax = fig.add_subplot(1, 3, index + 1, projection="3d")
        ax.set_facecolor("#111d38")
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=cmap(norm(points[:, 2])),
            s=0.72,
            alpha=0.82,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )
        ax.view_init(elev=28, azim=-62)
        ax.set_axis_off()
        ax.set_box_aspect((1.2, 1.0, 0.34))
        ax.text2D(
            0.04,
            0.97,
            label,
            transform=ax.transAxes,
            color=accent,
            fontsize=12,
            fontweight="bold",
            family="DejaVu Sans",
            va="top",
        )
        ax.text2D(
            0.04,
            0.90,
            detail,
            transform=ax.transAxes,
            color="#c9d4ea",
            fontsize=9.5,
            family="DejaVu Sans",
            va="top",
        )

    fig.text(
        0.045,
        0.055,
        "Deterministic 14k-point display sample · elevation-colored · source artifacts and batch metrics are checked in",
        color="#7183a7",
        fontsize=9.5,
        family="DejaVu Sans",
        ha="left",
        va="bottom",
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.79, bottom=0.11, wspace=0.015)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.output.resolve())
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
