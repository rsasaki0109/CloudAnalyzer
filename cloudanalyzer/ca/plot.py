"""Multi-comparison and 3D heatmap plotting."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from ca.io import load_point_cloud
from ca.metrics import compute_nn_distance
from ca.visualization import colorize, save_snapshot


def plot_multi_f1(
    results: list[dict],
    labels: list[str],
    output_path: str,
) -> None:
    """Plot multiple F1 curves on the same chart for comparison.

    Args:
        results: List of evaluate() result dicts.
        labels: Label for each result.
        output_path: Output image path (png).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for result, label in zip(results, labels):
        scores = result["f1_scores"]
        thresholds = [s["threshold"] for s in scores]
        f1s = [s["f1"] for s in scores]
        auc = result["auc"]
        ax.plot(thresholds, f1s, "o-", label=f"{label} (AUC={auc:.4f})", linewidth=2)

    ax.set_xlabel("Distance Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Comparison")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def heatmap3d(
    source_path: str,
    target_path: str,
    output_path: str,
    width: int = 1920,
    height: int = 1080,
) -> dict:
    """Render source point cloud colored by distance to target.

    Args:
        source_path: Source point cloud file.
        target_path: Target (reference) point cloud file.
        output_path: Output snapshot image path (png).
        width: Image width.
        height: Image height.

    Returns:
        Dict with distance stats and output path.
    """
    source = load_point_cloud(source_path)
    target = load_point_cloud(target_path)

    distances = compute_nn_distance(source, target)
    colorize(source, distances)
    save_snapshot(source, output_path, width=width, height=height)

    return {
        "source": source_path,
        "target": target_path,
        "output": output_path,
        "num_points": len(source.points),
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
    }
