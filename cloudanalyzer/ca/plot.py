"""Multi-comparison and 3D heatmap plotting."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from ca.io import load_point_cloud
from ca.metrics import compute_nn_distance
from ca.pareto import quality_size_pareto_results, recommended_quality_size_result
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


def plot_quality_vs_size(
    results: list[dict],
    output_path: str,
) -> None:
    """Plot compression size ratio vs quality (AUC)."""
    compression_results = [
        item for item in results
        if item.get("compression") is not None
    ]
    if not compression_results:
        raise ValueError("No compression metrics available for quality-vs-size plot")

    fig, ax = plt.subplots(figsize=(10, 6))
    pareto_items = quality_size_pareto_results(results)
    pareto_item_ids = {id(item) for item in pareto_items}
    recommended_item = recommended_quality_size_result(results)

    for item in compression_results:
        compression = item["compression"]
        gate = item.get("quality_gate")
        color = "#dc2626" if gate is not None and not gate["passed"] else "#0f766e"
        is_pareto = id(item) in pareto_item_ids
        ax.scatter(
            compression["size_ratio"],
            item["auc"],
            s=110 if is_pareto else 70,
            marker="D" if is_pareto else "o",
            color=color,
            edgecolor="#1d4ed8" if is_pareto else "white",
            linewidth=1.2 if is_pareto else 0.8,
            zorder=3,
        )
        ax.annotate(
            Path(item["path"]).stem,
            (compression["size_ratio"], item["auc"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    if pareto_items:
        ax.plot(
            [item["compression"]["size_ratio"] for item in pareto_items],
            [item["auc"] for item in pareto_items],
            color="#1d4ed8",
            linewidth=1.5,
            linestyle="--",
            zorder=2,
        )

    if recommended_item is not None:
        recommended_compression = recommended_item["compression"]
        ax.scatter(
            recommended_compression["size_ratio"],
            recommended_item["auc"],
            s=260,
            marker="*",
            color="#f59e0b",
            edgecolor="#92400e",
            linewidth=1.0,
            zorder=4,
        )
        ax.annotate(
            "RECOMMENDED",
            (recommended_compression["size_ratio"], recommended_item["auc"]),
            textcoords="offset points",
            xytext=(10, -14),
            fontsize=8,
            color="#92400e",
            weight="bold",
        )

    ax.set_xlabel("Size Ratio (compressed / baseline)")
    ax.set_ylabel("AUC (F1)")
    ax.set_title("Quality vs Size")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.axhline(1.0, color="#d1d5db", linewidth=1, linestyle="--")
    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="#0f766e",
            markeredgecolor="white",
            markersize=8,
            label="Pass",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="#dc2626",
            markeredgecolor="white",
            markersize=8,
            label="Fail",
        ),
    ]
    if pareto_items:
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="D",
                color="#1d4ed8",
                linestyle="--",
                markerfacecolor="white",
                markeredgecolor="#1d4ed8",
                markersize=7,
                label="Pareto frontier",
            )
        )
    if recommended_item is not None:
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="*",
                linestyle="",
                markerfacecolor="#f59e0b",
                markeredgecolor="#92400e",
                markersize=11,
                label="Recommended",
            )
        )
    ax.legend(handles=legend_handles, loc="lower right")

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
