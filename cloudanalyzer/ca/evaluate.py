"""Point cloud evaluation metrics (F1, Chamfer, Hausdorff, AUC)."""

import numpy as np
import open3d as o3d

from ca.io import load_point_cloud
from ca.metrics import compute_nn_distance


def _f1_at_threshold(
    dist_s2t: np.ndarray,
    dist_t2s: np.ndarray,
    threshold: float,
) -> dict:
    """Compute precision, recall, F1 at a given distance threshold."""
    precision = float(np.mean(dist_s2t <= threshold))
    recall = float(np.mean(dist_t2s <= threshold))
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1}


def evaluate(
    source_path: str,
    target_path: str,
    thresholds: list[float] | None = None,
) -> dict:
    """Evaluate point cloud similarity with multiple metrics.

    Args:
        source_path: Path to source (estimated) point cloud.
        target_path: Path to target (reference) point cloud.
        thresholds: List of distance thresholds for F1/AUC.
                    Defaults to [0.05, 0.1, 0.2, 0.3, 0.5, 1.0].

    Returns:
        Dict with chamfer, hausdorff, F1 scores, and AUC.
    """
    if thresholds is None:
        thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    thresholds = sorted(thresholds)

    source = load_point_cloud(source_path)
    target = load_point_cloud(target_path)

    # Bidirectional nearest neighbor distances
    dist_s2t = compute_nn_distance(source, target)
    dist_t2s = compute_nn_distance(target, source)

    # Chamfer Distance (mean of bidirectional means)
    chamfer = float((np.mean(dist_s2t) + np.mean(dist_t2s)) / 2.0)

    # Hausdorff Distance (max of bidirectional maxes)
    hausdorff = float(max(np.max(dist_s2t), np.max(dist_t2s)))

    # F1 at each threshold
    f1_scores = []
    for t in thresholds:
        f1_scores.append(_f1_at_threshold(dist_s2t, dist_t2s, t))

    # AUC (trapezoidal integration of F1 over thresholds)
    f1_values = [s["f1"] for s in f1_scores]
    if len(thresholds) >= 2:
        y = np.asarray(f1_values, dtype=np.float64)
        x = np.asarray(thresholds, dtype=np.float64)
        auc = float(np.sum(0.5 * (x[1:] - x[:-1]) * (y[1:] + y[:-1]))) / (float(x[-1] - x[0]))
    else:
        auc = f1_values[0] if f1_values else 0.0

    # Summary stats
    dist_stats = {
        "source_to_target": {
            "mean": float(np.mean(dist_s2t)),
            "median": float(np.median(dist_s2t)),
            "max": float(np.max(dist_s2t)),
            "std": float(np.std(dist_s2t)),
        },
        "target_to_source": {
            "mean": float(np.mean(dist_t2s)),
            "median": float(np.median(dist_t2s)),
            "max": float(np.max(dist_t2s)),
            "std": float(np.std(dist_t2s)),
        },
    }

    return {
        "source_path": source_path,
        "target_path": target_path,
        "source_points": len(source.points),
        "target_points": len(target.points),
        "chamfer_distance": chamfer,
        "hausdorff_distance": hausdorff,
        "f1_scores": f1_scores,
        "auc": auc,
        "distance_stats": dist_stats,
    }


def plot_f1_curve(result: dict, output_path: str) -> None:
    """Plot F1/Precision/Recall curves over thresholds.

    Args:
        result: Result dict from evaluate().
        output_path: Output image path (png).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = result["f1_scores"]
    thresholds = [s["threshold"] for s in scores]
    f1s = [s["f1"] for s in scores]
    precs = [s["precision"] for s in scores]
    recs = [s["recall"] for s in scores]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1s, "o-", label=f"F1 (AUC={result['auc']:.4f})", linewidth=2)
    ax.plot(thresholds, precs, "s--", label="Precision", linewidth=1.5)
    ax.plot(thresholds, recs, "^--", label="Recall", linewidth=1.5)

    ax.set_xlabel("Distance Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"Evaluation: Chamfer={result['chamfer_distance']:.4f}, Hausdorff={result['hausdorff_distance']:.4f}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
