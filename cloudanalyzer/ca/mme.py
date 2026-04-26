"""Mean Map Entropy (MME) — information-theoretic local consistency metric."""

import numpy as np
from scipy.spatial import KDTree

from ca.io import load_point_cloud


def compute_mme(
    path: str,
    k_neighbors: int = 20,
    max_points: int = 500_000,
    workers: int = -1,
) -> dict:
    """Compute Mean Map Entropy for a single point cloud.

    MME measures local geometric consistency without requiring ground truth.
    Lower values indicate more ordered, structured local geometry.

    Args:
        path: Path to point cloud file.
        k_neighbors: Number of nearest neighbors per point (min 4).
        max_points: Random-sample threshold; clouds larger than this are
                    sampled down to max_points before computation.
        workers: Parallel workers for k-NN query (-1 = all CPUs).

    Returns:
        Dict with mme, k_neighbors, num_points, num_points_used, sampled, path.

    Raises:
        ValueError: If k_neighbors < 4 or num_points_used <= k_neighbors.
        FileNotFoundError: If file does not exist.
    """
    if k_neighbors < 4:
        raise ValueError(f"k_neighbors must be >= 4, got {k_neighbors}")

    pcd = load_point_cloud(path)
    points = np.asarray(pcd.points)
    num_points = len(points)

    # Sampling
    if num_points > max_points:
        idx = np.random.default_rng(42).choice(num_points, size=max_points, replace=False)
        points_used = points[idx]
        sampled = True
    else:
        points_used = points
        sampled = False

    num_points_used = len(points_used)

    if num_points_used <= k_neighbors:
        raise ValueError(
            f"Not enough points for k_neighbors={k_neighbors}: "
            f"got {num_points_used} point(s). Use --neighbors < {num_points_used}."
        )

    # Build KDTree and batch query all points (k+1 to include self)
    tree = KDTree(points_used)
    _, neighbor_idx = tree.query(points_used, k=k_neighbors + 1, workers=workers)
    # neighbor_idx shape: (M, k+1) — first column is self; drop it
    neighbor_idx = neighbor_idx[:, 1:]  # (M, k)

    # Gather neighbor coordinates: (M, k, 3)
    neighbors = points_used[neighbor_idx]

    # Vectorized sample covariance: center each neighborhood
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)  # (M, k, 3)
    # cov shape: (M, 3, 3)
    cov = np.einsum("nik,nil->nkl", centered, centered) / (k_neighbors - 1)

    # Entropy: H_i = 0.5 * ln(|det(Σ_i)| + ε)
    det = np.linalg.det(cov)  # (M,)
    entropy = 0.5 * np.log(np.abs(det) + 1e-10)  # (M,)
    mme = float(np.mean(entropy))

    return {
        "path": path,
        "mme": mme,
        "k_neighbors": k_neighbors,
        "num_points": num_points,
        "num_points_used": num_points_used,
        "sampled": sampled,
    }
