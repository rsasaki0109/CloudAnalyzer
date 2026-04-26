"""GT-free map self-consistency via voxel occupancy entropy (MME proxy).

MapEval uses Mean Map Entropy (MME) as a GT-free local consistency metric.
Here we implement a lightweight proxy:
- Voxelize the map.
- For each occupied voxel, compute entropy of its neighbor occupancy pattern.
- Lower average entropy implies more locally consistent structure.

This is intentionally simple and fast on CPU, with no external deps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ca.experiments.map_evaluate.common import (
    MapEvaluateRequest,
    MapEvaluateResult,
    _require_xyz,
    voxel_downsample,
)


def _voxel_keys(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int64)
    return np.floor(points / voxel_size).astype(np.int64)


def _entropy_from_probs(p: np.ndarray) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


@dataclass(slots=True)
class VoxelEntropyMapEvaluateStrategy:
    name: str = "voxel_entropy"
    design: str = "functional"

    def evaluate(self, request: MapEvaluateRequest) -> MapEvaluateResult:
        if request.reference_points is not None:
            raise ValueError("voxel_entropy is intended for GT-free evaluation (reference_points must be None).")

        est = voxel_downsample(_require_xyz(request.estimated_points, "estimated_points"), request.downsample_voxel_size)
        vs = float(request.structure_voxel_size)
        if vs <= 0:
            raise ValueError("structure_voxel_size must be > 0 for voxel_entropy.")

        keys = _voxel_keys(est, vs)
        if keys.shape[0] == 0:
            return MapEvaluateResult(
                strategy=self.name,
                design=self.design,
                metrics={"n_est": 0.0, "occupied_voxels": 0.0, "mean_neighbor_entropy_bits": 0.0},
                artifacts={"structure_voxel_size": vs},
            )

        # Unique occupied voxels
        uniq = np.unique(keys, axis=0)
        occupied = {tuple(map(int, row)) for row in uniq}

        # 3x3x3 neighborhood occupancy pattern around each voxel (26 neighbors).
        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        entropies: list[float] = []
        for vx, vy, vz in occupied:
            occ = np.array(
                [1.0 if (vx + dx, vy + dy, vz + dz) in occupied else 0.0 for (dx, dy, dz) in offsets],
                dtype=np.float64,
            )
            # Probability of neighbor being occupied vs empty.
            p_occ = float(occ.mean())
            p = np.array([p_occ, 1.0 - p_occ], dtype=np.float64)
            entropies.append(_entropy_from_probs(p))

        mean_entropy = float(np.mean(entropies)) if entropies else 0.0
        # A simple score where higher is better (more consistent).
        consistency_score = float(max(0.0, 1.0 - (mean_entropy / 1.0)))  # 1 bit max for binary

        return MapEvaluateResult(
            strategy=self.name,
            design=self.design,
            metrics={
                "n_est": float(est.shape[0]),
                "occupied_voxels": float(len(occupied)),
                "mean_neighbor_entropy_bits": mean_entropy,
                "consistency_score": consistency_score,
            },
            artifacts={"structure_voxel_size": vs, "downsample_voxel_size": float(request.downsample_voxel_size)},
        )

