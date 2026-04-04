"""Spatial-shuffle planner for progressive browser loading."""

from __future__ import annotations

import numpy as np

from ca.core.web_progressive_loading import WebProgressiveLoadingRequest, WebProgressiveLoadingResult
from ca.experiments.web_progressive_loading.common import build_result_from_order, normalize_positions


def _hashed_cell_order(cell_ids: np.ndarray) -> np.ndarray:
    """Produce a deterministic pseudo-random order from coarse cell ids."""

    hashed = (cell_ids * 2654435761) % (2**32)
    return np.argsort(hashed, kind="stable")


class SpatialShuffleStrategy:
    """Interleave coarse cells with a deterministic spatial shuffle."""

    name = "spatial_shuffle"
    design = "functional"

    def plan(self, request: WebProgressiveLoadingRequest) -> WebProgressiveLoadingResult:
        positions = request.positions
        if positions.shape[0] == 0:
            return build_result_from_order(
                request=request,
                ordered_indices=np.zeros(0, dtype=int),
                strategy=self.name,
                design=self.design,
                metadata={"grid_side": 0, "bucket_count": 0},
            )

        normalized = normalize_positions(positions)
        side = max(1, int(np.ceil(np.cbrt(min(request.initial_points, positions.shape[0])))))
        coords = np.floor(normalized * side).astype(int)
        coords = np.clip(coords, 0, side - 1)
        packed = (coords[:, 0] * side * side) + (coords[:, 1] * side) + coords[:, 2]
        cell_order = _hashed_cell_order(np.unique(packed))
        unique_cells = np.unique(packed)[cell_order]

        groups: list[np.ndarray] = []
        for cell_id in unique_cells.tolist():
            groups.append(np.flatnonzero(packed == cell_id))

        ordered: list[int] = []
        depth = 0
        while len(ordered) < positions.shape[0]:
            added = False
            for group in groups:
                if depth < group.size:
                    ordered.append(int(group[depth]))
                    added = True
            if not added:
                break
            depth += 1

        return build_result_from_order(
            request=request,
            ordered_indices=np.asarray(ordered, dtype=int),
            strategy=self.name,
            design=self.design,
            metadata={"grid_side": side, "bucket_count": len(groups)},
        )
