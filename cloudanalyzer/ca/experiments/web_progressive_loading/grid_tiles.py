"""Grid-tile planner for progressive browser loading."""

from __future__ import annotations

import numpy as np

from ca.core.web_progressive_loading import WebProgressiveLoadingRequest, WebProgressiveLoadingResult
from ca.experiments.web_progressive_loading.common import build_result_from_order, normalize_positions


class GridTilesStrategy:
    """Partition points into coarse tiles and round-robin across occupied cells."""

    name = "grid_tiles"
    design = "grid"

    def plan(self, request: WebProgressiveLoadingRequest) -> WebProgressiveLoadingResult:
        positions = request.positions
        if positions.shape[0] == 0:
            return build_result_from_order(
                request=request,
                ordered_indices=np.zeros(0, dtype=int),
                strategy=self.name,
                design=self.design,
                metadata={"grid_side": 0, "tile_count": 0},
            )

        normalized = normalize_positions(positions)
        side = max(1, int(np.ceil(np.cbrt(min(request.initial_points, positions.shape[0])))))
        coords = np.floor(normalized * side).astype(int)
        coords = np.clip(coords, 0, side - 1)
        packed = (coords[:, 0] * side * side) + (coords[:, 1] * side) + coords[:, 2]
        sorted_indices = np.argsort(packed, kind="stable")
        sorted_ids = packed[sorted_indices]

        groups: list[tuple[int, np.ndarray]] = []
        start = 0
        while start < sorted_indices.size:
            end = start + 1
            while end < sorted_indices.size and sorted_ids[end] == sorted_ids[start]:
                end += 1
            groups.append((int(sorted_ids[start]), sorted_indices[start:end]))
            start = end

        groups.sort(
            key=lambda item: ((item[0] * 2654435761) % (2**32), item[0])
        )
        ordered: list[int] = []
        level = 0
        while len(ordered) < positions.shape[0]:
            added = False
            for _, group in groups:
                if level < group.size:
                    ordered.append(int(group[level]))
                    added = True
            if not added:
                break
            level += 1

        return build_result_from_order(
            request=request,
            ordered_indices=np.asarray(ordered, dtype=int),
            strategy=self.name,
            design=self.design,
            metadata={"grid_side": side, "tile_count": len(groups)},
        )
