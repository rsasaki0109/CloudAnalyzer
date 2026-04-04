"""Distance-shell planner for progressive browser loading."""

from __future__ import annotations

import numpy as np

from ca.core.web_progressive_loading import WebProgressiveLoadingRequest, WebProgressiveLoadingResult
from ca.experiments.web_progressive_loading.common import build_result_from_order


class DistanceShellsStrategy:
    """Group points by radius from the cloud centroid and interleave shells."""

    name = "distance_shells"
    design = "radial"

    def __init__(self, shell_count: int = 8):
        self.shell_count = shell_count

    def plan(self, request: WebProgressiveLoadingRequest) -> WebProgressiveLoadingResult:
        positions = request.positions
        if positions.shape[0] == 0:
            return build_result_from_order(
                request=request,
                ordered_indices=np.zeros(0, dtype=int),
                strategy=self.name,
                design=self.design,
                metadata={"shell_count": 0},
            )

        center = np.mean(positions, axis=0)
        radii = np.linalg.norm(positions - center[None, :], axis=1)
        shell_edges = np.quantile(radii, np.linspace(0.0, 1.0, self.shell_count + 1))
        shell_ids = np.searchsorted(shell_edges[1:-1], radii, side="right")

        groups: list[np.ndarray] = []
        for shell_id in range(self.shell_count):
            group = np.flatnonzero(shell_ids == shell_id)
            if group.size > 0:
                order = np.argsort(radii[group], kind="stable")
                groups.append(group[order])

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
            metadata={"shell_count": len(groups)},
        )
