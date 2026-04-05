"""Class-oriented implementation of web point-cloud reduction."""

from dataclasses import dataclass

import numpy as np

from ca.core.web_sampling import WebSampleRequest, WebSampleResult


@dataclass(slots=True)
class RandomBudgetConfig:
    """Configuration carried by the reducer object."""

    seed: int = 7


class RandomBudgetSamplingStrategy:
    """Reducer that treats reduction as a stateful policy object."""

    name = "random_budget"
    design = "oop"

    def __init__(self, config: RandomBudgetConfig | None = None):
        self.config = config or RandomBudgetConfig()

    def _select_indices(self, total_points: int, max_points: int) -> list[int]:
        rng = np.random.default_rng(self.config.seed)
        indices = rng.choice(total_points, size=max_points, replace=False)
        indices.sort()
        return [int(index) for index in indices.tolist()]

    def reduce(self, request: WebSampleRequest) -> WebSampleResult:
        original_points = len(request.point_cloud.points)
        if original_points <= request.max_points:
            return WebSampleResult(
                point_cloud=request.point_cloud,
                strategy=self.name,
                design=self.design,
                original_points=original_points,
                reduced_points=original_points,
                metadata={"label": request.label, "seed": self.config.seed},
            )

        indices = self._select_indices(original_points, request.max_points)
        reduced = request.point_cloud.select_by_index(indices)
        return WebSampleResult(
            point_cloud=reduced,
            strategy=self.name,
            design=self.design,
            original_points=original_points,
            reduced_points=len(reduced.points),
            metadata={
                "label": request.label,
                "seed": self.config.seed,
                "selected_points": len(indices),
            },
        )
