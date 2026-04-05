"""Stable, minimal interface for web display point-cloud reduction."""

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import open3d as o3d


@dataclass(slots=True)
class WebSampleRequest:
    """Input contract shared by all web reduction strategies."""

    point_cloud: o3d.geometry.PointCloud
    max_points: int
    label: str = "point cloud"

    def __post_init__(self) -> None:
        if self.max_points <= 0:
            raise ValueError(f"max_points must be positive, got {self.max_points}")


@dataclass(slots=True)
class WebSampleResult:
    """Output contract shared by all web reduction strategies."""

    point_cloud: o3d.geometry.PointCloud
    strategy: str
    design: str
    original_points: int
    reduced_points: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reduction_ratio(self) -> float:
        if self.original_points == 0:
            return 0.0
        return 1.0 - (self.reduced_points / self.original_points)


class WebSamplingStrategy(Protocol):
    """Protocol kept in core after comparing concrete implementations."""

    name: str
    design: str

    def reduce(self, request: WebSampleRequest) -> WebSampleResult:
        """Reduce a point cloud to a browser-friendly point budget."""


class RandomBudgetWebSamplingStrategy:
    """Stable reducer selected after experiment comparison."""

    name = "random_budget"
    design = "oop"

    def __init__(self, seed: int = 7):
        self.seed = seed

    def reduce(self, request: WebSampleRequest) -> WebSampleResult:
        original_points = len(request.point_cloud.points)
        if original_points <= request.max_points:
            return WebSampleResult(
                point_cloud=request.point_cloud,
                strategy=self.name,
                design=self.design,
                original_points=original_points,
                reduced_points=original_points,
                metadata={
                    "label": request.label,
                    "seed": self.seed,
                },
            )

        rng = np.random.default_rng(self.seed)
        indices = rng.choice(original_points, size=request.max_points, replace=False)
        indices.sort()
        reduced = request.point_cloud.select_by_index(indices.tolist())

        return WebSampleResult(
            point_cloud=reduced,
            strategy=self.name,
            design=self.design,
            original_points=original_points,
            reduced_points=len(reduced.points),
            metadata={
                "label": request.label,
                "seed": self.seed,
                "selected_points": len(indices),
            },
        )


def reduce_point_cloud_for_web(
    point_cloud: o3d.geometry.PointCloud,
    max_points: int,
    label: str = "point cloud",
    strategy: WebSamplingStrategy | None = None,
) -> WebSampleResult:
    """Reduce a point cloud for browser display using the stable strategy."""

    reducer = strategy or RandomBudgetWebSamplingStrategy()
    return reducer.reduce(
        WebSampleRequest(
            point_cloud=point_cloud,
            max_points=max_points,
            label=label,
        )
    )
