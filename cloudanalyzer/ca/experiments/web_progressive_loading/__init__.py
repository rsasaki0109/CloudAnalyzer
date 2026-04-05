"""Registry of concrete experiment variants."""

from __future__ import annotations

from .grid_tiles import GridTilesStrategy
from .spatial_shuffle import SpatialShuffleStrategy
from .distance_shells import DistanceShellsStrategy


def get_strategies() -> list:
    """Return all comparable experiment variants."""

    return [
        GridTilesStrategy(),
        SpatialShuffleStrategy(),
        DistanceShellsStrategy(),
    ]
