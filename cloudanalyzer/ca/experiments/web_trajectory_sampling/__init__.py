"""Registry of concrete experiment variants."""

from .uniform_stride import UniformStrideStrategy
from .distance_accumulator import DistanceAccumulatorStrategy
from .turn_aware import TurnAwareStrategy


def get_strategies():
    """Return all comparable experiment variants."""

    return [
        UniformStrideStrategy(),
        DistanceAccumulatorStrategy(),
        TurnAwareStrategy(),
    ]
