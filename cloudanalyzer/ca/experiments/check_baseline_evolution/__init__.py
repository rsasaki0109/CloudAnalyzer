"""Experimental strategy set for baseline evolution decisions."""

from ca.experiments.check_baseline_evolution.pareto_promote import (
    ParetoPromoteBaselineEvolutionStrategy,
)
from ca.experiments.check_baseline_evolution.stability_window import (
    StabilityWindowExperimentalBaselineEvolutionStrategy,
)
from ca.experiments.check_baseline_evolution.threshold_guard import (
    ThresholdGuardBaselineEvolutionStrategy,
)


def get_check_baseline_evolution_strategies():
    """Return the comparable concrete strategy set for this experiment slice."""

    return [
        ThresholdGuardBaselineEvolutionStrategy(),
        ParetoPromoteBaselineEvolutionStrategy(),
        StabilityWindowExperimentalBaselineEvolutionStrategy(),
    ]


__all__ = [
    "get_check_baseline_evolution_strategies",
    "ParetoPromoteBaselineEvolutionStrategy",
    "StabilityWindowExperimentalBaselineEvolutionStrategy",
    "ThresholdGuardBaselineEvolutionStrategy",
]
