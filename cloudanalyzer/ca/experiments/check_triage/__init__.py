"""Experimental strategy set for failed-check regression triage."""

from ca.experiments.check_triage.pareto_frontier import ParetoFrontierCheckTriageStrategy
from ca.experiments.check_triage.severity_weighted import SeverityWeightedExperimentalCheckTriageStrategy
from ca.experiments.check_triage.signature_cluster import SignatureClusterCheckTriageStrategy


def get_check_triage_strategies():
    """Return the concrete implementations to compare."""

    return [
        SeverityWeightedExperimentalCheckTriageStrategy(),
        ParetoFrontierCheckTriageStrategy(),
        SignatureClusterCheckTriageStrategy(),
    ]


__all__ = [
    "ParetoFrontierCheckTriageStrategy",
    "SeverityWeightedExperimentalCheckTriageStrategy",
    "SignatureClusterCheckTriageStrategy",
    "get_check_triage_strategies",
]
