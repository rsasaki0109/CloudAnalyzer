"""Experimental strategy set for starter `cloudanalyzer.yaml` generation."""

from ca.experiments.check_scaffolding.literal_profiles import LiteralProfilesStrategy
from ca.experiments.check_scaffolding.object_sections import ObjectSectionsStrategy
from ca.experiments.check_scaffolding.pipeline_overlays import PipelineOverlaysStrategy


def get_check_scaffolding_strategies():
    """Return the concrete implementations to compare."""

    return [
        LiteralProfilesStrategy(),
        ObjectSectionsStrategy(),
        PipelineOverlaysStrategy(),
    ]


__all__ = [
    "LiteralProfilesStrategy",
    "ObjectSectionsStrategy",
    "PipelineOverlaysStrategy",
    "get_check_scaffolding_strategies",
]
