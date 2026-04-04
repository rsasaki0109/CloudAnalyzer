"""Concrete OOP triage strategy based on Pareto fronts of failure severity."""

from __future__ import annotations

from dataclasses import dataclass

from ca.core.check_triage import (
    CheckTriageItem,
    CheckTriageRequest,
    CheckTriageResult,
    RankedCheckTriageItem,
    normalized_dimension_gaps,
)


def _dominates(left: dict[str, float], right: dict[str, float], dimensions: tuple[str, ...]) -> bool:
    left_values = [left.get(name, 0.0) for name in dimensions]
    right_values = [right.get(name, 0.0) for name in dimensions]
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


@dataclass(slots=True)
class Candidate:
    item: CheckTriageItem
    gaps: dict[str, float]
    front_rank: int = 0

    @property
    def max_gap(self) -> float:
        return max(self.gaps.values(), default=0.0)

    @property
    def total_gap(self) -> float:
        return sum(self.gaps.values())


class ParetoFrontierCheckTriageStrategy:
    """Rank failures by Pareto front before using local tie-breakers."""

    name = "pareto_frontier"
    design = "oop"

    def rank(self, request: CheckTriageRequest) -> CheckTriageResult:
        candidates = [Candidate(item=item, gaps=normalized_dimension_gaps(item)) for item in request.failed_items]
        dimensions = tuple(
            sorted({name for candidate in candidates for name in candidate.gaps})
        )

        remaining = candidates[:]
        front_rank = 1
        while remaining:
            front = [
                candidate
                for candidate in remaining
                if not any(
                    _dominates(other.gaps, candidate.gaps, dimensions)
                    for other in remaining
                    if other is not candidate
                )
            ]
            for candidate in front:
                candidate.front_rank = front_rank
            remaining = [candidate for candidate in remaining if candidate not in front]
            front_rank += 1

        ordered = sorted(
            candidates,
            key=lambda candidate: (
                candidate.front_rank,
                -candidate.max_gap,
                -candidate.total_gap,
                candidate.item.check_id,
            ),
        )
        ranked_items = []
        for index, candidate in enumerate(ordered, start=1):
            ranked_items.append(
                RankedCheckTriageItem(
                    check_id=candidate.item.check_id,
                    kind=candidate.item.kind,
                    rank=index,
                    severity_score=round(
                        float((1.0 / candidate.front_rank) + candidate.total_gap),
                        6,
                    ),
                    failed_dimensions=tuple(candidate.gaps.keys()),
                    reasons=candidate.item.reasons,
                    metrics=dict(candidate.item.metrics),
                    gate=dict(candidate.item.gate),
                    report_path=candidate.item.report_path,
                    json_path=candidate.item.json_path,
                    metadata={
                        **dict(candidate.item.metadata),
                        "front_rank": candidate.front_rank,
                    },
                )
            )
        return CheckTriageResult(
            ranked_items=tuple(ranked_items),
            strategy=self.name,
            design=self.design,
            metadata={"dimensions": dimensions},
        )
