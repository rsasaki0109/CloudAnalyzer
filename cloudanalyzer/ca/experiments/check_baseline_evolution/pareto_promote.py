"""Concrete OOP strategy that compares candidate quality on a Pareto frontier."""

from __future__ import annotations

from dataclasses import dataclass

from ca.core.check_baseline_evolution import (
    BaselineEvolutionRequest,
    BaselineEvolutionResult,
    BaselineEvolutionSnapshot,
    snapshot_margin_stats,
)


@dataclass(slots=True)
class _DecisionPoint:
    label: str
    mean_margin: float
    worst_margin: float
    pass_ratio: float
    failed_checks: float

    @classmethod
    def from_snapshot(cls, snapshot: BaselineEvolutionSnapshot) -> "_DecisionPoint":
        stats = snapshot_margin_stats(snapshot)
        return cls(
            label=snapshot.label,
            mean_margin=float(stats["mean_margin"]),
            worst_margin=float(stats["worst_margin"]),
            pass_ratio=float(stats["pass_ratio"]),
            failed_checks=float(stats["failed_checks"]),
        )

    def dominates(self, other: "_DecisionPoint") -> bool:
        non_worse = (
            self.mean_margin >= other.mean_margin
            and self.worst_margin >= other.worst_margin
            and self.pass_ratio >= other.pass_ratio
            and self.failed_checks <= other.failed_checks
        )
        strictly_better = (
            self.mean_margin > other.mean_margin
            or self.worst_margin > other.worst_margin
            or self.pass_ratio > other.pass_ratio
            or self.failed_checks < other.failed_checks
        )
        return non_worse and strictly_better


class ParetoPromoteBaselineEvolutionStrategy:
    """Promote only when the candidate sits on the nondominated frontier."""

    name = "pareto_promote"
    design = "oop"

    def __init__(self, *, min_dominated_history: int = 1):
        self.min_dominated_history = min_dominated_history

    def decide(self, request: BaselineEvolutionRequest) -> BaselineEvolutionResult:
        candidate = request.candidate
        if not candidate.passed:
            return BaselineEvolutionResult(
                decision="reject",
                confidence=0.98,
                reasons=("candidate_failed_quality_gate",),
                strategy=self.name,
                design=self.design,
                metadata={"history_count": len(request.history)},
            )

        candidate_point = _DecisionPoint.from_snapshot(candidate)
        history_points = [_DecisionPoint.from_snapshot(snapshot) for snapshot in request.history]
        if not history_points:
            return BaselineEvolutionResult(
                decision="keep",
                confidence=0.71,
                reasons=("missing_history_for_frontier",),
                strategy=self.name,
                design=self.design,
                metadata={"history_count": 0},
            )

        dominated_by_history = [point.label for point in history_points if point.dominates(candidate_point)]
        dominated_history = [point.label for point in history_points if candidate_point.dominates(point)]
        if dominated_by_history:
            return BaselineEvolutionResult(
                decision="keep",
                confidence=0.79,
                reasons=("candidate_is_dominated_by_history",),
                strategy=self.name,
                design=self.design,
                metadata={
                    "history_count": len(history_points),
                    "dominated_by": dominated_by_history,
                },
            )

        if len(dominated_history) >= self.min_dominated_history:
            return BaselineEvolutionResult(
                decision="promote",
                confidence=0.8,
                reasons=("candidate_is_nondominated", "candidate_dominates_history_frontier"),
                strategy=self.name,
                design=self.design,
                metadata={
                    "history_count": len(history_points),
                    "dominated_history": dominated_history,
                },
            )

        return BaselineEvolutionResult(
            decision="keep",
            confidence=0.75,
            reasons=("candidate_is_nondominated_but_not_decisive",),
            strategy=self.name,
            design=self.design,
            metadata={
                "history_count": len(history_points),
                "dominated_history": dominated_history,
            },
        )
