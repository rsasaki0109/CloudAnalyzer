"""Concrete functional baseline based on direct threshold margin guards."""

from __future__ import annotations

from typing import cast

from ca.core.check_baseline_evolution import (
    BaselineEvolutionRequest,
    BaselineEvolutionResult,
    DecisionLabel,
    snapshot_margin_stats,
)


class ThresholdGuardBaselineEvolutionStrategy:
    """Promote as soon as the candidate clearly beats the recent baseline margins."""

    name = "threshold_guard"
    design = "functional"

    def __init__(self, *, min_improvement: float = 0.03, worst_margin_floor: float = 0.0):
        self.min_improvement = min_improvement
        self.worst_margin_floor = worst_margin_floor

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

        candidate_stats = snapshot_margin_stats(candidate)
        history_best_mean = max(
            (snapshot_margin_stats(snapshot)["mean_margin"] for snapshot in request.history),
            default=None,
        )

        if history_best_mean is None:
            decision = cast(
                DecisionLabel,
                "promote" if candidate_stats["worst_margin"] >= self.worst_margin_floor else "keep",
            )
            reason = (
                "candidate_clears_margin_floor_without_history"
                if decision == "promote"
                else "candidate_margin_too_thin_without_history"
            )
            confidence = 0.72 if decision == "promote" else 0.7
        elif (
            candidate_stats["mean_margin"] >= history_best_mean + self.min_improvement
            and candidate_stats["worst_margin"] >= self.worst_margin_floor
        ):
            decision = "promote"
            reason = "candidate_beats_best_history_margin"
            confidence = 0.82
        else:
            decision = "keep"
            reason = "candidate_margin_not_far_enough_from_history"
            confidence = 0.76

        return BaselineEvolutionResult(
            decision=decision,
            confidence=confidence,
            reasons=(reason,),
            strategy=self.name,
            design=self.design,
            metadata={
                "history_count": len(request.history),
                "candidate_mean_margin": round(candidate_stats["mean_margin"], 6),
                "candidate_worst_margin": round(candidate_stats["worst_margin"], 6),
                "history_best_mean_margin": (
                    round(history_best_mean, 6) if history_best_mean is not None else None
                ),
            },
        )
