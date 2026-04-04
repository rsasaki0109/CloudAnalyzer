"""Concrete pipeline strategy that waits for a stable window before promotion."""

from __future__ import annotations

from statistics import fmean

from ca.core.check_baseline_evolution import (
    BaselineEvolutionRequest,
    BaselineEvolutionResult,
    snapshot_margin_stats,
)


class StabilityWindowExperimentalBaselineEvolutionStrategy:
    """Promote only after a stable recent window and net margin improvement."""

    name = "stability_window"
    design = "pipeline"

    def __init__(self, *, window_size: int = 3, min_improvement: float = 0.03):
        self.window_size = window_size
        self.min_improvement = min_improvement

    def _recent_window(self, request: BaselineEvolutionRequest):
        return request.history[-(self.window_size - 1):]

    def _recent_pass_streak(self, request: BaselineEvolutionRequest) -> int:
        consecutive_passes = 1
        for snapshot in reversed(request.history):
            if not snapshot.passed:
                break
            consecutive_passes += 1
        return consecutive_passes

    def decide(self, request: BaselineEvolutionRequest) -> BaselineEvolutionResult:
        candidate = request.candidate
        if not candidate.passed:
            return BaselineEvolutionResult(
                decision="reject",
                confidence=0.98,
                reasons=("candidate_failed_quality_gate",),
                strategy=self.name,
                design=self.design,
                metadata={
                    "window_size": self.window_size,
                    "history_count": len(request.history),
                },
            )

        candidate_stats = snapshot_margin_stats(candidate)
        trailing_history = self._recent_window(request)
        if len(trailing_history) < self.window_size - 1:
            return BaselineEvolutionResult(
                decision="keep",
                confidence=0.7,
                reasons=("insufficient_history_window",),
                strategy=self.name,
                design=self.design,
                metadata={
                    "window_size": self.window_size,
                    "history_count": len(request.history),
                    "candidate_mean_margin": round(candidate_stats["mean_margin"], 6),
                },
            )

        history_stats = [snapshot_margin_stats(snapshot) for snapshot in trailing_history]
        mean_history_margin = float(fmean(item["mean_margin"] for item in history_stats))
        mean_history_worst = float(fmean(item["worst_margin"] for item in history_stats))
        margin_gain = candidate_stats["mean_margin"] - mean_history_margin
        worst_gain = candidate_stats["worst_margin"] - mean_history_worst
        consecutive_passes = self._recent_pass_streak(request)

        if (
            consecutive_passes >= self.window_size
            and margin_gain >= self.min_improvement
            and worst_gain >= -0.01
        ):
            return BaselineEvolutionResult(
                decision="promote",
                confidence=0.88,
                reasons=("stable_pass_window", "candidate_improves_margin_window"),
                strategy=self.name,
                design=self.design,
                metadata={
                    "window_size": self.window_size,
                    "history_count": len(request.history),
                    "consecutive_passes": consecutive_passes,
                    "margin_gain": round(margin_gain, 6),
                    "worst_gain": round(worst_gain, 6),
                },
            )

        return BaselineEvolutionResult(
            decision="keep",
            confidence=0.76,
            reasons=("candidate_not_yet_stable_enough",),
            strategy=self.name,
            design=self.design,
            metadata={
                "window_size": self.window_size,
                "history_count": len(request.history),
                "consecutive_passes": consecutive_passes,
                "margin_gain": round(margin_gain, 6),
                "worst_gain": round(worst_gain, 6),
            },
        )
