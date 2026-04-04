"""Concrete functional baseline for failed-check triage."""

from __future__ import annotations

from ca.core.check_triage import (
    CheckTriageRequest,
    CheckTriageResult,
    RankedCheckTriageItem,
    normalized_dimension_gaps,
)


class SeverityWeightedExperimentalCheckTriageStrategy:
    """Rank failures by weighted threshold exceedance."""

    name = "severity_weighted"
    design = "functional"

    _WEIGHTS = {
        "auc": 1.35,
        "map_auc": 1.35,
        "chamfer": 1.15,
        "map_chamfer": 1.15,
        "ate": 1.3,
        "trajectory_ate": 1.3,
        "rpe": 1.1,
        "trajectory_rpe": 1.1,
        "drift": 1.0,
        "trajectory_drift": 1.0,
        "coverage": 1.4,
    }

    def rank(self, request: CheckTriageRequest) -> CheckTriageResult:
        ranked: list[RankedCheckTriageItem] = []
        for item in request.failed_items:
            gaps = normalized_dimension_gaps(item)
            score = sum(gaps[name] * self._WEIGHTS.get(name, 1.0) for name in gaps)
            if len(gaps) > 1:
                score += 0.08 * (len(gaps) - 1)
            ranked.append(
                RankedCheckTriageItem(
                    check_id=item.check_id,
                    kind=item.kind,
                    rank=0,
                    severity_score=round(float(score), 6),
                    failed_dimensions=tuple(gaps.keys()),
                    reasons=item.reasons,
                    metrics=dict(item.metrics),
                    gate=dict(item.gate),
                    report_path=item.report_path,
                    json_path=item.json_path,
                    metadata=dict(item.metadata),
                )
            )
        ordered = sorted(ranked, key=lambda item: (-item.severity_score, item.check_id))
        for index, ranked_item in enumerate(ordered, start=1):
            ranked_item.rank = index
        return CheckTriageResult(
            ranked_items=tuple(ordered),
            strategy=self.name,
            design=self.design,
            metadata={"rank_basis": "weighted_threshold_gap"},
        )
