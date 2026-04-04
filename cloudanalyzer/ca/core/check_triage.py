"""Stable, minimal interface for ranking failed QA checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import fmean
from typing import Any, Protocol


_DIMENSION_GATE_KEYS = {
    "auc": "min_auc",
    "map_auc": "min_auc",
    "chamfer": "max_chamfer",
    "map_chamfer": "max_chamfer",
    "ate": "max_ate",
    "trajectory_ate": "max_ate",
    "rpe": "max_rpe",
    "trajectory_rpe": "max_rpe",
    "drift": "max_drift",
    "trajectory_drift": "max_drift",
    "coverage": "min_coverage",
}

_DIMENSION_LABELS = {
    "auc": "auc",
    "map_auc": "map_auc",
    "chamfer": "chamfer",
    "map_chamfer": "map_chamfer",
    "ate": "ate",
    "trajectory_ate": "trajectory_ate",
    "rpe": "rpe",
    "trajectory_rpe": "trajectory_rpe",
    "drift": "drift",
    "trajectory_drift": "trajectory_drift",
    "coverage": "coverage",
}


@dataclass(slots=True)
class CheckTriageItem:
    """Comparable failed-check record shared across triage strategies."""

    check_id: str
    kind: str
    metrics: dict[str, float]
    gate: dict[str, float]
    reasons: tuple[str, ...] = ()
    report_path: str | None = None
    json_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CheckTriageRequest:
    """Input contract shared by all failed-check triage strategies."""

    failed_items: tuple[CheckTriageItem, ...]
    project: str | None = None


@dataclass(slots=True)
class RankedCheckTriageItem:
    """One ranked failure candidate in the triage output."""

    check_id: str
    kind: str
    rank: int
    severity_score: float
    failed_dimensions: tuple[str, ...]
    reasons: tuple[str, ...]
    metrics: dict[str, float]
    gate: dict[str, float]
    report_path: str | None = None
    json_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CheckTriageResult:
    """Ranked triage output shared by all strategies."""

    ranked_items: tuple[RankedCheckTriageItem, ...]
    strategy: str
    design: str
    metadata: dict[str, Any] = field(default_factory=dict)


class CheckTriageStrategy(Protocol):
    """Protocol kept in core after comparing concrete triage implementations."""

    name: str
    design: str

    def rank(self, request: CheckTriageRequest) -> CheckTriageResult:
        """Rank failed checks from most informative to least informative."""


def normalized_dimension_gaps(item: CheckTriageItem) -> dict[str, float]:
    """Return normalized threshold gaps for every failed dimension on one item."""

    gaps: dict[str, float] = {}
    for dimension, metric_value in item.metrics.items():
        gate_key = _DIMENSION_GATE_KEYS.get(dimension)
        if gate_key is None or gate_key not in item.gate:
            continue
        threshold = float(item.gate[gate_key])
        baseline = max(abs(threshold), 1e-6)
        if gate_key.startswith("min_"):
            gap = max((threshold - float(metric_value)) / baseline, 0.0)
        else:
            gap = max((float(metric_value) - threshold) / baseline, 0.0)
        if gap > 0.0:
            gaps[_DIMENSION_LABELS[dimension]] = float(gap)
    return gaps


def failed_dimensions(item: CheckTriageItem) -> tuple[str, ...]:
    """Return the failed dimensions in deterministic order."""

    return tuple(normalized_dimension_gaps(item).keys())


class SeverityWeightedCheckTriageStrategy:
    """Stable triage strategy selected after experiment comparison."""

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
        ranked_items: list[RankedCheckTriageItem] = []
        for item in request.failed_items:
            gaps = normalized_dimension_gaps(item)
            severity_score = sum(
                gaps[dimension] * self._WEIGHTS.get(dimension, 1.0)
                for dimension in gaps
            )
            if len(gaps) > 1:
                severity_score += 0.08 * (len(gaps) - 1)
            ranked_items.append(
                RankedCheckTriageItem(
                    check_id=item.check_id,
                    kind=item.kind,
                    rank=0,
                    severity_score=round(float(severity_score), 6),
                    failed_dimensions=tuple(gaps.keys()),
                    reasons=item.reasons,
                    metrics=dict(item.metrics),
                    gate=dict(item.gate),
                    report_path=item.report_path,
                    json_path=item.json_path,
                    metadata=dict(item.metadata),
                )
            )

        ordered = sorted(
            ranked_items,
            key=lambda item: (-item.severity_score, item.check_id),
        )
        for index, ranked_item in enumerate(ordered, start=1):
            ranked_item.rank = index
        return CheckTriageResult(
            ranked_items=tuple(ordered),
            strategy=self.name,
            design=self.design,
            metadata={
                "failed_count": len(ordered),
            },
        )


def _numeric_gate(gate: dict[str, Any] | None) -> dict[str, float]:
    if gate is None:
        return {}
    numeric: dict[str, float] = {}
    for key, value in gate.items():
        if not key.startswith(("min_", "max_")):
            continue
        if isinstance(value, (int, float)):
            numeric[key] = float(value)
    return numeric


def _artifact_batch_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    return {
        "auc": float(fmean(item["auc"] for item in results)),
        "chamfer": float(fmean(item["chamfer_distance"] for item in results)),
    }


def _trajectory_batch_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    return {
        "ate": float(fmean(item["ate"]["rmse"] for item in results)),
        "rpe": float(fmean(item["rpe_translation"]["rmse"] for item in results)),
        "drift": float(fmean(item["drift"]["endpoint"] for item in results)),
        "coverage": float(fmean(item["coverage_ratio"] for item in results)),
    }


def _run_batch_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    return {
        "map_auc": float(fmean(item["map"]["auc"] for item in results)),
        "map_chamfer": float(fmean(item["map"]["chamfer_distance"] for item in results)),
        "trajectory_ate": float(fmean(item["trajectory"]["ate"]["rmse"] for item in results)),
        "trajectory_rpe": float(
            fmean(item["trajectory"]["rpe_translation"]["rmse"] for item in results)
        ),
        "trajectory_drift": float(
            fmean(item["trajectory"]["drift"]["endpoint"] for item in results)
        ),
        "coverage": float(
            fmean(item["trajectory"]["matching"]["coverage_ratio"] for item in results)
        ),
    }


def _extract_gate_payload(check: dict[str, Any]) -> tuple[dict[str, float], tuple[str, ...]]:
    result = check.get("result")
    summary = check.get("summary", {})
    gate = None
    if check["kind"] in {"artifact", "trajectory"} and isinstance(result, dict):
        gate = result.get("quality_gate")
    elif check["kind"] == "run" and isinstance(result, dict):
        gate = result.get("overall_quality_gate")
    else:
        gate = summary.get("quality_gate")
    reasons = tuple(gate.get("reasons", ())) if isinstance(gate, dict) else ()
    return _numeric_gate(gate if isinstance(gate, dict) else None), reasons


def _extract_metrics(check: dict[str, Any]) -> dict[str, float]:
    kind = check["kind"]
    summary = check["summary"]
    result = check.get("result")
    if kind == "artifact":
        return {
            "auc": float(summary["auc"]),
            "chamfer": float(summary["chamfer_distance"]),
        }
    if kind == "artifact_batch":
        return _artifact_batch_metrics(result if isinstance(result, list) else [])
    if kind == "trajectory":
        return {
            "ate": float(summary["ate_rmse"]),
            "rpe": float(summary["rpe_rmse"]),
            "drift": float(summary["drift_endpoint"]),
            "coverage": float(summary["coverage_ratio"]),
        }
    if kind == "trajectory_batch":
        return _trajectory_batch_metrics(result if isinstance(result, list) else [])
    if kind == "run":
        return {
            "map_auc": float(summary["map_auc"]),
            "map_chamfer": float(summary["map_chamfer_distance"]),
            "trajectory_ate": float(summary["trajectory_ate_rmse"]),
            "trajectory_rpe": float(summary["trajectory_rpe_rmse"]),
            "trajectory_drift": float(summary["trajectory_drift_endpoint"]),
            "coverage": float(summary["coverage_ratio"]),
        }
    return _run_batch_metrics(result if isinstance(result, list) else [])


def build_check_triage_request(
    executed_checks: list[dict[str, Any]],
    *,
    project: str | None = None,
) -> CheckTriageRequest:
    """Build a triage request from `run_check_suite` check results."""

    failed_items: list[CheckTriageItem] = []
    for check in executed_checks:
        if check.get("passed") is not False:
            continue
        gate, reasons = _extract_gate_payload(check)
        failed_items.append(
            CheckTriageItem(
                check_id=str(check["id"]),
                kind=str(check["kind"]),
                metrics=_extract_metrics(check),
                gate=gate,
                reasons=reasons,
                report_path=check.get("report_path"),
                json_path=check.get("json_path"),
                metadata={
                    "source": "run_check_suite",
                    "batch_fail_count": (
                        int(check["summary"]["quality_gate"]["fail_count"])
                        if isinstance(check.get("summary", {}).get("quality_gate"), dict)
                        and isinstance(check["summary"]["quality_gate"].get("fail_count"), int)
                        else None
                    ),
                },
            )
        )
    return CheckTriageRequest(failed_items=tuple(failed_items), project=project)


def rank_failed_checks(
    request: CheckTriageRequest,
    strategy: CheckTriageStrategy | None = None,
) -> CheckTriageResult:
    """Rank failed checks using the stabilized strategy."""

    triage_strategy = strategy or SeverityWeightedCheckTriageStrategy()
    return triage_strategy.rank(request)


def summarize_failed_checks(
    executed_checks: list[dict[str, Any]],
    *,
    project: str | None = None,
    strategy: CheckTriageStrategy | None = None,
) -> dict[str, Any]:
    """Return a JSON-friendly triage summary for failed checks."""

    triage_result = rank_failed_checks(
        build_check_triage_request(executed_checks, project=project),
        strategy=strategy,
    )
    return {
        "strategy": triage_result.strategy,
        "design": triage_result.design,
        "failed_count": len(triage_result.ranked_items),
        "ranked_ids": [item.check_id for item in triage_result.ranked_items],
        "items": [asdict(item) for item in triage_result.ranked_items],
        "metadata": dict(triage_result.metadata),
    }
