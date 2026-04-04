"""Stable, minimal interface for baseline promotion decisions from QA summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any, Literal, Protocol

DecisionLabel = Literal["promote", "keep", "reject"]


@dataclass(slots=True)
class BaselineCheckSnapshot:
    """One QA check snapshot used for baseline evolution decisions."""

    check_id: str
    kind: str
    passed: bool | None
    metrics: dict[str, float]
    gate: dict[str, float]
    triage_rank: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BaselineEvolutionSnapshot:
    """One candidate or historical QA summary in baseline evolution."""

    label: str
    checks: tuple[BaselineCheckSnapshot, ...]
    passed: bool
    failed_check_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BaselineEvolutionRequest:
    """Input contract shared by baseline evolution strategies."""

    candidate: BaselineEvolutionSnapshot
    history: tuple[BaselineEvolutionSnapshot, ...] = ()


@dataclass(slots=True)
class BaselineEvolutionResult:
    """Decision output shared by baseline evolution strategies."""

    decision: DecisionLabel
    confidence: float
    reasons: tuple[str, ...]
    strategy: str
    design: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaselineEvolutionStrategy(Protocol):
    """Protocol kept in core after comparing concrete decision strategies."""

    name: str
    design: str

    def decide(self, request: BaselineEvolutionRequest) -> BaselineEvolutionResult:
        """Return a baseline promotion decision for the candidate summary."""


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


def _normalize_metric_margin(metric_name: str, metric_value: float, gate_value: float) -> float:
    baseline = max(abs(gate_value), 1e-6)
    if _DIMENSION_GATE_KEYS[metric_name].startswith("min_"):
        return float((metric_value - gate_value) / baseline)
    return float((gate_value - metric_value) / baseline)


def snapshot_metric_margins(snapshot: BaselineEvolutionSnapshot) -> dict[str, float]:
    """Return normalized gate margins for one snapshot."""

    margins: dict[str, float] = {}
    for check in snapshot.checks:
        for metric_name, metric_value in check.metrics.items():
            gate_key = _DIMENSION_GATE_KEYS.get(metric_name)
            if gate_key is None or gate_key not in check.gate:
                continue
            margins[f"{check.check_id}:{metric_name}"] = _normalize_metric_margin(
                metric_name,
                float(metric_value),
                float(check.gate[gate_key]),
            )
    return margins


def snapshot_margin_stats(snapshot: BaselineEvolutionSnapshot) -> dict[str, float]:
    """Summarize normalized gate margins for one snapshot."""

    margins = list(snapshot_metric_margins(snapshot).values())
    if not margins:
        return {
            "mean_margin": 0.0,
            "worst_margin": 0.0,
            "pass_ratio": 1.0 if snapshot.passed else 0.0,
            "failed_checks": float(len(snapshot.failed_check_ids)),
        }
    passed_checks = sum(1 for check in snapshot.checks if check.passed is True)
    gated_checks = sum(1 for check in snapshot.checks if check.passed is not None)
    return {
        "mean_margin": float(fmean(margins)),
        "worst_margin": float(min(margins)),
        "pass_ratio": float(passed_checks / gated_checks) if gated_checks else 1.0,
        "failed_checks": float(len(snapshot.failed_check_ids)),
    }


class StabilityWindowBaselineEvolutionStrategy:
    """Stable decision strategy selected after experiment comparison."""

    name = "stability_window"
    design = "pipeline"

    def __init__(self, *, window_size: int = 3, min_improvement: float = 0.03):
        self.window_size = window_size
        self.min_improvement = min_improvement

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
        trailing_history = request.history[-(self.window_size - 1):]
        consecutive_passes = 1
        for snapshot in reversed(request.history):
            if not snapshot.passed:
                break
            consecutive_passes += 1

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
                    "candidate_mean_margin": candidate_stats["mean_margin"],
                },
            )

        history_stats = [snapshot_margin_stats(snapshot) for snapshot in trailing_history]
        mean_history_margin = float(fmean(item["mean_margin"] for item in history_stats))
        mean_history_worst = float(fmean(item["worst_margin"] for item in history_stats))
        margin_gain = candidate_stats["mean_margin"] - mean_history_margin
        worst_gain = candidate_stats["worst_margin"] - mean_history_worst

        if consecutive_passes >= self.window_size and margin_gain >= self.min_improvement and worst_gain >= -0.01:
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


def _extract_numeric_gate(gate: dict[str, Any] | None) -> dict[str, float]:
    if gate is None:
        return {}
    numeric: dict[str, float] = {}
    for key, value in gate.items():
        if not key.startswith(("min_", "max_")):
            continue
        if isinstance(value, (int, float)):
            numeric[key] = float(value)
    return numeric


def _extract_metrics(check: dict[str, Any]) -> dict[str, float]:
    kind = str(check["kind"])
    summary = check["summary"]
    result = check.get("result")
    if kind == "artifact":
        return {
            "auc": float(summary["auc"]),
            "chamfer": float(summary["chamfer_distance"]),
        }
    if kind == "artifact_batch" and isinstance(result, list) and result:
        return {
            "auc": float(fmean(item["auc"] for item in result)),
            "chamfer": float(fmean(item["chamfer_distance"] for item in result)),
        }
    if kind == "trajectory":
        return {
            "ate": float(summary["ate_rmse"]),
            "rpe": float(summary["rpe_rmse"]),
            "drift": float(summary["drift_endpoint"]),
            "coverage": float(summary["coverage_ratio"]),
        }
    if kind == "trajectory_batch" and isinstance(result, list) and result:
        return {
            "ate": float(fmean(item["ate"]["rmse"] for item in result)),
            "rpe": float(fmean(item["rpe_translation"]["rmse"] for item in result)),
            "drift": float(fmean(item["drift"]["endpoint"] for item in result)),
            "coverage": float(fmean(item["coverage_ratio"] for item in result)),
        }
    if kind == "run":
        return {
            "map_auc": float(summary["map_auc"]),
            "map_chamfer": float(summary["map_chamfer_distance"]),
            "trajectory_ate": float(summary["trajectory_ate_rmse"]),
            "trajectory_rpe": float(summary["trajectory_rpe_rmse"]),
            "trajectory_drift": float(summary["trajectory_drift_endpoint"]),
            "coverage": float(summary["coverage_ratio"]),
        }
    if kind == "run_batch" and isinstance(result, list) and result:
        return {
            "map_auc": float(fmean(item["map"]["auc"] for item in result)),
            "map_chamfer": float(fmean(item["map"]["chamfer_distance"] for item in result)),
            "trajectory_ate": float(fmean(item["trajectory"]["ate"]["rmse"] for item in result)),
            "trajectory_rpe": float(
                fmean(item["trajectory"]["rpe_translation"]["rmse"] for item in result)
            ),
            "trajectory_drift": float(
                fmean(item["trajectory"]["drift"]["endpoint"] for item in result)
            ),
            "coverage": float(
                fmean(item["trajectory"]["matching"]["coverage_ratio"] for item in result)
            ),
        }
    return {}


def _extract_gate(check: dict[str, Any]) -> dict[str, float]:
    result = check.get("result")
    summary = check.get("summary", {})
    gate = None
    if check["kind"] in {"artifact", "trajectory"} and isinstance(result, dict):
        gate = result.get("quality_gate")
    elif check["kind"] == "run" and isinstance(result, dict):
        gate = result.get("overall_quality_gate")
    else:
        gate = summary.get("quality_gate")
    return _extract_numeric_gate(gate if isinstance(gate, dict) else None)


def snapshot_from_check_result(
    result: dict[str, Any],
    *,
    label: str | None = None,
) -> BaselineEvolutionSnapshot:
    """Convert one `ca check` JSON result into a baseline-evolution snapshot."""

    triage_ranks: dict[str, int] = {}
    triage = result.get("summary", {}).get("triage")
    if isinstance(triage, dict):
        for item in triage.get("items", []):
            if isinstance(item, dict) and "check_id" in item and "rank" in item:
                triage_ranks[str(item["check_id"])] = int(item["rank"])

    checks = tuple(
        BaselineCheckSnapshot(
            check_id=str(check["id"]),
            kind=str(check["kind"]),
            passed=check.get("passed"),
            metrics=_extract_metrics(check),
            gate=_extract_gate(check),
            triage_rank=triage_ranks.get(str(check["id"])),
            metadata={
                "report_path": check.get("report_path"),
                "json_path": check.get("json_path"),
            },
        )
        for check in result.get("checks", [])
    )
    resolved_label = label or Path(str(result.get("config_path", "candidate"))).stem
    return BaselineEvolutionSnapshot(
        label=resolved_label,
        checks=checks,
        passed=bool(result.get("summary", {}).get("passed", False)),
        failed_check_ids=tuple(result.get("summary", {}).get("failed_check_ids", [])),
        metadata={
            "project": result.get("project"),
            "config_path": result.get("config_path"),
        },
    )


def build_baseline_evolution_request(
    candidate_result: dict[str, Any],
    history_results: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> BaselineEvolutionRequest:
    """Build a baseline-evolution request from `ca check` JSON summaries."""

    return BaselineEvolutionRequest(
        candidate=snapshot_from_check_result(candidate_result),
        history=tuple(snapshot_from_check_result(item) for item in history_results),
    )


def decide_baseline_evolution(
    request: BaselineEvolutionRequest,
    strategy: BaselineEvolutionStrategy | None = None,
) -> BaselineEvolutionResult:
    """Decide whether to promote, keep, or reject a candidate baseline."""

    decider = strategy or StabilityWindowBaselineEvolutionStrategy()
    return decider.decide(request)


def summarize_baseline_evolution(
    candidate_result: dict[str, Any],
    history_results: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    strategy: BaselineEvolutionStrategy | None = None,
) -> dict[str, Any]:
    """Return a JSON-friendly baseline-evolution decision summary."""

    request = build_baseline_evolution_request(candidate_result, history_results)
    result = decide_baseline_evolution(request, strategy=strategy)
    return {
        "candidate_label": request.candidate.label,
        "history_labels": [snapshot.label for snapshot in request.history],
        "decision": result.decision,
        "confidence": result.confidence,
        "reasons": list(result.reasons),
        "strategy": result.strategy,
        "design": result.design,
        "metadata": dict(result.metadata),
        "candidate": asdict(request.candidate),
        "history": [asdict(snapshot) for snapshot in request.history],
    }
