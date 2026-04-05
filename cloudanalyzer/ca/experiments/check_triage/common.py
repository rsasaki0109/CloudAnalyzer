"""Shared datasets and metrics for failed-check triage experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ca.core import CheckTriageItem, CheckTriageRequest


@dataclass(slots=True)
class TriageDatasetCase:
    """Comparable failure set for triage experiments."""

    name: str
    description: str
    request: CheckTriageRequest
    expected_order: tuple[str, ...]


def make_item(
    check_id: str,
    kind: str,
    metrics: dict[str, float],
    gate: dict[str, float],
    *,
    reasons: tuple[str, ...] = (),
    signature: str | None = None,
) -> CheckTriageItem:
    """Build a deterministic triage item for synthetic experiments."""

    return CheckTriageItem(
        check_id=check_id,
        kind=kind,
        metrics=dict(metrics),
        gate=dict(gate),
        reasons=reasons,
        metadata={"signature": signature or kind},
    )


def build_default_datasets() -> list[TriageDatasetCase]:
    """Create deterministic triage datasets with different failure shapes."""

    return [
        TriageDatasetCase(
            name="integrated_cascade",
            description="Integrated run failure should outrank milder single-artifact and trajectory regressions.",
            request=CheckTriageRequest(
                failed_items=(
                    make_item(
                        "mapping-postprocess",
                        "artifact",
                        {"auc": 0.89, "chamfer": 0.028},
                        {"min_auc": 0.95, "max_chamfer": 0.02},
                        reasons=("AUC below gate", "Chamfer above gate"),
                        signature="map_geometry",
                    ),
                    make_item(
                        "localization-run",
                        "trajectory",
                        {"ate": 0.62, "rpe": 0.27, "drift": 0.12, "coverage": 0.82},
                        {
                            "max_ate": 0.5,
                            "max_rpe": 0.2,
                            "max_drift": 0.2,
                            "min_coverage": 0.9,
                        },
                        reasons=("ATE above gate", "RPE above gate", "Coverage below gate"),
                        signature="trajectory_shape",
                    ),
                    make_item(
                        "integrated-run",
                        "run",
                        {
                            "map_auc": 0.81,
                            "map_chamfer": 0.06,
                            "trajectory_ate": 0.8,
                            "trajectory_rpe": 0.22,
                            "trajectory_drift": 0.25,
                            "coverage": 0.72,
                        },
                        {
                            "min_auc": 0.95,
                            "max_chamfer": 0.02,
                            "max_ate": 0.5,
                            "max_rpe": 0.2,
                            "max_drift": 0.2,
                            "min_coverage": 0.9,
                        },
                        reasons=("Map and trajectory gate failures",),
                        signature="integrated",
                    ),
                ),
                project="triage-synthetic",
            ),
            expected_order=("integrated-run", "localization-run", "mapping-postprocess"),
        ),
        TriageDatasetCase(
            name="batch_tradeoff",
            description="Run-batch with several moderate failures should outrank a single-dimension collapse.",
            request=CheckTriageRequest(
                failed_items=(
                    make_item(
                        "artifact-batch",
                        "artifact_batch",
                        {"auc": 0.87, "chamfer": 0.031},
                        {"min_auc": 0.95, "max_chamfer": 0.02},
                        reasons=("Mean map quality below gate",),
                        signature="batch_map",
                    ),
                    make_item(
                        "trajectory-batch",
                        "trajectory_batch",
                        {"ate": 0.21, "rpe": 0.08, "drift": 0.03, "coverage": 0.58},
                        {
                            "max_ate": 0.5,
                            "max_rpe": 0.2,
                            "max_drift": 0.2,
                            "min_coverage": 0.9,
                        },
                        reasons=("Coverage collapse",),
                        signature="batch_coverage",
                    ),
                    make_item(
                        "run-batch",
                        "run_batch",
                        {
                            "map_auc": 0.9,
                            "map_chamfer": 0.029,
                            "trajectory_ate": 0.58,
                            "trajectory_rpe": 0.28,
                            "trajectory_drift": 0.23,
                            "coverage": 0.84,
                        },
                        {
                            "min_auc": 0.95,
                            "max_chamfer": 0.02,
                            "max_ate": 0.5,
                            "max_rpe": 0.2,
                            "max_drift": 0.2,
                            "min_coverage": 0.9,
                        },
                        reasons=("Map and localization batch drift",),
                        signature="batch_integrated",
                    ),
                ),
                project="triage-synthetic",
            ),
            expected_order=("run-batch", "trajectory-batch", "artifact-batch"),
        ),
        TriageDatasetCase(
            name="duplicate_geometry_regressions",
            description="Near-duplicate geometry failures should stay below the single most severe one.",
            request=CheckTriageRequest(
                failed_items=(
                    make_item(
                        "mapping-postprocess",
                        "artifact",
                        {"auc": 0.77, "chamfer": 0.052},
                        {"min_auc": 0.95, "max_chamfer": 0.02},
                        reasons=("Strong geometry regression",),
                        signature="geometry_duplicate",
                    ),
                    make_item(
                        "perception-output",
                        "artifact",
                        {"auc": 0.79, "chamfer": 0.047},
                        {"min_auc": 0.95, "max_chamfer": 0.02},
                        reasons=("Second geometry regression",),
                        signature="geometry_duplicate",
                    ),
                    make_item(
                        "localization-run",
                        "trajectory",
                        {"ate": 0.54, "rpe": 0.24, "drift": 0.1, "coverage": 0.91},
                        {
                            "max_ate": 0.5,
                            "max_rpe": 0.2,
                            "max_drift": 0.2,
                            "min_coverage": 0.9,
                        },
                        reasons=("Trajectory drift",),
                        signature="trajectory_shape",
                    ),
                ),
                project="triage-synthetic",
            ),
            expected_order=("mapping-postprocess", "perception-output", "localization-run"),
        ),
    ]


def perturb_request(request: CheckTriageRequest, factor: float) -> CheckTriageRequest:
    """Create a deterministic perturbed request for ranking-stability checks."""

    perturbed_items: list[CheckTriageItem] = []
    for item in request.failed_items:
        metrics: dict[str, float] = {}
        for name, value in item.metrics.items():
            if "coverage" in name or name.endswith("auc"):
                metrics[name] = float(value / factor)
            else:
                metrics[name] = float(value * factor)
        perturbed_items.append(
            CheckTriageItem(
                check_id=item.check_id,
                kind=item.kind,
                metrics=metrics,
                gate=dict(item.gate),
                reasons=item.reasons,
                report_path=item.report_path,
                json_path=item.json_path,
                metadata=dict(item.metadata),
            )
        )
    return CheckTriageRequest(
        failed_items=tuple(perturbed_items),
        project=request.project,
    )


def ranking_ndcg(ranked_ids: Iterable[str], expected_order: tuple[str, ...]) -> float:
    """Score a ranking against the expected ordering using a small NDCG variant."""

    ranked = list(ranked_ids)
    if not expected_order:
        return 1.0
    relevance = {
        check_id: float(len(expected_order) - index)
        for index, check_id in enumerate(expected_order)
    }

    def _dcg(ids: list[str]) -> float:
        score = 0.0
        for index, check_id in enumerate(ids, start=1):
            gain = relevance.get(check_id, 0.0)
            if gain <= 0.0:
                continue
            score += gain / index
        return score

    ideal = _dcg(list(expected_order))
    if ideal <= 0.0:
        return 1.0
    return _dcg(ranked) / ideal
