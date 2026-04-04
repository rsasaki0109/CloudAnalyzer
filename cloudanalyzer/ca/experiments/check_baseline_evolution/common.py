"""Shared datasets and perturbations for baseline evolution experiments."""

from __future__ import annotations

from dataclasses import dataclass

from ca.core import (
    BaselineCheckSnapshot,
    BaselineEvolutionRequest,
    BaselineEvolutionSnapshot,
)


@dataclass(slots=True)
class BaselineEvolutionDatasetCase:
    """Comparable baseline-promotion scenario."""

    name: str
    description: str
    request: BaselineEvolutionRequest
    expected_decision: str


def make_check(
    check_id: str,
    kind: str,
    *,
    passed: bool,
    metrics: dict[str, float],
    gate: dict[str, float],
    triage_rank: int | None = None,
) -> BaselineCheckSnapshot:
    """Build a deterministic QA check snapshot."""

    return BaselineCheckSnapshot(
        check_id=check_id,
        kind=kind,
        passed=passed,
        metrics=dict(metrics),
        gate=dict(gate),
        triage_rank=triage_rank,
    )


def make_snapshot(
    label: str,
    checks: tuple[BaselineCheckSnapshot, ...],
    *,
    passed: bool,
    failed_check_ids: tuple[str, ...] = (),
) -> BaselineEvolutionSnapshot:
    """Build a baseline-evolution snapshot from fixed checks."""

    return BaselineEvolutionSnapshot(
        label=label,
        checks=checks,
        passed=passed,
        failed_check_ids=failed_check_ids,
        metadata={"source": label},
    )


_MAP_GATE = {"min_auc": 0.95, "max_chamfer": 0.02}
_TRAJECTORY_GATE = {
    "max_ate": 0.5,
    "max_rpe": 0.2,
    "max_drift": 0.2,
    "min_coverage": 0.9,
}


def _passing_checks(
    *,
    map_auc: float,
    map_chamfer: float,
    ate: float,
    rpe: float,
    drift: float,
    coverage: float,
) -> tuple[BaselineCheckSnapshot, ...]:
    return (
        make_check(
            "mapping-postprocess",
            "artifact",
            passed=True,
            metrics={"auc": map_auc, "chamfer": map_chamfer},
            gate=_MAP_GATE,
            triage_rank=None,
        ),
        make_check(
            "localization-run",
            "trajectory",
            passed=True,
            metrics={
                "ate": ate,
                "rpe": rpe,
                "drift": drift,
                "coverage": coverage,
            },
            gate=_TRAJECTORY_GATE,
            triage_rank=None,
        ),
    )


def build_default_datasets() -> list[BaselineEvolutionDatasetCase]:
    """Create deterministic promote/keep/reject scenarios."""

    return [
        BaselineEvolutionDatasetCase(
            name="stable_improvement_window",
            description=(
                "Candidate should be promoted after a stable passing window with stronger margins."
            ),
            request=BaselineEvolutionRequest(
                candidate=make_snapshot(
                    "candidate-promote",
                    _passing_checks(
                        map_auc=0.975,
                        map_chamfer=0.014,
                        ate=0.34,
                        rpe=0.12,
                        drift=0.11,
                        coverage=0.95,
                    ),
                    passed=True,
                ),
                history=(
                    make_snapshot(
                        "history-a",
                        _passing_checks(
                            map_auc=0.956,
                            map_chamfer=0.019,
                            ate=0.46,
                            rpe=0.18,
                            drift=0.17,
                            coverage=0.91,
                        ),
                        passed=True,
                    ),
                    make_snapshot(
                        "history-b",
                        _passing_checks(
                            map_auc=0.959,
                            map_chamfer=0.0185,
                            ate=0.44,
                            rpe=0.17,
                            drift=0.16,
                            coverage=0.92,
                        ),
                        passed=True,
                    ),
                ),
            ),
            expected_decision="promote",
        ),
        BaselineEvolutionDatasetCase(
            name="candidate_failure_reject",
            description="Candidate that fails the quality gate should be rejected immediately.",
            request=BaselineEvolutionRequest(
                candidate=make_snapshot(
                    "candidate-reject",
                    (
                        make_check(
                            "mapping-postprocess",
                            "artifact",
                            passed=False,
                            metrics={"auc": 0.89, "chamfer": 0.032},
                            gate=_MAP_GATE,
                            triage_rank=1,
                        ),
                        make_check(
                            "localization-run",
                            "trajectory",
                            passed=False,
                            metrics={
                                "ate": 0.72,
                                "rpe": 0.28,
                                "drift": 0.31,
                                "coverage": 0.81,
                            },
                            gate=_TRAJECTORY_GATE,
                            triage_rank=2,
                        ),
                    ),
                    passed=False,
                    failed_check_ids=("mapping-postprocess", "localization-run"),
                ),
                history=(
                    make_snapshot(
                        "history-pass",
                        _passing_checks(
                            map_auc=0.961,
                            map_chamfer=0.018,
                            ate=0.43,
                            rpe=0.16,
                            drift=0.15,
                            coverage=0.92,
                        ),
                        passed=True,
                    ),
                ),
            ),
            expected_decision="reject",
        ),
        BaselineEvolutionDatasetCase(
            name="insufficient_history_keep",
            description="A strong candidate without enough history should stay in keep mode.",
            request=BaselineEvolutionRequest(
                candidate=make_snapshot(
                    "candidate-keep-insufficient",
                    _passing_checks(
                        map_auc=0.977,
                        map_chamfer=0.013,
                        ate=0.32,
                        rpe=0.11,
                        drift=0.1,
                        coverage=0.96,
                    ),
                    passed=True,
                ),
                history=(
                    make_snapshot(
                        "history-only",
                        _passing_checks(
                            map_auc=0.958,
                            map_chamfer=0.018,
                            ate=0.45,
                            rpe=0.17,
                            drift=0.15,
                            coverage=0.92,
                        ),
                        passed=True,
                    ),
                ),
            ),
            expected_decision="keep",
        ),
        BaselineEvolutionDatasetCase(
            name="recent_failure_keep",
            description=(
                "A recovering candidate should not be promoted immediately after a recent failure."
            ),
            request=BaselineEvolutionRequest(
                candidate=make_snapshot(
                    "candidate-after-failure",
                    _passing_checks(
                        map_auc=0.973,
                        map_chamfer=0.0145,
                        ate=0.35,
                        rpe=0.13,
                        drift=0.11,
                        coverage=0.95,
                    ),
                    passed=True,
                ),
                history=(
                    make_snapshot(
                        "older-pass",
                        _passing_checks(
                            map_auc=0.962,
                            map_chamfer=0.017,
                            ate=0.42,
                            rpe=0.16,
                            drift=0.15,
                            coverage=0.93,
                        ),
                        passed=True,
                    ),
                    make_snapshot(
                        "recent-fail",
                        (
                            make_check(
                                "mapping-postprocess",
                                "artifact",
                                passed=False,
                                metrics={"auc": 0.92, "chamfer": 0.025},
                                gate=_MAP_GATE,
                                triage_rank=1,
                            ),
                            make_check(
                                "localization-run",
                                "trajectory",
                                passed=True,
                                metrics={
                                    "ate": 0.44,
                                    "rpe": 0.16,
                                    "drift": 0.14,
                                    "coverage": 0.92,
                                },
                                gate=_TRAJECTORY_GATE,
                            ),
                        ),
                        passed=False,
                        failed_check_ids=("mapping-postprocess",),
                    ),
                ),
            ),
            expected_decision="keep",
        ),
        BaselineEvolutionDatasetCase(
            name="mixed_tradeoff_keep",
            description=(
                "Candidate with a stronger mean margin but worse weakest margin should stay keep."
            ),
            request=BaselineEvolutionRequest(
                candidate=make_snapshot(
                    "candidate-tradeoff",
                    _passing_checks(
                        map_auc=0.983,
                        map_chamfer=0.013,
                        ate=0.47,
                        rpe=0.19,
                        drift=0.18,
                        coverage=0.905,
                    ),
                    passed=True,
                ),
                history=(
                    make_snapshot(
                        "history-c",
                        _passing_checks(
                            map_auc=0.969,
                            map_chamfer=0.016,
                            ate=0.39,
                            rpe=0.15,
                            drift=0.13,
                            coverage=0.93,
                        ),
                        passed=True,
                    ),
                    make_snapshot(
                        "history-d",
                        _passing_checks(
                            map_auc=0.968,
                            map_chamfer=0.0165,
                            ate=0.38,
                            rpe=0.15,
                            drift=0.12,
                            coverage=0.94,
                        ),
                        passed=True,
                    ),
                ),
            ),
            expected_decision="keep",
        ),
    ]


def _perturb_metric(metric_name: str, value: float, factor: float) -> float:
    if metric_name in {"auc", "map_auc", "coverage"}:
        return float(value / factor)
    return float(value * factor)


def perturb_request(request: BaselineEvolutionRequest, factor: float) -> BaselineEvolutionRequest:
    """Create a deterministic perturbed request for decision-stability checks."""

    def _perturb_snapshot(snapshot: BaselineEvolutionSnapshot) -> BaselineEvolutionSnapshot:
        checks = []
        for check in snapshot.checks:
            metrics = {
                name: _perturb_metric(name, value, factor)
                for name, value in check.metrics.items()
            }
            checks.append(
                BaselineCheckSnapshot(
                    check_id=check.check_id,
                    kind=check.kind,
                    passed=check.passed,
                    metrics=metrics,
                    gate=dict(check.gate),
                    triage_rank=check.triage_rank,
                    metadata=dict(check.metadata),
                )
            )
        return BaselineEvolutionSnapshot(
            label=snapshot.label,
            checks=tuple(checks),
            passed=snapshot.passed,
            failed_check_ids=snapshot.failed_check_ids,
            metadata=dict(snapshot.metadata),
        )

    return BaselineEvolutionRequest(
        candidate=_perturb_snapshot(request.candidate),
        history=tuple(_perturb_snapshot(snapshot) for snapshot in request.history),
    )
