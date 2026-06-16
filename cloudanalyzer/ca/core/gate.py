"""Shared gate severity and CI exit policy helpers."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence, cast


GateSeverity = Literal["fail", "warn", "soft_fail", "skip", "not_applicable"]
GateMode = Literal["default", "warn_only", "strict"]
GateStatus = Literal[
    "pass",
    "fail",
    "warn",
    "soft_fail",
    "skip",
    "not_applicable",
    "info",
]

GATE_SUMMARY_SCHEMA_VERSION = "cloudanalyzer.gate_summary.v0.1"
GATE_SEVERITIES: tuple[GateSeverity, ...] = (
    "fail",
    "warn",
    "soft_fail",
    "skip",
    "not_applicable",
)


def normalize_gate_severity(value: object, context: str = "severity") -> GateSeverity:
    """Normalize a user-facing gate severity string."""
    if value is None:
        return "fail"
    if not isinstance(value, str):
        raise ValueError(f"{context} must be one of: {', '.join(GATE_SEVERITIES)}")
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in GATE_SEVERITIES:
        raise ValueError(f"{context} must be one of: {', '.join(GATE_SEVERITIES)}")
    return cast(GateSeverity, normalized)


def gate_status_for_check(check: Mapping[str, Any]) -> GateStatus:
    """Return the normalized gate status for one executed check summary."""
    severity = normalize_gate_severity(check.get("severity"))
    if severity in {"skip", "not_applicable"}:
        return severity

    passed = check.get("passed")
    if passed is True:
        return "pass"
    if passed is None:
        return "info"
    if severity == "warn":
        return "warn"
    if severity == "soft_fail":
        return "soft_fail"
    return "fail"


def summarize_gate_policy(
    checks: Sequence[Mapping[str, Any]],
    *,
    mode: GateMode = "default",
) -> dict[str, Any]:
    """Summarize check statuses into a stable CI gate policy block."""
    if mode not in {"default", "warn_only", "strict"}:
        raise ValueError("mode must be one of: default, warn_only, strict")

    status_ids: dict[GateStatus, list[str]] = {
        "pass": [],
        "fail": [],
        "warn": [],
        "soft_fail": [],
        "skip": [],
        "not_applicable": [],
        "info": [],
    }
    for check in checks:
        status = gate_status_for_check(check)
        status_ids[status].append(str(check.get("id", "")))

    if mode == "warn_only":
        blocking_failed_ids: list[str] = []
    elif mode == "strict":
        blocking_failed_ids = [
            check_id
            for status in ("fail", "warn", "soft_fail", "skip", "not_applicable", "info")
            for check_id in status_ids[status]
        ]
    else:
        blocking_failed_ids = list(status_ids["fail"])

    return {
        "schema_version": GATE_SUMMARY_SCHEMA_VERSION,
        "mode": mode,
        "passed": not blocking_failed_ids,
        "exit_code": 0 if not blocking_failed_ids else 1,
        "blocking_failed_ids": blocking_failed_ids,
        "pass_count": len(status_ids["pass"]),
        "fail_count": len(status_ids["fail"]),
        "warn_count": len(status_ids["warn"]),
        "soft_fail_count": len(status_ids["soft_fail"]),
        "skip_count": len(status_ids["skip"]),
        "not_applicable_count": len(status_ids["not_applicable"]),
        "info_count": len(status_ids["info"]),
        "passed_ids": status_ids["pass"],
        "failed_ids": status_ids["fail"],
        "warning_ids": status_ids["warn"],
        "soft_failed_ids": status_ids["soft_fail"],
        "skipped_ids": status_ids["skip"],
        "not_applicable_ids": status_ids["not_applicable"],
        "ungated_ids": status_ids["info"],
    }


__all__ = [
    "GATE_SEVERITIES",
    "GATE_SUMMARY_SCHEMA_VERSION",
    "GateMode",
    "GateSeverity",
    "GateStatus",
    "gate_status_for_check",
    "normalize_gate_severity",
    "summarize_gate_policy",
]
