"""Render a PR-comment Markdown blob from a CloudAnalyzer summary JSON.

Two input shapes are supported, auto-detected by structure:

- **`ca check` suite summary** — top-level has ``summary`` and ``checks``
  (the JSON written by ``summary_output_json:`` in ``cloudanalyzer.yaml``).
- **Single-run summary** — top-level has ``overall_quality_gate`` and
  ``map``/``trajectory`` (the JSON written by ``ca run-evaluate
  --output-json`` and ``ca benchmark eval --output-json``).

When a ``baseline`` JSON of the same shape is supplied, metric deltas
appear next to each value: ``↑`` if the number went up, ``↓`` if it went
down, ``◦`` if unchanged within rounding. Whether ``↑``/``↓`` is good or
bad is left to the reader because the gate direction is right next to
the metric (e.g. ``Chamfer 0.084 ↑ from 0.041, gate <= 0.05``).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ca.core.check_triage import (
    CheckTriageItem,
    build_check_triage_request,
    rank_failed_checks,
)


# --------------------------------------------------------------------- detection


def _looks_like_check_suite(data: Mapping[str, Any]) -> bool:
    return isinstance(data.get("summary"), Mapping) and isinstance(
        data.get("checks"), Sequence
    )


def _looks_like_single_run(data: Mapping[str, Any]) -> bool:
    if "overall_quality_gate" not in data:
        return False
    return isinstance(data.get("map"), Mapping) or isinstance(
        data.get("trajectory"), Mapping
    )


# --------------------------------------------------------------------- delta


def _format_delta(current: float, previous: float | None, metric_key: str) -> str:
    """Return ``" (was 0.987 ↓)"`` style suffix or empty string.

    Arrows track *value direction* (``↑`` = number went up, ``↓`` = went
    down). Whether that is good or bad depends on the gate direction
    shown right next to the metric, so the reader can judge in context.
    """
    if previous is None:
        return ""
    try:
        diff = float(current) - float(previous)
    except (TypeError, ValueError):
        return ""
    if abs(diff) < 1e-6:
        arrow = "◦"
    elif diff > 0:
        arrow = "↑"
    else:
        arrow = "↓"
    return f" (was {_fmt_metric_value(metric_key, float(previous))} {arrow})"


# --------------------------------------------------------------------- check suite


def _index_baseline_checks(baseline: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if baseline is None:
        return {}
    checks = baseline.get("checks")
    if not isinstance(checks, Sequence):
        return {}
    out: dict[str, Mapping[str, Any]] = {}
    for entry in checks:
        if isinstance(entry, Mapping) and isinstance(entry.get("id"), str):
            out[entry["id"]] = entry
    return out


_ARTIFACT_METRIC_KEYS = ("auc", "chamfer_distance", "hausdorff_distance", "awd_m", "scs")
_TRAJECTORY_METRIC_KEYS = ("ate_rmse", "rpe_rmse", "drift_endpoint", "coverage_ratio")
_RUN_METRIC_KEYS = (
    "map_auc",
    "map_chamfer_distance",
    "trajectory_ate_rmse",
    "trajectory_rpe_rmse",
    "trajectory_drift_endpoint",
    "coverage_ratio",
)
_IMAGE_METRIC_KEYS = ("psnr_mean", "ssim_mean")
_RENDERED_METRIC_KEYS = ("psnr_mean", "ssim_mean", "lpips_mean", "auc", "chamfer_distance")
_STRUCTURE_METRIC_KEYS = ("plane_normal_dispersion", "coplanar_offset_rmse", "num_plane_patches")


_METRIC_LABELS = {
    "auc": "AUC",
    "chamfer_distance": "Chamfer",
    "hausdorff_distance": "Hausdorff",
    "ate_rmse": "ATE",
    "rpe_rmse": "RPE",
    "drift_endpoint": "Drift",
    "coverage_ratio": "Coverage",
    "map_auc": "Map AUC",
    "map_chamfer_distance": "Map Chamfer",
    "trajectory_ate_rmse": "Traj ATE",
    "trajectory_rpe_rmse": "Traj RPE",
    "trajectory_drift_endpoint": "Traj Drift",
    "best_f1.f1": "Best F1",
    "f1": "F1",
    "iou": "IoU",
    "precision": "Precision",
    "recall": "Recall",
    "matched_poses": "Matched poses",
    "psnr_mean": "PSNR",
    "ssim_mean": "SSIM",
    "lpips_mean": "LPIPS",
    "awd_m": "AWD",
    "scs": "SCS",
    "plane_normal_dispersion": "Plane normal dispersion",
    "coplanar_offset_rmse": "Coplanar offset RMSE",
    "num_plane_patches": "Plane patches",
}


def _check_metric_keys(kind: str) -> tuple[str, ...]:
    if kind == "trajectory":
        return _TRAJECTORY_METRIC_KEYS
    if kind == "run":
        return _RUN_METRIC_KEYS
    if kind == "image":
        return _IMAGE_METRIC_KEYS
    if kind == "rendered":
        return _RENDERED_METRIC_KEYS
    if kind == "structure":
        return _STRUCTURE_METRIC_KEYS
    return _ARTIFACT_METRIC_KEYS


def _fmt_metric_value(key: str, value: float) -> str:
    if key == "coverage_ratio":
        return f"{value:.1%}"
    if key == "psnr_mean":
        return f"{value:.4f} dB"
    if key in {"awd_m", "coplanar_offset_rmse"}:
        return f"{value:.4f} m"
    return f"{value:.4f}"


def _format_check_metrics(
    check: Mapping[str, Any],
    baseline_check: Mapping[str, Any] | None,
) -> str:
    summary_raw = check.get("summary")
    summary: Mapping[str, Any] = summary_raw if isinstance(summary_raw, Mapping) else {}
    base_summary: Mapping[str, Any] | None = None
    if isinstance(baseline_check, Mapping):
        base_raw = baseline_check.get("summary")
        if isinstance(base_raw, Mapping):
            base_summary = base_raw
    pieces: list[str] = []
    for key in _check_metric_keys(str(check.get("kind", ""))):
        if key not in summary:
            continue
        value = summary[key]
        if not isinstance(value, (int, float)):
            continue
        previous: float | None = None
        if base_summary is not None:
            base_value = base_summary.get(key)
            if isinstance(base_value, (int, float)):
                previous = float(base_value)
        pieces.append(
            f"{_METRIC_LABELS.get(key, key)}={_fmt_metric_value(key, float(value))}"
            f"{_format_delta(float(value), previous, key)}"
        )
    return ", ".join(pieces) if pieces else "—"


def _check_reasons(check: Mapping[str, Any]) -> list[str]:
    result = check.get("result") if isinstance(check.get("result"), Mapping) else None
    if isinstance(result, Mapping):
        gate = result.get("quality_gate") or result.get("overall_quality_gate")
        if isinstance(gate, Mapping):
            reasons = gate.get("reasons")
            if isinstance(reasons, Sequence):
                return [str(r) for r in reasons if isinstance(r, str)]
    return []


def _triage_block(checks: Sequence[Mapping[str, Any]], project: str | None) -> list[str]:
    executed = [dict(c) for c in checks if isinstance(c, Mapping)]
    if not any(c.get("passed") is False for c in executed):
        return []
    request = build_check_triage_request(executed, project=project)
    result = rank_failed_checks(request)
    if not result.ranked_items:
        return []
    lines = ["", "### Recommended triage"]
    for item in result.ranked_items:
        primary_reason = item.reasons[0] if item.reasons else "no reason recorded"
        lines.append(
            f"{item.rank}. `{item.check_id}` ({item.kind}) — {primary_reason}"
        )
    return lines


def _format_check_suite(
    data: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any] | None,
    project: str | None,
) -> str:
    summary = data["summary"]
    checks = data["checks"]
    passed = bool(summary.get("passed"))
    proj = project or data.get("project")

    header = "## CloudAnalyzer QA: " + ("PASS ✅" if passed else "FAIL ❌")
    if proj:
        header += f" — `{proj}`"
    lines = [header, ""]

    totals = summary.get("total_checks")
    failed = summary.get("failed_checks", 0)
    passed_count = summary.get("passed_checks", 0)
    if isinstance(totals, int):
        lines.append(f"**Checks**: {passed_count}/{totals} passed, {failed} failed")
        lines.append("")

    baseline_index = _index_baseline_checks(baseline)

    lines.append("| Status | Check | Kind | Metrics | Reasons |")
    lines.append("|---|---|---|---|---|")
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        status = "✅" if check.get("passed") else "❌"
        check_id = check.get("id", "?")
        kind = check.get("kind", "?")
        metrics_str = _format_check_metrics(check, baseline_index.get(str(check_id)))
        reasons = _check_reasons(check)
        reasons_str = "; ".join(reasons) if reasons else "—"
        lines.append(
            f"| {status} | `{check_id}` | {kind} | {metrics_str} | {reasons_str} |"
        )

    lines.extend(_triage_block(checks, proj))
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------- single run


def _format_single_run(
    data: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any] | None,
    project: str | None,
) -> str:
    overall = data.get("overall_quality_gate")
    if isinstance(overall, Mapping):
        passed = bool(overall.get("passed"))
        reasons = [str(r) for r in overall.get("reasons", []) if isinstance(r, str)]
    else:
        passed = True
        reasons = []

    header = "## CloudAnalyzer QA: " + ("PASS ✅" if passed else "FAIL ❌")
    benchmark = data.get("benchmark") if isinstance(data.get("benchmark"), Mapping) else None
    if benchmark is not None:
        suite_name = benchmark.get("suite", "?")
        sequence = benchmark.get("sequence", "?")
        version = benchmark.get("version", "?")
        header += f" — `{suite_name}` v{version} (sequence: `{sequence}`)"
    elif project:
        header += f" — `{project}`"

    lines = [header, ""]

    map_raw = data.get("map")
    map_block: Mapping[str, Any] = map_raw if isinstance(map_raw, Mapping) else {}
    traj_raw = data.get("trajectory")
    traj_block: Mapping[str, Any] = traj_raw if isinstance(traj_raw, Mapping) else {}
    base_map: Mapping[str, Any] | None = None
    base_traj: Mapping[str, Any] | None = None
    if isinstance(baseline, Mapping):
        bm = baseline.get("map")
        bt = baseline.get("trajectory")
        if isinstance(bm, Mapping):
            base_map = bm
        if isinstance(bt, Mapping):
            base_traj = bt

    def _maybe_metric(parent: Mapping[str, Any], dotted: str) -> float | None:
        current: Any = parent
        for part in dotted.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return None
            current = current[part]
        if isinstance(current, (int, float)):
            return float(current)
        return None

    bullets: list[str] = []

    for metric_path, label, key in (
        ("auc", "Map AUC", "auc"),
        ("chamfer_distance", "Map Chamfer", "chamfer_distance"),
        ("hausdorff_distance", "Map Hausdorff", "hausdorff_distance"),
        ("best_f1.f1", "Map Best F1", "f1"),
    ):
        current = _maybe_metric(map_block, metric_path)
        if current is None:
            continue
        previous = _maybe_metric(base_map, metric_path) if base_map is not None else None
        bullets.append(
            f"- {label}: {_fmt_metric_value(key, current)}{_format_delta(current, previous, metric_path.split('.')[-1])}"
        )

    for metric_path, label, key in (
        ("ate.rmse", "Trajectory ATE", "ate_rmse"),
        ("rpe_translation.rmse", "Trajectory RPE", "rpe_rmse"),
        ("drift.endpoint", "Trajectory Drift", "drift_endpoint"),
        ("matching.coverage_ratio", "Trajectory Coverage", "coverage_ratio"),
    ):
        current = _maybe_metric(traj_block, metric_path)
        if current is None:
            continue
        previous = _maybe_metric(base_traj, metric_path) if base_traj is not None else None
        bullets.append(
            f"- {label}: {_fmt_metric_value(key, current)}{_format_delta(current, previous, key)}"
        )

    lines.extend(bullets)

    if reasons:
        lines.append("")
        lines.append("**Failed gates:**")
        for reason in reasons:
            lines.append(f"- {reason}")

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------- entry

_REPO_URL = "https://github.com/rsasaki0109/CloudAnalyzer"
_GENERATED_BY_FOOTER = (
    f"\n---\n\nGenerated by [CloudAnalyzer]({_REPO_URL})\n"
)


def build_pr_comment(
    summary: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any] | None = None,
    project: str | None = None,
) -> str:
    """Render a Markdown PR comment for a CloudAnalyzer summary."""
    if _looks_like_check_suite(summary):
        body = _format_check_suite(summary, baseline=baseline, project=project)
    elif _looks_like_single_run(summary):
        body = _format_single_run(summary, baseline=baseline, project=project)
    else:
        raise ValueError(
            "Unrecognized summary JSON shape. Expected `ca check` suite summary "
            "(`summary` + `checks`) or `ca run-evaluate`/`ca benchmark eval` "
            "single-run (`overall_quality_gate` + `map`/`trajectory`)."
        )
    return body.rstrip() + _GENERATED_BY_FOOTER
