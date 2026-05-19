"""Build a time-series view of QA gate metrics across many bundles.

`ca bundle diff` answers "is run X better or worse than run Y?". This
module answers "how has the gate moved across the last N runs?" by
reading any number of `qa_bundle.zip` archives, sorting them by their
metadata's ``created_at`` stamp, and rendering a per-metric trend table.

Both bundle summary shapes are supported, matching :mod:`ca.bundle` /
:mod:`ca.pr_comment`:

- ``check_suite``: per-check timelines (one section per ``check_id``).
- ``single_run``: a single table covering map + trajectory metrics.

Mixing ``summary_kind`` across input bundles is rejected — the metric
columns wouldn't line up and a half-blank table is more misleading
than a clean error.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


METADATA_FILENAME = "metadata.json"
SUMMARY_FILENAME = "summary.json"

HISTORY_VERSION = 1


@dataclass(slots=True)
class HistoryEntry:
    """One point in the time-series: a single QA bundle reduced to metrics."""

    bundle_path: str
    created_at: str            # ISO-8601 UTC from metadata.json
    summary_kind: str          # "check_suite" | "single_run"
    project: str | None
    git_commit: str | None
    pr_number: str | None
    overall_passed: bool | None
    # For single_run, flat dotted keys like "map.auc", "trajectory.ate.rmse".
    metrics: dict[str, float] = field(default_factory=dict)
    # For check_suite, per-check status + metric dicts.
    per_check_passed: dict[str, bool] = field(default_factory=dict)
    per_check_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    per_check_kind: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------- metric keys

_SINGLE_RUN_METRIC_PATHS: tuple[tuple[str, str, str], ...] = (
    ("map.auc",                    "Map AUC",        "auc"),
    ("map.chamfer_distance",       "Map Chamfer",    "chamfer_distance"),
    ("map.hausdorff_distance",     "Map Hausdorff",  "hausdorff_distance"),
    ("map.best_f1.f1",             "Map Best F1",    "f1"),
    ("trajectory.ate.rmse",        "Traj ATE",       "ate_rmse"),
    ("trajectory.rpe_translation.rmse", "Traj RPE",  "rpe_rmse"),
    ("trajectory.drift.endpoint",  "Traj Drift",     "drift_endpoint"),
    ("trajectory.matching.coverage_ratio", "Coverage", "coverage_ratio"),
)

_ARTIFACT_METRIC_KEYS: tuple[str, ...] = (
    "auc",
    "chamfer_distance",
    "hausdorff_distance",
    "f1",
    "precision",
    "recall",
)
_TRAJECTORY_METRIC_KEYS: tuple[str, ...] = (
    "ate_rmse",
    "rpe_translation_rmse",
    "drift_endpoint",
    "coverage_ratio",
)
_RUN_METRIC_KEYS: tuple[str, ...] = _ARTIFACT_METRIC_KEYS + _TRAJECTORY_METRIC_KEYS


def _check_metric_keys(kind: str) -> tuple[str, ...]:
    if kind == "trajectory":
        return _TRAJECTORY_METRIC_KEYS
    if kind == "run":
        return _RUN_METRIC_KEYS
    return _ARTIFACT_METRIC_KEYS


def _fmt_metric_value(key: str, value: float) -> str:
    if key == "coverage_ratio":
        return f"{value:.1%}"
    return f"{value:.4f}"


# --------------------------------------------------------------------- extraction


def _read_bundle_payload(bundle_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with zipfile.ZipFile(bundle_path, mode="r") as zf:
        names = zf.namelist()
        if METADATA_FILENAME not in names:
            raise ValueError(f"{bundle_path}: missing {METADATA_FILENAME}")
        if SUMMARY_FILENAME not in names:
            raise ValueError(f"{bundle_path}: missing {SUMMARY_FILENAME}")
        with zf.open(METADATA_FILENAME) as fp:
            metadata = json.loads(io.TextIOWrapper(fp, encoding="utf-8").read())
        with zf.open(SUMMARY_FILENAME) as fp:
            summary = json.loads(io.TextIOWrapper(fp, encoding="utf-8").read())
    if not isinstance(metadata, dict) or not isinstance(summary, dict):
        raise ValueError(f"{bundle_path}: metadata/summary must be JSON objects")
    return metadata, summary


def _maybe_metric(parent: Mapping[str, Any], dotted: str) -> float | None:
    current: Any = parent
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _extract_single_run_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for path, _, _ in _SINGLE_RUN_METRIC_PATHS:
        value = _maybe_metric(summary, path)
        if value is not None:
            metrics[path] = value
    return metrics


def _extract_check_suite_per_check(
    summary: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, dict[str, float]], dict[str, str]]:
    passed: dict[str, bool] = {}
    metrics: dict[str, dict[str, float]] = {}
    kinds: dict[str, str] = {}
    checks = summary.get("checks")
    if not isinstance(checks, Sequence):
        return passed, metrics, kinds
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        check_id = str(check.get("id") or check.get("check_id") or "")
        if not check_id:
            continue
        kind = str(check.get("kind", ""))
        kinds[check_id] = kind
        passed[check_id] = bool(check.get("passed"))
        check_metrics: dict[str, float] = {}
        check_summary = check.get("summary")
        if isinstance(check_summary, Mapping):
            for key in _check_metric_keys(kind):
                value = check_summary.get(key)
                if isinstance(value, (int, float)):
                    check_metrics[key] = float(value)
        metrics[check_id] = check_metrics
    return passed, metrics, kinds


def extract_history_entry(bundle_path: str | Path) -> HistoryEntry:
    """Reduce one QA bundle to a normalized HistoryEntry."""
    path = Path(bundle_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata, summary = _read_bundle_payload(path)
    summary_kind = str(metadata.get("summary_kind") or "")
    if summary_kind not in {"check_suite", "single_run"}:
        raise ValueError(
            f"{path}: unsupported summary_kind {summary_kind!r}; "
            "expected 'check_suite' or 'single_run'"
        )

    overall_passed: bool | None
    metrics: dict[str, float] = {}
    per_check_passed: dict[str, bool] = {}
    per_check_metrics: dict[str, dict[str, float]] = {}
    per_check_kind: dict[str, str] = {}

    if summary_kind == "single_run":
        gate = summary.get("overall_quality_gate")
        overall_passed = bool(gate.get("passed")) if isinstance(gate, Mapping) else None
        metrics = _extract_single_run_metrics(summary)
    else:
        suite = summary.get("summary")
        overall_passed = bool(suite.get("passed")) if isinstance(suite, Mapping) else None
        per_check_passed, per_check_metrics, per_check_kind = _extract_check_suite_per_check(
            summary
        )

    return HistoryEntry(
        bundle_path=str(path),
        created_at=str(metadata.get("created_at") or ""),
        summary_kind=summary_kind,
        project=metadata.get("project") if isinstance(metadata.get("project"), str) else None,
        git_commit=metadata.get("git_commit") if isinstance(metadata.get("git_commit"), str) else None,
        pr_number=metadata.get("pr_number") if isinstance(metadata.get("pr_number"), str) else None,
        overall_passed=overall_passed,
        metrics=metrics,
        per_check_passed=per_check_passed,
        per_check_metrics=per_check_metrics,
        per_check_kind=per_check_kind,
    )


def build_history_series(bundles: Sequence[str | Path]) -> list[HistoryEntry]:
    """Read N bundles, validate uniform summary_kind, return them oldest → newest."""
    if not bundles:
        raise ValueError("build_history_series requires at least one bundle")
    entries = [extract_history_entry(b) for b in bundles]
    kinds = {entry.summary_kind for entry in entries}
    if len(kinds) > 1:
        raise ValueError(
            f"Cannot build history across mixed summary_kinds: {sorted(kinds)}"
        )
    entries.sort(key=lambda e: e.created_at)
    return entries


# --------------------------------------------------------------------- rendering


def _short_commit(commit: str | None) -> str:
    if not commit:
        return "—"
    return commit[:8]


def _label_for_entry(entry: HistoryEntry) -> str:
    parts: list[str] = []
    if entry.created_at:
        parts.append(entry.created_at)
    parts.append(_short_commit(entry.git_commit))
    return " · ".join(parts) if parts else Path(entry.bundle_path).stem


def _status_cell(passed: bool | None) -> str:
    if passed is True:
        return "✅"
    if passed is False:
        return "❌"
    return "—"


def _format_single_run_table(entries: Sequence[HistoryEntry]) -> list[str]:
    headers = ["When", "Commit", "PR", "Status"]
    keys: list[str] = []
    labels: list[str] = []
    metric_keys: list[str] = []
    for path, label, key in _SINGLE_RUN_METRIC_PATHS:
        if any(path in entry.metrics for entry in entries):
            keys.append(path)
            labels.append(label)
            metric_keys.append(key)
    headers.extend(labels)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for entry in entries:
        row = [
            entry.created_at or "—",
            _short_commit(entry.git_commit),
            f"#{entry.pr_number}" if entry.pr_number else "—",
            _status_cell(entry.overall_passed),
        ]
        for path, mkey in zip(keys, metric_keys):
            value = entry.metrics.get(path)
            row.append(_fmt_metric_value(mkey, value) if value is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _format_check_suite_section(
    check_id: str,
    kind: str,
    entries: Sequence[HistoryEntry],
) -> list[str]:
    metric_keys = _check_metric_keys(kind)
    used_keys = [
        key
        for key in metric_keys
        if any(key in entry.per_check_metrics.get(check_id, {}) for entry in entries)
    ]
    lines = [f"### `{check_id}` ({kind or '?'})", ""]
    headers = ["When", "Commit", "Status"] + list(used_keys)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for entry in entries:
        row = [
            entry.created_at or "—",
            _short_commit(entry.git_commit),
            _status_cell(entry.per_check_passed.get(check_id)),
        ]
        check_metrics = entry.per_check_metrics.get(check_id, {})
        for key in used_keys:
            value = check_metrics.get(key)
            row.append(_fmt_metric_value(key, value) if value is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def render_history_markdown(entries: Sequence[HistoryEntry]) -> str:
    """Render a Markdown history report for a list of HistoryEntry."""
    if not entries:
        return "## CloudAnalyzer QA history\n\n_No bundles supplied._\n"

    kind = entries[0].summary_kind
    header = ["## CloudAnalyzer QA history", ""]
    header.append(f"**Bundles**: {len(entries)} ({kind})")
    project = next((entry.project for entry in entries if entry.project), None)
    if project:
        header.append(f"**Project**: `{project}`")
    header.append("")

    if kind == "single_run":
        return "\n".join(header + _format_single_run_table(entries)).rstrip() + "\n"

    # check_suite: gather every check_id seen across all entries; preserve
    # first-appearance order so the output is stable per run.
    seen: list[str] = []
    seen_set: set[str] = set()
    kinds: dict[str, str] = {}
    for entry in entries:
        for check_id in entry.per_check_passed:
            if check_id not in seen_set:
                seen.append(check_id)
                seen_set.add(check_id)
            if check_id not in kinds and entry.per_check_kind.get(check_id):
                kinds[check_id] = entry.per_check_kind[check_id]
    body: list[str] = []
    for check_id in seen:
        body.extend(_format_check_suite_section(check_id, kinds.get(check_id, ""), entries))
    return "\n".join(header + body).rstrip() + "\n"


def render_history_json(entries: Sequence[HistoryEntry]) -> dict[str, Any]:
    """Render a JSON-friendly payload for dashboards / tooling."""
    return {
        "history_version": HISTORY_VERSION,
        "summary_kind": entries[0].summary_kind if entries else None,
        "entries": [
            {
                "bundle_path": entry.bundle_path,
                "created_at": entry.created_at,
                "summary_kind": entry.summary_kind,
                "project": entry.project,
                "git_commit": entry.git_commit,
                "pr_number": entry.pr_number,
                "overall_passed": entry.overall_passed,
                "metrics": entry.metrics,
                "per_check_passed": entry.per_check_passed,
                "per_check_metrics": entry.per_check_metrics,
                "per_check_kind": entry.per_check_kind,
            }
            for entry in entries
        ],
    }


# --------------------------------------------------------------------- discovery


def discover_bundles(directory: str | Path, pattern: str = "*.zip") -> list[Path]:
    """List bundle archives in ``directory`` matching ``pattern``, sorted lexicographically."""
    root = Path(directory).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted(p for p in root.glob(pattern) if p.is_file())


__all__ = [
    "HISTORY_VERSION",
    "HistoryEntry",
    "build_history_series",
    "discover_bundles",
    "extract_history_entry",
    "render_history_json",
    "render_history_markdown",
]
