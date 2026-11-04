"""Trajectory batch reports."""

from html import escape
from pathlib import Path
from typing import Any
from ca.reports.trajectory import (
    _TRAJECTORY_LOW_COVERAGE_THRESHOLD,
    _format_optional_ratio,
    _trajectory_gate_for_item,
)
from ca.reports.common import _format_optional_float, _html_command_block, _html_summary_action


def make_trajectory_batch_summary(
    results: list[dict],
    reference_dir: str,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_rpe_rotation_deg: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> dict:
    """Build summary data for trajectory batch evaluation results."""
    low_coverage_threshold = (
        min_coverage if min_coverage is not None else _TRAJECTORY_LOW_COVERAGE_THRESHOLD
    )
    gate_enabled = (
        max_ate is not None
        or max_rpe is not None
        or max_rpe_rotation_deg is not None
        or max_drift is not None
        or min_coverage is not None
        or any(item.get("quality_gate") is not None for item in results)
    )
    quality_gate: dict[str, Any] | None = None

    if not results:
        if gate_enabled:
            quality_gate = {
                "max_ate": max_ate,
                "max_rpe": max_rpe,
                "max_rpe_rotation_deg": max_rpe_rotation_deg,
                "max_drift": max_drift,
                "min_coverage": min_coverage,
                "pass_count": 0,
                "fail_count": 0,
                "failed_paths": [],
            }
        return {
            "reference_dir": reference_dir,
            "total_files": 0,
            "mean_ate_rmse": 0.0,
            "mean_rpe_rmse": 0.0,
            "mean_rpe_rotation_rmse_deg": None,
            "rpe_rotation_evaluable_count": 0,
            "mean_coverage_ratio": 0.0,
            "low_coverage_threshold": low_coverage_threshold,
            "low_coverage_count": 0,
            "best_ate": None,
            "worst_ate": None,
            "best_rpe": None,
            "worst_rpe": None,
            "quality_gate": quality_gate,
            "results": [],
        }

    if gate_enabled:
        item_gates = [
            (
                item,
                _trajectory_gate_for_item(
                    item,
                    max_ate=max_ate,
                    max_rpe=max_rpe,
                    max_rpe_rotation_deg=max_rpe_rotation_deg,
                    max_drift=max_drift,
                    min_coverage=min_coverage,
                ),
            )
            for item in results
        ]
        quality_gate = {
            "max_ate": max_ate,
            "max_rpe": max_rpe,
            "max_rpe_rotation_deg": max_rpe_rotation_deg,
            "max_drift": max_drift,
            "min_coverage": min_coverage,
            "pass_count": sum(
                1
                for _, item_gate in item_gates
                if item_gate is not None and item_gate["passed"]
            ),
            "fail_count": sum(
                1
                for _, item_gate in item_gates
                if item_gate is not None and not item_gate["passed"]
            ),
            "failed_paths": [
                item["path"]
                for item, item_gate in item_gates
                if item_gate is not None and not item_gate["passed"]
            ],
        }

    rotation_rmses = [
        item["rpe_rotation"]["rmse"]
        for item in results
        if item.get("rpe_rotation") is not None
    ]
    return {
        "reference_dir": reference_dir,
        "total_files": len(results),
        "mean_ate_rmse": float(sum(item["ate"]["rmse"] for item in results) / len(results)),
        "mean_rpe_rmse": float(
            sum(item["rpe_translation"]["rmse"] for item in results) / len(results)
        ),
        "mean_rpe_rotation_rmse_deg": (
            float(sum(rotation_rmses) / len(rotation_rmses)) if rotation_rmses else None
        ),
        "rpe_rotation_evaluable_count": len(rotation_rmses),
        "mean_coverage_ratio": float(
            sum(item["coverage_ratio"] for item in results) / len(results)
        ),
        "low_coverage_threshold": low_coverage_threshold,
        "low_coverage_count": sum(
            1
            for item in results
            if item["coverage_ratio"] < low_coverage_threshold
        ),
        "best_ate": min(results, key=lambda item: item["ate"]["rmse"]),
        "worst_ate": max(results, key=lambda item: item["ate"]["rmse"]),
        "best_rpe": min(results, key=lambda item: item["rpe_translation"]["rmse"]),
        "worst_rpe": max(results, key=lambda item: item["rpe_translation"]["rmse"]),
        "quality_gate": quality_gate,
        "results": results,
    }


def make_trajectory_batch_markdown(
    results: list[dict],
    reference_dir: str,
    output_path: str,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_rpe_rotation_deg: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> None:
    """Generate a Markdown report for trajectory batch evaluation results."""
    summary = make_trajectory_batch_summary(
        results,
        reference_dir,
        max_ate=max_ate,
        max_rpe=max_rpe,
        max_rpe_rotation_deg=max_rpe_rotation_deg,
        max_drift=max_drift,
        min_coverage=min_coverage,
    )

    lines = [
        "# CloudAnalyzer Trajectory Batch Evaluation Report",
        "",
        "## Summary",
        f"- Reference Directory: {summary['reference_dir']}",
        f"- Files: {summary['total_files']}",
        f"- Mean ATE RMSE: {summary['mean_ate_rmse']:.4f}",
        f"- Mean RPE RMSE: {summary['mean_rpe_rmse']:.4f}",
        "- Mean RPE Rotation RMSE: "
        + (
            f"{summary['mean_rpe_rotation_rmse_deg']:.4f} deg"
            if summary["mean_rpe_rotation_rmse_deg"] is not None
            else "n/a (quaternion orientations unavailable)"
        ),
        f"- Mean Coverage: {summary['mean_coverage_ratio']:.1%}",
        (
            f"- Low Coverage (<{summary['low_coverage_threshold']:.0%}): "
            f"{summary['low_coverage_count']}"
        ),
    ]

    if summary["best_ate"] is not None:
        lines += [
            f"- Best ATE: {summary['best_ate']['path']} ({summary['best_ate']['ate']['rmse']:.4f})",
            f"- Worst ATE: {summary['worst_ate']['path']} ({summary['worst_ate']['ate']['rmse']:.4f})",
            f"- Best RPE: {summary['best_rpe']['path']} ({summary['best_rpe']['rpe_translation']['rmse']:.4f})",
            f"- Worst RPE: {summary['worst_rpe']['path']} ({summary['worst_rpe']['rpe_translation']['rmse']:.4f})",
        ]

    gate = summary["quality_gate"]
    if gate is not None:
        lines += [
            "",
            "## Quality Gate",
            f"- Max ATE RMSE: {_format_optional_float(gate['max_ate'])}",
            f"- Max RPE RMSE: {_format_optional_float(gate['max_rpe'])}",
            f"- Max RPE Rotation: {_format_optional_float(gate['max_rpe_rotation_deg'])} deg",
            f"- Max Drift: {_format_optional_float(gate['max_drift'])}",
            f"- Min Coverage: {_format_optional_ratio(gate['min_coverage'])}",
            f"- Pass: {gate['pass_count']}",
            f"- Fail: {gate['fail_count']}",
        ]

    lines += [
        "",
        "## Results",
        "",
        "| Path | Matched | Coverage | ATE RMSE | RPE RMSE | RPE Rot (deg) | Drift | Alignment |"
        + (" Status |" if gate is not None else ""),
        "|---|---:|---:|---:|---:|---:|---:|---|"
        + ("---|" if gate is not None else ""),
    ]

    for item in summary["results"]:
        item_gate = _trajectory_gate_for_item(
            item,
            max_ate=max_ate,
            max_rpe=max_rpe,
            max_rpe_rotation_deg=max_rpe_rotation_deg,
            max_drift=max_drift,
            min_coverage=min_coverage,
        )
        row = [
            item["path"],
            str(item["matched_poses"]),
            f"{item['coverage_ratio']:.1%}",
            f"{item['ate']['rmse']:.4f}",
            f"{item['rpe_translation']['rmse']:.4f}",
            _format_optional_float(
                None if item.get("rpe_rotation") is None else item["rpe_rotation"]["rmse"]
            ),
            f"{item['drift']['endpoint']:.4f}",
            item["alignment"]["mode"],
        ]
        if gate is not None:
            status = "PASS" if item_gate is not None and item_gate["passed"] else "FAIL"
            row.append(status)
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Inspection Commands",
        "",
    ]
    for item in summary["results"]:
        inspect = item.get("inspect", {})
        if inspect.get("traj_evaluate"):
            lines.append(f"- {item['path']}: `{inspect['traj_evaluate']}`")
        if inspect.get("traj_evaluate_aligned"):
            lines.append(f"  - Aligned: `{inspect['traj_evaluate_aligned']}`")
        if inspect.get("traj_evaluate_rigid"):
            lines.append(f"  - Rigid: `{inspect['traj_evaluate_rigid']}`")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def make_trajectory_batch_html(
    results: list[dict],
    reference_dir: str,
    output_path: str,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_rpe_rotation_deg: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> None:
    """Generate an HTML report for trajectory batch evaluation results."""
    summary = make_trajectory_batch_summary(
        results,
        reference_dir,
        max_ate=max_ate,
        max_rpe=max_rpe,
        max_rpe_rotation_deg=max_rpe_rotation_deg,
        max_drift=max_drift,
        min_coverage=min_coverage,
    )
    gate = summary["quality_gate"]
    failed_count = gate["fail_count"] if gate is not None else 0
    pass_count = gate["pass_count"] if gate is not None else 0
    low_coverage_count = summary["low_coverage_count"]
    low_coverage_label = f"Low Coverage (<{summary['low_coverage_threshold']:.0%})"

    summary_rows = [
        ("Reference Directory", summary["reference_dir"]),
        ("Files", str(summary["total_files"])),
        ("Mean ATE RMSE", f"{summary['mean_ate_rmse']:.4f}"),
        ("Mean RPE RMSE", f"{summary['mean_rpe_rmse']:.4f}"),
        (
            "Mean RPE Rotation RMSE",
            (
                f"{summary['mean_rpe_rotation_rmse_deg']:.4f} deg"
                if summary["mean_rpe_rotation_rmse_deg"] is not None
                else "n/a (quaternion orientations unavailable)"
            ),
        ),
        ("Mean Coverage", f"{summary['mean_coverage_ratio']:.1%}"),
        (low_coverage_label, str(low_coverage_count)),
    ]
    if summary["best_ate"] is not None:
        summary_rows += [
            ("Best ATE", f"{summary['best_ate']['path']} ({summary['best_ate']['ate']['rmse']:.4f})"),
            ("Worst ATE", f"{summary['worst_ate']['path']} ({summary['worst_ate']['ate']['rmse']:.4f})"),
            ("Best RPE", f"{summary['best_rpe']['path']} ({summary['best_rpe']['rpe_translation']['rmse']:.4f})"),
            ("Worst RPE", f"{summary['worst_rpe']['path']} ({summary['worst_rpe']['rpe_translation']['rmse']:.4f})"),
        ]
    if gate is not None:
        summary_rows += [
            ("Max ATE RMSE", _format_optional_float(gate["max_ate"])),
            ("Max RPE RMSE", _format_optional_float(gate["max_rpe"])),
            (
                "Max RPE Rotation",
                f"{_format_optional_float(gate['max_rpe_rotation_deg'])} deg",
            ),
            ("Max Drift", _format_optional_float(gate["max_drift"])),
            ("Min Coverage", _format_optional_ratio(gate["min_coverage"])),
            ("Pass", str(gate["pass_count"])),
            ("Fail", str(gate["fail_count"])),
        ]

    summary_row_actions = {
        "Pass": _html_summary_action(
            "trajectory-summary-show-pass",
            "pass",
            "Show pass",
            pass_count,
            disabled=gate is None or pass_count == 0,
            onclick_function="applyTrajectoryQuickAction",
        ),
        "Fail": _html_summary_action(
            "trajectory-summary-show-failed",
            "failed",
            "Show failed",
            failed_count,
            disabled=gate is None or failed_count == 0,
            onclick_function="applyTrajectoryQuickAction",
        ),
        low_coverage_label: _html_summary_action(
            "trajectory-summary-show-low-coverage",
            "low-coverage",
            "Show low coverage",
            low_coverage_count,
            disabled=low_coverage_count == 0,
            onclick_function="applyTrajectoryQuickAction",
        ),
    }

    summary_html = "\n".join(
        (
            f"<tr><th>{escape(label)}</th><td>"
            f"<span>{escape(value)}</span>"
            f"{summary_row_actions.get(label, '')}"
            "</td></tr>"
        )
        for label, value in summary_rows
    )
    row_parts = []
    inspection_row_parts = []
    for item in summary["results"]:
        low_coverage = item["coverage_ratio"] < summary["low_coverage_threshold"]
        rotation_rmse = (
            None if item.get("rpe_rotation") is None else item["rpe_rotation"]["rmse"]
        )
        rotation_data = "" if rotation_rmse is None else f"{rotation_rmse:.8f}"
        item_gate = _trajectory_gate_for_item(
            item,
            max_ate=max_ate,
            max_rpe=max_rpe,
            max_rpe_rotation_deg=max_rpe_rotation_deg,
            max_drift=max_drift,
            min_coverage=min_coverage,
        )
        status = None
        row_classes = []
        row_attrs = [
            f'data-path="{escape(item["path"])}"',
            f'data-matched="{item["matched_poses"]}"',
            f'data-coverage="{item["coverage_ratio"]:.8f}"',
            f'data-ate="{item["ate"]["rmse"]:.8f}"',
            f'data-rpe="{item["rpe_translation"]["rmse"]:.8f}"',
            f'data-rpe-rotation="{rotation_data}"',
            f'data-drift="{item["drift"]["endpoint"]:.8f}"',
            f'data-low-coverage="{"true" if low_coverage else "false"}"',
        ]
        if gate is not None and item_gate is not None:
            status = "PASS" if item_gate["passed"] else "FAIL"
            row_attrs.append(f'data-passed="{"true" if status == "PASS" else "false"}"')
            row_attrs.append(f'data-failed="{"true" if status == "FAIL" else "false"}"')
            if status == "FAIL":
                row_classes.append("fail-row")
        else:
            row_attrs.append('data-passed="false"')
            row_attrs.append('data-failed="false"')
        if low_coverage:
            row_classes.append("low-coverage-row")

        class_attr = f' class="{" ".join(row_classes)}"' if row_classes else ""
        row_parts.append(
            f"<tr{class_attr} {' '.join(row_attrs)}>"
            f"<td>{escape(item['path'])}</td>"
            f"<td>{item['matched_poses']}</td>"
            f"<td>{item['coverage_ratio']:.1%}</td>"
            f"<td>{item['ate']['rmse']:.4f}</td>"
            f"<td>{item['rpe_translation']['rmse']:.4f}</td>"
            f"<td>{_format_optional_float(rotation_rmse)}</td>"
            f"<td>{item['drift']['endpoint']:.4f}</td>"
            f"<td>{escape(item['alignment']['mode'])}</td>"
            + (f"<td>{status}</td>" if gate is not None else "")
            + "</tr>"
        )
        inspection_row_parts.append(
            "<tr "
            + " ".join(row_attrs)
            + ">"
            f"<td>{escape(item['path'])}</td>"
            "<td>"
            f"{_html_command_block(item.get('inspect', {}).get('traj_evaluate', ''))}"
            f"{_html_command_block(item.get('inspect', {}).get('traj_evaluate_aligned', ''))}"
            f"{_html_command_block(item.get('inspect', {}).get('traj_evaluate_rigid', ''))}"
            "</td>"
            "</tr>"
        )
    results_html = "\n".join(row_parts)
    inspection_html = "\n".join(inspection_row_parts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CloudAnalyzer Trajectory Batch Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #111827; }}
    h1, h2 {{ margin-bottom: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f3f4f6; }}
    tbody tr:nth-child(even) {{ background: #f9fafb; }}
    code {{ white-space: pre-wrap; word-break: break-word; }}
    .summary-table td {{ display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }}
    .summary-chip {{
      margin-left: 0; border-radius: 999px; padding: 0.25rem 0.4rem 0.25rem 0.65rem;
      font-size: 0.82rem; font-weight: 600; display: inline-flex; align-items: center; gap: 0.45rem;
    }}
    .summary-chip-count {{
      display: inline-flex; align-items: center; justify-content: center;
      min-width: 1.45rem; height: 1.45rem; padding: 0 0.35rem; border-radius: 999px;
      background: rgba(255, 255, 255, 0.78); font-size: 0.76rem; line-height: 1;
    }}
    .summary-chip-pass {{ border-color: #93c5fd; color: #1d4ed8; background: #eff6ff; }}
    .summary-chip-failed {{ border-color: #fca5a5; color: #991b1b; background: #fef2f2; }}
    .summary-chip-low-coverage {{ border-color: #fcd34d; color: #92400e; background: #fffbeb; }}
    .summary-chip-active {{
      box-shadow: 0 0 0 2px rgba(17, 24, 39, 0.12) inset;
      transform: translateY(-1px);
    }}
    .quick-action-chip-active {{
      box-shadow: 0 0 0 2px rgba(17, 24, 39, 0.12) inset;
      transform: translateY(-1px);
    }}
    .quick-actions {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 0 0 1.5rem 0; }}
    .filter-bar {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 0 0 1rem 0; }}
    .filter-bar label {{ display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.95rem; }}
    .filter-control {{
      display: inline-flex; align-items: center; gap: 0.45rem;
      padding: 0.3rem 0.65rem; border: 1px solid #d1d5db; border-radius: 999px;
      background: #ffffff; transition: border-color 120ms ease, background 120ms ease, color 120ms ease;
    }}
    .filter-control-active {{
      border-color: #93c5fd; background: #eff6ff; color: #1d4ed8;
    }}
    .filter-control-active select {{
      border-color: #93c5fd; background: #dbeafe; color: #1d4ed8;
    }}
    .filter-control-active input {{
      accent-color: #2563eb;
    }}
    .filter-control-disabled {{
      opacity: 0.6;
    }}
    .filter-bar select {{
      padding: 0.25rem 0.4rem; border: 1px solid #cbd5e1; border-radius: 6px; background: #ffffff;
    }}
    .filter-actions {{ display: inline-flex; align-items: center; gap: 0.75rem; }}
    .filter-summary {{ color: #4b5563; font-size: 0.9rem; }}
    .command-row {{ display: flex; align-items: flex-start; gap: 0.5rem; margin: 0.4rem 0; }}
    .command-row code {{ flex: 1; }}
    button {{
      margin-left: 0.75rem; padding: 0.3rem 0.55rem; border: 1px solid #cbd5e1;
      border-radius: 6px; background: #ffffff; cursor: pointer;
    }}
    .fail-row td {{ background: #fef2f2; }}
    .low-coverage-row td:first-child {{ box-shadow: inset 4px 0 0 #f59e0b; }}
  </style>
</head>
<body>
  <h1>CloudAnalyzer Trajectory Batch Evaluation Report</h1>
  <h2>Summary</h2>
  <table class="summary-table">
    <tbody>
      {summary_html}
    </tbody>
  </table>
  <div class="quick-actions">
    <button type="button" id="trajectory-quick-show-pass" data-quick-action="pass" {"disabled" if gate is None or pass_count == 0 else ""} onclick="applyTrajectoryQuickAction('pass')">
      Show pass ({pass_count})
    </button>
    <button type="button" id="trajectory-quick-show-failed" data-quick-action="failed" {"disabled" if gate is None or failed_count == 0 else ""} onclick="applyTrajectoryQuickAction('failed')">
      Show failed ({failed_count})
    </button>
    <button type="button" id="trajectory-quick-show-low-coverage" data-quick-action="low-coverage" {"disabled" if low_coverage_count == 0 else ""} onclick="applyTrajectoryQuickAction('low-coverage')">
      Show low coverage ({low_coverage_count})
    </button>
    <button type="button" id="trajectory-quick-reset-view" data-quick-action="reset" onclick="applyTrajectoryQuickAction('reset')">
      Reset view
    </button>
  </div>

  <h2>Results</h2>
  <div class="filter-bar">
    <label class="filter-control" id="trajectory-sort-results-control">
      Sort by
      <select id="trajectory-sort-results" onchange="refreshTrajectoryResultsView()">
        <option value="path-asc">Path (A-Z)</option>
        <option value="ate-desc">ATE RMSE (high-low)</option>
        <option value="ate-asc">ATE RMSE (low-high)</option>
        <option value="rpe-desc">RPE RMSE (high-low)</option>
        <option value="rpe-asc">RPE RMSE (low-high)</option>
        <option value="coverage-desc">Coverage (high-low)</option>
        <option value="coverage-asc">Coverage (low-high)</option>
        <option value="drift-desc">Drift (high-low)</option>
      </select>
    </label>
    <label class="filter-control{' filter-control-disabled' if gate is None else ''}" id="trajectory-filter-pass-only-control">
      <input type="checkbox" id="trajectory-filter-pass-only" {"disabled" if gate is None else ""} onchange="refreshTrajectoryResultsView()">
      Pass only
    </label>
    <label class="filter-control{' filter-control-disabled' if gate is None else ''}" id="trajectory-filter-failed-only-control">
      <input type="checkbox" id="trajectory-filter-failed-only" {"disabled" if gate is None else ""} onchange="refreshTrajectoryResultsView()">
      Failed only
    </label>
    <label class="filter-control{' filter-control-disabled' if low_coverage_count == 0 else ''}" id="trajectory-filter-low-coverage-only-control">
      <input type="checkbox" id="trajectory-filter-low-coverage-only" {"disabled" if low_coverage_count == 0 else ""} onchange="refreshTrajectoryResultsView()">
      Low coverage only
    </label>
    <span class="filter-actions">
      <button type="button" id="trajectory-reset-filters" onclick="resetTrajectoryFilters()">Reset filters</button>
    </span>
    <span class="filter-summary" id="trajectory-filter-summary"></span>
  </div>
  <table>
    <thead>
      <tr>
        <th>Path</th>
        <th>Matched</th>
        <th>Coverage</th>
        <th>ATE RMSE</th>
        <th>RPE RMSE</th>
        <th>RPE Rot (deg)</th>
        <th>Drift</th>
        <th>Alignment</th>
        {"<th>Status</th>" if gate is not None else ""}
      </tr>
    </thead>
    <tbody id="trajectory-results-table-body">
      {results_html}
    </tbody>
  </table>

  <h2>Inspection Commands</h2>
  <table>
    <thead>
      <tr>
        <th>Path</th>
        <th>Inspection</th>
      </tr>
    </thead>
    <tbody id="trajectory-inspection-table-body">
      {inspection_html}
    </tbody>
  </table>
  <script>
    function compareTrajectoryRows(a, b, sortValue) {{
      const separatorIndex = sortValue.lastIndexOf('-');
      const sortKey = separatorIndex >= 0 ? sortValue.slice(0, separatorIndex) : sortValue;
      const sortDirection = separatorIndex >= 0 ? sortValue.slice(separatorIndex + 1) : 'asc';

      function compareValues(left, right) {{
        if (left < right) return sortDirection === 'desc' ? 1 : -1;
        if (left > right) return sortDirection === 'desc' ? -1 : 1;
        return 0;
      }}

      if (sortKey === 'path') {{
        return compareValues(a.dataset.path || '', b.dataset.path || '');
      }}
      if (sortKey === 'ate') {{
        return compareValues(Number(a.dataset.ate || 0), Number(b.dataset.ate || 0));
      }}
      if (sortKey === 'rpe') {{
        return compareValues(Number(a.dataset.rpe || 0), Number(b.dataset.rpe || 0));
      }}
      if (sortKey === 'coverage') {{
        return compareValues(Number(a.dataset.coverage || 0), Number(b.dataset.coverage || 0));
      }}
      if (sortKey === 'drift') {{
        return compareValues(Number(a.dataset.drift || 0), Number(b.dataset.drift || 0));
      }}
      return compareValues(a.dataset.path || '', b.dataset.path || '');
    }}

    function setTrajectoryCheckedIfEnabled(id, checked) {{
      const input = document.getElementById(id);
      if (input && !input.disabled) {{
        input.checked = checked;
      }}
    }}

    function setTrajectorySortValue(value) {{
      const sortResults = document.getElementById('trajectory-sort-results');
      if (sortResults) {{
        const option = Array.from(sortResults.options).find((item) => item.value === value && !item.disabled);
        if (option) {{
          sortResults.value = value;
        }}
      }}
    }}

    function updateTrajectoryActionStates(action) {{
      const summaryButtons = Array.from(document.querySelectorAll('[data-summary-action]'));
      const quickButtons = Array.from(document.querySelectorAll('[data-quick-action]'));

      summaryButtons.forEach((button) => {{
        button.classList.toggle('summary-chip-active', button.dataset.summaryAction === action);
      }});
      quickButtons.forEach((button) => {{
        button.classList.toggle('quick-action-chip-active', button.dataset.quickAction === action);
      }});
    }}

    function updateTrajectoryFilterControlStates(passEnabled, failedEnabled, lowCoverageEnabled, sortValue) {{
      const controlStates = [
        ['trajectory-filter-pass-only-control', passEnabled],
        ['trajectory-filter-failed-only-control', failedEnabled],
        ['trajectory-filter-low-coverage-only-control', lowCoverageEnabled],
        ['trajectory-sort-results-control', sortValue !== 'path-asc'],
      ];

      controlStates.forEach(([id, enabled]) => {{
        const control = document.getElementById(id);
        if (control) {{
          control.classList.toggle('filter-control-active', enabled);
        }}
      }});
    }}

    function refreshTrajectoryResultsView() {{
      const passOnly = document.getElementById('trajectory-filter-pass-only');
      const failedOnly = document.getElementById('trajectory-filter-failed-only');
      const lowCoverageOnly = document.getElementById('trajectory-filter-low-coverage-only');
      const sortResults = document.getElementById('trajectory-sort-results');
      const passEnabled = passOnly && !passOnly.disabled && passOnly.checked;
      const failedEnabled = failedOnly && !failedOnly.disabled && failedOnly.checked;
      const lowCoverageEnabled = lowCoverageOnly && !lowCoverageOnly.disabled && lowCoverageOnly.checked;
      const sortValue = sortResults ? sortResults.value : 'path-asc';

      const resultsTableBody = document.getElementById('trajectory-results-table-body');
      const inspectionTableBody = document.getElementById('trajectory-inspection-table-body');
      const resultRows = Array.from(document.querySelectorAll('#trajectory-results-table-body tr'));
      const inspectionRows = Array.from(document.querySelectorAll('#trajectory-inspection-table-body tr'));
      let visibleCount = 0;

      function rowMatches(row) {{
        const isPassed = row.dataset.passed === 'true';
        const isFailed = row.dataset.failed === 'true';
        const isLowCoverage = row.dataset.lowCoverage === 'true';
        if (passEnabled && !isPassed) return false;
        if (failedEnabled && !isFailed) return false;
        if (lowCoverageEnabled && !isLowCoverage) return false;
        return true;
      }}

      resultRows.sort((a, b) => compareTrajectoryRows(a, b, sortValue));
      if (resultsTableBody) {{
        resultRows.forEach((row) => {{
          resultsTableBody.appendChild(row);
        }});
      }}

      const resultOrder = new Map(resultRows.map((row, index) => [row.dataset.path || '', index]));
      inspectionRows.sort((a, b) => {{
        const aOrder = resultOrder.get(a.dataset.path || '') ?? Number.MAX_SAFE_INTEGER;
        const bOrder = resultOrder.get(b.dataset.path || '') ?? Number.MAX_SAFE_INTEGER;
        return aOrder - bOrder;
      }});
      if (inspectionTableBody) {{
        inspectionRows.forEach((row) => {{
          inspectionTableBody.appendChild(row);
        }});
      }}

      resultRows.forEach((row) => {{
        const visible = rowMatches(row);
        row.style.display = visible ? '' : 'none';
        if (visible) visibleCount += 1;
      }});
      inspectionRows.forEach((row) => {{
        row.style.display = rowMatches(row) ? '' : 'none';
      }});

      const summary = document.getElementById('trajectory-filter-summary');
      if (summary) {{
        const activeFilters = [];
        if (passEnabled) activeFilters.push('pass');
        if (failedEnabled) activeFilters.push('failed');
        if (lowCoverageEnabled) activeFilters.push('low-coverage');
        summary.textContent = `Showing ${{visibleCount}} / ${{resultRows.length}} result(s)` + (
          activeFilters.length ? ` | Filters: ${{activeFilters.join(', ')}}` : ''
        ) + (
          sortValue ? ` | Sort: ${{sortValue}}` : ''
        );
      }}

      let activeAction = 'reset';
      if (passEnabled && !failedEnabled && !lowCoverageEnabled && sortValue === 'ate-asc') {{
        activeAction = 'pass';
      }} else if (!passEnabled && failedEnabled && !lowCoverageEnabled && sortValue === 'ate-desc') {{
        activeAction = 'failed';
      }} else if (!passEnabled && !failedEnabled && lowCoverageEnabled && sortValue === 'coverage-asc') {{
        activeAction = 'low-coverage';
      }}
      updateTrajectoryActionStates(activeAction);
      updateTrajectoryFilterControlStates(passEnabled, failedEnabled, lowCoverageEnabled, sortValue);
    }}

    function applyTrajectoryQuickAction(action) {{
      if (action === 'pass') {{
        setTrajectoryCheckedIfEnabled('trajectory-filter-pass-only', true);
        setTrajectoryCheckedIfEnabled('trajectory-filter-failed-only', false);
        setTrajectoryCheckedIfEnabled('trajectory-filter-low-coverage-only', false);
        setTrajectorySortValue('ate-asc');
      }} else if (action === 'failed') {{
        setTrajectoryCheckedIfEnabled('trajectory-filter-pass-only', false);
        setTrajectoryCheckedIfEnabled('trajectory-filter-failed-only', true);
        setTrajectoryCheckedIfEnabled('trajectory-filter-low-coverage-only', false);
        setTrajectorySortValue('ate-desc');
      }} else if (action === 'low-coverage') {{
        setTrajectoryCheckedIfEnabled('trajectory-filter-pass-only', false);
        setTrajectoryCheckedIfEnabled('trajectory-filter-failed-only', false);
        setTrajectoryCheckedIfEnabled('trajectory-filter-low-coverage-only', true);
        setTrajectorySortValue('coverage-asc');
      }} else {{
        resetTrajectoryFilters();
        return;
      }}
      refreshTrajectoryResultsView();
    }}

    function resetTrajectoryFilters() {{
      const filterIds = [
        'trajectory-filter-pass-only',
        'trajectory-filter-failed-only',
        'trajectory-filter-low-coverage-only',
      ];
      filterIds.forEach((id) => {{
        const input = document.getElementById(id);
        if (input && !input.disabled) {{
          input.checked = false;
        }}
      }});
      const sortResults = document.getElementById('trajectory-sort-results');
      if (sortResults) {{
        sortResults.value = 'path-asc';
      }}
      refreshTrajectoryResultsView();
    }}

    async function copyCommand(text, button) {{
      if (!text) return;
      try {{
        if (navigator.clipboard) {{
          await navigator.clipboard.writeText(text);
        }} else {{
          window.prompt('Copy command:', text);
        }}
        const original = button.textContent;
        button.textContent = 'Copied';
        setTimeout(() => {{ button.textContent = original; }}, 1200);
      }} catch (err) {{
        window.prompt('Copy command:', text);
      }}
    }}
    refreshTrajectoryResultsView();
  </script>
</body>
</html>
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def save_trajectory_batch_report(
    results: list[dict],
    reference_dir: str,
    output_path: str,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_rpe_rotation_deg: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> None:
    """Write a trajectory batch evaluation report based on the file extension."""
    ext = Path(output_path).suffix.lower()
    if ext in {".md", ".markdown"}:
        make_trajectory_batch_markdown(
            results,
            reference_dir,
            output_path,
            max_ate=max_ate,
            max_rpe=max_rpe,
            max_rpe_rotation_deg=max_rpe_rotation_deg,
            max_drift=max_drift,
            min_coverage=min_coverage,
        )
        return
    if ext == ".html":
        make_trajectory_batch_html(
            results,
            reference_dir,
            output_path,
            max_ate=max_ate,
            max_rpe=max_rpe,
            max_rpe_rotation_deg=max_rpe_rotation_deg,
            max_drift=max_drift,
            min_coverage=min_coverage,
        )
        return
    raise ValueError("Unsupported report format. Use .md, .markdown, or .html")
