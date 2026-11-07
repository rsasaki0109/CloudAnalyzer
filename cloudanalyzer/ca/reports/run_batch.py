"""Run batch reports."""

from html import escape
from pathlib import Path
from typing import Any, cast
from ca.reports.common import (
    _format_optional_float,
    _gate_for_item,
    _html_command_block,
    _html_summary_action,
)
from ca.reports.trajectory import _format_optional_ratio, _trajectory_gate_for_item


def _run_batch_gate_for_item(
    item: dict,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_rpe_rotation_deg: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> dict | None:
    """Return existing or derived overall quality gate metadata for one run-batch item."""
    if item.get("overall_quality_gate") is not None:
        return cast(dict[str, Any], item["overall_quality_gate"])
    if (
        min_auc is None
        and max_chamfer is None
        and max_ate is None
        and max_rpe is None
        and max_rpe_rotation_deg is None
        and max_drift is None
        and min_coverage is None
    ):
        return None

    map_item = {
        "auc": item["map"]["auc"],
        "chamfer_distance": item["map"]["chamfer_distance"],
        "quality_gate": item["map"].get("quality_gate"),
    }
    trajectory_item = {
        "ate": item["trajectory"]["ate"],
        "rpe_translation": item["trajectory"]["rpe_translation"],
        "rpe_rotation": item["trajectory"].get("rpe_rotation"),
        "drift": item["trajectory"]["drift"],
        "coverage_ratio": item["trajectory"]["matching"]["coverage_ratio"],
        "quality_gate": item["trajectory"].get("quality_gate"),
    }
    map_gate = _gate_for_item(map_item, min_auc=min_auc, max_chamfer=max_chamfer)
    trajectory_gate = _trajectory_gate_for_item(
        trajectory_item,
        max_ate=max_ate,
        max_rpe=max_rpe,
        max_rpe_rotation_deg=max_rpe_rotation_deg,
        max_drift=max_drift,
        min_coverage=min_coverage,
    )

    reasons: list[str] = []
    if map_gate is not None and not map_gate["passed"]:
        reasons.extend(f"Map: {reason}" for reason in map_gate["reasons"])
    if trajectory_gate is not None and not trajectory_gate["passed"]:
        reasons.extend(f"Trajectory: {reason}" for reason in trajectory_gate["reasons"])
    return {
        "passed": not reasons,
        "map_passed": None if map_gate is None else map_gate["passed"],
        "trajectory_passed": None if trajectory_gate is None else trajectory_gate["passed"],
        "reasons": reasons,
    }


def make_run_batch_summary(
    results: list[dict],
    map_reference_dir: str,
    trajectory_reference_dir: str,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_rpe_rotation_deg: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> dict:
    """Build summary data for combined run-batch evaluation results."""
    gate_enabled = (
        min_auc is not None
        or max_chamfer is not None
        or max_ate is not None
        or max_rpe is not None
        or max_rpe_rotation_deg is not None
        or max_drift is not None
        or min_coverage is not None
        or any(item.get("overall_quality_gate") is not None for item in results)
    )
    quality_gate: dict[str, Any] | None = None

    if not results:
        if gate_enabled:
            quality_gate = {
                "min_auc": min_auc,
                "max_chamfer": max_chamfer,
                "max_ate": max_ate,
                "max_rpe": max_rpe,
                "max_rpe_rotation_deg": max_rpe_rotation_deg,
                "max_drift": max_drift,
                "min_coverage": min_coverage,
                "map_fail_count": 0,
                "trajectory_fail_count": 0,
                "pass_count": 0,
                "fail_count": 0,
                "failed_ids": [],
            }
        return {
            "map_reference_dir": map_reference_dir,
            "trajectory_reference_dir": trajectory_reference_dir,
            "total_runs": 0,
            "mean_map_auc": 0.0,
            "mean_map_chamfer": 0.0,
            "mean_traj_ate_rmse": 0.0,
            "mean_traj_rpe_rmse": 0.0,
            "mean_traj_rpe_rotation_rmse_deg": None,
            "mean_traj_drift": 0.0,
            "mean_traj_coverage": 0.0,
            "best_map_auc": None,
            "worst_traj_ate": None,
            "quality_gate": quality_gate,
            "results": [],
        }

    if gate_enabled:
        item_gates = [
            (
                item,
                _run_batch_gate_for_item(
                    item,
                    min_auc=min_auc,
                    max_chamfer=max_chamfer,
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
            "min_auc": min_auc,
            "max_chamfer": max_chamfer,
            "max_ate": max_ate,
            "max_rpe": max_rpe,
            "max_rpe_rotation_deg": max_rpe_rotation_deg,
            "max_drift": max_drift,
            "min_coverage": min_coverage,
            "map_fail_count": sum(
                1 for _, gate in item_gates if gate is not None and gate["map_passed"] is False
            ),
            "trajectory_fail_count": sum(
                1 for _, gate in item_gates if gate is not None and gate["trajectory_passed"] is False
            ),
            "pass_count": sum(1 for _, gate in item_gates if gate is not None and gate["passed"]),
            "fail_count": sum(1 for _, gate in item_gates if gate is not None and not gate["passed"]),
            "failed_ids": [
                item["id"] for item, gate in item_gates if gate is not None and not gate["passed"]
            ],
        }

    rotation_rmses = [
        item["trajectory"]["rpe_rotation"]["rmse"]
        for item in results
        if item["trajectory"].get("rpe_rotation") is not None
    ]
    return {
        "map_reference_dir": map_reference_dir,
        "trajectory_reference_dir": trajectory_reference_dir,
        "total_runs": len(results),
        "mean_map_auc": float(sum(item["map"]["auc"] for item in results) / len(results)),
        "mean_map_chamfer": float(
            sum(item["map"]["chamfer_distance"] for item in results) / len(results)
        ),
        "mean_traj_ate_rmse": float(
            sum(item["trajectory"]["ate"]["rmse"] for item in results) / len(results)
        ),
        "mean_traj_rpe_rmse": float(
            sum(item["trajectory"]["rpe_translation"]["rmse"] for item in results) / len(results)
        ),
        "mean_traj_rpe_rotation_rmse_deg": (
            float(sum(rotation_rmses) / len(rotation_rmses)) if rotation_rmses else None
        ),
        "mean_traj_drift": float(
            sum(item["trajectory"]["drift"]["endpoint"] for item in results) / len(results)
        ),
        "mean_traj_coverage": float(
            sum(item["trajectory"]["matching"]["coverage_ratio"] for item in results) / len(results)
        ),
        "best_map_auc": max(results, key=lambda item: item["map"]["auc"]),
        "worst_traj_ate": max(results, key=lambda item: item["trajectory"]["ate"]["rmse"]),
        "quality_gate": quality_gate,
        "results": results,
    }


def make_run_batch_markdown(
    results: list[dict],
    map_reference_dir: str,
    trajectory_reference_dir: str,
    output_path: str,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_rpe_rotation_deg: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> None:
    """Generate a Markdown report for combined run-batch evaluation results."""
    summary = make_run_batch_summary(
        results,
        map_reference_dir,
        trajectory_reference_dir,
        min_auc=min_auc,
        max_chamfer=max_chamfer,
        max_ate=max_ate,
        max_rpe=max_rpe,
        max_rpe_rotation_deg=max_rpe_rotation_deg,
        max_drift=max_drift,
        min_coverage=min_coverage,
    )
    gate = summary["quality_gate"]

    lines = [
        "# CloudAnalyzer Run Batch Evaluation Report",
        "",
        "## Summary",
        f"- Map Reference Directory: {summary['map_reference_dir']}",
        f"- Trajectory Reference Directory: {summary['trajectory_reference_dir']}",
        f"- Runs: {summary['total_runs']}",
        f"- Mean Map AUC: {summary['mean_map_auc']:.4f}",
        f"- Mean Map Chamfer: {summary['mean_map_chamfer']:.4f}",
        f"- Mean Trajectory ATE RMSE: {summary['mean_traj_ate_rmse']:.4f}",
        f"- Mean Trajectory RPE RMSE: {summary['mean_traj_rpe_rmse']:.4f}",
        "- Mean Trajectory RPE Rotation RMSE: "
        + (
            f"{summary['mean_traj_rpe_rotation_rmse_deg']:.4f} deg"
            if summary["mean_traj_rpe_rotation_rmse_deg"] is not None
            else "n/a (quaternion orientations unavailable)"
        ),
        f"- Mean Trajectory Drift: {summary['mean_traj_drift']:.4f}",
        f"- Mean Trajectory Coverage: {summary['mean_traj_coverage']:.1%}",
    ]
    if summary["best_map_auc"] is not None:
        lines += [
            f"- Best Map AUC: {summary['best_map_auc']['id']} ({summary['best_map_auc']['map']['auc']:.4f})",
            (
                f"- Worst Trajectory ATE: {summary['worst_traj_ate']['id']} "
                f"({summary['worst_traj_ate']['trajectory']['ate']['rmse']:.4f})"
            ),
        ]
    if gate is not None:
        lines += [
            "",
            "## Quality Gate",
            f"- Min AUC: {_format_optional_float(gate['min_auc'])}",
            f"- Max Chamfer: {_format_optional_float(gate['max_chamfer'])}",
            f"- Max ATE RMSE: {_format_optional_float(gate['max_ate'])}",
            f"- Max RPE RMSE: {_format_optional_float(gate['max_rpe'])}",
            f"- Max RPE Rotation: {_format_optional_float(gate['max_rpe_rotation_deg'])} deg",
            f"- Max Drift: {_format_optional_float(gate['max_drift'])}",
            f"- Min Coverage: {_format_optional_ratio(gate['min_coverage'])}",
            f"- Map Failures: {gate['map_fail_count']}",
            f"- Trajectory Failures: {gate['trajectory_fail_count']}",
            f"- Pass: {gate['pass_count']}",
            f"- Fail: {gate['fail_count']}",
        ]

    lines += [
        "",
        "## Results",
        "",
        "| ID | Map AUC | Map Chamfer | Traj ATE | Traj RPE | Traj RPE Rot (deg) | Traj Drift | Coverage |"
        + (" Map Status | Trajectory Status | Overall |" if gate is not None else ""),
        "|---|---:|---:|---:|---:|---:|---:|---:|"
        + ("---|---|---|" if gate is not None else ""),
    ]
    for item in summary["results"]:
        item_gate = _run_batch_gate_for_item(
            item,
            min_auc=min_auc,
            max_chamfer=max_chamfer,
            max_ate=max_ate,
            max_rpe=max_rpe,
            max_rpe_rotation_deg=max_rpe_rotation_deg,
            max_drift=max_drift,
            min_coverage=min_coverage,
        )
        row = [
            item["id"],
            f"{item['map']['auc']:.4f}",
            f"{item['map']['chamfer_distance']:.4f}",
            f"{item['trajectory']['ate']['rmse']:.4f}",
            f"{item['trajectory']['rpe_translation']['rmse']:.4f}",
            _format_optional_float(
                None
                if item["trajectory"].get("rpe_rotation") is None
                else item["trajectory"]["rpe_rotation"]["rmse"]
            ),
            f"{item['trajectory']['drift']['endpoint']:.4f}",
            f"{item['trajectory']['matching']['coverage_ratio']:.1%}",
        ]
        if gate is not None:
            row.append("PASS" if item_gate is not None and item_gate["map_passed"] is not False else "FAIL")
            row.append(
                "PASS" if item_gate is not None and item_gate["trajectory_passed"] is not False else "FAIL"
            )
            row.append("PASS" if item_gate is not None and item_gate["passed"] else "FAIL")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Inspection Commands",
        "",
    ]
    for item in summary["results"]:
        lines.append(f"- {item['id']}:")
        lines.append(f"  - Run viewer: `{item['inspect']['run_web']}`")
        lines.append(f"  - Combined: `{item['inspect']['run_evaluate']}`")
        lines.append(f"  - Map heatmap: `{item['inspect']['map']['web_heatmap']}`")
        lines.append(f"  - Map snapshot: `{item['inspect']['map']['heatmap3d']}`")
        lines.append(f"  - Trajectory: `{item['inspect']['trajectory']['traj_evaluate']}`")
        lines.append(
            f"  - Trajectory aligned: `{item['inspect']['trajectory']['traj_evaluate_aligned']}`"
        )
        lines.append(
            f"  - Trajectory rigid: `{item['inspect']['trajectory']['traj_evaluate_rigid']}`"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def make_run_batch_html(
    results: list[dict],
    map_reference_dir: str,
    trajectory_reference_dir: str,
    output_path: str,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_rpe_rotation_deg: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> None:
    """Generate an HTML report for combined run-batch evaluation results."""
    summary = make_run_batch_summary(
        results,
        map_reference_dir,
        trajectory_reference_dir,
        min_auc=min_auc,
        max_chamfer=max_chamfer,
        max_ate=max_ate,
        max_rpe=max_rpe,
        max_rpe_rotation_deg=max_rpe_rotation_deg,
        max_drift=max_drift,
        min_coverage=min_coverage,
    )
    gate = summary["quality_gate"]
    pass_count = gate["pass_count"] if gate is not None else 0
    failed_count = gate["fail_count"] if gate is not None else 0
    map_failed_count = gate["map_fail_count"] if gate is not None else 0
    trajectory_failed_count = gate["trajectory_fail_count"] if gate is not None else 0
    summary_rows = [
        ("Map Reference Directory", summary["map_reference_dir"]),
        ("Trajectory Reference Directory", summary["trajectory_reference_dir"]),
        ("Runs", str(summary["total_runs"])),
        ("Mean Map AUC", f"{summary['mean_map_auc']:.4f}"),
        ("Mean Map Chamfer", f"{summary['mean_map_chamfer']:.4f}"),
        ("Mean Trajectory ATE RMSE", f"{summary['mean_traj_ate_rmse']:.4f}"),
        ("Mean Trajectory RPE RMSE", f"{summary['mean_traj_rpe_rmse']:.4f}"),
        (
            "Mean Trajectory RPE Rotation RMSE",
            (
                f"{summary['mean_traj_rpe_rotation_rmse_deg']:.4f} deg"
                if summary["mean_traj_rpe_rotation_rmse_deg"] is not None
                else "n/a (quaternion orientations unavailable)"
            ),
        ),
        ("Mean Trajectory Drift", f"{summary['mean_traj_drift']:.4f}"),
        ("Mean Trajectory Coverage", f"{summary['mean_traj_coverage']:.1%}"),
    ]
    if summary["best_map_auc"] is not None:
        summary_rows += [
            ("Best Map AUC", f"{summary['best_map_auc']['id']} ({summary['best_map_auc']['map']['auc']:.4f})"),
            (
                "Worst Trajectory ATE",
                f"{summary['worst_traj_ate']['id']} ({summary['worst_traj_ate']['trajectory']['ate']['rmse']:.4f})",
            ),
        ]
    if gate is not None:
        summary_rows += [
            ("Min AUC", _format_optional_float(gate["min_auc"])),
            ("Max Chamfer", _format_optional_float(gate["max_chamfer"])),
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
            ("Map Failures", str(map_failed_count)),
            ("Trajectory Failures", str(trajectory_failed_count)),
        ]
    summary_row_actions = {
        "Pass": _html_summary_action(
            "run-batch-summary-show-pass",
            "pass",
            "Show pass",
            pass_count,
            disabled=gate is None or pass_count == 0,
            onclick_function="applyRunBatchQuickAction",
        ),
        "Fail": _html_summary_action(
            "run-batch-summary-show-failed",
            "failed",
            "Show failed",
            failed_count,
            disabled=gate is None or failed_count == 0,
            onclick_function="applyRunBatchQuickAction",
        ),
        "Map Failures": _html_summary_action(
            "run-batch-summary-show-map-failed",
            "map-failed",
            "Show map issues",
            map_failed_count,
            disabled=gate is None or map_failed_count == 0,
            onclick_function="applyRunBatchQuickAction",
        ),
        "Trajectory Failures": _html_summary_action(
            "run-batch-summary-show-trajectory-failed",
            "trajectory-failed",
            "Show trajectory issues",
            trajectory_failed_count,
            disabled=gate is None or trajectory_failed_count == 0,
            onclick_function="applyRunBatchQuickAction",
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

    result_rows_html = []
    inspection_rows_html = []
    for item in summary["results"]:
        trajectory_rotation = item["trajectory"].get("rpe_rotation")
        trajectory_rotation_data = (
            ""
            if trajectory_rotation is None
            else f"{trajectory_rotation['rmse']:.8f}"
        )
        item_gate = _run_batch_gate_for_item(
            item,
            min_auc=min_auc,
            max_chamfer=max_chamfer,
            max_ate=max_ate,
            max_rpe=max_rpe,
            max_rpe_rotation_deg=max_rpe_rotation_deg,
            max_drift=max_drift,
            min_coverage=min_coverage,
        )
        status = None
        passed = False
        failed = False
        map_failed = False
        trajectory_failed = False
        if gate is not None:
            status = "PASS" if item_gate is not None and item_gate["passed"] else "FAIL"
            passed = item_gate is not None and item_gate["passed"]
            failed = item_gate is not None and not item_gate["passed"]
            map_failed = item_gate is not None and item_gate["map_passed"] is False
            trajectory_failed = item_gate is not None and item_gate["trajectory_passed"] is False
        row_attrs = [
            f'data-id="{escape(item["id"])}"',
            f'data-map-auc="{item["map"]["auc"]:.8f}"',
            f'data-map-chamfer="{item["map"]["chamfer_distance"]:.8f}"',
            f'data-traj-ate="{item["trajectory"]["ate"]["rmse"]:.8f}"',
            f'data-traj-rpe="{item["trajectory"]["rpe_translation"]["rmse"]:.8f}"',
            f'data-traj-rpe-rotation="{trajectory_rotation_data}"',
            f'data-traj-drift="{item["trajectory"]["drift"]["endpoint"]:.8f}"',
            f'data-coverage="{item["trajectory"]["matching"]["coverage_ratio"]:.8f}"',
            f'data-passed="{"true" if passed else "false"}"',
            f'data-failed="{"true" if failed else "false"}"',
            f'data-map-failed="{"true" if map_failed else "false"}"',
            f'data-trajectory-failed="{"true" if trajectory_failed else "false"}"',
        ]
        result_rows_html.append(
            f"<tr {' '.join(row_attrs)}>"
            f"<td>{escape(item['id'])}</td>"
            f"<td>{item['map']['auc']:.4f}</td>"
            f"<td>{item['map']['chamfer_distance']:.4f}</td>"
            f"<td>{item['trajectory']['ate']['rmse']:.4f}</td>"
            f"<td>{item['trajectory']['rpe_translation']['rmse']:.4f}</td>"
            f"<td>{_format_optional_float(None if trajectory_rotation is None else trajectory_rotation['rmse'])}</td>"
            f"<td>{item['trajectory']['drift']['endpoint']:.4f}</td>"
            f"<td>{item['trajectory']['matching']['coverage_ratio']:.1%}</td>"
            + (
                f"<td>{'PASS' if item_gate is not None and item_gate['map_passed'] is not False else 'FAIL'}</td>"
                f"<td>{'PASS' if item_gate is not None and item_gate['trajectory_passed'] is not False else 'FAIL'}</td>"
                f"<td>{status}</td>"
                if gate is not None
                else ""
            )
            + "</tr>"
        )
        inspection_rows_html.append(
            f"<tr {' '.join(row_attrs)}>"
            f"<td>{escape(item['id'])}</td>"
            "<td>"
            f"{_html_command_block(item['inspect']['run_web'])}"
            f"{_html_command_block(item['inspect']['run_evaluate'])}"
            f"{_html_command_block(item['inspect']['map']['web_heatmap'])}"
            f"{_html_command_block(item['inspect']['map']['heatmap3d'])}"
            f"{_html_command_block(item['inspect']['trajectory']['traj_evaluate'])}"
            f"{_html_command_block(item['inspect']['trajectory']['traj_evaluate_aligned'])}"
            f"{_html_command_block(item['inspect']['trajectory']['traj_evaluate_rigid'])}"
            "</td>"
            "</tr>"
        )
    results_html = "\n".join(result_rows_html)
    inspection_html = "\n".join(inspection_rows_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CloudAnalyzer Run Batch Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #111827; }}
    h1, h2 {{ margin-bottom: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f3f4f6; }}
    tbody tr:nth-child(even) {{ background: #f9fafb; }}
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
    .summary-chip-map-failed {{ border-color: #f59e0b; color: #92400e; background: #fffbeb; }}
    .summary-chip-trajectory-failed {{ border-color: #c084fc; color: #6b21a8; background: #faf5ff; }}
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
    .command-row code {{ flex: 1; white-space: pre-wrap; word-break: break-word; }}
    button {{
      padding: 0.3rem 0.55rem; border: 1px solid #cbd5e1;
      border-radius: 6px; background: #ffffff; cursor: pointer;
    }}
  </style>
</head>
<body>
  <h1>CloudAnalyzer Run Batch Evaluation Report</h1>
  <h2>Summary</h2>
  <table class="summary-table"><tbody>{summary_html}</tbody></table>

  <div class="quick-actions">
    <button type="button" id="run-batch-quick-show-pass" data-quick-action="pass" {"disabled" if gate is None or pass_count == 0 else ""} onclick="applyRunBatchQuickAction('pass')">
      Show pass ({pass_count})
    </button>
    <button type="button" id="run-batch-quick-show-failed" data-quick-action="failed" {"disabled" if gate is None or failed_count == 0 else ""} onclick="applyRunBatchQuickAction('failed')">
      Show failed ({failed_count})
    </button>
    <button type="button" id="run-batch-quick-show-map-failed" data-quick-action="map-failed" {"disabled" if gate is None or map_failed_count == 0 else ""} onclick="applyRunBatchQuickAction('map-failed')">
      Show map issues ({map_failed_count})
    </button>
    <button type="button" id="run-batch-quick-show-trajectory-failed" data-quick-action="trajectory-failed" {"disabled" if gate is None or trajectory_failed_count == 0 else ""} onclick="applyRunBatchQuickAction('trajectory-failed')">
      Show trajectory issues ({trajectory_failed_count})
    </button>
    <button type="button" id="run-batch-quick-reset-view" data-quick-action="reset" onclick="applyRunBatchQuickAction('reset')">
      Reset view
    </button>
  </div>

  <h2>Results</h2>
  <div class="filter-bar">
    <label class="filter-control" id="run-batch-sort-results-control">
      Sort by
      <select id="run-batch-sort-results" onchange="refreshRunBatchResultsView()">
        <option value="id-asc">ID (A-Z)</option>
        <option value="map-auc-desc">Map AUC (high-low)</option>
        <option value="map-auc-asc">Map AUC (low-high)</option>
        <option value="map-chamfer-desc">Map Chamfer (high-low)</option>
        <option value="map-chamfer-asc">Map Chamfer (low-high)</option>
        <option value="traj-ate-desc">Trajectory ATE (high-low)</option>
        <option value="traj-ate-asc">Trajectory ATE (low-high)</option>
        <option value="traj-drift-desc">Trajectory Drift (high-low)</option>
        <option value="traj-drift-asc">Trajectory Drift (low-high)</option>
        <option value="coverage-desc">Coverage (high-low)</option>
        <option value="coverage-asc">Coverage (low-high)</option>
      </select>
    </label>
    <label class="filter-control{' filter-control-disabled' if gate is None else ''}" id="run-batch-filter-pass-only-control">
      <input type="checkbox" id="run-batch-filter-pass-only" {"disabled" if gate is None else ""} onchange="refreshRunBatchResultsView()">
      Pass only
    </label>
    <label class="filter-control{' filter-control-disabled' if gate is None else ''}" id="run-batch-filter-failed-only-control">
      <input type="checkbox" id="run-batch-filter-failed-only" {"disabled" if gate is None else ""} onchange="refreshRunBatchResultsView()">
      Failed only
    </label>
    <label class="filter-control{' filter-control-disabled' if gate is None else ''}" id="run-batch-filter-map-failed-only-control">
      <input type="checkbox" id="run-batch-filter-map-failed-only" {"disabled" if gate is None else ""} onchange="refreshRunBatchResultsView()">
      Map issue only
    </label>
    <label class="filter-control{' filter-control-disabled' if gate is None else ''}" id="run-batch-filter-trajectory-failed-only-control">
      <input type="checkbox" id="run-batch-filter-trajectory-failed-only" {"disabled" if gate is None else ""} onchange="refreshRunBatchResultsView()">
      Trajectory issue only
    </label>
    <span class="filter-actions">
      <button type="button" id="run-batch-reset-filters" onclick="resetRunBatchFilters()">Reset filters</button>
    </span>
    <span class="filter-summary" id="run-batch-filter-summary"></span>
  </div>
  <table>
    <thead>
      <tr>
        <th>ID</th>
        <th>Map AUC</th>
        <th>Map Chamfer</th>
        <th>Traj ATE</th>
        <th>Traj RPE</th>
        <th>Traj RPE Rot (deg)</th>
        <th>Traj Drift</th>
        <th>Coverage</th>
        {"<th>Map Status</th><th>Trajectory Status</th><th>Overall</th>" if gate is not None else ""}
      </tr>
    </thead>
    <tbody id="run-batch-results-table-body">
      {results_html}
    </tbody>
  </table>

  <h2>Inspection Commands</h2>
  <table>
    <thead>
      <tr><th>ID</th><th>Inspection</th></tr>
    </thead>
    <tbody id="run-batch-inspection-table-body">
      {inspection_html}
    </tbody>
  </table>
  <script>
    function compareRunBatchRows(a, b, sortValue) {{
      const separatorIndex = sortValue.lastIndexOf('-');
      const sortKey = separatorIndex >= 0 ? sortValue.slice(0, separatorIndex) : sortValue;
      const sortDirection = separatorIndex >= 0 ? sortValue.slice(separatorIndex + 1) : 'asc';

      function compareValues(left, right) {{
        if (left < right) return sortDirection === 'desc' ? 1 : -1;
        if (left > right) return sortDirection === 'desc' ? -1 : 1;
        return 0;
      }}

      if (sortKey === 'id') return compareValues(a.dataset.id || '', b.dataset.id || '');
      if (sortKey === 'map-auc') return compareValues(Number(a.dataset.mapAuc || 0), Number(b.dataset.mapAuc || 0));
      if (sortKey === 'map-chamfer') return compareValues(Number(a.dataset.mapChamfer || 0), Number(b.dataset.mapChamfer || 0));
      if (sortKey === 'traj-ate') return compareValues(Number(a.dataset.trajAte || 0), Number(b.dataset.trajAte || 0));
      if (sortKey === 'traj-drift') return compareValues(Number(a.dataset.trajDrift || 0), Number(b.dataset.trajDrift || 0));
      if (sortKey === 'coverage') return compareValues(Number(a.dataset.coverage || 0), Number(b.dataset.coverage || 0));
      return compareValues(a.dataset.id || '', b.dataset.id || '');
    }}

    function setRunBatchCheckedIfEnabled(id, checked) {{
      const input = document.getElementById(id);
      if (input && !input.disabled) {{
        input.checked = checked;
      }}
    }}

    function setRunBatchSortValue(value) {{
      const sortResults = document.getElementById('run-batch-sort-results');
      if (sortResults) {{
        const option = Array.from(sortResults.options).find((item) => item.value === value && !item.disabled);
        if (option) {{
          sortResults.value = value;
        }}
      }}
    }}

    function updateRunBatchActionStates(action) {{
      const summaryButtons = Array.from(document.querySelectorAll('[data-summary-action]'));
      const quickButtons = Array.from(document.querySelectorAll('[data-quick-action]'));

      summaryButtons.forEach((button) => {{
        button.classList.toggle('summary-chip-active', button.dataset.summaryAction === action);
      }});
      quickButtons.forEach((button) => {{
        button.classList.toggle('quick-action-chip-active', button.dataset.quickAction === action);
      }});
    }}

    function updateRunBatchFilterControlStates(passEnabled, failedEnabled, mapFailedEnabled, trajectoryFailedEnabled, sortValue) {{
      const controlStates = [
        ['run-batch-filter-pass-only-control', passEnabled],
        ['run-batch-filter-failed-only-control', failedEnabled],
        ['run-batch-filter-map-failed-only-control', mapFailedEnabled],
        ['run-batch-filter-trajectory-failed-only-control', trajectoryFailedEnabled],
        ['run-batch-sort-results-control', sortValue !== 'id-asc'],
      ];

      controlStates.forEach(([id, enabled]) => {{
        const control = document.getElementById(id);
        if (control) {{
          control.classList.toggle('filter-control-active', enabled);
        }}
      }});
    }}

    function refreshRunBatchResultsView() {{
      const passOnly = document.getElementById('run-batch-filter-pass-only');
      const failedOnly = document.getElementById('run-batch-filter-failed-only');
      const mapFailedOnly = document.getElementById('run-batch-filter-map-failed-only');
      const trajectoryFailedOnly = document.getElementById('run-batch-filter-trajectory-failed-only');
      const sortResults = document.getElementById('run-batch-sort-results');
      const passEnabled = passOnly && !passOnly.disabled && passOnly.checked;
      const failedEnabled = failedOnly && !failedOnly.disabled && failedOnly.checked;
      const mapFailedEnabled = mapFailedOnly && !mapFailedOnly.disabled && mapFailedOnly.checked;
      const trajectoryFailedEnabled = trajectoryFailedOnly && !trajectoryFailedOnly.disabled && trajectoryFailedOnly.checked;
      const sortValue = sortResults ? sortResults.value : 'id-asc';

      const resultsTableBody = document.getElementById('run-batch-results-table-body');
      const inspectionTableBody = document.getElementById('run-batch-inspection-table-body');
      const resultRows = Array.from(document.querySelectorAll('#run-batch-results-table-body tr'));
      const inspectionRows = Array.from(document.querySelectorAll('#run-batch-inspection-table-body tr'));
      let visibleCount = 0;

      function rowMatches(row) {{
        const isPassed = row.dataset.passed === 'true';
        const isFailed = row.dataset.failed === 'true';
        const isMapFailed = row.dataset.mapFailed === 'true';
        const isTrajectoryFailed = row.dataset.trajectoryFailed === 'true';
        if (passEnabled && !isPassed) return false;
        if (failedEnabled && !isFailed) return false;
        if (mapFailedEnabled && !isMapFailed) return false;
        if (trajectoryFailedEnabled && !isTrajectoryFailed) return false;
        return true;
      }}

      resultRows.sort((a, b) => compareRunBatchRows(a, b, sortValue));
      if (resultsTableBody) {{
        resultRows.forEach((row) => {{ resultsTableBody.appendChild(row); }});
      }}

      const resultOrder = new Map(resultRows.map((row, index) => [row.dataset.id || '', index]));
      inspectionRows.sort((a, b) => {{
        const aOrder = resultOrder.get(a.dataset.id || '') ?? Number.MAX_SAFE_INTEGER;
        const bOrder = resultOrder.get(b.dataset.id || '') ?? Number.MAX_SAFE_INTEGER;
        return aOrder - bOrder;
      }});
      if (inspectionTableBody) {{
        inspectionRows.forEach((row) => {{ inspectionTableBody.appendChild(row); }});
      }}

      resultRows.forEach((row) => {{
        const visible = rowMatches(row);
        row.style.display = visible ? '' : 'none';
        if (visible) visibleCount += 1;
      }});
      inspectionRows.forEach((row) => {{
        row.style.display = rowMatches(row) ? '' : 'none';
      }});

      const summary = document.getElementById('run-batch-filter-summary');
      if (summary) {{
        const activeFilters = [];
        if (passEnabled) activeFilters.push('pass');
        if (failedEnabled) activeFilters.push('failed');
        if (mapFailedEnabled) activeFilters.push('map-failed');
        if (trajectoryFailedEnabled) activeFilters.push('trajectory-failed');
        summary.textContent = `Showing ${{visibleCount}} / ${{resultRows.length}} run(s)` + (
          activeFilters.length ? ` | Filters: ${{activeFilters.join(', ')}}` : ''
        ) + (
          sortValue ? ` | Sort: ${{sortValue}}` : ''
        );
      }}

      let activeAction = 'reset';
      if (passEnabled && !failedEnabled && !mapFailedEnabled && !trajectoryFailedEnabled && sortValue === 'map-auc-desc') {{
        activeAction = 'pass';
      }} else if (!passEnabled && failedEnabled && !mapFailedEnabled && !trajectoryFailedEnabled && sortValue === 'traj-ate-desc') {{
        activeAction = 'failed';
      }} else if (!passEnabled && !failedEnabled && mapFailedEnabled && !trajectoryFailedEnabled && sortValue === 'map-auc-asc') {{
        activeAction = 'map-failed';
      }} else if (!passEnabled && !failedEnabled && !mapFailedEnabled && trajectoryFailedEnabled && sortValue === 'traj-ate-desc') {{
        activeAction = 'trajectory-failed';
      }}
      updateRunBatchActionStates(activeAction);
      updateRunBatchFilterControlStates(passEnabled, failedEnabled, mapFailedEnabled, trajectoryFailedEnabled, sortValue);
    }}

    function applyRunBatchQuickAction(action) {{
      if (action === 'pass') {{
        setRunBatchCheckedIfEnabled('run-batch-filter-pass-only', true);
        setRunBatchCheckedIfEnabled('run-batch-filter-failed-only', false);
        setRunBatchCheckedIfEnabled('run-batch-filter-map-failed-only', false);
        setRunBatchCheckedIfEnabled('run-batch-filter-trajectory-failed-only', false);
        setRunBatchSortValue('map-auc-desc');
      }} else if (action === 'failed') {{
        setRunBatchCheckedIfEnabled('run-batch-filter-pass-only', false);
        setRunBatchCheckedIfEnabled('run-batch-filter-failed-only', true);
        setRunBatchCheckedIfEnabled('run-batch-filter-map-failed-only', false);
        setRunBatchCheckedIfEnabled('run-batch-filter-trajectory-failed-only', false);
        setRunBatchSortValue('traj-ate-desc');
      }} else if (action === 'map-failed') {{
        setRunBatchCheckedIfEnabled('run-batch-filter-pass-only', false);
        setRunBatchCheckedIfEnabled('run-batch-filter-failed-only', false);
        setRunBatchCheckedIfEnabled('run-batch-filter-map-failed-only', true);
        setRunBatchCheckedIfEnabled('run-batch-filter-trajectory-failed-only', false);
        setRunBatchSortValue('map-auc-asc');
      }} else if (action === 'trajectory-failed') {{
        setRunBatchCheckedIfEnabled('run-batch-filter-pass-only', false);
        setRunBatchCheckedIfEnabled('run-batch-filter-failed-only', false);
        setRunBatchCheckedIfEnabled('run-batch-filter-map-failed-only', false);
        setRunBatchCheckedIfEnabled('run-batch-filter-trajectory-failed-only', true);
        setRunBatchSortValue('traj-ate-desc');
      }} else {{
        resetRunBatchFilters();
        return;
      }}
      refreshRunBatchResultsView();
    }}

    function resetRunBatchFilters() {{
      const filterIds = [
        'run-batch-filter-pass-only',
        'run-batch-filter-failed-only',
        'run-batch-filter-map-failed-only',
        'run-batch-filter-trajectory-failed-only',
      ];
      filterIds.forEach((id) => {{
        const input = document.getElementById(id);
        if (input && !input.disabled) {{
          input.checked = false;
        }}
      }});
      const sortResults = document.getElementById('run-batch-sort-results');
      if (sortResults) {{
        sortResults.value = 'id-asc';
      }}
      refreshRunBatchResultsView();
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
    refreshRunBatchResultsView();
  </script>
</body>
</html>
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def save_run_batch_report(
    results: list[dict],
    map_reference_dir: str,
    trajectory_reference_dir: str,
    output_path: str,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_rpe_rotation_deg: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> None:
    """Write a combined run-batch report based on the file extension."""
    ext = Path(output_path).suffix.lower()
    if ext in {".md", ".markdown"}:
        make_run_batch_markdown(
            results,
            map_reference_dir,
            trajectory_reference_dir,
            output_path,
            min_auc=min_auc,
            max_chamfer=max_chamfer,
            max_ate=max_ate,
            max_rpe=max_rpe,
            max_rpe_rotation_deg=max_rpe_rotation_deg,
            max_drift=max_drift,
            min_coverage=min_coverage,
        )
        return
    if ext == ".html":
        make_run_batch_html(
            results,
            map_reference_dir,
            trajectory_reference_dir,
            output_path,
            min_auc=min_auc,
            max_chamfer=max_chamfer,
            max_ate=max_ate,
            max_rpe=max_rpe,
            max_rpe_rotation_deg=max_rpe_rotation_deg,
            max_drift=max_drift,
            min_coverage=min_coverage,
        )
        return
    raise ValueError("Unsupported report format. Use .md, .markdown, or .html")
