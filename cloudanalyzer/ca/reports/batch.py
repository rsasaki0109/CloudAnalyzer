"""Batch evaluation reports."""

from html import escape
from pathlib import Path
from typing import Any
from ca.pareto import mark_quality_size_recommended
from ca.plot import plot_quality_vs_size
from ca.reports.common import (
    _compression_for_item,
    _gate_for_item,
    _html_command_block,
    _html_summary_action,
    _quality_vs_size_plot_path,
)


def make_batch_summary(
    results: list[dict],
    reference_path: str,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
) -> dict:
    """Build summary data for batch evaluation results."""
    gate_enabled = (
        min_auc is not None
        or max_chamfer is not None
        or any(item.get("quality_gate") is not None for item in results)
    )
    recommended_item = mark_quality_size_recommended(
        results,
        min_auc=min_auc,
        max_chamfer=max_chamfer,
    )
    quality_gate: dict[str, Any] | None = None
    compression: dict[str, Any] | None = None

    if not results:
        if gate_enabled:
            quality_gate = {
                "min_auc": min_auc,
                "max_chamfer": max_chamfer,
                "pass_count": 0,
                "fail_count": 0,
                "failed_paths": [],
            }
        return {
            "reference_path": reference_path,
            "total_files": 0,
            "mean_auc": 0.0,
            "mean_chamfer_distance": 0.0,
            "best_auc": None,
            "worst_auc": None,
            "best_chamfer": None,
            "worst_chamfer": None,
            "quality_gate": quality_gate,
            "compression": compression,
            "results": [],
        }

    mean_auc = sum(item["auc"] for item in results) / len(results)
    mean_chamfer = sum(item["chamfer_distance"] for item in results) / len(results)
    compressed_items = [
        item for item in results
        if _compression_for_item(item) is not None
    ]
    if compressed_items:
        pareto_items = [
            item for item in compressed_items
            if (item_compression := _compression_for_item(item)) is not None
            and item_compression.get("pareto_optimal")
        ]
        recommendation_note = (
            "smallest candidate on the gate-filtered Pareto frontier"
            if gate_enabled
            else "smallest Pareto candidate by size ratio"
        )
        if recommended_item is None and gate_enabled:
            recommendation_note = "no compression candidate passed the quality gate"
        size_ratios = [
            _compression_for_item(item)["size_ratio"]  # type: ignore[index]
            for item in compressed_items
        ]
        space_savings = [
            _compression_for_item(item)["space_saving_ratio"]  # type: ignore[index]
            for item in compressed_items
        ]
        compression = {
            "count": len(compressed_items),
            "mean_size_ratio": sum(size_ratios) / len(size_ratios),
            "mean_space_saving_ratio": sum(space_savings) / len(space_savings),
            "best_size_ratio": min(
                compressed_items,
                key=lambda item: _compression_for_item(item)["size_ratio"],  # type: ignore[index]
            ),
            "worst_size_ratio": max(
                compressed_items,
                key=lambda item: _compression_for_item(item)["size_ratio"],  # type: ignore[index]
            ),
            "pareto_optimal_count": len(pareto_items),
            "pareto_optimal_items": pareto_items,
            "recommended_item": recommended_item,
            "recommendation_note": recommendation_note,
        }
    if gate_enabled:
        first_gate = next((_gate_for_item(item, min_auc, max_chamfer) for item in results), None)
        if first_gate is not None:
            if min_auc is None:
                min_auc = first_gate.get("min_auc")
            if max_chamfer is None:
                max_chamfer = first_gate.get("max_chamfer")
        gated_results = [
            gate
            for item in results
            if (gate := _gate_for_item(item, min_auc, max_chamfer)) is not None
        ]
        failed_paths = [
            item["path"]
            for item in results
            if (gate := _gate_for_item(item, min_auc, max_chamfer)) is not None and not gate["passed"]
        ]
        quality_gate = {
            "min_auc": min_auc,
            "max_chamfer": max_chamfer,
            "pass_count": sum(1 for gate in gated_results if gate["passed"]),
            "fail_count": sum(1 for gate in gated_results if not gate["passed"]),
            "failed_paths": failed_paths,
        }

    return {
        "reference_path": reference_path,
        "total_files": len(results),
        "mean_auc": float(mean_auc),
        "mean_chamfer_distance": float(mean_chamfer),
        "best_auc": max(results, key=lambda item: item["auc"]),
        "worst_auc": min(results, key=lambda item: item["auc"]),
        "best_chamfer": min(results, key=lambda item: item["chamfer_distance"]),
        "worst_chamfer": max(results, key=lambda item: item["chamfer_distance"]),
        "quality_gate": quality_gate,
        "compression": compression,
        "results": results,
    }


def make_batch_markdown(
    results: list[dict],
    reference_path: str,
    output_path: str,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
    report_title: str = "CloudAnalyzer Batch Evaluation Report",
    report_notes: list[str] | None = None,
) -> None:
    """Generate a Markdown report for batch evaluation results."""
    summary = make_batch_summary(
        results,
        reference_path,
        min_auc=min_auc,
        max_chamfer=max_chamfer,
    )

    lines = [f"# {report_title}", ""]
    notes = [note.strip() for note in (report_notes or []) if note.strip()]
    for note in notes:
        lines += [note, ""]

    lines += [
        "## Summary",
        f"- Reference: {summary['reference_path']}",
        f"- Files: {summary['total_files']}",
        f"- Mean AUC: {summary['mean_auc']:.4f}",
        f"- Mean Chamfer: {summary['mean_chamfer_distance']:.4f}",
    ]

    if summary["best_auc"] is not None:
        lines += [
            f"- Best AUC: {summary['best_auc']['path']} ({summary['best_auc']['auc']:.4f})",
            f"- Worst AUC: {summary['worst_auc']['path']} ({summary['worst_auc']['auc']:.4f})",
            f"- Best Chamfer: {summary['best_chamfer']['path']} ({summary['best_chamfer']['chamfer_distance']:.4f})",
            f"- Worst Chamfer: {summary['worst_chamfer']['path']} ({summary['worst_chamfer']['chamfer_distance']:.4f})",
        ]

    compression = summary["compression"]
    if compression is not None:
        plot_path = _quality_vs_size_plot_path(output_path)
        plot_quality_vs_size(summary["results"], str(plot_path))
        best_size_item = compression["best_size_ratio"]
        worst_size_item = compression["worst_size_ratio"]
        best_item_compression = _compression_for_item(best_size_item)
        worst_item_compression = _compression_for_item(worst_size_item)
        if best_item_compression is None or worst_item_compression is None:
            raise ValueError("Compression summary is inconsistent")
        pareto_paths = ", ".join(
            item["path"] for item in compression["pareto_optimal_items"]
        ) or "-"
        recommended = compression["recommended_item"]
        recommended_text = "none"
        if recommended is not None:
            recommended_compression = _compression_for_item(recommended)
            if recommended_compression is not None:
                recommended_text = (
                    f"{recommended['path']} "
                    f"(size={recommended_compression['size_ratio']:.4f}, auc={recommended['auc']:.4f})"
                )
        lines += [
            "",
            "## Compression",
            f"- Files with compressed artifacts: {compression['count']}",
            f"- Mean Size Ratio: {compression['mean_size_ratio']:.4f}",
            f"- Mean Space Saving: {compression['mean_space_saving_ratio']:.1%}",
            f"- Pareto Candidates: {compression['pareto_optimal_count']}",
            f"- Pareto Paths: {pareto_paths}",
            f"- Recommended: {recommended_text}",
            f"- Recommendation Rule: {compression['recommendation_note']}",
            (
                f"- Best Size Ratio: {best_size_item['path']} "
                f"({best_item_compression['size_ratio']:.4f})"
            ),
            (
                f"- Worst Size Ratio: {worst_size_item['path']} "
                f"({worst_item_compression['size_ratio']:.4f})"
            ),
            "",
            f"![Quality vs Size]({plot_path.name})",
        ]

    gate = summary["quality_gate"]
    if gate is not None:
        lines += [
            "",
            "## Quality Gate",
            f"- Min AUC: {gate['min_auc']:.4f}" if gate["min_auc"] is not None else "- Min AUC: disabled",
            (
                f"- Max Chamfer: {gate['max_chamfer']:.4f}"
                if gate["max_chamfer"] is not None
                else "- Max Chamfer: disabled"
            ),
            f"- Pass: {gate['pass_count']}",
            f"- Fail: {gate['fail_count']}",
        ]

    lines += [
        "",
        "## Results",
        "",
        "| " + " | ".join(
            [
                *["Path", "Points", "Chamfer", "AUC"],
                *(["Size Ratio", "Pareto", "Recommended"] if compression is not None else []),
                *["Best F1", "Threshold"],
                *(["Status"] if gate is not None else []),
            ]
        ) + " |",
        "|" + "|".join(
            [
                *["---", "---:", "---:", "---:"],
                *(["---:", "---", "---"] if compression is not None else []),
                *["---:", "---:"],
                *(["---"] if gate is not None else []),
            ]
        ) + "|",
    ]

    for item in summary["results"]:
        best_f1 = item["best_f1"]
        row = [
            item["path"],
            str(item["num_points"]),
            f"{item['chamfer_distance']:.4f}",
            f"{item['auc']:.4f}",
        ]
        if compression is not None:
            item_compression = _compression_for_item(item)
            row += [
                f"{item_compression['size_ratio']:.4f}" if item_compression is not None else "-",
                (
                    "Yes"
                    if item_compression is not None and item_compression.get("pareto_optimal")
                    else ""
                ),
                (
                    "Yes"
                    if item_compression is not None and item_compression.get("recommended")
                    else ""
                ),
            ]
        row += [
            f"{best_f1['f1']:.4f}",
            f"{best_f1['threshold']:.2f}",
        ]
        if gate is not None:
            item_gate = _gate_for_item(item, gate["min_auc"], gate["max_chamfer"])
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
        if inspect.get("web_heatmap"):
            lines.append(f"- {item['path']}: `{inspect['web_heatmap']}`")
        if inspect.get("heatmap3d"):
            lines.append(f"  - Snapshot: `{inspect['heatmap3d']}`")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def make_batch_html(
    results: list[dict],
    reference_path: str,
    output_path: str,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
    report_title: str = "CloudAnalyzer Batch Evaluation Report",
    report_notes: list[str] | None = None,
) -> None:
    """Generate an HTML report for batch evaluation results."""
    summary = make_batch_summary(
        results,
        reference_path,
        min_auc=min_auc,
        max_chamfer=max_chamfer,
    )

    summary_rows = [
        ("Reference", summary["reference_path"]),
        ("Files", str(summary["total_files"])),
        ("Mean AUC", f"{summary['mean_auc']:.4f}"),
        ("Mean Chamfer", f"{summary['mean_chamfer_distance']:.4f}"),
    ]
    if summary["best_auc"] is not None:
        summary_rows += [
            ("Best AUC", f"{summary['best_auc']['path']} ({summary['best_auc']['auc']:.4f})"),
            ("Worst AUC", f"{summary['worst_auc']['path']} ({summary['worst_auc']['auc']:.4f})"),
            (
                "Best Chamfer",
                f"{summary['best_chamfer']['path']} ({summary['best_chamfer']['chamfer_distance']:.4f})",
            ),
            (
                "Worst Chamfer",
                f"{summary['worst_chamfer']['path']} ({summary['worst_chamfer']['chamfer_distance']:.4f})",
            ),
        ]
    compression = summary["compression"]
    if compression is not None:
        plot_path = _quality_vs_size_plot_path(output_path)
        plot_quality_vs_size(summary["results"], str(plot_path))
        pareto_paths = ", ".join(
            item["path"] for item in compression["pareto_optimal_items"]
        ) or "-"
        recommended = compression["recommended_item"]
        recommended_text = "none"
        if recommended is not None:
            recommended_compression = _compression_for_item(recommended)
            if recommended_compression is not None:
                recommended_text = (
                    f"{recommended['path']} "
                    f"(size={recommended_compression['size_ratio']:.4f}, auc={recommended['auc']:.4f})"
                )
        summary_rows += [
            ("Compressed Files", str(compression["count"])),
            ("Mean Size Ratio", f"{compression['mean_size_ratio']:.4f}"),
            ("Mean Space Saving", f"{compression['mean_space_saving_ratio']:.1%}"),
            ("Pareto Candidates", str(compression["pareto_optimal_count"])),
            ("Pareto Paths", pareto_paths),
            ("Recommended", recommended_text),
            ("Recommendation Rule", compression["recommendation_note"]),
        ]
    gate = summary["quality_gate"]
    if gate is not None:
        summary_rows += [
            ("Min AUC", f"{gate['min_auc']:.4f}" if gate["min_auc"] is not None else "disabled"),
            (
                "Max Chamfer",
                f"{gate['max_chamfer']:.4f}" if gate["max_chamfer"] is not None else "disabled",
            ),
            ("Pass", str(gate["pass_count"])),
            ("Fail", str(gate["fail_count"])),
        ]
    failed_count = gate["fail_count"] if gate is not None else 0
    pass_count = gate["pass_count"] if gate is not None else 0
    pareto_count = compression["pareto_optimal_count"] if compression is not None else 0
    recommended_count = (
        1 if compression is not None and compression["recommended_item"] is not None else 0
    )
    sort_control_class = "filter-control"
    pass_control_class = "filter-control"
    failed_control_class = "filter-control"
    recommended_control_class = "filter-control"
    pareto_control_class = "filter-control"
    if gate is None:
        pass_control_class += " filter-control-disabled"
        failed_control_class += " filter-control-disabled"
    if compression is None:
        recommended_control_class += " filter-control-disabled"
        pareto_control_class += " filter-control-disabled"
    summary_row_actions = {
        "Pass": _html_summary_action(
            "summary-show-pass",
            "pass",
            "Show pass",
            pass_count,
            disabled=gate is None or pass_count == 0,
        ),
        "Fail": _html_summary_action(
            "summary-show-failed",
            "failed",
            "Show failed",
            failed_count,
            disabled=gate is None or failed_count == 0,
        ),
        "Pareto Candidates": _html_summary_action(
            "summary-show-pareto",
            "pareto",
            "Show pareto",
            pareto_count,
            disabled=compression is None or pareto_count == 0,
        ),
        "Recommended": _html_summary_action(
            "summary-show-recommended",
            "recommended",
            "Show recommended",
            recommended_count,
            disabled=compression is None or recommended_count == 0,
        ),
    }

    row_parts = []
    for item in summary["results"]:
        row_classes = []
        row_attrs = []
        row = (
            "<tr>"
            f"<td>{escape(item['path'])}</td>"
            f"<td>{item['num_points']}</td>"
            f"<td>{item['chamfer_distance']:.4f}</td>"
            f"<td>{item['auc']:.4f}</td>"
        )
        if compression is not None:
            item_compression = _compression_for_item(item)
            row += (
                f"<td>{item_compression['size_ratio']:.4f}</td>"
                if item_compression is not None
                else "<td>-</td>"
            )
            row += (
                "<td>Yes</td>"
                if item_compression is not None and item_compression.get("pareto_optimal")
                else "<td></td>"
            )
            row += (
                "<td>Yes</td>"
                if item_compression is not None and item_compression.get("recommended")
                else "<td></td>"
            )
        row += (
            f"<td>{item['best_f1']['f1']:.4f}</td>"
            f"<td>{item['best_f1']['threshold']:.2f}</td>"
        )
        if gate is not None:
            item_gate = _gate_for_item(item, gate["min_auc"], gate["max_chamfer"])
            status = "PASS" if item_gate is not None and item_gate["passed"] else "FAIL"
            row_attrs.append(
                f'data-passed="{"true" if status == "PASS" else "false"}"'
            )
            row_attrs.append(
                f'data-failed="{"true" if status == "FAIL" else "false"}"'
            )
            if status == "FAIL":
                row_classes.append("fail-row")
            row += f"<td>{status}</td>"
        else:
            row_attrs.append('data-passed="false"')
            row_attrs.append('data-failed="false"')
        if compression is not None:
            item_compression = _compression_for_item(item)
            row_attrs.append(
                f'data-pareto="{"true" if item_compression is not None and item_compression.get("pareto_optimal") else "false"}"'
            )
            row_attrs.append(
                f'data-recommended="{"true" if item_compression is not None and item_compression.get("recommended") else "false"}"'
            )
            if item_compression is not None and item_compression.get("pareto_optimal"):
                row_classes.append("pareto-row")
            if item_compression is not None and item_compression.get("recommended"):
                row_classes.append("recommended-row")
        else:
            row_attrs.append('data-pareto="false"')
            row_attrs.append('data-recommended="false"')
        row_attrs += [
            f'data-path="{escape(item["path"])}"',
            f'data-points="{item["num_points"]}"',
            f'data-chamfer="{item["chamfer_distance"]:.8f}"',
            f'data-auc="{item["auc"]:.8f}"',
            (
                f'data-size-ratio="{item_compression["size_ratio"]:.8f}"'
                if compression is not None and item_compression is not None
                else 'data-size-ratio=""'
            ),
        ]
        if row_classes:
            row = row.replace(
                "<tr>",
                f"<tr class=\"{' '.join(row_classes)}\" {' '.join(row_attrs)}>",
                1,
            )
        else:
            row = row.replace("<tr>", f"<tr {' '.join(row_attrs)}>", 1)
        row += "</tr>"
        row_parts.append(row)
    rows_html = "\n".join(row_parts)
    summary_html = "\n".join(
        (
            f"<tr><th>{escape(label)}</th>"
            "<td>"
            f"<span>{escape(value)}</span>"
            f"{summary_row_actions.get(label, '')}"
            "</td></tr>"
        )
        for label, value in summary_rows
    )
    inspection_row_parts = []
    for item in summary["results"]:
        item_gate = (
            _gate_for_item(item, gate["min_auc"], gate["max_chamfer"])
            if gate is not None
            else None
        )
        item_compression = _compression_for_item(item)
        item_size_ratio_attr = (
            f'data-size-ratio="{item_compression["size_ratio"]:.8f}"'
            if item_compression is not None
            else 'data-size-ratio=""'
        )
        inspection_row_parts.append(
            "<tr "
            f'data-passed="{"true" if item_gate is not None and item_gate["passed"] else "false"}" '
            f'data-failed="{"true" if item_gate is not None and not item_gate["passed"] else "false"}" '
            f'data-pareto="{"true" if item_compression is not None and item_compression.get("pareto_optimal") else "false"}" '
            f'data-recommended="{"true" if item_compression is not None and item_compression.get("recommended") else "false"}" '
            f'data-path="{escape(item["path"])}" '
            f'data-points="{item["num_points"]}" '
            f'data-chamfer="{item["chamfer_distance"]:.8f}" '
            f'data-auc="{item["auc"]:.8f}" '
            f"{item_size_ratio_attr}"
            ">"
            f"<td>{escape(item['path'])}</td>"
            "<td>"
            f"{_html_command_block(item.get('inspect', {}).get('web_heatmap', ''))}"
            f"{_html_command_block(item.get('inspect', {}).get('heatmap3d', ''))}"
            "</td>"
            "</tr>"
        )
    inspection_rows_html = "\n".join(inspection_row_parts)
    notes = [note.strip() for note in (report_notes or []) if note.strip()]
    notes_html = "\n".join(
        f'<p class="report-note">{escape(note)}</p>'
        for note in notes
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(report_title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #111827; }}
    h1, h2 {{ margin-bottom: 0.5rem; }}
    .report-notes {{ margin: 0 0 1.5rem 0; max-width: 72rem; color: #374151; line-height: 1.6; }}
    .report-note {{ margin: 0.35rem 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f3f4f6; }}
    tbody tr:nth-child(even) {{ background: #f9fafb; }}
    code {{ white-space: pre-wrap; word-break: break-word; }}
    button {{
      margin-left: 0.75rem; padding: 0.3rem 0.55rem; border: 1px solid #cbd5e1;
      border-radius: 6px; background: #ffffff; cursor: pointer;
    }}
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
      border-color: #93c5fd;
      background: #dbeafe;
      color: #1d4ed8;
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
    .quick-actions {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 0 0 1.5rem 0; }}
    .command-row {{ display: flex; align-items: flex-start; gap: 0.5rem; margin: 0.4rem 0; }}
    .command-row code {{ flex: 1; }}
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
    .summary-chip-pareto {{ border-color: #86efac; color: #166534; background: #f0fdf4; }}
    .summary-chip-recommended {{ border-color: #fcd34d; color: #92400e; background: #fffbeb; }}
    .summary-chip-active {{
      box-shadow: 0 0 0 2px rgba(17, 24, 39, 0.12) inset;
      transform: translateY(-1px);
    }}
    .quick-action-chip-active {{
      box-shadow: 0 0 0 2px rgba(17, 24, 39, 0.12) inset;
      transform: translateY(-1px);
    }}
    .fail-row td {{ background: #fef2f2; }}
    .pareto-row td:first-child {{ box-shadow: inset 4px 0 0 #0f766e; }}
    .recommended-row td:first-child {{ box-shadow: inset 8px 0 0 #f59e0b; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>{escape(report_title)}</h1>
  {f'<div class="report-notes">{notes_html}</div>' if notes_html else ""}
  <h2>Summary</h2>
  <table class="summary-table">
    <tbody>
      {summary_html}
    </tbody>
  </table>
  <div class="quick-actions">
    <button type="button" id="quick-show-pass" data-quick-action="pass" {"disabled" if gate is None else ""} onclick="applyQuickAction('pass')">
      Show pass ({pass_count})
    </button>
    <button type="button" id="quick-show-failed" data-quick-action="failed" {"disabled" if gate is None else ""} onclick="applyQuickAction('failed')">
      Show failed ({failed_count})
    </button>
    <button type="button" id="quick-show-pareto" data-quick-action="pareto" {"disabled" if compression is None else ""} onclick="applyQuickAction('pareto')">
      Show pareto ({pareto_count})
    </button>
    <button type="button" id="quick-show-recommended" data-quick-action="recommended" {"disabled" if compression is None else ""} onclick="applyQuickAction('recommended')">
      Show recommended ({recommended_count})
    </button>
    <button type="button" id="quick-reset-view" data-quick-action="reset" onclick="applyQuickAction('reset')">
      Reset view
    </button>
  </div>
  {f'<h2>Quality vs Size</h2><img src="{escape(plot_path.name)}" alt="Quality vs Size plot" style="max-width: 100%; height: auto; margin: 0 0 2rem 0;">' if compression is not None else ""}

  <h2>Results</h2>
  <div class="filter-bar">
    <label class="{sort_control_class}" id="sort-results-control">
      Sort by
      <select id="sort-results" onchange="refreshResultsView()">
        <option value="path-asc">Path (A-Z)</option>
        <option value="auc-desc">AUC (high-low)</option>
        <option value="auc-asc">AUC (low-high)</option>
        <option value="chamfer-asc">Chamfer (low-high)</option>
        <option value="size-ratio-asc" {"disabled" if compression is None else ""}>Size Ratio (low-high)</option>
        <option value="failed-first" {"disabled" if gate is None else ""}>Failed first</option>
        <option value="recommended-first" {"disabled" if compression is None else ""}>Recommended first</option>
        <option value="points-desc">Points (high-low)</option>
      </select>
    </label>
    <label class="{pass_control_class}" id="filter-pass-only-control">
      <input type="checkbox" id="filter-pass-only" {"disabled" if gate is None else ""} onchange="refreshResultsView()">
      Pass only
    </label>
    <label class="{failed_control_class}" id="filter-failed-only-control">
      <input type="checkbox" id="filter-failed-only" {"disabled" if gate is None else ""} onchange="refreshResultsView()">
      Failed only
    </label>
    <label class="{recommended_control_class}" id="filter-recommended-only-control">
      <input type="checkbox" id="filter-recommended-only" {"disabled" if compression is None else ""} onchange="refreshResultsView()">
      Recommended only
    </label>
    <label class="{pareto_control_class}" id="filter-pareto-only-control">
      <input type="checkbox" id="filter-pareto-only" {"disabled" if compression is None else ""} onchange="refreshResultsView()">
      Pareto only
    </label>
    <span class="filter-actions">
      <button type="button" id="reset-filters" onclick="resetFilters()">Reset filters</button>
    </span>
    <span class="filter-summary" id="filter-summary"></span>
  </div>
  <table>
    <thead>
      <tr>
        <th>Path</th>
        <th>Points</th>
        <th>Chamfer</th>
        <th>AUC</th>
        {"<th>Size Ratio</th>" if compression is not None else ""}
        {"<th>Pareto</th>" if compression is not None else ""}
        {"<th>Recommended</th>" if compression is not None else ""}
        <th>Best F1</th>
        <th>Threshold</th>
        {"<th>Status</th>" if gate is not None else ""}
      </tr>
    </thead>
    <tbody id="results-table-body">
      {rows_html}
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
    <tbody id="inspection-table-body">
      {inspection_rows_html}
    </tbody>
  </table>
  <script>
    function compareRows(a, b, sortValue) {{
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
      if (sortKey === 'points') {{
        return compareValues(Number(a.dataset.points || 0), Number(b.dataset.points || 0));
      }}
      if (sortKey === 'auc') {{
        return compareValues(Number(a.dataset.auc || 0), Number(b.dataset.auc || 0));
      }}
      if (sortKey === 'chamfer') {{
        return compareValues(Number(a.dataset.chamfer || 0), Number(b.dataset.chamfer || 0));
      }}
      if (sortKey === 'size-ratio') {{
        const aValue = a.dataset.sizeRatio === '' ? Number.POSITIVE_INFINITY : Number(a.dataset.sizeRatio);
        const bValue = b.dataset.sizeRatio === '' ? Number.POSITIVE_INFINITY : Number(b.dataset.sizeRatio);
        return compareValues(aValue, bValue);
      }}
      if (sortKey === 'failed' && sortDirection === 'first') {{
        const aFailed = a.dataset.failed === 'true';
        const bFailed = b.dataset.failed === 'true';
        if (aFailed !== bFailed) {{
          return aFailed ? -1 : 1;
        }}
        const aucComparison = compareValues(Number(a.dataset.auc || 0), Number(b.dataset.auc || 0));
        if (aucComparison !== 0) {{
          return aucComparison;
        }}
        return compareValues(a.dataset.path || '', b.dataset.path || '');
      }}
      if (sortKey === 'recommended' && sortDirection === 'first') {{
        const aRecommended = a.dataset.recommended === 'true';
        const bRecommended = b.dataset.recommended === 'true';
        if (aRecommended !== bRecommended) {{
          return aRecommended ? -1 : 1;
        }}
        const aValue = a.dataset.sizeRatio === '' ? Number.POSITIVE_INFINITY : Number(a.dataset.sizeRatio);
        const bValue = b.dataset.sizeRatio === '' ? Number.POSITIVE_INFINITY : Number(b.dataset.sizeRatio);
        const sizeComparison = compareValues(aValue, bValue);
        if (sizeComparison !== 0) {{
          return sizeComparison;
        }}
        return compareValues(a.dataset.path || '', b.dataset.path || '');
      }}
      return compareValues(a.dataset.path || '', b.dataset.path || '');
    }}

    function setCheckedIfEnabled(id, checked) {{
      const input = document.getElementById(id);
      if (input && !input.disabled) {{
        input.checked = checked;
      }}
    }}

    function setSortValue(value) {{
      const sortResults = document.getElementById('sort-results');
      if (sortResults) {{
        const option = Array.from(sortResults.options).find((item) => item.value === value && !item.disabled);
        if (option) {{
          sortResults.value = value;
        }}
      }}
    }}

    function updateActionStates(action) {{
      const summaryButtons = Array.from(document.querySelectorAll('[data-summary-action]'));
      const quickButtons = Array.from(document.querySelectorAll('[data-quick-action]'));

      summaryButtons.forEach((button) => {{
        button.classList.toggle('summary-chip-active', button.dataset.summaryAction === action);
      }});
      quickButtons.forEach((button) => {{
        button.classList.toggle('quick-action-chip-active', button.dataset.quickAction === action);
      }});
    }}

    function updateFilterControlStates(passEnabled, failedEnabled, paretoEnabled, recommendedEnabled, sortValue) {{
      const controlStates = [
        ['filter-pass-only-control', passEnabled],
        ['filter-failed-only-control', failedEnabled],
        ['filter-pareto-only-control', paretoEnabled],
        ['filter-recommended-only-control', recommendedEnabled],
        ['sort-results-control', sortValue !== 'path-asc'],
      ];

      controlStates.forEach(([id, enabled]) => {{
        const control = document.getElementById(id);
        if (control) {{
          control.classList.toggle('filter-control-active', enabled);
        }}
      }});
    }}

    function refreshResultsView() {{
      const failedOnly = document.getElementById('filter-failed-only');
      const passOnly = document.getElementById('filter-pass-only');
      const recommendedOnly = document.getElementById('filter-recommended-only');
      const paretoOnly = document.getElementById('filter-pareto-only');
      const sortResults = document.getElementById('sort-results');
      const passEnabled = passOnly && !passOnly.disabled && passOnly.checked;
      const failedEnabled = failedOnly && !failedOnly.disabled && failedOnly.checked;
      const recommendedEnabled = recommendedOnly && !recommendedOnly.disabled && recommendedOnly.checked;
      const paretoEnabled = paretoOnly && !paretoOnly.disabled && paretoOnly.checked;
      const sortValue = sortResults ? sortResults.value : 'path-asc';

      const resultsTableBody = document.getElementById('results-table-body');
      const inspectionTableBody = document.getElementById('inspection-table-body');
      const resultRows = Array.from(document.querySelectorAll('#results-table-body tr'));
      const inspectionRows = Array.from(document.querySelectorAll('#inspection-table-body tr'));
      let visibleCount = 0;

      function rowMatches(row) {{
        const isPassed = row.dataset.passed === 'true';
        const isFailed = row.dataset.failed === 'true';
        const isPareto = row.dataset.pareto === 'true';
        const isRecommended = row.dataset.recommended === 'true';
        if (passEnabled && !isPassed) return false;
        if (failedEnabled && !isFailed) return false;
        if (paretoEnabled && !isPareto) return false;
        if (recommendedEnabled && !isRecommended) return false;
        return true;
      }}

      resultRows.sort((a, b) => compareRows(a, b, sortValue));
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

      const summary = document.getElementById('filter-summary');
      if (summary) {{
        const activeFilters = [];
        if (passEnabled) activeFilters.push('pass');
        if (failedEnabled) activeFilters.push('failed');
        if (paretoEnabled) activeFilters.push('pareto');
        if (recommendedEnabled) activeFilters.push('recommended');
        summary.textContent = `Showing ${{visibleCount}} / ${{resultRows.length}} result(s)` + (
          activeFilters.length ? ` | Filters: ${{activeFilters.join(', ')}}` : ''
        ) + (
          sortValue ? ` | Sort: ${{sortValue}}` : ''
        );
      }}

      let activeAction = 'reset';
      if (passEnabled && !failedEnabled && !paretoEnabled && !recommendedEnabled && sortValue === 'auc-desc') {{
        activeAction = 'pass';
      }} else if (failedEnabled && !passEnabled && !paretoEnabled && !recommendedEnabled && sortValue === 'auc-asc') {{
        activeAction = 'failed';
      }} else if (!passEnabled && !failedEnabled && paretoEnabled && !recommendedEnabled && sortValue === 'size-ratio-asc') {{
        activeAction = 'pareto';
      }} else if (!passEnabled && !failedEnabled && !paretoEnabled && recommendedEnabled && sortValue === 'size-ratio-asc') {{
        activeAction = 'recommended';
      }}
      updateActionStates(activeAction);
      updateFilterControlStates(passEnabled, failedEnabled, paretoEnabled, recommendedEnabled, sortValue);
    }}

    function applyQuickAction(action) {{
      if (action === 'pass') {{
        setCheckedIfEnabled('filter-pass-only', true);
        setCheckedIfEnabled('filter-failed-only', false);
        setCheckedIfEnabled('filter-pareto-only', false);
        setCheckedIfEnabled('filter-recommended-only', false);
        setSortValue('auc-desc');
      }} else if (action === 'failed') {{
        setCheckedIfEnabled('filter-pass-only', false);
        setCheckedIfEnabled('filter-failed-only', true);
        setCheckedIfEnabled('filter-pareto-only', false);
        setCheckedIfEnabled('filter-recommended-only', false);
        setSortValue('auc-asc');
      }} else if (action === 'pareto') {{
        setCheckedIfEnabled('filter-pass-only', false);
        setCheckedIfEnabled('filter-failed-only', false);
        setCheckedIfEnabled('filter-pareto-only', true);
        setCheckedIfEnabled('filter-recommended-only', false);
        setSortValue('size-ratio-asc');
      }} else if (action === 'recommended') {{
        setCheckedIfEnabled('filter-pass-only', false);
        setCheckedIfEnabled('filter-failed-only', false);
        setCheckedIfEnabled('filter-pareto-only', false);
        setCheckedIfEnabled('filter-recommended-only', true);
        setSortValue('size-ratio-asc');
      }} else {{
        resetFilters();
        return;
      }}
      refreshResultsView();
    }}

    function resetFilters() {{
      const filterIds = [
        'filter-pass-only',
        'filter-failed-only',
        'filter-pareto-only',
        'filter-recommended-only',
      ];
      filterIds.forEach((id) => {{
        const input = document.getElementById(id);
        if (input && !input.disabled) {{
          input.checked = false;
        }}
      }});
      const sortResults = document.getElementById('sort-results');
      if (sortResults) {{
        sortResults.value = 'path-asc';
      }}
      refreshResultsView();
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
    refreshResultsView();
  </script>
</body>
</html>
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def save_batch_report(
    results: list[dict],
    reference_path: str,
    output_path: str,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
    report_title: str = "CloudAnalyzer Batch Evaluation Report",
    report_notes: list[str] | None = None,
) -> None:
    """Write a batch evaluation report based on the file extension."""
    ext = Path(output_path).suffix.lower()
    if ext in {".md", ".markdown"}:
        make_batch_markdown(
            results,
            reference_path,
            output_path,
            min_auc=min_auc,
            max_chamfer=max_chamfer,
            report_title=report_title,
            report_notes=report_notes,
        )
        return
    if ext == ".html":
        make_batch_html(
            results,
            reference_path,
            output_path,
            min_auc=min_auc,
            max_chamfer=max_chamfer,
            report_title=report_title,
            report_notes=report_notes,
        )
        return
    raise ValueError("Unsupported report format. Use .md, .markdown, or .html")
