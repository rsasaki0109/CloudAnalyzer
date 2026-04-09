"""Report generation module (JSON, Markdown, and HTML)."""

import json
from html import escape
from pathlib import Path
from typing import Any, cast

from ca.pareto import mark_quality_size_recommended
from ca.plot import plot_quality_vs_size
from ca.trajectory import plot_trajectory_error_timeline, plot_trajectory_overlay


def make_json(
    source_points: int,
    target_points: int,
    fitness: float | None,
    rmse: float | None,
    distance_stats: dict,
) -> dict:
    """Build JSON-serializable report dict.

    Args:
        source_points: Number of source points.
        target_points: Number of target points.
        fitness: Registration fitness (None if no registration).
        rmse: Registration RMSE (None if no registration).
        distance_stats: Distance summary statistics.

    Returns:
        Report dict.
    """
    data = {
        "source_points": source_points,
        "target_points": target_points,
        "fitness": fitness,
        "rmse": rmse,
        "distance_stats": distance_stats,
    }
    return data


def save_json(data: dict, path: str) -> None:
    """Write report dict to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def make_markdown(data: dict, output_path: str) -> None:
    """Generate Markdown report from report dict.

    Args:
        data: Report dict (from make_json).
        output_path: Path to write .md file.
    """
    lines = [
        "# CloudAnalyzer Report",
        "",
    ]

    if data.get("fitness") is not None:
        lines += [
            "## Registration",
            f"- Fitness: {data['fitness']:.4f}",
            f"- RMSE: {data['rmse']:.4f}",
            "",
        ]

    stats = data.get("distance_stats", {})
    lines += [
        "## Distance Stats",
        f"- Mean: {stats.get('mean', 0):.4f}",
        f"- Median: {stats.get('median', 0):.4f}",
        f"- Max: {stats.get('max', 0):.4f}",
        f"- Min: {stats.get('min', 0):.4f}",
        f"- Std: {stats.get('std', 0):.4f}",
        "",
        "## Point Counts",
        f"- Source: {data['source_points']}",
        f"- Target: {data['target_points']}",
        "",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def _gate_for_item(
    item: dict,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
) -> dict | None:
    """Return existing or derived quality gate metadata for one batch item."""
    if item.get("quality_gate") is not None:
        return cast(dict[str, Any], item["quality_gate"])
    if min_auc is None and max_chamfer is None:
        return None

    reasons = []
    if min_auc is not None and item["auc"] < min_auc:
        reasons.append(f"AUC {item['auc']:.4f} < min_auc {min_auc:.4f}")
    if max_chamfer is not None and item["chamfer_distance"] > max_chamfer:
        reasons.append(
            f"Chamfer {item['chamfer_distance']:.4f} > max_chamfer {max_chamfer:.4f}"
        )
    return {
        "passed": not reasons,
        "min_auc": min_auc,
        "max_chamfer": max_chamfer,
        "reasons": reasons,
    }


def _html_command_block(command: str) -> str:
    """Render one copyable HTML command block."""
    return (
        "<div class=\"command-row\">"
        f"<code>{escape(command)}</code>"
        f"<button type=\"button\" onclick='copyCommand({json.dumps(command)}, this)'>Copy</button>"
        "</div>"
    )


def _html_summary_action(
    action_id: str,
    action: str,
    label: str,
    count: int,
    disabled: bool = False,
    onclick_function: str = "applyQuickAction",
) -> str:
    """Render one summary-row action button."""
    disabled_attr = " disabled" if disabled else ""
    return (
        f"<button type=\"button\" id=\"{escape(action_id)}\" "
        f'class="summary-chip summary-chip-{escape(action)}" '
        f'data-summary-action="{escape(action)}" '
        f'onclick="{escape(onclick_function)}(\'{escape(action)}\')"{disabled_attr}>'
        f"<span>{escape(label)}</span>"
        f"<span class=\"summary-chip-count\">{count}</span>"
        "</button>"
    )


def _compression_for_item(item: dict) -> dict | None:
    """Return compression metadata for one batch item."""
    compression = item.get("compression")
    if compression is None:
        return None
    return cast(dict[str, Any], compression)


def _quality_vs_size_plot_path(report_path: str) -> Path:
    """Return sibling PNG path for the quality-vs-size plot."""
    report = Path(report_path)
    return report.with_name(f"{report.stem}_quality_vs_size.png")


def _trajectory_overlay_plot_path(report_path: str) -> Path:
    """Return sibling PNG path for the trajectory overlay plot."""
    report = Path(report_path)
    return report.with_name(f"{report.stem}_trajectory_overlay.png")


def _trajectory_error_plot_path(report_path: str) -> Path:
    """Return sibling PNG path for the trajectory error plot."""
    report = Path(report_path)
    return report.with_name(f"{report.stem}_trajectory_errors.png")


def _run_f1_plot_path(report_path: str) -> Path:
    """Return sibling PNG path for the run-evaluation F1 plot."""
    report = Path(report_path)
    return report.with_name(f"{report.stem}_map_f1.png")


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
) -> None:
    """Generate a Markdown report for batch evaluation results."""
    summary = make_batch_summary(
        results,
        reference_path,
        min_auc=min_auc,
        max_chamfer=max_chamfer,
    )

    lines = [
        "# CloudAnalyzer Batch Evaluation Report",
        "",
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

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CloudAnalyzer Batch Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #111827; }}
    h1, h2 {{ margin-bottom: 0.5rem; }}
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
  <h1>CloudAnalyzer Batch Evaluation Report</h1>
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
        )
        return
    if ext == ".html":
        make_batch_html(
            results,
            reference_path,
            output_path,
            min_auc=min_auc,
            max_chamfer=max_chamfer,
        )
        return
    raise ValueError("Unsupported report format. Use .md, .markdown, or .html")


def make_ground_markdown(result: dict, output_path: str) -> None:
    """Generate a Markdown report for ground segmentation evaluation."""
    counts = result["counts"]
    cm = result["confusion_matrix"]
    gate = cast(dict[str, Any] | None, result.get("quality_gate"))

    lines = [
        "# CloudAnalyzer Ground Segmentation Report",
        "",
        "## Summary",
        f"- Estimated ground: {result['estimated_ground_path']}",
        f"- Estimated non-ground: {result['estimated_nonground_path']}",
        f"- Reference ground: {result['reference_ground_path']}",
        f"- Reference non-ground: {result['reference_nonground_path']}",
        f"- Voxel size: {result['voxel_size']:.4f}m",
        "",
        "## Metrics",
        f"- Precision: {result['precision']:.4f}",
        f"- Recall: {result['recall']:.4f}",
        f"- F1: {result['f1']:.4f}",
        f"- IoU: {result['iou']:.4f}",
        f"- Accuracy: {result['accuracy']:.4f}",
        "",
        "## Confusion Matrix",
        "",
        "| Estimate \\/ Reference | Ground | Non-ground |",
        "|---|---:|---:|",
        f"| Ground | {cm['tp']} | {cm['fp']} |",
        f"| Non-ground | {cm['fn']} | {cm['tn']} |",
        "",
        "## Point Counts",
        "",
        "| Split | Ground | Non-ground | Total |",
        "|---|---:|---:|---:|",
        (
            f"| Reference | {counts['reference_ground_points']} | "
            f"{counts['reference_nonground_points']} | "
            f"{counts['reference_ground_points'] + counts['reference_nonground_points']} |"
        ),
        (
            f"| Estimated | {counts['estimated_ground_points']} | "
            f"{counts['estimated_nonground_points']} | "
            f"{counts['estimated_ground_points'] + counts['estimated_nonground_points']} |"
        ),
    ]

    if gate is not None:
        lines += [
            "",
            "## Quality Gate",
            f"- Status: {'PASS' if gate['passed'] else 'FAIL'}",
            f"- Min Precision: {_format_optional_float(cast(float | None, gate.get('min_precision')))}",
            f"- Min Recall: {_format_optional_float(cast(float | None, gate.get('min_recall')))}",
            f"- Min F1: {_format_optional_float(cast(float | None, gate.get('min_f1')))}",
            f"- Min IoU: {_format_optional_float(cast(float | None, gate.get('min_iou')))}",
        ]
        if gate["reasons"]:
            lines.append("- Reasons:")
            lines.extend(f"  - {reason}" for reason in gate["reasons"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def make_ground_html(result: dict, output_path: str) -> None:
    """Generate an HTML report for ground segmentation evaluation."""
    counts = result["counts"]
    cm = result["confusion_matrix"]
    gate = cast(dict[str, Any] | None, result.get("quality_gate"))

    summary_rows = [
        ("Estimated ground", result["estimated_ground_path"]),
        ("Estimated non-ground", result["estimated_nonground_path"]),
        ("Reference ground", result["reference_ground_path"]),
        ("Reference non-ground", result["reference_nonground_path"]),
        ("Voxel size", f"{result['voxel_size']:.4f}m"),
    ]
    metrics_rows = [
        ("Precision", f"{result['precision']:.4f}"),
        ("Recall", f"{result['recall']:.4f}"),
        ("F1", f"{result['f1']:.4f}"),
        ("IoU", f"{result['iou']:.4f}"),
        ("Accuracy", f"{result['accuracy']:.4f}"),
    ]
    counts_rows = [
        (
            "Reference",
            str(counts["reference_ground_points"]),
            str(counts["reference_nonground_points"]),
            str(counts["reference_ground_points"] + counts["reference_nonground_points"]),
        ),
        (
            "Estimated",
            str(counts["estimated_ground_points"]),
            str(counts["estimated_nonground_points"]),
            str(counts["estimated_ground_points"] + counts["estimated_nonground_points"]),
        ),
    ]

    summary_html = "\n".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in summary_rows
    )
    metrics_html = "\n".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in metrics_rows
    )
    counts_html = "\n".join(
        (
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td>{escape(ground)}</td>"
            f"<td>{escape(nonground)}</td>"
            f"<td>{escape(total)}</td>"
            "</tr>"
        )
        for label, ground, nonground, total in counts_rows
    )

    gate_html = ""
    if gate is not None:
        gate_rows = [
            ("Status", "PASS" if gate["passed"] else "FAIL"),
            ("Min Precision", _format_optional_float(cast(float | None, gate.get("min_precision")))),
            ("Min Recall", _format_optional_float(cast(float | None, gate.get("min_recall")))),
            ("Min F1", _format_optional_float(cast(float | None, gate.get("min_f1")))),
            ("Min IoU", _format_optional_float(cast(float | None, gate.get("min_iou")))),
        ]
        gate_rows_html = "\n".join(
            f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
            for label, value in gate_rows
        )
        reasons_html = ""
        if gate["reasons"]:
            reasons_html = (
                "<h3>Gate Reasons</h3><ul>"
                + "".join(f"<li>{escape(reason)}</li>" for reason in gate["reasons"])
                + "</ul>"
            )
        gate_html = (
            "<h2>Quality Gate</h2>"
            f"<table><tbody>{gate_rows_html}</tbody></table>"
            f"{reasons_html}"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CloudAnalyzer Ground Segmentation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #111827; }}
    h1, h2, h3 {{ margin-bottom: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f3f4f6; width: 18rem; }}
    tbody tr:nth-child(even) {{ background: #f9fafb; }}
    .matrix th, .matrix td {{ text-align: center; width: auto; }}
    .matrix th:first-child, .matrix td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>CloudAnalyzer Ground Segmentation Report</h1>

  <h2>Summary</h2>
  <table><tbody>{summary_html}</tbody></table>

  <h2>Metrics</h2>
  <table><tbody>{metrics_html}</tbody></table>

  <h2>Confusion Matrix</h2>
  <table class="matrix">
    <thead>
      <tr><th>Estimate / Reference</th><th>Ground</th><th>Non-ground</th></tr>
    </thead>
    <tbody>
      <tr><td>Ground</td><td>{cm['tp']}</td><td>{cm['fp']}</td></tr>
      <tr><td>Non-ground</td><td>{cm['fn']}</td><td>{cm['tn']}</td></tr>
    </tbody>
  </table>

  <h2>Point Counts</h2>
  <table>
    <thead>
      <tr><th>Split</th><th>Ground</th><th>Non-ground</th><th>Total</th></tr>
    </thead>
    <tbody>{counts_html}</tbody>
  </table>

  {gate_html}
</body>
</html>
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def save_ground_report(result: dict, output_path: str) -> None:
    """Write a ground evaluation report based on the file extension."""
    ext = Path(output_path).suffix.lower()
    if ext in {".md", ".markdown"}:
        make_ground_markdown(result, output_path)
        return
    if ext == ".html":
        make_ground_html(result, output_path)
        return
    raise ValueError("Unsupported report format. Use .md, .markdown, or .html")


def _format_optional_float(value: float | None) -> str:
    """Format a float or render n/a."""
    return f"{value:.4f}" if value is not None else "n/a"


def _format_optional_ratio(value: float | None) -> str:
    """Format a ratio as a percentage or render n/a."""
    return f"{value:.1%}" if value is not None else "n/a"


def _trajectory_gate_for_item(
    item: dict,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> dict | None:
    """Return existing or derived trajectory quality gate metadata for one batch item."""
    if item.get("quality_gate") is not None:
        return cast(dict[str, Any], item["quality_gate"])
    if max_ate is None and max_rpe is None and max_drift is None and min_coverage is None:
        return None

    reasons = []
    if max_ate is not None and item["ate"]["rmse"] > max_ate:
        reasons.append(f"ATE RMSE {item['ate']['rmse']:.4f} > max_ate {max_ate:.4f}")
    if max_rpe is not None and item["rpe_translation"]["rmse"] > max_rpe:
        reasons.append(f"RPE RMSE {item['rpe_translation']['rmse']:.4f} > max_rpe {max_rpe:.4f}")
    if max_drift is not None and item["drift"]["endpoint"] > max_drift:
        reasons.append(
            f"Endpoint Drift {item['drift']['endpoint']:.4f} > max_drift {max_drift:.4f}"
        )
    if min_coverage is not None and item["coverage_ratio"] < min_coverage:
        reasons.append(
            f"Coverage {item['coverage_ratio']:.1%} < min_coverage {min_coverage:.1%}"
        )
    return {
        "passed": not reasons,
        "max_ate": max_ate,
        "max_rpe": max_rpe,
        "max_drift": max_drift,
        "min_coverage": min_coverage,
        "reasons": reasons,
    }


_TRAJECTORY_LOW_COVERAGE_THRESHOLD = 0.95


def make_trajectory_markdown(result: dict, output_path: str) -> None:
    """Generate a Markdown report for trajectory evaluation."""
    alignment = result["alignment"]
    matching = result["matching"]
    ate = result["ate"]
    rpe = result["rpe_translation"]
    drift = result["drift"]
    gate = result.get("quality_gate")
    overlay_plot_path = _trajectory_overlay_plot_path(output_path)
    error_plot_path = _trajectory_error_plot_path(output_path)
    plot_trajectory_overlay(result, str(overlay_plot_path))
    plot_trajectory_error_timeline(result, str(error_plot_path))

    lines = [
        "# CloudAnalyzer Trajectory Evaluation Report",
        "",
        "## Summary",
        f"- Estimated: {result['estimated_path']}",
        f"- Reference: {result['reference_path']}",
        f"- Estimated poses: {matching['estimated_poses']}",
        f"- Reference poses: {matching['reference_poses']}",
        f"- Matched poses: {matching['matched_poses']} ({matching['coverage_ratio']:.1%})",
        f"- Alignment: {alignment['mode']}",
        (
            f"- Alignment Translation: [{alignment['translation'][0]:.4f}, "
            f"{alignment['translation'][1]:.4f}, {alignment['translation'][2]:.4f}]"
        ),
        f"- Max time delta: {matching['max_time_delta']:.4f}s",
        f"- Mean abs time delta: {matching['mean_abs_time_delta']:.4f}s",
        f"- Max abs time delta: {matching['max_abs_time_delta']:.4f}s",
        f"- Matched duration: {matching['matched_duration']:.4f}s",
        f"- Duration coverage: {matching['duration_coverage_ratio']:.1%}",
        "",
        "## Absolute Trajectory Error (ATE)",
        f"- RMSE: {ate['rmse']:.4f}",
        f"- Mean: {ate['mean']:.4f}",
        f"- Median: {ate['median']:.4f}",
        f"- Min: {ate['min']:.4f}",
        f"- Max: {ate['max']:.4f}",
        f"- Std: {ate['std']:.4f}",
        "",
        "## Relative Pose Error (Translation)",
        f"- RMSE: {rpe['rmse']:.4f}",
        f"- Mean: {rpe['mean']:.4f}",
        f"- Median: {rpe['median']:.4f}",
        f"- Min: {rpe['min']:.4f}",
        f"- Max: {rpe['max']:.4f}",
        f"- Std: {rpe['std']:.4f}",
        "",
        "## Drift",
        f"- Endpoint Drift: {drift['endpoint']:.4f}",
        f"- Reference Path Length: {drift['reference_path_length']:.4f}",
        f"- Estimated Path Length: {drift['estimated_path_length']:.4f}",
        f"- Path Length Ratio: {_format_optional_float(drift['path_length_ratio'])}",
        f"- Drift Ratio: {_format_optional_float(drift['ratio_to_reference_path_length'])}",
        "",
        "## Visualizations",
        "",
        f"![Trajectory Overlay]({overlay_plot_path.name})",
        "",
        f"![Trajectory Errors]({error_plot_path.name})",
    ]

    if gate is not None:
        status = "PASS" if gate["passed"] else "FAIL"
        lines += [
            "",
            "## Quality Gate",
            f"- Max ATE RMSE: {_format_optional_float(gate['max_ate'])}",
            f"- Max RPE RMSE: {_format_optional_float(gate['max_rpe'])}",
            f"- Max Drift: {_format_optional_float(gate['max_drift'])}",
            f"- Min Coverage: {_format_optional_ratio(gate['min_coverage'])}",
            f"- Status: {status}",
        ]
        if gate["reasons"]:
            lines.append("- Reasons:")
            for reason in gate["reasons"]:
                lines.append(f"  - {reason}")

    lines += [
        "",
        "## Worst ATE Samples",
        "",
        "| Timestamp | Position Error | Time Delta |",
        "|---:|---:|---:|",
    ]
    for sample in result["worst_ate_samples"]:
        lines.append(
            f"| {sample['timestamp']:.4f} | {sample['position_error']:.4f} | {sample['time_delta']:.4f} |"
        )

    lines += [
        "",
        "## Worst RPE Segments",
        "",
        "| Start | End | Translation Error |",
        "|---:|---:|---:|",
    ]
    for segment in result["worst_rpe_segments"]:
        lines.append(
            f"| {segment['start_timestamp']:.4f} | {segment['end_timestamp']:.4f} | {segment['translation_error']:.4f} |"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def make_trajectory_html(result: dict, output_path: str) -> None:
    """Generate an HTML report for trajectory evaluation."""
    alignment = result["alignment"]
    matching = result["matching"]
    ate = result["ate"]
    rpe = result["rpe_translation"]
    drift = result["drift"]
    gate = result.get("quality_gate")
    overlay_plot_path = _trajectory_overlay_plot_path(output_path)
    error_plot_path = _trajectory_error_plot_path(output_path)
    plot_trajectory_overlay(result, str(overlay_plot_path))
    plot_trajectory_error_timeline(result, str(error_plot_path))
    gate_status = None
    if gate is not None:
        gate_status = "PASS" if gate["passed"] else "FAIL"
    gate_status_html = ""
    if gate is not None:
        gate_class = "status-pass" if gate["passed"] else "status-fail"
        gate_status_html = (
            f'<p class="{gate_class}">Quality Gate: {escape(gate_status or "")}</p>'
        )

    summary_rows = [
        ("Estimated", result["estimated_path"]),
        ("Reference", result["reference_path"]),
        ("Estimated poses", str(matching["estimated_poses"])),
        ("Reference poses", str(matching["reference_poses"])),
        ("Matched poses", f"{matching['matched_poses']} ({matching['coverage_ratio']:.1%})"),
        ("Alignment", alignment["mode"]),
        (
            "Alignment Translation",
            f"[{alignment['translation'][0]:.4f}, {alignment['translation'][1]:.4f}, {alignment['translation'][2]:.4f}]",
        ),
        ("Max time delta", f"{matching['max_time_delta']:.4f}s"),
        ("Mean abs time delta", f"{matching['mean_abs_time_delta']:.4f}s"),
        ("Max abs time delta", f"{matching['max_abs_time_delta']:.4f}s"),
        ("Matched duration", f"{matching['matched_duration']:.4f}s"),
        ("Duration coverage", f"{matching['duration_coverage_ratio']:.1%}"),
        ("ATE RMSE", f"{ate['rmse']:.4f}"),
        ("RPE RMSE", f"{rpe['rmse']:.4f}"),
        ("Endpoint Drift", f"{drift['endpoint']:.4f}"),
        ("Path Length Ratio", _format_optional_float(drift["path_length_ratio"])),
        ("Drift Ratio", _format_optional_float(drift["ratio_to_reference_path_length"])),
    ]
    if gate is not None:
        summary_rows += [
            ("Max ATE RMSE", _format_optional_float(gate["max_ate"])),
            ("Max RPE RMSE", _format_optional_float(gate["max_rpe"])),
            ("Max Drift", _format_optional_float(gate["max_drift"])),
            ("Min Coverage", _format_optional_ratio(gate["min_coverage"])),
            ("Quality Gate", gate_status or "n/a"),
        ]

    summary_html = "\n".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in summary_rows
    )
    gate_reasons_html = ""
    if gate is not None and gate["reasons"]:
        gate_reasons_html = (
            "<h2>Gate Reasons</h2><ul>"
            + "".join(f"<li>{escape(reason)}</li>" for reason in gate["reasons"])
            + "</ul>"
        )
    ate_rows_html = "\n".join(
        (
            "<tr>"
            f"<td>{sample['timestamp']:.4f}</td>"
            f"<td>{sample['position_error']:.4f}</td>"
            f"<td>{sample['time_delta']:.4f}</td>"
            "</tr>"
        )
        for sample in result["worst_ate_samples"]
    )
    rpe_rows_html = "\n".join(
        (
            "<tr>"
            f"<td>{segment['start_timestamp']:.4f}</td>"
            f"<td>{segment['end_timestamp']:.4f}</td>"
            f"<td>{segment['translation_error']:.4f}</td>"
            "</tr>"
        )
        for segment in result["worst_rpe_segments"]
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CloudAnalyzer Trajectory Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #111827; }}
    h1, h2 {{ margin-bottom: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f3f4f6; width: 20rem; }}
    tbody tr:nth-child(even) {{ background: #f9fafb; }}
    .status-pass {{ color: #166534; font-weight: 700; }}
    .status-fail {{ color: #991b1b; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>CloudAnalyzer Trajectory Evaluation Report</h1>
  <h2>Summary</h2>
  <table>
    <tbody>
      {summary_html}
    </tbody>
  </table>
  {gate_status_html}
  {gate_reasons_html}

  <h2>Visualizations</h2>
  <p><img src="{escape(overlay_plot_path.name)}" alt="Trajectory overlay plot" style="max-width: 100%; height: auto;"></p>
  <p><img src="{escape(error_plot_path.name)}" alt="Trajectory error plot" style="max-width: 100%; height: auto;"></p>

  <h2>Worst ATE Samples</h2>
  <table>
    <thead>
      <tr>
        <th>Timestamp</th>
        <th>Position Error</th>
        <th>Time Delta</th>
      </tr>
    </thead>
    <tbody>
      {ate_rows_html}
    </tbody>
  </table>

  <h2>Worst RPE Segments</h2>
  <table>
    <thead>
      <tr>
        <th>Start</th>
        <th>End</th>
        <th>Translation Error</th>
      </tr>
    </thead>
    <tbody>
      {rpe_rows_html}
    </tbody>
  </table>
</body>
</html>
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def save_trajectory_report(result: dict, output_path: str) -> None:
    """Write a trajectory evaluation report based on the file extension."""
    ext = Path(output_path).suffix.lower()
    if ext in {".md", ".markdown"}:
        make_trajectory_markdown(result, output_path)
        return
    if ext == ".html":
        make_trajectory_html(result, output_path)
        return
    raise ValueError("Unsupported report format. Use .md, .markdown, or .html")


def make_trajectory_batch_summary(
    results: list[dict],
    reference_dir: str,
    max_ate: float | None = None,
    max_rpe: float | None = None,
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
            (item, _trajectory_gate_for_item(item, max_ate, max_rpe, max_drift, min_coverage))
            for item in results
        ]
        quality_gate = {
            "max_ate": max_ate,
            "max_rpe": max_rpe,
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

    return {
        "reference_dir": reference_dir,
        "total_files": len(results),
        "mean_ate_rmse": float(sum(item["ate"]["rmse"] for item in results) / len(results)),
        "mean_rpe_rmse": float(
            sum(item["rpe_translation"]["rmse"] for item in results) / len(results)
        ),
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
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> None:
    """Generate a Markdown report for trajectory batch evaluation results."""
    summary = make_trajectory_batch_summary(
        results,
        reference_dir,
        max_ate=max_ate,
        max_rpe=max_rpe,
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
            f"- Max Drift: {_format_optional_float(gate['max_drift'])}",
            f"- Min Coverage: {_format_optional_ratio(gate['min_coverage'])}",
            f"- Pass: {gate['pass_count']}",
            f"- Fail: {gate['fail_count']}",
        ]

    lines += [
        "",
        "## Results",
        "",
        "| Path | Matched | Coverage | ATE RMSE | RPE RMSE | Drift | Alignment |"
        + (" Status |" if gate is not None else ""),
        "|---|---:|---:|---:|---:|---:|---|" + ("---|" if gate is not None else ""),
    ]

    for item in summary["results"]:
        item_gate = _trajectory_gate_for_item(
            item,
            max_ate=max_ate,
            max_rpe=max_rpe,
            max_drift=max_drift,
            min_coverage=min_coverage,
        )
        row = [
            item["path"],
            str(item["matched_poses"]),
            f"{item['coverage_ratio']:.1%}",
            f"{item['ate']['rmse']:.4f}",
            f"{item['rpe_translation']['rmse']:.4f}",
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
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> None:
    """Generate an HTML report for trajectory batch evaluation results."""
    summary = make_trajectory_batch_summary(
        results,
        reference_dir,
        max_ate=max_ate,
        max_rpe=max_rpe,
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
        item_gate = _trajectory_gate_for_item(
            item,
            max_ate=max_ate,
            max_rpe=max_rpe,
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
            max_drift=max_drift,
            min_coverage=min_coverage,
        )
        return
    raise ValueError("Unsupported report format. Use .md, .markdown, or .html")


def make_run_markdown(result: dict, output_path: str) -> None:
    """Generate a Markdown report for combined map + trajectory evaluation."""
    from ca.evaluate import plot_f1_curve

    map_result = result["map"]
    trajectory_result = result["trajectory"]
    overall_gate = result.get("overall_quality_gate")
    map_gate = map_result.get("quality_gate")
    trajectory_gate = trajectory_result.get("quality_gate")
    best_f1 = map_result["best_f1"]
    alignment = trajectory_result["alignment"]
    matching = trajectory_result["matching"]
    drift = trajectory_result["drift"]
    f1_plot_path = _run_f1_plot_path(output_path)
    overlay_plot_path = _trajectory_overlay_plot_path(output_path)
    error_plot_path = _trajectory_error_plot_path(output_path)
    plot_f1_curve(map_result, str(f1_plot_path))
    plot_trajectory_overlay(trajectory_result, str(overlay_plot_path))
    plot_trajectory_error_timeline(trajectory_result, str(error_plot_path))

    lines = [
        "# CloudAnalyzer Run Evaluation Report",
        "",
        "## Overall Summary",
        f"- Map: {map_result['source_path']} vs {map_result['target_path']}",
        f"- Trajectory: {trajectory_result['estimated_path']} vs {trajectory_result['reference_path']}",
        f"- Map AUC: {map_result['auc']:.4f}",
        f"- Trajectory ATE RMSE: {trajectory_result['ate']['rmse']:.4f}",
        f"- Trajectory Drift: {drift['endpoint']:.4f}",
    ]
    if overall_gate is not None:
        lines += [
            f"- Overall Quality Gate: {'PASS' if overall_gate['passed'] else 'FAIL'}",
        ]
        if overall_gate["reasons"]:
            lines.append("- Overall Gate Reasons:")
            for reason in overall_gate["reasons"]:
                lines.append(f"  - {reason}")

    lines += [
        "",
        "## Map Quality",
        f"- Source points: {map_result['source_points']}",
        f"- Reference points: {map_result['target_points']}",
        f"- Chamfer Distance: {map_result['chamfer_distance']:.4f}",
        f"- Hausdorff Distance: {map_result['hausdorff_distance']:.4f}",
        f"- AUC (F1): {map_result['auc']:.4f}",
        f"- Best F1: {best_f1['f1']:.4f} @ d={best_f1['threshold']:.2f}",
    ]
    if map_gate is not None:
        lines += [
            f"- Map Quality Gate: {'PASS' if map_gate['passed'] else 'FAIL'}",
            f"- Min AUC: {_format_optional_float(map_gate['min_auc'])}",
            f"- Max Chamfer: {_format_optional_float(map_gate['max_chamfer'])}",
        ]
        if map_gate["reasons"]:
            lines.append("- Map Gate Reasons:")
            for reason in map_gate["reasons"]:
                lines.append(f"  - {reason}")

    lines += [
        "",
        "## Trajectory Quality",
        f"- Estimated poses: {matching['estimated_poses']}",
        f"- Reference poses: {matching['reference_poses']}",
        f"- Matched poses: {matching['matched_poses']} ({matching['coverage_ratio']:.1%})",
        f"- Alignment: {alignment['mode']}",
        f"- ATE RMSE: {trajectory_result['ate']['rmse']:.4f}",
        f"- RPE RMSE: {trajectory_result['rpe_translation']['rmse']:.4f}",
        f"- Endpoint Drift: {drift['endpoint']:.4f}",
    ]
    if trajectory_gate is not None:
        lines += [
            f"- Trajectory Quality Gate: {'PASS' if trajectory_gate['passed'] else 'FAIL'}",
            f"- Max ATE RMSE: {_format_optional_float(trajectory_gate['max_ate'])}",
            f"- Max RPE RMSE: {_format_optional_float(trajectory_gate['max_rpe'])}",
            f"- Max Drift: {_format_optional_float(trajectory_gate['max_drift'])}",
            f"- Min Coverage: {_format_optional_ratio(trajectory_gate['min_coverage'])}",
        ]
        if trajectory_gate["reasons"]:
            lines.append("- Trajectory Gate Reasons:")
            for reason in trajectory_gate["reasons"]:
                lines.append(f"  - {reason}")

    lines += [
        "",
        "## Visualizations",
        "",
        f"![Map F1 Curve]({f1_plot_path.name})",
        "",
        f"![Trajectory Overlay]({overlay_plot_path.name})",
        "",
        f"![Trajectory Errors]({error_plot_path.name})",
        "",
        "## Inspection Commands",
        "",
        f"- Run viewer: `{result['inspect']['run_web']}`",
        f"- Map heatmap: `{result['inspect']['map']['web_heatmap']}`",
        f"- Map snapshot: `{result['inspect']['map']['heatmap3d']}`",
        f"- Trajectory: `{result['inspect']['trajectory']['traj_evaluate']}`",
        f"- Trajectory aligned: `{result['inspect']['trajectory']['traj_evaluate_aligned']}`",
        f"- Trajectory rigid: `{result['inspect']['trajectory']['traj_evaluate_rigid']}`",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def make_run_html(result: dict, output_path: str) -> None:
    """Generate an HTML report for combined map + trajectory evaluation."""
    from ca.evaluate import plot_f1_curve

    map_result = result["map"]
    trajectory_result = result["trajectory"]
    overall_gate = result.get("overall_quality_gate")
    map_gate = map_result.get("quality_gate")
    trajectory_gate = trajectory_result.get("quality_gate")
    best_f1 = map_result["best_f1"]
    alignment = trajectory_result["alignment"]
    matching = trajectory_result["matching"]
    drift = trajectory_result["drift"]
    f1_plot_path = _run_f1_plot_path(output_path)
    overlay_plot_path = _trajectory_overlay_plot_path(output_path)
    error_plot_path = _trajectory_error_plot_path(output_path)
    plot_f1_curve(map_result, str(f1_plot_path))
    plot_trajectory_overlay(trajectory_result, str(overlay_plot_path))
    plot_trajectory_error_timeline(trajectory_result, str(error_plot_path))

    summary_rows = [
        ("Map", f"{map_result['source_path']} vs {map_result['target_path']}"),
        (
            "Trajectory",
            f"{trajectory_result['estimated_path']} vs {trajectory_result['reference_path']}",
        ),
        ("Map AUC", f"{map_result['auc']:.4f}"),
        ("Trajectory ATE RMSE", f"{trajectory_result['ate']['rmse']:.4f}"),
        ("Trajectory Drift", f"{drift['endpoint']:.4f}"),
    ]
    if overall_gate is not None:
        summary_rows.append(
            ("Overall Quality Gate", "PASS" if overall_gate["passed"] else "FAIL")
        )
    summary_html = "\n".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in summary_rows
    )

    overall_reasons_html = ""
    if overall_gate is not None and overall_gate["reasons"]:
        overall_reasons_html = (
            "<h2>Overall Gate Reasons</h2><ul>"
            + "".join(f"<li>{escape(reason)}</li>" for reason in overall_gate["reasons"])
            + "</ul>"
        )

    map_rows = [
        ("Source points", str(map_result["source_points"])),
        ("Reference points", str(map_result["target_points"])),
        ("Chamfer Distance", f"{map_result['chamfer_distance']:.4f}"),
        ("Hausdorff Distance", f"{map_result['hausdorff_distance']:.4f}"),
        ("AUC (F1)", f"{map_result['auc']:.4f}"),
        ("Best F1", f"{best_f1['f1']:.4f} @ d={best_f1['threshold']:.2f}"),
    ]
    if map_gate is not None:
        map_rows += [
            ("Map Quality Gate", "PASS" if map_gate["passed"] else "FAIL"),
            ("Min AUC", _format_optional_float(map_gate["min_auc"])),
            ("Max Chamfer", _format_optional_float(map_gate["max_chamfer"])),
        ]
    map_html = "\n".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in map_rows
    )

    map_reasons_html = ""
    if map_gate is not None and map_gate["reasons"]:
        map_reasons_html = (
            "<h3>Map Gate Reasons</h3><ul>"
            + "".join(f"<li>{escape(reason)}</li>" for reason in map_gate["reasons"])
            + "</ul>"
        )

    trajectory_rows = [
        ("Estimated poses", str(matching["estimated_poses"])),
        ("Reference poses", str(matching["reference_poses"])),
        ("Matched poses", f"{matching['matched_poses']} ({matching['coverage_ratio']:.1%})"),
        ("Alignment", alignment["mode"]),
        ("ATE RMSE", f"{trajectory_result['ate']['rmse']:.4f}"),
        ("RPE RMSE", f"{trajectory_result['rpe_translation']['rmse']:.4f}"),
        ("Endpoint Drift", f"{drift['endpoint']:.4f}"),
    ]
    if trajectory_gate is not None:
        trajectory_rows += [
            ("Trajectory Quality Gate", "PASS" if trajectory_gate["passed"] else "FAIL"),
            ("Max ATE RMSE", _format_optional_float(trajectory_gate["max_ate"])),
            ("Max RPE RMSE", _format_optional_float(trajectory_gate["max_rpe"])),
            ("Max Drift", _format_optional_float(trajectory_gate["max_drift"])),
            ("Min Coverage", _format_optional_ratio(trajectory_gate["min_coverage"])),
        ]
    trajectory_html = "\n".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in trajectory_rows
    )

    trajectory_reasons_html = ""
    if trajectory_gate is not None and trajectory_gate["reasons"]:
        trajectory_reasons_html = (
            "<h3>Trajectory Gate Reasons</h3><ul>"
            + "".join(f"<li>{escape(reason)}</li>" for reason in trajectory_gate["reasons"])
            + "</ul>"
        )

    inspection_html = "".join(
        [
            "<h2>Inspection Commands</h2>",
            _html_command_block(result["inspect"]["run_web"]),
            _html_command_block(result["inspect"]["map"]["web_heatmap"]),
            _html_command_block(result["inspect"]["map"]["heatmap3d"]),
            _html_command_block(result["inspect"]["trajectory"]["traj_evaluate"]),
            _html_command_block(result["inspect"]["trajectory"]["traj_evaluate_aligned"]),
            _html_command_block(result["inspect"]["trajectory"]["traj_evaluate_rigid"]),
        ]
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CloudAnalyzer Run Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #111827; }}
    h1, h2, h3 {{ margin-bottom: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f3f4f6; width: 20rem; }}
    tbody tr:nth-child(even) {{ background: #f9fafb; }}
    .command-row {{ display: flex; align-items: flex-start; gap: 0.5rem; margin: 0.4rem 0; }}
    .command-row code {{ flex: 1; white-space: pre-wrap; word-break: break-word; }}
    button {{
      padding: 0.3rem 0.55rem; border: 1px solid #cbd5e1;
      border-radius: 6px; background: #ffffff; cursor: pointer;
    }}
  </style>
</head>
<body>
  <h1>CloudAnalyzer Run Evaluation Report</h1>
  <h2>Overall Summary</h2>
  <table><tbody>{summary_html}</tbody></table>
  {overall_reasons_html}

  <h2>Map Quality</h2>
  <table><tbody>{map_html}</tbody></table>
  {map_reasons_html}

  <h2>Trajectory Quality</h2>
  <table><tbody>{trajectory_html}</tbody></table>
  {trajectory_reasons_html}

  <h2>Visualizations</h2>
  <p><img src="{escape(f1_plot_path.name)}" alt="Map F1 curve" style="max-width: 100%; height: auto;"></p>
  <p><img src="{escape(overlay_plot_path.name)}" alt="Trajectory overlay plot" style="max-width: 100%; height: auto;"></p>
  <p><img src="{escape(error_plot_path.name)}" alt="Trajectory error plot" style="max-width: 100%; height: auto;"></p>

  {inspection_html}

  <script>
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
  </script>
</body>
</html>
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def save_run_report(result: dict, output_path: str) -> None:
    """Write a combined run-evaluation report based on the file extension."""
    ext = Path(output_path).suffix.lower()
    if ext in {".md", ".markdown"}:
        make_run_markdown(result, output_path)
        return
    if ext == ".html":
        make_run_html(result, output_path)
        return
    raise ValueError("Unsupported report format. Use .md, .markdown, or .html")


def _run_batch_gate_for_item(
    item: dict,
    min_auc: float | None = None,
    max_chamfer: float | None = None,
    max_ate: float | None = None,
    max_rpe: float | None = None,
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
        "drift": item["trajectory"]["drift"],
        "coverage_ratio": item["trajectory"]["matching"]["coverage_ratio"],
        "quality_gate": item["trajectory"].get("quality_gate"),
    }
    map_gate = _gate_for_item(map_item, min_auc=min_auc, max_chamfer=max_chamfer)
    trajectory_gate = _trajectory_gate_for_item(
        trajectory_item,
        max_ate=max_ate,
        max_rpe=max_rpe,
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
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> dict:
    """Build summary data for combined run-batch evaluation results."""
    gate_enabled = (
        min_auc is not None
        or max_chamfer is not None
        or max_ate is not None
        or max_rpe is not None
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
        "| ID | Map AUC | Map Chamfer | Traj ATE | Traj RPE | Traj Drift | Coverage |"
        + (" Map Status | Trajectory Status | Overall |" if gate is not None else ""),
        "|---|---:|---:|---:|---:|---:|---:|" + ("---|---|---|" if gate is not None else ""),
    ]
    for item in summary["results"]:
        item_gate = _run_batch_gate_for_item(
            item,
            min_auc=min_auc,
            max_chamfer=max_chamfer,
            max_ate=max_ate,
            max_rpe=max_rpe,
            max_drift=max_drift,
            min_coverage=min_coverage,
        )
        row = [
            item["id"],
            f"{item['map']['auc']:.4f}",
            f"{item['map']['chamfer_distance']:.4f}",
            f"{item['trajectory']['ate']['rmse']:.4f}",
            f"{item['trajectory']['rpe_translation']['rmse']:.4f}",
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
        item_gate = _run_batch_gate_for_item(
            item,
            min_auc=min_auc,
            max_chamfer=max_chamfer,
            max_ate=max_ate,
            max_rpe=max_rpe,
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
            max_drift=max_drift,
            min_coverage=min_coverage,
        )
        return
    raise ValueError("Unsupported report format. Use .md, .markdown, or .html")
