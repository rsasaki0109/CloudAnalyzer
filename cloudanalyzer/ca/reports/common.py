"""Report helpers shared by every report kind: JSON, Markdown, gates and plot paths."""

import json
from html import escape
from pathlib import Path
from typing import Any, cast


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
    onclick_command = escape(json.dumps(command), quote=True)
    return (
        "<div class=\"command-row\">"
        f"<code>{escape(command)}</code>"
        f"<button type=\"button\" onclick='copyCommand({onclick_command}, this)'>Copy</button>"
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


def _format_optional_float(value: float | None) -> str:
    """Format a float or render n/a."""
    return f"{value:.4f}" if value is not None else "n/a"
