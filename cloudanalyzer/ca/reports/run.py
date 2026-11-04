"""Single-run (map and trajectory) evaluation reports."""

from html import escape
from pathlib import Path
from ca.trajectory import plot_trajectory_error_timeline, plot_trajectory_overlay
from ca.reports.common import (
    _format_optional_float,
    _html_command_block,
    _run_f1_plot_path,
    _trajectory_error_plot_path,
    _trajectory_overlay_plot_path,
)
from ca.reports.trajectory import _format_optional_ratio


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
