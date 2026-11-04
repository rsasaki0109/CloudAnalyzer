"""Trajectory evaluation reports."""

from html import escape
from pathlib import Path
from typing import Any, cast
from ca.trajectory import plot_trajectory_error_timeline, plot_trajectory_overlay
from ca.reports.common import (
    _format_optional_float,
    _trajectory_error_plot_path,
    _trajectory_overlay_plot_path,
)


def _format_optional_ratio(value: float | None) -> str:
    """Format a ratio as a percentage or render n/a."""
    return f"{value:.1%}" if value is not None else "n/a"


def _trajectory_gate_for_item(
    item: dict,
    max_ate: float | None = None,
    max_rpe: float | None = None,
    max_rpe_rotation_deg: float | None = None,
    max_drift: float | None = None,
    min_coverage: float | None = None,
) -> dict | None:
    """Return existing or derived trajectory quality gate metadata for one batch item."""
    if item.get("quality_gate") is not None:
        return cast(dict[str, Any], item["quality_gate"])
    if (
        max_ate is None
        and max_rpe is None
        and max_rpe_rotation_deg is None
        and max_drift is None
        and min_coverage is None
    ):
        return None

    reasons = []
    if max_ate is not None and item["ate"]["rmse"] > max_ate:
        reasons.append(f"ATE RMSE {item['ate']['rmse']:.4f} > max_ate {max_ate:.4f}")
    if max_rpe is not None and item["rpe_translation"]["rmse"] > max_rpe:
        reasons.append(f"RPE RMSE {item['rpe_translation']['rmse']:.4f} > max_rpe {max_rpe:.4f}")
    if max_rpe_rotation_deg is not None:
        rpe_rotation = item.get("rpe_rotation")
        if rpe_rotation is None:
            reasons.append("RPE rotation unavailable: quaternion orientations are required")
        elif rpe_rotation["rmse"] > max_rpe_rotation_deg:
            reasons.append(
                f"RPE rotation RMSE {rpe_rotation['rmse']:.4f} deg > "
                f"max_rpe_rotation_deg {max_rpe_rotation_deg:.4f} deg"
            )
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
        "max_rpe_rotation_deg": max_rpe_rotation_deg,
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
    ate_rotation = result.get("ate_rotation")
    rpe_rotation = result.get("rpe_rotation")
    distance_rpe = result.get("rpe_distance", [])
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
    ]

    if ate_rotation is not None:
        lines += [
            "",
            "## Absolute Trajectory Error (Rotation)",
            f"- RMSE: {ate_rotation['rmse']:.4f} deg",
            f"- Mean: {ate_rotation['mean']:.4f} deg",
            f"- Max: {ate_rotation['max']:.4f} deg",
        ]
    if rpe_rotation is not None:
        lines += [
            "",
            "## Relative Pose Error (Rotation)",
            f"- RMSE: {rpe_rotation['rmse']:.4f} deg",
            f"- Mean: {rpe_rotation['mean']:.4f} deg",
            f"- Max: {rpe_rotation['max']:.4f} deg",
        ]
    if distance_rpe:
        lines += [
            "",
            "## Distance-based Relative Pose Error",
            "",
            "| Distance | Pairs | Translation RMSE | Translation % RMSE | Rotation RMSE |",
            "|---:|---:|---:|---:|---:|",
        ]
        for distance_result in distance_rpe:
            translation = distance_result.get("translation")
            translation_percent = distance_result.get("translation_percent")
            rotation = distance_result.get("rotation")
            lines.append(
                "| "
                f"{distance_result['delta_m']:.1f}m | {distance_result['pairs']} | "
                f"{_format_optional_float(translation['rmse'] if translation else None)} | "
                f"{_format_optional_float(translation_percent['rmse'] if translation_percent else None)} | "
                f"{_format_optional_float(rotation['rmse'] if rotation else None)} |"
            )

    lines += [
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
            f"- Max RPE Rotation: {_format_optional_float(gate.get('max_rpe_rotation_deg'))} deg",
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
        "| Start | End | Translation Error | Rotation Error (deg) |",
        "|---:|---:|---:|---:|",
    ]
    for segment in result["worst_rpe_segments"]:
        lines.append(
            f"| {segment['start_timestamp']:.4f} | {segment['end_timestamp']:.4f} | "
            f"{segment['translation_error']:.4f} | "
            f"{_format_optional_float(segment.get('rotation_error_deg'))} |"
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
    ate_rotation = result.get("ate_rotation")
    rpe_rotation = result.get("rpe_rotation")
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
    if ate_rotation is not None:
        summary_rows.append(("ATE Rotation RMSE", f"{ate_rotation['rmse']:.4f} deg"))
    if rpe_rotation is not None:
        summary_rows.append(("RPE Rotation RMSE", f"{rpe_rotation['rmse']:.4f} deg"))
    if gate is not None:
        summary_rows += [
            ("Max ATE RMSE", _format_optional_float(gate["max_ate"])),
            ("Max RPE RMSE", _format_optional_float(gate["max_rpe"])),
            (
                "Max RPE Rotation",
                f"{_format_optional_float(gate.get('max_rpe_rotation_deg'))} deg",
            ),
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
            f"<td>{_format_optional_float(segment.get('rotation_error_deg'))}</td>"
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
        <th>Rotation Error (deg)</th>
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
