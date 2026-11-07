"""Ground segmentation reports."""

from html import escape
from pathlib import Path
from typing import Any, cast
from ca.reports.common import _format_optional_float


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
