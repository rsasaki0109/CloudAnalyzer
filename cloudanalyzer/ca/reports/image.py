"""Image and rendered-view evaluation reports."""

from html import escape
from pathlib import Path
from typing import Any, cast
from ca.reports.common import _format_optional_float


def _format_optional_metric(value: object, *, unit: str = "") -> str:
    """Format a possibly-None aggregate (mean/median/min/max) for a report."""
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}{unit}"
    return "--"


def make_image_markdown(result: dict, output_path: str) -> None:
    """Generate a Markdown report for a photometric (PSNR/SSIM) image check."""
    summary = result["summary"]
    metadata = result.get("metadata", {})
    gate = cast(dict[str, Any] | None, result.get("quality_gate"))
    metrics = cast(list, metadata.get("metrics", ["psnr", "ssim"]))

    lines = [
        "# CloudAnalyzer Image Evaluation Report",
        "",
        "## Summary",
        f"- Rendered dir: {metadata.get('rendered_dir', '')}",
        f"- Reference dir: {metadata.get('reference_dir', '')}",
        f"- Pairs evaluated: {summary['pairs_evaluated']}",
        f"- Missing in reference: {summary['pairs_missing_in_reference']}",
        f"- Size mismatch: {summary['pairs_size_mismatch']}",
        "",
        "## Metrics",
        "",
        "| Metric | Mean | Median | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in metrics:
        unit = " dB" if m == "psnr" else ""
        lines.append(
            f"| {m.upper()} "
            f"| {_format_optional_metric(summary.get(f'{m}_mean'), unit=unit)} "
            f"| {_format_optional_metric(summary.get(f'{m}_median'), unit=unit)} "
            f"| {_format_optional_metric(summary.get(f'{m}_min'), unit=unit)} "
            f"| {_format_optional_metric(summary.get(f'{m}_max'), unit=unit)} |"
        )

    if gate is not None:
        lines += [
            "",
            "## Quality Gate",
            f"- Status: {'PASS' if gate['passed'] else 'FAIL'}",
            f"- Min PSNR: {_format_optional_float(cast(float | None, gate.get('min_psnr')))}",
            f"- Min SSIM: {_format_optional_float(cast(float | None, gate.get('min_ssim')))}",
            f"- Max frequency consistency: {_format_optional_float(cast(float | None, gate.get('max_frequency_consistency')))}",
        ]
        if gate["reasons"]:
            lines.append("- Reasons:")
            lines.extend(f"  - {reason}" for reason in gate["reasons"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def make_image_html(result: dict, output_path: str) -> None:
    """Generate an HTML report for a photometric (PSNR/SSIM) image check."""
    summary = result["summary"]
    metadata = result.get("metadata", {})
    gate = cast(dict[str, Any] | None, result.get("quality_gate"))
    metrics = cast(list, metadata.get("metrics", ["psnr", "ssim"]))

    summary_rows = [
        ("Rendered dir", str(metadata.get("rendered_dir", ""))),
        ("Reference dir", str(metadata.get("reference_dir", ""))),
        ("Pairs evaluated", str(summary["pairs_evaluated"])),
        ("Missing in reference", str(summary["pairs_missing_in_reference"])),
        ("Size mismatch", str(summary["pairs_size_mismatch"])),
    ]
    summary_html = "\n".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in summary_rows
    )
    metric_rows_html = "\n".join(
        (
            "<tr>"
            f"<td>{escape(m.upper())}</td>"
            f"<td>{escape(_format_optional_metric(summary.get(f'{m}_mean'), unit=' dB' if m == 'psnr' else ''))}</td>"
            f"<td>{escape(_format_optional_metric(summary.get(f'{m}_median'), unit=' dB' if m == 'psnr' else ''))}</td>"
            f"<td>{escape(_format_optional_metric(summary.get(f'{m}_min'), unit=' dB' if m == 'psnr' else ''))}</td>"
            f"<td>{escape(_format_optional_metric(summary.get(f'{m}_max'), unit=' dB' if m == 'psnr' else ''))}</td>"
            "</tr>"
        )
        for m in metrics
    )

    gate_html = ""
    if gate is not None:
        gate_rows = [
            ("Status", "PASS" if gate["passed"] else "FAIL"),
            ("Min PSNR", _format_optional_float(cast(float | None, gate.get("min_psnr")))),
            ("Min SSIM", _format_optional_float(cast(float | None, gate.get("min_ssim")))),
            (
                "Max frequency consistency",
                _format_optional_float(
                    cast(float | None, gate.get("max_frequency_consistency"))
                ),
            ),
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
  <title>CloudAnalyzer Image Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #111827; }}
    h1, h2, h3 {{ margin-bottom: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f3f4f6; width: 18rem; }}
    tbody tr:nth-child(even) {{ background: #f9fafb; }}
  </style>
</head>
<body>
  <h1>CloudAnalyzer Image Evaluation Report</h1>

  <h2>Summary</h2>
  <table><tbody>{summary_html}</tbody></table>

  <h2>Metrics</h2>
  <table>
    <thead>
      <tr><th>Metric</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th></tr>
    </thead>
    <tbody>{metric_rows_html}</tbody>
  </table>

  {gate_html}
</body>
</html>
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def save_image_report(result: dict, output_path: str) -> None:
    """Write a photometric image evaluation report based on the file extension."""
    ext = Path(output_path).suffix.lower()
    if ext in {".md", ".markdown"}:
        make_image_markdown(result, output_path)
        return
    if ext == ".html":
        make_image_html(result, output_path)
        return
    raise ValueError("Unsupported report format. Use .md, .markdown, or .html")


def make_rendered_markdown(result: dict, output_path: str) -> None:
    """Generate a Markdown report for rendered 3DGS evaluation."""

    metadata = result.get("metadata", {})
    renderer = result.get("renderer", {})
    photometric = result.get("photometric", {})
    geometry = result.get("geometry")
    gate = result.get("quality_gate")

    lines = [
        "# CloudAnalyzer Rendered 3DGS Evaluation Report",
        "",
        "## Summary",
        f"- Splat: {metadata.get('splat_path', '')}",
        f"- Cameras: {metadata.get('cameras_path', '')}",
        f"- Reference dir: {metadata.get('reference_dir', '')}",
        f"- Renderer: {renderer.get('backend', '')} ({renderer.get('frames_rendered', 0)} frames)",
        f"- Rendered dir: {renderer.get('rendered_dir', '')}",
        "",
        "## Photometric metrics",
        "",
    ]
    summary = photometric.get("summary", {})
    metrics = metadata.get("metrics", ["psnr", "ssim"])
    lines += [
        "| Metric | Mean | Median | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in metrics:
        unit = " dB" if m == "psnr" else ""
        lines.append(
            f"| {str(m).upper()} "
            f"| {_format_optional_metric(summary.get(f'{m}_mean'), unit=unit)} "
            f"| {_format_optional_metric(summary.get(f'{m}_median'), unit=unit)} "
            f"| {_format_optional_metric(summary.get(f'{m}_min'), unit=unit)} "
            f"| {_format_optional_metric(summary.get(f'{m}_max'), unit=unit)} |"
        )

    if geometry is not None:
        lines += [
            "",
            "## Geometry metrics",
            "",
            f"- Chamfer: {geometry.get('chamfer_distance', '--')}",
            f"- AUC: {geometry.get('auc', '--')}",
            f"- F1 @ primary threshold: "
            f"{(geometry.get('best_f1') or {}).get('f1', '--')}",
        ]
        rep = geometry.get("representation")
        if isinstance(rep, dict):
            lines.append(
                f"- Representation: {rep.get('detected')} "
                f"({rep.get('final_count')} points after filters)"
            )

    if isinstance(gate, dict):
        lines += [
            "",
            "## Quality Gate",
            f"- Status: {'PASS' if gate.get('passed') else 'FAIL'}",
            "- Min PSNR: "
            f"{_format_optional_float(cast(float | None, gate.get('min_psnr')))}",
            "- Min SSIM: "
            f"{_format_optional_float(cast(float | None, gate.get('min_ssim')))}",
            "- Max LPIPS: "
            f"{_format_optional_float(cast(float | None, gate.get('max_lpips')))}",
            "- Max DreamSim distance: "
            f"{_format_optional_float(cast(float | None, gate.get('max_dreamsim_distance')))}",
            "- Max frequency consistency: "
            f"{_format_optional_float(cast(float | None, gate.get('max_frequency_consistency')))}",
            "- Min AUC: "
            f"{_format_optional_float(cast(float | None, gate.get('min_auc')))}",
            "- Max Chamfer: "
            f"{_format_optional_float(cast(float | None, gate.get('max_chamfer')))}",
        ]
        reasons = gate.get("reasons")
        if isinstance(reasons, list) and reasons:
            lines.append("- Reasons:")
            lines.extend(f"  - {reason}" for reason in reasons)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def make_rendered_html(result: dict, output_path: str) -> None:
    """Generate an HTML report for rendered 3DGS evaluation."""

    metadata = result.get("metadata", {})
    renderer = result.get("renderer", {})
    photometric = result.get("photometric", {})
    geometry = result.get("geometry")
    gate = result.get("quality_gate")
    summary = photometric.get("summary", {})
    metrics = metadata.get("metrics", ["psnr", "ssim"])

    summary_rows = [
        ("Splat", str(metadata.get("splat_path", ""))),
        ("Cameras", str(metadata.get("cameras_path", ""))),
        ("Reference dir", str(metadata.get("reference_dir", ""))),
        ("Renderer", str(renderer.get("backend", ""))),
        ("Frames rendered", str(renderer.get("frames_rendered", ""))),
        ("Rendered dir", str(renderer.get("rendered_dir", ""))),
    ]
    summary_html = "\n".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in summary_rows
    )
    metric_rows_html = "\n".join(
        (
            "<tr>"
            f"<td>{escape(str(m).upper())}</td>"
            f"<td>{escape(_format_optional_metric(summary.get(f'{m}_mean'), unit=' dB' if m == 'psnr' else ''))}</td>"
            f"<td>{escape(_format_optional_metric(summary.get(f'{m}_median'), unit=' dB' if m == 'psnr' else ''))}</td>"
            f"<td>{escape(_format_optional_metric(summary.get(f'{m}_min'), unit=' dB' if m == 'psnr' else ''))}</td>"
            f"<td>{escape(_format_optional_metric(summary.get(f'{m}_max'), unit=' dB' if m == 'psnr' else ''))}</td>"
            "</tr>"
        )
        for m in metrics
    )

    geometry_html = ""
    if geometry is not None:
        rep = geometry.get("representation") if isinstance(geometry.get("representation"), dict) else {}
        geometry_rows = [
            ("Chamfer", f"{geometry.get('chamfer_distance', '--')}"),
            ("AUC", f"{geometry.get('auc', '--')}"),
            (
                "Best F1",
                str((geometry.get("best_f1") or {}).get("f1", "--")),
            ),
            ("Representation", str(rep.get("detected", "--"))),
            ("Adapted points", str(rep.get("final_count", "--"))),
        ]
        geometry_html = (
            "<h2>Geometry metrics</h2><table><tbody>"
            + "".join(
                f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
                for label, value in geometry_rows
            )
            + "</tbody></table>"
        )

    gate_html = ""
    if isinstance(gate, dict):
        gate_rows = [
            ("Status", "PASS" if gate.get("passed") else "FAIL"),
            ("Min PSNR", _format_optional_float(cast(float | None, gate.get("min_psnr")))),
            ("Min SSIM", _format_optional_float(cast(float | None, gate.get("min_ssim")))),
            ("Max LPIPS", _format_optional_float(cast(float | None, gate.get("max_lpips")))),
            (
                "Max DreamSim distance",
                _format_optional_float(
                    cast(float | None, gate.get("max_dreamsim_distance"))
                ),
            ),
            (
                "Max frequency consistency",
                _format_optional_float(
                    cast(float | None, gate.get("max_frequency_consistency"))
                ),
            ),
            ("Min AUC", _format_optional_float(cast(float | None, gate.get("min_auc")))),
            (
                "Max Chamfer",
                _format_optional_float(cast(float | None, gate.get("max_chamfer"))),
            ),
        ]
        reasons = gate.get("reasons")
        reasons_html = ""
        if isinstance(reasons, list) and reasons:
            reasons_html = "<h3>Gate Reasons</h3><ul>" + "".join(
                f"<li>{escape(str(reason))}</li>" for reason in reasons
            ) + "</ul>"
        gate_html = (
            "<h2>Quality Gate</h2><table><tbody>"
            + "".join(
                f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
                for label, value in gate_rows
            )
            + f"</tbody></table>{reasons_html}"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CloudAnalyzer Rendered 3DGS Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #111827; }}
    h1, h2, h3 {{ margin-bottom: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f3f4f6; width: 18rem; }}
    tbody tr:nth-child(even) {{ background: #f9fafb; }}
  </style>
</head>
<body>
  <h1>CloudAnalyzer Rendered 3DGS Evaluation Report</h1>

  <h2>Summary</h2>
  <table><tbody>{summary_html}</tbody></table>

  <h2>Photometric metrics</h2>
  <table>
    <thead>
      <tr><th>Metric</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th></tr>
    </thead>
    <tbody>{metric_rows_html}</tbody>
  </table>

  {geometry_html}
  {gate_html}
</body>
</html>
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def save_rendered_report(result: dict, output_path: str) -> None:
    """Write a rendered 3DGS evaluation report based on the file extension."""
    ext = Path(output_path).suffix.lower()
    if ext in {".md", ".markdown"}:
        make_rendered_markdown(result, output_path)
        return
    if ext == ".html":
        make_rendered_html(result, output_path)
        return
    raise ValueError("Unsupported report format. Use .md, .markdown, or .html")
