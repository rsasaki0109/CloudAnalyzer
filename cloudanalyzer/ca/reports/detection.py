"""Detection and tracking evaluation reports."""

from html import escape
from pathlib import Path
from typing import Any, cast
from ca.reports.common import _format_optional_float


def make_detection_markdown(result: dict, output_path: str) -> None:
    """Generate a Markdown report for 3D detection evaluation."""

    counts = result["counts"]
    primary = result["primary_threshold_result"]
    gate = cast(dict[str, Any] | None, result.get("quality_gate"))

    lines = [
        "# CloudAnalyzer Detection Report",
        "",
        "## Summary",
        f"- Estimated: {result['estimated_path']}",
        f"- Reference: {result['reference_path']}",
        f"- Geometry: {result['matching_policy']['geometry']}",
        f"- IoU thresholds: {', '.join(f'{value:.2f}' for value in result['matching_policy']['iou_thresholds'])}",
        f"- Primary IoU threshold: {result['matching_policy']['primary_iou_threshold']:.2f}",
        "",
        "## Dataset Counts",
        f"- Shared frames: {counts['shared_frames']}",
        f"- Estimated frames / boxes: {counts['estimated_frames']} / {counts['estimated_boxes']}",
        f"- Reference frames / boxes: {counts['reference_frames']} / {counts['reference_boxes']}",
        "",
        "## Overall Metrics",
        f"- mAP: {result['mAP']:.4f}",
        f"- Primary precision: {primary['precision']:.4f}",
        f"- Primary recall: {primary['recall']:.4f}",
        f"- Primary F1: {primary['f1']:.4f}",
        f"- Primary mean IoU: {primary['mean_iou']:.4f}",
        f"- Primary mean center distance: {primary['mean_center_distance']:.4f}",
        "",
        "## Threshold Sweep",
        "",
        "| IoU | mAP | Precision | Recall | F1 | Mean IoU | Mean Center Dist |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold in result["threshold_results"]:
        lines.append(
            f"| {threshold['iou_threshold']:.2f} | {threshold['map']:.4f} | "
            f"{threshold['precision']:.4f} | {threshold['recall']:.4f} | {threshold['f1']:.4f} | "
            f"{threshold['mean_iou']:.4f} | {threshold['mean_center_distance']:.4f} |"
        )

    lines += [
        "",
        "## Per-class Summary",
        "",
        "| Class | Ref | Est | Precision | Recall | F1 | Mean AP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in sorted(result["per_class"].items()):
        lines.append(
            f"| {label} | {summary['reference_boxes']} | {summary['estimated_boxes']} | "
            f"{summary['precision']:.4f} | {summary['recall']:.4f} | {summary['f1']:.4f} | "
            f"{_format_optional_float(cast(float | None, summary['mean_ap']))} |"
        )

    if gate is not None:
        lines += [
            "",
            "## Quality Gate",
            f"- Status: {'PASS' if gate['passed'] else 'FAIL'}",
            f"- Min mAP: {_format_optional_float(cast(float | None, gate.get('min_map')))}",
            f"- Min Precision: {_format_optional_float(cast(float | None, gate.get('min_precision')))}",
            f"- Min Recall: {_format_optional_float(cast(float | None, gate.get('min_recall')))}",
            f"- Min F1: {_format_optional_float(cast(float | None, gate.get('min_f1')))}",
        ]
        if gate["reasons"]:
            lines.append("- Reasons:")
            lines.extend(f"  - {reason}" for reason in gate["reasons"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def make_detection_html(result: dict, output_path: str) -> None:
    """Generate an HTML report for 3D detection evaluation."""

    counts = result["counts"]
    primary = result["primary_threshold_result"]
    gate = cast(dict[str, Any] | None, result.get("quality_gate"))
    summary_rows = [
        ("Estimated", result["estimated_path"]),
        ("Reference", result["reference_path"]),
        ("Geometry", result["matching_policy"]["geometry"]),
        (
            "IoU thresholds",
            ", ".join(f"{value:.2f}" for value in result["matching_policy"]["iou_thresholds"]),
        ),
        ("Primary IoU threshold", f"{result['matching_policy']['primary_iou_threshold']:.2f}"),
        ("mAP", f"{result['mAP']:.4f}"),
        ("Primary precision", f"{primary['precision']:.4f}"),
        ("Primary recall", f"{primary['recall']:.4f}"),
        ("Primary F1", f"{primary['f1']:.4f}"),
    ]
    threshold_rows = "\n".join(
        (
            "<tr>"
            f"<td>{threshold['iou_threshold']:.2f}</td>"
            f"<td>{threshold['map']:.4f}</td>"
            f"<td>{threshold['precision']:.4f}</td>"
            f"<td>{threshold['recall']:.4f}</td>"
            f"<td>{threshold['f1']:.4f}</td>"
            f"<td>{threshold['mean_iou']:.4f}</td>"
            f"<td>{threshold['mean_center_distance']:.4f}</td>"
            "</tr>"
        )
        for threshold in result["threshold_results"]
    )
    class_rows = "\n".join(
        (
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td>{summary['reference_boxes']}</td>"
            f"<td>{summary['estimated_boxes']}</td>"
            f"<td>{summary['precision']:.4f}</td>"
            f"<td>{summary['recall']:.4f}</td>"
            f"<td>{summary['f1']:.4f}</td>"
            f"<td>{escape(_format_optional_float(cast(float | None, summary['mean_ap'])))}</td>"
            "</tr>"
        )
        for label, summary in sorted(result["per_class"].items())
    )
    gate_html = ""
    if gate is not None:
        gate_rows = "\n".join(
            f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
            for label, value in [
                ("Status", "PASS" if gate["passed"] else "FAIL"),
                ("Min mAP", _format_optional_float(cast(float | None, gate.get("min_map")))),
                (
                    "Min Precision",
                    _format_optional_float(cast(float | None, gate.get("min_precision"))),
                ),
                ("Min Recall", _format_optional_float(cast(float | None, gate.get("min_recall")))),
                ("Min F1", _format_optional_float(cast(float | None, gate.get("min_f1")))),
            ]
        )
        reasons_html = (
            "<ul>" + "".join(f"<li>{escape(reason)}</li>" for reason in gate["reasons"]) + "</ul>"
            if gate["reasons"]
            else ""
        )
        gate_html = (
            "<h2>Quality Gate</h2>"
            f"<table>{gate_rows}</table>"
            f"{reasons_html}"
        )

    summary_html = "\n".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in summary_rows
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CloudAnalyzer Detection Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; line-height: 1.5; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 1.5rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.55rem 0.7rem; text-align: left; }}
    th {{ background: #f3f4f6; }}
    h1, h2 {{ color: #111827; }}
  </style>
</head>
<body>
  <h1>CloudAnalyzer Detection Report</h1>
  <h2>Summary</h2>
  <table>{summary_html}</table>
  <h2>Dataset Counts</h2>
  <table>
    <tr><th>Shared Frames</th><td>{counts['shared_frames']}</td></tr>
    <tr><th>Estimated Frames / Boxes</th><td>{counts['estimated_frames']} / {counts['estimated_boxes']}</td></tr>
    <tr><th>Reference Frames / Boxes</th><td>{counts['reference_frames']} / {counts['reference_boxes']}</td></tr>
  </table>
  <h2>Threshold Sweep</h2>
  <table>
    <thead>
      <tr><th>IoU</th><th>mAP</th><th>Precision</th><th>Recall</th><th>F1</th><th>Mean IoU</th><th>Mean Center Dist</th></tr>
    </thead>
    <tbody>{threshold_rows}</tbody>
  </table>
  <h2>Per-class Summary</h2>
  <table>
    <thead>
      <tr><th>Class</th><th>Ref</th><th>Est</th><th>Precision</th><th>Recall</th><th>F1</th><th>Mean AP</th></tr>
    </thead>
    <tbody>{class_rows}</tbody>
  </table>
  {gate_html}
</body>
</html>
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def save_detection_report(result: dict, output_path: str) -> None:
    """Write a detection evaluation report based on the file extension."""
    ext = Path(output_path).suffix.lower()
    if ext in {".md", ".markdown"}:
        make_detection_markdown(result, output_path)
        return
    if ext == ".html":
        make_detection_html(result, output_path)
        return
    raise ValueError("Unsupported report format. Use .md, .markdown, or .html")


def make_tracking_markdown(result: dict, output_path: str) -> None:
    """Generate a Markdown report for 3D tracking evaluation."""

    counts = result["counts"]
    detection = result["detection"]
    tracking = result["tracking"]
    gate = cast(dict[str, Any] | None, result.get("quality_gate"))

    lines = [
        "# CloudAnalyzer Tracking Report",
        "",
        "## Summary",
        f"- Estimated: {result['estimated_path']}",
        f"- Reference: {result['reference_path']}",
        f"- Geometry: {result['matching_policy']['geometry']}",
        f"- IoU threshold: {result['matching_policy']['iou_threshold']:.2f}",
        "",
        "## Dataset Counts",
        f"- Shared frames: {counts['shared_frames']}",
        f"- Estimated detections / tracks: {counts['estimated_detections']} / {counts['estimated_tracks']}",
        f"- Reference detections / tracks: {counts['reference_detections']} / {counts['reference_tracks']}",
        f"- Matched detections: {counts['matched_detections']}",
        "",
        "## Metrics",
        f"- Precision: {detection['precision']:.4f}",
        f"- Recall: {detection['recall']:.4f}",
        f"- F1: {detection['f1']:.4f}",
        f"- MOTA: {tracking['mota']:.4f}",
        f"- ID switches: {tracking['id_switches']}",
        f"- Track fragmentations: {tracking['track_fragmentations']}",
        f"- Mostly tracked ratio: {tracking['mostly_tracked_ratio']:.4f}",
        f"- Mean IoU: {tracking['mean_iou']:.4f}",
        f"- Mean center distance: {tracking['mean_center_distance']:.4f}",
        "",
        "## Per-class Summary",
        "",
        "| Class | Ref | Est | Matched | Precision | Recall | F1 | IDSW |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in sorted(result["per_class"].items()):
        lines.append(
            f"| {label} | {summary['reference_detections']} | {summary['estimated_detections']} | "
            f"{summary['matched_detections']} | {summary['precision']:.4f} | {summary['recall']:.4f} | "
            f"{summary['f1']:.4f} | {summary['id_switches']} |"
        )

    if result["matched_samples"]:
        lines += [
            "",
            "## Sample Matches",
            "",
            "| Frame | Class | Ref Track | Est Track | IoU | Center Dist |",
            "|---|---|---|---|---:|---:|",
        ]
        for sample in result["matched_samples"]:
            lines.append(
                f"| {sample['frame_id']} | {sample['label']} | {sample['reference_track_id']} | "
                f"{sample['estimated_track_id']} | {sample['iou']:.4f} | {sample['center_distance']:.4f} |"
            )

    if gate is not None:
        lines += [
            "",
            "## Quality Gate",
            f"- Status: {'PASS' if gate['passed'] else 'FAIL'}",
            f"- Min MOTA: {_format_optional_float(cast(float | None, gate.get('min_mota')))}",
            f"- Min Recall: {_format_optional_float(cast(float | None, gate.get('min_recall')))}",
            f"- Max ID switches: {gate.get('max_id_switches', 'n/a')}",
        ]
        if gate["reasons"]:
            lines.append("- Reasons:")
            lines.extend(f"  - {reason}" for reason in gate["reasons"])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def make_tracking_html(result: dict, output_path: str) -> None:
    """Generate an HTML report for 3D tracking evaluation."""

    counts = result["counts"]
    detection = result["detection"]
    tracking = result["tracking"]
    gate = cast(dict[str, Any] | None, result.get("quality_gate"))
    summary_rows = [
        ("Estimated", result["estimated_path"]),
        ("Reference", result["reference_path"]),
        ("IoU threshold", f"{result['matching_policy']['iou_threshold']:.2f}"),
        ("Precision", f"{detection['precision']:.4f}"),
        ("Recall", f"{detection['recall']:.4f}"),
        ("F1", f"{detection['f1']:.4f}"),
        ("MOTA", f"{tracking['mota']:.4f}"),
        ("ID switches", str(tracking["id_switches"])),
        ("Track fragmentations", str(tracking["track_fragmentations"])),
        ("Mostly tracked ratio", f"{tracking['mostly_tracked_ratio']:.4f}"),
        ("Mean IoU", f"{tracking['mean_iou']:.4f}"),
        ("Mean center distance", f"{tracking['mean_center_distance']:.4f}"),
    ]
    class_rows = "\n".join(
        (
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td>{summary['reference_detections']}</td>"
            f"<td>{summary['estimated_detections']}</td>"
            f"<td>{summary['matched_detections']}</td>"
            f"<td>{summary['precision']:.4f}</td>"
            f"<td>{summary['recall']:.4f}</td>"
            f"<td>{summary['f1']:.4f}</td>"
            f"<td>{summary['id_switches']}</td>"
            "</tr>"
        )
        for label, summary in sorted(result["per_class"].items())
    )
    sample_rows = "\n".join(
        (
            "<tr>"
            f"<td>{escape(sample['frame_id'])}</td>"
            f"<td>{escape(sample['label'])}</td>"
            f"<td>{escape(sample['reference_track_id'])}</td>"
            f"<td>{escape(sample['estimated_track_id'])}</td>"
            f"<td>{sample['iou']:.4f}</td>"
            f"<td>{sample['center_distance']:.4f}</td>"
            "</tr>"
        )
        for sample in result["matched_samples"]
    )
    gate_html = ""
    if gate is not None:
        gate_rows = "\n".join(
            f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
            for label, value in [
                ("Status", "PASS" if gate["passed"] else "FAIL"),
                ("Min MOTA", _format_optional_float(cast(float | None, gate.get("min_mota")))),
                ("Min Recall", _format_optional_float(cast(float | None, gate.get("min_recall")))),
                ("Max ID switches", str(gate.get("max_id_switches", "n/a"))),
            ]
        )
        reasons_html = (
            "<ul>" + "".join(f"<li>{escape(reason)}</li>" for reason in gate["reasons"]) + "</ul>"
            if gate["reasons"]
            else ""
        )
        gate_html = f"<h2>Quality Gate</h2><table>{gate_rows}</table>{reasons_html}"

    summary_html = "\n".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in summary_rows
    )
    sample_table = ""
    if result["matched_samples"]:
        sample_table = (
            "<h2>Sample Matches</h2>"
            "<table><thead><tr><th>Frame</th><th>Class</th><th>Ref Track</th><th>Est Track</th><th>IoU</th><th>Center Dist</th></tr></thead>"
            f"<tbody>{sample_rows}</tbody></table>"
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CloudAnalyzer Tracking Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem; line-height: 1.5; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 1.5rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: 0.55rem 0.7rem; text-align: left; }}
    th {{ background: #f3f4f6; }}
    h1, h2 {{ color: #111827; }}
  </style>
</head>
<body>
  <h1>CloudAnalyzer Tracking Report</h1>
  <h2>Summary</h2>
  <table>{summary_html}</table>
  <h2>Dataset Counts</h2>
  <table>
    <tr><th>Shared Frames</th><td>{counts['shared_frames']}</td></tr>
    <tr><th>Estimated Detections / Tracks</th><td>{counts['estimated_detections']} / {counts['estimated_tracks']}</td></tr>
    <tr><th>Reference Detections / Tracks</th><td>{counts['reference_detections']} / {counts['reference_tracks']}</td></tr>
    <tr><th>Matched Detections</th><td>{counts['matched_detections']}</td></tr>
  </table>
  <h2>Per-class Summary</h2>
  <table>
    <thead>
      <tr><th>Class</th><th>Ref</th><th>Est</th><th>Matched</th><th>Precision</th><th>Recall</th><th>F1</th><th>IDSW</th></tr>
    </thead>
    <tbody>{class_rows}</tbody>
  </table>
  {sample_table}
  {gate_html}
</body>
</html>
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


def save_tracking_report(result: dict, output_path: str) -> None:
    """Write a tracking evaluation report based on the file extension."""
    ext = Path(output_path).suffix.lower()
    if ext in {".md", ".markdown"}:
        make_tracking_markdown(result, output_path)
        return
    if ext == ".html":
        make_tracking_html(result, output_path)
        return
    raise ValueError("Unsupported report format. Use .md, .markdown, or .html")
