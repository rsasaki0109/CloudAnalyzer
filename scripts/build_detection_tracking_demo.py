#!/usr/bin/env python3
"""Build a public detection/tracking demo from the RELLIS-3D-derived examples.

Generates evaluation reports for good and regressed candidates against reference
data, plus a wrapper index page for GitHub Pages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "cloudanalyzer"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from ca.detection import evaluate_detection
from ca.tracking import evaluate_tracking
from ca.report import save_detection_report, save_tracking_report

OBJECT_EVAL_DIR = REPO_ROOT / "demo_assets" / "public" / "rellis3d-frame-000001" / "object_eval"


def _build_index_html(
    output_dir: Path,
    det_good: dict,
    det_regressed: dict,
    trk_good: dict,
    trk_regressed: dict,
) -> None:
    """Generate the wrapper index.html with comparison tables."""

    def _gate_badge(result: dict) -> str:
        gate = result.get("quality_gate")
        if gate is None:
            return '<span style="color:#6b7280">No gate</span>'
        if gate["passed"]:
            return '<span style="color:#059669;font-weight:600">PASS</span>'
        return '<span style="color:#dc2626;font-weight:600">FAIL</span>'

    def _fmt(val: float) -> str:
        return f"{val:.4f}"

    det_policy = det_good.get("matching_policy", {})
    trk_policy = trk_good.get("matching_policy", {})

    html = dedent(f"""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>CloudAnalyzer Detection &amp; Tracking Demo</title>
    <style>
      :root {{
        --bg: #fafaf9; --fg: #1c1917; --accent: #2563eb;
        --card-bg: #fff; --border: #e7e5e4; --muted: #78716c;
        --pass: #059669; --fail: #dc2626;
      }}
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      body {{ font-family: 'Space Grotesk', system-ui, sans-serif; background: var(--bg); color: var(--fg); line-height: 1.6; padding: 2rem; max-width: 1100px; margin: 0 auto; }}
      h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
      h2 {{ font-size: 1.3rem; margin: 2rem 0 0.8rem; color: var(--fg); }}
      .subtitle {{ color: var(--muted); margin-bottom: 2rem; }}
      table {{ width: 100%; border-collapse: collapse; margin-bottom: 1rem; }}
      th, td {{ padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }}
      th {{ background: #f5f5f4; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
      td {{ font-size: 0.95rem; }}
      .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }}
      .meta {{ font-size: 0.85rem; color: var(--muted); }}
      a {{ color: var(--accent); text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
      .reports {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }}
      .reports a {{ display: block; padding: 0.8rem 1.2rem; background: var(--accent); color: white; border-radius: 6px; text-align: center; font-weight: 500; }}
      .reports a:hover {{ opacity: 0.9; text-decoration: none; }}
    </style>
    </head>
    <body>

    <h1>Detection &amp; Tracking Demo</h1>
    <p class="subtitle">3D object detection (mAP/AP) and multi-object tracking (MOTA) evaluation on public RELLIS-3D-derived examples.</p>

    <h2>Detection Evaluation</h2>
    <div class="card">
      <table>
        <thead><tr><th>Candidate</th><th>mAP</th><th>Precision</th><th>Recall</th><th>F1</th><th>Gate</th></tr></thead>
        <tbody>
          <tr>
            <td>Good</td>
            <td>{_fmt(det_good["mAP"])}</td>
            <td>{_fmt(det_good["primary_threshold_result"]["precision"])}</td>
            <td>{_fmt(det_good["primary_threshold_result"]["recall"])}</td>
            <td>{_fmt(det_good["primary_threshold_result"]["f1"])}</td>
            <td>{_gate_badge(det_good)}</td>
          </tr>
          <tr>
            <td>Regressed</td>
            <td>{_fmt(det_regressed["mAP"])}</td>
            <td>{_fmt(det_regressed["primary_threshold_result"]["precision"])}</td>
            <td>{_fmt(det_regressed["primary_threshold_result"]["recall"])}</td>
            <td>{_fmt(det_regressed["primary_threshold_result"]["f1"])}</td>
            <td>{_gate_badge(det_regressed)}</td>
          </tr>
        </tbody>
      </table>
      <p class="meta">Geometry: {det_policy.get("geometry", "N/A")} &middot; IoU thresholds: {det_policy.get("iou_thresholds", [])} &middot; Yaw ignored: {det_policy.get("yaw_ignored", True)}</p>
    </div>

    <h2>Tracking Evaluation</h2>
    <div class="card">
      <table>
        <thead><tr><th>Candidate</th><th>MOTA</th><th>Recall</th><th>ID Switches</th><th>Mostly Tracked</th><th>Gate</th></tr></thead>
        <tbody>
          <tr>
            <td>Good</td>
            <td>{_fmt(trk_good["tracking"]["mota"])}</td>
            <td>{_fmt(trk_good["detection"]["recall"])}</td>
            <td>{trk_good["tracking"]["id_switches"]}</td>
            <td>{_fmt(trk_good["tracking"]["mostly_tracked_ratio"])}</td>
            <td>{_gate_badge(trk_good)}</td>
          </tr>
          <tr>
            <td>Regressed</td>
            <td>{_fmt(trk_regressed["tracking"]["mota"])}</td>
            <td>{_fmt(trk_regressed["detection"]["recall"])}</td>
            <td>{trk_regressed["tracking"]["id_switches"]}</td>
            <td>{_fmt(trk_regressed["tracking"]["mostly_tracked_ratio"])}</td>
            <td>{_gate_badge(trk_regressed)}</td>
          </tr>
        </tbody>
      </table>
      <p class="meta">Geometry: {trk_policy.get("geometry", "N/A")} &middot; IoU threshold: {trk_policy.get("iou_threshold", "N/A")} &middot; Yaw ignored: {trk_policy.get("yaw_ignored", True)}</p>
    </div>

    <h2>Detailed Reports</h2>
    <div class="reports">
      <a href="detection_good.html">Detection &mdash; Good</a>
      <a href="detection_regressed.html">Detection &mdash; Regressed</a>
      <a href="tracking_good.html">Tracking &mdash; Good</a>
      <a href="tracking_regressed.html">Tracking &mdash; Regressed</a>
    </div>

    <h2>Build Locally</h2>
    <div class="card">
      <pre style="font-size:0.85rem;overflow-x:auto"><code>python scripts/build_detection_tracking_demo.py \\
  --output docs/demo/detection-tracking</code></pre>
    </div>

    <p class="meta" style="margin-top:2rem">
      Data derived from <a href="https://github.com/unmannedlab/RELLIS-3D">RELLIS-3D</a> (CC&nbsp;BY-NC-SA&nbsp;3.0).
      Tracking examples are deterministic synthetic sequences.
      Built with <a href="https://github.com/rsasaki0109/CloudAnalyzer">CloudAnalyzer</a>.
    </p>

    </body>
    </html>
    """)
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def _write_attribution(output_dir: Path) -> None:
    text = dedent("""\
    # Attribution

    Detection and tracking examples are derived from the public RELLIS-3D dataset:
    - Repository: https://github.com/unmannedlab/RELLIS-3D
    - License: CC BY-NC-SA 3.0
    - Citation: Jiang et al., "RELLIS-3D Dataset: Data, Benchmarks and Analysis
      for Off-Road Robotics", 2021.

    Tracking examples are deterministic synthetic 3-frame sequences seeded from
    the same public RELLIS-3D frame. They do not correspond to real temporal data.

    The CloudAnalyzer Python package itself is MIT-licensed.
    """)
    (output_dir / "ATTRIBUTION.md").write_text(text, encoding="utf-8")


def _write_readme(output_dir: Path) -> None:
    text = dedent("""\
    # Detection & Tracking Demo

    Public demo comparing good and regressed candidates for 3D object detection
    and multi-object tracking evaluation.

    ## Contents

    - `index.html` — Overview page with comparison tables
    - `detection_good.html` / `detection_regressed.html` — Detection reports
    - `tracking_good.html` / `tracking_regressed.html` — Tracking reports
    - `results.json` — Raw evaluation results

    ## Rebuild

    ```bash
    python scripts/build_detection_tracking_demo.py --output docs/demo/detection-tracking
    ```

    ## Data Source

    All examples are derived from the public RELLIS-3D dataset.
    See ATTRIBUTION.md for license details.
    """)
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build detection/tracking demo.")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    det_ref = str(OBJECT_EVAL_DIR / "detection_reference.json")
    det_est_good = str(OBJECT_EVAL_DIR / "detection_estimated_good.json")
    det_est_regressed = str(OBJECT_EVAL_DIR / "detection_estimated_regressed.json")
    trk_ref = str(OBJECT_EVAL_DIR / "tracking_reference.json")
    trk_est_good = str(OBJECT_EVAL_DIR / "tracking_estimated_good.json")
    trk_est_regressed = str(OBJECT_EVAL_DIR / "tracking_estimated_regressed.json")

    print("Running detection evaluation (good)...")
    det_good = evaluate_detection(
        det_est_good, det_ref,
        iou_thresholds=[0.25, 0.5],
        min_map=0.8, min_recall=0.7,
    )

    print("Running detection evaluation (regressed)...")
    det_regressed = evaluate_detection(
        det_est_regressed, det_ref,
        iou_thresholds=[0.25, 0.5],
        min_map=0.8, min_recall=0.7,
    )

    print("Running tracking evaluation (good)...")
    trk_good = evaluate_tracking(
        trk_est_good, trk_ref,
        iou_threshold=0.5,
        min_mota=0.5, min_recall=0.5, max_id_switches=5,
    )

    print("Running tracking evaluation (regressed)...")
    trk_regressed = evaluate_tracking(
        trk_est_regressed, trk_ref,
        iou_threshold=0.5,
        min_mota=0.5, min_recall=0.5, max_id_switches=5,
    )

    # Generate individual HTML reports
    print("Generating reports...")
    save_detection_report(det_good, str(output_dir / "detection_good.html"))
    save_detection_report(det_regressed, str(output_dir / "detection_regressed.html"))
    save_tracking_report(trk_good, str(output_dir / "tracking_good.html"))
    save_tracking_report(trk_regressed, str(output_dir / "tracking_regressed.html"))

    # Generate wrapper page
    _build_index_html(output_dir, det_good, det_regressed, trk_good, trk_regressed)

    # Write results JSON
    results = {
        "detection_good": det_good,
        "detection_regressed": det_regressed,
        "tracking_good": trk_good,
        "tracking_regressed": trk_regressed,
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8"
    )

    _write_attribution(output_dir)
    _write_readme(output_dir)

    print(f"\nDemo generated at: {output_dir}")
    print(f"  Detection good  mAP: {det_good['mAP']:.4f}")
    print(f"  Detection regr  mAP: {det_regressed['mAP']:.4f}")
    print(f"  Tracking good  MOTA: {trk_good['tracking']['mota']:.4f}")
    print(f"  Tracking regr  MOTA: {trk_regressed['tracking']['mota']:.4f}")


if __name__ == "__main__":
    main()
