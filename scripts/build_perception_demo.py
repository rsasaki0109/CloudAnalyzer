#!/usr/bin/env python3
"""Build a public perception QA demo using the Stanford Bunny.

Generates ground/non-ground point clouds from the Stanford Bunny mesh and
runs ca ground-evaluate to demonstrate perception QA capabilities.

Usage:
    python3 scripts/build_perception_demo.py --output docs/demo/perception
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "cloudanalyzer"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from public_benchmark_assets import download_bunny_mesh
from ca.ground_evaluate import evaluate_ground_segmentation


def _make_ground_scene(mesh: o3d.geometry.TriangleMesh, rng: np.random.Generator) -> dict:
    """Create a synthetic ground scene from the Stanford Bunny.

    Returns dict with ground/nonground arrays for both reference and estimation.
    """
    # Sample points from the bunny mesh
    pcd = mesh.sample_points_uniformly(number_of_points=5000)
    points = np.asarray(pcd.points, dtype=np.float64)

    # Normalize and place bunny on a ground plane
    points -= points.mean(axis=0)
    points *= 10.0  # scale to ~10m scene

    # Create a ground plane (z near 0)
    ground_plane = rng.uniform([-15, -15, -0.3], [15, 15, 0.3], size=(3000, 3))

    # Bunny points are "non-ground" (objects above ground)
    bunny_lifted = points.copy()
    bunny_lifted[:, 2] += 2.0  # lift above ground

    all_points = np.vstack([ground_plane, bunny_lifted])
    labels = np.concatenate([
        np.ones(len(ground_plane), dtype=np.int8),    # 1 = ground
        np.zeros(len(bunny_lifted), dtype=np.int8),   # 0 = non-ground
    ])

    # Reference: perfect labels
    ref_ground = all_points[labels == 1]
    ref_nonground = all_points[labels == 0]

    # Estimation: simulate imperfect segmentation
    # - 5% of ground points misclassified as non-ground (boundary noise)
    # - 3% of non-ground points misclassified as ground (low objects)
    ground_mask = labels == 1
    nonground_mask = labels == 0

    ground_leak = rng.random(ground_mask.sum()) < 0.05
    nonground_leak = rng.random(nonground_mask.sum()) < 0.03

    est_ground_mask = ground_mask.copy()
    est_ground_indices = np.where(ground_mask)[0]
    est_ground_mask[est_ground_indices[ground_leak]] = False

    est_nonground_mask = nonground_mask.copy()
    est_nonground_indices = np.where(nonground_mask)[0]
    est_nonground_mask[est_nonground_indices[nonground_leak]] = False

    # Leaked points swap labels
    est_ground = np.vstack([
        all_points[est_ground_mask],
        all_points[est_nonground_indices[nonground_leak]],
    ])
    est_nonground = np.vstack([
        all_points[est_nonground_mask],
        all_points[est_ground_indices[ground_leak]],
    ])

    return {
        "ref_ground": ref_ground,
        "ref_nonground": ref_nonground,
        "est_ground": est_ground,
        "est_nonground": est_nonground,
    }


def _write_dashboard(output_dir: Path, result: dict, scene: dict) -> None:
    """Write a single-page HTML dashboard for the ground evaluation result."""
    cm = result["confusion_matrix"]
    counts = result["counts"]
    gate = result.get("quality_gate") or {}
    gate_status = "PASS" if gate.get("passed", True) else "FAIL"
    gate_color = "#16a34a" if gate.get("passed", True) else "#dc2626"
    gate_reasons = gate.get("reasons", [])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CloudAnalyzer — Ground Segmentation QA</title>
<style>
  :root {{ --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --accent: #38bdf8;
           --green: #4ade80; --red: #f87171; --dim: #94a3b8; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: var(--bg); color: var(--text); min-height: 100vh; padding: 2rem; }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
  .subtitle {{ color: var(--dim); margin-bottom: 2rem; font-size: 0.95rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
  .metric-card {{ background: var(--card); border-radius: 12px; padding: 1.5rem; text-align: center; }}
  .metric-value {{ font-size: 2.2rem; font-weight: 700; color: var(--accent); }}
  .metric-label {{ color: var(--dim); font-size: 0.85rem; margin-top: 0.25rem; }}
  .gate-badge {{ display: inline-block; padding: 0.4rem 1.2rem; border-radius: 8px;
                  font-weight: 700; font-size: 1.1rem; }}
  .section {{ background: var(--card); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }}
  .section h2 {{ font-size: 1.1rem; margin-bottom: 1rem; color: var(--accent); }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ padding: 0.6rem 1rem; text-align: right; border-bottom: 1px solid #334155; }}
  th {{ color: var(--dim); font-weight: 500; font-size: 0.85rem; text-transform: uppercase; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .cm-grid {{ display: grid; grid-template-columns: auto 1fr 1fr; gap: 0; width: fit-content; margin: 0 auto; }}
  .cm-cell {{ width: 100px; height: 60px; display: flex; align-items: center; justify-content: center;
              font-weight: 700; font-size: 1.2rem; border: 1px solid #334155; }}
  .cm-header {{ display: flex; align-items: center; justify-content: center;
                color: var(--dim); font-size: 0.8rem; font-weight: 500; }}
  .cm-tp {{ background: rgba(74, 222, 128, 0.15); color: var(--green); }}
  .cm-tn {{ background: rgba(74, 222, 128, 0.08); color: var(--green); }}
  .cm-fp {{ background: rgba(248, 113, 113, 0.15); color: var(--red); }}
  .cm-fn {{ background: rgba(248, 113, 113, 0.08); color: var(--red); }}
  .reasons {{ margin-top: 0.5rem; }}
  .reasons li {{ color: var(--dim); font-size: 0.9rem; margin-left: 1.2rem; }}
  footer {{ text-align: center; color: var(--dim); font-size: 0.8rem; margin-top: 2rem; }}
  footer a {{ color: var(--accent); text-decoration: none; }}
</style>
</head>
<body>
<div class="container">
  <h1>Ground Segmentation QA</h1>
  <p class="subtitle">
    Stanford Bunny on synthetic ground plane &middot; CloudAnalyzer <code>ca ground-evaluate</code>
  </p>

  <div class="grid">
    <div class="metric-card">
      <div class="metric-value">{result['precision']:.3f}</div>
      <div class="metric-label">Precision</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{result['recall']:.3f}</div>
      <div class="metric-label">Recall</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{result['f1']:.3f}</div>
      <div class="metric-label">F1 Score</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{result['iou']:.3f}</div>
      <div class="metric-label">IoU</div>
    </div>
    <div class="metric-card">
      <div class="metric-value" style="color: {gate_color}">{gate_status}</div>
      <div class="metric-label">Quality Gate</div>
    </div>
  </div>

  <div class="section">
    <h2>Confusion Matrix (voxel-level)</h2>
    <div class="cm-grid">
      <div class="cm-header"></div>
      <div class="cm-header">Ref: Ground</div>
      <div class="cm-header">Ref: Non-ground</div>
      <div class="cm-header">Est: Ground</div>
      <div class="cm-cell cm-tp">{cm['tp']}</div>
      <div class="cm-cell cm-fp">{cm['fp']}</div>
      <div class="cm-header">Est: Non-ground</div>
      <div class="cm-cell cm-fn">{cm['fn']}</div>
      <div class="cm-cell cm-tn">{cm['tn']}</div>
    </div>
  </div>

  <div class="section">
    <h2>Point Counts</h2>
    <table>
      <tr><th></th><th>Ground</th><th>Non-ground</th><th>Total</th></tr>
      <tr>
        <td>Reference</td>
        <td>{counts['reference_ground_points']:,}</td>
        <td>{counts['reference_nonground_points']:,}</td>
        <td>{counts['reference_ground_points'] + counts['reference_nonground_points']:,}</td>
      </tr>
      <tr>
        <td>Estimated</td>
        <td>{counts['estimated_ground_points']:,}</td>
        <td>{counts['estimated_nonground_points']:,}</td>
        <td>{counts['estimated_ground_points'] + counts['estimated_nonground_points']:,}</td>
      </tr>
    </table>
  </div>

  <div class="section">
    <h2>Scene</h2>
    <table>
      <tr><td>Dataset</td><td>Stanford Bunny + synthetic ground plane</td></tr>
      <tr><td>Ground plane</td><td>3,000 points, z &in; [-0.3, 0.3]</td></tr>
      <tr><td>Non-ground (bunny)</td><td>5,000 points, lifted z+2.0</td></tr>
      <tr><td>Simulated errors</td><td>5% ground leak, 3% non-ground leak</td></tr>
      <tr><td>Voxel size</td><td>{result['voxel_size']}m</td></tr>
    </table>
  </div>

  <div class="section">
    <h2>Quality Gate</h2>
    <table>
      <tr><td>min_precision</td><td>{gate.get('min_precision', '—')}</td></tr>
      <tr><td>min_recall</td><td>{gate.get('min_recall', '—')}</td></tr>
      <tr><td>min_f1</td><td>{gate.get('min_f1', '—')}</td></tr>
      <tr><td>min_iou</td><td>{gate.get('min_iou', '—')}</td></tr>
    </table>
    {"<ul class='reasons'>" + "".join(f"<li>{r}</li>" for r in gate_reasons) + "</ul>" if gate_reasons else ""}
  </div>

  <div class="section">
    <h2>Reproduce</h2>
    <div style="background: #0f172a; padding: 1rem; border-radius: 8px; font-family: monospace; font-size: 0.85rem; overflow-x: auto;">
      <div style="color: var(--dim);"># generate demo data + evaluate</div>
      <div>python3 scripts/build_perception_demo.py</div>
      <br>
      <div style="color: var(--dim);"># or run directly</div>
      <div>ca ground-evaluate \\</div>
      <div>&nbsp;&nbsp;estimated_ground.pcd estimated_nonground.pcd \\</div>
      <div>&nbsp;&nbsp;reference_ground.pcd reference_nonground.pcd \\</div>
      <div>&nbsp;&nbsp;--min-f1 0.9 --voxel-size 0.5</div>
    </div>
  </div>

  <footer>
    <p>
      Built with <a href="https://github.com/rsasaki0109/CloudAnalyzer">CloudAnalyzer</a> &middot;
      Data: <a href="https://graphics.stanford.edu/data/3Dscanrep/">Stanford 3D Scanning Repository</a>
    </p>
  </footer>
</div>
</body>
</html>"""

    (output_dir / "index.html").write_text(html, encoding="utf-8")


def _write_pcd(path: Path, points: np.ndarray) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/demo/perception"))
    args = parser.parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Stanford Bunny...")
    mesh = download_bunny_mesh()
    mesh.compute_vertex_normals()

    print("Generating ground scene...")
    rng = np.random.default_rng(42)
    scene = _make_ground_scene(mesh, rng)

    # Write PCD files
    _write_pcd(output_dir / "reference_ground.pcd", scene["ref_ground"])
    _write_pcd(output_dir / "reference_nonground.pcd", scene["ref_nonground"])
    _write_pcd(output_dir / "estimated_ground.pcd", scene["est_ground"])
    _write_pcd(output_dir / "estimated_nonground.pcd", scene["est_nonground"])

    print(f"  Reference: {len(scene['ref_ground'])} ground, {len(scene['ref_nonground'])} non-ground")
    print(f"  Estimated: {len(scene['est_ground'])} ground, {len(scene['est_nonground'])} non-ground")

    # Run ground evaluation
    print("Running ground-evaluate...")
    result = evaluate_ground_segmentation(
        str(output_dir / "estimated_ground.pcd"),
        str(output_dir / "estimated_nonground.pcd"),
        str(output_dir / "reference_ground.pcd"),
        str(output_dir / "reference_nonground.pcd"),
        voxel_size=0.5,
        min_precision=0.9,
        min_recall=0.9,
        min_f1=0.9,
    )

    # Write result JSON
    result_path = output_dir / "ground_evaluate_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Print summary
    print(f"\nResults:")
    print(f"  Precision: {result['precision']:.4f}")
    print(f"  Recall:    {result['recall']:.4f}")
    print(f"  F1:        {result['f1']:.4f}")
    print(f"  IoU:       {result['iou']:.4f}")
    gate = result.get("quality_gate")
    if gate:
        print(f"  Gate:      {'PASS' if gate['passed'] else 'FAIL'}")

    # Write HTML dashboard
    _write_dashboard(output_dir, result, scene)

    # Write a README for the demo
    readme = output_dir / "README.md"
    readme.write_text(
        f"""# Perception QA Demo — Ground Segmentation

This demo uses the [Stanford Bunny](https://graphics.stanford.edu/data/3Dscanrep/) placed on a synthetic ground plane to demonstrate CloudAnalyzer's ground segmentation evaluation.

## Scene

- **Ground plane**: 3000 points, z near 0
- **Non-ground (bunny)**: 5000 points, lifted above ground
- **Simulated errors**: 5% ground leak, 3% non-ground leak

## Files

| File | Description |
|---|---|
| `reference_ground.pcd` | Reference ground truth ground points |
| `reference_nonground.pcd` | Reference ground truth non-ground points |
| `estimated_ground.pcd` | Simulated segmentation result (ground) |
| `estimated_nonground.pcd` | Simulated segmentation result (non-ground) |
| `ground_evaluate_result.json` | Evaluation result |

## How to reproduce

```bash
# Run the demo script
python3 scripts/build_perception_demo.py --output docs/demo/perception

# Or evaluate directly
ca ground-evaluate \\
  docs/demo/perception/estimated_ground.pcd \\
  docs/demo/perception/estimated_nonground.pcd \\
  docs/demo/perception/reference_ground.pcd \\
  docs/demo/perception/reference_nonground.pcd \\
  --min-f1 0.9 --voxel-size 0.5
```

## Results

| Metric | Value |
|---|---|
| Precision | {result['precision']:.4f} |
| Recall | {result['recall']:.4f} |
| F1 | {result['f1']:.4f} |
| IoU | {result['iou']:.4f} |

Data source: [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/)
""",
        encoding="utf-8",
    )

    print(f"\nDemo written to {output_dir}")
    print(f"  PCD files: 4")
    print(f"  Result JSON: {result_path}")
    print(f"  README: {readme}")


if __name__ == "__main__":
    main()
