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
