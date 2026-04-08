#!/usr/bin/env python3
"""Build a public perception QA demo from the RELLIS-3D LiDAR example.

The demo derives reference ground/non-ground splits from public semantic labels
and then injects deterministic label noise to emulate a model prediction.

Usage:
    python3 scripts/build_perception_demo.py --output docs/demo/perception
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "cloudanalyzer"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from public_benchmark_assets import (
    RELLIS_LABEL_CONFIG_URL,
    RELLIS_LIDAR_EXAMPLE_URL,
    RELLIS_README_URL,
    RELLIS_REPO_URL,
    download_rellis_lidar_example,
)
from ca.ground_evaluate import evaluate_ground_segmentation
from ca.report import save_ground_report

RELLIS_LABEL_NAMES = {
    0: "void",
    1: "dirt",
    3: "grass",
    4: "tree",
    5: "pole",
    6: "water",
    7: "sky",
    8: "vehicle",
    9: "object",
    10: "asphalt",
    12: "building",
    15: "log",
    17: "person",
    18: "fence",
    19: "bush",
    23: "concrete",
    27: "barrier",
    31: "puddle",
    33: "mud",
    34: "rubble",
}
GROUND_LABEL_IDS = (1, 3, 10, 23, 31, 33)
IGNORED_LABEL_IDS = (0,)
DEFAULT_FRAME = "000001"
RNG_SEED = 42


def _read_rellis_scan(example_root: Path, frame: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one RELLIS-3D example scan in SemanticKITTI-style format."""
    bin_path = example_root / "os1_cloud_node_kitti_bin" / f"{frame}.bin"
    label_path = example_root / "os1_cloud_node_semantickitti_label_id" / f"{frame}.label"
    if not bin_path.exists():
        raise FileNotFoundError(f"missing point cloud frame: {bin_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"missing label frame: {label_path}")

    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    labels = np.fromfile(label_path, dtype=np.uint32)
    if points.shape[0] != labels.shape[0]:
        raise RuntimeError("point and label counts do not match")
    return np.asarray(points, dtype=np.float64), labels


def _split_reference(points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labeled points and the reference ground mask."""
    ignore_mask = np.isin(labels, IGNORED_LABEL_IDS)
    labeled_points = points[~ignore_mask]
    labeled_labels = labels[~ignore_mask]
    ground_mask = np.isin(labeled_labels, GROUND_LABEL_IDS)
    return labeled_points, labeled_labels, ground_mask


def _make_estimated_ground_mask(points: np.ndarray, reference_ground_mask: np.ndarray) -> np.ndarray:
    """Create a deterministic noisy estimate from the reference ground split."""
    rng = np.random.default_rng(RNG_SEED)
    estimated_ground_mask = reference_ground_mask.copy()

    ground_indices = np.flatnonzero(reference_ground_mask)
    nonground_indices = np.flatnonzero(~reference_ground_mask)

    ground_error_count = max(10, ground_indices.size // 800)
    nonground_error_count = max(10, nonground_indices.size // 900)

    ground_candidates = rng.choice(ground_indices, size=ground_error_count, replace=False)
    estimated_ground_mask[ground_candidates] = False

    nonground_order = np.argsort(points[nonground_indices, 2])
    low_nonground_candidates = nonground_indices[nonground_order[: max(nonground_error_count * 3, nonground_error_count)]]
    nonground_candidates = rng.choice(
        low_nonground_candidates,
        size=nonground_error_count,
        replace=False,
    )
    estimated_ground_mask[nonground_candidates] = True

    return estimated_ground_mask


def _write_pcd(path: Path, points: np.ndarray) -> None:
    """Write Nx3 points to a PCD file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)


def _label_names(ids: tuple[int, ...]) -> str:
    return ", ".join(f"{RELLIS_LABEL_NAMES[label_id]} ({label_id})" for label_id in ids)


def _counts_by_label(labels: np.ndarray) -> list[str]:
    unique, counts = np.unique(labels, return_counts=True)
    rows = []
    for label_id, count in zip(unique, counts):
        rows.append(f"- `{int(label_id)}` {RELLIS_LABEL_NAMES.get(int(label_id), 'unknown')}: {int(count):,} points")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/demo/perception"))
    parser.add_argument(
        "--frame",
        default=DEFAULT_FRAME,
        help="RELLIS-3D example frame stem to use (default: %(default)s)",
    )
    args = parser.parse_args()

    output_dir = args.output
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        example_root = download_rellis_lidar_example(Path(tmp_dir))
        points, labels = _read_rellis_scan(example_root, args.frame)

    labeled_points, labeled_labels, reference_ground_mask = _split_reference(points, labels)
    estimated_ground_mask = _make_estimated_ground_mask(labeled_points, reference_ground_mask)

    reference_ground = labeled_points[reference_ground_mask]
    reference_nonground = labeled_points[~reference_ground_mask]
    estimated_ground = labeled_points[estimated_ground_mask]
    estimated_nonground = labeled_points[~estimated_ground_mask]

    _write_pcd(output_dir / "reference_ground.pcd", reference_ground)
    _write_pcd(output_dir / "reference_nonground.pcd", reference_nonground)
    _write_pcd(output_dir / "estimated_ground.pcd", estimated_ground)
    _write_pcd(output_dir / "estimated_nonground.pcd", estimated_nonground)

    result = evaluate_ground_segmentation(
        str(output_dir / "estimated_ground.pcd"),
        str(output_dir / "estimated_nonground.pcd"),
        str(output_dir / "reference_ground.pcd"),
        str(output_dir / "reference_nonground.pcd"),
        voxel_size=0.2,
        min_precision=0.97,
        min_recall=0.97,
        min_f1=0.97,
    )
    result["report_metadata"] = [
        {"label": "Dataset", "value": "RELLIS-3D Ouster LiDAR with Annotation Examples"},
        {"label": "Frame", "value": args.frame},
        {"label": "Ignored labels", "value": _label_names(IGNORED_LABEL_IDS)},
        {"label": "Ground labels", "value": _label_names(GROUND_LABEL_IDS)},
        {"label": "Source repository", "value": RELLIS_REPO_URL},
    ]
    result["report_notes"] = [
        "Reference ground/non-ground splits come from the official semantic labels.",
        "The estimated split is a deterministic perturbation of the reference labels to emulate boundary leakage and low-obstacle confusion.",
        "This demo uses only non-void labeled points from the public example bundle.",
    ]

    result_path = output_dir / "ground_evaluate_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    save_ground_report(result, str(output_dir / "index.html"))
    save_ground_report(result, str(output_dir / "report.md"))

    attribution_lines = [
        "# Perception Demo Attribution",
        "",
        "- Dataset: [RELLIS-3D](%s)" % RELLIS_REPO_URL,
        "- Dataset README: [%s](%s)" % ("RELLIS-3D README", RELLIS_README_URL),
        "- Example bundle: [Ouster LiDAR with Annotation Examples](%s)" % RELLIS_LIDAR_EXAMPLE_URL,
        "- Label ontology: [rellis.yaml](%s)" % RELLIS_LABEL_CONFIG_URL,
        "- License: CC BY-NC-SA 3.0 (see the dataset README)",
        "- Frame used: `%s`" % args.frame,
        "- Ignored labels: %s" % _label_names(IGNORED_LABEL_IDS),
        "- Ground labels: %s" % _label_names(GROUND_LABEL_IDS),
    ]
    (output_dir / "ATTRIBUTION.md").write_text("\n".join(attribution_lines) + "\n", encoding="utf-8")

    readme_lines = [
        "# Perception QA Demo",
        "",
        "This demo evaluates ground segmentation on the public RELLIS-3D LiDAR example.",
        "Reference splits come from the official semantic labels, and the estimated splits",
        "are a deterministic perturbation used to emulate perception errors in a reproducible way.",
        "",
        "## Source Data",
        "",
        "- Dataset: [RELLIS-3D](%s)" % RELLIS_REPO_URL,
        "- Example bundle: [Ouster LiDAR with Annotation Examples](%s)" % RELLIS_LIDAR_EXAMPLE_URL,
        "- Label ontology: [rellis.yaml](%s)" % RELLIS_LABEL_CONFIG_URL,
        "- Frame: `%s`" % args.frame,
        "",
        "## Label Policy",
        "",
        "- Ignored labels: %s" % _label_names(IGNORED_LABEL_IDS),
        "- Ground labels: %s" % _label_names(GROUND_LABEL_IDS),
        "- Non-ground labels are every remaining labeled class in the frame.",
        "",
        "## Labeled Points In This Frame",
        "",
        *_counts_by_label(labeled_labels),
        "",
        "## Files",
        "",
        "| File | Description |",
        "|---|---|",
        "| `reference_ground.pcd` | Reference terrain points from the official labels |",
        "| `reference_nonground.pcd` | Reference non-ground points from the official labels |",
        "| `estimated_ground.pcd` | Deterministic noisy estimate used as a demo prediction |",
        "| `estimated_nonground.pcd` | Complementary deterministic noisy estimate |",
        "| `ground_evaluate_result.json` | Raw evaluation result |",
        "| `index.html` | Ground segmentation report for GitHub Pages |",
        "| `report.md` | Markdown version of the same report |",
        "| `ATTRIBUTION.md` | Dataset provenance and license note |",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        "| Precision | %.4f |" % result["precision"],
        "| Recall | %.4f |" % result["recall"],
        "| F1 | %.4f |" % result["f1"],
        "| IoU | %.4f |" % result["iou"],
        "| Accuracy | %.4f |" % result["accuracy"],
        "",
        "## Reproduce",
        "",
        "```bash",
        "python3 scripts/build_perception_demo.py --output docs/demo/perception --frame %s" % args.frame,
        "",
        "ca ground-evaluate \\",
        "  docs/demo/perception/estimated_ground.pcd \\",
        "  docs/demo/perception/estimated_nonground.pcd \\",
        "  docs/demo/perception/reference_ground.pcd \\",
        "  docs/demo/perception/reference_nonground.pcd \\",
        "  --min-precision 0.97 --min-recall 0.97 --min-f1 0.97 --voxel-size 0.2 \\",
        "  --report docs/demo/perception/index.html",
        "```",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"Frame:        {args.frame}")
    print(f"Labels kept:  {int(labeled_points.shape[0])}")
    print(f"Ref ground:   {int(reference_ground.shape[0])}")
    print(f"Ref non-grnd: {int(reference_nonground.shape[0])}")
    print(f"Est ground:   {int(estimated_ground.shape[0])}")
    print(f"Est non-grnd: {int(estimated_nonground.shape[0])}")
    print(f"Precision:    {result['precision']:.4f}")
    print(f"Recall:       {result['recall']:.4f}")
    print(f"F1:           {result['f1']:.4f}")
    print(f"IoU:          {result['iou']:.4f}")
    print(f"Report:       {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
