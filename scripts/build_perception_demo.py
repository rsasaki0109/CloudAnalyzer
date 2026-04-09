#!/usr/bin/env python3
"""Build a public perception artifact-comparison demo from the RELLIS-3D example.

The demo compares two deterministic candidate outputs on the same public LiDAR
frame:

- `nondeep_baseline`: coarse geometry-only proxy
- `deep_baseline`: higher-fidelity learned-style proxy

Both are evaluated against a public reference artifact with `ca batch`.
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
from ca.batch import batch_evaluate
from ca.report import make_batch_summary, save_batch_report

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
IGNORED_LABEL_IDS = (0,)
DEFAULT_FRAME = "000001"
DEFAULT_THRESHOLDS = [0.02, 0.05, 0.1, 0.2, 0.3]
MIN_AUC = 0.90
MAX_CHAMFER = 0.05


def _read_rellis_scan(example_root: Path, frame: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one RELLIS-3D example LiDAR frame and its semantic labels."""
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


def _write_pcd(path: Path, points: np.ndarray) -> None:
    """Write Nx3 points to a point cloud file."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), cloud)


def _nonvoid_reference(points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the non-void labeled points for the reference artifact."""
    keep_mask = ~np.isin(labels, IGNORED_LABEL_IDS)
    return points[keep_mask], labels[keep_mask]


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Return voxel-downsampled points as an ndarray."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return np.asarray(cloud.voxel_down_sample(voxel_size).points, dtype=np.float64)


def _build_nondeep_candidate(reference_points: np.ndarray) -> np.ndarray:
    """Build a coarse geometry-only proxy for a non-deep baseline."""
    points = _voxel_downsample(reference_points, voxel_size=0.20)
    planar_distance = np.linalg.norm(points[:, :2], axis=1)
    near_points = points[planar_distance <= 20.0]
    far_points = points[planar_distance > 20.0][::3]
    candidate = np.vstack([near_points, far_points])
    candidate += np.array([0.10, -0.05, 0.04], dtype=np.float64)
    return candidate


def _build_deep_candidate(reference_points: np.ndarray) -> np.ndarray:
    """Build a higher-fidelity proxy for a learning-based baseline."""
    candidate = _voxel_downsample(reference_points, voxel_size=0.10)
    candidate += np.array([0.02, -0.01, 0.008], dtype=np.float64)
    return candidate


def _label_count_lines(labels: np.ndarray) -> list[str]:
    """Format labeled point counts for README output."""
    unique, counts = np.unique(labels, return_counts=True)
    return [
        f"- `{int(label_id)}` {RELLIS_LABEL_NAMES.get(int(label_id), 'unknown')}: {int(count):,} points"
        for label_id, count in zip(unique, counts)
    ]


def _result_rows(results: list[dict]) -> list[str]:
    """Render metric rows for the demo README."""
    rows = []
    for item in results:
        gate = item.get("quality_gate") or {}
        rows.append(
            "| `%s` | %.4f | %.4f | %.4f | %s |"
            % (
                Path(item["path"]).name,
                item["auc"],
                item["chamfer_distance"],
                item["best_f1"]["f1"],
                "PASS" if gate.get("passed") else "FAIL",
            )
        )
    return rows


def _normalize_result_paths(results: list[dict], reference_path: Path, output_dir: Path) -> tuple[list[dict], str]:
    """Render summary/report paths relative to the demo root when possible."""
    normalized_results: list[dict] = []
    for item in results:
        normalized = dict(item)
        try:
            normalized["path"] = str(Path(item["path"]).relative_to(output_dir))
        except ValueError:
            normalized["path"] = item["path"]
        try:
            normalized["reference_path"] = str(Path(item["reference_path"]).relative_to(output_dir))
        except ValueError:
            normalized["reference_path"] = item["reference_path"]
        normalized_results.append(normalized)
    try:
        normalized_reference = str(reference_path.relative_to(output_dir))
    except ValueError:
        normalized_reference = str(reference_path)
    return normalized_results, normalized_reference


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

    reference_points, reference_labels = _nonvoid_reference(points, labels)

    reference_path = output_dir / "reference_scene.pcd"
    candidates_dir = output_dir / "candidates"
    nondeep_path = candidates_dir / "nondeep_baseline.pcd"
    deep_path = candidates_dir / "deep_baseline.pcd"

    _write_pcd(reference_path, reference_points)
    _write_pcd(nondeep_path, _build_nondeep_candidate(reference_points))
    _write_pcd(deep_path, _build_deep_candidate(reference_points))

    results = batch_evaluate(
        str(candidates_dir),
        str(reference_path),
        thresholds=DEFAULT_THRESHOLDS,
        min_auc=MIN_AUC,
        max_chamfer=MAX_CHAMFER,
    )
    results, normalized_reference_path = _normalize_result_paths(results, reference_path, output_dir)
    summary = make_batch_summary(
        results,
        normalized_reference_path,
        min_auc=MIN_AUC,
        max_chamfer=MAX_CHAMFER,
    )

    save_batch_report(
        results,
        normalized_reference_path,
        str(output_dir / "index.html"),
        min_auc=MIN_AUC,
        max_chamfer=MAX_CHAMFER,
    )
    save_batch_report(
        results,
        normalized_reference_path,
        str(output_dir / "report.md"),
        min_auc=MIN_AUC,
        max_chamfer=MAX_CHAMFER,
    )
    (output_dir / "results.json").write_text(
        json.dumps({"results": results, "summary": summary}, indent=2),
        encoding="utf-8",
    )

    attribution_lines = [
        "# Perception Demo Attribution",
        "",
        "- Dataset: [RELLIS-3D](%s)" % RELLIS_REPO_URL,
        "- Dataset README: [RELLIS-3D README](%s)" % RELLIS_README_URL,
        "- Example bundle: [Ouster LiDAR with Annotation Examples](%s)" % RELLIS_LIDAR_EXAMPLE_URL,
        "- Label ontology: [rellis.yaml](%s)" % RELLIS_LABEL_CONFIG_URL,
        "- License: CC BY-NC-SA 3.0 (see the dataset README)",
        "- Frame used: `%s`" % args.frame,
        "- Ignored labels: `%s`" % ", ".join(str(label_id) for label_id in IGNORED_LABEL_IDS),
        "",
        "The generated artifacts in this directory are derived from the public example bundle above.",
        "Keep the upstream attribution and non-commercial / share-alike terms with these files.",
    ]
    (output_dir / "ATTRIBUTION.md").write_text("\n".join(attribution_lines) + "\n", encoding="utf-8")

    readme_lines = [
        "# Perception Batch QA Demo",
        "",
        "This demo compares two candidate perception artifacts against the same public reference frame.",
        "It uses the RELLIS-3D Ouster LiDAR example and evaluates both candidates with `ca batch`.",
        "",
        "## What The Candidates Mean",
        "",
        "- `nondeep_baseline.pcd`: deterministic geometry-only proxy with coarse voxelization, long-range thinning, and a small rigid bias.",
        "- `deep_baseline.pcd`: higher-fidelity proxy with denser sampling and a smaller rigid bias.",
        "- `reference_scene.pcd`: non-void labeled points from the official public RELLIS-3D example frame.",
        "",
        "These are demo proxies, not archived model outputs. The point is to show how CloudAnalyzer",
        "compares a non-deep candidate and a deep candidate against the same public reference artifact.",
        "",
        "## Source Data",
        "",
        "- Dataset: [RELLIS-3D](%s)" % RELLIS_REPO_URL,
        "- Example bundle: [Ouster LiDAR with Annotation Examples](%s)" % RELLIS_LIDAR_EXAMPLE_URL,
        "- Label ontology: [rellis.yaml](%s)" % RELLIS_LABEL_CONFIG_URL,
        "- Frame: `%s`" % args.frame,
        "",
        "## Label Counts In The Reference Frame",
        "",
        *_label_count_lines(reference_labels),
        "",
        "## Batch Metrics",
        "",
        "| Candidate | AUC | Chamfer | Best F1 | Gate |",
        "|---|---:|---:|---:|---|",
        *_result_rows(results),
        "",
        "Gate settings: `min_auc=%.2f`, `max_chamfer=%.2f`" % (MIN_AUC, MAX_CHAMFER),
        "",
        "## Files",
        "",
        "| File | Description |",
        "|---|---|",
        "| `reference_scene.pcd` | Public reference artifact derived from the official labels |",
        "| `candidates/nondeep_baseline.pcd` | Geometry-only non-deep proxy candidate |",
        "| `candidates/deep_baseline.pcd` | Higher-fidelity deep proxy candidate |",
        "| `index.html` | HTML batch report for GitHub Pages |",
        "| `report.md` | Markdown version of the same batch report |",
        "| `results.json` | Raw batch results plus summary |",
        "| `ATTRIBUTION.md` | Dataset provenance and license note |",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python3 scripts/build_perception_demo.py --output docs/demo/perception --frame %s" % args.frame,
        "",
        "ca batch docs/demo/perception/candidates \\",
        "  --evaluate docs/demo/perception/reference_scene.pcd \\",
        "  --thresholds 0.02,0.05,0.1,0.2,0.3 \\",
        "  --min-auc 0.90 --max-chamfer 0.05 \\",
        "  --report docs/demo/perception/index.html",
        "```",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"Frame:         {args.frame}")
    print(f"Reference pts: {int(reference_points.shape[0])}")
    for item in results:
        gate = item.get("quality_gate") or {}
        print(
            f"{Path(item['path']).name}: "
            f"AUC={item['auc']:.4f} "
            f"Chamfer={item['chamfer_distance']:.4f} "
            f"BestF1={item['best_f1']['f1']:.4f} "
            f"Gate={'PASS' if gate.get('passed') else 'FAIL'}"
        )
    print(f"Report:        {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
