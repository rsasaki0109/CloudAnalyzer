#!/usr/bin/env python3
"""Build a public benchmark pack for CloudAnalyzer QA workflows."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "cloudanalyzer"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from public_benchmark_assets import (  # noqa: E402
    BUNNY_ARCHIVE_URL,
    BUNNY_SOURCE_PAGE,
    download_bunny_mesh,
    make_trajectory_rows,
    write_csv_trajectory,
)
from ca.core import load_check_suite, run_check_suite  # noqa: E402


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center and scale points to a unit-ish working space."""
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = bbox_max - bbox_min
    scale = float(max(np.max(extent), 1e-8))
    normalized = (points - center[None, :]) / scale
    return normalized, np.zeros(3, dtype=float), extent / scale


def _make_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Create an Open3D point cloud from a numpy array."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    return cloud


def _write_point_cloud(path: Path, points: np.ndarray) -> None:
    """Write a point cloud to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), _make_point_cloud(points)):
        raise RuntimeError(f"failed to write point cloud: {path}")


def _mapping_pass_points(points: np.ndarray) -> np.ndarray:
    """Generate a mild map post-processing candidate that should pass."""
    passed = points.copy()
    passed[:, 0] += 0.0015 * np.sin(passed[:, 1] * 9.0)
    passed[:, 1] += 0.0010 * np.cos(passed[:, 2] * 7.0)
    passed[:, 2] += 0.0015 * np.sin(passed[:, 0] * 8.0)
    return np.asarray(passed, dtype=float)


def _mapping_fail_points(points: np.ndarray) -> np.ndarray:
    """Generate a distorted map candidate that should fail."""
    failed = points.copy()
    failed[:, 0] = failed[:, 0] * 1.25 + 0.05
    failed[:, 2] *= 0.88
    keep_mask = failed[:, 1] < 0.24
    return np.asarray(failed[keep_mask], dtype=float)


def _perception_pass_points(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a mild reconstruction candidate that should pass."""
    noisy = points.copy()
    noisy += rng.normal(scale=0.0025, size=noisy.shape)
    noisy[:, 2] += 0.003 * np.sin(noisy[:, 0] * 8.0)
    return np.asarray(noisy, dtype=float)


def _perception_fail_points(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a visibly degraded reconstruction candidate that should fail."""
    failed = points.copy()
    failed += rng.normal(scale=0.02, size=failed.shape)
    failed[:, 2] *= 0.78
    keep_mask = (failed[:, 0] + failed[:, 1]) < 0.18
    return np.asarray(failed[keep_mask], dtype=float)


def _localization_pass_rows(center: np.ndarray, extent: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Generate a close-to-reference trajectory."""
    reference = make_trajectory_rows(
        center,
        extent,
        phase=0.0,
        radial_wobble=0.0,
        vertical_wobble=0.03,
        sample_count=80,
    )
    rows: list[tuple[float, float, float, float]] = []
    for timestamp, x, y, z in reference:
        progress = timestamp / max(reference[-1][0], 1e-8)
        rows.append(
            (
                timestamp,
                x + (0.003 * np.sin(progress * 6.0)),
                y + (0.002 * np.cos(progress * 4.0)),
                z + (0.002 * np.sin(progress * 5.0)),
            )
        )
    return rows


def _localization_fail_rows(center: np.ndarray, extent: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Generate a drifting trajectory with reduced coverage."""
    reference = make_trajectory_rows(
        center,
        extent,
        phase=0.0,
        radial_wobble=0.0,
        vertical_wobble=0.03,
        sample_count=80,
    )
    rows: list[tuple[float, float, float, float]] = []
    kept = reference[:64]
    total = max(len(reference) - 1, 1)
    for index, (timestamp, x, y, z) in enumerate(kept):
        progress = index / total
        rows.append(
            (
                timestamp,
                x + (0.12 * progress),
                y - (0.08 * progress),
                z + (0.05 * progress),
            )
        )
    return rows


def _write_check_config(path: Path, content: str) -> None:
    """Write a YAML config file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _write_pack_readme(output_dir: Path) -> None:
    """Write benchmark pack usage docs."""
    lines = [
        "# Public Benchmark Pack",
        "",
        "This pack demonstrates CloudAnalyzer as a QA platform across:",
        "",
        "- mapping post-processing",
        "- localization trajectory validation",
        "- perception / reconstruction regression checks",
        "- integrated run evaluation",
        "",
        "Generate the pack with:",
        "",
        "```bash",
        "python scripts/build_public_benchmark_pack.py --output benchmarks/public/stanford-bunny-mini",
        "```",
        "",
        "Run the configs with:",
        "",
        "```bash",
        "ca check configs/suite-pass.cloudanalyzer.yaml",
        "ca check configs/suite-regression.cloudanalyzer.yaml",
        "```",
        "",
        "The pack stores expected summaries under `expected/` and a manifest under `manifest.json`.",
        "",
        "The source geometry is derived from the public Stanford Bunny mesh:",
        f"- Source page: {BUNNY_SOURCE_PAGE}",
        f"- Archive: {BUNNY_ARCHIVE_URL}",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_attribution(output_dir: Path) -> None:
    """Write attribution for the public source geometry."""
    lines = [
        "# Attribution",
        "",
        "Source geometry: The Stanford Bunny from the Stanford 3D Scanning Repository.",
        f"Source page: {BUNNY_SOURCE_PAGE}",
        f"Archive: {BUNNY_ARCHIVE_URL}",
        "",
        "The benchmark pack derives point cloud fixtures from the public bunny mesh.",
        "Trajectory fixtures, degraded outputs, and config files are generated by CloudAnalyzer",
        "to demonstrate config-driven QA across mapping, localization, perception, and integrated runs.",
        "",
        "The Stanford repository states that the models may be used for research, mirrored or",
        "redistributed for free with credit to the Stanford Computer Graphics Laboratory, and are",
        "not for commercial use without permission.",
    ]
    (output_dir / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_configs(output_dir: Path) -> list[dict[str, Any]]:
    """Write pass/regression configs and return their manifest entries."""
    configs_dir = output_dir / "configs"
    pass_config = """
version: 1
project: public-benchmark-pass
summary_output_json: ../expected/suite-pass.summary.json
defaults:
  thresholds: [0.01, 0.02, 0.05]
  max_time_delta: 0.15
  report_dir: ../reports/pass
  json_dir: ../results/pass
checks:
  - id: mapping-postprocess
    kind: artifact
    source: ../outputs/mapping_pass.pcd
    reference: ../baselines/mapping_ref.pcd
    gate:
      min_auc: 0.97
      max_chamfer: 0.01
  - id: localization-run
    kind: trajectory
    estimated: ../outputs/localization_pass.csv
    reference: ../baselines/localization_ref.csv
    alignment: none
    gate:
      max_ate: 0.02
      max_rpe: 0.02
      max_drift: 0.03
      min_coverage: 0.95
  - id: perception-output
    kind: artifact
    source: ../outputs/perception_pass.pcd
    reference: ../baselines/perception_ref.pcd
    gate:
      min_auc: 0.95
      max_chamfer: 0.015
  - id: integrated-run
    kind: run
    map: ../outputs/mapping_pass.pcd
    map_reference: ../baselines/mapping_ref.pcd
    trajectory: ../outputs/localization_pass.csv
    trajectory_reference: ../baselines/localization_ref.csv
    gate:
      min_auc: 0.97
      max_chamfer: 0.01
      max_ate: 0.02
      max_rpe: 0.02
      max_drift: 0.03
      min_coverage: 0.95
"""
    regression_config = """
version: 1
project: public-benchmark-regression
summary_output_json: ../expected/suite-regression.summary.json
defaults:
  thresholds: [0.01, 0.02, 0.05]
  max_time_delta: 0.15
  report_dir: ../reports/regression
  json_dir: ../results/regression
checks:
  - id: mapping-postprocess
    kind: artifact
    source: ../outputs/mapping_fail.pcd
    reference: ../baselines/mapping_ref.pcd
    gate:
      min_auc: 0.97
      max_chamfer: 0.01
  - id: localization-run
    kind: trajectory
    estimated: ../outputs/localization_fail.csv
    reference: ../baselines/localization_ref.csv
    alignment: none
    gate:
      max_ate: 0.02
      max_rpe: 0.02
      max_drift: 0.03
      min_coverage: 0.95
  - id: perception-output
    kind: artifact
    source: ../outputs/perception_fail.pcd
    reference: ../baselines/perception_ref.pcd
    gate:
      min_auc: 0.95
      max_chamfer: 0.015
  - id: integrated-run
    kind: run
    map: ../outputs/mapping_fail.pcd
    map_reference: ../baselines/mapping_ref.pcd
    trajectory: ../outputs/localization_fail.csv
    trajectory_reference: ../baselines/localization_ref.csv
    gate:
      min_auc: 0.97
      max_chamfer: 0.01
      max_ate: 0.02
      max_rpe: 0.02
      max_drift: 0.03
      min_coverage: 0.95
"""
    pass_path = configs_dir / "suite-pass.cloudanalyzer.yaml"
    regression_path = configs_dir / "suite-regression.cloudanalyzer.yaml"
    _write_check_config(pass_path, pass_config)
    _write_check_config(regression_path, regression_config)
    return [
        {"id": "suite-pass", "config_path": str(pass_path), "expected_pass": True},
        {"id": "suite-regression", "config_path": str(regression_path), "expected_pass": False},
    ]


def _write_manifest(
    output_dir: Path,
    manifest_entries: list[dict[str, Any]],
    *,
    source_label: str,
) -> dict[str, Any]:
    """Write a benchmark manifest file."""
    manifest = {
        "pack": "public-benchmark-pack",
        "source": source_label,
        "configs": [
            {
                "id": entry["id"],
                "config_path": str(Path(entry["config_path"]).relative_to(output_dir)),
                "expected_pass": entry["expected_pass"],
                "expected_summary_path": f"expected/{entry['id']}.summary.json",
            }
            for entry in manifest_entries
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_public_benchmark_pack(
    output_dir: Path,
    *,
    source_points: np.ndarray | None = None,
    seed: int = 7,
) -> dict[str, Any]:
    """Build a public benchmark pack in the target directory."""
    rng = np.random.default_rng(seed)
    source_label = "custom"
    if source_points is None:
        mesh = download_bunny_mesh()
        source_points = np.asarray(mesh.vertices, dtype=float)
        source_label = "stanford-bunny"

    normalized_points, center, extent = _normalize_points(np.asarray(source_points, dtype=float))
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baselines_dir = output_dir / "baselines"
    outputs_dir = output_dir / "outputs"
    expected_dir = output_dir / "expected"
    reports_dir = output_dir / "reports"
    results_dir = output_dir / "results"
    for directory in [baselines_dir, outputs_dir, expected_dir, reports_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    mapping_ref = normalized_points
    perception_ref = normalized_points
    mapping_pass = _mapping_pass_points(normalized_points)
    mapping_fail = _mapping_fail_points(normalized_points)
    perception_pass = _perception_pass_points(normalized_points, rng)
    perception_fail = _perception_fail_points(normalized_points, rng)

    localization_ref_rows = make_trajectory_rows(
        center,
        extent,
        phase=0.0,
        radial_wobble=0.0,
        vertical_wobble=0.03,
        sample_count=80,
    )
    localization_pass_rows = _localization_pass_rows(center, extent)
    localization_fail_rows = _localization_fail_rows(center, extent)

    _write_point_cloud(baselines_dir / "mapping_ref.pcd", mapping_ref)
    _write_point_cloud(baselines_dir / "perception_ref.pcd", perception_ref)
    _write_point_cloud(outputs_dir / "mapping_pass.pcd", mapping_pass)
    _write_point_cloud(outputs_dir / "mapping_fail.pcd", mapping_fail)
    _write_point_cloud(outputs_dir / "perception_pass.pcd", perception_pass)
    _write_point_cloud(outputs_dir / "perception_fail.pcd", perception_fail)
    write_csv_trajectory(baselines_dir / "localization_ref.csv", localization_ref_rows)
    write_csv_trajectory(outputs_dir / "localization_pass.csv", localization_pass_rows)
    write_csv_trajectory(outputs_dir / "localization_fail.csv", localization_fail_rows)

    manifest_entries = _build_configs(output_dir)
    manifest = _write_manifest(output_dir, manifest_entries, source_label=source_label)
    _write_pack_readme(output_dir)
    _write_attribution(output_dir)

    summaries: list[dict[str, Any]] = []
    for entry in manifest_entries:
        result = run_check_suite(load_check_suite(entry["config_path"]))
        summary_path = output_dir / "expected" / f"{entry['id']}.summary.json"
        summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        summaries.append(
            {
                "id": entry["id"],
                "config_path": str(Path(entry["config_path"]).relative_to(output_dir)),
                "expected_pass": result["summary"]["passed"],
                "failed_check_ids": result["summary"]["failed_check_ids"],
                "summary_path": str(summary_path.relative_to(output_dir)),
            }
        )

    manifest["configs"] = summaries
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "pack_name": "public-benchmark-pack",
        "source": source_label,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="benchmarks/public/stanford-bunny-mini",
        help="Output directory for the generated benchmark pack",
    )
    args = parser.parse_args()

    result = build_public_benchmark_pack(Path(args.output))
    print(f"Generated:    {result['output_dir']}")
    print(f"Pack:         {result['pack_name']}")
    print(f"Configs:      {len(result['manifest']['configs'])}")
    for item in result["manifest"]["configs"]:
        print(
            f"  {item['id']}: pass={item['expected_pass']} "
            f"failed={len(item['failed_check_ids'])}"
        )


if __name__ == "__main__":
    main()
