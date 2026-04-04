"""Tests for the public benchmark pack builder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_public_benchmark_pack import build_public_benchmark_pack  # noqa: E402
from ca.core import load_check_suite, run_check_suite  # noqa: E402


def _synthetic_source_points() -> np.ndarray:
    """Create a deterministic nontrivial 3D shape for offline tests."""
    angles = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    heights = np.linspace(-0.6, 0.6, 18)
    points: list[list[float]] = []
    for z in heights:
        radius = 0.75 - (0.2 * abs(z))
        for angle in angles:
            wobble = 1.0 + (0.08 * np.sin(angle * 3.0))
            x = radius * wobble * np.cos(angle)
            y = radius * wobble * np.sin(angle)
            points.append([x, y, z + (0.04 * np.sin(angle * 2.0))])
    return np.asarray(points, dtype=float)


class TestPublicBenchmarkPack:
    def test_builder_generates_manifest_expected_summaries_and_artifacts(self, tmp_path: Path):
        output_dir = tmp_path / "public-pack"

        result = build_public_benchmark_pack(
            output_dir,
            source_points=_synthetic_source_points(),
            seed=11,
        )

        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert result["output_dir"] == str(output_dir.resolve())
        assert result["source"] == "custom"
        assert manifest["pack"] == "public-benchmark-pack"
        assert [item["id"] for item in manifest["configs"]] == [
            "suite-pass",
            "suite-regression",
        ]

        pass_entry = manifest["configs"][0]
        regression_entry = manifest["configs"][1]
        assert pass_entry["expected_pass"] is True
        assert regression_entry["expected_pass"] is False
        assert pass_entry["failed_check_ids"] == []
        assert set(regression_entry["failed_check_ids"]) == {
            "mapping-postprocess",
            "localization-run",
            "perception-output",
            "integrated-run",
        }

        assert (output_dir / "README.md").exists()
        assert (output_dir / "ATTRIBUTION.md").exists()
        assert (output_dir / "baselines" / "mapping_ref.pcd").exists()
        assert (output_dir / "outputs" / "mapping_pass.pcd").exists()
        assert (output_dir / "outputs" / "mapping_fail.pcd").exists()
        assert (output_dir / "expected" / "suite-pass.summary.json").exists()
        assert (output_dir / "expected" / "suite-regression.summary.json").exists()
        assert (output_dir / "reports" / "pass" / "mapping-postprocess.html").exists()
        assert (output_dir / "reports" / "regression" / "integrated-run.html").exists()
        assert (output_dir / "results" / "pass" / "integrated-run.json").exists()

    def test_generated_configs_match_manifest_expectations(self, tmp_path: Path):
        output_dir = tmp_path / "public-pack"
        build_public_benchmark_pack(
            output_dir,
            source_points=_synthetic_source_points(),
            seed=19,
        )
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

        for item in manifest["configs"]:
            config_path = output_dir / item["config_path"]
            rerun = run_check_suite(load_check_suite(str(config_path)))
            assert rerun["summary"]["passed"] is item["expected_pass"]
            assert rerun["summary"]["failed_check_ids"] == item["failed_check_ids"]
