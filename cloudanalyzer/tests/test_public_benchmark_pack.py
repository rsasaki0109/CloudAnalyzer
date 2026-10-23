"""Tests for the public benchmark pack builder."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import build_public_benchmark_pack as builder  # noqa: E402
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
    def test_synthetic_builder_runs_offline_and_detects_regressions(self, tmp_path: Path, monkeypatch):
        output_dir = tmp_path / "public-pack"

        def forbid_download(*args, **kwargs):
            pytest.fail("Synthetic fixtures must not download geometry")

        monkeypatch.setattr(builder, "download_bunny_mesh", forbid_download)
        monkeypatch.setattr(urllib.request, "urlretrieve", forbid_download)
        monkeypatch.setattr(urllib.request, "urlopen", forbid_download)
        result = build_public_benchmark_pack(
            output_dir,
            fixture="synthetic",
        )

        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert result["output_dir"] == str(output_dir.resolve())
        assert result["source"] == manifest["source"] == "synthetic-smoke"
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
        for name in ["README.md", "ATTRIBUTION.md"]:
            text = (output_dir / name).read_text(encoding="utf-8")
            assert "procedural surface" in text
            assert "Stanford" not in text
        assert "--fixture synthetic" in (output_dir / "README.md").read_text(encoding="utf-8")

    def test_generated_configs_match_manifest_expectations(self, tmp_path: Path):
        output_dir = tmp_path / "public-pack"
        build_public_benchmark_pack(
            output_dir,
            source_points=_synthetic_source_points(),
            seed=19,
        )
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["source"] == "custom"
        for name in ["README.md", "ATTRIBUTION.md"]:
            assert "Stanford" not in (output_dir / name).read_text(encoding="utf-8")

        for item in manifest["configs"]:
            config_path = output_dir / item["config_path"]
            rerun = run_check_suite(load_check_suite(str(config_path)))
            assert rerun["summary"]["passed"] is item["expected_pass"]
            assert rerun["summary"]["failed_check_ids"] == item["failed_check_ids"]


def test_synthetic_source_is_small_finite_and_deterministic():
    points = builder._synthetic_smoke_points()
    assert points.shape == (1440, 3)
    assert np.isfinite(points).all()
    assert (np.ptp(points, axis=0) > 1.0).all()
    np.testing.assert_array_equal(points, builder._synthetic_smoke_points())


def test_default_download_failure_is_not_replaced_by_synthetic(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    def fail_download():
        raise TimeoutError("Stanford unavailable")

    monkeypatch.setattr(builder, "download_bunny_mesh", fail_download)
    with pytest.raises(TimeoutError, match="Stanford unavailable"):
        build_public_benchmark_pack(output_dir)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_default_fixture_retains_stanford_download_and_attribution(tmp_path: Path, monkeypatch):
    downloads = []

    def fake_download():
        downloads.append(True)
        return SimpleNamespace(vertices=_synthetic_source_points())

    monkeypatch.setattr(builder, "download_bunny_mesh", fake_download)
    result = build_public_benchmark_pack(tmp_path / "stanford-pack")
    assert downloads == [True]
    assert result["source"] == result["manifest"]["source"] == "stanford-bunny"
    for name in ["README.md", "ATTRIBUTION.md"]:
        text = (tmp_path / "stanford-pack" / name).read_text(encoding="utf-8")
        assert "Stanford Bunny" in text
        assert builder.BUNNY_SOURCE_PAGE in text
        assert builder.BUNNY_ARCHIVE_URL in text
        assert "synthetic" not in text


@pytest.mark.parametrize("fixture,custom", [("invalid", False), ("synthetic", True)])
def test_invalid_source_selection_preserves_existing_output(tmp_path: Path, fixture, custom):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError):
        build_public_benchmark_pack(
            tmp_path, fixture=fixture,
            source_points=_synthetic_source_points() if custom else None,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "arguments,fixture,output",
    [
        ([], "stanford-bunny", "benchmarks/public/stanford-bunny-mini"),
        (["--fixture", "synthetic"], "synthetic", "benchmarks/public/self-qa-synthetic"),
        (["--fixture", "synthetic", "--output", "my-pack"], "synthetic", "my-pack"),
    ],
)
def test_cli_selects_fixture_and_separate_default_output(monkeypatch, capsys, arguments, fixture, output):
    calls = []

    def capture_build(output_dir, *, fixture):
        calls.append((output_dir, fixture))
        return {"output_dir": str(output_dir), "pack_name": "test", "source": fixture, "manifest": {"configs": []}}

    monkeypatch.setattr(builder, "build_public_benchmark_pack", capture_build)
    monkeypatch.setattr(sys, "argv", ["build_public_benchmark_pack.py", *arguments])
    builder.main()
    assert calls == [(Path(output), fixture)]
    assert f"Source:       {fixture}" in capsys.readouterr().out


def test_self_qa_uses_offline_fixture_with_matching_config():
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/self-qa.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["qa"]["steps"]
    build_step = next(step for step in steps if "build_public_benchmark_pack.py" in step.get("run", ""))
    action_step = next(step for step in steps if step.get("uses") == "./cloudanalyzer-action")
    assert "--fixture synthetic" in build_step["run"]
    assert "--output benchmarks/public/self-qa-synthetic" in build_step["run"]
    assert action_step["with"]["config"] == "benchmarks/public/self-qa-synthetic/configs/suite-pass.cloudanalyzer.yaml"
