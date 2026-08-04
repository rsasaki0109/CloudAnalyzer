"""Tests for the reproducible upstream MapEval parity harness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ca.mapeval_parity import (
    UPSTREAM_COMMIT,
    compute_official_compatible_metrics,
    make_mapeval_smoke_fixture,
    run_parity_harness,
    write_ascii_pcd,
)


def test_mapeval_fixture_is_deterministic_and_dense() -> None:
    first = make_mapeval_smoke_fixture(seed=17, points_per_voxel=128)
    second = make_mapeval_smoke_fixture(seed=17, points_per_voxel=128)

    assert np.array_equal(first.reference_points, second.reference_points)
    assert np.array_equal(first.estimated_points, second.estimated_points)
    assert first.reference_points.shape == (256, 3)


def test_official_compatibility_lane_has_two_awd_voxels() -> None:
    fixture = make_mapeval_smoke_fixture()
    metrics = compute_official_compatible_metrics(
        fixture.estimated_points,
        fixture.reference_points,
    )

    assert metrics["n_awd_voxels"] == 2
    assert metrics["n_scs_voxels"] == 2
    assert np.isfinite(metrics["awd_m"])
    assert np.isfinite(metrics["scs"])
    assert metrics["awd_m"] > 0


def test_parity_harness_writes_ci_safe_report_without_binary(tmp_path: Path) -> None:
    output = tmp_path / "parity"
    report = run_parity_harness(output)

    assert report["status"] == "smoke_pass"
    assert report["comparison"]["status"] == "not_run"
    assert report["upstream"]["commit"] == UPSTREAM_COMMIT
    assert report["lanes"]["official_compatibility"]["metrics"]["n_awd_voxels"] == 2
    assert (output / "fixture" / "estimated" / "map.pcd").is_file()
    saved = json.loads((output / "mapeval_parity.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == "cloudanalyzer.mapeval_parity.v1"
    assert saved["fixture"]["sha256"]["reference"]


def test_parity_harness_records_missing_official_binary(tmp_path: Path) -> None:
    report = run_parity_harness(
        tmp_path / "missing-binary",
        official_executable=tmp_path / "map_eval",
    )

    assert report["official_execution"]["status"] == "unavailable"
    assert report["comparison"]["status"] == "unavailable"
    assert report["comparison"]["passed"] is None


def test_ascii_pcd_writer_emits_expected_point_count(tmp_path: Path) -> None:
    path = write_ascii_pcd(tmp_path / "points.pcd", np.eye(3))
    text = path.read_text(encoding="ascii")
    assert "WIDTH 3" in text
    assert "POINTS 3" in text
    assert "1 0 0" in text
