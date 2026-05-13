"""Tests for scan matching diagnostics."""

import numpy as np

from ca.scan_match_debug import run_scan_match_debug


def test_run_scan_match_debug_reports_before_after(source_and_target_files, tmp_path):
    src, tgt = source_and_target_files
    artifact_dir = tmp_path / "artifacts"

    result = run_scan_match_debug(
        scan_path=src,
        map_path=tgt,
        method="icp",
        max_correspondence_distance=0.5,
        threshold=0.05,
        artifact_dir=str(artifact_dir),
    )

    assert result["method"] == "icp"
    assert result["registration"]["fitness"] >= 0.0
    assert np.asarray(result["registration"]["final_transform"]).shape == (4, 4)
    assert result["distance_before"]["stats"]["mean"] >= result["distance_after"]["stats"]["mean"]
    assert result["distance_before"]["threshold"]["threshold"] == 0.05
    assert (artifact_dir / "scan_initial_error.ply").exists()
    assert (artifact_dir / "scan_aligned_error.ply").exists()
    assert (artifact_dir / "map_debug.ply").exists()


def test_run_scan_match_debug_crops_map(source_and_target_files):
    src, tgt = source_and_target_files

    result = run_scan_match_debug(
        scan_path=src,
        map_path=tgt,
        method="icp",
        max_correspondence_distance=0.5,
        crop_margin=0.2,
    )

    assert result["preprocess"]["map_points_used"] <= result["preprocess"]["map_points_raw"]
    assert result["preprocess"]["crop_margin"] == 0.2
