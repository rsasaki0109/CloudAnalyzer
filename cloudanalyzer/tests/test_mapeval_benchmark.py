"""Tests for the streaming MapEval scale benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ca.mapeval_benchmark import iter_synthetic_map_chunks, run_scale_benchmark


def test_synthetic_scale_chunks_are_bounded_and_complete() -> None:
    chunks = list(
        iter_synthetic_map_chunks(
            257,
            chunk_size=64,
            seed=4,
            estimated=False,
        )
    )
    assert [len(chunk) for chunk in chunks] == [64, 64, 64, 64, 1]
    assert all(chunk.shape[1:] == (3,) for chunk in chunks)
    assert all(np.isfinite(chunk).all() for chunk in chunks)


def test_scale_benchmark_records_metrics_and_resources(tmp_path: Path) -> None:
    output = tmp_path / "scale.json"
    report = run_scale_benchmark((2_000,), chunk_size=128, output=output)

    assert report["schema_version"] == "cloudanalyzer.mapeval_scale.v1"
    assert report["runs"][0]["points_per_map"] == 2_000
    assert report["runs"][0]["metrics"]["n_awd_voxels"] > 0
    assert report["runs"][0]["runtime"]["elapsed_seconds"] >= 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["plan"]["target_points_per_map"] == [1_000_000, 10_000_000]
