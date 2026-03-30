"""Tests for ca.batch module."""

from pathlib import Path

import open3d as o3d
import pytest

from ca.batch import batch_evaluate, batch_info


class TestBatchInfo:
    def test_finds_files(self, tmp_path, simple_pcd):
        for name in ["a.pcd", "b.pcd", "c.pcd"]:
            o3d.io.write_point_cloud(str(tmp_path / name), simple_pcd)
        results = batch_info(str(tmp_path))
        assert len(results) == 3

    def test_recursive(self, tmp_path, simple_pcd):
        sub = tmp_path / "sub"
        sub.mkdir()
        o3d.io.write_point_cloud(str(tmp_path / "a.pcd"), simple_pcd)
        o3d.io.write_point_cloud(str(sub / "b.pcd"), simple_pcd)
        results = batch_info(str(tmp_path), recursive=True)
        assert len(results) == 2

    def test_no_recursive(self, tmp_path, simple_pcd):
        sub = tmp_path / "sub"
        sub.mkdir()
        o3d.io.write_point_cloud(str(tmp_path / "a.pcd"), simple_pcd)
        o3d.io.write_point_cloud(str(sub / "b.pcd"), simple_pcd)
        results = batch_info(str(tmp_path), recursive=False)
        assert len(results) == 1

    def test_empty_directory(self, tmp_path):
        results = batch_info(str(tmp_path))
        assert results == []

    def test_dir_not_found(self):
        with pytest.raises(FileNotFoundError):
            batch_info("/no/such/dir")


class TestBatchEvaluate:
    def test_evaluates_all_files(self, tmp_path, identical_pcd, shifted_pcd):
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        ref = tmp_path / "reference.pcd"
        o3d.io.write_point_cloud(str(ref), identical_pcd)
        o3d.io.write_point_cloud(str(batch_dir / "a.pcd"), identical_pcd)
        o3d.io.write_point_cloud(str(batch_dir / "b.pcd"), shifted_pcd)

        results = batch_evaluate(str(batch_dir), str(ref))

        assert len(results) == 2
        assert results[0]["path"].endswith("a.pcd")
        assert results[0]["auc"] == pytest.approx(1.0)
        assert results[0]["best_f1"]["f1"] == pytest.approx(1.0)
        assert len(results[0]["f1_scores"]) == 6
        assert results[0]["inspect"]["web_heatmap"].startswith("ca web ")
        assert results[0]["inspect"]["heatmap3d"].startswith("ca heatmap3d ")
        assert results[1]["chamfer_distance"] > 0

    def test_custom_thresholds(self, tmp_path, identical_pcd):
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        ref = tmp_path / "reference.pcd"
        o3d.io.write_point_cloud(str(ref), identical_pcd)
        o3d.io.write_point_cloud(str(batch_dir / "a.pcd"), identical_pcd)

        results = batch_evaluate(str(batch_dir), str(ref), thresholds=[0.01, 0.1])

        assert len(results[0]["f1_scores"]) == 2

    def test_quality_gate(self, tmp_path, identical_pcd, shifted_pcd):
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        ref = tmp_path / "reference.pcd"
        o3d.io.write_point_cloud(str(ref), identical_pcd)
        o3d.io.write_point_cloud(str(batch_dir / "a.pcd"), identical_pcd)
        o3d.io.write_point_cloud(str(batch_dir / "b.pcd"), shifted_pcd)

        results = batch_evaluate(
            str(batch_dir),
            str(ref),
            min_auc=0.95,
            max_chamfer=0.02,
        )

        assert results[0]["quality_gate"]["passed"] is True
        assert results[1]["quality_gate"]["passed"] is False
        assert results[1]["quality_gate"]["reasons"]

    def test_compression_metrics(self, tmp_path, identical_pcd):
        batch_dir = tmp_path / "decoded"
        compressed_dir = tmp_path / "compressed"
        baseline_dir = tmp_path / "original"
        batch_dir.mkdir()
        compressed_dir.mkdir()
        baseline_dir.mkdir()
        ref = tmp_path / "reference.pcd"
        o3d.io.write_point_cloud(str(ref), identical_pcd)

        decoded = batch_dir / "a.pcd"
        baseline = baseline_dir / "a.pcd"
        compressed = compressed_dir / "a.cloudini"
        o3d.io.write_point_cloud(str(decoded), identical_pcd)
        o3d.io.write_point_cloud(str(baseline), identical_pcd)
        compressed.write_bytes(b"1234567890")

        results = batch_evaluate(
            str(batch_dir),
            str(ref),
            compressed_dir=str(compressed_dir),
            baseline_dir=str(baseline_dir),
        )

        compression = results[0]["compression"]
        assert compression is not None
        assert compression["compressed_path"].endswith("a.cloudini")
        assert compression["baseline_path"].endswith("a.pcd")
        assert compression["compressed_size_bytes"] == 10
        assert 0 < compression["size_ratio"] < 1
        assert compression["pareto_optimal"] is True
        assert compression["recommended"] is True

    def test_compression_metrics_marks_dominated_result(self, tmp_path, identical_pcd, shifted_pcd):
        batch_dir = tmp_path / "decoded"
        compressed_dir = tmp_path / "compressed"
        baseline_dir = tmp_path / "original"
        batch_dir.mkdir()
        compressed_dir.mkdir()
        baseline_dir.mkdir()
        ref = tmp_path / "reference.pcd"
        o3d.io.write_point_cloud(str(ref), identical_pcd)

        o3d.io.write_point_cloud(str(batch_dir / "a.pcd"), identical_pcd)
        o3d.io.write_point_cloud(str(batch_dir / "b.pcd"), shifted_pcd)
        o3d.io.write_point_cloud(str(baseline_dir / "a.pcd"), identical_pcd)
        o3d.io.write_point_cloud(str(baseline_dir / "b.pcd"), shifted_pcd)
        (compressed_dir / "a.cloudini").write_bytes(b"1234567890")
        (compressed_dir / "b.cloudini").write_bytes(b"123456789012345678901234567890")

        results = batch_evaluate(
            str(batch_dir),
            str(ref),
            compressed_dir=str(compressed_dir),
            baseline_dir=str(baseline_dir),
        )

        by_name = {Path(item["path"]).name: item for item in results}
        assert by_name["a.pcd"]["compression"]["pareto_optimal"] is True
        assert by_name["b.pcd"]["compression"]["pareto_optimal"] is False
        assert by_name["a.pcd"]["compression"]["recommended"] is True
        assert by_name["b.pcd"]["compression"]["recommended"] is False

    def test_reference_not_found(self, tmp_path, identical_pcd):
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        o3d.io.write_point_cloud(str(batch_dir / "a.pcd"), identical_pcd)

        with pytest.raises(FileNotFoundError):
            batch_evaluate(str(batch_dir), str(tmp_path / "missing.pcd"))
