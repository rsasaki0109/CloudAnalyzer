"""Tests for ca.compare pipeline."""

import json

import pytest

from ca.compare import run_compare


class TestRunCompare:
    def test_basic_compare(self, source_and_target_files):
        src, tgt = source_and_target_files
        result = run_compare(src, tgt, method="gicp")
        assert "source_points" in result
        assert "target_points" in result
        assert "distance_stats" in result
        assert result["fitness"] is not None
        assert result["rmse"] is not None

    def test_no_registration(self, source_and_target_files):
        src, tgt = source_and_target_files
        result = run_compare(src, tgt, method=None)
        assert result["fitness"] is None
        assert result["rmse"] is None
        assert result["distance_stats"]["mean"] > 0

    def test_json_output(self, source_and_target_files, tmp_path):
        src, tgt = source_and_target_files
        json_path = str(tmp_path / "result.json")
        run_compare(src, tgt, method="gicp", json_path=json_path)
        data = json.loads(open(json_path).read())
        assert "distance_stats" in data

    def test_report_output(self, source_and_target_files, tmp_path):
        src, tgt = source_and_target_files
        report_path = str(tmp_path / "report.md")
        run_compare(src, tgt, method="gicp", report_path=report_path)
        content = open(report_path).read()
        assert "# CloudAnalyzer Report" in content

    def test_snapshot_output(self, source_and_target_files, tmp_path):
        src, tgt = source_and_target_files
        snap_path = str(tmp_path / "snap.png")
        run_compare(src, tgt, method="gicp", snapshot_path=snap_path)
        assert (tmp_path / "snap.png").exists()

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_compare("/no/such/file.pcd", "/no/such/other.pcd")

    def test_icp_method(self, source_and_target_files):
        src, tgt = source_and_target_files
        result = run_compare(src, tgt, method="icp")
        assert result["fitness"] is not None
